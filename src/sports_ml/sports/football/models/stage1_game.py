"""NFL Stage 1: calibrated win probability and expected team scores (`models/stage1_game.py`).

Components (2-way outcome; ties are `0.5` in Elo and excluded from classifier fit/calibration):

1. **Expected scores**: shallow, heavily regularized Tweedie LightGBM regressors with a
   log-link anchored on the previous season's league-mean points (`init_score`) and monotone
   constraints on key QB / EPA / Elo features, yielding `(mu_home, mu_away)`, implied margin
   and total.
2. **Hybrid win-probability ensemble**: a logit-space blend of QB-adjusted Elo, the score-implied
   probability `Phi(margin / sigma_margin)` (`sigma_margin ~ 13.5`, empirically shrunk), and a
   regularized logistic / shallow-GBDT classifier.
3. **Calibration**: Platt / temperature scaling (`BinaryProbabilityCalibrator`) on pooled
   out-of-fold predictions of prior seasons, with an identity fallback below a minimum sample.
4. **Neutral sites** are symmetric under home/away relabeling (predictions are averaged over the
   original and swapped orientation).
5. **Walk-forward backtest** with `FoldBoundaryAudit`, season-bootstrap confidence intervals,
   per-era home-win priors, and playoff / Week 1-4 slices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Literal

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sports_ml.config import RANDOM_SEED
from sports_ml.core.calibration import (
    BinaryCalibrationMethod,
    BinaryProbabilityCalibrator,
    clip_probabilities,
)
from sports_ml.core.evaluation import (
    BenchmarkComparisonReport,
    EvaluationMetrics,
    compare_against_baselines,
    evaluate_probabilities,
)
from sports_ml.core.temporal import sort_events_chronologically
from sports_ml.sports.football.constants import get_league_priors
from sports_ml.sports.football.features.pipeline import (
    STAGE1_FEATURE_COLUMNS,
    is_forbidden_feature_column,
)
from sports_ml.sports.hockey.models.stage1_game import FoldBoundaryAudit

PRIOR_SIGMA_MARGIN: Final[float] = 13.5
SIGMA_PRIOR_STRENGTH: Final[float] = 200.0
SIGMA_BOUNDS: Final[tuple[float, float]] = (11.0, 16.0)
ERAS: Final[tuple[tuple[str, int, int], ...]] = (
    ("1999-2010", 1999, 2010),
    ("2011-2019", 2011, 2019),
    ("2020+", 2020, 9999),
)
_NEGATE_COLUMNS: Final[frozenset[str]] = frozenset({"rest_diff", "qb_elo_diff_pre"})
_FLIP_PROB_COLUMNS: Final[frozenset[str]] = frozenset({"elo_home_win_prob", "qb_elo_home_win_prob"})
_SWAP_PAIRS: Final[tuple[tuple[str, str], ...]] = (("elo_home_pre", "elo_away_pre"),)

# Monotone constraints on the *home* regressor: +1 raises home points, -1 lowers them.
# The away regressor uses the mirrored sign.
HOME_MONOTONE_CONSTRAINTS: Final[dict[str, int]] = {
    "diff_qb_value": 1,
    "diff_qb_epa_per_dropback": 1,
    "diff_off_epa_per_play_std": 1,
    "diff_def_epa_per_play_std": -1,
    "diff_adj_epa_net": 1,
    "diff_elo_pre": 1,
    "qb_elo_diff_pre": 1,
}

ClassifierKind = Literal["logistic", "gbdt", "both"]


def _f64(df: pl.DataFrame, col: str, fill: float) -> NDArray[np.float64]:
    return np.asarray(df[col].cast(pl.Float64).fill_null(fill).to_numpy(), dtype=np.float64)


def swap_home_away_features(df: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Relabel home/away: swap `home_*`/`away_*`, negate `diff_*`, flip home-win probabilities."""
    cols = set(feature_columns)
    exprs: list[pl.Expr] = []
    for c in feature_columns:
        partner: str | None = None
        if c.startswith("home_") and f"away_{c[5:]}" in cols:
            partner = f"away_{c[5:]}"
        elif c.startswith("away_") and f"home_{c[5:]}" in cols:
            partner = f"home_{c[5:]}"
        else:
            for a, b in _SWAP_PAIRS:
                if c == a and b in cols:
                    partner = b
                elif c == b and a in cols:
                    partner = a
        if partner is not None:
            exprs.append(pl.col(partner).alias(c))
        elif c.startswith("diff_") or c in _NEGATE_COLUMNS:
            exprs.append((-pl.col(c).cast(pl.Float64)).alias(c))
        elif c in _FLIP_PROB_COLUMNS:
            exprs.append((1.0 - pl.col(c).cast(pl.Float64)).alias(c))
    return df.with_columns(exprs) if exprs else df


@dataclass(frozen=True)
class FitBoundaryAudit:
    """Chronological boundaries used by one `Stage1FootballGameModel.fit` call."""

    gbdt_train_max_time: datetime
    calibrator_min_time: datetime | None
    calibrator_max_time: datetime | None


class Stage1FootballGameModel:
    """Mandatory NFL Stage 1: calibrated `P(Home Win)` plus expected team scores."""

    def __init__(
        self,
        *,
        calibration_method: BinaryCalibrationMethod = "platt",
        min_calibration_samples: int = 400,
        calibration_seasons: int = 2,
        elo_weight: float = 0.40,
        score_weight: float = 0.30,
        classifier_weight: float = 0.30,
        classifier: ClassifierKind = "both",
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        num_leaves: int = 4,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        total = elo_weight + score_weight + classifier_weight
        if total <= 0 or min(elo_weight, score_weight, classifier_weight) < 0:
            raise ValueError("Blend weights must be non-negative with a positive sum.")
        self.calibration_method: BinaryCalibrationMethod = calibration_method
        self.min_calibration_samples = min_calibration_samples
        self.calibration_seasons = calibration_seasons
        self.weights = (elo_weight / total, score_weight / total, classifier_weight / total)
        self.classifier_kind: ClassifierKind = classifier
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_seed = random_seed

        self.feature_columns: list[str] = []
        self.calibrator = BinaryProbabilityCalibrator(
            method=calibration_method,
            min_samples=min_calibration_samples,
            random_seed=random_seed,
        )
        self.sigma_margin: float = PRIOR_SIGMA_MARGIN
        self.fit_audit_: FitBoundaryAudit | None = None
        self._season_means: dict[int, float] = {}
        self._home_reg: LGBMRegressor | None = None
        self._away_reg: LGBMRegressor | None = None
        self._scaler: StandardScaler | None = None
        self._logit_clf: LogisticRegression | None = None
        self._gbdt_clf: LGBMClassifier | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def resolve_feature_columns(df: pl.DataFrame) -> list[str]:
        """Stage 1 feature columns present in `df`, in `df` column order (firewall-checked)."""
        allowed = set(STAGE1_FEATURE_COLUMNS)
        cols = [c for c in df.columns if c in allowed]
        if not cols:
            raise ValueError("No Stage 1 football feature columns found in DataFrame.")
        bad = [c for c in cols if is_forbidden_feature_column(c)]
        if bad:
            raise ValueError(f"Forbidden odds/observed-outcome feature columns: {bad}.")
        return cols

    def _matrix(self, df: pl.DataFrame) -> NDArray[np.float64]:
        mat = df.select(
            [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in self.feature_columns]
        ).to_numpy()
        return np.asarray(mat, dtype=np.float64)

    @staticmethod
    def classifier_mask(df: pl.DataFrame) -> pl.Expr:
        """Rows usable as 2-way classifier targets: completed games that are not ties."""
        tie = pl.col("is_tie").fill_null(0) if "is_tie" in df.columns else pl.lit(0)
        return pl.col("home_win").is_not_null() & (tie != 1)

    @staticmethod
    def binary_target(df: pl.DataFrame) -> NDArray[np.int64]:
        return np.asarray(df["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)

    def _anchor(self, season: int) -> float:
        earlier = [s for s in self._season_means if s < season]
        if earlier:
            return self._season_means[max(earlier)]
        return get_league_priors(season).points_per_team_game

    def _anchors(self, df: pl.DataFrame) -> NDArray[np.float64]:
        return np.array([self._anchor(int(s)) for s in df["season"].to_list()], dtype=np.float64)

    def _elo_prob(self, df: pl.DataFrame) -> NDArray[np.float64]:
        for col in ("qb_elo_home_win_prob", "elo_home_win_prob"):
            if col in df.columns:
                return clip_probabilities(_f64(df, col, 0.5))
        return np.full(df.height, 0.5, dtype=np.float64)

    def _monotone(self, home: bool) -> list[int]:
        sign = 1 if home else -1
        return [sign * HOME_MONOTONE_CONSTRAINTS.get(c, 0) for c in self.feature_columns]

    def _new_calibrator(self) -> BinaryProbabilityCalibrator:
        return BinaryProbabilityCalibrator(
            method=self.calibration_method,
            min_samples=self.min_calibration_samples,
            random_seed=self.random_seed,
        )

    # --------------------------------------------------------------------- fit

    def _split(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame | None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        sorted_df = sort_events_chronologically(train_df)
        if val_df is not None:
            if val_df.is_empty():
                raise ValueError("val_df cannot be empty when explicitly provided.")
            max_train = sorted_df["start_time_utc"].max()
            min_val = val_df["start_time_utc"].min()
            if (
                isinstance(max_train, datetime)
                and isinstance(min_val, datetime)
                and max_train >= min_val
            ):
                raise ValueError(
                    f"Temporal leakage detected: train max timestamp ({max_train}) >= "
                    f"calibration validation min timestamp ({min_val})."
                )
            return sorted_df, sort_events_chronologically(val_df)

        seasons = sorted(set(sorted_df["season"].to_list()))
        if len(seasons) >= 3:
            n_cal = min(self.calibration_seasons, len(seasons) - 1)
            cutoff = seasons[-n_cal]
            return (
                sorted_df.filter(pl.col("season") < cutoff),
                sorted_df.filter(pl.col("season") >= cutoff),
            )
        blocks = (
            sorted_df.select("season", "week").unique().sort("season", "week").rows()
            if "week" in sorted_df.columns
            else []
        )
        if len(blocks) >= 4:
            cut_season, cut_week = blocks[max(1, int(len(blocks) * 0.8))]
            before = (pl.col("season") < cut_season) | (
                (pl.col("season") == cut_season) & (pl.col("week") < cut_week)
            )
            return sorted_df.filter(before), sorted_df.filter(~before)
        split_row = max(1, int(sorted_df.height * 0.8))
        return sorted_df.slice(0, split_row), sorted_df.slice(split_row)

    def fit(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame | None = None
    ) -> Stage1FootballGameModel:
        """Fit score regressors and classifiers on early data; calibrate on later pooled data."""
        if train_df.is_empty():
            raise ValueError("train_df cannot be empty.")
        gbdt_df, cal_df = self._split(train_df, val_df)
        if gbdt_df.is_empty():
            raise ValueError("Not enough history to fit Stage 1 (empty GBDT training block).")
        self.feature_columns = self.resolve_feature_columns(gbdt_df)

        gbdt_max = gbdt_df["start_time_utc"].max()
        cal_min = cal_df["start_time_utc"].min() if not cal_df.is_empty() else None
        cal_max = cal_df["start_time_utc"].max() if not cal_df.is_empty() else None
        assert isinstance(gbdt_max, datetime)
        self.fit_audit_ = FitBoundaryAudit(
            gbdt_train_max_time=gbdt_max,
            calibrator_min_time=cal_min if isinstance(cal_min, datetime) else None,
            calibrator_max_time=cal_max if isinstance(cal_max, datetime) else None,
        )

        scored = gbdt_df.filter(
            pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        )
        if scored.is_empty():
            raise ValueError("No completed games with scores available to fit Stage 1.")
        season_pts = (
            scored.group_by("season")
            .agg(((pl.col("home_score") + pl.col("away_score")) / 2.0).mean().alias("m"))
            .sort("season")
        )
        self._season_means = {
            int(r["season"]): float(r["m"]) for r in season_pts.iter_rows(named=True)
        }

        x_s = self._matrix(scored)
        anchors = self._anchors(scored)
        y_h = np.clip(_f64(scored, "home_score", 0.0), 0.0, None)
        y_a = np.clip(_f64(scored, "away_score", 0.0), 0.0, None)
        min_child = max(10, min(40, x_s.shape[0] // 20))
        reg_params: dict[str, object] = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.3,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": 3,
            "min_child_samples": min_child,
            "colsample_bytree": 0.6,
            "subsample": 0.8,
            "subsample_freq": 1,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "random_state": self.random_seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        reg_h = LGBMRegressor(**reg_params, monotone_constraints=self._monotone(True))  # type: ignore[arg-type]
        reg_a = LGBMRegressor(**reg_params, monotone_constraints=self._monotone(False))  # type: ignore[arg-type]
        reg_h.fit(x_s, y_h, init_score=np.log(anchors))
        reg_a.fit(x_s, y_a, init_score=np.log(anchors))
        self._home_reg, self._away_reg = reg_h, reg_a

        # 2-way classifiers: ties are dropped from the targets
        clf_df = gbdt_df.filter(self.classifier_mask(gbdt_df))
        self._scaler = self._logit_clf = self._gbdt_clf = None
        y_c = self.binary_target(clf_df) if not clf_df.is_empty() else np.array([], dtype=np.int64)
        if y_c.size >= 20 and np.unique(y_c).size == 2:
            x_c = self._matrix(clf_df)
            if self.classifier_kind in ("logistic", "both"):
                scaler = StandardScaler().fit(x_c)
                lr = LogisticRegression(
                    C=0.05, solver="lbfgs", max_iter=500, random_state=self.random_seed
                )
                lr.fit(scaler.transform(x_c), y_c)
                self._scaler, self._logit_clf = scaler, lr
            if self.classifier_kind in ("gbdt", "both"):
                gb = LGBMClassifier(
                    objective="binary",
                    n_estimators=min(self.n_estimators, 120),
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    max_depth=3,
                    min_child_samples=max(10, min(40, x_c.shape[0] // 15)),
                    colsample_bytree=0.6,
                    subsample=0.8,
                    subsample_freq=1,
                    reg_alpha=1.0,
                    reg_lambda=10.0,
                    random_state=self.random_seed,
                    n_jobs=1,
                    verbose=-1,
                )
                gb.fit(x_c, y_c)
                self._gbdt_clf = gb
        self._is_fitted = True

        # Empirically estimated margin sigma (shrunk toward the 13.5-point prior)
        self.sigma_margin = PRIOR_SIGMA_MARGIN
        cal_scored = cal_df.filter(
            pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        )
        if not cal_scored.is_empty():
            mu_h, mu_a = self._expected_scores(cal_scored)
            resid = (_f64(cal_scored, "home_score", 0.0) - _f64(cal_scored, "away_score", 0.0)) - (
                mu_h - mu_a
            )
            n = float(resid.size)
            empirical = float(np.sqrt(np.mean(resid**2)))
            blended = (n * empirical + SIGMA_PRIOR_STRENGTH * PRIOR_SIGMA_MARGIN) / (
                n + SIGMA_PRIOR_STRENGTH
            )
            self.sigma_margin = float(np.clip(blended, *SIGMA_BOUNDS))

        # Calibrate on pooled later-season predictions (ties excluded)
        cal_clf = cal_df.filter(self.classifier_mask(cal_df)) if not cal_df.is_empty() else cal_df
        self.calibrator = self._new_calibrator()
        if cal_clf.is_empty():
            self.calibrator.fit(np.array([0.5]), np.array([1], dtype=np.int64))
        else:
            raw = self._raw_logit(cal_clf)[0]
            self.calibrator.fit(
                np.asarray(expit(raw), dtype=np.float64), self.binary_target(cal_clf)
            )
        return self

    # --------------------------------------------------------------- inference

    def _expected_scores(self, df: pl.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert self._home_reg is not None and self._away_reg is not None
        x = self._matrix(df)
        anchors = self._anchors(df)
        raw_h = np.asarray(self._home_reg.predict(x, raw_score=True), dtype=np.float64)
        raw_a = np.asarray(self._away_reg.predict(x, raw_score=True), dtype=np.float64)
        return anchors * np.exp(raw_h), anchors * np.exp(raw_a)

    def _classifier_prob(self, df: pl.DataFrame) -> NDArray[np.float64] | None:
        parts: list[NDArray[np.float64]] = []
        x = self._matrix(df)
        if self._logit_clf is not None and self._scaler is not None:
            parts.append(
                np.asarray(
                    np.asarray(self._logit_clf.predict_proba(self._scaler.transform(x)))[:, 1],
                    dtype=np.float64,
                )
            )
        if self._gbdt_clf is not None:
            parts.append(
                np.asarray(np.asarray(self._gbdt_clf.predict_proba(x))[:, 1], dtype=np.float64)
            )
        if not parts:
            return None
        return clip_probabilities(np.mean(np.vstack(parts), axis=0))

    def _raw_logit(
        self, df: pl.DataFrame
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return `(blended_logit, mu_home, mu_away)` for `df` in its given orientation."""
        mu_h, mu_a = self._expected_scores(df)
        p_score = clip_probabilities(np.asarray(norm.cdf((mu_h - mu_a) / self.sigma_margin)))
        w_elo, w_score, w_clf = self.weights
        p_clf = self._classifier_prob(df)
        z = w_elo * logit(self._elo_prob(df)) + w_score * logit(p_score)
        if p_clf is None:  # renormalize when no classifier could be fit
            return np.asarray(z / max(w_elo + w_score, 1e-12)), mu_h, mu_a
        z = z + w_clf * logit(p_clf)
        return np.asarray(z, dtype=np.float64), mu_h, mu_a

    def predict(self, features_df: pl.DataFrame) -> pl.DataFrame:
        """Return calibrated `P(Home)`, `P(Away) = 1 - P(Home)`, expected scores, and components."""
        if not self._is_fitted:
            raise RuntimeError("Stage1FootballGameModel must be fitted before predict().")
        df = sort_events_chronologically(features_df)
        n = df.height
        z, mu_h, mu_a = self._raw_logit(df)
        p_raw = clip_probabilities(np.asarray(expit(z), dtype=np.float64))
        p_cal = self.calibrator.predict_proba(p_raw)

        if "is_neutral_site" in df.columns and n > 0:
            neutral = df["is_neutral_site"].cast(pl.Int64).fill_null(0).to_numpy() == 1
        else:
            neutral = np.zeros(n, dtype=bool)
        if neutral.any():
            swapped = swap_home_away_features(df, self.feature_columns)
            z_s, mu_h_s, mu_a_s = self._raw_logit(swapped)
            p_raw_s = clip_probabilities(np.asarray(expit(z_s), dtype=np.float64))
            p_cal_s = self.calibrator.predict_proba(p_raw_s)
            p_cal = np.where(neutral, 0.5 * (p_cal + (1.0 - p_cal_s)), p_cal)
            p_raw = np.where(neutral, 0.5 * (p_raw + (1.0 - p_raw_s)), p_raw)
            mu_h, mu_a = (
                np.where(neutral, 0.5 * (mu_h + mu_a_s), mu_h),
                np.where(neutral, 0.5 * (mu_a + mu_h_s), mu_a),
            )
        p_home = clip_probabilities(p_cal)

        keep = [
            c
            for c in (
                "game_id",
                "event_seq",
                "season",
                "season_type",
                "week",
                "start_time_utc",
                "home_team",
                "away_team",
                "game_state",
                "is_neutral_site",
                "home_score",
                "away_score",
                "home_win",
                "is_tie",
                "elo_home_win_prob",
                "qb_elo_home_win_prob",
            )
            if c in df.columns
        ]
        return df.select(keep).with_columns(
            pl.Series("home_win_prob", p_home),
            pl.Series("away_win_prob", 1.0 - p_home),
            pl.lit(0.0).alias("draw_prob"),
            pl.Series("raw_home_win_prob", p_raw),
            pl.Series("expected_home_score", mu_h),
            pl.Series("expected_away_score", mu_a),
            pl.Series("expected_margin", mu_h - mu_a),
            pl.Series("expected_total", mu_h + mu_a),
        )


# ----------------------------------------------------------------------------------------
# Walk-forward backtest
# ----------------------------------------------------------------------------------------


class ScoreMetrics(BaseModel):
    """Expected-score accuracy against a league-mean baseline."""

    model_config = ConfigDict(frozen=True)

    n_games: int
    home_mae: float
    away_mae: float
    margin_mae: float
    total_mae: float
    home_rmse: float
    away_rmse: float
    margin_rmse: float
    total_rmse: float
    baseline_total_mae: float
    baseline_margin_mae: float


class BootstrapInterval(BaseModel):
    """Season-level bootstrap interval for a metric difference (negative = model better)."""

    model_config = ConfigDict(frozen=True)

    mean: float
    lower: float
    upper: float
    n_units: int


class SliceReport(BaseModel):
    """Model vs. baselines on a subset of games (e.g., playoffs or Weeks 1-4)."""

    model_config = ConfigDict(frozen=True)

    name: str
    n_games: int
    model: EvaluationMetrics
    qb_elo_baseline: EvaluationMetrics | None
    home_prior_baseline: EvaluationMetrics
    coin_flip_baseline: EvaluationMetrics


@dataclass(frozen=True)
class FootballStage1BacktestResult:
    """Out-of-sample walk-forward output (non-tie metrics; scores include all completed games)."""

    oof_predictions: pl.DataFrame
    benchmark_report: BenchmarkComparisonReport
    plain_elo_metrics: EvaluationMetrics | None
    score_metrics: ScoreMetrics
    slices: dict[str, SliceReport]
    era_home_win_rates: dict[str, float]
    log_loss_vs_home_prior: BootstrapInterval
    log_loss_vs_qb_elo: BootstrapInterval | None
    fold_audits: list[FoldBoundaryAudit]
    calibration_identity_folds: list[int] = field(default_factory=list)


def era_home_win_rates(df: pl.DataFrame) -> dict[str, float]:
    """Empirical home-win rate (ties excluded) per era."""
    out: dict[str, float] = {}
    mask = Stage1FootballGameModel.classifier_mask(df)
    for name, lo, hi in ERAS:
        sub = df.filter(mask & (pl.col("season") >= lo) & (pl.col("season") <= hi))
        if sub.height > 0:
            out[name] = float(sub.select(pl.col("home_win").cast(pl.Float64).mean()).item())
    return out


def _season_bootstrap(
    seasons: NDArray[np.int64],
    loss_model: NDArray[np.float64],
    loss_base: NDArray[np.float64],
    *,
    n_bootstrap: int,
    seed: int,
) -> BootstrapInterval:
    units = np.unique(seasons)
    diff_by_unit = {int(u): (loss_model[seasons == u] - loss_base[seasons == u]) for u in units}
    point = float(np.mean(loss_model - loss_base))
    if units.size < 2:
        return BootstrapInterval(mean=point, lower=point, upper=point, n_units=int(units.size))
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        picks = rng.choice(units, size=units.size, replace=True)
        means[b] = float(np.mean(np.concatenate([diff_by_unit[int(u)] for u in picks])))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return BootstrapInterval(mean=point, lower=float(lo), upper=float(hi), n_units=int(units.size))


def _per_game_log_loss(p: NDArray[np.float64], y: NDArray[np.int64]) -> NDArray[np.float64]:
    q = clip_probabilities(p)
    return np.asarray(-(y * np.log(q) + (1 - y) * np.log(1.0 - q)), dtype=np.float64)


def _score_metrics(oof: pl.DataFrame, league_mean: float) -> ScoreMetrics:
    s = oof.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    hs, as_ = _f64(s, "home_score", 0.0), _f64(s, "away_score", 0.0)
    ph, pa = _f64(s, "expected_home_score", 0.0), _f64(s, "expected_away_score", 0.0)

    def mae(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.mean(np.abs(a - b)))

    def rmse(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return ScoreMetrics(
        n_games=s.height,
        home_mae=mae(hs, ph),
        away_mae=mae(as_, pa),
        margin_mae=mae(hs - as_, ph - pa),
        total_mae=mae(hs + as_, ph + pa),
        home_rmse=rmse(hs, ph),
        away_rmse=rmse(as_, pa),
        margin_rmse=rmse(hs - as_, ph - pa),
        total_rmse=rmse(hs + as_, ph + pa),
        baseline_total_mae=mae(hs + as_, np.full(hs.size, 2.0 * league_mean)),
        baseline_margin_mae=mae(hs - as_, np.zeros(hs.size)),
    )


def _slice_report(name: str, oof: pl.DataFrame, home_rate: float) -> SliceReport | None:
    sub = oof.filter(Stage1FootballGameModel.classifier_mask(oof))
    if sub.height < 5:
        return None
    y = Stage1FootballGameModel.binary_target(sub)
    base = compare_against_baselines(
        _f64(sub, "home_win_prob", 0.5),
        y,
        train_home_win_rate=home_rate,
        elo_probs=_f64(sub, "qb_elo_home_win_prob", 0.5)
        if "qb_elo_home_win_prob" in sub.columns
        else None,
    )
    return SliceReport(
        name=name,
        n_games=sub.height,
        model=base.model,
        qb_elo_baseline=base.elo_baseline,
        home_prior_baseline=base.home_win_rate_baseline,
        coin_flip_baseline=base.coin_flip_baseline,
    )


def _make_folds(
    completed: pl.DataFrame,
    *,
    min_train_seasons: int,
    test_unit: Literal["season", "week_block"],
    week_block_size: int,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    seasons = sorted(set(completed["season"].to_list()))
    folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    for idx, season in enumerate(seasons):
        if idx < min_train_seasons:
            continue
        if test_unit == "season":
            folds.append(
                (
                    completed.filter(pl.col("season") < season),
                    completed.filter(pl.col("season") == season),
                )
            )
            continue
        weeks = sorted(set(completed.filter(pl.col("season") == season)["week"].to_list()))
        for i in range(0, len(weeks), week_block_size):
            block = weeks[i : i + week_block_size]
            block_df = completed.filter((pl.col("season") == season) & pl.col("week").is_in(block))
            start = block_df["start_time_utc"].min()
            folds.append((completed.filter(pl.col("start_time_utc") < start), block_df))
    return folds


def _home_rate(df: pl.DataFrame) -> float | None:
    sub = df.filter(Stage1FootballGameModel.classifier_mask(df))
    if sub.is_empty():
        return None
    return float(sub.select(pl.col("home_win").cast(pl.Float64).mean()).item())


def run_football_stage1_walk_forward_backtest(
    features_df: pl.DataFrame,
    *,
    min_train_seasons: int = 3,
    test_unit: Literal["season", "week_block"] = "season",
    week_block_size: int = 6,
    calibration_method: BinaryCalibrationMethod = "platt",
    min_calibration_samples: int = 400,
    n_bootstrap: int = 1000,
    random_seed: int = RANDOM_SEED,
) -> FootballStage1BacktestResult:
    """Expanding-window, week-aligned walk-forward backtest (train -> calibrate -> test).

    Every fold trains on games strictly before the test block, calibrates on the latest pooled
    prior seasons *after* the GBDT block, and tests on whole future seasons (or week blocks).
    Ties stay in Elo and score evaluation but are excluded from probability metrics.
    """
    completed = sort_events_chronologically(
        features_df.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    )
    folds = _make_folds(
        completed,
        min_train_seasons=min_train_seasons,
        test_unit=test_unit,
        week_block_size=week_block_size,
    )
    if not folds:
        raise ValueError(
            f"Need more than {min_train_seasons} seasons of completed games for a backtest."
        )

    oof_frames: list[pl.DataFrame] = []
    audits: list[FoldBoundaryAudit] = []
    identity_folds: list[int] = []
    for fold_idx, (train, test) in enumerate(folds):
        model = Stage1FootballGameModel(
            calibration_method=calibration_method,
            min_calibration_samples=min_calibration_samples,
            random_seed=random_seed,
        ).fit(train)
        fa = model.fit_audit_
        assert fa is not None
        test_min, test_max = test["start_time_utc"].min(), test["start_time_utc"].max()
        assert isinstance(test_min, datetime) and isinstance(test_max, datetime)
        if fa.calibrator_min_time is not None and fa.calibrator_max_time is not None:
            audits.append(
                FoldBoundaryAudit(
                    fold_index=fold_idx,
                    gbdt_train_max_time=fa.gbdt_train_max_time,
                    calibrator_min_time=fa.calibrator_min_time,
                    calibrator_max_time=fa.calibrator_max_time,
                    val_min_time=test_min,
                    val_max_time=test_max,
                )
            )
        if model.calibrator.is_identity:
            identity_folds.append(fold_idx)
        oof_frames.append(
            model.predict(test).with_columns(pl.lit(fold_idx, dtype=pl.Int32).alias("fold_index"))
        )
    oof = pl.concat(oof_frames, how="vertical")

    clf = oof.filter(Stage1FootballGameModel.classifier_mask(oof))
    y = Stage1FootballGameModel.binary_target(clf)
    p = _f64(clf, "home_win_prob", 0.5)
    first_test_season = int(oof.select(pl.col("season").min()).item())
    prior_rate = _home_rate(completed.filter(pl.col("season") < first_test_season))
    if prior_rate is None:
        prior_rate = float(np.mean(y))
    qb_elo = (
        _f64(clf, "qb_elo_home_win_prob", 0.5) if "qb_elo_home_win_prob" in clf.columns else None
    )
    report = compare_against_baselines(p, y, train_home_win_rate=prior_rate, elo_probs=qb_elo)
    plain_elo = (
        evaluate_probabilities(_f64(clf, "elo_home_win_prob", 0.5), y)
        if "elo_home_win_prob" in clf.columns
        else None
    )

    seasons = np.asarray(clf["season"].to_numpy(), dtype=np.int64)
    loss_m = _per_game_log_loss(p, y)
    ci_prior = _season_bootstrap(
        seasons,
        loss_m,
        _per_game_log_loss(np.full(p.size, prior_rate), y),
        n_bootstrap=n_bootstrap,
        seed=random_seed,
    )
    ci_elo = (
        _season_bootstrap(
            seasons,
            loss_m,
            _per_game_log_loss(qb_elo, y),
            n_bootstrap=n_bootstrap,
            seed=random_seed,
        )
        if qb_elo is not None
        else None
    )

    slices: dict[str, SliceReport] = {}
    candidates = {
        "playoffs": oof.filter(pl.col("season_type") != "REG"),
        "weeks_1_4": oof.filter((pl.col("season_type") == "REG") & (pl.col("week") <= 4)),
    }
    for name, frame in candidates.items():
        rep = _slice_report(name, frame, prior_rate)
        if rep is not None:
            slices[name] = rep

    league_mean = float(
        completed.select(((pl.col("home_score") + pl.col("away_score")) / 2.0).mean()).item()
    )
    return FootballStage1BacktestResult(
        oof_predictions=oof,
        benchmark_report=report,
        plain_elo_metrics=plain_elo,
        score_metrics=_score_metrics(oof, league_mean),
        slices=slices,
        era_home_win_rates=era_home_win_rates(completed),
        log_loss_vs_home_prior=ci_prior,
        log_loss_vs_qb_elo=ci_elo,
        fold_audits=audits,
        calibration_identity_folds=identity_folds,
    )


__all__ = [
    "BootstrapInterval",
    "FitBoundaryAudit",
    "FootballStage1BacktestResult",
    "HOME_MONOTONE_CONSTRAINTS",
    "ScoreMetrics",
    "SliceReport",
    "Stage1FootballGameModel",
    "era_home_win_rates",
    "run_football_stage1_walk_forward_backtest",
    "swap_home_away_features",
]
