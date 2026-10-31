"""Stage 1 League-Aware Calibrated Game Outcome & Run Environment (`models/stage1_game.py`).

Implements:
1. Auxiliary Negative-Binomial / Tweedie team-run regressors (`objective="tweedie"`,
   `tweedie_variance_power=1.5`) estimating pre-game expected runs
   `(expected_home_runs, expected_away_runs)`.
2. Bivariate Negative-Binomial 9-inning + extra-inning run distribution model:
   - MLB (`2-Way Binary`): Unlimited extra innings with Manfred runner boost ->
     `P(Home Win) + P(Away Win) == 1.0`, `P(Draw) == 0.0`.
   - NPB (`3-Way 12-Inning 1X2`): Explicit 12-inning cap (3 extra innings: 10, 11, 12) ->
     `P(Home Win) + P(Draw) + P(Away Win) == 1.0`, plus conditional Draw-No-Bet
     `home_dnb_prob = P(Home Win) / (P(Home Win) + P(Away Win))`.
3. Out-of-fold probability calibration (`ProbabilityCalibrator` for MLB,
   `MulticlassProbabilityCalibrator` for NPB) and walk-forward backtesting with
   `FoldBoundaryAudit`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray

from sports_ml.config import RANDOM_SEED
from sports_ml.core.calibration import (
    CalibrationMethod,
    MulticlassProbabilityCalibrator,
    ProbabilityCalibrator,
    clip_probabilities,
    normalize_multiclass_probabilities,
)
from sports_ml.core.evaluation import (
    BenchmarkComparisonReport,
    MulticlassBenchmarkReport,
    compare_against_baselines,
    compare_multiclass_against_baselines,
    evaluate_probabilities,
)
from sports_ml.core.market import devig_probabilities, is_forbidden_market_column
from sports_ml.core.ratings import (
    compute_elo_1x2_probabilities,
    compute_elo_win_probability,
)
from sports_ml.core.temporal import sort_events_chronologically
from sports_ml.core.validation import WalkForwardValidator
from sports_ml.sports.baseball.features.pipeline import (
    BASEBALL_STAGE1_CANDIDATE_FEATURE_COLS,
)
from sports_ml.sports.hockey.models.stage1_game import FoldBoundaryAudit


def _neg_binom_pmf(max_k: int, mean_lam: float, dispersion_r: float) -> list[float]:
    """Compute Negative-Binomial PMF `P(Y = k)` for `k = 0..max_k` (`mean_lam`, size `r`)."""
    lam = max(0.05, float(mean_lam))
    r = max(0.5, float(dispersion_r))
    p = r / (r + lam)
    q = 1.0 - p

    probs: list[float] = []
    # P(0) = p^r
    p_k = math.pow(p, r)
    probs.append(p_k)
    for k in range(1, max_k + 1):
        p_k = p_k * ((k + r - 1.0) / float(k)) * q
        probs.append(p_k)

    total = sum(probs)
    if total <= 0.0:
        return [1.0] + [0.0] * max_k
    return [val / total for val in probs]


def compute_negative_binomial_game_probabilities(
    lambda_home: float,
    lambda_away: float,
    *,
    league: str = "MLB",
    dispersion_r: float = 4.2,
    max_runs: int = 22,
) -> tuple[float, float, float]:
    """Compute `(p_home_win, p_draw, p_away_win)` from Negative-Binomial run expectations.

    - For `league == "MLB"`: Models 9-inning regulation via Negative-Binomial PMFs and
      resolves regulation ties across unlimited extra innings (with Manfred runner scoring
      boost), returning `p_draw == 0.0` and `p_home_win + p_away_win == 1.0`.
    - For `league == "NPB"`: Models 9-inning regulation and 3 extra innings (innings 10, 11, 12).
      If both teams remain tied through 3 extra innings, the game ends in a 12-inning Draw/Tie
      (`p_draw > 0.0`), with `p_home_win + p_draw + p_away_win == 1.0`.
      By construction, `p_draw` is highest when `lambda_home == lambda_away` in low-scoring
      pitching duels (e.g., `lambda = 2.8` vs `lambda = 5.5`).
    """
    lh = max(0.50, min(float(lambda_home), 12.5))
    la = max(0.50, min(float(lambda_away), 12.5))
    lg = league.strip().upper()

    pmf_h = _neg_binom_pmf(max_runs, lh, dispersion_r)
    pmf_a = _neg_binom_pmf(max_runs, la, dispersion_r)

    p_reg_home = 0.0
    p_reg_tie = 0.0
    p_reg_away = 0.0

    for i in range(max_runs + 1):
        pi = pmf_h[i]
        for j in range(max_runs + 1):
            joint = pi * pmf_a[j]
            if i > j:
                p_reg_home += joint
            elif i == j:
                p_reg_tie += joint
            else:
                p_reg_away += joint

    reg_decisive = max(p_reg_home + p_reg_away, 1e-9)
    reg_home_share = p_reg_home / reg_decisive

    if lg != "NPB":
        # MLB unlimited extra innings: home team has a modest walk-off / last-ups edge in extras
        extra_home_share = min(max(0.512 + 0.65 * (reg_home_share - 0.50), 0.15), 0.85)
        p_home = p_reg_home + p_reg_tie * extra_home_share
        p_away = 1.0 - p_home
        return (float(p_home), 0.0, float(p_away))

    # NPB 12-inning cap: 3 extra innings (10th, 11th, 12th)
    lh_1 = lh / 9.0
    la_1 = la / 9.0
    r_1 = max(0.6, dispersion_r / 9.0)
    pmf_h_1 = _neg_binom_pmf(8, lh_1, r_1)
    pmf_a_1 = _neg_binom_pmf(8, la_1, r_1)

    p_1inn_home = 0.0
    p_1inn_tie = 0.0
    p_1inn_away = 0.0
    for i in range(9):
        for j in range(9):
            joint_1 = pmf_h_1[i] * pmf_a_1[j]
            if i > j:
                p_1inn_home += joint_1
            elif i == j:
                p_1inn_tie += joint_1
            else:
                p_1inn_away += joint_1

    # Probability all 3 extra innings (10, 11, 12) remain tied, penalized by team mismatch
    mismatch_penalty = math.exp(-0.28 * abs(lh - la))
    p_survive_3_extras = math.pow(p_1inn_tie, 2.0) * mismatch_penalty
    # Empirical NPB tactical late-inning bullpen/bunt factor (~0.52) matching ~3% overall ties
    p_draw_12 = min(max(p_reg_tie * p_survive_3_extras * 0.52, 0.004), 0.18)

    extra_decisive_mass = max(0.0, p_reg_tie - p_draw_12)
    inn_decisive = max(p_1inn_home + p_1inn_away, 1e-9)
    extra_home_share_npb = min(max(p_1inn_home / inn_decisive + 0.010, 0.15), 0.85)

    p_home_npb = p_reg_home + extra_decisive_mass * extra_home_share_npb
    p_away_npb = p_reg_away + extra_decisive_mass * (1.0 - extra_home_share_npb)

    total = p_home_npb + p_draw_12 + p_away_npb
    return (
        float(p_home_npb / total),
        float(p_draw_12 / total),
        float(p_away_npb / total),
    )


@dataclass(frozen=True)
class BaseballStage1BacktestResult:
    """Out-of-sample walk-forward backtest output for MLB (2-way) & NPB (3-way + DNB)."""

    oof_predictions: pl.DataFrame
    mlb_benchmark_report: BenchmarkComparisonReport | None
    npb_benchmark_report: MulticlassBenchmarkReport | None
    npb_dnb_log_loss: float | None
    fold_audits: list[FoldBoundaryAudit]


class Stage1BaseballGameModel:
    """Mandatory Stage 1 MLB (2-Way) & NPB (3-Way 12-Inning) outcome and run environment model."""

    def __init__(
        self,
        *,
        calibration_method: CalibrationMethod = "sigmoid",
        elo_blend_weight: float = 0.50,
        nb_run_blend_weight: float = 0.30,
        dispersion_r_mlb: float = 4.2,
        dispersion_r_npb: float = 4.6,
        npb_draw_param: float = 0.032,
        n_estimators: int = 60,
        learning_rate: float = 0.035,
        num_leaves: int = 8,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.calibration_method: CalibrationMethod = calibration_method
        self.elo_blend_weight = elo_blend_weight
        self.nb_run_blend_weight = nb_run_blend_weight
        self.dispersion_r_mlb = dispersion_r_mlb
        self.dispersion_r_npb = dispersion_r_npb
        self.npb_draw_param = npb_draw_param
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_seed = random_seed

        self.feature_columns: list[str] = []
        self.mlb_calibrator = ProbabilityCalibrator(
            method=calibration_method, random_seed=random_seed
        )
        self.npb_calibrator = MulticlassProbabilityCalibrator(
            method=calibration_method, n_classes=3, random_seed=random_seed
        )
        self._mlb_classifier: LGBMClassifier | None = None
        self._npb_classifier: LGBMClassifier | None = None
        self._home_runs_regressor: LGBMRegressor | None = None
        self._away_runs_regressor: LGBMRegressor | None = None
        self._default_mlb_runs: float = 4.45
        self._default_npb_runs: float = 3.55
        self._mlb_cal_fitted: bool = False
        self._npb_cal_fitted: bool = False
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @staticmethod
    def resolve_feature_columns(df: pl.DataFrame) -> list[str]:
        """Select ordered numerical pre-game feature columns (excluding `market_*` / `*odds*`)."""
        candidate_set = set(BASEBALL_STAGE1_CANDIDATE_FEATURE_COLS)
        preferred = [
            c for c in df.columns if c in candidate_set and not is_forbidden_market_column(c)
        ]
        if preferred:
            return preferred

        excluded = {
            "sport",
            "league",
            "sub_league",
            "season",
            "game_id",
            "event_id",
            "event_seq",
            "game_date",
            "start_time_utc",
            "home_team",
            "away_team",
            "venue_id",
            "venue_name",
            "venue_timezone",
            "game_state",
            "home_sp_id",
            "away_sp_id",
            "home_sp_hand",
            "away_sp_hand",
            "home_runs",
            "away_runs",
            "home_score",
            "away_score",
            "outcome_1x2",
            "home_win",
            "is_tie",
            "scheduled_innings",
            "innings_played",
            "elo_home_post",
            "elo_away_post",
        }
        fallback = [
            col
            for col, dtype in df.schema.items()
            if col not in excluded
            and not is_forbidden_market_column(col)
            and (dtype.is_numeric() or dtype == pl.Boolean)
        ]
        if not fallback:
            raise ValueError("No valid numeric pre-game feature columns found in DataFrame.")
        return fallback

    def _extract_matrix(self, df: pl.DataFrame) -> NDArray[np.float64]:
        mat = df.select(
            [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in self.feature_columns]
        ).to_numpy()
        return np.asarray(mat, dtype=np.float64)

    def _extract_composite_elo_diff(self, df: pl.DataFrame) -> NDArray[np.float64]:
        n = df.height
        elo_diff = (
            np.asarray(
                df["diff_elo_pre"].cast(pl.Float64).fill_null(24.0).to_numpy(),
                dtype=np.float64,
            )
            if "diff_elo_pre" in df.columns
            else (
                np.asarray(
                    df["elo_diff_pre"].cast(pl.Float64).fill_null(24.0).to_numpy(),
                    dtype=np.float64,
                )
                if "elo_diff_pre" in df.columns
                else np.full(n, 24.0, dtype=np.float64)
            )
        )
        sp_edge = (
            np.asarray(
                df["diff_sp_xfip_edge"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "diff_sp_xfip_edge" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        lineup_edge = (
            np.asarray(
                df["diff_lineup_platoon_woba"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "diff_lineup_platoon_woba" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        bp_edge = (
            np.asarray(
                df["diff_bp_xfip_edge"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "diff_bp_xfip_edge" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        return elo_diff + 28.0 * sp_edge + 650.0 * lineup_edge + 12.0 * bp_edge

    def _predict_expected_runs(
        self, df: pl.DataFrame
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict expected runs `(lam_home, lam_away)` using Tweedie + sabermetric prior."""
        n = df.height
        leagues = (
            [str(x).upper() for x in df["league"].to_list()]
            if "league" in df.columns
            else ["MLB"] * n
        )
        base_runs = np.array(
            [self._default_npb_runs if lg == "NPB" else self._default_mlb_runs for lg in leagues],
            dtype=np.float64,
        )
        venue_rf = (
            np.asarray(
                df["venue_run_factor"].cast(pl.Float64).fill_null(1.0).to_numpy(),
                dtype=np.float64,
            )
            if "venue_run_factor" in df.columns
            else np.ones(n, dtype=np.float64)
        )
        h_sp_xfip = (
            np.asarray(
                df["home_sp_xfip_shrunk"].cast(pl.Float64).fill_null(4.0).to_numpy(),
                dtype=np.float64,
            )
            if "home_sp_xfip_shrunk" in df.columns
            else np.full(n, 4.0, dtype=np.float64)
        )
        a_sp_xfip = (
            np.asarray(
                df["away_sp_xfip_shrunk"].cast(pl.Float64).fill_null(4.0).to_numpy(),
                dtype=np.float64,
            )
            if "away_sp_xfip_shrunk" in df.columns
            else np.full(n, 4.0, dtype=np.float64)
        )
        h_woba = (
            np.asarray(
                df["home_lineup_platoon_woba"].cast(pl.Float64).fill_null(0.312).to_numpy(),
                dtype=np.float64,
            )
            if "home_lineup_platoon_woba" in df.columns
            else np.full(n, 0.312, dtype=np.float64)
        )
        a_woba = (
            np.asarray(
                df["away_lineup_platoon_woba"].cast(pl.Float64).fill_null(0.312).to_numpy(),
                dtype=np.float64,
            )
            if "away_lineup_platoon_woba" in df.columns
            else np.full(n, 0.312, dtype=np.float64)
        )
        lg_woba_base = np.array([0.298 if lg == "NPB" else 0.315 for lg in leagues])
        lg_xfip_base = np.array([3.35 if lg == "NPB" else 4.15 for lg in leagues])

        prior_h = np.clip(
            (base_runs + 0.12 + 0.62 * (a_sp_xfip - lg_xfip_base) + 14.0 * (h_woba - lg_woba_base))
            * venue_rf,
            1.20,
            9.50,
        )
        prior_a = np.clip(
            (base_runs - 0.08 + 0.62 * (h_sp_xfip - lg_xfip_base) + 14.0 * (a_woba - lg_woba_base))
            * venue_rf,
            1.10,
            9.50,
        )

        if self._home_runs_regressor is not None and self._away_runs_regressor is not None:
            x_mat = self._extract_matrix(df)
            gbdt_h = np.clip(
                np.asarray(self._home_runs_regressor.predict(x_mat), dtype=np.float64),
                1.10,
                10.50,
            )
            gbdt_a = np.clip(
                np.asarray(self._away_runs_regressor.predict(x_mat), dtype=np.float64),
                1.10,
                10.50,
            )
            return (
                np.clip(0.50 * gbdt_h + 0.50 * prior_h, 1.10, 10.50),
                np.clip(0.50 * gbdt_a + 0.50 * prior_a, 1.10, 10.50),
            )
        return prior_h, prior_a

    def _raw_mlb_binary_probs(self, df: pl.DataFrame) -> NDArray[np.float64]:
        comp_diff = self._extract_composite_elo_diff(df)
        elo_p = np.array(
            [compute_elo_win_probability(float(d)) for d in comp_diff], dtype=np.float64
        )
        lam_h, lam_a = self._predict_expected_runs(df)
        nb_p = np.array(
            [
                compute_negative_binomial_game_probabilities(
                    float(lam_h[i]),
                    float(lam_a[i]),
                    league="MLB",
                    dispersion_r=self.dispersion_r_mlb,
                )[0]
                for i in range(df.height)
            ],
            dtype=np.float64,
        )
        if self._mlb_classifier is not None:
            x_mat = self._extract_matrix(df)
            proba_mat = np.asarray(self._mlb_classifier.predict_proba(x_mat), dtype=np.float64)
            gbdt_p = proba_mat[:, 1]
            w_elo = self.elo_blend_weight
            w_nb = self.nb_run_blend_weight
            w_gbdt = max(0.0, 1.0 - w_elo - w_nb)
            raw = w_gbdt * gbdt_p + w_elo * elo_p + w_nb * nb_p
        else:
            raw = 0.60 * elo_p + 0.40 * nb_p
        return clip_probabilities(raw)

    def _raw_npb_1x2_probs(self, df: pl.DataFrame) -> NDArray[np.float64]:
        comp_diff = self._extract_composite_elo_diff(df)
        elo_1x2 = np.zeros((df.height, 3), dtype=np.float64)
        for i in range(df.height):
            ph, pd, pa = compute_elo_1x2_probabilities(
                float(comp_diff[i]), draw_param=self.npb_draw_param
            )
            elo_1x2[i, 0] = ph
            elo_1x2[i, 1] = pd
            elo_1x2[i, 2] = pa

        lam_h, lam_a = self._predict_expected_runs(df)
        nb_1x2 = np.zeros((df.height, 3), dtype=np.float64)
        for i in range(df.height):
            ph, pd, pa = compute_negative_binomial_game_probabilities(
                float(lam_h[i]),
                float(lam_a[i]),
                league="NPB",
                dispersion_r=self.dispersion_r_npb,
            )
            nb_1x2[i, 0] = ph
            nb_1x2[i, 1] = pd
            nb_1x2[i, 2] = pa

        if self._npb_classifier is not None:
            x_mat = self._extract_matrix(df)
            gbdt_1x2 = normalize_multiclass_probabilities(
                np.asarray(self._npb_classifier.predict_proba(x_mat), dtype=np.float64)
            )
            w_elo = self.elo_blend_weight
            w_nb = self.nb_run_blend_weight
            w_gbdt = max(0.0, 1.0 - w_elo - w_nb)
            blended = w_gbdt * gbdt_1x2 + w_elo * elo_1x2 + w_nb * nb_1x2
        else:
            blended = 0.55 * elo_1x2 + 0.45 * nb_1x2
        return normalize_multiclass_probabilities(blended)

    @staticmethod
    def _extract_target_binary(df: pl.DataFrame) -> NDArray[np.int64]:
        if "home_win" in df.columns and df["home_win"].null_count() == 0:
            return np.asarray(df["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        if {"home_runs", "away_runs"}.issubset(df.columns):
            hr = np.asarray(df["home_runs"].cast(pl.Int64).to_numpy(), dtype=np.int64)
            ar = np.asarray(df["away_runs"].cast(pl.Int64).to_numpy(), dtype=np.int64)
            return (hr > ar).astype(np.int64)
        raise ValueError("DataFrame must contain 'home_win' or ('home_runs', 'away_runs').")

    @staticmethod
    def _extract_target_1x2(df: pl.DataFrame) -> NDArray[np.int64]:
        if "outcome_1x2" in df.columns and df["outcome_1x2"].null_count() == 0:
            return np.asarray(df["outcome_1x2"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        if {"home_runs", "away_runs"}.issubset(df.columns):
            hr = np.asarray(df["home_runs"].cast(pl.Int64).to_numpy(), dtype=np.int64)
            ar = np.asarray(df["away_runs"].cast(pl.Int64).to_numpy(), dtype=np.int64)
            return np.where(hr > ar, 0, np.where(hr == ar, 1, 2)).astype(np.int64)
        raise ValueError("DataFrame must contain 'outcome_1x2' or ('home_runs', 'away_runs').")

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame | None = None,
    ) -> Stage1BaseballGameModel:
        """Fit Tweedie run regressors, MLB/NPB LightGBM classifiers, and out-of-fold calibrators."""
        if train_df.is_empty():
            raise ValueError("train_df cannot be empty.")

        if val_df is None:
            sorted_all = sort_events_chronologically(train_df)
            unique_ts = sorted_all["start_time_utc"].unique().sort().to_list()
            if len(unique_ts) >= 4:
                split_idx = max(1, int(len(unique_ts) * 0.8))
                cutoff_ts = unique_ts[split_idx]
                gbdt_train_df = sorted_all.filter(pl.col("start_time_utc") < cutoff_ts)
                cal_val_df = sorted_all.filter(pl.col("start_time_utc") >= cutoff_ts)
            else:
                split_row = max(1, int(sorted_all.height * 0.8))
                gbdt_train_df = sorted_all.slice(0, split_row)
                cal_val_df = sorted_all.slice(split_row)
        else:
            if val_df.is_empty():
                raise ValueError("val_df cannot be empty when explicitly provided.")
            if "start_time_utc" in train_df.columns and "start_time_utc" in val_df.columns:
                max_train_ts = train_df["start_time_utc"].max()
                min_val_ts = val_df["start_time_utc"].min()
                if (
                    isinstance(max_train_ts, datetime)
                    and isinstance(min_val_ts, datetime)
                    and max_train_ts >= min_val_ts
                ):
                    raise ValueError(
                        f"Temporal leakage detected: train max timestamp ({max_train_ts}) "
                        f">= calibration validation min timestamp ({min_val_ts})."
                    )
            gbdt_train_df = train_df
            cal_val_df = val_df

        self.feature_columns = self.resolve_feature_columns(gbdt_train_df)
        x_all = self._extract_matrix(gbdt_train_df)

        min_child = max(4, min(30, x_all.shape[0] // 8))
        leaves = max(3, min(self.num_leaves, max(3, x_all.shape[0] // 6)))

        # 1. Fit Auxiliary Negative-Binomial / Tweedie Team-Run Regressors
        if {"home_runs", "away_runs"}.issubset(gbdt_train_df.columns):
            y_hr = np.clip(
                np.asarray(
                    gbdt_train_df["home_runs"].cast(pl.Float64).fill_null(4.2).to_numpy(),
                    dtype=np.float64,
                ),
                0.0,
                None,
            )
            y_ar = np.clip(
                np.asarray(
                    gbdt_train_df["away_runs"].cast(pl.Float64).fill_null(4.0).to_numpy(),
                    dtype=np.float64,
                ),
                0.0,
                None,
            )
            reg_h = LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                n_estimators=max(25, self.n_estimators // 2),
                learning_rate=self.learning_rate,
                num_leaves=leaves,
                min_child_samples=min_child,
                random_state=self.random_seed,
                n_jobs=1,
                verbose=-1,
            )
            reg_a = LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                n_estimators=max(25, self.n_estimators // 2),
                learning_rate=self.learning_rate,
                num_leaves=leaves,
                min_child_samples=min_child,
                random_state=self.random_seed,
                n_jobs=1,
                verbose=-1,
            )
            reg_h.fit(x_all, y_hr)
            reg_a.fit(x_all, y_ar)
            self._home_runs_regressor = reg_h
            self._away_runs_regressor = reg_a

        # 2. Fit Binary MLB Head + Calibrator
        mlb_train = (
            gbdt_train_df.filter(pl.col("league") == "MLB")
            if "league" in gbdt_train_df.columns
            else gbdt_train_df
        )
        if mlb_train.height == 0:
            mlb_train = gbdt_train_df

        x_mlb = self._extract_matrix(mlb_train)
        y_mlb = self._extract_target_binary(mlb_train)
        if len(np.unique(y_mlb)) < 2:
            x_mlb = np.vstack([x_mlb, x_mlb[:1]])
            y_mlb = np.concatenate([y_mlb, np.array([1 - y_mlb[0]], dtype=np.int64)])

        clf_mlb = LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=leaves,
            min_child_samples=max(3, min(25, x_mlb.shape[0] // 6)),
            colsample_bytree=0.75,
            subsample=0.80,
            subsample_freq=1,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=self.random_seed,
            n_jobs=1,
            verbose=-1,
        )
        clf_mlb.fit(x_mlb, y_mlb)
        self._mlb_classifier = clf_mlb

        mlb_val = (
            cal_val_df.filter(pl.col("league") == "MLB")
            if "league" in cal_val_df.columns
            else cal_val_df
        )
        if mlb_val.height == 0:
            mlb_val = cal_val_df
        raw_mlb_val = self._raw_mlb_binary_probs(mlb_val)
        y_mlb_val = self._extract_target_binary(mlb_val)
        self.mlb_calibrator.fit(raw_mlb_val, y_mlb_val)
        self._mlb_cal_fitted = True

        # 3. Fit 3-Way NPB Head + Multiclass Calibrator
        npb_train = (
            gbdt_train_df.filter(pl.col("league") == "NPB")
            if "league" in gbdt_train_df.columns
            else gbdt_train_df
        )
        if npb_train.height == 0:
            npb_train = gbdt_train_df

        x_npb = self._extract_matrix(npb_train)
        y_npb = self._extract_target_1x2(npb_train)
        present_c = set(np.unique(y_npb).tolist())
        if len(present_c) < 3:
            pad_x = [x_npb]
            pad_y = [y_npb]
            for mc in (0, 1, 2):
                if mc not in present_c:
                    pad_x.append(x_npb[:1])
                    pad_y.append(np.array([mc], dtype=np.int64))
            x_npb = np.vstack(pad_x)
            y_npb = np.concatenate(pad_y)

        clf_npb = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=leaves,
            min_child_samples=max(3, min(25, x_npb.shape[0] // 6)),
            colsample_bytree=0.75,
            subsample=0.80,
            subsample_freq=1,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=self.random_seed,
            n_jobs=1,
            verbose=-1,
        )
        clf_npb.fit(x_npb, y_npb)
        self._npb_classifier = clf_npb

        npb_val = (
            cal_val_df.filter(pl.col("league") == "NPB")
            if "league" in cal_val_df.columns
            else cal_val_df
        )
        if npb_val.height == 0:
            npb_val = cal_val_df
        raw_npb_val = self._raw_npb_1x2_probs(npb_val)
        y_npb_val = self._extract_target_1x2(npb_val)
        self.npb_calibrator.fit(raw_npb_val, y_npb_val)
        self._npb_cal_fitted = True

        self._is_fitted = True
        return self

    def predict(self, features_df: pl.DataFrame) -> pl.DataFrame:
        """Predict calibrated outcome probabilities (`2-way MLB` & `3-way NPB`) & expected runs."""
        if not self._is_fitted:
            raise RuntimeError("Stage1BaseballGameModel must be fitted before calling predict.")

        if features_df.is_empty():
            return features_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("home_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("draw_prob"),
                pl.lit(None, dtype=pl.Float64).alias("away_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("home_dnb_prob"),
                pl.lit(None, dtype=pl.Float64).alias("away_dnb_prob"),
                pl.lit(None, dtype=pl.Float64).alias("expected_home_runs"),
                pl.lit(None, dtype=pl.Float64).alias("expected_away_runs"),
            )

        n = features_df.height
        leagues = (
            [str(x).upper() for x in features_df["league"].to_list()]
            if "league" in features_df.columns
            else ["MLB"] * n
        )

        # Compute MLB calibrated 2-way probabilities and NPB calibrated 3-way probabilities
        raw_mlb = self._raw_mlb_binary_probs(features_df)
        cal_mlb_home = clip_probabilities(self.mlb_calibrator.predict_proba(raw_mlb))

        raw_npb_1x2 = self._raw_npb_1x2_probs(features_df)
        cal_npb_1x2 = normalize_multiclass_probabilities(
            self.npb_calibrator.predict_proba(raw_npb_1x2)
        )

        lam_h, lam_a = self._predict_expected_runs(features_df)

        p_home_out = np.zeros(n, dtype=np.float64)
        p_draw_out = np.zeros(n, dtype=np.float64)
        p_away_out = np.zeros(n, dtype=np.float64)
        p_home_dnb_out = np.zeros(n, dtype=np.float64)
        p_away_dnb_out = np.zeros(n, dtype=np.float64)
        exp_h_out = np.zeros(n, dtype=np.float64)
        exp_a_out = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if leagues[i] == "NPB":
                # Blend calibrated 1X2 with Negative-Binomial 12-inning structural tie sensitivity
                nb_ph, nb_pd, nb_pa = compute_negative_binomial_game_probabilities(
                    float(lam_h[i]),
                    float(lam_a[i]),
                    league="NPB",
                    dispersion_r=self.dispersion_r_npb,
                )
                ph = 0.65 * float(cal_npb_1x2[i, 0]) + 0.35 * nb_ph
                pd = 0.45 * float(cal_npb_1x2[i, 1]) + 0.55 * nb_pd
                pa = 0.65 * float(cal_npb_1x2[i, 2]) + 0.35 * nb_pa
                tot = ph + pd + pa
                ph = float(np.clip(ph / tot, 1e-6, 1.0 - 2e-6))
                pd = float(np.clip(pd / tot, 1e-6, 1.0 - ph - 1e-6))
                pa = 1.0 - ph - pd

                p_home_out[i] = ph
                p_draw_out[i] = pd
                p_away_out[i] = pa
                decisive = max(ph + pa, 1e-9)
                p_home_dnb_out[i] = ph / decisive
                p_away_dnb_out[i] = 1.0 - p_home_dnb_out[i]
            else:
                nb_ph, _, _ = compute_negative_binomial_game_probabilities(
                    float(lam_h[i]),
                    float(lam_a[i]),
                    league="MLB",
                    dispersion_r=self.dispersion_r_mlb,
                )
                ph = float(np.clip(0.75 * float(cal_mlb_home[i]) + 0.25 * nb_ph, 1e-6, 1.0 - 1e-6))
                pa = 1.0 - ph
                p_home_out[i] = ph
                p_draw_out[i] = 0.0
                p_away_out[i] = pa
                p_home_dnb_out[i] = ph
                p_away_dnb_out[i] = pa

            # Reconcile expected runs with win probability direction
            tot_runs = float(np.clip(lam_h[i] + lam_a[i], 3.0, 16.0))
            share_h = float(
                np.clip(0.50 + 0.35 * (p_home_dnb_out[i] - p_away_dnb_out[i]), 0.25, 0.75)
            )
            exp_h_out[i] = float(np.clip(0.60 * lam_h[i] + 0.40 * (tot_runs * share_h), 1.10, 10.5))
            exp_a_out[i] = float(
                np.clip(0.60 * lam_a[i] + 0.40 * (tot_runs * (1.0 - share_h)), 1.10, 10.5)
            )

        return features_df.with_columns(
            pl.Series("home_win_prob", p_home_out, dtype=pl.Float64),
            pl.Series("draw_prob", p_draw_out, dtype=pl.Float64),
            pl.Series("away_win_prob", p_away_out, dtype=pl.Float64),
            pl.Series("home_dnb_prob", p_home_dnb_out, dtype=pl.Float64),
            pl.Series("away_dnb_prob", p_away_dnb_out, dtype=pl.Float64),
            pl.Series("expected_home_runs", exp_h_out, dtype=pl.Float64),
            pl.Series("expected_away_runs", exp_a_out, dtype=pl.Float64),
            pl.Series("stage1_home_win_prob", p_home_out, dtype=pl.Float64),
            pl.Series("stage1_away_win_prob", p_away_out, dtype=pl.Float64),
            pl.Series("stage1_expected_home_runs", exp_h_out, dtype=pl.Float64),
            pl.Series("stage1_expected_away_runs", exp_a_out, dtype=pl.Float64),
        )


def run_baseball_stage1_walk_forward_backtest(
    features_df: pl.DataFrame,
    *,
    n_splits: int = 3,
    min_train_groups: int = 8,
    calibration_method: CalibrationMethod = "sigmoid",
    random_seed: int = RANDOM_SEED,
) -> BaseballStage1BacktestResult:
    """Execute leak-free walk-forward backtest for MLB (2-Way) & NPB (3-Way + DNB)."""
    completed = features_df.filter(
        pl.col("outcome_1x2").is_not_null()
        if "outcome_1x2" in features_df.columns
        else pl.col("home_runs").is_not_null()
    )
    sorted_df = sort_events_chronologically(completed)

    validator = WalkForwardValidator(
        n_splits=n_splits,
        min_train_groups=min_train_groups,
        expanding=True,
    )
    splits = validator.split(sorted_df)

    oof_frames: list[pl.DataFrame] = []
    fold_audits: list[FoldBoundaryAudit] = []

    for split in splits:
        fold_train = sort_events_chronologically(split.train_df)
        fold_val = sort_events_chronologically(split.val_df)

        unique_train_ts = fold_train["start_time_utc"].unique().sort().to_list()
        if len(unique_train_ts) < 2:
            continue

        cal_split_idx = max(1, int(len(unique_train_ts) * 0.8))
        cal_cutoff_ts = unique_train_ts[cal_split_idx]
        gbdt_sub_train = fold_train.filter(pl.col("start_time_utc") < cal_cutoff_ts)
        cal_sub_val = fold_train.filter(pl.col("start_time_utc") >= cal_cutoff_ts)

        gbdt_max_ts = gbdt_sub_train["start_time_utc"].max()
        cal_min_ts = cal_sub_val["start_time_utc"].min()
        cal_max_ts = cal_sub_val["start_time_utc"].max()
        val_min_ts = fold_val["start_time_utc"].min()
        val_max_ts = fold_val["start_time_utc"].max()

        assert isinstance(gbdt_max_ts, datetime)
        assert isinstance(cal_min_ts, datetime)
        assert isinstance(cal_max_ts, datetime)
        assert isinstance(val_min_ts, datetime)
        assert isinstance(val_max_ts, datetime)

        fold_audits.append(
            FoldBoundaryAudit(
                fold_index=split.fold_index,
                gbdt_train_max_time=gbdt_max_ts,
                calibrator_min_time=cal_min_ts,
                calibrator_max_time=cal_max_ts,
                val_min_time=val_min_ts,
                val_max_time=val_max_ts,
            )
        )

        model = Stage1BaseballGameModel(
            calibration_method=calibration_method,
            random_seed=random_seed,
        )
        model.fit(gbdt_sub_train, cal_sub_val)
        preds = model.predict(fold_val).with_columns(
            pl.lit(split.fold_index, dtype=pl.Int32).alias("fold_index")
        )
        oof_frames.append(preds)

    if not oof_frames:
        raise ValueError("No valid walk-forward splits produced out-of-fold predictions.")

    oof_df = pl.concat(oof_frames, how="vertical")

    # 1. MLB 2-Way Evaluation
    mlb_oof = oof_df.filter(pl.col("league") == "MLB") if "league" in oof_df.columns else oof_df
    mlb_report: BenchmarkComparisonReport | None = None
    if not mlb_oof.is_empty():
        y_mlb = Stage1BaseballGameModel._extract_target_binary(mlb_oof)
        p_mlb = np.asarray(mlb_oof["home_win_prob"].to_numpy(), dtype=np.float64)
        elo_mlb = (
            np.asarray(mlb_oof["elo_home_win_prob"].to_numpy(), dtype=np.float64)
            if "elo_home_win_prob" in mlb_oof.columns
            else None
        )
        mkt_p_mlb: NDArray[np.float64] | None = None
        mkt_odds_mlb: NDArray[np.float64] | None = None
        if {"market_prob_home", "market_prob_away"}.issubset(mlb_oof.columns):
            mkt_p_mlb = np.asarray(
                mlb_oof["market_prob_home"].cast(pl.Float64).to_numpy(), dtype=np.float64
            )
            if {"market_home_decimal_odds", "market_away_decimal_odds"}.issubset(mlb_oof.columns):
                mkt_odds_mlb = np.column_stack(
                    [
                        mlb_oof["market_home_decimal_odds"].cast(pl.Float64).to_numpy(),
                        mlb_oof["market_away_decimal_odds"].cast(pl.Float64).to_numpy(),
                    ]
                )
        elif {"market_home_decimal_odds", "market_away_decimal_odds"}.issubset(mlb_oof.columns):
            raw_o = np.column_stack(
                [
                    mlb_oof["market_home_decimal_odds"].cast(pl.Float64).to_numpy(),
                    mlb_oof["market_away_decimal_odds"].cast(pl.Float64).to_numpy(),
                ]
            )
            mkt_odds_mlb = raw_o
            valid_mask = np.all(np.isfinite(raw_o) & (raw_o > 1.0), axis=1)
            mkt_p_mlb = np.full(mlb_oof.height, np.nan, dtype=np.float64)
            if np.any(valid_mask):
                mkt_p_mlb[valid_mask] = devig_probabilities(raw_o[valid_mask], method="power")[:, 0]

        mlb_report = compare_against_baselines(
            p_mlb,
            y_mlb,
            train_home_win_rate=float(np.mean(y_mlb)),
            elo_probs=elo_mlb,
            market_probs=mkt_p_mlb,
            market_decimal_odds=mkt_odds_mlb,
        )

    # 2. NPB 3-Way (1X2 + RPS + DNB) Evaluation
    npb_oof = (
        oof_df.filter(pl.col("league") == "NPB") if "league" in oof_df.columns else pl.DataFrame()
    )
    npb_report: MulticlassBenchmarkReport | None = None
    npb_dnb_ll: float | None = None
    if not npb_oof.is_empty():
        y_npb = Stage1BaseballGameModel._extract_target_1x2(npb_oof)
        p_npb = np.column_stack(
            [
                npb_oof["home_win_prob"].to_numpy(),
                npb_oof["draw_prob"].to_numpy(),
                npb_oof["away_win_prob"].to_numpy(),
            ]
        )
        elo_npb: NDArray[np.float64] | None = None
        if {"elo_home_win_prob", "elo_draw_prob", "elo_away_win_prob"}.issubset(npb_oof.columns):
            elo_npb = np.column_stack(
                [
                    npb_oof["elo_home_win_prob"].to_numpy(),
                    npb_oof["elo_draw_prob"].to_numpy(),
                    npb_oof["elo_away_win_prob"].to_numpy(),
                ]
            )
        counts = np.bincount(y_npb, minlength=3).astype(np.float64)
        emp_prior = (counts + 0.5) / np.sum(counts + 0.5)

        mkt_p_npb: NDArray[np.float64] | None = None
        mkt_odds_npb: NDArray[np.float64] | None = None
        if {"market_prob_home", "market_prob_draw", "market_prob_away"}.issubset(npb_oof.columns):
            mkt_p_npb = np.column_stack(
                [
                    npb_oof["market_prob_home"].cast(pl.Float64).to_numpy(),
                    npb_oof["market_prob_draw"].cast(pl.Float64).to_numpy(),
                    npb_oof["market_prob_away"].cast(pl.Float64).to_numpy(),
                ]
            )
            if {
                "market_home_decimal_odds",
                "market_draw_decimal_odds",
                "market_away_decimal_odds",
            }.issubset(npb_oof.columns):
                mkt_odds_npb = np.column_stack(
                    [
                        npb_oof["market_home_decimal_odds"].cast(pl.Float64).to_numpy(),
                        npb_oof["market_draw_decimal_odds"].cast(pl.Float64).to_numpy(),
                        npb_oof["market_away_decimal_odds"].cast(pl.Float64).to_numpy(),
                    ]
                )

        npb_report = compare_multiclass_against_baselines(
            p_npb,
            y_npb,
            elo_probs=elo_npb,
            empirical_prior=(float(emp_prior[0]), float(emp_prior[1]), float(emp_prior[2])),
            market_probs=mkt_p_npb,
            market_decimal_odds=mkt_odds_npb,
        )
        decisive_mask = y_npb != 1
        if np.any(decisive_mask):
            y_dnb = (y_npb[decisive_mask] == 0).astype(np.int64)
            p_dnb = np.asarray(npb_oof["home_dnb_prob"].to_numpy()[decisive_mask], dtype=np.float64)
            npb_dnb_ll = evaluate_probabilities(p_dnb, y_dnb).log_loss

    return BaseballStage1BacktestResult(
        oof_predictions=oof_df,
        mlb_benchmark_report=mlb_report,
        npb_benchmark_report=npb_report,
        npb_dnb_log_loss=npb_dnb_ll,
        fold_audits=fold_audits,
    )
