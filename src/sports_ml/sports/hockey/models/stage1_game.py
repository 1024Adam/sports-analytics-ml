"""Stage 1 Mandatory Calibrated Game Win Probability pipeline (`models/stage1_game.py`).

Combines dynamic Elo baseline, LightGBM classifier, out-of-fold probability
calibration, and auxiliary Poisson team-goals pace estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray

from sports_ml.config import RANDOM_SEED
from sports_ml.core.calibration import CalibrationMethod, ProbabilityCalibrator, clip_probabilities
from sports_ml.core.evaluation import BenchmarkComparisonReport, compare_against_baselines
from sports_ml.core.temporal import sort_events_chronologically
from sports_ml.core.validation import WalkForwardValidator

STAGE1_CANDIDATE_FEATURE_COLS: Final[tuple[str, ...]] = (
    "elo_diff_pre",
    "elo_home_win_prob",
    "xgf_pct_5v5_w10_diff",
    "cf_pct_5v5_w10_diff",
    "st_net_edge_home_vs_away",
    "rest_advantage",
    "goalie_gsax_per_60_diff",
    "forward_ixg_per_60_diff",
    "defense_xgd_per_60_diff",
    "elo_home_pre",
    "elo_away_pre",
    "xgf_pct_5v5_w10_home",
    "xgf_pct_5v5_w10_away",
    "xgf_pct_5v5_ewma_home",
    "xgf_pct_5v5_ewma_away",
    "xgf_per_60_5v5_w10_home",
    "xgf_per_60_5v5_w10_away",
    "xga_per_60_5v5_w10_home",
    "xga_per_60_5v5_w10_away",
    "cf_pct_5v5_w10_home",
    "cf_pct_5v5_w10_away",
    "ff_pct_5v5_w10_home",
    "ff_pct_5v5_w10_away",
    "hd_xgf_pct_5v5_w10_home",
    "hd_xgf_pct_5v5_w10_away",
    "pdo_5v5_w10_home",
    "pdo_5v5_w10_away",
    "pdo_regression_signal_home",
    "pdo_regression_signal_away",
    "finishing_above_xg_w10_home",
    "finishing_above_xg_w10_away",
    "pp_xgf_per_60_home",
    "pp_xgf_per_60_away",
    "pk_xga_per_60_home",
    "pk_xga_per_60_away",
    "rest_days_home",
    "rest_days_away",
    "is_b2b_home",
    "is_b2b_away",
    "games_last_4d_home",
    "games_last_4d_away",
    "gsax_per_60_shrunk_home",
    "gsax_per_60_shrunk_away",
    "save_pct_shrunk_home",
    "save_pct_shrunk_away",
    "goalie_days_rest_home",
    "goalie_days_rest_away",
    "forward_weighted_ixg_per_60_home",
    "forward_weighted_ixg_per_60_away",
    "defense_weighted_xgd_per_60_home",
    "defense_weighted_xgd_per_60_away",
    "missing_regular_toi_home",
    "missing_regular_toi_away",
)


@dataclass(frozen=True)
class FoldBoundaryAudit:
    """Chronological boundary audit for a single walk-forward backtest fold."""

    fold_index: int
    gbdt_train_max_time: datetime
    calibrator_min_time: datetime
    calibrator_max_time: datetime
    val_min_time: datetime
    val_max_time: datetime


@dataclass(frozen=True)
class Stage1BacktestResult:
    """Out-of-sample walk-forward backtest output and chronological audit trail."""

    oof_predictions: pl.DataFrame
    benchmark_report: BenchmarkComparisonReport
    fold_audits: list[FoldBoundaryAudit]


class Stage1GameOutcomeModel:
    """Mandatory Stage 1 NHL game win probability and expected team-goals estimator."""

    def __init__(
        self,
        *,
        calibration_method: CalibrationMethod = "sigmoid",
        elo_blend_weight: float = 0.55,
        n_estimators: int = 80,
        learning_rate: float = 0.04,
        num_leaves: int = 12,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.calibration_method: CalibrationMethod = calibration_method
        self.elo_blend_weight = elo_blend_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_seed = random_seed

        self.feature_columns: list[str] = []
        self.calibrator = ProbabilityCalibrator(method=calibration_method, random_seed=random_seed)
        self._classifier: LGBMClassifier | None = None
        self._home_goals_regressor: LGBMRegressor | None = None
        self._away_goals_regressor: LGBMRegressor | None = None
        self._default_home_goals: float = 3.10
        self._default_away_goals: float = 2.90
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _build_monotone_constraints(self) -> list[int]:
        """Build LightGBM monotone constraints for Stage 1 matchup features."""
        pos_cols = {
            "elo_home_pre",
            "elo_diff_pre",
            "elo_home_win_prob",
            "xgf_pct_5v5_w10_home",
            "xgf_pct_5v5_w10_diff",
            "xgf_pct_5v5_ewma_home",
            "xgf_per_60_5v5_w10_home",
            "xga_per_60_5v5_w10_away",
            "cf_pct_5v5_w10_home",
            "cf_pct_5v5_w10_diff",
            "ff_pct_5v5_w10_home",
            "hd_xgf_pct_5v5_w10_home",
            "pp_xgf_per_60_home",
            "pk_xga_per_60_away",
            "st_net_edge_home_vs_away",
            "rest_days_home",
            "rest_advantage",
            "gsax_per_60_shrunk_home",
            "goalie_gsax_per_60_diff",
            "save_pct_shrunk_home",
            "forward_weighted_ixg_per_60_home",
            "forward_ixg_per_60_diff",
            "defense_weighted_xgd_per_60_home",
            "defense_xgd_per_60_diff",
            "missing_regular_toi_away",
        }
        neg_cols = {
            "elo_away_pre",
            "xgf_pct_5v5_w10_away",
            "xgf_pct_5v5_ewma_away",
            "xgf_per_60_5v5_w10_away",
            "xga_per_60_5v5_w10_home",
            "cf_pct_5v5_w10_away",
            "ff_pct_5v5_w10_away",
            "hd_xgf_pct_5v5_w10_away",
            "pp_xgf_per_60_away",
            "pk_xga_per_60_home",
            "rest_days_away",
            "gsax_per_60_shrunk_away",
            "save_pct_shrunk_away",
            "forward_weighted_ixg_per_60_away",
            "defense_weighted_xgd_per_60_away",
            "missing_regular_toi_home",
        }
        return [1 if c in pos_cols else (-1 if c in neg_cols else 0) for c in self.feature_columns]

    @staticmethod
    def resolve_feature_columns(df: pl.DataFrame) -> list[str]:
        """Select ordered numerical pre-game feature columns present in `df`."""
        candidate_set = set(STAGE1_CANDIDATE_FEATURE_COLS)
        preferred = [c for c in df.columns if c in candidate_set]
        if preferred:
            return preferred

        excluded = {
            "sport",
            "season",
            "game_type",
            "game_id",
            "event_id",
            "event_seq",
            "game_date",
            "start_time_utc",
            "home_team",
            "away_team",
            "venue_timezone",
            "game_state",
            "finish_period_type",
            "home_goals",
            "away_goals",
            "home_reg_goals",
            "away_reg_goals",
            "home_win",
            "elo_home_post",
            "elo_away_post",
        }
        fallback = [
            col
            for col, dtype in df.schema.items()
            if col not in excluded and (dtype.is_numeric() or dtype == pl.Boolean)
        ]
        if not fallback:
            raise ValueError("No valid numeric pre-game feature columns found in DataFrame.")
        return fallback

    def _extract_matrix(self, df: pl.DataFrame) -> NDArray[np.float64]:
        mat = df.select(
            [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in self.feature_columns]
        ).to_numpy()
        return np.asarray(mat, dtype=np.float64)

    def _extract_elo_probs(self, df: pl.DataFrame) -> NDArray[np.float64]:
        """Compute composite analytical win probability from Elo, 5v5 xGF%, GSAx, and roster."""
        if "elo_diff_pre" in df.columns:
            elo_diff = np.asarray(
                df["elo_diff_pre"].cast(pl.Float64).fill_null(35.0).to_numpy(),
                dtype=np.float64,
            )
        else:
            elo_diff = np.full(df.height, 35.0, dtype=np.float64)

        xgf_diff = (
            np.asarray(
                df["xgf_pct_5v5_w10_diff"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "xgf_pct_5v5_w10_diff" in df.columns
            else np.zeros(df.height, dtype=np.float64)
        )
        gsax_diff = (
            np.asarray(
                df["goalie_gsax_per_60_diff"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "goalie_gsax_per_60_diff" in df.columns
            else np.zeros(df.height, dtype=np.float64)
        )
        fwd_diff = (
            np.asarray(
                df["forward_ixg_per_60_diff"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "forward_ixg_per_60_diff" in df.columns
            else np.zeros(df.height, dtype=np.float64)
        )

        # Convert 5v5 xGF% diff (+5% -> +45 Elo), goalie GSAx diff (+0.5 -> +25 Elo),
        # and active roster forward ixG/60 diff (+0.2 -> +30 Elo) into effective Elo points
        composite_elo_diff = elo_diff + 900.0 * xgf_diff + 50.0 * gsax_diff + 150.0 * fwd_diff
        comp_probs = 1.0 / (1.0 + np.power(10.0, -composite_elo_diff / 400.0))
        return clip_probabilities(np.asarray(comp_probs, dtype=np.float64))

    def _raw_blended_probs(self, df: pl.DataFrame) -> NDArray[np.float64]:
        if self._classifier is None:
            raise RuntimeError("Stage1GameOutcomeModel must be fitted before prediction.")
        x_mat = self._extract_matrix(df)
        prob_mat = np.asarray(self._classifier.predict_proba(x_mat), dtype=np.float64)
        gbdt_probs = np.asarray(prob_mat[:, 1], dtype=np.float64)
        elo_probs = self._extract_elo_probs(df)
        has_elo = "elo_diff_pre" in df.columns or "elo_home_win_prob" in df.columns
        w = self.elo_blend_weight if has_elo else 0.0
        blended = (1.0 - w) * gbdt_probs + w * elo_probs
        return clip_probabilities(blended)

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame | None = None,
    ) -> Stage1GameOutcomeModel:
        """Fit GBDT classifier, team-goals regressors, and out-of-fold probability calibrator.

        Enforces strict chronological separation: `val_df` (used to fit the calibrator)
        must occur strictly after `train_df` (used to fit GBDT trees).
        """
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
        x_train = self._extract_matrix(gbdt_train_df)
        y_train = np.asarray(gbdt_train_df["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)

        min_child = max(2, min(20, gbdt_train_df.height // 6))
        leaves = max(3, min(self.num_leaves, max(3, gbdt_train_df.height // 3)))
        mono = self._build_monotone_constraints()

        clf = LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=leaves,
            min_child_samples=min_child,
            monotone_constraints=mono,
            random_state=self.random_seed,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(x_train, y_train)
        self._classifier = clf

        # Fit auxiliary Poisson expected team-goals regressors for Stage 2 pace conditioning
        home_g_col = (
            "home_reg_goals"
            if "home_reg_goals" in gbdt_train_df.columns
            else ("home_goals" if "home_goals" in gbdt_train_df.columns else None)
        )
        away_g_col = (
            "away_reg_goals"
            if "away_reg_goals" in gbdt_train_df.columns
            else ("away_goals" if "away_goals" in gbdt_train_df.columns else None)
        )

        if home_g_col is not None and away_g_col is not None:
            y_home_g = np.clip(
                np.asarray(gbdt_train_df[home_g_col].cast(pl.Float64).to_numpy(), dtype=np.float64),
                0.0,
                None,
            )
            y_away_g = np.clip(
                np.asarray(gbdt_train_df[away_g_col].cast(pl.Float64).to_numpy(), dtype=np.float64),
                0.0,
                None,
            )
            self._default_home_goals = max(0.5, float(np.mean(y_home_g)))
            self._default_away_goals = max(0.5, float(np.mean(y_away_g)))

            reg_h = LGBMRegressor(
                objective="poisson",
                n_estimators=max(30, self.n_estimators // 2),
                learning_rate=self.learning_rate,
                num_leaves=leaves,
                min_child_samples=min_child,
                random_state=self.random_seed,
                n_jobs=1,
                verbose=-1,
            )
            reg_a = LGBMRegressor(
                objective="poisson",
                n_estimators=max(30, self.n_estimators // 2),
                learning_rate=self.learning_rate,
                num_leaves=leaves,
                min_child_samples=min_child,
                random_state=self.random_seed,
                n_jobs=1,
                verbose=-1,
            )
            reg_h.fit(x_train, y_home_g)
            reg_a.fit(x_train, y_away_g)
            self._home_goals_regressor = reg_h
            self._away_goals_regressor = reg_a

        # Fit out-of-fold probability calibrator strictly on held-out chronological cal_val_df
        raw_val_probs = self._raw_blended_probs(cal_val_df)
        y_val = np.asarray(cal_val_df["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        self.calibrator.fit(raw_val_probs, y_val)
        self._is_fitted = True
        return self

    def predict(self, features_df: pl.DataFrame) -> pl.DataFrame:
        """Predict calibrated `home_win_prob`, `away_win_prob`, and expected team goals."""
        if not self._is_fitted:
            raise RuntimeError("Stage1GameOutcomeModel must be fitted before calling predict.")

        if features_df.is_empty():
            return features_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("home_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("away_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("expected_home_goals"),
                pl.lit(None, dtype=pl.Float64).alias("expected_away_goals"),
            )

        raw_probs = self._raw_blended_probs(features_df)
        cal_home_probs = clip_probabilities(self.calibrator.predict_proba(raw_probs))
        cal_away_probs = 1.0 - cal_home_probs

        x_mat = self._extract_matrix(features_df)
        if self._home_goals_regressor is not None and self._away_goals_regressor is not None:
            gbdt_h_goals = np.clip(
                np.asarray(self._home_goals_regressor.predict(x_mat), dtype=np.float64),
                0.50,
                8.0,
            )
            gbdt_a_goals = np.clip(
                np.asarray(self._away_goals_regressor.predict(x_mat), dtype=np.float64),
                0.50,
                8.0,
            )
            # Blend GBDT Poisson pace with win-probability-consistent goal split
            total_pace = np.clip(gbdt_h_goals + gbdt_a_goals, 4.5, 7.5)
            prob_implied_h = total_pace * (0.50 + 0.45 * (cal_home_probs - 0.50))
            prob_implied_a = total_pace - prob_implied_h
            pred_h_goals = np.clip(0.45 * gbdt_h_goals + 0.55 * prob_implied_h, 0.50, 8.0)
            pred_a_goals = np.clip(0.45 * gbdt_a_goals + 0.55 * prob_implied_a, 0.50, 8.0)
        else:
            pred_h_goals = np.full(features_df.height, self._default_home_goals, dtype=np.float64)
            pred_a_goals = np.full(features_df.height, self._default_away_goals, dtype=np.float64)

        return features_df.with_columns(
            pl.Series("home_win_prob", cal_home_probs, dtype=pl.Float64),
            pl.Series("away_win_prob", cal_away_probs, dtype=pl.Float64),
            pl.Series("expected_home_goals", pred_h_goals, dtype=pl.Float64),
            pl.Series("expected_away_goals", pred_a_goals, dtype=pl.Float64),
        )


def run_stage1_walk_forward_backtest(
    features_df: pl.DataFrame,
    *,
    n_splits: int = 3,
    min_train_groups: int = 6,
    calibration_method: CalibrationMethod = "sigmoid",
    elo_blend_weight: float = 0.25,
    random_seed: int = RANDOM_SEED,
) -> Stage1BacktestResult:
    """Execute walk-forward expanding time-series backtest for Stage 1 game outcome model."""
    validator = WalkForwardValidator(
        n_splits=n_splits,
        expanding=True,
        timestamp_col="start_time_utc",
        min_train_groups=min_train_groups,
    )
    splits = validator.split(features_df)

    oof_frames: list[pl.DataFrame] = []
    audits: list[FoldBoundaryAudit] = []

    for split in splits:
        train_fold = sort_events_chronologically(split.train_df)
        val_fold = sort_events_chronologically(split.val_df)

        unique_train_ts = train_fold["start_time_utc"].unique().sort().to_list()
        if len(unique_train_ts) < 2:
            raise ValueError("Each walk-forward training fold requires >= 2 unique timestamps.")

        cal_split_idx = max(1, int(len(unique_train_ts) * 0.75))
        cal_cutoff = unique_train_ts[cal_split_idx]

        gbdt_sub_train = train_fold.filter(pl.col("start_time_utc") < cal_cutoff)
        cal_sub_val = train_fold.filter(pl.col("start_time_utc") >= cal_cutoff)

        gbdt_max_ts = gbdt_sub_train["start_time_utc"].max()
        cal_min_ts = cal_sub_val["start_time_utc"].min()
        cal_max_ts = cal_sub_val["start_time_utc"].max()
        val_min_ts = val_fold["start_time_utc"].min()
        val_max_ts = val_fold["start_time_utc"].max()

        assert isinstance(gbdt_max_ts, datetime)
        assert isinstance(cal_min_ts, datetime) and isinstance(cal_max_ts, datetime)
        assert isinstance(val_min_ts, datetime) and isinstance(val_max_ts, datetime)

        audits.append(
            FoldBoundaryAudit(
                fold_index=split.fold_index,
                gbdt_train_max_time=gbdt_max_ts,
                calibrator_min_time=cal_min_ts,
                calibrator_max_time=cal_max_ts,
                val_min_time=val_min_ts,
                val_max_time=val_max_ts,
            )
        )

        model = Stage1GameOutcomeModel(
            calibration_method=calibration_method,
            elo_blend_weight=elo_blend_weight,
            random_seed=random_seed,
        )
        model.fit(gbdt_sub_train, cal_sub_val)
        pred_val = model.predict(val_fold).with_columns(
            pl.lit(split.fold_index, dtype=pl.Int32).alias("fold_index")
        )
        oof_frames.append(pred_val)

    oof_df = pl.concat(oof_frames, how="vertical")
    y_true = np.asarray(oof_df["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)
    model_probs = np.asarray(oof_df["home_win_prob"].to_numpy(), dtype=np.float64)
    elo_probs = (
        np.asarray(oof_df["elo_home_win_prob"].to_numpy(), dtype=np.float64)
        if "elo_home_win_prob" in oof_df.columns
        else None
    )

    first_train_arr = np.asarray(
        splits[0].train_df["home_win"].cast(pl.Float64).to_numpy(), dtype=np.float64
    )
    first_train_home_rate = float(np.mean(first_train_arr)) if first_train_arr.size > 0 else 0.535
    report = compare_against_baselines(
        model_probs,
        y_true,
        train_home_win_rate=first_train_home_rate,
        elo_probs=elo_probs,
    )
    return Stage1BacktestResult(
        oof_predictions=oof_df,
        benchmark_report=report,
        fold_audits=audits,
    )
