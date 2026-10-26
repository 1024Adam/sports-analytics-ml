"""Stage 1 Mandatory Calibrated 3-Way (`1X2`) & Goals Environment pipeline (`stage1_match.py`).

Combines:
1. Auxiliary Poisson team-goals regressors (`expected_home_goals`, `expected_away_goals`)
   paired with a Dixon-Coles bivariate Poisson low-score correlation adjustment (`rho`)
   to derive structural scoreline and goal-implied `1X2` probabilities.
2. Hybrid 3-Way Outcome Ensemble blending analytical 3-way Elo probabilities,
   Dixon-Coles goal-implied `1X2` probabilities, and a Multiclass LightGBM Classifier
   (`objective="multiclass"`, `num_class=3`).
3. Out-of-fold 3-way calibration (`MulticlassProbabilityCalibrator`) fitted strictly on
   held-out chronological validation folds so `P(Home Win) + P(Draw) + P(Away Win) == 1.0`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray

from sports_ml.config import RANDOM_SEED
from sports_ml.core.calibration import (
    CalibrationMethod,
    MulticlassProbabilityCalibrator,
    normalize_multiclass_probabilities,
)
from sports_ml.core.evaluation import (
    MulticlassBenchmarkReport,
    compare_multiclass_against_baselines,
)
from sports_ml.core.ratings import compute_elo_1x2_probabilities
from sports_ml.core.temporal import sort_events_chronologically
from sports_ml.core.validation import WalkForwardValidator
from sports_ml.sports.hockey.models.stage1_game import FoldBoundaryAudit

SOCCER_STAGE1_CANDIDATE_FEATURE_COLS: Final[tuple[str, ...]] = (
    "elo_diff_pre",
    "elo_home_win_prob",
    "elo_draw_prob",
    "elo_away_win_prob",
    "npxgd_per_90_w10_diff",
    "npxgf_pct_w10_diff",
    "pressing_intensity_diff",
    "deep_completions_per_90_w10_diff",
    "rest_advantage",
    "gk_gsaa_per_90_diff",
    "fwd_npxg_per_90_diff",
    "mid_xg_chain_per_90_diff",
    "elo_home_pre",
    "elo_away_pre",
    "npxgf_per_90_w10_home",
    "npxgf_per_90_w10_away",
    "npxga_per_90_w10_home",
    "npxga_per_90_w10_away",
    "npxgd_per_90_w10_home",
    "npxgd_per_90_w10_away",
    "npxgf_pct_w10_home",
    "npxgf_pct_w10_away",
    "npxgf_pct_ewma_home",
    "npxgf_pct_ewma_away",
    "npxgd_per_90_ewma_home",
    "npxgd_per_90_ewma_away",
    "xgf_per_90_w10_home",
    "xgf_per_90_w10_away",
    "xga_per_90_w10_home",
    "xga_per_90_w10_away",
    "shots_per_90_w10_home",
    "shots_per_90_w10_away",
    "sot_pct_w10_home",
    "sot_pct_w10_away",
    "ppda_w10_home",
    "ppda_w10_away",
    "opp_ppda_w10_home",
    "opp_ppda_w10_away",
    "deep_completions_per_90_w10_home",
    "deep_completions_per_90_w10_away",
    "corner_share_w10_home",
    "corner_share_w10_away",
    "finishing_above_xg_w10_home",
    "finishing_above_xg_w10_away",
    "defensive_ga_above_xga_w10_home",
    "defensive_ga_above_xga_w10_away",
    "rest_days_home",
    "rest_days_away",
    "is_midweek_turnaround_home",
    "is_midweek_turnaround_away",
    "matches_last_14d_home",
    "matches_last_14d_away",
    "travel_km_prev_game_home",
    "travel_km_prev_game_away",
    "is_promoted_team_home",
    "is_promoted_team_away",
    "gsaa_per_90_shrunk_home",
    "gsaa_per_90_shrunk_away",
    "save_pct_shrunk_home",
    "save_pct_shrunk_away",
    "clean_sheet_rate_shrunk_home",
    "clean_sheet_rate_shrunk_away",
    "gk_days_rest_home",
    "gk_days_rest_away",
    "fwd_weighted_npxg_per_90_home",
    "fwd_weighted_npxg_per_90_away",
    "fwd_weighted_xa_per_90_home",
    "fwd_weighted_xa_per_90_away",
    "mid_weighted_npxg_per_90_home",
    "mid_weighted_npxg_per_90_away",
    "mid_weighted_xa_per_90_home",
    "mid_weighted_xa_per_90_away",
    "mid_weighted_xg_chain_per_90_home",
    "mid_weighted_xg_chain_per_90_away",
    "def_weighted_xg_buildup_per_90_home",
    "def_weighted_xg_buildup_per_90_away",
    "missing_regular_minutes_home",
    "missing_regular_minutes_away",
)


def compute_dixon_coles_1x2_probabilities(
    lambda_home: float,
    lambda_away: float,
    *,
    rho: float = -0.11,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """Compute 3-way (`1X2`) probabilities from expected goals with Dixon-Coles correlation `rho`.

    Dixon & Coles (1997) low-score adjustment `tau(i, j)` for `0 <= i, j <= 1`:
        `tau(0, 0) = 1 - lambda_home * lambda_away * rho`
        `tau(0, 1) = 1 + lambda_home * rho`
        `tau(1, 0) = 1 + lambda_away * rho`
        `tau(1, 1) = 1 - rho`
        `tau(i, j) = 1` otherwise.
    In soccer, `rho < 0` (typically `-0.13 <= rho <= -0.08`) increases the probability of
    `0-0` and `1-1` draws relative to independent Poisson marginals.
    """
    lh = max(0.15, min(float(lambda_home), 6.5))
    la = max(0.15, min(float(lambda_away), 6.5))

    p_h_marginal = [math.exp(-lh) * (lh**i) / math.factorial(i) for i in range(max_goals + 1)]
    p_a_marginal = [math.exp(-la) * (la**j) / math.factorial(j) for j in range(max_goals + 1)]

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            base_p = p_h_marginal[i] * p_a_marginal[j]
            if i == 0 and j == 0:
                tau = max(0.05, 1.0 - lh * la * rho)
            elif i == 0 and j == 1:
                tau = max(0.05, 1.0 + lh * rho)
            elif i == 1 and j == 0:
                tau = max(0.05, 1.0 + la * rho)
            elif i == 1 and j == 1:
                tau = max(0.05, 1.0 - rho)
            else:
                tau = 1.0

            joint_p = base_p * tau
            if i > j:
                p_home += joint_p
            elif i == j:
                p_draw += joint_p
            else:
                p_away += joint_p

    total = p_home + p_draw + p_away
    if total <= 0.0:
        return (0.45, 0.25, 0.30)
    return (p_home / total, p_draw / total, p_away / total)


@dataclass(frozen=True)
class SoccerStage1BacktestResult:
    """Out-of-sample walk-forward backtest output and chronological audit trail for EPL Stage 1."""

    oof_predictions: pl.DataFrame
    benchmark_report: MulticlassBenchmarkReport
    fold_audits: list[FoldBoundaryAudit]


class Stage1SoccerMatchModel:
    """Mandatory Stage 1 EPL 3-way (`1X2`) match outcome and expected team-goals estimator."""

    def __init__(
        self,
        *,
        calibration_method: CalibrationMethod = "sigmoid",
        elo_blend_weight: float = 0.55,
        dixon_coles_blend_weight: float = 0.30,
        dixon_coles_rho: float = -0.11,
        draw_param: float = 0.26,
        n_estimators: int = 55,
        learning_rate: float = 0.03,
        num_leaves: int = 7,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.calibration_method: CalibrationMethod = calibration_method
        self.elo_blend_weight = elo_blend_weight
        self.dixon_coles_blend_weight = dixon_coles_blend_weight
        self.dixon_coles_rho = dixon_coles_rho
        self.draw_param = draw_param
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_seed = random_seed

        self.feature_columns: list[str] = []
        self.calibrator = MulticlassProbabilityCalibrator(
            method=calibration_method, n_classes=3, random_seed=random_seed
        )
        self._classifier: LGBMClassifier | None = None
        self._home_goals_regressor: LGBMRegressor | None = None
        self._away_goals_regressor: LGBMRegressor | None = None
        self._default_home_goals: float = 1.55
        self._default_away_goals: float = 1.28
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @staticmethod
    def resolve_feature_columns(df: pl.DataFrame) -> list[str]:
        """Select ordered numerical pre-match feature columns present in `df`."""
        candidate_set = set(SOCCER_STAGE1_CANDIDATE_FEATURE_COLS)
        preferred = [c for c in df.columns if c in candidate_set]
        if preferred:
            return preferred

        excluded = {
            "sport",
            "season",
            "matchweek",
            "match_id",
            "game_id",
            "event_id",
            "event_seq",
            "match_date",
            "game_date",
            "start_time_utc",
            "home_team",
            "away_team",
            "venue_timezone",
            "game_state",
            "home_goals",
            "away_goals",
            "outcome_1x2",
            "home_win",
            "is_draw",
            "away_win",
            "elo_home_post",
            "elo_away_post",
        }
        fallback = [
            col
            for col, dtype in df.schema.items()
            if col not in excluded and (dtype.is_numeric() or dtype == pl.Boolean)
        ]
        if not fallback:
            raise ValueError("No valid numeric pre-match feature columns found in DataFrame.")
        return fallback

    def _extract_matrix(self, df: pl.DataFrame) -> NDArray[np.float64]:
        mat = df.select(
            [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in self.feature_columns]
        ).to_numpy()
        return np.asarray(mat, dtype=np.float64)

    def _extract_composite_elo_1x2(self, df: pl.DataFrame) -> NDArray[np.float64]:
        """Compute composite 3-way (`1X2`) probabilities from Elo, npxGD, GK GSAA, and FWD npxG."""
        n = df.height
        elo_diff = (
            np.asarray(
                df["elo_diff_pre"].cast(pl.Float64).fill_null(48.0).to_numpy(),
                dtype=np.float64,
            )
            if "elo_diff_pre" in df.columns
            else np.full(n, 48.0, dtype=np.float64)
        )
        npxgd_diff = (
            np.asarray(
                df["npxgd_per_90_w10_diff"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "npxgd_per_90_w10_diff" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        gk_diff = (
            np.asarray(
                df["gk_gsaa_per_90_diff"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "gk_gsaa_per_90_diff" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        fwd_diff = (
            np.asarray(
                df["fwd_npxg_per_90_diff"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "fwd_npxg_per_90_diff" in df.columns
            else np.zeros(n, dtype=np.float64)
        )

        comp_diff = elo_diff + 85.0 * npxgd_diff + 35.0 * gk_diff + 55.0 * fwd_diff
        out = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            p_h, p_d, p_a = compute_elo_1x2_probabilities(
                float(comp_diff[i]), draw_param=self.draw_param
            )
            out[i, 0] = p_h
            out[i, 1] = p_d
            out[i, 2] = p_a
        return normalize_multiclass_probabilities(out)

    def _predict_expected_goals(
        self, df: pl.DataFrame
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = df.height
        x_mat = self._extract_matrix(df)

        # Analytical expected goals baseline from Elo + npxG
        elo_diff = (
            np.asarray(
                df["elo_diff_pre"].cast(pl.Float64).fill_null(48.0).to_numpy(),
                dtype=np.float64,
            )
            if "elo_diff_pre" in df.columns
            else np.full(n, 48.0, dtype=np.float64)
        )
        npxgd_diff = (
            np.asarray(
                df["npxgd_per_90_w10_diff"].cast(pl.Float64).fill_null(0.0).to_numpy(),
                dtype=np.float64,
            )
            if "npxgd_per_90_w10_diff" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        prior_h = np.clip(
            self._default_home_goals + 0.0022 * (elo_diff - 48.0) + 0.35 * npxgd_diff,
            0.30,
            4.80,
        )
        prior_a = np.clip(
            self._default_away_goals - 0.0020 * (elo_diff - 48.0) - 0.35 * npxgd_diff,
            0.25,
            4.50,
        )

        if self._home_goals_regressor is not None and self._away_goals_regressor is not None:
            gbdt_h = np.clip(
                np.asarray(self._home_goals_regressor.predict(x_mat), dtype=np.float64),
                0.25,
                5.50,
            )
            gbdt_a = np.clip(
                np.asarray(self._away_goals_regressor.predict(x_mat), dtype=np.float64),
                0.25,
                5.50,
            )
            lam_h = np.clip(0.55 * gbdt_h + 0.45 * prior_h, 0.25, 5.50)
            lam_a = np.clip(0.55 * gbdt_a + 0.45 * prior_a, 0.25, 5.50)
            return lam_h, lam_a
        return prior_h, prior_a

    def _raw_blended_probs(self, df: pl.DataFrame) -> NDArray[np.float64]:
        if self._classifier is None:
            raise RuntimeError("Stage1SoccerMatchModel must be fitted before prediction.")
        x_mat = self._extract_matrix(df)
        gbdt_probs = normalize_multiclass_probabilities(
            np.asarray(self._classifier.predict_proba(x_mat), dtype=np.float64)
        )
        elo_probs = self._extract_composite_elo_1x2(df)

        lam_h, lam_a = self._predict_expected_goals(df)
        dc_probs = np.zeros_like(gbdt_probs)
        for i in range(df.height):
            ph, pd, pa = compute_dixon_coles_1x2_probabilities(
                float(lam_h[i]), float(lam_a[i]), rho=self.dixon_coles_rho
            )
            dc_probs[i, 0] = ph
            dc_probs[i, 1] = pd
            dc_probs[i, 2] = pa

        w_elo = self.elo_blend_weight
        w_dc = self.dixon_coles_blend_weight
        w_gbdt = max(0.0, 1.0 - w_elo - w_dc)
        blended = w_gbdt * gbdt_probs + w_elo * elo_probs + w_dc * dc_probs
        return normalize_multiclass_probabilities(blended)

    @staticmethod
    def _extract_target_1x2(df: pl.DataFrame) -> NDArray[np.int64]:
        if "outcome_1x2" in df.columns and df["outcome_1x2"].null_count() == 0:
            return np.asarray(df["outcome_1x2"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        if {"home_goals", "away_goals"}.issubset(df.columns):
            hg = np.asarray(df["home_goals"].cast(pl.Int64).to_numpy(), dtype=np.int64)
            ag = np.asarray(df["away_goals"].cast(pl.Int64).to_numpy(), dtype=np.int64)
            return np.where(hg > ag, 0, np.where(hg == ag, 1, 2)).astype(np.int64)
        raise ValueError("DataFrame must contain 'outcome_1x2' or ('home_goals', 'away_goals').")

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame | None = None,
    ) -> Stage1SoccerMatchModel:
        """Fit Multiclass LightGBM, Poisson goal regressors, and out-of-fold 3-way calibrator."""
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
        y_train = self._extract_target_1x2(gbdt_train_df)

        # Ensure all 3 classes (0, 1, 2) are represented for LightGBM multiclass fitting
        present_classes = set(np.unique(y_train).tolist())
        if len(present_classes) < 3:
            pad_x = [x_train]
            pad_y = [y_train]
            for missing_c in (0, 1, 2):
                if missing_c not in present_classes:
                    pad_x.append(x_train[:1])
                    pad_y.append(np.array([missing_c], dtype=np.int64))
            x_fit = np.vstack(pad_x)
            y_fit = np.concatenate(pad_y)
        else:
            x_fit = x_train
            y_fit = y_train

        min_child = max(5, min(35, x_fit.shape[0] // 8))
        leaves = max(3, min(self.num_leaves, max(3, x_fit.shape[0] // 6)))

        clf = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=leaves,
            min_child_samples=min_child,
            colsample_bytree=0.70,
            subsample=0.80,
            subsample_freq=1,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=self.random_seed,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(x_fit, y_fit)
        self._classifier = clf

        if {"home_goals", "away_goals"}.issubset(gbdt_train_df.columns):
            y_hg = np.clip(
                np.asarray(
                    gbdt_train_df["home_goals"].cast(pl.Float64).to_numpy(),
                    dtype=np.float64,
                ),
                0.0,
                None,
            )
            y_ag = np.clip(
                np.asarray(
                    gbdt_train_df["away_goals"].cast(pl.Float64).to_numpy(),
                    dtype=np.float64,
                ),
                0.0,
                None,
            )
            self._default_home_goals = max(0.4, float(np.mean(y_hg)))
            self._default_away_goals = max(0.4, float(np.mean(y_ag)))

            reg_h = LGBMRegressor(
                objective="poisson",
                n_estimators=max(25, self.n_estimators // 2),
                learning_rate=self.learning_rate,
                num_leaves=leaves,
                min_child_samples=min_child,
                random_state=self.random_seed,
                n_jobs=1,
                verbose=-1,
            )
            reg_a = LGBMRegressor(
                objective="poisson",
                n_estimators=max(25, self.n_estimators // 2),
                learning_rate=self.learning_rate,
                num_leaves=leaves,
                min_child_samples=min_child,
                random_state=self.random_seed,
                n_jobs=1,
                verbose=-1,
            )
            reg_h.fit(x_train, y_hg)
            reg_a.fit(x_train, y_ag)
            self._home_goals_regressor = reg_h
            self._away_goals_regressor = reg_a

        raw_val_probs = self._raw_blended_probs(cal_val_df)
        y_val = self._extract_target_1x2(cal_val_df)
        self.calibrator.fit(raw_val_probs, y_val)
        self._is_fitted = True
        return self

    def predict(self, features_df: pl.DataFrame) -> pl.DataFrame:
        """Predict calibrated `home_win_prob`, `draw_prob`, `away_win_prob`, and expected goals."""
        if not self._is_fitted:
            raise RuntimeError("Stage1SoccerMatchModel must be fitted before calling predict.")

        if features_df.is_empty():
            return features_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("home_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("draw_prob"),
                pl.lit(None, dtype=pl.Float64).alias("away_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("expected_home_goals"),
                pl.lit(None, dtype=pl.Float64).alias("expected_away_goals"),
            )

        raw_probs = self._raw_blended_probs(features_df)
        cal_probs = normalize_multiclass_probabilities(self.calibrator.predict_proba(raw_probs))

        p_home = cal_probs[:, 0]
        p_draw = cal_probs[:, 1]
        # Set p_away = 1.0 - p_home - p_draw to guarantee exact sum == 1.0 to machine precision
        p_away = np.clip(1.0 - p_home - p_draw, 1e-6, 1.0 - 1e-6)
        total = p_home + p_draw + p_away
        p_home = p_home / total
        p_draw = p_draw / total
        p_away = 1.0 - p_home - p_draw

        lam_h, lam_a = self._predict_expected_goals(features_df)
        total_goals = np.clip(lam_h + lam_a, 1.60, 4.80)
        strength_share_h = np.clip(0.50 + 0.42 * (p_home - p_away), 0.18, 0.82)
        reconciled_h = np.clip(0.55 * lam_h + 0.45 * (total_goals * strength_share_h), 0.20, 5.0)
        reconciled_a = np.clip(
            0.55 * lam_a + 0.45 * (total_goals * (1.0 - strength_share_h)), 0.20, 5.0
        )

        return features_df.with_columns(
            pl.Series("home_win_prob", p_home, dtype=pl.Float64),
            pl.Series("draw_prob", p_draw, dtype=pl.Float64),
            pl.Series("away_win_prob", p_away, dtype=pl.Float64),
            pl.Series("expected_home_goals", reconciled_h, dtype=pl.Float64),
            pl.Series("expected_away_goals", reconciled_a, dtype=pl.Float64),
        )


def run_soccer_stage1_walk_forward_backtest(
    features_df: pl.DataFrame,
    *,
    n_splits: int = 3,
    min_train_groups: int = 10,
    calibration_method: CalibrationMethod = "sigmoid",
    random_seed: int = RANDOM_SEED,
) -> SoccerStage1BacktestResult:
    """Execute leak-free walk-forward backtest for EPL Stage 1 `1X2` probabilities."""
    completed = features_df.filter(
        pl.col("outcome_1x2").is_not_null()
        if "outcome_1x2" in features_df.columns
        else pl.col("home_goals").is_not_null()
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

        model = Stage1SoccerMatchModel(
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
    model_probs = np.column_stack(
        [
            oof_df["home_win_prob"].to_numpy(),
            oof_df["draw_prob"].to_numpy(),
            oof_df["away_win_prob"].to_numpy(),
        ]
    )
    y_true = Stage1SoccerMatchModel._extract_target_1x2(oof_df)

    elo_probs: NDArray[np.float64] | None = None
    if {"elo_home_win_prob", "elo_draw_prob", "elo_away_win_prob"}.issubset(oof_df.columns):
        elo_probs = np.column_stack(
            [
                oof_df["elo_home_win_prob"].to_numpy(),
                oof_df["elo_draw_prob"].to_numpy(),
                oof_df["elo_away_win_prob"].to_numpy(),
            ]
        )

    report = compare_multiclass_against_baselines(
        model_probs,
        y_true,
        empirical_prior=(0.45, 0.24, 0.31),
        elo_probs=elo_probs,
    )

    return SoccerStage1BacktestResult(
        oof_predictions=oof_df,
        benchmark_report=report,
        fold_audits=fold_audits,
    )
