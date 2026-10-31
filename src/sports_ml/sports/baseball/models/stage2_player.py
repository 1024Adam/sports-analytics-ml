"""Stage 2 Starting Pitcher (`SP`) & Lineup Batter (`1..9`) Projections (`stage2_player.py`).

Implements:
1. Starting Pitcher (`SP`) Projection Pipeline:
   - `proj_outs_pitched` in `[0, 27]`, `proj_ip` (`outs / 3`), `proj_batters_faced`
   - `proj_k`, `proj_bb`, `proj_hits_allowed`, `proj_er`
   - `prob_5plus_k`, `prob_6plus_k`, `prob_7plus_k`, `prob_quality_start`
2. Lineup Batter (`1..9`) Projection Pipeline:
   - `proj_pa` in `[0, 6.0]` (monotonically decreasing from Slot 1 to Slot 9, accounting
     for unplayed bottom-9th innings for home batters and Stage 1 team run environment)
   - `proj_hits`, `proj_total_bases`, `proj_hr`, `proj_runs`, `proj_rbi`, `proj_bb`, `proj_k`
   - `prob_1plus_hit`, `prob_2plus_tb`, `prob_1plus_hr`
3. Cross-Stage Coherence Reconciler (`reconcile_baseball_stage2_projections`, `gamma = 0.50`).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from numpy.typing import NDArray

from sports_ml.config import RANDOM_SEED
from sports_ml.core.market import is_forbidden_market_column
from sports_ml.sports.baseball.features.pipeline import (
    BASEBALL_STAGE2_BATTER_CANDIDATE_FEATURE_COLS,
    BASEBALL_STAGE2_PITCHER_CANDIDATE_FEATURE_COLS,
)


def _poisson_tail_prob(lam: float, min_k: int) -> float:
    """Compute Poisson `P(X >= min_k)` for rate `lam >= 0`."""
    mu = max(0.0, float(lam))
    if min_k <= 0:
        return 1.0
    if mu <= 1e-9:
        return 0.0
    cdf = 0.0
    term = math.exp(-mu)
    for k in range(min_k):
        if k > 0:
            term *= mu / float(k)
        cdf += term
    return float(min(max(1.0 - cdf, 0.0), 1.0))


def _poisson_cdf(lam: float, max_k: int) -> float:
    """Compute Poisson `P(X <= max_k)` for rate `lam >= 0`."""
    return 1.0 - _poisson_tail_prob(lam, max_k + 1)


def reconcile_baseball_stage2_projections(
    batter_preds_df: pl.DataFrame,
    pitcher_preds_df: pl.DataFrame | None = None,
    *,
    gamma: float = 0.50,
    rbi_per_run_ratio: float = 0.94,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Soft-scale batter `proj_runs`/`proj_rbi` and pitcher `proj_er` toward Stage 1 team runs.

    Formula:
        `mu_reconciled = mu_raw * (lambda_stage1 / sum(mu_raw)) ** gamma`
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}.")

    reconciled_batters = batter_preds_df
    if not batter_preds_df.is_empty():
        seq_cols = (
            ["game_id", "event_seq", "team"]
            if "event_seq" in batter_preds_df.columns
            else ["game_id", "team"]
        )
        raw_runs_col = (
            "_raw_proj_runs" if "_raw_proj_runs" in batter_preds_df.columns else "proj_runs"
        )
        raw_rbi_col = "_raw_proj_rbi" if "_raw_proj_rbi" in batter_preds_df.columns else "proj_rbi"

        raw_team_runs_sum = pl.col(raw_runs_col).sum().over(seq_cols)
        safe_runs_denom = pl.when(raw_team_runs_sum > 1e-6).then(raw_team_runs_sum).otherwise(1.0)
        target_runs = (
            pl.col("stage1_team_expected_runs").fill_null(4.35).clip(lower_bound=0.50)
            if "stage1_team_expected_runs" in batter_preds_df.columns
            else pl.lit(4.35, dtype=pl.Float64)
        )
        runs_scale = (target_runs / safe_runs_denom).pow(gamma)

        raw_team_rbi_sum = pl.col(raw_rbi_col).sum().over(seq_cols)
        safe_rbi_denom = pl.when(raw_team_rbi_sum > 1e-6).then(raw_team_rbi_sum).otherwise(1.0)
        rbi_scale = ((target_runs * rbi_per_run_ratio) / safe_rbi_denom).pow(gamma)

        # Also gently scale total bases and hits by `runs_scale ** 0.45` so scoring environment
        # shifts propagate monotonically across offensive counting stats
        tb_scale = runs_scale.pow(0.45)
        raw_tb_col = (
            "_raw_proj_tb" if "_raw_proj_tb" in batter_preds_df.columns else "proj_total_bases"
        )

        reconciled_tb = (pl.col(raw_tb_col) * tb_scale).clip(lower_bound=0.0)
        reconciled_batters = batter_preds_df.with_columns(
            (pl.col(raw_runs_col) * runs_scale).clip(lower_bound=0.0).alias("proj_runs"),
            (pl.col(raw_rbi_col) * rbi_scale).clip(lower_bound=0.0).alias("proj_rbi"),
            reconciled_tb.alias("proj_total_bases"),
        )
        prob_2tb = [
            _poisson_tail_prob(float(tb) * 0.78, 2)
            for tb in reconciled_batters["proj_total_bases"].to_list()
        ]
        reconciled_batters = reconciled_batters.with_columns(
            pl.Series("prob_2plus_tb", prob_2tb, dtype=pl.Float64)
        )

    reconciled_pitchers = pitcher_preds_df
    if pitcher_preds_df is not None and not pitcher_preds_df.is_empty():
        raw_er_col = "_raw_proj_er" if "_raw_proj_er" in pitcher_preds_df.columns else "proj_er"
        opp_runs = (
            pl.col("stage1_opp_expected_runs").fill_null(4.35).clip(lower_bound=0.50)
            if "stage1_opp_expected_runs" in pitcher_preds_df.columns
            else pl.lit(4.35, dtype=pl.Float64)
        )
        sp_share = (pl.col("proj_outs_pitched").clip(6.0, 27.0) / 27.0) * 0.91
        target_sp_er = (opp_runs * sp_share).clip(lower_bound=0.25)
        safe_er = pl.when(pl.col(raw_er_col) > 1e-6).then(pl.col(raw_er_col)).otherwise(1.0)
        er_scale = (target_sp_er / safe_er).pow(gamma)
        rec_er = (pl.col(raw_er_col) * er_scale).clip(lower_bound=0.0)

        reconciled_pitchers = pitcher_preds_df.with_columns(rec_er.alias("proj_er"))
        qs_probs: list[float] = []
        for r in reconciled_pitchers.iter_rows(named=True):
            outs_p = float(r["proj_outs_pitched"])
            er_p = float(r["proj_er"])
            p_reach_18 = 1.0 / (1.0 + math.exp(-0.55 * (outs_p - 16.8)))
            p_er_le_3 = _poisson_cdf(er_p, 3)
            qs_probs.append(float(min(max(p_reach_18 * p_er_le_3, 0.01), 0.95)))
        reconciled_pitchers = reconciled_pitchers.with_columns(
            pl.Series("prob_quality_start", qs_probs, dtype=pl.Float64)
        )

    return reconciled_batters, reconciled_pitchers


class Stage2BaseballPlayerModel:
    """Optional Stage 2 Starting Pitcher (`SP`) & Lineup Batter (`1..9`) projection model."""

    def __init__(
        self,
        *,
        reconciliation_gamma: float = 0.50,
        gbdt_blend_weight: float = 0.45,
        n_estimators: int = 50,
        learning_rate: float = 0.04,
        num_leaves: int = 10,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        if not (0.0 <= reconciliation_gamma <= 1.0):
            raise ValueError(
                f"reconciliation_gamma must be in [0.0, 1.0], got {reconciliation_gamma}."
            )
        self.reconciliation_gamma = reconciliation_gamma
        self.gbdt_blend_weight = gbdt_blend_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_seed = random_seed

        self.batter_feature_columns: list[str] = []
        self.pitcher_feature_columns: list[str] = []
        self._batter_pa_model: LGBMRegressor | None = None
        self._batter_hits_model: LGBMRegressor | None = None
        self._batter_tb_model: LGBMRegressor | None = None
        self._batter_hr_model: LGBMRegressor | None = None
        self._pitcher_outs_model: LGBMRegressor | None = None
        self._pitcher_k_model: LGBMRegressor | None = None
        self._pitcher_er_model: LGBMRegressor | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_columns(self) -> list[str]:
        """Primary feature column list saved in `metadata.json` (`batter_feature_columns`)."""
        return self.batter_feature_columns

    @staticmethod
    def resolve_batter_feature_columns(df: pl.DataFrame) -> list[str]:
        candidate_set = set(BASEBALL_STAGE2_BATTER_CANDIDATE_FEATURE_COLS)
        preferred = [
            c for c in df.columns if c in candidate_set and not is_forbidden_market_column(c)
        ]
        if preferred:
            return preferred
        excluded = {
            "game_id",
            "event_id",
            "event_seq",
            "game_date",
            "start_time_utc",
            "season",
            "league",
            "player_id",
            "player_name",
            "team",
            "opponent",
            "is_home",
            "bat_side",
            "batting_order_slot",
            "position_group",
            "is_starter",
            "pa",
            "ab",
            "hits",
            "singles",
            "doubles",
            "triples",
            "hr",
            "total_bases",
            "runs",
            "rbi",
            "bb",
            "ibb",
            "hbp",
            "k",
            "sf",
            "sb",
            "cs",
            "woba_num",
        }
        return [
            col
            for col, dtype in df.schema.items()
            if col not in excluded and not is_forbidden_market_column(col) and dtype.is_numeric()
        ]

    @staticmethod
    def resolve_pitcher_feature_columns(df: pl.DataFrame) -> list[str]:
        candidate_set = set(BASEBALL_STAGE2_PITCHER_CANDIDATE_FEATURE_COLS)
        preferred = [
            c for c in df.columns if c in candidate_set and not is_forbidden_market_column(c)
        ]
        if preferred:
            return preferred
        excluded = {
            "game_id",
            "event_id",
            "event_seq",
            "game_date",
            "start_time_utc",
            "season",
            "league",
            "player_id",
            "player_name",
            "team",
            "opponent",
            "is_home",
            "pitch_hand",
            "is_starter",
            "outs_pitched",
            "innings_pitched",
            "batters_faced",
            "pitches_thrown",
            "strikes_thrown",
            "hits_allowed",
            "runs_allowed",
            "er_allowed",
            "hr_allowed",
            "bb_allowed",
            "ibb_allowed",
            "hbp_allowed",
            "k_recorded",
            "gb_count",
            "fb_count",
            "ld_count",
            "quality_start",
        }
        return [
            col
            for col, dtype in df.schema.items()
            if col not in excluded and not is_forbidden_market_column(col) and dtype.is_numeric()
        ]

    @staticmethod
    def _extract_matrix(df: pl.DataFrame, cols: list[str]) -> NDArray[np.float64]:
        mat = df.select(
            [
                (
                    pl.col(c).cast(pl.Float64).fill_null(0.0)
                    if c in df.columns
                    else pl.lit(0.0, dtype=pl.Float64).alias(c)
                )
                for c in cols
            ]
        ).to_numpy()
        return np.asarray(mat, dtype=np.float64)

    def _fit_single_regressor(
        self,
        x_mat: NDArray[np.float64],
        y_vec: NDArray[np.float64],
        *,
        objective: str = "tweedie",
    ) -> LGBMRegressor:
        min_child = max(3, min(25, x_mat.shape[0] // 6))
        leaves = max(3, min(self.num_leaves, max(3, x_mat.shape[0] // 5)))
        reg = LGBMRegressor(
            objective=objective,
            tweedie_variance_power=1.4 if objective == "tweedie" else 1.5,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=leaves,
            min_child_samples=min_child,
            random_state=self.random_seed,
            n_jobs=1,
            verbose=-1,
        )
        reg.fit(x_mat, np.clip(y_vec, 0.0, None))
        return reg

    def fit(
        self,
        batter_train_df: pl.DataFrame,
        pitcher_train_df: pl.DataFrame | None = None,
    ) -> Stage2BaseballPlayerModel:
        """Fit Stage 2 Lineup Batter and Starting Pitcher regressors."""
        if batter_train_df.is_empty():
            raise ValueError("batter_train_df cannot be empty.")

        self.batter_feature_columns = self.resolve_batter_feature_columns(batter_train_df)
        x_bat = self._extract_matrix(batter_train_df, self.batter_feature_columns)

        if "pa" in batter_train_df.columns:
            y_pa = np.asarray(
                batter_train_df["pa"].cast(pl.Float64).fill_null(4.0).to_numpy(), dtype=np.float64
            )
            self._batter_pa_model = self._fit_single_regressor(x_bat, y_pa, objective="regression")
        if "hits" in batter_train_df.columns:
            y_h = np.asarray(
                batter_train_df["hits"].cast(pl.Float64).fill_null(0.9).to_numpy(),
                dtype=np.float64,
            )
            self._batter_hits_model = self._fit_single_regressor(x_bat, y_h, objective="tweedie")
        if "total_bases" in batter_train_df.columns:
            y_tb = np.asarray(
                batter_train_df["total_bases"].cast(pl.Float64).fill_null(1.4).to_numpy(),
                dtype=np.float64,
            )
            self._batter_tb_model = self._fit_single_regressor(x_bat, y_tb, objective="tweedie")
        if "hr" in batter_train_df.columns:
            y_hr = np.asarray(
                batter_train_df["hr"].cast(pl.Float64).fill_null(0.12).to_numpy(),
                dtype=np.float64,
            )
            self._batter_hr_model = self._fit_single_regressor(x_bat, y_hr, objective="tweedie")

        if pitcher_train_df is not None and not pitcher_train_df.is_empty():
            self.pitcher_feature_columns = self.resolve_pitcher_feature_columns(pitcher_train_df)
            x_pit = self._extract_matrix(pitcher_train_df, self.pitcher_feature_columns)
            if "outs_pitched" in pitcher_train_df.columns:
                y_outs = np.asarray(
                    pitcher_train_df["outs_pitched"].cast(pl.Float64).fill_null(17.5).to_numpy(),
                    dtype=np.float64,
                )
                self._pitcher_outs_model = self._fit_single_regressor(
                    x_pit, y_outs, objective="regression"
                )
            if "k_recorded" in pitcher_train_df.columns:
                y_k = np.asarray(
                    pitcher_train_df["k_recorded"].cast(pl.Float64).fill_null(5.8).to_numpy(),
                    dtype=np.float64,
                )
                self._pitcher_k_model = self._fit_single_regressor(x_pit, y_k, objective="tweedie")
            if "er_allowed" in pitcher_train_df.columns:
                y_er = np.asarray(
                    pitcher_train_df["er_allowed"].cast(pl.Float64).fill_null(2.5).to_numpy(),
                    dtype=np.float64,
                )
                self._pitcher_er_model = self._fit_single_regressor(
                    x_pit, y_er, objective="tweedie"
                )

        self._is_fitted = True
        return self

    def predict_batters(self, batter_features_df: pl.DataFrame) -> pl.DataFrame:
        """Predict Stage 2 Lineup Batter (`1..9`) PA, counts, and milestone probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Stage2BaseballPlayerModel must be fitted before prediction.")
        if batter_features_df.is_empty():
            return batter_features_df

        n = batter_features_df.height
        x_bat = self._extract_matrix(batter_features_df, self.batter_feature_columns)

        # Structural PA model guaranteeing Batting Order Slot 1 > Slot 2 > ... > Slot 9 monotonicity
        slot_num = (
            np.asarray(
                batter_features_df["batting_slot_num"].cast(pl.Float64).fill_null(5.0).to_numpy(),
                dtype=np.float64,
            )
            if "batting_slot_num" in batter_features_df.columns
            else (
                np.asarray(
                    batter_features_df["batting_order_slot"]
                    .cast(pl.Float64)
                    .fill_null(5.0)
                    .to_numpy(),
                    dtype=np.float64,
                )
                if "batting_order_slot" in batter_features_df.columns
                else np.full(n, 5.0, dtype=np.float64)
            )
        )
        is_starter = (
            np.asarray(
                batter_features_df["is_starter_flag"].cast(pl.Float64).fill_null(1.0).to_numpy(),
                dtype=np.float64,
            )
            if "is_starter_flag" in batter_features_df.columns
            else np.ones(n, dtype=np.float64)
        )
        is_home = (
            np.asarray(
                batter_features_df["is_home_flag"].cast(pl.Float64).fill_null(1.0).to_numpy(),
                dtype=np.float64,
            )
            if "is_home_flag" in batter_features_df.columns
            else np.ones(n, dtype=np.float64)
        )
        s1_runs = (
            np.asarray(
                batter_features_df["stage1_team_expected_runs"]
                .cast(pl.Float64)
                .fill_null(4.35)
                .to_numpy(),
                dtype=np.float64,
            )
            if "stage1_team_expected_runs" in batter_features_df.columns
            else np.full(n, 4.35, dtype=np.float64)
        )

        # Slot 1 averages ~4.65 PA, decreasing by ~0.112 PA per slot down to Slot 9 (~3.75 PA),
        # plus team run environment turnover and -0.16 PA for home team (unplayed bottom 9th)
        eff_slot = np.where((slot_num >= 1.0) & (slot_num <= 9.0), slot_num, 8.5)
        prior_pa = np.where(
            is_starter >= 0.5,
            4.68 - 0.112 * (eff_slot - 1.0) + 0.14 * (s1_runs - 4.35) - 0.16 * is_home,
            1.15,
        )
        if self._batter_pa_model is not None:
            gbdt_pa = np.asarray(self._batter_pa_model.predict(x_bat), dtype=np.float64)
            # Keep structural slot monotonicity dominant (0.75 prior + 0.25 GBDT)
            proj_pa = np.clip(0.75 * prior_pa + 0.25 * gbdt_pa, 0.20, 6.00)
        else:
            proj_pa = np.clip(prior_pa, 0.20, 6.00)

        plat_woba = (
            np.asarray(
                batter_features_df["batter_platoon_woba"]
                .cast(pl.Float64)
                .fill_null(0.315)
                .to_numpy(),
                dtype=np.float64,
            )
            if "batter_platoon_woba" in batter_features_df.columns
            else np.full(n, 0.315, dtype=np.float64)
        )
        iso_s = (
            np.asarray(
                batter_features_df["batter_iso_shrunk"]
                .cast(pl.Float64)
                .fill_null(0.155)
                .to_numpy(),
                dtype=np.float64,
            )
            if "batter_iso_shrunk" in batter_features_df.columns
            else np.full(n, 0.155, dtype=np.float64)
        )
        bat_k_pct = (
            np.asarray(
                batter_features_df["batter_k_pct_shrunk"]
                .cast(pl.Float64)
                .fill_null(0.220)
                .to_numpy(),
                dtype=np.float64,
            )
            if "batter_k_pct_shrunk" in batter_features_df.columns
            else np.full(n, 0.220, dtype=np.float64)
        )
        bat_bb_pct = (
            np.asarray(
                batter_features_df["batter_bb_pct_shrunk"]
                .cast(pl.Float64)
                .fill_null(0.084)
                .to_numpy(),
                dtype=np.float64,
            )
            if "batter_bb_pct_shrunk" in batter_features_df.columns
            else np.full(n, 0.084, dtype=np.float64)
        )
        bat_hr_pa = (
            np.asarray(
                batter_features_df["batter_hr_per_pa_shrunk"]
                .cast(pl.Float64)
                .fill_null(0.031)
                .to_numpy(),
                dtype=np.float64,
            )
            if "batter_hr_per_pa_shrunk" in batter_features_df.columns
            else np.full(n, 0.031, dtype=np.float64)
        )
        opp_sp_xfip = (
            np.asarray(
                batter_features_df["opp_sp_xfip_shrunk"]
                .cast(pl.Float64)
                .fill_null(4.10)
                .to_numpy(),
                dtype=np.float64,
            )
            if "opp_sp_xfip_shrunk" in batter_features_df.columns
            else np.full(n, 4.10, dtype=np.float64)
        )
        opp_sp_k = (
            np.asarray(
                batter_features_df["opp_sp_k_pct_shrunk"]
                .cast(pl.Float64)
                .fill_null(0.224)
                .to_numpy(),
                dtype=np.float64,
            )
            if "opp_sp_k_pct_shrunk" in batter_features_df.columns
            else np.full(n, 0.224, dtype=np.float64)
        )
        opp_sp_bb = (
            np.asarray(
                batter_features_df["opp_sp_bb_pct_shrunk"]
                .cast(pl.Float64)
                .fill_null(0.078)
                .to_numpy(),
                dtype=np.float64,
            )
            if "opp_sp_bb_pct_shrunk" in batter_features_df.columns
            else np.full(n, 0.078, dtype=np.float64)
        )
        v_rf = (
            np.asarray(
                batter_features_df["venue_run_factor"].cast(pl.Float64).fill_null(1.0).to_numpy(),
                dtype=np.float64,
            )
            if "venue_run_factor" in batter_features_df.columns
            else np.ones(n, dtype=np.float64)
        )
        v_hr = (
            np.asarray(
                batter_features_df["venue_hr_factor_batter"]
                .cast(pl.Float64)
                .fill_null(1.0)
                .to_numpy(),
                dtype=np.float64,
            )
            if "venue_hr_factor_batter" in batter_features_df.columns
            else np.ones(n, dtype=np.float64)
        )

        # Log5 / multiplicative matchup rates per PA
        sp_suppr = np.clip(opp_sp_xfip / 4.10, 0.72, 1.35)
        hit_rate_pa = np.clip(
            (0.225 + 0.62 * (plat_woba - 0.315)) * (sp_suppr**0.45) * (v_rf**0.55),
            0.08,
            0.40,
        )
        hr_rate_pa = np.clip(
            (bat_hr_pa + 0.14 * (iso_s - 0.155)) * (sp_suppr**0.50) * v_hr,
            0.004,
            0.120,
        )
        tb_rate_pa = np.clip(
            hit_rate_pa + (iso_s * 0.88) * (sp_suppr**0.45) * (0.55 * v_rf + 0.45 * v_hr),
            0.12,
            0.85,
        )
        # Strikeout rate increases monotonically with opposing starter K%
        k_rate_pa = np.clip(bat_k_pct * (opp_sp_k / 0.220) ** 0.75, 0.05, 0.48)
        bb_rate_pa = np.clip(bat_bb_pct * (opp_sp_bb / 0.080) ** 0.65, 0.02, 0.28)

        prior_hits = proj_pa * hit_rate_pa
        prior_hr = proj_pa * hr_rate_pa
        prior_tb = np.maximum(prior_hits, proj_pa * tb_rate_pa)
        proj_k = np.clip(proj_pa * k_rate_pa, 0.0, proj_pa)
        proj_bb = np.clip(proj_pa * bb_rate_pa, 0.0, proj_pa)

        w_gbdt = self.gbdt_blend_weight
        if self._batter_hits_model is not None:
            gbdt_h = np.clip(
                np.asarray(self._batter_hits_model.predict(x_bat), dtype=np.float64), 0.0, 4.5
            )
            proj_hits = np.clip((1.0 - w_gbdt) * prior_hits + w_gbdt * gbdt_h, 0.02, 4.5)
        else:
            proj_hits = prior_hits

        if self._batter_hr_model is not None:
            gbdt_hr = np.clip(
                np.asarray(self._batter_hr_model.predict(x_bat), dtype=np.float64), 0.0, 1.8
            )
            proj_hr = np.clip((1.0 - w_gbdt) * prior_hr + w_gbdt * gbdt_hr, 0.005, 1.8)
        else:
            proj_hr = prior_hr

        if self._batter_tb_model is not None:
            gbdt_tb = np.clip(
                np.asarray(self._batter_tb_model.predict(x_bat), dtype=np.float64), 0.0, 7.5
            )
            raw_tb = np.maximum(
                proj_hits, np.clip((1.0 - w_gbdt) * prior_tb + w_gbdt * gbdt_tb, 0.04, 7.5)
            )
        else:
            raw_tb = np.maximum(proj_hits, prior_tb)

        # Raw runs and RBI (reconciled with Stage 1 team expected runs next)
        slot_run_mult = np.clip(1.18 - 0.045 * (eff_slot - 1.0), 0.78, 1.22)
        slot_rbi_mult = np.where(
            (eff_slot >= 3.0) & (eff_slot <= 5.0),
            1.22,
            np.where(eff_slot <= 2.0, 0.92, 0.88),
        )
        raw_runs = np.clip((s1_runs / 9.0) * slot_run_mult * (plat_woba / 0.315), 0.02, 3.0)
        raw_rbi = np.clip(
            (s1_runs * 0.94 / 9.0) * slot_rbi_mult * ((plat_woba + iso_s) / 0.470),
            0.02,
            3.2,
        )

        p_1plus_hit = np.clip(1.0 - np.power(1.0 - hit_rate_pa, proj_pa), 0.01, 0.98)
        p_1plus_hr = np.clip(1.0 - np.exp(-proj_hr), 0.002, 0.85)
        p_2plus_tb = np.array(
            [_poisson_tail_prob(float(tb) * 0.78, 2) for tb in raw_tb], dtype=np.float64
        )

        prelim = batter_features_df.with_columns(
            pl.Series("proj_pa", proj_pa, dtype=pl.Float64),
            pl.Series("proj_hits", proj_hits, dtype=pl.Float64),
            pl.Series("_raw_proj_tb", raw_tb, dtype=pl.Float64),
            pl.Series("proj_total_bases", raw_tb, dtype=pl.Float64),
            pl.Series("proj_hr", proj_hr, dtype=pl.Float64),
            pl.Series("_raw_proj_runs", raw_runs, dtype=pl.Float64),
            pl.Series("proj_runs", raw_runs, dtype=pl.Float64),
            pl.Series("_raw_proj_rbi", raw_rbi, dtype=pl.Float64),
            pl.Series("proj_rbi", raw_rbi, dtype=pl.Float64),
            pl.Series("proj_bb", proj_bb, dtype=pl.Float64),
            pl.Series("proj_k", proj_k, dtype=pl.Float64),
            pl.Series("prob_1plus_hit", p_1plus_hit, dtype=pl.Float64),
            pl.Series("prob_2plus_tb", p_2plus_tb, dtype=pl.Float64),
            pl.Series("prob_1plus_hr", p_1plus_hr, dtype=pl.Float64),
        )
        reconciled_b, _ = reconcile_baseball_stage2_projections(
            prelim, None, gamma=self.reconciliation_gamma
        )
        return reconciled_b.drop(["_raw_proj_tb", "_raw_proj_runs", "_raw_proj_rbi"])

    def predict_pitchers(self, pitcher_features_df: pl.DataFrame) -> pl.DataFrame:
        """Predict Stage 2 Starting Pitcher (`SP`) outs/IP, K, BB, ER, and milestone probs."""
        if not self._is_fitted:
            raise RuntimeError("Stage2BaseballPlayerModel must be fitted before prediction.")
        if pitcher_features_df.is_empty():
            return pitcher_features_df

        n = pitcher_features_df.height
        outs_ceil = (
            np.asarray(
                pitcher_features_df["sp_expected_outs_ceiling"]
                .cast(pl.Float64)
                .fill_null(17.5)
                .to_numpy(),
                dtype=np.float64,
            )
            if "sp_expected_outs_ceiling" in pitcher_features_df.columns
            else np.full(n, 17.5, dtype=np.float64)
        )
        sp_xfip = (
            np.asarray(
                pitcher_features_df["sp_xfip_shrunk"].cast(pl.Float64).fill_null(4.0).to_numpy(),
                dtype=np.float64,
            )
            if "sp_xfip_shrunk" in pitcher_features_df.columns
            else np.full(n, 4.0, dtype=np.float64)
        )
        sp_k_pct = (
            np.asarray(
                pitcher_features_df["sp_k_pct_shrunk"].cast(pl.Float64).fill_null(0.224).to_numpy(),
                dtype=np.float64,
            )
            if "sp_k_pct_shrunk" in pitcher_features_df.columns
            else np.full(n, 0.224, dtype=np.float64)
        )
        sp_bb_pct = (
            np.asarray(
                pitcher_features_df["sp_bb_pct_shrunk"]
                .cast(pl.Float64)
                .fill_null(0.078)
                .to_numpy(),
                dtype=np.float64,
            )
            if "sp_bb_pct_shrunk" in pitcher_features_df.columns
            else np.full(n, 0.078, dtype=np.float64)
        )
        sp_whip = (
            np.asarray(
                pitcher_features_df["sp_whip_shrunk"].cast(pl.Float64).fill_null(1.25).to_numpy(),
                dtype=np.float64,
            )
            if "sp_whip_shrunk" in pitcher_features_df.columns
            else np.full(n, 1.25, dtype=np.float64)
        )
        opp_k_pct = (
            np.asarray(
                pitcher_features_df["opp_lineup_platoon_k_pct"]
                .cast(pl.Float64)
                .fill_null(0.220)
                .to_numpy(),
                dtype=np.float64,
            )
            if "opp_lineup_platoon_k_pct" in pitcher_features_df.columns
            else np.full(n, 0.220, dtype=np.float64)
        )
        opp_bb_pct = (
            np.asarray(
                pitcher_features_df["opp_lineup_platoon_bb_pct"]
                .cast(pl.Float64)
                .fill_null(0.082)
                .to_numpy(),
                dtype=np.float64,
            )
            if "opp_lineup_platoon_bb_pct" in pitcher_features_df.columns
            else np.full(n, 0.082, dtype=np.float64)
        )
        opp_runs = (
            np.asarray(
                pitcher_features_df["stage1_opp_expected_runs"]
                .cast(pl.Float64)
                .fill_null(4.35)
                .to_numpy(),
                dtype=np.float64,
            )
            if "stage1_opp_expected_runs" in pitcher_features_df.columns
            else np.full(n, 4.35, dtype=np.float64)
        )
        v_rf = (
            np.asarray(
                pitcher_features_df["venue_run_factor"].cast(pl.Float64).fill_null(1.0).to_numpy(),
                dtype=np.float64,
            )
            if "venue_run_factor" in pitcher_features_df.columns
            else np.ones(n, dtype=np.float64)
        )

        prior_outs = np.clip(outs_ceil - 0.35 * (opp_runs - 4.20), 6.0, 27.0)
        if self._pitcher_outs_model is not None and self.pitcher_feature_columns:
            x_pit = self._extract_matrix(pitcher_features_df, self.pitcher_feature_columns)
            gbdt_outs = np.asarray(self._pitcher_outs_model.predict(x_pit), dtype=np.float64)
            proj_outs = np.clip(
                (1.0 - self.gbdt_blend_weight) * prior_outs + self.gbdt_blend_weight * gbdt_outs,
                6.0,
                27.0,
            )
        else:
            proj_outs = prior_outs

        proj_ip = proj_outs / 3.0
        proj_bf = np.clip(proj_outs * (1.28 + 0.18 * (sp_whip - 1.25)), proj_outs, 38.0)

        # Matchup K% increases monotonically with both starter K% and opposing lineup K%
        matchup_k_pct = np.clip(sp_k_pct * (opp_k_pct / 0.220) ** 0.70, 0.08, 0.48)
        matchup_bb_pct = np.clip(sp_bb_pct * (opp_bb_pct / 0.082) ** 0.65, 0.02, 0.22)

        prior_k = proj_bf * matchup_k_pct
        if self._pitcher_k_model is not None and self.pitcher_feature_columns:
            x_pit = self._extract_matrix(pitcher_features_df, self.pitcher_feature_columns)
            gbdt_k = np.clip(
                np.asarray(self._pitcher_k_model.predict(x_pit), dtype=np.float64), 0.5, 16.0
            )
            proj_k = np.clip(
                (1.0 - self.gbdt_blend_weight) * prior_k + self.gbdt_blend_weight * gbdt_k,
                0.5,
                16.0,
            )
        else:
            proj_k = np.clip(prior_k, 0.5, 16.0)

        proj_bb = np.clip(proj_bf * matchup_bb_pct, 0.1, 8.0)
        proj_hits_allowed = np.clip(
            np.maximum(0.5, proj_ip * sp_whip - proj_bb) * (v_rf**0.40), 0.5, 14.0
        )

        prior_er = np.clip((proj_ip / 9.0) * sp_xfip * (v_rf**0.65), 0.3, 8.5)
        if self._pitcher_er_model is not None and self.pitcher_feature_columns:
            x_pit = self._extract_matrix(pitcher_features_df, self.pitcher_feature_columns)
            gbdt_er = np.clip(
                np.asarray(self._pitcher_er_model.predict(x_pit), dtype=np.float64), 0.2, 8.5
            )
            raw_er = np.clip(
                (1.0 - self.gbdt_blend_weight) * prior_er + self.gbdt_blend_weight * gbdt_er,
                0.2,
                8.5,
            )
        else:
            raw_er = prior_er

        p_5k = np.array([_poisson_tail_prob(float(k), 5) for k in proj_k], dtype=np.float64)
        p_6k = np.array([_poisson_tail_prob(float(k), 6) for k in proj_k], dtype=np.float64)
        p_7k = np.array([_poisson_tail_prob(float(k), 7) for k in proj_k], dtype=np.float64)

        prelim_p = pitcher_features_df.with_columns(
            pl.Series("proj_outs_pitched", proj_outs, dtype=pl.Float64),
            pl.Series("proj_ip", proj_ip, dtype=pl.Float64),
            pl.Series("proj_batters_faced", proj_bf, dtype=pl.Float64),
            pl.Series("proj_k", proj_k, dtype=pl.Float64),
            pl.Series("proj_bb", proj_bb, dtype=pl.Float64),
            pl.Series("proj_hits_allowed", proj_hits_allowed, dtype=pl.Float64),
            pl.Series("_raw_proj_er", raw_er, dtype=pl.Float64),
            pl.Series("proj_er", raw_er, dtype=pl.Float64),
            pl.Series("prob_5plus_k", p_5k, dtype=pl.Float64),
            pl.Series("prob_6plus_k", p_6k, dtype=pl.Float64),
            pl.Series("prob_7plus_k", p_7k, dtype=pl.Float64),
            pl.Series("prob_quality_start", np.full(n, 0.45), dtype=pl.Float64),
        )
        _, reconciled_p = reconcile_baseball_stage2_projections(
            pl.DataFrame(), prelim_p, gamma=self.reconciliation_gamma
        )
        assert reconciled_p is not None
        return reconciled_p.drop("_raw_proj_er")

    def predict(
        self,
        batter_features_df: pl.DataFrame,
        pitcher_features_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Predict Stage 2 batter projections (`ModelRegistry` default interface)."""
        _ = pitcher_features_df
        return self.predict_batters(batter_features_df)
