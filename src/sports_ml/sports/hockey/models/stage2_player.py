"""Stage 2 Optional Player Matchup Stat Projections and coherence reconciler (`stage2_player.py`).

Predicts skater TOI (`proj_toi_ev_min`, `proj_toi_pp_min`, `proj_toi_min`),
counting stats (`proj_sog`, `proj_goals`, `proj_assists`, `proj_points`),
Poisson milestone probabilities (`prob_1plus_goal`, `prob_1plus_point`), and
goalie projections (`proj_shots_against`, `proj_goals_against`, `proj_saves`,
`proj_save_pct`), conditioned on Stage 1 game context and reconciled via
damped proportional scaling (`gamma = 0.50`).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from numpy.typing import NDArray

from sports_ml.config import RANDOM_SEED
from sports_ml.core.market import is_forbidden_market_column

STAGE2_SKATER_FEATURE_COLS: Final[tuple[str, ...]] = (
    "rolling_toi_ev_min",
    "rolling_toi_pp_min",
    "rolling_toi_total_min",
    "rolling_shifts_per_game",
    "team_injury_vacated_toi_f_or_d",
    "rolling_sog_per_60",
    "rolling_icf_per_60",
    "rolling_ixg_per_60",
    "rolling_goals_per_60",
    "rolling_assists_per_60",
    "rolling_points_per_60",
    "career_shooting_pct_shrunk",
    "opp_xga_per_60_5v5",
    "opp_sa_per_60_5v5",
    "opp_pk_xga_per_60",
    "opp_penalties_taken_per_60",
    "opp_goalie_gsax_per_60_shrunk",
    "stage1_team_win_prob",
    "stage1_expected_team_goals",
    "is_home",
    "team_rest_days",
    "is_b2b",
)


def reconcile_skater_team_goals(
    skater_preds_df: pl.DataFrame,
    *,
    gamma: float = 0.50,
    assist_share_per_goal: float = 1.68,
) -> pl.DataFrame:
    """Apply cross-stage coherence scaling toward Stage 1's `stage1_expected_team_goals`.

    Formula:
        `mu_G_reconciled = mu_G_raw * (lambda_team / sum(mu_G_raw)) ** gamma`
    Preserves relative player rankings while ensuring skater goal/assist/point
    projections scale coherently with Stage 1's scoring environment.
    """
    if skater_preds_df.is_empty():
        return skater_preds_df

    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}.")

    event_col = "game_id" if "game_id" in skater_preds_df.columns else "event_id"
    group_cols = [event_col, "team"]

    raw_team_goals_sum = pl.col("_raw_proj_goals").sum().over(group_cols)
    safe_denom = pl.when(raw_team_goals_sum > 1e-6).then(raw_team_goals_sum).otherwise(1.0)
    target_lambda = (
        pl.col("stage1_expected_team_goals").fill_null(3.05).clip(lower_bound=0.25)
        if "stage1_expected_team_goals" in skater_preds_df.columns
        else pl.lit(3.05, dtype=pl.Float64)
    )

    scale_factor = (target_lambda / safe_denom).pow(gamma)

    reconciled_goals = (pl.col("_raw_proj_goals") * scale_factor).clip(lower_bound=0.0)
    raw_team_assists_sum = pl.col("_raw_proj_assists").sum().over(group_cols)
    safe_assist_denom = (
        pl.when(raw_team_assists_sum > 1e-6).then(raw_team_assists_sum).otherwise(1.0)
    )
    target_team_assists = target_lambda * assist_share_per_goal
    assist_scale = (target_team_assists / safe_assist_denom).pow(gamma)
    reconciled_assists = (pl.col("_raw_proj_assists") * assist_scale).clip(lower_bound=0.0)
    reconciled_points = reconciled_goals + reconciled_assists

    return skater_preds_df.with_columns(
        reconciled_goals.alias("proj_goals"),
        reconciled_assists.alias("proj_assists"),
        reconciled_points.alias("proj_points"),
        (1.0 - (-reconciled_goals).exp()).clip(0.0, 1.0).alias("prob_1plus_goal"),
        (1.0 - (-reconciled_points).exp()).clip(0.0, 1.0).alias("prob_1plus_point"),
    )


class Stage2PlayerProjectionModel:
    """Optional Stage 2 NHL skater and goalie matchup stat projection pipeline."""

    def __init__(
        self,
        reconciliation_gamma: float = 0.50,
        gbdt_blend_weight: float = 0.55,
        n_estimators: int = 60,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
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

        self.feature_columns: list[str] = []
        self._toi_ev_model: LGBMRegressor | None = None
        self._toi_pp_model: LGBMRegressor | None = None
        self._sog_model: LGBMRegressor | None = None
        self._goals_model: LGBMRegressor | None = None
        self._assists_model: LGBMRegressor | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @staticmethod
    def resolve_feature_columns(df: pl.DataFrame) -> list[str]:
        """Return ordered Stage 2 skater feature columns (excluding `market_*` / `*odds*`)."""
        candidate_set = set(STAGE2_SKATER_FEATURE_COLS)
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
            "player_id",
            "player_name",
            "team",
            "opponent",
            "position_group",
            "toi_total_min",
            "toi_ev_min",
            "toi_pp_min",
            "toi_pk_min",
            "goals",
            "assists",
            "points",
            "sog",
        }
        fallback = [
            col
            for col, dtype in df.schema.items()
            if col not in excluded
            and not is_forbidden_market_column(col)
            and (dtype.is_numeric() or dtype == pl.Boolean)
        ]
        if not fallback:
            raise ValueError("No valid numeric Stage 2 feature columns found in DataFrame.")
        return fallback

    def _extract_matrix(self, df: pl.DataFrame) -> NDArray[np.float64]:
        mat = df.select(
            [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in self.feature_columns]
        ).to_numpy()
        return np.asarray(mat, dtype=np.float64)

    def _build_monotone_constraints(self, for_scoring: bool = True) -> list[int]:
        """Build LightGBM monotone constraint vector for Stage 2 feature columns."""
        if not for_scoring:
            return [0] * len(self.feature_columns)

        positive_cols = {
            "rolling_sog_per_60",
            "rolling_icf_per_60",
            "rolling_ixg_per_60",
            "rolling_goals_per_60",
            "rolling_assists_per_60",
            "rolling_points_per_60",
            "opp_xga_per_60_5v5",
            "stage1_expected_team_goals",
        }
        negative_cols = {
            "opp_goalie_gsax_per_60_shrunk",
        }
        constraints: list[int] = []
        for col in self.feature_columns:
            if col in positive_cols:
                constraints.append(1)
            elif col in negative_cols:
                constraints.append(-1)
            else:
                constraints.append(0)
        return constraints

    def _make_regressor(
        self,
        n_rows: int,
        objective: str = "poisson",
        monotone_constraints: list[int] | None = None,
    ) -> LGBMRegressor:
        min_child = max(2, min(20, n_rows // 6))
        leaves = max(3, min(self.num_leaves, max(3, n_rows // 4)))
        return LGBMRegressor(
            objective=objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=leaves,
            min_child_samples=min_child,
            monotone_constraints=monotone_constraints,
            random_state=self.random_seed,
            n_jobs=1,
            verbose=-1,
        )

    def fit(self, train_df: pl.DataFrame) -> Stage2PlayerProjectionModel:
        """Fit skater TOI projectors and Poisson/Tweedie count/rate regressors."""
        if train_df.is_empty():
            raise ValueError("train_df cannot be empty.")

        self.feature_columns = self.resolve_feature_columns(train_df)
        x_train = self._extract_matrix(train_df)
        n_rows = train_df.height
        scoring_constraints = self._build_monotone_constraints(for_scoring=True)

        y_toi_ev = np.clip(
            np.asarray(train_df["toi_ev_min"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.1,
            35.0,
        )
        y_toi_pp = np.clip(
            np.asarray(train_df["toi_pp_min"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            15.0,
        )
        y_sog = np.clip(
            np.asarray(train_df["sog"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            20.0,
        )
        y_goals = np.clip(
            np.asarray(train_df["goals"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            10.0,
        )
        y_assists = np.clip(
            np.asarray(train_df["assists"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            10.0,
        )

        self._toi_ev_model = self._make_regressor(n_rows, objective="regression")
        self._toi_ev_model.fit(x_train, y_toi_ev)

        pp_objective = "tweedie" if float(np.sum(y_toi_pp)) > 0.0 else "regression"
        self._toi_pp_model = self._make_regressor(n_rows, objective=pp_objective)
        self._toi_pp_model.fit(x_train, y_toi_pp)

        sog_obj = "poisson" if float(np.sum(y_sog)) > 0.0 else "regression"
        self._sog_model = self._make_regressor(
            n_rows, objective=sog_obj, monotone_constraints=scoring_constraints
        )
        self._sog_model.fit(x_train, y_sog)

        goals_obj = "poisson" if float(np.sum(y_goals)) > 0.0 else "regression"
        self._goals_model = self._make_regressor(
            n_rows, objective=goals_obj, monotone_constraints=scoring_constraints
        )
        self._goals_model.fit(x_train, y_goals)

        assists_obj = "poisson" if float(np.sum(y_assists)) > 0.0 else "regression"
        self._assists_model = self._make_regressor(
            n_rows, objective=assists_obj, monotone_constraints=scoring_constraints
        )
        self._assists_model.fit(x_train, y_assists)

        self._is_fitted = True
        return self

    def predict(self, player_features_df: pl.DataFrame) -> pl.DataFrame:
        """Generate Stage 2 skater projections with cross-stage coherence scaling."""
        if not self._is_fitted:
            raise RuntimeError("Stage2PlayerProjectionModel must be fitted before calling predict.")

        if player_features_df.is_empty():
            return player_features_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("proj_toi_ev_min"),
                pl.lit(None, dtype=pl.Float64).alias("proj_toi_pp_min"),
                pl.lit(None, dtype=pl.Float64).alias("proj_toi_min"),
                pl.lit(None, dtype=pl.Float64).alias("proj_sog"),
                pl.lit(None, dtype=pl.Float64).alias("proj_goals"),
                pl.lit(None, dtype=pl.Float64).alias("proj_assists"),
                pl.lit(None, dtype=pl.Float64).alias("proj_points"),
                pl.lit(None, dtype=pl.Float64).alias("prob_1plus_goal"),
                pl.lit(None, dtype=pl.Float64).alias("prob_1plus_point"),
            )

        assert self._toi_ev_model is not None
        assert self._toi_pp_model is not None
        assert self._sog_model is not None
        assert self._goals_model is not None
        assert self._assists_model is not None

        x_mat = self._extract_matrix(player_features_df)
        w_gbdt = self.gbdt_blend_weight

        # 1. Skater TOI Projector (blends GBDT with rolling usage + vacated injury TOI bump)
        roll_ev = np.asarray(
            player_features_df["rolling_toi_ev_min"].cast(pl.Float64).to_numpy(), dtype=np.float64
        )
        roll_pp = np.asarray(
            player_features_df["rolling_toi_pp_min"].cast(pl.Float64).to_numpy(), dtype=np.float64
        )
        vacated = (
            np.asarray(
                player_features_df["team_injury_vacated_toi_f_or_d"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "team_injury_vacated_toi_f_or_d" in player_features_df.columns
            else np.zeros(player_features_df.height, dtype=np.float64)
        )

        gbdt_ev = np.asarray(self._toi_ev_model.predict(x_mat), dtype=np.float64)
        gbdt_pp = np.asarray(self._toi_pp_model.predict(x_mat), dtype=np.float64)

        prior_ev = np.clip(roll_ev + 0.08 * vacated, 2.0, 28.0)
        prior_pp = np.clip(roll_pp + 0.02 * vacated, 0.0, 10.0)

        proj_ev = np.clip(w_gbdt * gbdt_ev + (1.0 - w_gbdt) * prior_ev, 1.0, 30.0)
        proj_pp = np.clip(w_gbdt * gbdt_pp + (1.0 - w_gbdt) * prior_pp, 0.0, 12.0)
        proj_tot_toi = np.clip(proj_ev + proj_pp, 1.0, 35.0)

        # 2. Structural rate*exposure priors conditioned on opponent defense, goalie GSAx & Stage 1
        sog_60 = np.asarray(
            player_features_df["rolling_sog_per_60"].cast(pl.Float64).to_numpy(), dtype=np.float64
        )
        goals_60 = np.asarray(
            player_features_df["rolling_goals_per_60"].cast(pl.Float64).to_numpy(), dtype=np.float64
        )
        assists_60 = np.asarray(
            player_features_df["rolling_assists_per_60"].cast(pl.Float64).to_numpy(),
            dtype=np.float64,
        )
        opp_xga_60 = (
            np.asarray(
                player_features_df["opp_xga_per_60_5v5"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "opp_xga_per_60_5v5" in player_features_df.columns
            else np.full(player_features_df.height, 2.55, dtype=np.float64)
        )
        opp_gsax_60 = (
            np.asarray(
                player_features_df["opp_goalie_gsax_per_60_shrunk"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "opp_goalie_gsax_per_60_shrunk" in player_features_df.columns
            else np.zeros(player_features_df.height, dtype=np.float64)
        )
        stage1_lambda = (
            np.asarray(
                player_features_df["stage1_expected_team_goals"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "stage1_expected_team_goals" in player_features_df.columns
            else np.full(player_features_df.height, 3.05, dtype=np.float64)
        )

        # Opponent goalie suppression modifier: higher opponent GSAx/60 lowers scoring expectation
        goalie_env_mod = np.exp(-0.12 * opp_gsax_60)
        def_env_mod = np.clip(opp_xga_60 / 2.55, 0.65, 1.45)
        pace_env_mod = np.clip(stage1_lambda / 3.05, 0.50, 2.00)

        struct_sog = np.clip((sog_60 * (proj_tot_toi / 60.0)) * np.sqrt(pace_env_mod), 0.05, 12.0)
        struct_goals = np.clip(
            (goals_60 * (proj_tot_toi / 60.0)) * def_env_mod * goalie_env_mod,
            0.01,
            5.0,
        )
        struct_assists = np.clip(
            (assists_60 * (proj_tot_toi / 60.0)) * def_env_mod * goalie_env_mod,
            0.01,
            5.0,
        )

        gbdt_sog = np.clip(np.asarray(self._sog_model.predict(x_mat), dtype=np.float64), 0.0, 15.0)
        gbdt_goals = (
            np.clip(np.asarray(self._goals_model.predict(x_mat), dtype=np.float64), 0.0, 6.0)
            * goalie_env_mod
        )
        gbdt_assists = (
            np.clip(np.asarray(self._assists_model.predict(x_mat), dtype=np.float64), 0.0, 6.0)
            * goalie_env_mod
        )

        proj_sog = np.clip(w_gbdt * gbdt_sog + (1.0 - w_gbdt) * struct_sog, 0.0, 15.0)
        raw_goals = np.clip(w_gbdt * gbdt_goals + (1.0 - w_gbdt) * struct_goals, 1e-4, 6.0)
        raw_assists = np.clip(w_gbdt * gbdt_assists + (1.0 - w_gbdt) * struct_assists, 1e-4, 6.0)

        with_raw = player_features_df.with_columns(
            pl.Series("proj_toi_ev_min", proj_ev, dtype=pl.Float64),
            pl.Series("proj_toi_pp_min", proj_pp, dtype=pl.Float64),
            pl.Series("proj_toi_min", proj_tot_toi, dtype=pl.Float64),
            pl.Series("proj_sog", proj_sog, dtype=pl.Float64),
            pl.Series("_raw_proj_goals", raw_goals, dtype=pl.Float64),
            pl.Series("_raw_proj_assists", raw_assists, dtype=pl.Float64),
        )

        reconciled = reconcile_skater_team_goals(with_raw, gamma=self.reconciliation_gamma)
        return reconciled.drop(["_raw_proj_goals", "_raw_proj_assists"])

    def predict_goalies(
        self,
        goalie_features_df: pl.DataFrame,
        *,
        skater_projections_df: pl.DataFrame | None = None,
        stage1_predictions_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Predict goalie `proj_shots_against`, `proj_goals_against`, `proj_saves`, and `SV%`."""
        if goalie_features_df.is_empty():
            return goalie_features_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("proj_shots_against"),
                pl.lit(None, dtype=pl.Float64).alias("proj_goals_against"),
                pl.lit(None, dtype=pl.Float64).alias("proj_saves"),
                pl.lit(None, dtype=pl.Float64).alias("proj_save_pct"),
            )

        event_col = "game_id" if "game_id" in goalie_features_df.columns else "event_id"
        working = goalie_features_df

        # Anchor shots against to opponent skater sum(proj_sog) when available
        if skater_projections_df is not None and not skater_projections_df.is_empty():
            s_ev_col = "game_id" if "game_id" in skater_projections_df.columns else "event_id"
            opp_sog_sum = (
                skater_projections_df.group_by([s_ev_col, "team"])
                .agg(pl.col("proj_sog").sum().alias("_opp_sum_proj_sog"))
                .rename({s_ev_col: event_col, "team": "opponent"})
            )
            working = working.join(opp_sog_sum, on=[event_col, "opponent"], how="left")
        else:
            working = working.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("_opp_sum_proj_sog")
            )

        # Join Stage 1 opponent expected goals if available
        if stage1_predictions_df is not None and not stage1_predictions_df.is_empty():
            s1_ev_col = "game_id" if "game_id" in stage1_predictions_df.columns else "event_id"
            s1_lookup = stage1_predictions_df.select(
                [
                    pl.col(s1_ev_col).alias(event_col),
                    pl.col("expected_home_goals").alias("_s1_h_g"),
                    pl.col("expected_away_goals").alias("_s1_a_g"),
                ]
            ).unique(subset=[event_col], keep="last")
            working = (
                working.join(s1_lookup, on=[event_col], how="left")
                .with_columns(
                    pl.when(pl.col("is_home"))
                    .then(pl.col("_s1_a_g"))
                    .otherwise(pl.col("_s1_h_g"))
                    .alias("_opp_expected_goals")
                )
                .drop(["_s1_h_g", "_s1_a_g"])
            )
        else:
            working = working.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("_opp_expected_goals")
            )

        gsax_col = (
            pl.col("gsax_per_60_shrunk")
            if "gsax_per_60_shrunk" in working.columns
            else pl.lit(0.0, dtype=pl.Float64)
        )
        sv_pct_col = (
            pl.col("save_pct_shrunk")
            if "save_pct_shrunk" in working.columns
            else pl.lit(0.905, dtype=pl.Float64)
        )

        proj_sa = (
            pl.when(pl.col("_opp_sum_proj_sog").is_not_null() & (pl.col("_opp_sum_proj_sog") > 5.0))
            .then(pl.col("_opp_sum_proj_sog"))
            .otherwise(29.5)
            .clip(lower_bound=1.0)
        )

        base_ga_from_sv = proj_sa * (1.0 - sv_pct_col) - 0.50 * gsax_col
        proj_ga = (
            pl.when(pl.col("_opp_expected_goals").is_not_null())
            .then(0.60 * (pl.col("_opp_expected_goals") - 0.35 * gsax_col) + 0.40 * base_ga_from_sv)
            .otherwise(base_ga_from_sv)
            .clip(lower_bound=0.10)
        )
        # Ensure proj_ga never exceeds proj_sa
        capped_ga = pl.min_horizontal(proj_ga, proj_sa)
        proj_sv = (proj_sa - capped_ga).clip(lower_bound=0.0)
        proj_sv_pct = (proj_sv / proj_sa).clip(0.0, 1.0)

        return working.with_columns(
            proj_sa.alias("proj_shots_against"),
            capped_ga.alias("proj_goals_against"),
            proj_sv.alias("proj_saves"),
            proj_sv_pct.alias("proj_save_pct"),
        ).drop(["_opp_sum_proj_sog", "_opp_expected_goals"])
