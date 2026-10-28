"""Stage 2 Optional EPL Outfield & Goalkeeper Matchup Projections (`stage2_player.py`).

Predicts outfield player minutes (`proj_minutes` in `[0, 90]`), counting stats
(`proj_goals`, `proj_assists`, `proj_goal_involvements`, `proj_shots`, `proj_sot`),
Poisson milestone probabilities (`prob_1plus_goal`, `prob_1plus_assist`, `prob_1plus_gi`),
and goalkeeper projections (`proj_saves`, `proj_goals_against`, `proj_save_pct`,
`prob_clean_sheet`), conditioned on Stage 1 match context and reconciled via
damped proportional scaling (`gamma = 0.50`, EPL assist share per goal `0.72`).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from numpy.typing import NDArray

from sports_ml.config import RANDOM_SEED
from sports_ml.core.market import is_forbidden_market_column

SOCCER_STAGE2_OUTFIELD_FEATURE_COLS: Final[tuple[str, ...]] = (
    "rolling_minutes",
    "rolling_start_rate",
    "availability_weight",
    "team_injury_vacated_minutes_pos",
    "rolling_npxg_per_90",
    "rolling_xg_per_90",
    "rolling_xa_per_90",
    "rolling_xg_chain_per_90",
    "rolling_xg_buildup_per_90",
    "rolling_shots_per_90",
    "rolling_sot_per_90",
    "rolling_key_passes_per_90",
    "rolling_goals_per_90",
    "rolling_assists_per_90",
    "career_shot_conversion_shrunk",
    "is_penalty_taker",
    "is_set_piece_taker",
    "opp_xga_per_90",
    "opp_sot_against_per_90",
    "opp_ppda",
    "opp_gk_gsaa_per_90_shrunk",
    "stage1_team_win_prob",
    "stage1_draw_prob",
    "stage1_expected_team_goals",
    "is_home",
    "team_rest_days",
    "is_midweek_turnaround",
)


def reconcile_outfield_team_goals(
    outfield_preds_df: pl.DataFrame,
    *,
    gamma: float = 0.50,
    assist_share_per_goal: float = 0.72,
) -> pl.DataFrame:
    """Apply cross-stage coherence scaling toward Stage 1's `stage1_expected_team_goals`.

    Formula:
        `mu_G_reconciled = mu_G_raw * (lambda_team / sum(mu_G_raw)) ** gamma`
    Preserves relative player rankings while ensuring outfield goal, assist, and
    goal-involvement (`G+A`) projections scale coherently with Stage 1's scoring environment.
    """
    if outfield_preds_df.is_empty():
        return outfield_preds_df

    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}.")

    event_col = "match_id" if "match_id" in outfield_preds_df.columns else "event_id"
    group_cols = [event_col, "team"]

    raw_team_goals_sum = pl.col("_raw_proj_goals").sum().over(group_cols)
    safe_denom = pl.when(raw_team_goals_sum > 1e-6).then(raw_team_goals_sum).otherwise(1.0)
    target_lambda = (
        pl.col("stage1_expected_team_goals").fill_null(1.45).clip(lower_bound=0.15)
        if "stage1_expected_team_goals" in outfield_preds_df.columns
        else pl.lit(1.45, dtype=pl.Float64)
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
    reconciled_gi = reconciled_goals + reconciled_assists

    return outfield_preds_df.with_columns(
        reconciled_goals.alias("proj_goals"),
        reconciled_assists.alias("proj_assists"),
        reconciled_gi.alias("proj_goal_involvements"),
        (1.0 - (-reconciled_goals).exp()).clip(0.0, 1.0).alias("prob_1plus_goal"),
        (1.0 - (-reconciled_assists).exp()).clip(0.0, 1.0).alias("prob_1plus_assist"),
        (1.0 - (-reconciled_gi).exp()).clip(0.0, 1.0).alias("prob_1plus_gi"),
    )


class Stage2SoccerPlayerModel:
    """Optional Stage 2 EPL outfield and goalkeeper matchup stat projection pipeline."""

    def __init__(
        self,
        reconciliation_gamma: float = 0.50,
        gbdt_blend_weight: float = 0.50,
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
        self._minutes_model: LGBMRegressor | None = None
        self._shots_model: LGBMRegressor | None = None
        self._sot_model: LGBMRegressor | None = None
        self._goals_model: LGBMRegressor | None = None
        self._assists_model: LGBMRegressor | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @staticmethod
    def resolve_feature_columns(df: pl.DataFrame) -> list[str]:
        """Return ordered Stage 2 outfield feature columns (excluding `market_*` / `*odds*`)."""
        candidate_set = set(SOCCER_STAGE2_OUTFIELD_FEATURE_COLS)
        preferred = [
            c for c in df.columns if c in candidate_set and not is_forbidden_market_column(c)
        ]
        if preferred:
            return preferred
        excluded = {
            "match_id",
            "game_id",
            "event_id",
            "event_seq",
            "match_date",
            "game_date",
            "start_time_utc",
            "season",
            "player_id",
            "player_name",
            "team",
            "opponent",
            "position_group",
            "is_starter",
            "minutes",
            "goals",
            "assists",
            "shots",
            "sot",
            "key_passes",
            "xg",
            "npxg",
            "xa",
            "xg_chain",
            "xg_buildup",
            "yellow_cards",
            "red_cards",
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
        if not for_scoring:
            return [0] * len(self.feature_columns)

        positive_cols = {
            "rolling_npxg_per_90",
            "rolling_xg_per_90",
            "rolling_xa_per_90",
            "rolling_shots_per_90",
            "rolling_sot_per_90",
            "rolling_goals_per_90",
            "rolling_assists_per_90",
            "opp_xga_per_90",
            "stage1_expected_team_goals",
        }
        negative_cols = {
            "opp_gk_gsaa_per_90_shrunk",
        }
        return [
            1 if col in positive_cols else (-1 if col in negative_cols else 0)
            for col in self.feature_columns
        ]

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

    def fit(self, train_df: pl.DataFrame) -> Stage2SoccerPlayerModel:
        """Fit outfield minutes projector and Poisson/Tweedie count/rate regressors."""
        if train_df.is_empty():
            raise ValueError("train_df cannot be empty.")

        self.feature_columns = self.resolve_feature_columns(train_df)
        x_train = self._extract_matrix(train_df)
        n_rows = train_df.height
        scoring_constraints = self._build_monotone_constraints(for_scoring=True)

        y_min = np.clip(
            np.asarray(train_df["minutes"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            90.0,
        )
        y_shots = np.clip(
            np.asarray(train_df["shots"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            15.0,
        )
        y_sot = np.clip(
            np.asarray(train_df["sot"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            10.0,
        )
        y_goals = np.clip(
            np.asarray(train_df["goals"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            6.0,
        )
        y_assists = np.clip(
            np.asarray(train_df["assists"].cast(pl.Float64).to_numpy(), dtype=np.float64),
            0.0,
            6.0,
        )

        self._minutes_model = self._make_regressor(n_rows, objective="regression")
        self._minutes_model.fit(x_train, y_min)

        shots_obj = "poisson" if float(np.sum(y_shots)) > 0.0 else "regression"
        self._shots_model = self._make_regressor(
            n_rows, objective=shots_obj, monotone_constraints=scoring_constraints
        )
        self._shots_model.fit(x_train, y_shots)

        sot_obj = "poisson" if float(np.sum(y_sot)) > 0.0 else "regression"
        self._sot_model = self._make_regressor(
            n_rows, objective=sot_obj, monotone_constraints=scoring_constraints
        )
        self._sot_model.fit(x_train, y_sot)

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
        """Generate Stage 2 outfield projections with cross-stage coherence scaling."""
        if not self._is_fitted:
            raise RuntimeError("Stage2SoccerPlayerModel must be fitted before calling predict.")

        if player_features_df.is_empty():
            return player_features_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("proj_minutes"),
                pl.lit(None, dtype=pl.Float64).alias("proj_shots"),
                pl.lit(None, dtype=pl.Float64).alias("proj_sot"),
                pl.lit(None, dtype=pl.Float64).alias("proj_goals"),
                pl.lit(None, dtype=pl.Float64).alias("proj_assists"),
                pl.lit(None, dtype=pl.Float64).alias("proj_goal_involvements"),
                pl.lit(None, dtype=pl.Float64).alias("prob_1plus_goal"),
                pl.lit(None, dtype=pl.Float64).alias("prob_1plus_assist"),
                pl.lit(None, dtype=pl.Float64).alias("prob_1plus_gi"),
            )

        assert self._minutes_model is not None
        assert self._shots_model is not None
        assert self._sot_model is not None
        assert self._goals_model is not None
        assert self._assists_model is not None

        n = player_features_df.height
        x_mat = self._extract_matrix(player_features_df)
        w_gbdt = self.gbdt_blend_weight

        # 1. Outfield Minutes Projector in [0, 90] scaled by availability_weight
        roll_min = np.asarray(
            player_features_df["rolling_minutes"].cast(pl.Float64).to_numpy(), dtype=np.float64
        )
        vacated = (
            np.asarray(
                player_features_df["team_injury_vacated_minutes_pos"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "team_injury_vacated_minutes_pos" in player_features_df.columns
            else np.zeros(n, dtype=np.float64)
        )
        avail_w = (
            np.asarray(
                player_features_df["availability_weight"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "availability_weight" in player_features_df.columns
            else np.ones(n, dtype=np.float64)
        )

        gbdt_min = np.asarray(self._minutes_model.predict(x_mat), dtype=np.float64)
        prior_min = np.clip(roll_min + 0.12 * vacated, 0.0, 90.0)
        proj_minutes = np.clip(
            (w_gbdt * gbdt_min + (1.0 - w_gbdt) * prior_min) * np.clip(avail_w, 0.0, 1.0),
            0.0,
            90.0,
        )

        # 2. Structural per-90 priors conditioned on opponent defense, GK GSAA/90, & Stage 1
        shots_90 = np.asarray(
            player_features_df["rolling_shots_per_90"].cast(pl.Float64).to_numpy(),
            dtype=np.float64,
        )
        sot_90 = np.asarray(
            player_features_df["rolling_sot_per_90"].cast(pl.Float64).to_numpy(),
            dtype=np.float64,
        )
        npxg_90 = np.asarray(
            player_features_df["rolling_npxg_per_90"].cast(pl.Float64).to_numpy(),
            dtype=np.float64,
        )
        goals_90 = np.asarray(
            player_features_df["rolling_goals_per_90"].cast(pl.Float64).to_numpy(),
            dtype=np.float64,
        )
        xa_90 = np.asarray(
            player_features_df["rolling_xa_per_90"].cast(pl.Float64).to_numpy(),
            dtype=np.float64,
        )
        assists_90 = np.asarray(
            player_features_df["rolling_assists_per_90"].cast(pl.Float64).to_numpy(),
            dtype=np.float64,
        )
        is_pen = (
            np.asarray(
                player_features_df["is_penalty_taker"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "is_penalty_taker" in player_features_df.columns
            else np.zeros(n, dtype=np.float64)
        )
        is_sp = (
            np.asarray(
                player_features_df["is_set_piece_taker"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "is_set_piece_taker" in player_features_df.columns
            else np.zeros(n, dtype=np.float64)
        )
        opp_xga_90 = (
            np.asarray(
                player_features_df["opp_xga_per_90"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "opp_xga_per_90" in player_features_df.columns
            else np.full(n, 1.45, dtype=np.float64)
        )
        opp_gk_gsaa = (
            np.asarray(
                player_features_df["opp_gk_gsaa_per_90_shrunk"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "opp_gk_gsaa_per_90_shrunk" in player_features_df.columns
            else np.zeros(n, dtype=np.float64)
        )
        stage1_lambda = (
            np.asarray(
                player_features_df["stage1_expected_team_goals"].cast(pl.Float64).to_numpy(),
                dtype=np.float64,
            )
            if "stage1_expected_team_goals" in player_features_df.columns
            else np.full(n, 1.45, dtype=np.float64)
        )

        gk_env_mod = np.exp(-0.18 * opp_gk_gsaa)
        def_env_mod = np.clip(opp_xga_90 / 1.45, 0.60, 1.60)
        pace_env_mod = np.clip(stage1_lambda / 1.45, 0.35, 2.50)
        min_share = proj_minutes / 90.0

        struct_shots = np.clip(shots_90 * min_share * np.sqrt(pace_env_mod), 0.0, 10.0)
        struct_sot = np.clip(sot_90 * min_share * np.sqrt(pace_env_mod) * gk_env_mod, 0.0, 6.0)
        eff_goal_rate_90 = 0.60 * npxg_90 + 0.40 * goals_90 + 0.12 * is_pen
        eff_assist_rate_90 = 0.60 * xa_90 + 0.40 * assists_90 + 0.06 * is_sp

        struct_goals = np.clip(
            eff_goal_rate_90 * min_share * def_env_mod * gk_env_mod * pace_env_mod,
            1e-5,
            4.0,
        )
        struct_assists = np.clip(
            eff_assist_rate_90 * min_share * def_env_mod * gk_env_mod * pace_env_mod,
            1e-5,
            4.0,
        )

        gbdt_shots = np.clip(
            np.asarray(self._shots_model.predict(x_mat), dtype=np.float64), 0.0, 12.0
        ) * np.sqrt(pace_env_mod)
        gbdt_sot = (
            np.clip(np.asarray(self._sot_model.predict(x_mat), dtype=np.float64), 0.0, 8.0)
            * np.sqrt(pace_env_mod)
            * gk_env_mod
        )
        gbdt_goals = (
            np.clip(np.asarray(self._goals_model.predict(x_mat), dtype=np.float64), 0.0, 4.0)
            * gk_env_mod
            * pace_env_mod
        )
        gbdt_assists = (
            np.clip(np.asarray(self._assists_model.predict(x_mat), dtype=np.float64), 0.0, 4.0)
            * gk_env_mod
            * pace_env_mod
        )

        proj_shots = np.clip(
            (w_gbdt * gbdt_shots + (1.0 - w_gbdt) * struct_shots) * np.clip(avail_w, 0.0, 1.0),
            0.0,
            12.0,
        )
        proj_sot = np.clip(
            np.minimum(
                proj_shots,
                (w_gbdt * gbdt_sot + (1.0 - w_gbdt) * struct_sot) * np.clip(avail_w, 0.0, 1.0),
            ),
            0.0,
            8.0,
        )
        raw_goals = np.clip(
            (w_gbdt * gbdt_goals + (1.0 - w_gbdt) * struct_goals) * np.clip(avail_w, 0.0, 1.0),
            1e-5,
            4.0,
        )
        raw_assists = np.clip(
            (w_gbdt * gbdt_assists + (1.0 - w_gbdt) * struct_assists) * np.clip(avail_w, 0.0, 1.0),
            1e-5,
            4.0,
        )

        with_raw = player_features_df.with_columns(
            pl.Series("proj_minutes", proj_minutes, dtype=pl.Float64),
            pl.Series("proj_shots", proj_shots, dtype=pl.Float64),
            pl.Series("proj_sot", proj_sot, dtype=pl.Float64),
            pl.Series("_raw_proj_goals", raw_goals, dtype=pl.Float64),
            pl.Series("_raw_proj_assists", raw_assists, dtype=pl.Float64),
        )

        reconciled = reconcile_outfield_team_goals(
            with_raw,
            gamma=self.reconciliation_gamma,
        )
        return reconciled.drop(["_raw_proj_goals", "_raw_proj_assists"])

    def predict_goalkeepers(
        self,
        gk_features_df: pl.DataFrame,
        *,
        outfield_projections_df: pl.DataFrame | None = None,
        stage1_predictions_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Predict GK `proj_saves`, `proj_goals_against`, `proj_save_pct`, and `prob_clean_sheet`.

        Combines Stage 1 opponent expected goals `lambda_opp` (Poisson clean-sheet baseline
        `exp(-lambda_opp)`) with the goalkeeper's Bayesian-shrunk `gsaa_per_90_shrunk` and
        `save_pct_shrunk`.
        """
        if gk_features_df.is_empty():
            return gk_features_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("proj_sot_against"),
                pl.lit(None, dtype=pl.Float64).alias("proj_goals_against"),
                pl.lit(None, dtype=pl.Float64).alias("proj_saves"),
                pl.lit(None, dtype=pl.Float64).alias("proj_save_pct"),
                pl.lit(None, dtype=pl.Float64).alias("prob_clean_sheet"),
            )

        event_col = "match_id" if "match_id" in gk_features_df.columns else "event_id"
        working = gk_features_df

        if outfield_projections_df is not None and not outfield_projections_df.is_empty():
            o_ev_col = "match_id" if "match_id" in outfield_projections_df.columns else "event_id"
            opp_sot_sum = (
                outfield_projections_df.group_by([o_ev_col, "team"])
                .agg(pl.col("proj_sot").sum().alias("_opp_sum_sot"))
                .rename({o_ev_col: event_col, "team": "opponent"})
            )
            working = working.join(opp_sot_sum, on=[event_col, "opponent"], how="left")
        else:
            working = working.with_columns(pl.lit(None, dtype=pl.Float64).alias("_opp_sum_sot"))

        if stage1_predictions_df is not None and not stage1_predictions_df.is_empty():
            s1_ev_col = "match_id" if "match_id" in stage1_predictions_df.columns else "event_id"
            s1_lookup = stage1_predictions_df.select(
                [
                    pl.col(s1_ev_col).alias(event_col),
                    pl.col("expected_home_goals").alias("_s1_h_goals"),
                    pl.col("expected_away_goals").alias("_s1_a_goals"),
                ]
            ).unique(subset=[event_col], keep="last")
            working = working.join(s1_lookup, on=[event_col], how="left").with_columns(
                pl.when(pl.col("is_home"))
                .then(pl.col("_s1_a_goals"))
                .otherwise(pl.col("_s1_h_goals"))
                .alias("_opp_expected_goals")
            )
        else:
            working = working.with_columns(
                pl.lit(1.38, dtype=pl.Float64).alias("_opp_expected_goals")
            )

        sot_against = (
            pl.coalesce([pl.col("_opp_sum_sot"), pl.lit(4.4, dtype=pl.Float64)])
            .clip(lower_bound=1.0, upper_bound=14.0)
            .alias("proj_sot_against")
        )
        gsaa_adj = (
            pl.col("gsaa_per_90_shrunk").fill_null(0.0)
            if "gsaa_per_90_shrunk" in working.columns
            else pl.lit(0.0, dtype=pl.Float64)
        )
        sv_pct_prior = (
            pl.col("save_pct_shrunk").fill_null(0.710)
            if "save_pct_shrunk" in working.columns
            else pl.lit(0.710, dtype=pl.Float64)
        )

        with_sot = working.with_columns(sot_against)
        raw_ga_from_sv = pl.col("proj_sot_against") * (1.0 - sv_pct_prior)
        opp_lam = pl.col("_opp_expected_goals").fill_null(1.38)
        reconciled_ga = (0.65 * opp_lam + 0.35 * raw_ga_from_sv - 0.35 * gsaa_adj).clip(
            lower_bound=0.10
        )

        with_ga = with_sot.with_columns(
            pl.min_horizontal(reconciled_ga, pl.col("proj_sot_against")).alias("proj_goals_against")
        )
        with_saves = with_ga.with_columns(
            (pl.col("proj_sot_against") - pl.col("proj_goals_against"))
            .clip(lower_bound=0.0)
            .alias("proj_saves")
        )
        with_derived = with_saves.with_columns(
            (pl.col("proj_saves") / pl.col("proj_sot_against").clip(lower_bound=1.0))
            .clip(0.0, 1.0)
            .alias("proj_save_pct"),
            (-pl.col("proj_goals_against")).exp().clip(0.01, 0.95).alias("prob_clean_sheet"),
        )

        drop_cols = [
            c
            for c in ("_opp_sum_sot", "_s1_h_goals", "_s1_a_goals", "_opp_expected_goals")
            if c in with_derived.columns
        ]
        return with_derived.drop(drop_cols)
