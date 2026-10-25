"""Team 5v5 process, shot quality, PDO, and special teams rolling features (`features/team.py`).

Computes strictly leak-free pre-game rolling (5, 10, 25 games) and EWMA (alpha=0.12)
features for 5v5 xGF%, xGF/60, xGA/60, CF%, FF%, HD-xGF%, PDO, Finishing Above Expected,
and Special Teams (PP/PK) with Bayesian prior shrinkage (`k = 8.0`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import (
    pregame_ewma,
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_rolling_sum,
    pregame_sample_count,
    sort_events_chronologically,
)

TEAM_SHRINKAGE_K: Final[float] = 8.0
DEFAULT_TEAM_WINDOWS: Final[tuple[int, ...]] = (5, 10, 25)
DEFAULT_EWMA_ALPHA: Final[float] = 0.12

# Causal league-average priors for NHL team metrics
PRIOR_SHARE_50: Final[float] = 0.50
PRIOR_XGF_PER_60: Final[float] = 2.55
PRIOR_XGA_PER_60: Final[float] = 2.55
PRIOR_SA_PER_60: Final[float] = 29.5
PRIOR_PDO: Final[float] = 1.000
PRIOR_PP_XGF_PER_60: Final[float] = 7.50
PRIOR_PK_XGA_PER_60: Final[float] = 7.50
PRIOR_PP_PCT: Final[float] = 0.205
PRIOR_PK_PCT: Final[float] = 0.795
PRIOR_DRAW_RATE: Final[float] = 3.0


def _ensure_team_log_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Populate optional team game-log columns with defaults so minimal fixtures work cleanly."""
    defaults: list[tuple[str, float | int, pl.DataType]] = [
        ("sog_for", 30, pl.Int16()),
        ("sog_against", 30, pl.Int16()),
        ("cf_5v5", 50, pl.Int16()),
        ("ca_5v5", 50, pl.Int16()),
        ("ff_5v5", 38, pl.Int16()),
        ("fa_5v5", 38, pl.Int16()),
        ("hd_xgf_5v5", 1.0, pl.Float64()),
        ("hd_xga_5v5", 1.0, pl.Float64()),
        ("toi_5v5_minutes", 50.0, pl.Float64()),
        ("pp_goals", 0.6, pl.Float64()),
        ("pp_opportunities", 3.0, pl.Float64()),
        ("pp_xgf", 0.6, pl.Float64()),
        ("pp_toi_min", 4.8, pl.Float64()),
        ("pk_goals_against", 0.6, pl.Float64()),
        ("pk_times_shorthanded", 3.0, pl.Float64()),
        ("pk_xga", 0.6, pl.Float64()),
        ("pk_toi_min", 4.8, pl.Float64()),
    ]
    missing_exprs = [
        pl.lit(val, dtype=dtype).alias(col) for col, val, dtype in defaults if col not in df.columns
    ]
    return df.with_columns(missing_exprs) if missing_exprs else df


def _pregame_share_expr(
    for_col: str,
    against_col: str,
    window: int,
    group_by: str,
) -> pl.Expr:
    """Compute leak-free shifted rolling share `sum(for) / (sum(for) + sum(against))`."""
    f_sum = pregame_rolling_sum(for_col, window=window, group_by=group_by)
    a_sum = pregame_rolling_sum(against_col, window=window, group_by=group_by)
    denom = f_sum + a_sum
    return f_sum / pl.when(denom > 0.0).then(denom).otherwise(None)


def build_team_rolling_features(
    team_game_logs: pl.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_TEAM_WINDOWS,
    ewma_alpha: float = DEFAULT_EWMA_ALPHA,
    shrinkage_k: float = TEAM_SHRINKAGE_K,
) -> pl.DataFrame:
    """Compute leak-free pre-game team process and special-teams rolling features."""
    prepared = _ensure_team_log_columns(team_game_logs)
    sorted_df = sort_events_chronologically(prepared, extra_tiebreakers=["team"])

    # Compute per-game derived series strictly before shifting inside core/temporal helpers
    with_game_derived = sorted_df.with_columns(
        (pl.col("goals_for").cast(pl.Float64) - pl.col("xgf_5v5").cast(pl.Float64)).alias(
            "_game_finishing_above_xg"
        ),
        (
            pl.col("xgf_5v5").cast(pl.Float64)
            / pl.when((pl.col("xgf_5v5") + pl.col("xga_5v5")) > 0.0)
            .then(pl.col("xgf_5v5") + pl.col("xga_5v5"))
            .otherwise(0.50)
        ).alias("_game_xgf_share_5v5"),
        pregame_sample_count("xgf_5v5", group_by="team").alias("team_prior_games_n"),
    )

    window_exprs: list[pl.Expr] = []
    for w in windows:
        n_w_col = f"_n_w{w}"
        raw_xgf_pct = _pregame_share_expr("xgf_5v5", "xga_5v5", window=w, group_by="team")
        raw_cf_pct = _pregame_share_expr("cf_5v5", "ca_5v5", window=w, group_by="team")
        raw_ff_pct = _pregame_share_expr("ff_5v5", "fa_5v5", window=w, group_by="team")
        raw_hd_pct = _pregame_share_expr("hd_xgf_5v5", "hd_xga_5v5", window=w, group_by="team")

        raw_xgf_60 = pregame_exposure_rate(
            "xgf_5v5", "toi_5v5_minutes", window=w, group_by="team", scale_factor=60.0
        )
        raw_xga_60 = pregame_exposure_rate(
            "xga_5v5", "toi_5v5_minutes", window=w, group_by="team", scale_factor=60.0
        )
        raw_sa_60 = pregame_exposure_rate(
            "sog_against", "toi_5v5_minutes", window=w, group_by="team", scale_factor=60.0
        )

        # PDO = (GF / SOG_F) + (1 - GA / SOG_A)
        raw_sh_pct = pregame_exposure_rate(
            "goals_for", "sog_for", window=w, group_by="team", scale_factor=1.0
        )
        raw_ga_rate = pregame_exposure_rate(
            "goals_against", "sog_against", window=w, group_by="team", scale_factor=1.0
        )
        raw_pdo = raw_sh_pct + (1.0 - raw_ga_rate)

        raw_finishing = pregame_rolling_mean("_game_finishing_above_xg", window=w, group_by="team")

        # Special teams rates
        raw_pp_xgf_60 = pregame_exposure_rate(
            "pp_xgf", "pp_toi_min", window=w, group_by="team", scale_factor=60.0
        )
        raw_pk_xga_60 = pregame_exposure_rate(
            "pk_xga", "pk_toi_min", window=w, group_by="team", scale_factor=60.0
        )
        raw_pp_pct = pregame_exposure_rate(
            "pp_goals", "pp_opportunities", window=w, group_by="team", scale_factor=1.0
        )
        raw_pk_kill = 1.0 - pregame_exposure_rate(
            "pk_goals_against",
            "pk_times_shorthanded",
            window=w,
            group_by="team",
            scale_factor=1.0,
        )
        raw_draw_rate = pregame_rolling_mean("pp_opportunities", window=w, group_by="team")
        raw_penalties_taken = pregame_rolling_mean(
            "pk_times_shorthanded", window=w, group_by="team"
        )

        n_w_expr = pregame_sample_count("xgf_5v5", group_by="team", window=w)

        window_exprs.extend(
            [
                n_w_expr.alias(n_w_col),
                bayesian_shrinkage(raw_xgf_pct, n_w_expr, PRIOR_SHARE_50, shrinkage_k).alias(
                    f"xgf_pct_5v5_w{w}"
                ),
                bayesian_shrinkage(raw_xgf_60, n_w_expr, PRIOR_XGF_PER_60, shrinkage_k).alias(
                    f"xgf_per_60_5v5_w{w}"
                ),
                bayesian_shrinkage(raw_xga_60, n_w_expr, PRIOR_XGA_PER_60, shrinkage_k).alias(
                    f"xga_per_60_5v5_w{w}"
                ),
                bayesian_shrinkage(raw_sa_60, n_w_expr, PRIOR_SA_PER_60, shrinkage_k).alias(
                    f"sa_per_60_5v5_w{w}"
                ),
                bayesian_shrinkage(raw_cf_pct, n_w_expr, PRIOR_SHARE_50, shrinkage_k).alias(
                    f"cf_pct_5v5_w{w}"
                ),
                bayesian_shrinkage(raw_ff_pct, n_w_expr, PRIOR_SHARE_50, shrinkage_k).alias(
                    f"ff_pct_5v5_w{w}"
                ),
                bayesian_shrinkage(raw_hd_pct, n_w_expr, PRIOR_SHARE_50, shrinkage_k).alias(
                    f"hd_xgf_pct_5v5_w{w}"
                ),
                bayesian_shrinkage(raw_pdo, n_w_expr, PRIOR_PDO, shrinkage_k).alias(
                    f"pdo_5v5_w{w}"
                ),
                bayesian_shrinkage(raw_finishing, n_w_expr, 0.0, shrinkage_k).alias(
                    f"finishing_above_xg_w{w}"
                ),
                bayesian_shrinkage(raw_pp_xgf_60, n_w_expr, PRIOR_PP_XGF_PER_60, shrinkage_k).alias(
                    f"pp_xgf_per_60_w{w}"
                ),
                bayesian_shrinkage(raw_pk_xga_60, n_w_expr, PRIOR_PK_XGA_PER_60, shrinkage_k).alias(
                    f"pk_xga_per_60_w{w}"
                ),
                bayesian_shrinkage(raw_pp_pct, n_w_expr, PRIOR_PP_PCT, shrinkage_k).alias(
                    f"pp_pct_w{w}"
                ),
                bayesian_shrinkage(raw_pk_kill, n_w_expr, PRIOR_PK_PCT, shrinkage_k).alias(
                    f"pk_pct_w{w}"
                ),
                bayesian_shrinkage(raw_draw_rate, n_w_expr, PRIOR_DRAW_RATE, shrinkage_k).alias(
                    f"pp_draw_rate_w{w}"
                ),
                bayesian_shrinkage(
                    raw_penalties_taken, n_w_expr, PRIOR_DRAW_RATE, shrinkage_k
                ).alias(f"penalties_taken_per_60_w{w}"),
            ]
        )

    # EWMA features (shrunk by total prior games n)
    raw_ewma_xgf_share = pregame_ewma("_game_xgf_share_5v5", alpha=ewma_alpha, group_by="team")
    raw_ewma_xgf = pregame_ewma("xgf_5v5", alpha=ewma_alpha, group_by="team")
    raw_ewma_xga = pregame_ewma("xga_5v5", alpha=ewma_alpha, group_by="team")

    window_exprs.extend(
        [
            bayesian_shrinkage(
                raw_ewma_xgf_share,
                pl.col("team_prior_games_n"),
                PRIOR_SHARE_50,
                shrinkage_k,
            ).alias("xgf_pct_5v5_ewma"),
            bayesian_shrinkage(
                raw_ewma_xgf,
                pl.col("team_prior_games_n"),
                2.10,
                shrinkage_k,
            ).alias("xgf_5v5_ewma"),
            bayesian_shrinkage(
                raw_ewma_xga,
                pl.col("team_prior_games_n"),
                2.10,
                shrinkage_k,
            ).alias("xga_5v5_ewma"),
        ]
    )

    result = with_game_derived.with_columns(window_exprs)

    # Alias canonical 10-game rolling metrics for downstream matchup and Stage 2 builders
    ref_w = 10 if 10 in windows else windows[0]
    result = result.with_columns(
        (pl.col(f"pdo_5v5_w{ref_w}") - PRIOR_PDO).alias("pdo_regression_signal"),
        pl.col(f"xga_per_60_5v5_w{ref_w}").alias("xga_per_60_5v5"),
        pl.col(f"sa_per_60_5v5_w{ref_w}").alias("sa_per_60_5v5"),
        pl.col(f"pk_xga_per_60_w{ref_w}").alias("pk_xga_per_60"),
        pl.col(f"pp_xgf_per_60_w{ref_w}").alias("pp_xgf_per_60"),
        pl.col(f"penalties_taken_per_60_w{ref_w}").alias("penalties_taken_per_60"),
        pl.col(f"pp_draw_rate_w{ref_w}").alias("pp_draw_rate"),
    )

    drop_temp = [
        "_game_finishing_above_xg",
        "_game_xgf_share_5v5",
        *[f"_n_w{w}" for w in windows],
    ]
    return result.drop([c for c in drop_temp if c in result.columns])


def compute_special_teams_net_edge(
    home_pp_xgf_per_60: str | float | pl.Expr,
    away_pk_xga_per_60: str | float | pl.Expr,
    home_draw_rate: str | float | pl.Expr,
    away_pp_xgf_per_60: str | float | pl.Expr,
    home_pk_xga_per_60: str | float | pl.Expr,
    away_draw_rate: str | float | pl.Expr,
) -> pl.Expr:
    """Compute Special Teams Net Edge `(PP_H - PK_A) * Draw_H - (PP_A - PK_H) * Draw_A`."""
    h_pp = (
        pl.col(home_pp_xgf_per_60)
        if isinstance(home_pp_xgf_per_60, str)
        else pl.lit(home_pp_xgf_per_60)
    )
    a_pk = (
        pl.col(away_pk_xga_per_60)
        if isinstance(away_pk_xga_per_60, str)
        else pl.lit(away_pk_xga_per_60)
    )
    h_draw = pl.col(home_draw_rate) if isinstance(home_draw_rate, str) else pl.lit(home_draw_rate)
    a_pp = (
        pl.col(away_pp_xgf_per_60)
        if isinstance(away_pp_xgf_per_60, str)
        else pl.lit(away_pp_xgf_per_60)
    )
    h_pk = (
        pl.col(home_pk_xga_per_60)
        if isinstance(home_pk_xga_per_60, str)
        else pl.lit(home_pk_xga_per_60)
    )
    a_draw = pl.col(away_draw_rate) if isinstance(away_draw_rate, str) else pl.lit(away_draw_rate)

    home_edge = (h_pp - a_pk) * (h_draw / 60.0)
    away_edge = (a_pp - h_pk) * (a_draw / 60.0)
    return home_edge - away_edge


def _compute_postgame_team_features(team_with_sched: pl.DataFrame) -> pl.DataFrame:
    """Compute post-game rolling team metrics so upcoming (`FUT`) games can inherit latest state."""
    from sports_ml.core.shrinkage import bayesian_shrinkage
    from sports_ml.core.temporal import sort_events_chronologically

    sorted_t = sort_events_chronologically(team_with_sched, extra_tiebreakers=["team"])
    w = 10
    f_xg = pl.col("xgf_5v5").cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over("team")
    a_xg = pl.col("xga_5v5").cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over("team")
    f_cf = pl.col("cf_5v5").cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over("team")
    a_cf = pl.col("ca_5v5").cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over("team")
    f_ff = pl.col("ff_5v5").cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over("team")
    a_ff = pl.col("fa_5v5").cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over("team")
    f_hd = (
        pl.col("hd_xgf_5v5").cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over("team")
    )
    a_hd = (
        pl.col("hd_xga_5v5").cast(pl.Float64).rolling_sum(window_size=w, min_samples=1).over("team")
    )
    toi_5v5 = (
        pl.col("toi_5v5_minutes")
        .cast(pl.Float64)
        .rolling_sum(window_size=w, min_samples=1)
        .over("team")
    )
    n_post = (
        pl.col("xgf_5v5")
        .is_not_null()
        .cast(pl.Int32)
        .rolling_sum(window_size=w, min_samples=1)
        .over("team")
    )

    return sorted_t.with_columns(
        bayesian_shrinkage(f_xg / (f_xg + a_xg), n_post, PRIOR_SHARE_50, TEAM_SHRINKAGE_K).alias(
            "_post_xgf_pct_5v5_w10"
        ),
        bayesian_shrinkage(60.0 * f_xg / toi_5v5, n_post, PRIOR_XGF_PER_60, TEAM_SHRINKAGE_K).alias(
            "_post_xgf_per_60_5v5_w10"
        ),
        bayesian_shrinkage(60.0 * a_xg / toi_5v5, n_post, PRIOR_XGA_PER_60, TEAM_SHRINKAGE_K).alias(
            "_post_xga_per_60_5v5_w10"
        ),
        bayesian_shrinkage(f_cf / (f_cf + a_cf), n_post, PRIOR_SHARE_50, TEAM_SHRINKAGE_K).alias(
            "_post_cf_pct_5v5_w10"
        ),
        bayesian_shrinkage(f_ff / (f_ff + a_ff), n_post, PRIOR_SHARE_50, TEAM_SHRINKAGE_K).alias(
            "_post_ff_pct_5v5_w10"
        ),
        bayesian_shrinkage(f_hd / (f_hd + a_hd), n_post, PRIOR_SHARE_50, TEAM_SHRINKAGE_K).alias(
            "_post_hd_xgf_pct_5v5_w10"
        ),
    )


def build_stage1_game_features(
    games_df: pl.DataFrame,
    team_game_logs: pl.DataFrame,
    skater_game_logs: pl.DataFrame | None = None,
    goalie_game_logs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Assemble the complete Stage 1 pre-game matchup feature matrix (`Home`, `Away`, and `Diff`).

    Combines Dynamic Elo, schedule/rest/travel, team 5v5 process/PDO/Special Teams Net Edge,
    starting goalie shrunk GSAx/60, and TOI-weighted active roster strength.
    For upcoming (`FUT`) games without boxscore rows in `team_game_logs`, inherits each
    team's strictly prior completed state via backward `join_asof`.
    """
    from sports_ml.core.ratings import PairwiseEloEngine
    from sports_ml.core.schedule import compute_schedule_features
    from sports_ml.sports.hockey.features.goalie import (
        PRIOR_GOALIE_SAVE_PCT,
        PRIOR_GOALIE_GSAx_PER_60,
        build_goalie_rolling_features,
        resolve_team_starting_goalie_features,
    )
    from sports_ml.sports.hockey.features.roster import (
        PRIOR_GAMESCORE_PER_60_D,
        PRIOR_GAMESCORE_PER_60_F,
        PRIOR_IXG_PER_60_D,
        PRIOR_IXG_PER_60_F,
        build_roster_strength_features,
    )

    event_col = "game_id" if "game_id" in games_df.columns else "event_id"
    elo_df = PairwiseEloEngine().compute_ratings(games_df)

    team_feat = build_team_rolling_features(team_game_logs)
    team_with_sched = _compute_postgame_team_features(compute_schedule_features(team_feat))
    t_ev_col = "game_id" if "game_id" in team_with_sched.columns else "event_id"

    core_team_cols = [
        "xgf_pct_5v5_w10",
        "xgf_pct_5v5_ewma",
        "xgf_per_60_5v5_w10",
        "xga_per_60_5v5_w10",
        "cf_pct_5v5_w10",
        "ff_pct_5v5_w10",
        "hd_xgf_pct_5v5_w10",
        "pdo_5v5_w10",
        "pdo_regression_signal",
        "finishing_above_xg_w10",
        "pp_xgf_per_60",
        "pk_xga_per_60",
        "pp_pct_w10",
        "pk_pct_w10",
        "pp_draw_rate",
        "rest_days",
        "is_b2b",
        "games_last_4d",
        "games_last_6d",
    ]
    available_team_cols = [c for c in core_team_cols if c in team_with_sched.columns]

    home_team_lookup = team_with_sched.select(
        [
            pl.col(t_ev_col).alias(event_col),
            pl.col("team").alias("home_team"),
            *[pl.col(c).alias(f"{c}_home") for c in available_team_cols],
        ]
    ).unique(subset=[event_col, "home_team"], keep="last")
    away_team_lookup = team_with_sched.select(
        [
            pl.col(t_ev_col).alias(event_col),
            pl.col("team").alias("away_team"),
            *[pl.col(c).alias(f"{c}_away") for c in available_team_cols],
        ]
    ).unique(subset=[event_col, "away_team"], keep="last")

    matchup = elo_df.join(home_team_lookup, on=[event_col, "home_team"], how="left").join(
        away_team_lookup, on=[event_col, "away_team"], how="left"
    )

    # For upcoming FUT games missing from team_game_logs, backfill via causal backward join_asof
    if not team_with_sched.is_empty():
        asof_source = team_with_sched.sort("start_time_utc").select(
            [
                pl.col("start_time_utc"),
                pl.col("team"),
                pl.col("_post_xgf_pct_5v5_w10").alias("xgf_pct_5v5_w10"),
                pl.col("xgf_pct_5v5_ewma"),
                pl.col("_post_xgf_per_60_5v5_w10").alias("xgf_per_60_5v5_w10"),
                pl.col("_post_xga_per_60_5v5_w10").alias("xga_per_60_5v5_w10"),
                pl.col("_post_cf_pct_5v5_w10").alias("cf_pct_5v5_w10"),
                pl.col("_post_ff_pct_5v5_w10").alias("ff_pct_5v5_w10"),
                pl.col("_post_hd_xgf_pct_5v5_w10").alias("hd_xgf_pct_5v5_w10"),
                pl.col("pdo_5v5_w10"),
                pl.col("pdo_regression_signal"),
                pl.col("finishing_above_xg_w10"),
                pl.col("pp_xgf_per_60"),
                pl.col("pk_xga_per_60"),
                pl.col("pp_pct_w10"),
                pl.col("pk_pct_w10"),
                pl.col("pp_draw_rate"),
            ]
        )
        asof_cols = [c for c in asof_source.columns if c not in {"start_time_utc", "team"}]
        matchup = (
            matchup.sort(["home_team", "start_time_utc"])
            .join_asof(
                asof_source.sort(["team", "start_time_utc"]).rename(
                    {"team": "home_team", **{c: f"_asof_{c}_home" for c in asof_cols}}
                ),
                on="start_time_utc",
                by="home_team",
                strategy="backward",
                check_sortedness=False,
            )
            .sort(["away_team", "start_time_utc"])
            .join_asof(
                asof_source.sort(["team", "start_time_utc"]).rename(
                    {"team": "away_team", **{c: f"_asof_{c}_away" for c in asof_cols}}
                ),
                on="start_time_utc",
                by="away_team",
                strategy="backward",
                check_sortedness=False,
            )
            .sort("start_time_utc")
        )
        fill_exprs: list[pl.Expr] = []
        for c in asof_cols:
            for side in ("home", "away"):
                col_s = f"{c}_{side}"
                asof_s = f"_asof_{c}_{side}"
                if col_s in matchup.columns and asof_s in matchup.columns:
                    fill_exprs.append(pl.col(col_s).fill_null(pl.col(asof_s)).alias(col_s))
        matchup = (
            matchup.with_columns(fill_exprs)
            .drop([f"_asof_{c}_{side}" for c in asof_cols for side in ("home", "away")])
            .with_columns(
                pl.col("rest_days_home").fill_null(2),
                pl.col("rest_days_away").fill_null(2),
                pl.col("is_b2b_home").fill_null(False),
                pl.col("is_b2b_away").fill_null(False),
                pl.col("games_last_4d_home").fill_null(2),
                pl.col("games_last_4d_away").fill_null(2),
                pl.col("games_last_6d_home").fill_null(3),
                pl.col("games_last_6d_away").fill_null(3),
            )
        )

    if goalie_game_logs is not None and not goalie_game_logs.is_empty():
        g_feat = resolve_team_starting_goalie_features(
            build_goalie_rolling_features(goalie_game_logs)
        )
        g_ev_col = "game_id" if "game_id" in g_feat.columns else "event_id"
        g_cols = [
            "gsax_per_60_shrunk",
            "save_pct_shrunk",
            "goalie_days_rest",
            "goalie_starts_last_7d",
        ]
        home_g = g_feat.select(
            [
                pl.col(g_ev_col).alias(event_col),
                pl.col("team").alias("home_team"),
                *[pl.col(c).alias(f"{c}_home") for c in g_cols],
            ]
        )
        away_g = g_feat.select(
            [
                pl.col(g_ev_col).alias(event_col),
                pl.col("team").alias("away_team"),
                *[pl.col(c).alias(f"{c}_away") for c in g_cols],
            ]
        )
        matchup = matchup.join(home_g, on=[event_col, "home_team"], how="left").join(
            away_g, on=[event_col, "away_team"], how="left"
        )
        g_asof_src = g_feat.sort(["team", "start_time_utc"]).select(
            ["start_time_utc", "team", *g_cols]
        )
        matchup = (
            matchup.sort(["home_team", "start_time_utc"])
            .join_asof(
                g_asof_src.rename({"team": "home_team", **{c: f"_g_{c}_home" for c in g_cols}}),
                on="start_time_utc",
                by="home_team",
                strategy="backward",
                check_sortedness=False,
            )
            .sort(["away_team", "start_time_utc"])
            .join_asof(
                g_asof_src.rename({"team": "away_team", **{c: f"_g_{c}_away" for c in g_cols}}),
                on="start_time_utc",
                by="away_team",
                strategy="backward",
                check_sortedness=False,
            )
            .with_columns(
                [
                    pl.col(f"{c}_{s}").fill_null(pl.col(f"_g_{c}_{s}")).alias(f"{c}_{s}")
                    for c in g_cols
                    for s in ("home", "away")
                ]
            )
            .drop([f"_g_{c}_{s}" for c in g_cols for s in ("home", "away")])
            .with_columns(
                pl.col("gsax_per_60_shrunk_home").fill_null(PRIOR_GOALIE_GSAx_PER_60),
                pl.col("gsax_per_60_shrunk_away").fill_null(PRIOR_GOALIE_GSAx_PER_60),
                pl.col("save_pct_shrunk_home").fill_null(PRIOR_GOALIE_SAVE_PCT),
                pl.col("save_pct_shrunk_away").fill_null(PRIOR_GOALIE_SAVE_PCT),
                pl.col("goalie_days_rest_home").fill_null(7),
                pl.col("goalie_days_rest_away").fill_null(7),
                pl.col("goalie_starts_last_7d_home").fill_null(0),
                pl.col("goalie_starts_last_7d_away").fill_null(0),
            )
        )
    else:
        matchup = matchup.with_columns(
            pl.lit(PRIOR_GOALIE_GSAx_PER_60).alias("gsax_per_60_shrunk_home"),
            pl.lit(PRIOR_GOALIE_GSAx_PER_60).alias("gsax_per_60_shrunk_away"),
            pl.lit(PRIOR_GOALIE_SAVE_PCT).alias("save_pct_shrunk_home"),
            pl.lit(PRIOR_GOALIE_SAVE_PCT).alias("save_pct_shrunk_away"),
            pl.lit(7, dtype=pl.Int32).alias("goalie_days_rest_home"),
            pl.lit(7, dtype=pl.Int32).alias("goalie_days_rest_away"),
            pl.lit(0, dtype=pl.Int32).alias("goalie_starts_last_7d_home"),
            pl.lit(0, dtype=pl.Int32).alias("goalie_starts_last_7d_away"),
        )

    if skater_game_logs is not None and not skater_game_logs.is_empty():
        r_feat = build_roster_strength_features(skater_game_logs)
        r_ev_col = "game_id" if "game_id" in r_feat.columns else "event_id"
        r_cols = [
            "forward_weighted_ixg_per_60",
            "forward_weighted_xgd_per_60",
            "forward_weighted_gs_per_60",
            "defense_weighted_ixg_per_60",
            "defense_weighted_xgd_per_60",
            "defense_weighted_gs_per_60",
            "missing_regular_toi",
        ]
        home_r = r_feat.select(
            [
                pl.col(r_ev_col).alias(event_col),
                pl.col("team").alias("home_team"),
                *[pl.col(c).alias(f"{c}_home") for c in r_cols],
            ]
        )
        away_r = r_feat.select(
            [
                pl.col(r_ev_col).alias(event_col),
                pl.col("team").alias("away_team"),
                *[pl.col(c).alias(f"{c}_away") for c in r_cols],
            ]
        )
        matchup = matchup.join(home_r, on=[event_col, "home_team"], how="left").join(
            away_r, on=[event_col, "away_team"], how="left"
        )
        r_asof_src = r_feat.sort(["team", "start_time_utc"]).select(
            ["start_time_utc", "team", *r_cols]
        )
        matchup = (
            matchup.sort(["home_team", "start_time_utc"])
            .join_asof(
                r_asof_src.rename({"team": "home_team", **{c: f"_r_{c}_home" for c in r_cols}}),
                on="start_time_utc",
                by="home_team",
                strategy="backward",
                check_sortedness=False,
            )
            .sort(["away_team", "start_time_utc"])
            .join_asof(
                r_asof_src.rename({"team": "away_team", **{c: f"_r_{c}_away" for c in r_cols}}),
                on="start_time_utc",
                by="away_team",
                strategy="backward",
                check_sortedness=False,
            )
            .with_columns(
                [
                    pl.col(f"{c}_{s}").fill_null(pl.col(f"_r_{c}_{s}")).alias(f"{c}_{s}")
                    for c in r_cols
                    for s in ("home", "away")
                ]
            )
            .drop([f"_r_{c}_{s}" for c in r_cols for s in ("home", "away")])
            .with_columns(
                pl.col("forward_weighted_ixg_per_60_home").fill_null(PRIOR_IXG_PER_60_F),
                pl.col("forward_weighted_ixg_per_60_away").fill_null(PRIOR_IXG_PER_60_F),
                pl.col("forward_weighted_xgd_per_60_home").fill_null(0.0),
                pl.col("forward_weighted_xgd_per_60_away").fill_null(0.0),
                pl.col("forward_weighted_gs_per_60_home").fill_null(PRIOR_GAMESCORE_PER_60_F),
                pl.col("forward_weighted_gs_per_60_away").fill_null(PRIOR_GAMESCORE_PER_60_F),
                pl.col("defense_weighted_ixg_per_60_home").fill_null(PRIOR_IXG_PER_60_D),
                pl.col("defense_weighted_ixg_per_60_away").fill_null(PRIOR_IXG_PER_60_D),
                pl.col("defense_weighted_xgd_per_60_home").fill_null(0.0),
                pl.col("defense_weighted_xgd_per_60_away").fill_null(0.0),
                pl.col("defense_weighted_gs_per_60_home").fill_null(PRIOR_GAMESCORE_PER_60_D),
                pl.col("defense_weighted_gs_per_60_away").fill_null(PRIOR_GAMESCORE_PER_60_D),
                pl.col("missing_regular_toi_home").fill_null(0.0),
                pl.col("missing_regular_toi_away").fill_null(0.0),
            )
        )
        matchup = sort_events_chronologically(matchup)
    else:
        matchup = matchup.with_columns(
            pl.lit(PRIOR_IXG_PER_60_F).alias("forward_weighted_ixg_per_60_home"),
            pl.lit(PRIOR_IXG_PER_60_F).alias("forward_weighted_ixg_per_60_away"),
            pl.lit(0.0).alias("forward_weighted_xgd_per_60_home"),
            pl.lit(0.0).alias("forward_weighted_xgd_per_60_away"),
            pl.lit(PRIOR_GAMESCORE_PER_60_F).alias("forward_weighted_gs_per_60_home"),
            pl.lit(PRIOR_GAMESCORE_PER_60_F).alias("forward_weighted_gs_per_60_away"),
            pl.lit(PRIOR_IXG_PER_60_D).alias("defense_weighted_ixg_per_60_home"),
            pl.lit(PRIOR_IXG_PER_60_D).alias("defense_weighted_ixg_per_60_away"),
            pl.lit(0.0).alias("defense_weighted_xgd_per_60_home"),
            pl.lit(0.0).alias("defense_weighted_xgd_per_60_away"),
            pl.lit(PRIOR_GAMESCORE_PER_60_D).alias("defense_weighted_gs_per_60_home"),
            pl.lit(PRIOR_GAMESCORE_PER_60_D).alias("defense_weighted_gs_per_60_away"),
            pl.lit(0.0).alias("missing_regular_toi_home"),
            pl.lit(0.0).alias("missing_regular_toi_away"),
        )

    return matchup.with_columns(
        compute_special_teams_net_edge(
            "pp_xgf_per_60_home",
            "pk_xga_per_60_away",
            "pp_draw_rate_home",
            "pp_xgf_per_60_away",
            "pk_xga_per_60_home",
            "pp_draw_rate_away",
        ).alias("st_net_edge_home_vs_away"),
        (pl.col("rest_days_home") - pl.col("rest_days_away")).alias("rest_advantage"),
        (pl.col("xgf_pct_5v5_w10_home") - pl.col("xgf_pct_5v5_w10_away")).alias(
            "xgf_pct_5v5_w10_diff"
        ),
        (pl.col("cf_pct_5v5_w10_home") - pl.col("cf_pct_5v5_w10_away")).alias(
            "cf_pct_5v5_w10_diff"
        ),
        (pl.col("gsax_per_60_shrunk_home") - pl.col("gsax_per_60_shrunk_away")).alias(
            "goalie_gsax_per_60_diff"
        ),
        (
            pl.col("forward_weighted_ixg_per_60_home") - pl.col("forward_weighted_ixg_per_60_away")
        ).alias("forward_ixg_per_60_diff"),
        (
            pl.col("defense_weighted_xgd_per_60_home") - pl.col("defense_weighted_xgd_per_60_away")
        ).alias("defense_xgd_per_60_diff"),
    )
