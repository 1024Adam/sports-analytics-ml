"""Stage 2 individual skater and goalie matchup feature builder (`features/player.py`).

Constructs leak-free pre-game individual rolling usage (`5, 10, 25` games + EWMA),
vacated injury TOI (`team_injury_vacated_toi_f_or_d`), per-60 production & shot-quality
rates, shrunk career shooting %, opponent defensive suppression, opponent starting
goalie GSAx, and Stage 1 conditioning signals (`stage1_team_win_prob`,
`stage1_expected_team_goals`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from sports_ml.core.schedule import compute_schedule_features
from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import (
    pregame_ewma,
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.hockey.features.goalie import (
    PRIOR_GOALIE_GSAx_PER_60,
    resolve_team_starting_goalie_features,
)
from sports_ml.sports.hockey.features.roster import (
    PRIOR_IXG_PER_60_D,
    PRIOR_IXG_PER_60_F,
    PRIOR_TOI_EV_D,
    PRIOR_TOI_EV_F,
    PRIOR_TOI_PP_D,
    PRIOR_TOI_PP_F,
    PRIOR_TOI_TOTAL_D,
    PRIOR_TOI_TOTAL_F,
    SKATER_SHRINKAGE_K,
    _ensure_skater_log_columns,
    build_roster_strength_features,
)
from sports_ml.sports.hockey.features.team import (
    DEFAULT_EWMA_ALPHA,
    PRIOR_DRAW_RATE,
    PRIOR_PK_XGA_PER_60,
    PRIOR_SA_PER_60,
    PRIOR_XGA_PER_60,
)

DEFAULT_PLAYER_WINDOWS: Final[tuple[int, ...]] = (5, 10, 25)
CAREER_SH_PCT_SHRINKAGE_K: Final[float] = 12.0

PRIOR_SOG_PER_60_F: Final[float] = 7.50
PRIOR_SOG_PER_60_D: Final[float] = 4.80
PRIOR_ICF_PER_60_F: Final[float] = 13.50
PRIOR_ICF_PER_60_D: Final[float] = 10.50
PRIOR_GOALS_PER_60_F: Final[float] = 0.80
PRIOR_GOALS_PER_60_D: Final[float] = 0.25
PRIOR_ASSISTS_PER_60_F: Final[float] = 1.25
PRIOR_ASSISTS_PER_60_D: Final[float] = 0.90
PRIOR_POINTS_PER_60_F: Final[float] = 2.05
PRIOR_POINTS_PER_60_D: Final[float] = 1.15
PRIOR_SH_PCT_F: Final[float] = 0.105
PRIOR_SH_PCT_D: Final[float] = 0.052


def build_stage2_player_features(
    skater_game_logs: pl.DataFrame,
    team_features: pl.DataFrame,
    goalie_features: pl.DataFrame,
    *,
    skater_pbp_micro_logs: pl.DataFrame | None = None,
    stage1_predictions: pl.DataFrame | None = None,
    windows: Sequence[int] = DEFAULT_PLAYER_WINDOWS,
    ewma_alpha: float = DEFAULT_EWMA_ALPHA,
    shrinkage_k: float = SKATER_SHRINKAGE_K,
) -> pl.DataFrame:
    """Construct leak-free pre-game player-matchup features for Stage 2 stat projections."""
    prepared = _ensure_skater_log_columns(skater_game_logs)
    event_col = "game_id" if "game_id" in prepared.columns else "event_id"

    if skater_pbp_micro_logs is not None and not skater_pbp_micro_logs.is_empty():
        sm_ev_col = "game_id" if "game_id" in skater_pbp_micro_logs.columns else "event_id"
        sm_cols = [
            "rush_ixg_5v5",
            "rebound_ixg_5v5",
            "inner_slot_ixg_5v5",
            "qoc_opponent_toi_share",
            "qot_teammate_ixg_per_60",
        ]
        avail_sm = [c for c in sm_cols if c in skater_pbp_micro_logs.columns]
        sm_lookup = skater_pbp_micro_logs.select(
            [
                pl.col(sm_ev_col).cast(pl.Int64).alias(event_col),
                pl.col("player_id").cast(pl.Int64),
                *[pl.col(c).cast(pl.Float64).alias(f"_sm_{c}") for c in avail_sm],
            ]
        ).unique(subset=[event_col, "player_id"], keep="last")
        prepared = prepared.join(sm_lookup, on=[event_col, "player_id"], how="left")
        for c in avail_sm:
            if c in prepared.columns:
                prepared = prepared.with_columns(
                    pl.coalesce([pl.col(f"_sm_{c}"), pl.col(c).cast(pl.Float64)]).alias(c)
                )
            else:
                prepared = prepared.with_columns(pl.col(f"_sm_{c}").alias(c))
        prepared = prepared.drop([f"_sm_{c}" for c in avail_sm])

    skater_micro_defaults: list[tuple[str, pl.Expr]] = [
        ("rush_ixg_5v5", 0.14 * pl.col("ixg_5v5").cast(pl.Float64)),
        ("rebound_ixg_5v5", 0.14 * pl.col("ixg_5v5").cast(pl.Float64)),
        ("inner_slot_ixg_5v5", 0.42 * pl.col("ixg_5v5").cast(pl.Float64)),
        ("qoc_opponent_toi_share", pl.lit(0.30, dtype=pl.Float64)),
        ("qot_teammate_ixg_per_60", pl.lit(0.72, dtype=pl.Float64)),
    ]
    for col_name, fallback_expr in skater_micro_defaults:
        if col_name in prepared.columns:
            prepared = prepared.with_columns(
                pl.col(col_name).cast(pl.Float64).fill_null(fallback_expr).alias(col_name)
            )
        else:
            prepared = prepared.with_columns(fallback_expr.alias(col_name))

    prepared = prepared.with_columns(
        (
            pl.col("rush_ixg_5v5").cast(pl.Float64) + pl.col("rebound_ixg_5v5").cast(pl.Float64)
        ).alias("_game_rush_rebound_ixg")
    )
    if "shifts" not in prepared.columns:
        prepared = prepared.with_columns(
            (pl.col("toi_total_min").cast(pl.Float64) * 1.25).alias("shifts")
        )

    sorted_df = sort_events_chronologically(prepared, extra_tiebreakers=["player_id"])

    is_d = pl.col("position_group") == "D"
    with_priors = sorted_df.with_columns(
        pl.when(is_d).then(PRIOR_TOI_TOTAL_D).otherwise(PRIOR_TOI_TOTAL_F).alias("_p_toi_tot"),
        pl.when(is_d).then(PRIOR_TOI_EV_D).otherwise(PRIOR_TOI_EV_F).alias("_p_toi_ev"),
        pl.when(is_d).then(PRIOR_TOI_PP_D).otherwise(PRIOR_TOI_PP_F).alias("_p_toi_pp"),
        pl.when(is_d).then(PRIOR_SOG_PER_60_D).otherwise(PRIOR_SOG_PER_60_F).alias("_p_sog_60"),
        pl.when(is_d).then(PRIOR_ICF_PER_60_D).otherwise(PRIOR_ICF_PER_60_F).alias("_p_icf_60"),
        pl.when(is_d).then(PRIOR_IXG_PER_60_D).otherwise(PRIOR_IXG_PER_60_F).alias("_p_ixg_60"),
        pl.when(is_d).then(PRIOR_GOALS_PER_60_D).otherwise(PRIOR_GOALS_PER_60_F).alias("_p_g_60"),
        pl.when(is_d)
        .then(PRIOR_ASSISTS_PER_60_D)
        .otherwise(PRIOR_ASSISTS_PER_60_F)
        .alias("_p_a_60"),
        pl.when(is_d)
        .then(PRIOR_POINTS_PER_60_D)
        .otherwise(PRIOR_POINTS_PER_60_F)
        .alias("_p_pts_60"),
        pl.when(is_d).then(PRIOR_SH_PCT_D).otherwise(PRIOR_SH_PCT_F).alias("_p_sh_pct"),
        pregame_sample_count("toi_total_min", group_by="player_id").alias("player_prior_games_n"),
    )

    exprs: list[pl.Expr] = []
    for w in windows:
        n_w = pregame_sample_count("toi_total_min", group_by="player_id", window=w)
        r_ev = pregame_rolling_mean("toi_ev_min", window=w, group_by="player_id")
        r_pp = pregame_rolling_mean("toi_pp_min", window=w, group_by="player_id")
        r_tot = pregame_rolling_mean("toi_total_min", window=w, group_by="player_id")
        r_shifts = pregame_rolling_mean("shifts", window=w, group_by="player_id")

        r_sog_60 = pregame_exposure_rate(
            "sog", "toi_total_min", window=w, group_by="player_id", scale_factor=60.0
        )
        r_icf_60 = pregame_exposure_rate(
            "icf_all", "toi_total_min", window=w, group_by="player_id", scale_factor=60.0
        )
        r_ixg_60 = pregame_exposure_rate(
            "ixg_all", "toi_total_min", window=w, group_by="player_id", scale_factor=60.0
        )
        r_g_60 = pregame_exposure_rate(
            "goals", "toi_total_min", window=w, group_by="player_id", scale_factor=60.0
        )
        r_a_60 = pregame_exposure_rate(
            "assists", "toi_total_min", window=w, group_by="player_id", scale_factor=60.0
        )
        r_pts_60 = pregame_exposure_rate(
            "points", "toi_total_min", window=w, group_by="player_id", scale_factor=60.0
        )
        r_rush_reb_60 = pregame_exposure_rate(
            "_game_rush_rebound_ixg",
            "toi_total_min",
            window=w,
            group_by="player_id",
            scale_factor=60.0,
        )
        r_qoc = pregame_rolling_mean("qoc_opponent_toi_share", window=w, group_by="player_id")
        r_qot = pregame_rolling_mean("qot_teammate_ixg_per_60", window=w, group_by="player_id")

        exprs.extend(
            [
                bayesian_shrinkage(r_ev, n_w, pl.col("_p_toi_ev"), shrinkage_k).alias(
                    f"rolling_toi_ev_min_w{w}"
                ),
                bayesian_shrinkage(r_pp, n_w, pl.col("_p_toi_pp"), shrinkage_k).alias(
                    f"rolling_toi_pp_min_w{w}"
                ),
                bayesian_shrinkage(r_tot, n_w, pl.col("_p_toi_tot"), shrinkage_k).alias(
                    f"rolling_toi_total_min_w{w}"
                ),
                bayesian_shrinkage(r_shifts, n_w, 20.0, shrinkage_k).alias(
                    f"rolling_shifts_per_game_w{w}"
                ),
                bayesian_shrinkage(r_sog_60, n_w, pl.col("_p_sog_60"), shrinkage_k).alias(
                    f"rolling_sog_per_60_w{w}"
                ),
                bayesian_shrinkage(r_icf_60, n_w, pl.col("_p_icf_60"), shrinkage_k).alias(
                    f"rolling_icf_per_60_w{w}"
                ),
                bayesian_shrinkage(r_ixg_60, n_w, pl.col("_p_ixg_60"), shrinkage_k).alias(
                    f"rolling_ixg_per_60_w{w}"
                ),
                bayesian_shrinkage(r_g_60, n_w, pl.col("_p_g_60"), shrinkage_k).alias(
                    f"rolling_goals_per_60_w{w}"
                ),
                bayesian_shrinkage(r_a_60, n_w, pl.col("_p_a_60"), shrinkage_k).alias(
                    f"rolling_assists_per_60_w{w}"
                ),
                bayesian_shrinkage(r_pts_60, n_w, pl.col("_p_pts_60"), shrinkage_k).alias(
                    f"rolling_points_per_60_w{w}"
                ),
                bayesian_shrinkage(
                    r_rush_reb_60, n_w, 0.28 * pl.col("_p_ixg_60"), shrinkage_k
                ).alias(f"rolling_rush_rebound_ixg_per_60_w{w}"),
                bayesian_shrinkage(r_qoc, n_w, 0.30, shrinkage_k).alias(f"rolling_qoc_index_w{w}"),
                bayesian_shrinkage(r_qot, n_w, 0.72, shrinkage_k).alias(
                    f"rolling_qot_teammate_ixg_per_60_w{w}"
                ),
            ]
        )

    # EWMA usage & career shrunk shooting percentage
    n_total = pl.col("player_prior_games_n")
    raw_ewma_ev = pregame_ewma("toi_ev_min", alpha=ewma_alpha, group_by="player_id")
    raw_ewma_pp = pregame_ewma("toi_pp_min", alpha=ewma_alpha, group_by="player_id")
    raw_career_sh = pregame_exposure_rate(
        "goals", "sog", window=250, group_by="player_id", scale_factor=1.0
    )

    exprs.extend(
        [
            bayesian_shrinkage(raw_ewma_ev, n_total, pl.col("_p_toi_ev"), shrinkage_k).alias(
                "rolling_toi_ev_min_ewma"
            ),
            bayesian_shrinkage(raw_ewma_pp, n_total, pl.col("_p_toi_pp"), shrinkage_k).alias(
                "rolling_toi_pp_min_ewma"
            ),
            bayesian_shrinkage(
                raw_career_sh, n_total, pl.col("_p_sh_pct"), CAREER_SH_PCT_SHRINKAGE_K
            ).alias("career_shooting_pct_shrunk"),
        ]
    )

    ref_w = 10 if 10 in windows else windows[0]
    base_player_df = with_priors.with_columns(exprs).with_columns(
        pl.col(f"rolling_toi_ev_min_w{ref_w}").alias("rolling_toi_ev_min"),
        pl.col(f"rolling_toi_pp_min_w{ref_w}").alias("rolling_toi_pp_min"),
        pl.col(f"rolling_toi_total_min_w{ref_w}").alias("rolling_toi_total_min"),
        pl.col(f"rolling_shifts_per_game_w{ref_w}").alias("rolling_shifts_per_game"),
        pl.col(f"rolling_sog_per_60_w{ref_w}").alias("rolling_sog_per_60"),
        pl.col(f"rolling_icf_per_60_w{ref_w}").alias("rolling_icf_per_60"),
        pl.col(f"rolling_ixg_per_60_w{ref_w}").alias("rolling_ixg_per_60"),
        pl.col(f"rolling_goals_per_60_w{ref_w}").alias("rolling_goals_per_60"),
        pl.col(f"rolling_assists_per_60_w{ref_w}").alias("rolling_assists_per_60"),
        pl.col(f"rolling_points_per_60_w{ref_w}").alias("rolling_points_per_60"),
        pl.col(f"rolling_rush_rebound_ixg_per_60_w{ref_w}").alias(
            "rolling_rush_rebound_ixg_per_60"
        ),
        pl.col(f"rolling_qoc_index_w{ref_w}").alias("rolling_qoc_index"),
        pl.col(f"rolling_qot_teammate_ixg_per_60_w{ref_w}").alias(
            "rolling_qot_teammate_ixg_per_60"
        ),
        (pl.col("player_prior_games_n") < shrinkage_k).alias("is_low_sample_skater"),
    )

    # Join vacated injury TOI from active roster strength builder
    roster_df = build_roster_strength_features(skater_game_logs, window=ref_w)
    vacated_lookup = roster_df.select(
        [event_col, "team", "missing_regular_toi_f", "missing_regular_toi_d"]
    )
    joined = base_player_df.join(vacated_lookup, on=[event_col, "team"], how="left").with_columns(
        pl.when(pl.col("position_group") == "D")
        .then(pl.col("missing_regular_toi_d").fill_null(0.0))
        .otherwise(pl.col("missing_regular_toi_f").fill_null(0.0))
        .alias("team_injury_vacated_toi_f_or_d")
    )

    # Join opponent defensive suppression features & team rest/B2B schedule context
    if not team_features.is_empty():
        t_ev_col = "game_id" if "game_id" in team_features.columns else "event_id"
        opp_lookup = team_features.select(
            [
                pl.col(t_ev_col).alias(event_col),
                pl.col("team").alias("opponent"),
                pl.col("xga_per_60_5v5").alias("opp_xga_per_60_5v5"),
                pl.col("sa_per_60_5v5").alias("opp_sa_per_60_5v5"),
                pl.col("pk_xga_per_60").alias("opp_pk_xga_per_60"),
                pl.col("penalties_taken_per_60").alias("opp_penalties_taken_per_60"),
            ]
        ).unique(subset=[event_col, "opponent"], keep="last")
        joined = joined.join(opp_lookup, on=[event_col, "opponent"], how="left")

        if {"rest_days", "is_b2b"}.issubset(team_features.columns):
            sched_lookup = team_features.select(
                [
                    pl.col(t_ev_col).alias(event_col),
                    pl.col("team"),
                    pl.col("rest_days").alias("team_rest_days"),
                    pl.col("is_b2b"),
                ]
            ).unique(subset=[event_col, "team"], keep="last")
        else:
            sched_df = compute_schedule_features(team_features)
            sched_lookup = sched_df.select(
                [
                    pl.col(t_ev_col).alias(event_col),
                    pl.col("team"),
                    pl.col("rest_days").alias("team_rest_days"),
                    pl.col("is_b2b"),
                ]
            ).unique(subset=[event_col, "team"], keep="last")
        joined = joined.join(sched_lookup, on=[event_col, "team"], how="left")

    joined = joined.with_columns(
        pl.col("opp_xga_per_60_5v5").fill_null(PRIOR_XGA_PER_60)
        if "opp_xga_per_60_5v5" in joined.columns
        else pl.lit(PRIOR_XGA_PER_60).alias("opp_xga_per_60_5v5"),
        pl.col("opp_sa_per_60_5v5").fill_null(PRIOR_SA_PER_60)
        if "opp_sa_per_60_5v5" in joined.columns
        else pl.lit(PRIOR_SA_PER_60).alias("opp_sa_per_60_5v5"),
        pl.col("opp_pk_xga_per_60").fill_null(PRIOR_PK_XGA_PER_60)
        if "opp_pk_xga_per_60" in joined.columns
        else pl.lit(PRIOR_PK_XGA_PER_60).alias("opp_pk_xga_per_60"),
        pl.col("opp_penalties_taken_per_60").fill_null(PRIOR_DRAW_RATE)
        if "opp_penalties_taken_per_60" in joined.columns
        else pl.lit(PRIOR_DRAW_RATE).alias("opp_penalties_taken_per_60"),
        pl.col("team_rest_days").fill_null(3)
        if "team_rest_days" in joined.columns
        else pl.lit(3, dtype=pl.Int32).alias("team_rest_days"),
        pl.col("is_b2b").fill_null(False)
        if "is_b2b" in joined.columns
        else pl.lit(False).alias("is_b2b"),
    )

    # Join opponent starting goalie shrunk GSAx/60
    if not goalie_features.is_empty():
        g_starters = resolve_team_starting_goalie_features(goalie_features)
        g_ev_col = "game_id" if "game_id" in g_starters.columns else "event_id"
        g_lookup = g_starters.select(
            [
                pl.col(g_ev_col).alias(event_col),
                pl.col("team").alias("opponent"),
                pl.col("gsax_per_60_shrunk").alias("opp_goalie_gsax_per_60_shrunk"),
            ]
        ).unique(subset=[event_col, "opponent"], keep="last")
        joined = joined.join(g_lookup, on=[event_col, "opponent"], how="left")

    joined = joined.with_columns(
        pl.col("opp_goalie_gsax_per_60_shrunk").fill_null(PRIOR_GOALIE_GSAx_PER_60)
        if "opp_goalie_gsax_per_60_shrunk" in joined.columns
        else pl.lit(PRIOR_GOALIE_GSAx_PER_60).alias("opp_goalie_gsax_per_60_shrunk")
    )

    # Join Stage 1 macro game conditioning signals (P(Win) and Expected Team Goals)
    if stage1_predictions is not None and not stage1_predictions.is_empty():
        s1_ev_col = "game_id" if "game_id" in stage1_predictions.columns else "event_id"
        s1_lookup = stage1_predictions.select(
            [
                pl.col(s1_ev_col).alias(event_col),
                pl.col("home_win_prob").alias("_s1_home_prob"),
                pl.col("away_win_prob").alias("_s1_away_prob"),
                pl.col("expected_home_goals").alias("_s1_home_goals"),
                pl.col("expected_away_goals").alias("_s1_away_goals"),
            ]
        ).unique(subset=[event_col], keep="last")
        joined = (
            joined.join(s1_lookup, on=[event_col], how="left")
            .with_columns(
                pl.when(pl.col("is_home"))
                .then(pl.col("_s1_home_prob"))
                .otherwise(pl.col("_s1_away_prob"))
                .fill_null(0.50)
                .alias("stage1_team_win_prob"),
                pl.when(pl.col("is_home"))
                .then(pl.col("_s1_home_goals"))
                .otherwise(pl.col("_s1_away_goals"))
                .fill_null(3.05)
                .alias("stage1_expected_team_goals"),
            )
            .drop(["_s1_home_prob", "_s1_away_prob", "_s1_home_goals", "_s1_away_goals"])
        )
    else:
        joined = joined.with_columns(
            pl.lit(0.50, dtype=pl.Float64).alias("stage1_team_win_prob"),
            pl.lit(3.05, dtype=pl.Float64).alias("stage1_expected_team_goals"),
        )

    drop_temp = [
        "_game_rush_rebound_ixg",
        "rush_ixg_5v5",
        "rebound_ixg_5v5",
        "inner_slot_ixg_5v5",
        "qoc_opponent_toi_share",
        "qot_teammate_ixg_per_60",
        "_p_toi_tot",
        "_p_toi_ev",
        "_p_toi_pp",
        "_p_sog_60",
        "_p_icf_60",
        "_p_ixg_60",
        "_p_g_60",
        "_p_a_60",
        "_p_pts_60",
        "_p_sh_pct",
        "missing_regular_toi_f",
        "missing_regular_toi_d",
    ]
    return joined.drop([c for c in drop_temp if c in joined.columns]).sort(
        ["start_time_utc", "event_seq", event_col, "player_id"]
    )
