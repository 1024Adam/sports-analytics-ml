"""Stage 2 individual EPL outfield and goalkeeper matchup feature builder (`player.py`).

Constructs leak-free pre-match individual rolling usage (`5, 10, 19` matches + EWMA),
start rate, availability weight (`chance_of_playing_next_round`), per-90 rates (`npxg`,
`xa`, `goals`, `assists`, `shots`, `sot`, `key_passes`), set-piece/penalty taker flags,
vacated injury/suspension positional minutes, opponent defensive `xga_per_90` & `ppda`,
opponent starting GK `gsaa_per_90_shrunk`, and Stage 1 conditioning signals
(`stage1_team_win_prob`, `stage1_draw_prob`, `stage1_expected_team_goals`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import (
    pregame_ewma,
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.soccer.constants import normalize_soccer_club
from sports_ml.sports.soccer.features.goalkeeper import (
    PRIOR_GK_GSAA_PER_90,
    resolve_team_starting_gk_features,
)
from sports_ml.sports.soccer.features.roster import (
    OUTFIELD_SHRINKAGE_K,
    PRIOR_MINUTES_DEF,
    PRIOR_MINUTES_FWD,
    PRIOR_MINUTES_MID,
    PRIOR_NPXG_PER_90_DEF,
    PRIOR_NPXG_PER_90_FWD,
    PRIOR_NPXG_PER_90_MID,
    PRIOR_XA_PER_90_DEF,
    PRIOR_XA_PER_90_FWD,
    PRIOR_XA_PER_90_MID,
    PRIOR_XGBUILDUP_PER_90_DEF,
    PRIOR_XGBUILDUP_PER_90_FWD,
    PRIOR_XGBUILDUP_PER_90_MID,
    PRIOR_XGCHAIN_PER_90_DEF,
    PRIOR_XGCHAIN_PER_90_FWD,
    PRIOR_XGCHAIN_PER_90_MID,
    _ensure_outfield_log_columns,
    build_soccer_roster_strength_features,
)
from sports_ml.sports.soccer.features.team import (
    DEFAULT_SOCCER_EWMA_ALPHA,
    PRIOR_PPDA,
    PRIOR_SOT_AGAINST_PER_90,
    PRIOR_XGA_PER_90,
)

DEFAULT_SOCCER_PLAYER_WINDOWS: Final[tuple[int, ...]] = (5, 10, 19)
CAREER_FINISHING_SHRINKAGE_K: Final[float] = 10.0

PRIOR_SHOTS_PER_90_FWD: Final[float] = 2.65
PRIOR_SHOTS_PER_90_MID: Final[float] = 1.35
PRIOR_SHOTS_PER_90_DEF: Final[float] = 0.55

PRIOR_SOT_PER_90_FWD: Final[float] = 1.05
PRIOR_SOT_PER_90_MID: Final[float] = 0.45
PRIOR_SOT_PER_90_DEF: Final[float] = 0.16

PRIOR_KEY_PASSES_PER_90_FWD: Final[float] = 1.10
PRIOR_KEY_PASSES_PER_90_MID: Final[float] = 1.45
PRIOR_KEY_PASSES_PER_90_DEF: Final[float] = 0.65

PRIOR_GOALS_PER_90_FWD: Final[float] = 0.38
PRIOR_GOALS_PER_90_MID: Final[float] = 0.14
PRIOR_GOALS_PER_90_DEF: Final[float] = 0.05

PRIOR_ASSISTS_PER_90_FWD: Final[float] = 0.16
PRIOR_ASSISTS_PER_90_MID: Final[float] = 0.18
PRIOR_ASSISTS_PER_90_DEF: Final[float] = 0.07

PRIOR_CONVERSION_FWD: Final[float] = 0.145
PRIOR_CONVERSION_MID: Final[float] = 0.105
PRIOR_CONVERSION_DEF: Final[float] = 0.085


def build_soccer_stage2_player_features(
    outfield_match_logs: pl.DataFrame,
    team_features: pl.DataFrame,
    gk_features: pl.DataFrame,
    *,
    player_shot_micro_logs: pl.DataFrame | None = None,
    squad_metadata_df: pl.DataFrame | None = None,
    stage1_predictions: pl.DataFrame | None = None,
    excluded_ids_by_match_team: Mapping[tuple[int, str], Sequence[int]] | None = None,
    windows: Sequence[int] = DEFAULT_SOCCER_PLAYER_WINDOWS,
    ewma_alpha: float = DEFAULT_SOCCER_EWMA_ALPHA,
    shrinkage_k: float = OUTFIELD_SHRINKAGE_K,
) -> pl.DataFrame:
    """Construct leak-free pre-match player-matchup features for Stage 2 EPL projections."""
    prepared = _ensure_outfield_log_columns(outfield_match_logs).with_columns(
        pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
        pl.col("opponent").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
    )
    event_col = "match_id" if "match_id" in prepared.columns else "event_id"

    if player_shot_micro_logs is not None and not player_shot_micro_logs.is_empty():
        pm_ev_col = "match_id" if "match_id" in player_shot_micro_logs.columns else "event_id"
        pm_cols = ["open_play_npxg", "set_piece_xg"]
        avail_pm = [c for c in pm_cols if c in player_shot_micro_logs.columns]
        if avail_pm:
            pm_lookup = player_shot_micro_logs.select(
                [
                    pl.col(pm_ev_col).cast(pl.Int64).alias(event_col),
                    pl.col("player_id").cast(pl.Int64),
                    *[pl.col(c).cast(pl.Float64).alias(f"_pm_{c}") for c in avail_pm],
                ]
            ).unique(subset=[event_col, "player_id"], keep="last")
            prepared = prepared.join(pm_lookup, on=[event_col, "player_id"], how="left")
            for c in avail_pm:
                if c in prepared.columns:
                    prepared = prepared.with_columns(
                        pl.coalesce([pl.col(f"_pm_{c}"), pl.col(c).cast(pl.Float64)]).alias(c)
                    )
                else:
                    prepared = prepared.with_columns(pl.col(f"_pm_{c}").alias(c))
            prepared = prepared.drop([f"_pm_{c}" for c in avail_pm])

    for col_name, frac in (("open_play_npxg", 0.78), ("set_piece_xg", 0.22)):
        fallback_expr = frac * pl.col("npxg").cast(pl.Float64)
        if col_name in prepared.columns:
            prepared = prepared.with_columns(
                pl.col(col_name).cast(pl.Float64).fill_null(fallback_expr).alias(col_name)
            )
        else:
            prepared = prepared.with_columns(fallback_expr.alias(col_name))

    sorted_df = sort_events_chronologically(prepared, extra_tiebreakers=["player_id"])

    is_fwd = pl.col("position_group") == "FWD"
    is_def = pl.col("position_group") == "DEF"

    with_priors = sorted_df.with_columns(
        pl.col("is_starter").cast(pl.Float64).alias("_starter_num"),
        pl.when(is_fwd)
        .then(PRIOR_MINUTES_FWD)
        .when(is_def)
        .then(PRIOR_MINUTES_DEF)
        .otherwise(PRIOR_MINUTES_MID)
        .alias("_p_min"),
        pl.when(is_fwd)
        .then(PRIOR_NPXG_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_NPXG_PER_90_DEF)
        .otherwise(PRIOR_NPXG_PER_90_MID)
        .alias("_p_npxg_90"),
        pl.when(is_fwd)
        .then(PRIOR_XA_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_XA_PER_90_DEF)
        .otherwise(PRIOR_XA_PER_90_MID)
        .alias("_p_xa_90"),
        pl.when(is_fwd)
        .then(PRIOR_XGCHAIN_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_XGCHAIN_PER_90_DEF)
        .otherwise(PRIOR_XGCHAIN_PER_90_MID)
        .alias("_p_xgchain_90"),
        pl.when(is_fwd)
        .then(PRIOR_XGBUILDUP_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_XGBUILDUP_PER_90_DEF)
        .otherwise(PRIOR_XGBUILDUP_PER_90_MID)
        .alias("_p_xgbuildup_90"),
        pl.when(is_fwd)
        .then(PRIOR_SHOTS_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_SHOTS_PER_90_DEF)
        .otherwise(PRIOR_SHOTS_PER_90_MID)
        .alias("_p_shots_90"),
        pl.when(is_fwd)
        .then(PRIOR_SOT_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_SOT_PER_90_DEF)
        .otherwise(PRIOR_SOT_PER_90_MID)
        .alias("_p_sot_90"),
        pl.when(is_fwd)
        .then(PRIOR_KEY_PASSES_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_KEY_PASSES_PER_90_DEF)
        .otherwise(PRIOR_KEY_PASSES_PER_90_MID)
        .alias("_p_kp_90"),
        pl.when(is_fwd)
        .then(PRIOR_GOALS_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_GOALS_PER_90_DEF)
        .otherwise(PRIOR_GOALS_PER_90_MID)
        .alias("_p_g_90"),
        pl.when(is_fwd)
        .then(PRIOR_ASSISTS_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_ASSISTS_PER_90_DEF)
        .otherwise(PRIOR_ASSISTS_PER_90_MID)
        .alias("_p_a_90"),
        pl.when(is_fwd)
        .then(PRIOR_CONVERSION_FWD)
        .when(is_def)
        .then(PRIOR_CONVERSION_DEF)
        .otherwise(PRIOR_CONVERSION_MID)
        .alias("_p_conv"),
        pregame_sample_count("minutes", group_by="player_id").alias("player_prior_matches_n"),
    )

    exprs: list[pl.Expr] = []
    for w in windows:
        n_w = pregame_sample_count("minutes", group_by="player_id", window=w)
        r_min = pregame_rolling_mean("minutes", window=w, group_by="player_id")
        r_start_rate = pregame_rolling_mean("_starter_num", window=w, group_by="player_id")
        r_npxg_90 = pregame_exposure_rate(
            "npxg", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_xg_90 = pregame_exposure_rate(
            "xg", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_xa_90 = pregame_exposure_rate(
            "xa", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_xgchain_90 = pregame_exposure_rate(
            "xg_chain", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_xgbuildup_90 = pregame_exposure_rate(
            "xg_buildup", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_shots_90 = pregame_exposure_rate(
            "shots", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_sot_90 = pregame_exposure_rate(
            "sot", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_kp_90 = pregame_exposure_rate(
            "key_passes", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_g_90 = pregame_exposure_rate(
            "goals", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_a_90 = pregame_exposure_rate(
            "assists", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_op_npxg_90 = pregame_exposure_rate(
            "open_play_npxg", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        r_sp_xg_share = pregame_exposure_rate(
            "set_piece_xg", "xg", window=w, group_by="player_id", scale_factor=1.0
        )

        exprs.extend(
            [
                bayesian_shrinkage(r_min, n_w, pl.col("_p_min"), shrinkage_k)
                .clip(0.0, 90.0)
                .alias(f"rolling_minutes_w{w}"),
                bayesian_shrinkage(r_start_rate, n_w, 0.65, shrinkage_k)
                .clip(0.0, 1.0)
                .alias(f"rolling_start_rate_w{w}"),
                bayesian_shrinkage(r_npxg_90, n_w, pl.col("_p_npxg_90"), shrinkage_k).alias(
                    f"rolling_npxg_per_90_w{w}"
                ),
                bayesian_shrinkage(
                    r_op_npxg_90, n_w, 0.78 * pl.col("_p_npxg_90"), shrinkage_k
                ).alias(f"rolling_open_play_npxg_per_90_w{w}"),
                bayesian_shrinkage(r_sp_xg_share, n_w, 0.22, shrinkage_k)
                .clip(0.0, 1.0)
                .alias(f"rolling_set_piece_xg_share_w{w}"),
                bayesian_shrinkage(r_xg_90, n_w, pl.col("_p_npxg_90"), shrinkage_k).alias(
                    f"rolling_xg_per_90_w{w}"
                ),
                bayesian_shrinkage(r_xa_90, n_w, pl.col("_p_xa_90"), shrinkage_k).alias(
                    f"rolling_xa_per_90_w{w}"
                ),
                bayesian_shrinkage(r_xgchain_90, n_w, pl.col("_p_xgchain_90"), shrinkage_k).alias(
                    f"rolling_xg_chain_per_90_w{w}"
                ),
                bayesian_shrinkage(
                    r_xgbuildup_90, n_w, pl.col("_p_xgbuildup_90"), shrinkage_k
                ).alias(f"rolling_xg_buildup_per_90_w{w}"),
                bayesian_shrinkage(r_shots_90, n_w, pl.col("_p_shots_90"), shrinkage_k).alias(
                    f"rolling_shots_per_90_w{w}"
                ),
                bayesian_shrinkage(r_sot_90, n_w, pl.col("_p_sot_90"), shrinkage_k).alias(
                    f"rolling_sot_per_90_w{w}"
                ),
                bayesian_shrinkage(r_kp_90, n_w, pl.col("_p_kp_90"), shrinkage_k).alias(
                    f"rolling_key_passes_per_90_w{w}"
                ),
                bayesian_shrinkage(r_g_90, n_w, pl.col("_p_g_90"), shrinkage_k).alias(
                    f"rolling_goals_per_90_w{w}"
                ),
                bayesian_shrinkage(r_a_90, n_w, pl.col("_p_a_90"), shrinkage_k).alias(
                    f"rolling_assists_per_90_w{w}"
                ),
            ]
        )

    n_total = pl.col("player_prior_matches_n")
    raw_ewma_min = pregame_ewma("minutes", alpha=ewma_alpha, group_by="player_id")
    raw_career_conv = pregame_exposure_rate(
        "goals", "shots", window=200, group_by="player_id", scale_factor=1.0
    )

    exprs.extend(
        [
            bayesian_shrinkage(raw_ewma_min, n_total, pl.col("_p_min"), shrinkage_k)
            .clip(0.0, 90.0)
            .alias("rolling_minutes_ewma"),
            bayesian_shrinkage(
                raw_career_conv, n_total, pl.col("_p_conv"), CAREER_FINISHING_SHRINKAGE_K
            ).alias("career_shot_conversion_shrunk"),
        ]
    )

    ref_w = 10 if 10 in windows else windows[0]
    base_df = with_priors.with_columns(exprs).with_columns(
        pl.col(f"rolling_minutes_w{ref_w}").alias("rolling_minutes"),
        pl.col(f"rolling_start_rate_w{ref_w}").alias("rolling_start_rate"),
        pl.col(f"rolling_npxg_per_90_w{ref_w}").alias("rolling_npxg_per_90"),
        pl.col(f"rolling_open_play_npxg_per_90_w{ref_w}").alias("rolling_open_play_npxg_per_90"),
        pl.col(f"rolling_set_piece_xg_share_w{ref_w}").alias("rolling_set_piece_xg_share"),
        pl.col(f"rolling_xg_per_90_w{ref_w}").alias("rolling_xg_per_90"),
        pl.col(f"rolling_xa_per_90_w{ref_w}").alias("rolling_xa_per_90"),
        pl.col(f"rolling_xg_chain_per_90_w{ref_w}").alias("rolling_xg_chain_per_90"),
        pl.col(f"rolling_xg_buildup_per_90_w{ref_w}").alias("rolling_xg_buildup_per_90"),
        pl.col(f"rolling_shots_per_90_w{ref_w}").alias("rolling_shots_per_90"),
        pl.col(f"rolling_sot_per_90_w{ref_w}").alias("rolling_sot_per_90"),
        pl.col(f"rolling_key_passes_per_90_w{ref_w}").alias("rolling_key_passes_per_90"),
        pl.col(f"rolling_goals_per_90_w{ref_w}").alias("rolling_goals_per_90"),
        pl.col(f"rolling_assists_per_90_w{ref_w}").alias("rolling_assists_per_90"),
        (pl.col("player_prior_matches_n") < shrinkage_k).alias("is_low_sample_player"),
    )

    # Join squad availability & set-piece order (with dynamic redistribution when taker is out)
    if squad_metadata_df is not None and not squad_metadata_df.is_empty():
        sq_norm = squad_metadata_df.with_columns(
            pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
            pl.col("player_id").cast(pl.Int64),
        ).select(
            [
                "player_id",
                "team",
                (pl.col("chance_of_playing_next_round").cast(pl.Float64) / 100.0)
                .clip(0.0, 1.0)
                .alias("availability_weight"),
                pl.col("penalties_order").cast(pl.Int16),
                pl.col("direct_freekicks_order").cast(pl.Int16),
                pl.col("corners_order").cast(pl.Int16),
            ]
        )
        base_df = base_df.join(sq_norm, on=["player_id", "team"], how="left")
    else:
        base_df = base_df.with_columns(
            pl.lit(1.0, dtype=pl.Float64).alias("availability_weight"),
            pl.lit(99, dtype=pl.Int16).alias("penalties_order"),
            pl.lit(99, dtype=pl.Int16).alias("direct_freekicks_order"),
            pl.lit(99, dtype=pl.Int16).alias("corners_order"),
        )

    base_df = base_df.with_columns(
        pl.col("availability_weight").fill_null(1.0),
        pl.col("penalties_order").fill_null(99),
        pl.col("direct_freekicks_order").fill_null(99),
        pl.col("corners_order").fill_null(99),
    )

    if excluded_ids_by_match_team:
        excl_pairs = [
            {event_col: int(ev), "team": normalize_soccer_club(t), "player_id": int(pid)}
            for (ev, t), pids in excluded_ids_by_match_team.items()
            for pid in pids
        ]
        if excl_pairs:
            excl_df = pl.DataFrame(excl_pairs).with_columns(pl.lit(True).alias("_is_excl"))
            base_df = (
                base_df.join(excl_df, on=[event_col, "team", "player_id"], how="left")
                .with_columns(
                    pl.when(pl.col("_is_excl").fill_null(False))
                    .then(0.0)
                    .otherwise(pl.col("availability_weight"))
                    .alias("availability_weight")
                )
                .drop("_is_excl")
            )

    # Identify active primary penalty/set-piece takers among available players
    eff_pen_order = (
        pl.when(pl.col("availability_weight") > 0.0).then(pl.col("penalties_order")).otherwise(999)
    )
    eff_sp_order = (
        pl.when(pl.col("availability_weight") > 0.0)
        .then(pl.min_horizontal("corners_order", "direct_freekicks_order"))
        .otherwise(999)
    )
    min_team_pen = eff_pen_order.min().over([event_col, "team"])
    min_team_sp = eff_sp_order.min().over([event_col, "team"])

    base_df = base_df.with_columns(
        ((eff_pen_order == min_team_pen) & (eff_pen_order < 99)).alias("is_penalty_taker"),
        ((eff_sp_order == min_team_sp) & (eff_sp_order < 99)).alias("is_set_piece_taker"),
    )

    # Join vacated injury/suspension positional minutes from roster strength builder
    roster_df = build_soccer_roster_strength_features(
        outfield_match_logs,
        window=ref_w,
        excluded_ids_by_match_team=excluded_ids_by_match_team,
    )
    r_ev_col = "match_id" if "match_id" in roster_df.columns else "event_id"
    vacated_lookup = roster_df.select(
        [
            pl.col(r_ev_col).alias(event_col),
            "team",
            "missing_regular_minutes_fwd",
            "missing_regular_minutes_mid",
            "missing_regular_minutes_def",
        ]
    )
    joined = base_df.join(vacated_lookup, on=[event_col, "team"], how="left").with_columns(
        pl.when(pl.col("position_group") == "FWD")
        .then(pl.col("missing_regular_minutes_fwd").fill_null(0.0))
        .when(pl.col("position_group") == "DEF")
        .then(pl.col("missing_regular_minutes_def").fill_null(0.0))
        .otherwise(pl.col("missing_regular_minutes_mid").fill_null(0.0))
        .alias("team_injury_vacated_minutes_pos")
    )

    # Join opponent defensive suppression features & team rest/congestion
    if not team_features.is_empty():
        t_ev_col = "match_id" if "match_id" in team_features.columns else "event_id"
        opp_lookup = team_features.select(
            [
                pl.col(t_ev_col).alias(event_col),
                pl.col("team").alias("opponent"),
                pl.col("xga_per_90").alias("opp_xga_per_90"),
                pl.col("sot_against_per_90").alias("opp_sot_against_per_90"),
                pl.col("ppda_rolling").alias("opp_ppda"),
            ]
        ).unique(subset=[event_col, "opponent"], keep="last")
        joined = joined.join(opp_lookup, on=[event_col, "opponent"], how="left")

        if {"rest_days", "is_midweek_turnaround"}.issubset(team_features.columns):
            sched_lookup = team_features.select(
                [
                    pl.col(t_ev_col).alias(event_col),
                    pl.col("team"),
                    pl.col("rest_days").alias("team_rest_days"),
                    pl.col("is_midweek_turnaround"),
                ]
            ).unique(subset=[event_col, "team"], keep="last")
            joined = joined.join(sched_lookup, on=[event_col, "team"], how="left")

    joined = joined.with_columns(
        pl.col("opp_xga_per_90").fill_null(PRIOR_XGA_PER_90)
        if "opp_xga_per_90" in joined.columns
        else pl.lit(PRIOR_XGA_PER_90).alias("opp_xga_per_90"),
        pl.col("opp_sot_against_per_90").fill_null(PRIOR_SOT_AGAINST_PER_90)
        if "opp_sot_against_per_90" in joined.columns
        else pl.lit(PRIOR_SOT_AGAINST_PER_90).alias("opp_sot_against_per_90"),
        pl.col("opp_ppda").fill_null(PRIOR_PPDA)
        if "opp_ppda" in joined.columns
        else pl.lit(PRIOR_PPDA).alias("opp_ppda"),
        pl.col("team_rest_days").fill_null(6)
        if "team_rest_days" in joined.columns
        else pl.lit(6, dtype=pl.Int32).alias("team_rest_days"),
        pl.col("is_midweek_turnaround").fill_null(False)
        if "is_midweek_turnaround" in joined.columns
        else pl.lit(False).alias("is_midweek_turnaround"),
    )

    # Join opponent starting GK shrunk GSAA/90
    if not gk_features.is_empty():
        g_starters = resolve_team_starting_gk_features(gk_features)
        g_ev_col = "match_id" if "match_id" in g_starters.columns else "event_id"
        g_lookup = g_starters.select(
            [
                pl.col(g_ev_col).alias(event_col),
                pl.col("team").alias("opponent"),
                pl.col("gsaa_per_90_shrunk").alias("opp_gk_gsaa_per_90_shrunk"),
            ]
        ).unique(subset=[event_col, "opponent"], keep="last")
        joined = joined.join(g_lookup, on=[event_col, "opponent"], how="left")

    joined = joined.with_columns(
        pl.col("opp_gk_gsaa_per_90_shrunk").fill_null(PRIOR_GK_GSAA_PER_90)
        if "opp_gk_gsaa_per_90_shrunk" in joined.columns
        else pl.lit(PRIOR_GK_GSAA_PER_90).alias("opp_gk_gsaa_per_90_shrunk")
    )

    # Join Stage 1 macro match conditioning signals (P(Win), P(Draw), Expected Team Goals)
    if stage1_predictions is not None and not stage1_predictions.is_empty():
        s1_ev_col = "match_id" if "match_id" in stage1_predictions.columns else "event_id"
        s1_lookup = stage1_predictions.select(
            [
                pl.col(s1_ev_col).alias(event_col),
                pl.col("home_win_prob").alias("_s1_home_prob"),
                pl.col("draw_prob").alias("_s1_draw_prob"),
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
                .fill_null(0.38)
                .alias("stage1_team_win_prob"),
                pl.col("_s1_draw_prob").fill_null(0.24).alias("stage1_draw_prob"),
                pl.when(pl.col("is_home"))
                .then(pl.col("_s1_home_goals"))
                .otherwise(pl.col("_s1_away_goals"))
                .fill_null(1.45)
                .alias("stage1_expected_team_goals"),
            )
            .drop(
                [
                    "_s1_home_prob",
                    "_s1_draw_prob",
                    "_s1_away_prob",
                    "_s1_home_goals",
                    "_s1_away_goals",
                ]
            )
        )
    else:
        joined = joined.with_columns(
            pl.lit(0.38, dtype=pl.Float64).alias("stage1_team_win_prob"),
            pl.lit(0.24, dtype=pl.Float64).alias("stage1_draw_prob"),
            pl.lit(1.45, dtype=pl.Float64).alias("stage1_expected_team_goals"),
        )

    drop_internal = [
        "_starter_num",
        "open_play_npxg",
        "set_piece_xg",
        "_p_min",
        "_p_npxg_90",
        "_p_xa_90",
        "_p_xgchain_90",
        "_p_xgbuildup_90",
        "_p_shots_90",
        "_p_sot_90",
        "_p_kp_90",
        "_p_g_90",
        "_p_a_90",
        "_p_conv",
        "missing_regular_minutes_fwd",
        "missing_regular_minutes_mid",
        "missing_regular_minutes_def",
    ]
    cleaned = joined.drop([c for c in drop_internal if c in joined.columns])
    return sort_events_chronologically(cleaned, extra_tiebreakers=["player_id"])
