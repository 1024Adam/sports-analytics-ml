"""TOI-weighted active skater aggregation for Stage 1 game features (`features/roster.py`).

Weights active skater rolling shrunk `ixG/60`, `OnIce-xGD/60`, and `GameScore/60`
by projected TOI share across Forwards (`F`) and Defensemen (`D`), and computes
`missing_regular_toi` from recent regulars absent from tonight's active lineup.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import (
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_sample_count,
    sort_events_chronologically,
)

SKATER_SHRINKAGE_K: Final[float] = 12.0

# Position-group priors (Forwards vs. Defensemen)
PRIOR_TOI_TOTAL_F: Final[float] = 16.0
PRIOR_TOI_TOTAL_D: Final[float] = 20.0
PRIOR_TOI_EV_F: Final[float] = 14.0
PRIOR_TOI_EV_D: Final[float] = 18.0
PRIOR_TOI_PP_F: Final[float] = 1.8
PRIOR_TOI_PP_D: Final[float] = 1.5
PRIOR_IXG_PER_60_F: Final[float] = 0.75
PRIOR_IXG_PER_60_D: Final[float] = 0.30
PRIOR_ONICE_XGD_PER_60: Final[float] = 0.0
PRIOR_GAMESCORE_PER_60_F: Final[float] = 1.80
PRIOR_GAMESCORE_PER_60_D: Final[float] = 1.50


def _ensure_skater_log_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Populate optional skater game-log columns so minimal fixtures work cleanly."""
    working = df
    if "position_group" not in working.columns:
        working = working.with_columns(pl.lit("F", dtype=pl.Utf8).alias("position_group"))
    if "toi_ev_min" not in working.columns and "toi_total_min" in working.columns:
        working = working.with_columns(
            (pl.col("toi_total_min").cast(pl.Float64) * 0.85).alias("toi_ev_min")
        )
    if "toi_pp_min" not in working.columns and "toi_total_min" in working.columns:
        working = working.with_columns(
            (pl.col("toi_total_min").cast(pl.Float64) * 0.10).alias("toi_pp_min")
        )
    if "toi_pk_min" not in working.columns and "toi_total_min" in working.columns:
        working = working.with_columns(
            (pl.col("toi_total_min").cast(pl.Float64) * 0.05).alias("toi_pk_min")
        )

    defaults: list[tuple[str, float | int, pl.DataType]] = [
        ("goals", 0, pl.Int16()),
        ("assists", 0, pl.Int16()),
        ("points", 0, pl.Int16()),
        ("sog", 2, pl.Int16()),
        ("ixg_all", 0.18, pl.Float64()),
        ("ixg_5v5", 0.14, pl.Float64()),
        ("icf_all", 3.5, pl.Float64()),
        ("on_ice_xgf_5v5", 0.65, pl.Float64()),
        ("on_ice_xga_5v5", 0.65, pl.Float64()),
        ("game_score", 0.45, pl.Float64()),
    ]
    missing_exprs = [
        pl.lit(val, dtype=dtype).alias(col)
        for col, val, dtype in defaults
        if col not in working.columns
    ]
    return working.with_columns(missing_exprs) if missing_exprs else working


def build_skater_rolling_features(
    skater_game_logs: pl.DataFrame,
    *,
    window: int = 10,
    shrinkage_k: float = SKATER_SHRINKAGE_K,
) -> pl.DataFrame:
    """Compute leak-free pre-game rolling TOI and per-60 rates for each skater appearance."""
    prepared = _ensure_skater_log_columns(skater_game_logs)
    sorted_df = sort_events_chronologically(prepared, extra_tiebreakers=["player_id"])

    with_xgd = sorted_df.with_columns(
        (
            pl.col("on_ice_xgf_5v5").cast(pl.Float64) - pl.col("on_ice_xga_5v5").cast(pl.Float64)
        ).alias("_on_ice_xgd_5v5"),
        pl.when(pl.col("position_group") == "D")
        .then(PRIOR_TOI_TOTAL_D)
        .otherwise(PRIOR_TOI_TOTAL_F)
        .alias("_prior_toi_total"),
        pl.when(pl.col("position_group") == "D")
        .then(PRIOR_TOI_EV_D)
        .otherwise(PRIOR_TOI_EV_F)
        .alias("_prior_toi_ev"),
        pl.when(pl.col("position_group") == "D")
        .then(PRIOR_TOI_PP_D)
        .otherwise(PRIOR_TOI_PP_F)
        .alias("_prior_toi_pp"),
        pl.when(pl.col("position_group") == "D")
        .then(PRIOR_IXG_PER_60_D)
        .otherwise(PRIOR_IXG_PER_60_F)
        .alias("_prior_ixg_60"),
        pl.when(pl.col("position_group") == "D")
        .then(PRIOR_GAMESCORE_PER_60_D)
        .otherwise(PRIOR_GAMESCORE_PER_60_F)
        .alias("_prior_gs_60"),
        pregame_sample_count("toi_total_min", group_by="player_id").alias("skater_prior_games_n"),
    )

    n_w = pregame_sample_count("toi_total_min", group_by="player_id", window=window)
    raw_toi_total = pregame_rolling_mean("toi_total_min", window=window, group_by="player_id")
    raw_toi_ev = pregame_rolling_mean("toi_ev_min", window=window, group_by="player_id")
    raw_toi_pp = pregame_rolling_mean("toi_pp_min", window=window, group_by="player_id")

    raw_ixg_60 = pregame_exposure_rate(
        "ixg_all", "toi_total_min", window=window, group_by="player_id", scale_factor=60.0
    )
    raw_xgd_60 = pregame_exposure_rate(
        "_on_ice_xgd_5v5", "toi_ev_min", window=window, group_by="player_id", scale_factor=60.0
    )
    raw_gs_60 = pregame_exposure_rate(
        "game_score", "toi_total_min", window=window, group_by="player_id", scale_factor=60.0
    )

    result = with_xgd.with_columns(
        bayesian_shrinkage(raw_toi_total, n_w, pl.col("_prior_toi_total"), shrinkage_k).alias(
            "proj_toi_total_min"
        ),
        bayesian_shrinkage(raw_toi_ev, n_w, pl.col("_prior_toi_ev"), shrinkage_k).alias(
            "proj_toi_ev_min"
        ),
        bayesian_shrinkage(raw_toi_pp, n_w, pl.col("_prior_toi_pp"), shrinkage_k).alias(
            "proj_toi_pp_min"
        ),
        bayesian_shrinkage(raw_ixg_60, n_w, pl.col("_prior_ixg_60"), shrinkage_k).alias(
            "shrunk_ixg_per_60"
        ),
        bayesian_shrinkage(raw_xgd_60, n_w, PRIOR_ONICE_XGD_PER_60, shrinkage_k).alias(
            "shrunk_on_ice_xgd_per_60"
        ),
        bayesian_shrinkage(raw_gs_60, n_w, pl.col("_prior_gs_60"), shrinkage_k).alias(
            "shrunk_game_score_per_60"
        ),
    )

    return result.drop(
        [
            "_on_ice_xgd_5v5",
            "_prior_toi_total",
            "_prior_toi_ev",
            "_prior_toi_pp",
            "_prior_ixg_60",
            "_prior_gs_60",
        ]
    )


def _compute_missing_regular_toi(skater_features_df: pl.DataFrame) -> pl.DataFrame:
    """Compute leak-free `missing_regular_toi` (total, F, and D) per `(game_id, team)`."""
    event_col = "game_id" if "game_id" in skater_features_df.columns else "event_id"
    if skater_features_df.is_empty():
        return pl.DataFrame(
            schema={
                event_col: pl.Int64(),
                "team": pl.Utf8(),
                "missing_regular_toi": pl.Float64(),
                "missing_regular_toi_f": pl.Float64(),
                "missing_regular_toi_d": pl.Float64(),
            }
        )

    # Group chronologically by (start_time_utc, event_seq, event_id, team)
    grouped = (
        skater_features_df.group_by(
            ["season", "start_time_utc", "event_seq", event_col, "team"],
            maintain_order=True,
        )
        .agg(
            pl.col("player_id"),
            pl.col("position_group"),
            pl.col("proj_toi_total_min"),
            pl.col("toi_total_min"),
        )
        .sort(["start_time_utc", "event_seq", event_col, "team"])
    )

    # Track per-(team, season): player_id -> (pos_group, post_game_rolling_toi_estimate)
    known_regulars: dict[tuple[str, int], dict[int, tuple[str, float]]] = {}
    out_rows: list[dict[str, int | str | float]] = []

    for row in grouped.iter_rows(named=True):
        season = int(row["season"])
        team = str(row["team"])
        ev_id = int(row[event_col])
        active_pids: list[int] = [int(x) for x in row["player_id"]]
        pos_groups: list[str] = [str(x) for x in row["position_group"]]
        pre_tois: list[float] = [float(x) for x in row["proj_toi_total_min"]]
        actual_tois: list[float] = [float(x) for x in row["toi_total_min"]]

        active_set = set(active_pids)
        team_roster_state = known_regulars.setdefault((team, season), {})

        # 1. Compute missing regular TOI strictly before updating state with tonight's game
        missing_f = 0.0
        missing_d = 0.0
        for pid, (pos_grp, est_toi) in team_roster_state.items():
            if pid not in active_set and est_toi >= 10.0:
                if pos_grp == "D":
                    missing_d += est_toi
                else:
                    missing_f += est_toi

        out_rows.append(
            {
                event_col: ev_id,
                "team": team,
                "missing_regular_toi": missing_f + missing_d,
                "missing_regular_toi_f": missing_f,
                "missing_regular_toi_d": missing_d,
            }
        )

        # 2. Update known regulars state for future games (> t)
        for pid, pos_grp, pre_t, act_t in zip(
            active_pids, pos_groups, pre_tois, actual_tois, strict=True
        ):
            # Blend pre-game projected TOI with completed game TOI for future games' baseline
            updated_est = (
                (SKATER_SHRINKAGE_K * pre_t + act_t) / (SKATER_SHRINKAGE_K + 1.0)
                if act_t > 0.0
                else pre_t
            )
            team_roster_state[pid] = (pos_grp, updated_est)

    return pl.DataFrame(out_rows)


def build_roster_strength_features(
    skater_game_logs: pl.DataFrame,
    *,
    window: int = 10,
    shrinkage_k: float = SKATER_SHRINKAGE_K,
) -> pl.DataFrame:
    """Aggregate pre-game active skater rolling metrics weighted by normalized TOI share.

    Normalized separately across Forwards (`sum(w_F) == 1.0`) and Defensemen (`sum(w_D) == 1.0`)
    for each `(game_id, team)`.
    """
    skater_feat = build_skater_rolling_features(
        skater_game_logs, window=window, shrinkage_k=shrinkage_k
    )
    event_col = "game_id" if "game_id" in skater_feat.columns else "event_id"

    if skater_feat.is_empty():
        return pl.DataFrame(
            schema={
                event_col: pl.Int64(),
                "team": pl.Utf8(),
                "forward_weighted_ixg_per_60": pl.Float64(),
                "forward_weighted_xgd_per_60": pl.Float64(),
                "forward_weighted_gs_per_60": pl.Float64(),
                "defense_weighted_ixg_per_60": pl.Float64(),
                "defense_weighted_xgd_per_60": pl.Float64(),
                "defense_weighted_gs_per_60": pl.Float64(),
                "forward_toi_weight_sum": pl.Float64(),
                "defense_toi_weight_sum": pl.Float64(),
                "missing_regular_toi": pl.Float64(),
                "missing_regular_toi_f": pl.Float64(),
                "missing_regular_toi_d": pl.Float64(),
            }
        )

    # Normalize TOI weights within (game_id, team, position_group)
    pos_group_sum = pl.col("proj_toi_total_min").sum().over([event_col, "team", "position_group"])
    weighted_skaters = skater_feat.with_columns(
        (
            pl.col("proj_toi_total_min")
            / pl.when(pos_group_sum > 0.0).then(pos_group_sum).otherwise(1.0)
        ).alias("pos_toi_weight")
    )

    f_mask = pl.col("position_group") == "F"
    d_mask = pl.col("position_group") == "D"

    agg_df = (
        weighted_skaters.group_by(
            ["season", "start_time_utc", "event_seq", event_col, "team"],
            maintain_order=True,
        )
        .agg(
            (pl.col("pos_toi_weight") * pl.col("shrunk_ixg_per_60"))
            .filter(f_mask)
            .sum()
            .alias("forward_weighted_ixg_per_60"),
            (pl.col("pos_toi_weight") * pl.col("shrunk_on_ice_xgd_per_60"))
            .filter(f_mask)
            .sum()
            .alias("forward_weighted_xgd_per_60"),
            (pl.col("pos_toi_weight") * pl.col("shrunk_game_score_per_60"))
            .filter(f_mask)
            .sum()
            .alias("forward_weighted_gs_per_60"),
            (pl.col("pos_toi_weight") * pl.col("shrunk_ixg_per_60"))
            .filter(d_mask)
            .sum()
            .alias("defense_weighted_ixg_per_60"),
            (pl.col("pos_toi_weight") * pl.col("shrunk_on_ice_xgd_per_60"))
            .filter(d_mask)
            .sum()
            .alias("defense_weighted_xgd_per_60"),
            (pl.col("pos_toi_weight") * pl.col("shrunk_game_score_per_60"))
            .filter(d_mask)
            .sum()
            .alias("defense_weighted_gs_per_60"),
            pl.col("pos_toi_weight").filter(f_mask).sum().alias("forward_toi_weight_sum"),
            pl.col("pos_toi_weight").filter(d_mask).sum().alias("defense_toi_weight_sum"),
        )
        .sort(["start_time_utc", "event_seq", event_col, "team"])
    )

    missing_df = _compute_missing_regular_toi(skater_feat)
    return agg_df.join(missing_df, on=[event_col, "team"], how="left")
