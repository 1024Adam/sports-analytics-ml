"""Minutes-weighted active squad aggregation for Stage 1 EPL match features (`roster.py`).

Weights available outfield players' rolling Bayesian-shrunk per-90 `npxG`, `xA`,
`xGChain`, and `xGBuildup` by projected minutes share across positional units
(`FWD`, `MID`, `DEF`), and computes `missing_regular_minutes` from injured,
suspended, or absent regulars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import (
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.soccer.constants import normalize_soccer_club

OUTFIELD_SHRINKAGE_K: Final[float] = 9.0

# Positional priors for EPL outfield units (FWD, MID, DEF)
PRIOR_MINUTES_FWD: Final[float] = 68.0
PRIOR_MINUTES_MID: Final[float] = 74.0
PRIOR_MINUTES_DEF: Final[float] = 80.0

PRIOR_NPXG_PER_90_FWD: Final[float] = 0.36
PRIOR_NPXG_PER_90_MID: Final[float] = 0.14
PRIOR_NPXG_PER_90_DEF: Final[float] = 0.05

PRIOR_XA_PER_90_FWD: Final[float] = 0.16
PRIOR_XA_PER_90_MID: Final[float] = 0.18
PRIOR_XA_PER_90_DEF: Final[float] = 0.07

PRIOR_XGCHAIN_PER_90_FWD: Final[float] = 0.58
PRIOR_XGCHAIN_PER_90_MID: Final[float] = 0.52
PRIOR_XGCHAIN_PER_90_DEF: Final[float] = 0.34

PRIOR_XGBUILDUP_PER_90_FWD: Final[float] = 0.16
PRIOR_XGBUILDUP_PER_90_MID: Final[float] = 0.32
PRIOR_XGBUILDUP_PER_90_DEF: Final[float] = 0.28


def _ensure_outfield_log_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Populate optional outfield match-log columns so minimal fixtures work cleanly."""
    working = df
    if "match_id" in working.columns and "event_id" not in working.columns:
        working = working.with_columns(pl.col("match_id").cast(pl.Int64).alias("event_id"))
    if "position_group" not in working.columns:
        working = working.with_columns(pl.lit("MID", dtype=pl.Utf8).alias("position_group"))
    if "is_starter" not in working.columns and "minutes" in working.columns:
        working = working.with_columns((pl.col("minutes") >= 60.0).alias("is_starter"))

    defaults: list[tuple[str, float | int, pl.DataType]] = [
        ("goals", 0, pl.Int16()),
        ("assists", 0, pl.Int16()),
        ("shots", 1, pl.Int16()),
        ("sot", 0, pl.Int16()),
        ("key_passes", 1, pl.Int16()),
        ("xg", 0.12, pl.Float64()),
        ("npxg", 0.12, pl.Float64()),
        ("xa", 0.10, pl.Float64()),
        ("xg_chain", 0.38, pl.Float64()),
        ("xg_buildup", 0.22, pl.Float64()),
        ("yellow_cards", 0, pl.Int16()),
        ("red_cards", 0, pl.Int16()),
    ]
    missing_exprs = [
        pl.lit(val, dtype=dtype).alias(col)
        for col, val, dtype in defaults
        if col not in working.columns
    ]
    return working.with_columns(missing_exprs) if missing_exprs else working


def build_outfield_rolling_features(
    outfield_match_logs: pl.DataFrame,
    *,
    window: int = 10,
    shrinkage_k: float = OUTFIELD_SHRINKAGE_K,
) -> pl.DataFrame:
    """Compute leak-free pre-match rolling minutes and per-90 rates for each outfield appearance."""
    prepared = _ensure_outfield_log_columns(outfield_match_logs)
    sorted_df = sort_events_chronologically(prepared, extra_tiebreakers=["player_id"])

    is_fwd = pl.col("position_group") == "FWD"
    is_def = pl.col("position_group") == "DEF"

    with_priors = sorted_df.with_columns(
        pl.when(is_fwd)
        .then(PRIOR_MINUTES_FWD)
        .when(is_def)
        .then(PRIOR_MINUTES_DEF)
        .otherwise(PRIOR_MINUTES_MID)
        .alias("_prior_minutes"),
        pl.when(is_fwd)
        .then(PRIOR_NPXG_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_NPXG_PER_90_DEF)
        .otherwise(PRIOR_NPXG_PER_90_MID)
        .alias("_prior_npxg_90"),
        pl.when(is_fwd)
        .then(PRIOR_XA_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_XA_PER_90_DEF)
        .otherwise(PRIOR_XA_PER_90_MID)
        .alias("_prior_xa_90"),
        pl.when(is_fwd)
        .then(PRIOR_XGCHAIN_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_XGCHAIN_PER_90_DEF)
        .otherwise(PRIOR_XGCHAIN_PER_90_MID)
        .alias("_prior_xgchain_90"),
        pl.when(is_fwd)
        .then(PRIOR_XGBUILDUP_PER_90_FWD)
        .when(is_def)
        .then(PRIOR_XGBUILDUP_PER_90_DEF)
        .otherwise(PRIOR_XGBUILDUP_PER_90_MID)
        .alias("_prior_xgbuildup_90"),
        pregame_sample_count("minutes", group_by="player_id").alias("outfield_prior_matches_n"),
    )

    n_w = pregame_sample_count("minutes", group_by="player_id", window=window)
    raw_min = pregame_rolling_mean("minutes", window=window, group_by="player_id")
    raw_npxg_90 = pregame_exposure_rate(
        "npxg", "minutes", window=window, group_by="player_id", scale_factor=90.0
    )
    raw_xa_90 = pregame_exposure_rate(
        "xa", "minutes", window=window, group_by="player_id", scale_factor=90.0
    )
    raw_xgchain_90 = pregame_exposure_rate(
        "xg_chain", "minutes", window=window, group_by="player_id", scale_factor=90.0
    )
    raw_xgbuildup_90 = pregame_exposure_rate(
        "xg_buildup", "minutes", window=window, group_by="player_id", scale_factor=90.0
    )

    result = with_priors.with_columns(
        bayesian_shrinkage(raw_min, n_w, pl.col("_prior_minutes"), shrinkage_k)
        .clip(0.0, 90.0)
        .alias("proj_minutes"),
        bayesian_shrinkage(raw_npxg_90, n_w, pl.col("_prior_npxg_90"), shrinkage_k).alias(
            "shrunk_npxg_per_90"
        ),
        bayesian_shrinkage(raw_xa_90, n_w, pl.col("_prior_xa_90"), shrinkage_k).alias(
            "shrunk_xa_per_90"
        ),
        bayesian_shrinkage(raw_xgchain_90, n_w, pl.col("_prior_xgchain_90"), shrinkage_k).alias(
            "shrunk_xg_chain_per_90"
        ),
        bayesian_shrinkage(raw_xgbuildup_90, n_w, pl.col("_prior_xgbuildup_90"), shrinkage_k).alias(
            "shrunk_xg_buildup_per_90"
        ),
    )

    return result.drop(
        [
            "_prior_minutes",
            "_prior_npxg_90",
            "_prior_xa_90",
            "_prior_xgchain_90",
            "_prior_xgbuildup_90",
        ]
    )


def _compute_missing_regular_minutes(
    outfield_features_df: pl.DataFrame,
    *,
    excluded_ids_by_match_team: Mapping[tuple[int, str], Sequence[int]] | None = None,
) -> pl.DataFrame:
    """Compute leak-free `missing_regular_minutes` (total, FWD, MID, DEF) per `(match_id, team)`."""
    event_col = "match_id" if "match_id" in outfield_features_df.columns else "event_id"
    if outfield_features_df.is_empty():
        return pl.DataFrame(
            schema={
                event_col: pl.Int64(),
                "team": pl.Utf8(),
                "missing_regular_minutes": pl.Float64(),
                "missing_regular_minutes_fwd": pl.Float64(),
                "missing_regular_minutes_mid": pl.Float64(),
                "missing_regular_minutes_def": pl.Float64(),
            }
        )

    grouped = (
        outfield_features_df.group_by(
            ["season", "start_time_utc", "event_seq", event_col, "team"],
            maintain_order=True,
        )
        .agg(
            pl.col("player_id"),
            pl.col("position_group"),
            pl.col("proj_minutes"),
            pl.col("minutes"),
        )
        .sort(["start_time_utc", "event_seq", event_col, "team"])
    )

    known_regulars: dict[tuple[str, int], dict[int, tuple[str, float]]] = {}
    out_rows: list[dict[str, int | str | float]] = []

    for row in grouped.iter_rows(named=True):
        season = int(row["season"])
        team = normalize_soccer_club(str(row["team"]))
        ev_id = int(row[event_col])
        active_pids: list[int] = [int(x) for x in row["player_id"]]
        pos_groups: list[str] = [str(x) for x in row["position_group"]]
        pre_mins: list[float] = [float(x) for x in row["proj_minutes"]]
        actual_mins: list[float] = [float(x) for x in row["minutes"]]

        explicit_excluded = set(
            int(pid) for pid in (excluded_ids_by_match_team or {}).get((ev_id, team), ())
        )
        active_set = set(active_pids) - explicit_excluded
        team_roster_state = known_regulars.setdefault((team, season), {})

        # Also include any player present in `active_pids` but explicitly marked excluded/injured
        pre_lookup = {
            pid: (pos_grp, pre_m)
            for pid, pos_grp, pre_m in zip(active_pids, pos_groups, pre_mins, strict=True)
        }

        missing_fwd = 0.0
        missing_mid = 0.0
        missing_def = 0.0
        candidate_pids = set(team_roster_state.keys()) | explicit_excluded
        for pid in candidate_pids:
            if pid not in active_set:
                pos_grp, est_min = team_roster_state.get(pid) or pre_lookup.get(pid, ("MID", 0.0))
                if est_min >= 45.0:
                    if pos_grp == "FWD":
                        missing_fwd += est_min
                    elif pos_grp == "DEF":
                        missing_def += est_min
                    else:
                        missing_mid += est_min

        out_rows.append(
            {
                event_col: ev_id,
                "team": team,
                "missing_regular_minutes": missing_fwd + missing_mid + missing_def,
                "missing_regular_minutes_fwd": missing_fwd,
                "missing_regular_minutes_mid": missing_mid,
                "missing_regular_minutes_def": missing_def,
            }
        )

        # Update known regulars strictly after computing pre-match missing minutes
        for pid, pos_grp, pre_m, act_m in zip(
            active_pids, pos_groups, pre_mins, actual_mins, strict=True
        ):
            if pid in explicit_excluded:
                continue
            updated_est = (
                (OUTFIELD_SHRINKAGE_K * pre_m + act_m) / (OUTFIELD_SHRINKAGE_K + 1.0)
                if act_m > 0.0
                else pre_m
            )
            team_roster_state[pid] = (pos_grp, updated_est)

    return pl.DataFrame(out_rows)


def build_soccer_roster_strength_features(
    outfield_match_logs: pl.DataFrame,
    *,
    window: int = 10,
    shrinkage_k: float = OUTFIELD_SHRINKAGE_K,
    excluded_ids_by_match_team: Mapping[tuple[int, str], Sequence[int]] | None = None,
) -> pl.DataFrame:
    """Aggregate pre-match available outfield player rolling metrics weighted by minutes share.

    Normalized separately across Forwards (`FWD`), Midfielders (`MID`), and Defenders (`DEF`)
    for each `(match_id, team)`.
    """
    outfield_feat = build_outfield_rolling_features(
        outfield_match_logs, window=window, shrinkage_k=shrinkage_k
    )
    event_col = "match_id" if "match_id" in outfield_feat.columns else "event_id"

    if outfield_feat.is_empty():
        return pl.DataFrame(
            schema={
                event_col: pl.Int64(),
                "team": pl.Utf8(),
                "fwd_weighted_npxg_per_90": pl.Float64(),
                "fwd_weighted_xa_per_90": pl.Float64(),
                "mid_weighted_npxg_per_90": pl.Float64(),
                "mid_weighted_xa_per_90": pl.Float64(),
                "mid_weighted_xg_chain_per_90": pl.Float64(),
                "def_weighted_xg_buildup_per_90": pl.Float64(),
                "fwd_minutes_weight_sum": pl.Float64(),
                "mid_minutes_weight_sum": pl.Float64(),
                "def_minutes_weight_sum": pl.Float64(),
                "missing_regular_minutes": pl.Float64(),
                "missing_regular_minutes_fwd": pl.Float64(),
                "missing_regular_minutes_mid": pl.Float64(),
                "missing_regular_minutes_def": pl.Float64(),
            }
        )

    missing_df = _compute_missing_regular_minutes(
        outfield_feat, excluded_ids_by_match_team=excluded_ids_by_match_team
    )

    working = outfield_feat
    if excluded_ids_by_match_team:
        # Zero out projected minutes weight for players excluded by pre-match injury/suspension
        excluded_pairs = [
            {event_col: int(ev), "team": normalize_soccer_club(t), "player_id": int(pid)}
            for (ev, t), pids in excluded_ids_by_match_team.items()
            for pid in pids
        ]
        if excluded_pairs:
            excl_df = pl.DataFrame(excluded_pairs).with_columns(pl.lit(True).alias("_is_excl"))
            working = (
                working.join(excl_df, on=[event_col, "team", "player_id"], how="left")
                .with_columns(
                    pl.when(pl.col("_is_excl").fill_null(False))
                    .then(0.0)
                    .otherwise(pl.col("proj_minutes"))
                    .alias("proj_minutes")
                )
                .drop("_is_excl")
            )

    pos_sum = pl.col("proj_minutes").sum().over([event_col, "team", "position_group"])
    weighted = working.with_columns(
        (pl.col("proj_minutes") / pl.when(pos_sum > 0.0).then(pos_sum).otherwise(1.0)).alias(
            "_pos_min_weight"
        )
    )

    is_fwd = pl.col("position_group") == "FWD"
    is_mid = pl.col("position_group") == "MID"
    is_def = pl.col("position_group") == "DEF"

    agg_df = (
        weighted.group_by([event_col, "team"], maintain_order=True)
        .agg(
            (pl.col("_pos_min_weight") * pl.col("shrunk_npxg_per_90"))
            .filter(is_fwd)
            .sum()
            .alias("_fwd_npxg"),
            (pl.col("_pos_min_weight") * pl.col("shrunk_xa_per_90"))
            .filter(is_fwd)
            .sum()
            .alias("_fwd_xa"),
            (pl.col("_pos_min_weight") * pl.col("shrunk_npxg_per_90"))
            .filter(is_mid)
            .sum()
            .alias("_mid_npxg"),
            (pl.col("_pos_min_weight") * pl.col("shrunk_xa_per_90"))
            .filter(is_mid)
            .sum()
            .alias("_mid_xa"),
            (pl.col("_pos_min_weight") * pl.col("shrunk_xg_chain_per_90"))
            .filter(is_mid)
            .sum()
            .alias("_mid_xgchain"),
            (pl.col("_pos_min_weight") * pl.col("shrunk_xg_buildup_per_90"))
            .filter(is_def)
            .sum()
            .alias("_def_xgbuildup"),
            pl.col("_pos_min_weight").filter(is_fwd).sum().alias("fwd_minutes_weight_sum"),
            pl.col("_pos_min_weight").filter(is_mid).sum().alias("mid_minutes_weight_sum"),
            pl.col("_pos_min_weight").filter(is_def).sum().alias("def_minutes_weight_sum"),
        )
        .with_columns(
            pl.when(pl.col("fwd_minutes_weight_sum") > 0.0)
            .then(pl.col("_fwd_npxg"))
            .otherwise(PRIOR_NPXG_PER_90_FWD)
            .alias("fwd_weighted_npxg_per_90"),
            pl.when(pl.col("fwd_minutes_weight_sum") > 0.0)
            .then(pl.col("_fwd_xa"))
            .otherwise(PRIOR_XA_PER_90_FWD)
            .alias("fwd_weighted_xa_per_90"),
            pl.when(pl.col("mid_minutes_weight_sum") > 0.0)
            .then(pl.col("_mid_npxg"))
            .otherwise(PRIOR_NPXG_PER_90_MID)
            .alias("mid_weighted_npxg_per_90"),
            pl.when(pl.col("mid_minutes_weight_sum") > 0.0)
            .then(pl.col("_mid_xa"))
            .otherwise(PRIOR_XA_PER_90_MID)
            .alias("mid_weighted_xa_per_90"),
            pl.when(pl.col("mid_minutes_weight_sum") > 0.0)
            .then(pl.col("_mid_xgchain"))
            .otherwise(PRIOR_XGCHAIN_PER_90_MID)
            .alias("mid_weighted_xg_chain_per_90"),
            pl.when(pl.col("def_minutes_weight_sum") > 0.0)
            .then(pl.col("_def_xgbuildup"))
            .otherwise(PRIOR_XGBUILDUP_PER_90_DEF)
            .alias("def_weighted_xg_buildup_per_90"),
        )
        .drop(
            [
                "_fwd_npxg",
                "_fwd_xa",
                "_mid_npxg",
                "_mid_xa",
                "_mid_xgchain",
                "_def_xgbuildup",
            ]
        )
    )

    return agg_df.join(missing_df, on=[event_col, "team"], how="left")
