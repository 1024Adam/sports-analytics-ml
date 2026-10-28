"""NHL shift-chart overlap and TOI Quality of Competition / Teammates (`ingestion/shifts.py`).

Computes per-game skater deployment context from shift-interval overlaps (or pre-aggregated
shift summaries):
- `qoc_opponent_toi_share`: TOI-weighted average of opposing skaters' share of their team's
  even-strength ice time (measures whether a skater faces opponent top-pair/top-line
  matchups vs. sheltered 4th-line minutes).
- `qot_teammate_ixg_per_60`: TOI-weighted average of on-ice teammates' pre-game rolling
  `ixG/60` (strictly `< T_game`, never leaking same-game or future teammate goals/xG).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from sports_ml.sports.hockey.constants import normalize_tricode
from sports_ml.sports.hockey.ingestion.nhl_api import parse_toi_minutes

DEFAULT_QOC_TOI_SHARE: Final[float] = 0.30
DEFAULT_QOT_IXG_PER_60: Final[float] = 0.72


def compute_shift_overlap_qoc_qot(
    shift_records: Sequence[Mapping[str, Any]],
    *,
    skater_pregame_ixg_per_60: Mapping[int, float] | None = None,
) -> pl.DataFrame:
    """Compute per-`(game_id, player_id)` `qoc_opponent_toi_share` and `qot_teammate_ixg_per_60`.

    Accepts shift records with:
    - `game_id`, `player_id`, `team`, `period`, `start_sec` (or `startTime`),
      `end_sec` (or `endTime`).
    Or pre-summarized records already containing `qoc_opponent_toi_share` and
    `qot_teammate_ixg_per_60`.
    """
    schema = {
        "game_id": pl.Int64(),
        "player_id": pl.Int64(),
        "team": pl.Utf8(),
        "qoc_opponent_toi_share": pl.Float64(),
        "qot_teammate_ixg_per_60": pl.Float64(),
    }
    if not shift_records:
        return pl.DataFrame(schema=schema)

    # Fast path if records are already game-skater summaries
    first = shift_records[0]
    if "qoc_opponent_toi_share" in first and "start_sec" not in first and "startTime" not in first:
        rows = [
            {
                "game_id": int(r["game_id"]),
                "player_id": int(r["player_id"]),
                "team": normalize_tricode(str(r.get("team") or "BOS")),
                "qoc_opponent_toi_share": float(
                    r.get("qoc_opponent_toi_share") or DEFAULT_QOC_TOI_SHARE
                ),
                "qot_teammate_ixg_per_60": float(
                    r.get("qot_teammate_ixg_per_60") or DEFAULT_QOT_IXG_PER_60
                ),
            }
            for r in shift_records
        ]
        return pl.DataFrame(rows, schema=schema).unique(
            subset=["game_id", "player_id"], keep="last"
        )

    ixg_lookup = dict(skater_pregame_ixg_per_60 or {})

    # Group shift intervals by game_id: list of (player_id, team, period, start_s, end_s)
    by_game: dict[int, list[tuple[int, str, int, float, float]]] = {}
    for rec in shift_records:
        gid = int(rec.get("game_id") or rec.get("gameId") or 2024020001)
        pid = int(rec.get("player_id") or rec.get("playerId") or 0)
        if pid <= 0:
            continue
        team = normalize_tricode(str(rec.get("team") or rec.get("teamAbbrev") or "BOS"))
        period = int(rec.get("period") or 1)

        if "start_sec" in rec and rec.get("start_sec") is not None:
            s_sec = float(rec["start_sec"])
        else:
            s_sec = parse_toi_minutes(rec.get("startTime")) * 60.0

        if "end_sec" in rec and rec.get("end_sec") is not None:
            e_sec = float(rec["end_sec"])
        else:
            e_sec = parse_toi_minutes(rec.get("endTime")) * 60.0

        if e_sec > s_sec:
            by_game.setdefault(gid, []).append((pid, team, period, s_sec, e_sec))

    out_rows: list[dict[str, Any]] = []
    for gid, intervals in by_game.items():
        # 1. Total TOI per player and max/team TOI for normalizing opponent TOI share
        player_toi: dict[int, float] = {}
        player_team: dict[int, str] = {}
        team_total_skater_toi: dict[str, float] = {}

        for pid, team, _, s_sec, e_sec in intervals:
            dur = e_sec - s_sec
            player_toi[pid] = player_toi.get(pid, 0.0) + dur
            player_team[pid] = team
            team_total_skater_toi[team] = team_total_skater_toi.get(team, 0.0) + dur

        # A skater's share of their team's total skater TOI (monotonically preserved in (0, 1))
        player_toi_share: dict[int, float] = {}
        for pid, tot_s in player_toi.items():
            t_code = player_team[pid]
            team_tot_sec = max(60.0, team_total_skater_toi.get(t_code, 300.0))
            player_toi_share[pid] = max(0.01, min(0.99, tot_s / team_tot_sec))

        # 2. Pairwise shift overlap within each period
        by_period: dict[int, list[tuple[int, str, float, float]]] = {}
        for pid, team, period, s_sec, e_sec in intervals:
            by_period.setdefault(period, []).append((pid, team, s_sec, e_sec))

        opp_weighted_sum: dict[int, float] = {pid: 0.0 for pid in player_toi}
        opp_overlap_sec: dict[int, float] = {pid: 0.0 for pid in player_toi}
        tm_weighted_sum: dict[int, float] = {pid: 0.0 for pid in player_toi}
        tm_overlap_sec: dict[int, float] = {pid: 0.0 for pid in player_toi}

        for p_list in by_period.values():
            n_int = len(p_list)
            for i in range(n_int):
                pid_i, team_i, s_i, e_i = p_list[i]
                for j in range(i + 1, n_int):
                    pid_j, team_j, s_j, e_j = p_list[j]
                    if pid_i == pid_j:
                        continue
                    overlap = min(e_i, e_j) - max(s_i, s_j)
                    if overlap <= 0.0:
                        continue
                    if team_i != team_j:
                        opp_weighted_sum[pid_i] += overlap * player_toi_share[pid_j]
                        opp_overlap_sec[pid_i] += overlap
                        opp_weighted_sum[pid_j] += overlap * player_toi_share[pid_i]
                        opp_overlap_sec[pid_j] += overlap
                    else:
                        ixg_j = ixg_lookup.get(pid_j, DEFAULT_QOT_IXG_PER_60)
                        ixg_i = ixg_lookup.get(pid_i, DEFAULT_QOT_IXG_PER_60)
                        tm_weighted_sum[pid_i] += overlap * ixg_j
                        tm_overlap_sec[pid_i] += overlap
                        tm_weighted_sum[pid_j] += overlap * ixg_i
                        tm_overlap_sec[pid_j] += overlap

        for pid, t_code in sorted(player_team.items()):
            qoc = (
                opp_weighted_sum[pid] / opp_overlap_sec[pid]
                if opp_overlap_sec[pid] > 0.0
                else DEFAULT_QOC_TOI_SHARE
            )
            qot = (
                tm_weighted_sum[pid] / tm_overlap_sec[pid]
                if tm_overlap_sec[pid] > 0.0
                else DEFAULT_QOT_IXG_PER_60
            )
            out_rows.append(
                {
                    "game_id": gid,
                    "player_id": pid,
                    "team": t_code,
                    "qoc_opponent_toi_share": float(qoc),
                    "qot_teammate_ixg_per_60": float(qot),
                }
            )

    if not out_rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(out_rows, schema=schema).sort(["game_id", "player_id"])


def enrich_skater_micro_logs_with_shifts(
    skater_micro_df: pl.DataFrame,
    shifts_qoc_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Merge shift `qoc_opponent_toi_share` & `qot_teammate_ixg_per_60` into skater micro logs."""
    if skater_micro_df.is_empty() or shifts_qoc_df is None or shifts_qoc_df.is_empty():
        return skater_micro_df

    lookup = shifts_qoc_df.select(
        [
            pl.col("game_id").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64),
            pl.col("qoc_opponent_toi_share").cast(pl.Float64).alias("_sh_qoc"),
            pl.col("qot_teammate_ixg_per_60").cast(pl.Float64).alias("_sh_qot"),
        ]
    ).unique(subset=["game_id", "player_id"], keep="last")

    return (
        skater_micro_df.join(lookup, on=["game_id", "player_id"], how="left")
        .with_columns(
            pl.coalesce([pl.col("_sh_qoc"), pl.col("qoc_opponent_toi_share")]).alias(
                "qoc_opponent_toi_share"
            ),
            pl.coalesce([pl.col("_sh_qot"), pl.col("qot_teammate_ixg_per_60")]).alias(
                "qot_teammate_ixg_per_60"
            ),
        )
        .drop(["_sh_qoc", "_sh_qot"])
    )
