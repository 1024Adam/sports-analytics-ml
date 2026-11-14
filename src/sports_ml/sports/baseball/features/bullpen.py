"""Bullpen rolling quality and 3-day acute workload fatigue (`features/bullpen.py`).

Computes strictly pre-game relief corps quality (`bp_xfip_shrunk`, `bp_k_minus_bb_pct_shrunk`,
`bp_lob_pct_shrunk`) and acute 1-day/3-day workload (`bp_pitches_last_1d`,
`bp_pitches_last_3d`, `bp_outs_last_3d`, `high_leverage_arms_unavailable_count`).
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import date, datetime, timedelta
from typing import Any, Final

import polars as pl

from sports_ml.core.shrinkage import shrink_scalar
from sports_ml.core.temporal import sort_events_chronologically
from sports_ml.sports.baseball.constants import (
    SHRINKAGE_K_TEAM_GAMES,
    get_league_sabermetric_priors,
    normalize_baseball_team,
    resolve_team_league,
)

HIGH_PITCH_RELIEF_THRESHOLD: Final[int] = 28


def _row_key(row: dict[str, Any]) -> tuple[datetime, int]:
    return (row["start_time_utc"], int(row.get("event_seq") or 1))


def _row_date(row: dict[str, Any]) -> date:
    row_date: date = row.get("game_date") or row["start_time_utc"].date()
    return row_date


class _ClubTeamIndex:
    """Chronological team logs with date buckets so each game only touches nearby rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = sorted(rows, key=_row_key)
        self.keys = [_row_key(r) for r in self.rows]
        self.by_date: dict[date, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.by_date.setdefault(_row_date(r), []).append(i)

    def acute_workload(self, g_key: tuple[datetime, int], g_date: date) -> tuple[int, int, int]:
        n_prior = bisect_left(self.keys, g_key)
        pitches_1d = pitches_3d = outs_3d = 0
        for day_diff in range(4):
            for i in self.by_date.get(g_date - timedelta(days=day_diff), ()):
                if i >= n_prior:
                    break
                bp_p = int(self.rows[i].get("bullpen_pitches") or 0)
                bp_o = int(self.rows[i].get("bullpen_outs_pitched") or 0)
                if day_diff <= 1:
                    pitches_1d += bp_p
                pitches_3d += bp_p
                outs_3d += bp_o
        return pitches_1d, pitches_3d, outs_3d

    def last_prior(self, g_key: tuple[datetime, int], n: int) -> list[dict[str, Any]]:
        n_prior = bisect_left(self.keys, g_key)
        return self.rows[max(0, n_prior - n) : n_prior]


class _ClubReliefIndex:
    """Chronological relief appearances for one club."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = sorted(rows, key=_row_key)
        self.keys = [_row_key(r) for r in self.rows]
        self.dates = [_row_date(r) for r in self.rows]
        self.prefix_max_date: list[date] = []
        running: date | None = None
        for d in self.dates:
            running = d if running is None or d > running else running
            self.prefix_max_date.append(running)

    def recent_prior(
        self, g_key: tuple[datetime, int], g_date: date, window_days: int
    ) -> list[dict[str, Any]]:
        n_prior = bisect_left(self.keys, g_key)
        cutoff = g_date - timedelta(days=window_days)
        picked: list[dict[str, Any]] = []
        i = n_prior - 1
        # No earlier row can qualify once the running maximum date falls before the cutoff.
        while i >= 0 and self.prefix_max_date[i] >= cutoff:
            if self.dates[i] >= cutoff:
                picked.append(self.rows[i])
            i -= 1
        picked.reverse()
        return picked

    def high_leverage_unavailable(
        self,
        queries: list[tuple[tuple[datetime, int], int, str]],
        games: list[dict[str, Any]],
    ) -> list[tuple[int, str, int]]:
        """Top-3 relievers (by prior appearances, outs, fewer ER) that are fatigued per query."""
        results: list[tuple[int, str, int]] = []
        stats: dict[int, list[int]] = {}
        by_date: dict[int, dict[date, list[int]]] = {}
        pointer = 0
        for g_key, gi, side in sorted(queries, key=lambda q: q[0]):
            while pointer < len(self.rows) and self.keys[pointer] < g_key:
                row = self.rows[pointer]
                pid = int(row["player_id"])
                st = stats.setdefault(pid, [0, 0, 0])
                st[0] += 1
                st[1] += int(row.get("outs_pitched") or 0)
                st[2] += int(row.get("er_allowed") or 0)
                bucket = by_date.setdefault(pid, {}).setdefault(self.dates[pointer], [0, 0])
                bucket[0] += 1
                bucket[1] += int(row.get("pitches_thrown") or 0)
                pointer += 1
            g = games[gi]
            g_date: date = g.get("game_date") or g["start_time_utc"].date()
            ranked = sorted(
                stats, key=lambda p: (stats[p][0], stats[p][1], -stats[p][2]), reverse=True
            )[:3]
            count = 0
            for pid in ranked:
                dates = by_date[pid]
                d1 = dates.get(g_date - timedelta(days=1))
                d2 = dates.get(g_date - timedelta(days=2))
                if (
                    g_date in dates
                    or (d1 is not None and d2 is not None)
                    or (d1 is not None and d1[1] >= HIGH_PITCH_RELIEF_THRESHOLD)
                ):
                    count += 1
            results.append((gi, side, count))
        return results


def build_bullpen_pregame_features(
    games_df: pl.DataFrame,
    team_logs_df: pl.DataFrame,
    pitcher_logs_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute strictly pre-game bullpen skill and 3-day fatigue features for `games_df`.

    For every game `(start_time_utc, event_seq)`:
    - `bp_pitches_last_1d`: Relief pitches thrown by the team on `game_date - 1d` (plus
      earlier same-day Game 1 relief pitches when evaluating Game 2 of a doubleheader!).
    - `bp_pitches_last_3d` & `bp_outs_last_3d`: Relief pitches and outs over the prior 3
      calendar days (`0 <= (game_date - prev_date).days <= 3` and `< (start_time_utc, event_seq)`).
    - `high_leverage_arms_unavailable_count`: Number of the club's top-3 relievers (by
      prior relief appearances/outs) who pitched on consecutive prior days (`d-1` and `d-2`),
      pitched earlier today in Game 1 of a doubleheader, or threw `>= 28` pitches yesterday.
    """
    if games_df.is_empty():
        return games_df

    team_hist_by_club: dict[str, list[dict[str, Any]]] = {}
    if not team_logs_df.is_empty():
        t_work = team_logs_df
        if "game_id" in t_work.columns and "event_id" not in t_work.columns:
            t_work = t_work.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
        if "event_seq" not in t_work.columns:
            t_work = t_work.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
        t_sorted = sort_events_chronologically(t_work, extra_tiebreakers=["team"])
        for r in t_sorted.iter_rows(named=True):
            c = normalize_baseball_team(str(r["team"]))
            team_hist_by_club.setdefault(c, []).append(r)

    rp_hist_by_club: dict[str, list[dict[str, Any]]] = {}
    if pitcher_logs_df is not None and not pitcher_logs_df.is_empty():
        p_work = pitcher_logs_df.filter(pl.col("is_starter") == pl.lit(False))
        if not p_work.is_empty():
            if "game_id" in p_work.columns and "event_id" not in p_work.columns:
                p_work = p_work.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
            if "event_seq" not in p_work.columns:
                p_work = p_work.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
            p_sorted = sort_events_chronologically(p_work, extra_tiebreakers=["player_id"])
            for r in p_sorted.iter_rows(named=True):
                c = normalize_baseball_team(str(r["team"]))
                rp_hist_by_club.setdefault(c, []).append(r)

    team_idx = {c: _ClubTeamIndex(rows) for c, rows in team_hist_by_club.items()}
    rp_idx = {c: _ClubReliefIndex(rows) for c, rows in rp_hist_by_club.items()}

    games = list(games_df.iter_rows(named=True))
    game_keys: list[tuple[datetime, int]] = []
    hl_queries: dict[str, list[tuple[tuple[datetime, int], int, str]]] = {}
    for gi, g in enumerate(games):
        key = (g["start_time_utc"], int(g.get("event_seq") or 1))
        game_keys.append(key)
        for side in ("home", "away"):
            club = normalize_baseball_team(str(g[f"{side}_team"]))
            hl_queries.setdefault(club, []).append((key, gi, side))

    hl_counts: dict[tuple[int, str], int] = {}
    for club, queries in hl_queries.items():
        index = rp_idx.get(club)
        if index is None:
            continue
        for gi, side, count in index.high_leverage_unavailable(queries, games):
            hl_counts[(gi, side)] = count

    out_rows: list[dict[str, Any]] = []
    for gi, g in enumerate(games):
        gid = int(g.get("game_id") or g.get("event_id") or 0)
        g_seq = int(g.get("event_seq") or 1)
        g_utc: datetime = g["start_time_utc"]
        g_date: date = g.get("game_date") or g_utc.date()
        g_key = game_keys[gi]

        row_out: dict[str, Any] = {"game_id": gid, "event_seq": g_seq}

        for side in ("home", "away"):
            team_code = normalize_baseball_team(str(g[f"{side}_team"]))
            lg_code = resolve_team_league(team_code) or str(g.get("league") or "MLB").upper()
            lg_priors = get_league_sabermetric_priors("NPB" if lg_code == "NPB" else "MLB")

            t_index = team_idx.get(team_code)
            r_index = rp_idx.get(team_code)

            pitches_1d = 0
            pitches_3d = 0
            outs_3d = 0
            recent_t: list[dict[str, Any]] = []
            if t_index is not None:
                pitches_1d, pitches_3d, outs_3d = t_index.acute_workload(g_key, g_date)
                recent_t = t_index.last_prior(g_key, 30)

            recent_rp = r_index.recent_prior(g_key, g_date, 45) if r_index is not None else []

            if recent_rp:
                tot_rp_outs = sum(int(r.get("outs_pitched") or 0) for r in recent_rp)
                tot_rp_bf = sum(int(r.get("batters_faced") or 0) for r in recent_rp)
                tot_rp_k = sum(int(r.get("k_recorded") or 0) for r in recent_rp)
                tot_rp_bb = sum(
                    max(
                        0,
                        int(r.get("bb_allowed") or 0)
                        - int(r.get("ibb_allowed") or 0)
                        + int(r.get("hbp_allowed") or 0),
                    )
                    for r in recent_rp
                )
                tot_rp_hr = sum(int(r.get("hr_allowed") or 0) for r in recent_rp)
                tot_rp_fb = sum(int(r.get("fb_count") or 0) for r in recent_rp)
                rp_ip = max(tot_rp_outs / 3.0, 1.0)
                if tot_rp_fb > 0:
                    raw_bp_xfip = (
                        13.0 * (tot_rp_fb * lg_priors.lg_hr_per_fb)
                        + 3.0 * tot_rp_bb
                        - 2.0 * tot_rp_k
                    ) / rp_ip + lg_priors.fip_constant
                else:
                    raw_bp_xfip = (
                        13.0 * tot_rp_hr + 3.0 * tot_rp_bb - 2.0 * tot_rp_k
                    ) / rp_ip + lg_priors.fip_constant
                raw_bp_k_bb = (tot_rp_k - tot_rp_bb) / max(tot_rp_bf, 1)
                bp_xfip_s = shrink_scalar(
                    raw_bp_xfip, len(recent_t) or 10, lg_priors.lg_bp_xfip, SHRINKAGE_K_TEAM_GAMES
                )
                bp_k_bb_s = shrink_scalar(
                    raw_bp_k_bb,
                    len(recent_t) or 10,
                    lg_priors.lg_sp_k_pct - lg_priors.lg_sp_bb_pct + 0.020,
                    SHRINKAGE_K_TEAM_GAMES,
                )
            elif recent_t:
                tot_bp_outs = sum(int(t.get("bullpen_outs_pitched") or 0) for t in recent_t)
                tot_bp_er = sum(int(t.get("bullpen_er") or 0) for t in recent_t)
                bp_ip = max(tot_bp_outs / 3.0, 1.0)
                raw_bp_era = 9.0 * tot_bp_er / bp_ip
                bp_xfip_s = shrink_scalar(
                    raw_bp_era, len(recent_t), lg_priors.lg_bp_xfip, SHRINKAGE_K_TEAM_GAMES
                )
                bp_k_bb_s = lg_priors.lg_sp_k_pct - lg_priors.lg_sp_bb_pct + 0.020
            else:
                bp_xfip_s = lg_priors.lg_bp_xfip
                bp_k_bb_s = lg_priors.lg_sp_k_pct - lg_priors.lg_sp_bb_pct + 0.020

            unavail_hl_count = hl_counts.get((gi, side), 0)

            row_out[f"{side}_bp_xfip_shrunk"] = float(bp_xfip_s)
            row_out[f"{side}_bp_k_minus_bb_pct_shrunk"] = float(bp_k_bb_s)
            row_out[f"{side}_bp_pitches_last_1d"] = float(pitches_1d)
            row_out[f"{side}_bp_pitches_last_3d"] = float(pitches_3d)
            row_out[f"{side}_bp_outs_last_3d"] = float(outs_3d)
            row_out[f"{side}_high_leverage_arms_unavailable_count"] = float(unavail_hl_count)

        out_rows.append(row_out)

    bp_df = pl.DataFrame(out_rows)
    join_keys = ["game_id", "event_seq"] if "event_seq" in games_df.columns else ["game_id"]
    if "event_seq" not in games_df.columns:
        bp_df = bp_df.drop("event_seq")
    return games_df.join(bp_df, on=join_keys, how="left")
