"""Bullpen rolling quality and 3-day acute workload fatigue (`features/bullpen.py`).

Computes strictly pre-game relief corps quality (`bp_xfip_shrunk`, `bp_k_minus_bb_pct_shrunk`,
`bp_lob_pct_shrunk`) and acute 1-day/3-day workload (`bp_pitches_last_1d`,
`bp_pitches_last_3d`, `bp_outs_last_3d`, `high_leverage_arms_unavailable_count`).
"""

from __future__ import annotations

from datetime import date, datetime
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

    # Index team logs by normalized team
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

    # Index relief pitcher appearances by normalized team
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

    out_rows: list[dict[str, Any]] = []
    for g in games_df.iter_rows(named=True):
        gid = int(g.get("game_id") or g.get("event_id") or 0)
        g_seq = int(g.get("event_seq") or 1)
        g_utc: datetime = g["start_time_utc"]
        g_date: date = g.get("game_date") or g_utc.date()

        row_out: dict[str, Any] = {"game_id": gid, "event_seq": g_seq}

        for side in ("home", "away"):
            team_code = normalize_baseball_team(str(g[f"{side}_team"]))
            lg_code = resolve_team_league(team_code) or str(g.get("league") or "MLB").upper()
            lg_priors = get_league_sabermetric_priors("NPB" if lg_code == "NPB" else "MLB")

            prior_t_logs = [
                tl
                for tl in team_hist_by_club.get(team_code, [])
                if (tl["start_time_utc"], int(tl.get("event_seq") or 1)) < (g_utc, g_seq)
            ]
            prior_rp_logs = [
                rl
                for rl in rp_hist_by_club.get(team_code, [])
                if (rl["start_time_utc"], int(rl.get("event_seq") or 1)) < (g_utc, g_seq)
            ]

            # 1. Acute 1-day and 3-day team bullpen pitch/out counts
            pitches_1d = 0
            pitches_3d = 0
            outs_3d = 0
            for tl in prior_t_logs:
                tl_date: date = tl.get("game_date") or tl["start_time_utc"].date()
                day_diff = (g_date - tl_date).days
                bp_p = int(tl.get("bullpen_pitches") or 0)
                bp_o = int(tl.get("bullpen_outs_pitched") or 0)
                if 0 <= day_diff <= 1:
                    pitches_1d += bp_p
                if 0 <= day_diff <= 3:
                    pitches_3d += bp_p
                    outs_3d += bp_o

            # 2. Rolling 30-game bullpen ERA / xFIP & K-BB%
            recent_t = prior_t_logs[-30:]
            recent_rp = [
                rl
                for rl in prior_rp_logs
                if (g_date - (rl.get("game_date") or rl["start_time_utc"].date())).days <= 45
            ]

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

            # 3. High-leverage reliever unavailability (top-3 relievers by appearances/outs)
            unavail_hl_count = 0
            if prior_rp_logs:
                apps_by_pid: dict[int, list[dict[str, Any]]] = {}
                for rl in prior_rp_logs:
                    pid = int(rl["player_id"])
                    apps_by_pid.setdefault(pid, []).append(rl)

                # Rank relievers by appearances and low ER rate to identify top-3 high-leverage arms
                ranked_pids = sorted(
                    apps_by_pid.keys(),
                    key=lambda pid: (
                        len(apps_by_pid[pid]),
                        sum(int(x.get("outs_pitched") or 0) for x in apps_by_pid[pid]),
                        -sum(int(x.get("er_allowed") or 0) for x in apps_by_pid[pid]),
                    ),
                    reverse=True,
                )[:3]

                for pid in ranked_pids:
                    p_apps = apps_by_pid[pid]
                    pitched_today_g1 = False
                    pitched_d_minus_1 = False
                    pitched_d_minus_2 = False
                    pitches_d_minus_1 = 0

                    for a in p_apps:
                        a_date: date = a.get("game_date") or a["start_time_utc"].date()
                        d_gap = (g_date - a_date).days
                        p_cnt = int(a.get("pitches_thrown") or 0)
                        if d_gap == 0:
                            pitched_today_g1 = True
                        elif d_gap == 1:
                            pitched_d_minus_1 = True
                            pitches_d_minus_1 += p_cnt
                        elif d_gap == 2:
                            pitched_d_minus_2 = True

                    if (
                        pitched_today_g1
                        or (pitched_d_minus_1 and pitched_d_minus_2)
                        or pitches_d_minus_1 >= HIGH_PITCH_RELIEF_THRESHOLD
                    ):
                        unavail_hl_count += 1

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
