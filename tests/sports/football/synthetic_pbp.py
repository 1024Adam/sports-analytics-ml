"""Deterministic synthetic nflverse play-by-play and Next Gen Stats for offline micro tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from sports_ml.config import RANDOM_SEED
from tests.sports.football.synthetic import SyntheticSilver, player_ids

PLAYS_PER_TEAM = 64
PASS_ROLES = ("WR1", "WR2", "WR3", "TE1", "RB1")
RUSH_ROLES = ("RB1", "RB2")


def _gsis(pid: int) -> str:
    return f"00-{pid:07d}"


def build_synthetic_pbp(
    silver: SyntheticSilver, *, seed: int = RANDOM_SEED, plays_per_team: int = PLAYS_PER_TEAM
) -> pl.DataFrame:
    """One row per play for every FINAL game; ids are the nflverse `YYYY_WW_AWAY_HOME` strings."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    finals = silver.games.filter(pl.col("game_state") == "FINAL").sort("start_time_utc", "game_id")
    for g in finals.iter_rows(named=True):
        home, away = str(g["home_team"]), str(g["away_team"])
        counts = {home: 0, away: 0}
        pos_team, play_id, drive = home, 1, 0
        clock, margin = 3600.0, 0.0
        while min(counts.values()) < plays_per_team and clock > 0:
            drive += 1
            drive_len = int(rng.integers(4, 9))
            drive_epa = 0.0
            for k in range(drive_len):
                if counts[pos_team] >= plays_per_team:
                    break
                counts[pos_team] += 1
                down = 1 + min(k, 3)
                is_pass = rng.random() < 0.58
                kneel = rng.random() < 0.01
                spike = (not kneel) and rng.random() < 0.01
                punt = down == 4 and rng.random() < 0.55
                go_pass = is_pass and not kneel and not spike
                sack = go_pass and rng.random() < 0.07
                complete = go_pass and (not sack) and rng.random() < 0.64
                air = float(rng.integers(-2, 30)) if go_pass and not sack else None
                yac = float(rng.integers(0, 12)) if complete else None
                xyac = float(rng.uniform(2.0, 7.0)) if complete else None
                yards = float(rng.integers(-3, 12) if not complete else (air or 0.0) + (yac or 0.0))
                if sack:
                    yards = float(-rng.integers(3, 9))
                ep = float(rng.normal(0.0, 1.1))
                dt = float(rng.uniform(5.0, 42.0))
                clock = max(clock - dt, 0.0)
                half_sec = clock - 1800.0 if clock > 1800.0 else clock
                qtr = 1 + int((3600.0 - clock) // 900.0)
                receiver = rng.choice(PASS_ROLES) if go_pass and not sack else None
                rusher = rng.choice(RUSH_ROLES) if (not is_pass) and not kneel else None
                ids = player_ids(pos_team)
                score_diff = margin if pos_team == home else -margin
                rows.append(
                    {
                        "game_id": g["nflverse_game_id"],
                        "play_id": play_id,
                        "posteam": pos_team,
                        "defteam": away if pos_team == home else home,
                        "play_type": "punt" if punt else ("pass" if is_pass else "run"),
                        "epa": None if punt else ep,
                        "success": float(ep > 0.0),
                        "yards_gained": yards,
                        "pass": float(is_pass and not punt),
                        "rush": float((not is_pass) and not punt),
                        "qb_dropback": float(is_pass and not punt),
                        "wp": float(
                            np.clip(
                                0.5 + 0.35 * np.tanh(score_diff / 14.0) + rng.normal(0, 0.05),
                                0.01,
                                0.99,
                            )
                        ),
                        "down": float(down),
                        "ydstogo": float(rng.integers(1, 12)),
                        "yardline_100": float(rng.integers(3, 95)),
                        "fixed_drive_result": "Touchdown" if drive % 5 == 0 else "Punt",
                        "drive": float(drive),
                        "fumble": float(rng.random() < 0.01),
                        "fumble_lost": float(rng.random() < 0.005),
                        "interception": float(
                            complete is False and go_pass and rng.random() < 0.02
                        ),
                        "sack": float(sack),
                        "qb_hit": float(sack or (go_pass and rng.random() < 0.08)),
                        "penalty": float(rng.random() < 0.05),
                        "penalty_team": pos_team if rng.random() < 0.5 else None,
                        "penalty_yards": float(rng.choice([0, 5, 10])),
                        "drive_time_of_possession": "2:30",
                        "qb_kneel": float(kneel),
                        "qb_spike": float(spike),
                        "xpass": float(np.clip(rng.normal(0.58, 0.12), 0.05, 0.95)),
                        "cpoe": float(rng.normal(0.0, 12.0)) if go_pass and not sack else None,
                        "air_yards": air,
                        "yards_after_catch": yac,
                        "xyac_mean_yardage": xyac,
                        "complete_pass": float(complete),
                        "pass_attempt": float(go_pass and not sack),
                        "qtr": float(min(qtr, 4)),
                        "half_seconds_remaining": float(half_sec),
                        "game_seconds_remaining": float(clock),
                        "score_differential": float(score_diff),
                        "receiver_player_id": _gsis(ids[str(receiver)]) if receiver else None,
                        "rusher_player_id": _gsis(ids[str(rusher)]) if rusher else None,
                        "receiving_yards": yards if complete else None,
                    }
                )
                play_id += 1
                drive_epa += ep
            margin += float(rng.normal(0.0, 1.5)) * (1 if pos_team == home else -1)
            pos_team = away if pos_team == home else home
    return pl.DataFrame(rows)


def build_synthetic_ngs(
    silver: SyntheticSilver, *, first_season: int = 2016, seed: int = 7
) -> dict[str, pl.DataFrame]:
    """Weekly NGS passing/rushing/receiving rows (week > 0) for seasons >= `first_season`."""
    rng = np.random.default_rng(seed)
    passing: list[dict[str, Any]] = []
    rushing: list[dict[str, Any]] = []
    receiving: list[dict[str, Any]] = []
    games = silver.games.filter(
        (pl.col("game_state") == "FINAL") & (pl.col("season") >= first_season)
    )
    for g in games.iter_rows(named=True):
        for team in (g["home_team"], g["away_team"]):
            ids = player_ids(str(team))
            base = {
                "season": g["season"],
                "season_type": g["season_type"],
                "week": g["week"],
                "team_abbr": team,
            }
            att = float(rng.integers(25, 45))
            passing.append(
                {
                    **base,
                    "player_gsis_id": _gsis(ids["QB1"]),
                    "attempts": att,
                    "avg_time_to_throw": float(rng.normal(2.75, 0.2)),
                    "completion_percentage_above_expectation": float(rng.normal(0.0, 4.0)),
                }
            )
            rushing.append(
                {
                    **base,
                    "player_gsis_id": _gsis(ids["RB1"]),
                    "rush_attempts": float(rng.integers(8, 25)),
                    "rush_yards_over_expected": float(rng.normal(0.0, 10.0)),
                }
            )
            for role in ("WR1", "WR2", "TE1"):
                tg = float(rng.integers(3, 11))
                receiving.append(
                    {
                        **base,
                        "player_gsis_id": _gsis(ids[role]),
                        "targets": tg,
                        "receptions": float(int(tg * 0.65)),
                        "avg_separation": float(rng.normal(3.0, 0.4)),
                        "avg_yac_above_expectation": float(rng.normal(0.0, 1.0)),
                    }
                )
    return {
        "passing": pl.DataFrame(passing),
        "rushing": pl.DataFrame(rushing),
        "receiving": pl.DataFrame(receiving),
    }
