"""Deterministic synthetic NFL Silver tables for offline feature/pipeline tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from sports_ml.config import RANDOM_SEED
from sports_ml.sports.football.constants import (
    NFL_TEAMS_32,
    SILVER_FOOTBALL_AVAILABILITY_SCHEMA,
    SILVER_FOOTBALL_GAMES_SCHEMA,
    SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA,
    SILVER_FOOTBALL_QB_LOGS_SCHEMA,
    SILVER_FOOTBALL_TEAM_LOGS_SCHEMA,
    make_football_game_id,
)

TEAMS = ["BUF", "MIA", "NE", "NYJ", "KC", "DEN", "LV", "LAC"]
SEASON_BASES = {
    2014: datetime(2014, 9, 7, 17, 0, tzinfo=UTC),
    2015: datetime(2015, 9, 13, 17, 0, tzinfo=UTC),
    2016: datetime(2016, 9, 11, 17, 0, tzinfo=UTC),
    2017: datetime(2017, 9, 10, 17, 0, tzinfo=UTC),
    2021: datetime(2021, 9, 12, 17, 0, tzinfo=UTC),
    2022: datetime(2022, 9, 11, 17, 0, tzinfo=UTC),
    2023: datetime(2023, 9, 10, 17, 0, tzinfo=UTC),
    2024: datetime(2024, 9, 8, 17, 0, tzinfo=UTC),
}
N_WEEKS = 8
BYE_WEEK = 5


def qb_ids(team: str) -> tuple[int, int]:
    """Return `(starter_id, backup_id)` for a synthetic team."""
    i = TEAMS.index(team)
    return 100_000 + i * 10 + 1, 100_000 + i * 10 + 2


@dataclass
class SyntheticSilver:
    """Container for the five Silver tables."""

    games: pl.DataFrame
    team_logs: pl.DataFrame
    qb_logs: pl.DataFrame
    player_logs: pl.DataFrame
    availability: pl.DataFrame


def _circle_round(r: int) -> list[tuple[str, str]]:
    teams = list(TEAMS)
    n = len(teams)
    rot = [teams[0], *teams[1:][-r % (n - 1) :], *teams[1:][: -r % (n - 1)]] if r else teams
    return [(rot[i], rot[n - 1 - i]) for i in range(n // 2)]


def _kickoff(sunday: datetime, slot: int) -> datetime:
    if slot == 0:
        return sunday - timedelta(days=3) + timedelta(hours=3, minutes=20)  # Thursday night
    if slot == 1:
        return sunday + timedelta(days=1) + timedelta(hours=3, minutes=15)  # Monday night
    return sunday if slot % 2 == 0 else sunday + timedelta(hours=3, minutes=25)


def build_synthetic_silver(
    seasons: tuple[int, ...] = (2021, 2022, 2023),
    *,
    future_weeks: int = 0,
    backup_starts: frozenset[tuple[int, int, str]] = frozenset(),
) -> SyntheticSilver:
    """Build deterministic Silver tables for 8 teams over `seasons`.

    Week `BYE_WEEK` has two byes, every week has a Thursday and a Monday game, and the final
    week of each season adds a wild-card game. In the *last* season, the final `future_weeks`
    weeks are scheduled but unplayed (`FUT`). `backup_starts` holds `(season, week, team)`
    triples in which the backup QB starts.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    off = {t: float(rng.normal(0.0, 0.08)) for t in TEAMS}
    dfn = {t: float(rng.normal(0.0, 0.08)) for t in TEAMS}
    game_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    qb_rows: list[dict[str, object]] = []

    for season in seasons:
        base = SEASON_BASES[season]
        for t in TEAMS:  # season-to-season drift
            off[t] += float(rng.normal(0.0, 0.03))
            dfn[t] += float(rng.normal(0.0, 0.03))
        weeks_this_season = N_WEEKS + 1
        for week in range(1, weeks_this_season + 1):
            sunday = base + timedelta(days=7 * (week - 1))
            if week <= N_WEEKS:
                pairs = _circle_round((week - 1) % 7)
                if week == BYE_WEEK:
                    pairs = pairs[:-1]
                season_type = "REG"
            else:
                pairs = [("KC", "BUF")]
                season_type = "WC"
            unplayed = season == seasons[-1] and week > weeks_this_season - future_weeks
            for slot, (a, b) in enumerate(pairs):
                home, away = (a, b) if (week + slot) % 2 == 0 else (b, a)
                kickoff = _kickoff(sunday, slot)
                gid = make_football_game_id(season, week, away, home)
                venue = NFL_TEAMS_32[home].primary_venue_id
                stats: dict[str, dict[str, float]] = {}
                if not unplayed:
                    for team, opp in ((home, away), (away, home)):
                        epa = off[team] - dfn[opp] - 0.02 + float(rng.normal(0.0, 0.04))
                        plays = float(rng.integers(48, 60))
                        db = round(plays * 0.58)
                        pass_epa = plays * epa * 1.1
                        total_epa = plays * epa
                        stats[team] = {
                            "plays": plays,
                            "epa": total_epa,
                            "db": float(db),
                            "pass_epa": pass_epa,
                            "rush": plays - db,
                            "rush_epa": total_epa - pass_epa,
                            "success": float(round(plays * (0.45 + 0.6 * epa))),
                            "pts": float(max(0, round(22 + 35 * epa + rng.normal(0, 5)))),
                            "to": float(rng.poisson(0.9)),
                            "pen": float(rng.integers(20, 90)),
                        }
                    stats[home]["pa"] = stats[away]["pts"]
                    stats[away]["pa"] = stats[home]["pts"]
                hs = stats[home]["pts"] if stats else None
                as_ = stats[away]["pts"] if stats else None
                home_start = qb_ids(home)[1 if (season, week, home) in backup_starts else 0]
                away_start = qb_ids(away)[1 if (season, week, away) in backup_starts else 0]
                game_rows.append(
                    {
                        "sport": "football",
                        "league": "NFL",
                        "season": season,
                        "season_type": season_type,
                        "week": week,
                        "game_id": gid,
                        "event_seq": 1,
                        "game_date": kickoff.date(),
                        "start_time_utc": kickoff,
                        "home_team": home,
                        "away_team": away,
                        "venue_id": venue,
                        "venue_timezone": "America/New_York",
                        "is_neutral_site": False,
                        "is_international": False,
                        "game_state": "FUT" if unplayed else "FINAL",
                        "home_qb_id": None if unplayed else home_start,
                        "away_qb_id": None if unplayed else away_start,
                        "overtime": None if unplayed else False,
                        "home_score": None if unplayed else int(hs),  # type: ignore[arg-type]
                        "away_score": None if unplayed else int(as_),  # type: ignore[arg-type]
                        "home_win": None if unplayed or hs == as_ else int(hs > as_),  # type: ignore[operator]
                        "is_tie": None if unplayed else int(hs == as_),
                        "market_home_decimal_odds": 1.9,
                        "market_away_decimal_odds": 1.95,
                        "market_prob_home": 0.52,
                        "market_prob_away": 0.48,
                        "market_spread_home": -2.5,
                        "market_total_points": 44.5,
                        "obs_weather_temp": 61.0,
                        "obs_weather_wind": 7.0,
                        "obs_weather_roof": "outdoors",
                        "obs_weather_surface": "grass",
                        "nflverse_game_id": f"{season}_{week:02d}_{away}_{home}",
                    }
                )
                if unplayed:
                    continue
                for team, opp, is_home, starter in (
                    (home, away, True, home_start),
                    (away, home, False, away_start),
                ):
                    s = stats[team]
                    team_rows.append(
                        {
                            "game_id": gid,
                            "event_seq": 1,
                            "game_date": kickoff.date(),
                            "start_time_utc": kickoff,
                            "season": season,
                            "season_type": season_type,
                            "week": week,
                            "league": "NFL",
                            "team": team,
                            "opponent": opp,
                            "is_home": is_home,
                            "venue_id": venue,
                            "overtime": False,
                            "points_for": int(s["pts"]),
                            "points_against": int(s["pa"]),
                            "off_plays": int(s["plays"]) + 6,
                            "off_epa_total": s["epa"],
                            "off_dropbacks": int(s["db"]) + 4,
                            "off_rushes": int(s["rush"]) + 2,
                            "off_pass_epa_total": s["pass_epa"],
                            "off_rush_epa_total": s["rush_epa"],
                            "off_success_plays": int(s["success"]),
                            "off_yards": 300.0,
                            "off_turnovers": int(s["to"]),
                            "off_sacks_taken": 2,
                            "off_fumbles": 1,
                            "off_fumbles_lost": int(s["to"] > 1),
                            "off_rz_drives": 3,
                            "off_rz_td_drives": 2,
                            "f_off_plays": int(s["plays"]),
                            "f_off_epa": s["epa"],
                            "f_off_dropbacks": int(s["db"]),
                            "f_off_pass_epa": s["pass_epa"],
                            "f_off_rushes": int(s["rush"]),
                            "f_off_rush_epa": s["rush_epa"],
                            "f_off_success": int(s["success"]),
                            "f_off_early_plays": int(s["plays"] // 2),
                            "f_off_early_success": int(s["success"] // 2),
                            "f_off_explosive": 5,
                            "f_off_drives": 10,
                            "f_off_drive_points": float(s["pts"]),
                            "def_plays": int(stats[opp]["plays"]),
                            "def_epa_total": stats[opp]["epa"],
                            "penalties": 5,
                            "penalty_yards": int(s["pen"]),
                            "time_of_possession_s": 1800,
                        }
                    )
                    qb_rows.append(
                        {
                            "game_id": gid,
                            "event_seq": 1,
                            "game_date": kickoff.date(),
                            "start_time_utc": kickoff,
                            "season": season,
                            "season_type": season_type,
                            "week": week,
                            "team": team,
                            "opponent": opp,
                            "qb_id": starter,
                            "is_starter": True,
                            "dropbacks": int(s["db"]),
                            "pass_attempts": int(s["db"]) - 2,
                            "qb_epa_total": s["pass_epa"],
                            "cpoe_total": 3.0 * (s["pass_epa"] / max(s["db"], 1.0)) * s["db"],
                            "cpoe_plays": int(s["db"]) - 3,
                            "sacks": 2,
                            "air_yards_total": 7.5 * (int(s["db"]) - 2),
                            "scrambles": 1,
                            "rush_attempts": 2,
                            "rush_epa_total": 0.1,
                        }
                    )

    games = pl.DataFrame(game_rows, schema=SILVER_FOOTBALL_GAMES_SCHEMA).sort(
        "start_time_utc", "event_seq", "game_id"
    )
    team_logs = pl.DataFrame(team_rows, schema=SILVER_FOOTBALL_TEAM_LOGS_SCHEMA_PARTIAL()).sort(
        "start_time_utc", "event_seq", "game_id", "team"
    )
    qb_logs = pl.DataFrame(qb_rows, schema=SILVER_FOOTBALL_QB_LOGS_SCHEMA).sort(
        "start_time_utc", "event_seq", "game_id", "team", "qb_id"
    )
    return SyntheticSilver(
        games=games,
        team_logs=team_logs,
        qb_logs=qb_logs,
        player_logs=pl.DataFrame(schema=SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA),
        availability=pl.DataFrame(schema=SILVER_FOOTBALL_AVAILABILITY_SCHEMA),
    )


def SILVER_FOOTBALL_TEAM_LOGS_SCHEMA_PARTIAL() -> dict[str, pl.DataType]:
    """Team-log schema restricted to the columns the synthetic generator populates."""
    populated = {
        "game_id",
        "event_seq",
        "game_date",
        "start_time_utc",
        "season",
        "season_type",
        "week",
        "league",
        "team",
        "opponent",
        "is_home",
        "venue_id",
        "overtime",
        "points_for",
        "points_against",
        "off_plays",
        "off_epa_total",
        "off_dropbacks",
        "off_rushes",
        "off_pass_epa_total",
        "off_rush_epa_total",
        "off_success_plays",
        "off_yards",
        "off_turnovers",
        "off_sacks_taken",
        "off_fumbles",
        "off_fumbles_lost",
        "off_rz_drives",
        "off_rz_td_drives",
        "f_off_plays",
        "f_off_epa",
        "f_off_dropbacks",
        "f_off_pass_epa",
        "f_off_rushes",
        "f_off_rush_epa",
        "f_off_success",
        "f_off_early_plays",
        "f_off_early_success",
        "f_off_explosive",
        "f_off_drives",
        "f_off_drive_points",
        "def_plays",
        "def_epa_total",
        "penalties",
        "penalty_yards",
        "time_of_possession_s",
    }
    return {k: v for k, v in SILVER_FOOTBALL_TEAM_LOGS_SCHEMA.items() if k in populated}


ROLE_SHARES = {
    "WR1": ("WR", 0.28),
    "WR2": ("WR", 0.20),
    "WR3": ("WR", 0.14),
    "TE1": ("TE", 0.14),
    "RB1": ("RB", 0.10),
    "RB2": ("RB", 0.06),
}


def player_ids(team: str) -> dict[str, int]:
    """Return role -> player id for a synthetic team (QB1 is `qb_ids(team)[0]`)."""
    i = TEAMS.index(team)
    ids = {role: 200_000 + i * 100 + k for k, role in enumerate(ROLE_SHARES)}
    ids["QB1"], ids["QB2"] = qb_ids(team)
    return ids


def build_synthetic_player_logs(silver: SyntheticSilver) -> pl.DataFrame:
    """Deterministic player-game logs consistent with the synthetic team logs."""
    rng = np.random.default_rng(RANDOM_SEED + 7)
    rows: list[dict[str, object]] = []
    games = {int(r["game_id"]): r for r in silver.games.iter_rows(named=True)}
    qb_by = {
        (int(r["game_id"]), r["team"]): int(r["qb_id"])
        for r in silver.qb_logs.iter_rows(named=True)
    }
    for t in silver.team_logs.iter_rows(named=True):
        gid, team = int(t["game_id"]), str(t["team"])
        g = games[gid]
        att = max(int(t["off_dropbacks"]) - 3, 10)
        rush_att = max(int(t["off_rushes"]) - 2, 8)
        epa = float(t["f_off_epa"]) / max(float(t["f_off_plays"]), 1.0)
        ids = player_ids(team)
        base = {
            "game_id": gid,
            "event_seq": 1,
            "game_date": g["game_date"],
            "start_time_utc": g["start_time_utc"],
            "season": g["season"],
            "season_type": g["season_type"],
            "week": g["week"],
            "team": team,
            "opponent": t["opponent"],
        }
        qid = qb_by.get((gid, team), ids["QB1"])
        qb_yds = att * (6.6 + 8.0 * epa) + float(rng.normal(0, 25))
        qb_td = float(rng.poisson(max(0.2, 1.4 + 6.0 * epa)))
        rows.append(
            {
                **base,
                "player_id": qid,
                "player_name": f"QB {qid}",
                "position": "QB",
                "is_starter": True,
                "snaps": 62,
                "snap_share": 1.0,
                "pass_attempts": att,
                "completions": int(att * 0.64),
                "pass_air_yards": att * 7.2,
                "pass_yards": max(qb_yds, 0.0),
                "pass_tds": int(qb_td),
                "interceptions": int(rng.poisson(0.8)),
                "carries": 3,
                "rush_yards": 15.0,
                "rush_tds": int(rng.random() < 0.1),
                "targets": 0,
                "receptions": 0,
                "rec_yards": 0.0,
                "rec_tds": 0,
                "rec_air_yards": 0.0,
                "rz_targets": 0,
                "rz_carries": 1,
                "gl_carries": 0,
            }
        )
        for role, (pos, share) in ROLE_SHARES.items():
            tg = (
                int(round(att * share * float(rng.uniform(0.7, 1.3))))
                if pos != "RB"
                else int(round(att * share))
            )
            ca = int(round(rush_att * (0.6 if role == "RB1" else 0.3))) if pos == "RB" else 0
            if role == "RB2" and (gid + ids["RB2"]) % 5 == 0:
                continue
            rec = int(tg * 0.66)
            yds_t = 8.0 + 8.0 * epa if pos != "RB" else 5.5
            rows.append(
                {
                    **base,
                    "player_id": ids[role],
                    "player_name": f"{role} {team}",
                    "position": pos,
                    "is_starter": role in ("WR1", "WR2", "TE1", "RB1"),
                    "snaps": 50,
                    "snap_share": 0.8 if role in ("WR1", "WR2", "RB1") else 0.5,
                    "pass_attempts": 0,
                    "completions": 0,
                    "pass_air_yards": 0.0,
                    "pass_yards": 0.0,
                    "pass_tds": 0,
                    "interceptions": 0,
                    "carries": ca,
                    "rush_yards": ca * (4.2 + 3.0 * epa) if ca else 0.0,
                    "rush_tds": int(rng.random() < 0.04 * ca) if ca else 0,
                    "targets": tg,
                    "receptions": rec,
                    "rec_yards": max(tg * yds_t + float(rng.normal(0, 8)), 0.0),
                    "rec_tds": int(rng.random() < 0.05 * tg),
                    "rec_air_yards": tg * (9.0 if pos != "RB" else 2.0),
                    "rz_targets": int(rng.integers(0, 3)) if pos != "RB" else 0,
                    "rz_carries": int(ca // 4),
                    "gl_carries": int(ca // 8),
                }
            )
    return pl.DataFrame(rows, schema=SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA).sort(
        "start_time_utc", "event_seq", "game_id", "team", "player_id"
    )


def write_synthetic_silver(
    root: Path,
    silver: SyntheticSilver,
    *,
    availability: pl.DataFrame | None = None,
    player_logs: pl.DataFrame | None = None,
) -> None:
    """Persist synthetic Silver tables under `<root>/data/silver/football/`."""
    out = root / "data" / "silver" / "football"
    out.mkdir(parents=True, exist_ok=True)
    silver.games.write_parquet(out / "games.parquet")
    silver.team_logs.write_parquet(out / "team_game_logs.parquet")
    silver.qb_logs.write_parquet(out / "qb_game_logs.parquet")
    logs = player_logs if player_logs is not None else build_synthetic_player_logs(silver)
    logs.write_parquet(out / "player_game_logs.parquet")
    (availability if availability is not None else silver.availability).write_parquet(
        out / "team_week_availability.parquet"
    )
