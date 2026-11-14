"""Schedule/venue context and Elo features for NFL games (`features/schedule_context.py`).

Wraps the weekly-cadence hooks (`schedule_hooks.py`) into one row per game with home/away
rest, bye, travel, and body-clock features, game-level flags (division/conference/playoff
round/neutral/international), and pre-game Elo plus QB-adjusted Elo. Every value depends only on
the published schedule and on completed games before kickoff.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Final

import polars as pl

from sports_ml.sports.football.constants import get_football_team_metadata, normalize_football_team
from sports_ml.sports.football.features.elo_hooks import (
    FootballQBEloHook,
    FootballVenueEloHook,
    compute_football_elo_ratings,
)
from sports_ml.sports.football.features.schedule_hooks import build_football_schedule_features

PLAYOFF_ROUND: Final[dict[str, int]] = {"REG": 0, "WC": 1, "DIV": 2, "CON": 3, "SB": 4}
DEFAULT_DAYS_REST: Final[float] = 7.0

TEAM_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "days_rest",
    "is_short_week",
    "is_post_bye",
    "is_post_mini_bye",
    "prior_game_was_overtime",
    "travel_km",
    "timezone_shift",
    "consecutive_road_games",
    "kickoff_body_clock_hour",
)
GAME_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "is_thursday_game",
    "is_monday_game",
    "is_international",
    "is_neutral_site",
    "is_division_game",
    "is_conference_game",
    "is_playoff",
    "playoff_round",
)
ELO_COLUMNS: Final[tuple[str, ...]] = (
    "elo_home_pre",
    "elo_away_pre",
    "diff_elo_pre",
    "elo_home_win_prob",
    "qb_elo_diff_pre",
    "qb_elo_home_win_prob",
)


def _long_team_games(games_df: pl.DataFrame) -> pl.DataFrame:
    keep = ["game_id", "event_seq", "season", "week", "start_time_utc", "venue_id", "overtime"]
    home = games_df.select(*keep, pl.col("home_team").alias("team"), pl.lit(True).alias("is_home"))
    away = games_df.select(*keep, pl.col("away_team").alias("team"), pl.lit(False).alias("is_home"))
    return pl.concat([home, away]).sort("start_time_utc", "event_seq", "game_id", "team")


def _consecutive_road_games(long_df: pl.DataFrame) -> dict[tuple[int, str], int]:
    """Consecutive away games immediately before each game (within the season, byes ignored)."""
    streak: dict[tuple[str, int], int] = defaultdict(int)
    out: dict[tuple[int, str], int] = {}
    for r in long_df.select("game_id", "team", "season", "is_home").iter_rows(named=True):
        key = (str(r["team"]), int(r["season"]))
        out[(int(r["game_id"]), str(r["team"]))] = streak[key]
        streak[key] = 0 if r["is_home"] else streak[key] + 1
    return out


def build_schedule_context(games_df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per game with rest, travel, bye, round, and matchup-context features."""
    long_df = _long_team_games(games_df)
    feats = build_football_schedule_features(
        long_df.drop("week").with_columns(pl.col("overtime").cast(pl.Boolean))
    )
    road_df = pl.DataFrame(
        [
            {"game_id": gid, "team": team, "consecutive_road_games": n}
            for (gid, team), n in _consecutive_road_games(long_df).items()
        ],
        schema={"game_id": pl.Int64, "team": pl.Utf8, "consecutive_road_games": pl.Int32},
    )
    team_rows = feats.select(
        "game_id",
        "team",
        "days_rest",
        "is_short_week",
        "is_post_bye",
        "is_post_mini_bye",
        "prior_game_was_overtime",
        pl.col("travel_km_prev_game").alias("travel_km"),
        pl.col("tz_shift_hours").alias("timezone_shift"),
        pl.col("kickoff_local_hour_vs_body_clock").alias("kickoff_body_clock_hour"),
        "is_thursday_game",
        "is_monday_game",
    ).join(road_df, on=["game_id", "team"], how="left")

    def side(prefix: str, team_col: str) -> pl.DataFrame:
        base = games_df.select("game_id", pl.col(team_col).alias("team"))
        joined = base.join(team_rows, on=["game_id", "team"], how="left")
        return joined.select(
            "game_id",
            *[pl.col(c).alias(f"{prefix}_{c}") for c in TEAM_SCHEDULE_COLUMNS],
        )

    home = side("home", "home_team")
    away = side("away", "away_team")
    thursday = team_rows.group_by("game_id").agg(
        pl.col("is_thursday_game").first().alias("is_thursday_game"),
        pl.col("is_monday_game").first().alias("is_monday_game"),
    )

    meta = []
    for r in games_df.select(
        "game_id", "home_team", "away_team", "season_type", "is_neutral_site", "is_international"
    ).iter_rows(named=True):
        h = get_football_team_metadata(str(r["home_team"]))
        a = get_football_team_metadata(str(r["away_team"]))
        same_conf = h is not None and a is not None and h.conference == a.conference
        same_div = same_conf and h is not None and a is not None and h.division == a.division
        stype = str(r["season_type"]).upper()
        meta.append(
            {
                "game_id": int(r["game_id"]),
                "is_international": bool(r["is_international"]),
                "is_neutral_site": bool(r["is_neutral_site"]),
                "is_division_game": bool(same_div),
                "is_conference_game": bool(same_conf),
                "is_playoff": stype != "REG",
                "playoff_round": PLAYOFF_ROUND.get(stype, 0),
            }
        )
    meta_df = pl.DataFrame(
        meta,
        schema={
            "game_id": pl.Int64,
            "is_international": pl.Boolean,
            "is_neutral_site": pl.Boolean,
            "is_division_game": pl.Boolean,
            "is_conference_game": pl.Boolean,
            "is_playoff": pl.Boolean,
            "playoff_round": pl.Int32,
        },
    )

    out = (
        games_df.select("game_id")
        .join(home, on="game_id")
        .join(away, on="game_id")
        .join(thursday, on="game_id")
        .join(meta_df, on="game_id")
    )
    flags = {c: pl.col(c).is_null() for c in ("home_days_rest", "away_days_rest")}
    out = out.with_columns(
        *[expr.alias(f"{c}_is_imputed") for c, expr in flags.items()],
    ).with_columns(
        pl.col("home_days_rest").fill_null(DEFAULT_DAYS_REST),
        pl.col("away_days_rest").fill_null(DEFAULT_DAYS_REST),
        pl.col("home_travel_km", "away_travel_km").fill_null(0.0),
        pl.col("home_timezone_shift", "away_timezone_shift").fill_null(0.0),
        pl.col("home_kickoff_body_clock_hour", "away_kickoff_body_clock_hour").fill_null(13.0),
        pl.col(
            "home_is_short_week",
            "away_is_short_week",
            "home_is_post_bye",
            "away_is_post_bye",
            "home_is_post_mini_bye",
            "away_is_post_mini_bye",
            "home_prior_game_was_overtime",
            "away_prior_game_was_overtime",
            "is_thursday_game",
            "is_monday_game",
        ).fill_null(False),
        pl.col("home_consecutive_road_games", "away_consecutive_road_games").fill_null(0),
    )
    return out.with_columns(
        (pl.col("home_days_rest") - pl.col("away_days_rest")).alias("rest_diff")
    ).sort("game_id")


def build_elo_context(
    games_df: pl.DataFrame,
    *,
    qb_hook: FootballQBEloHook | None = None,
    schedule_context: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return per-game pre-game Elo and QB/venue-adjusted Elo (`ELO_COLUMNS`).

    The plain Elo uses only ratings and HFA. The adjusted Elo adds the starting-QB hook and the
    venue hook (altitude, dome, rest differential from `schedule_context`).
    """
    work = games_df.with_columns(pl.col("game_id").alias("event_id"))
    plain = compute_football_elo_ratings(work).select(
        "game_id",
        "elo_home_pre",
        "elo_away_pre",
        pl.col("elo_home_win_prob"),
    )
    rest_lookup: dict[tuple[int, str], float] = {}
    if schedule_context is not None:
        ctx = games_df.select("game_id", "home_team", "away_team").join(
            schedule_context.select("game_id", "home_days_rest", "away_days_rest"), on="game_id"
        )
        for r in ctx.iter_rows(named=True):
            gid = int(r["game_id"])
            rest_lookup[(gid, normalize_football_team(str(r["home_team"])))] = float(
                r["home_days_rest"]
            )
            rest_lookup[(gid, normalize_football_team(str(r["away_team"])))] = float(
                r["away_days_rest"]
            )
    venue_by_event = {
        int(r["game_id"]): str(r["venue_id"])
        for r in games_df.select("game_id", "venue_id").iter_rows(named=True)
        if r["venue_id"] is not None
    }
    neutral = [
        int(r["game_id"])
        for r in games_df.select("game_id", "is_neutral_site").iter_rows(named=True)
        if r["is_neutral_site"]
    ]
    venue_hook = FootballVenueEloHook(
        venue_by_event, neutral_event_ids=neutral, rest_days_by_event_team=rest_lookup
    )
    adjusted = compute_football_elo_ratings(work, qb_hook=qb_hook, venue_hook=venue_hook).select(
        "game_id",
        pl.col("elo_diff_pre").alias("qb_elo_diff_pre"),
        pl.col("elo_home_win_prob").alias("qb_elo_home_win_prob"),
    )
    return (
        plain.join(adjusted, on="game_id")
        .with_columns((pl.col("elo_home_pre") - pl.col("elo_away_pre")).alias("diff_elo_pre"))
        .select("game_id", *ELO_COLUMNS)
        .sort("game_id")
    )


__all__ = [
    "DEFAULT_DAYS_REST",
    "ELO_COLUMNS",
    "GAME_SCHEDULE_COLUMNS",
    "PLAYOFF_ROUND",
    "TEAM_SCHEDULE_COLUMNS",
    "build_elo_context",
    "build_schedule_context",
]
