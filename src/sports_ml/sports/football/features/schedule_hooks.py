"""Weekly-cadence schedule hooks for NFL team logs (`sports/football/features/schedule_hooks.py`).

Wraps the sport-agnostic `core/schedule.py` helpers and adds NFL-specific rules
(Thursday/Monday kickoffs, international travel, body-clock kickoff hour). Every
feature uses only scheduled kickoff metadata and the entity's *previous* appearance.
"""

from __future__ import annotations

import polars as pl

from sports_ml.core.schedule import (
    compute_schedule_features,
    compute_weekly_cadence_features,
    kickoff_local_hour_vs_body_clock,
)
from sports_ml.sports.football.constants import (
    FOOTBALL_STADIUMS,
    MINI_BYE_MIN_DAYS,
    POST_BYE_MIN_DAYS,
    SHORT_WEEK_MAX_DAYS,
    get_stadium_metadata,
)

_EASTERN = "America/New_York"
_THURSDAY = 4
_MONDAY = 1

FOOTBALL_SCHEDULE_FEATURE_COLUMNS: tuple[str, ...] = (
    "rest_hours",
    "days_rest",
    "is_short_week",
    "is_post_bye",
    "is_post_mini_bye",
    "prior_game_was_overtime",
    "is_thursday_game",
    "is_monday_game",
    "is_thursday_short_week",
    "kickoff_local_hour",
    "kickoff_local_hour_vs_body_clock",
    "is_international_travel",
    "travel_km_prev_game",
    "tz_shift_hours",
)


def build_football_schedule_features(team_logs: pl.DataFrame) -> pl.DataFrame:
    """Annotate a one-row-per-team-game frame with NFL cadence, kickoff, and travel features.

    Required columns: `start_time_utc`, `season`, `team`, `venue_id`, and `game_id` (or
    `event_id`). Optional: `overtime` (used only for the *prior* game flag).
    """
    working = team_logs
    if "event_id" not in working.columns:
        working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))

    venue_rows = [
        {
            "venue_id": v.venue_id,
            "venue_lat": v.latitude,
            "venue_lon": v.longitude,
            "venue_timezone": v.timezone,
            "venue_is_international": v.is_international,
        }
        for v in FOOTBALL_STADIUMS.values()
    ]
    venue_df = pl.DataFrame(venue_rows)
    working = working.join(venue_df, on="venue_id", how="left")

    cadence = compute_weekly_cadence_features(
        working,
        entity_col="team",
        overtime_col="overtime",
        short_week_max_days=SHORT_WEEK_MAX_DAYS,
        mini_bye_min_days=MINI_BYE_MIN_DAYS,
        post_bye_min_days=POST_BYE_MIN_DAYS,
    )

    travel = compute_schedule_features(
        cadence.select(
            "event_id",
            "event_seq",
            "season",
            "team",
            "start_time_utc",
            "venue_lat",
            "venue_lon",
            "venue_timezone",
        ),
        entity_col="team",
    ).select("event_id", "event_seq", "team", "travel_km_prev_game", "tz_shift_hours")
    out = cadence.join(travel, on=["event_id", "event_seq", "team"], how="left")

    eastern_start = pl.col("start_time_utc").dt.convert_time_zone(_EASTERN)
    out = out.with_columns(
        (eastern_start.dt.weekday() == _THURSDAY).alias("is_thursday_game"),
        (eastern_start.dt.weekday() == _MONDAY).alias("is_monday_game"),
        pl.col("venue_is_international").fill_null(False).alias("is_international_travel"),
    ).with_columns(
        (pl.col("is_thursday_game") & pl.col("is_short_week")).alias("is_thursday_short_week")
    )

    local_hours: list[float | None] = []
    body_clock_hours: list[float | None] = []
    for row in out.select("start_time_utc", "venue_timezone", "team").iter_rows(named=True):
        start = row["start_time_utc"]
        venue_tz = row["venue_timezone"]
        home_venue = get_stadium_metadata(str(row["team"]))
        local_hours.append(
            kickoff_local_hour_vs_body_clock(start, venue_tz) if venue_tz is not None else None
        )
        body_clock_hours.append(
            kickoff_local_hour_vs_body_clock(start, home_venue.timezone)
            if home_venue is not None
            else None
        )
    out = out.with_columns(
        pl.Series("kickoff_local_hour", local_hours, dtype=pl.Float64),
        pl.Series("kickoff_local_hour_vs_body_clock", body_clock_hours, dtype=pl.Float64),
    )
    return out.drop("venue_lat", "venue_lon", "venue_is_international")
