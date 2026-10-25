"""Schedule, rest, density, travel distance, and timezone shift calculators (`core/schedule.py`).

Handles back-to-backs (`is_b2b`), same-day doubleheader turnarounds (`event_seq`),
`k`-in-`m` night density (`games_last_4d`, `games_last_6d`, `is_3_in_4`, `is_4_in_6`),
geodesic haversine distance, and timezone shifts across month/year boundaries.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

from sports_ml.core.temporal import sort_events_chronologically

EARTH_RADIUS_KM: Final[float] = 6371.0088


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute geodesic great-circle distance in kilometers between two coordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return float(EARTH_RADIUS_KM * c)


def timezone_shift_hours(
    tz_from: str,
    tz_to: str,
    reference_utc: datetime | None = None,
) -> float:
    """Compute UTC offset difference in hours (`offset(tz_to) - offset(tz_from)`)."""
    ref = reference_utc or datetime(2025, 1, 15, 12, 0, tzinfo=ZoneInfo("UTC"))
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=ZoneInfo("UTC"))

    offset_from = ref.astimezone(ZoneInfo(tz_from)).utcoffset()
    offset_to = ref.astimezone(ZoneInfo(tz_to)).utcoffset()
    sec_from = offset_from.total_seconds() if offset_from is not None else 0.0
    sec_to = offset_to.total_seconds() if offset_to is not None else 0.0
    return float((sec_to - sec_from) / 3600.0)


def compute_schedule_features(
    df: pl.DataFrame,
    *,
    entity_col: str = "team",
    date_col: str = "game_date",
    max_rest_days: int = 7,
) -> pl.DataFrame:
    """Annotate chronological entity logs with rest days, B2B, doubleheader, density & travel.

    Schedule features depend strictly on pre-game schedule metadata (`start_time_utc`,
    `event_seq`, `game_date`, `venue_lat`, `venue_lon`, `venue_timezone`) at `<= t`
    and never inspect post-game scores or future dates.
    """
    sorted_df = sort_events_chronologically(df, extra_tiebreakers=[entity_col])
    if sorted_df.is_empty():
        return sorted_df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("rest_days"),
            pl.lit(None, dtype=pl.Boolean).alias("is_b2b"),
            pl.lit(None, dtype=pl.Boolean).alias("is_same_day_doubleheader"),
            pl.lit(None, dtype=pl.Float64).alias("turnaround_hours"),
            pl.lit(None, dtype=pl.Int32).alias("games_last_4d"),
            pl.lit(None, dtype=pl.Int32).alias("games_last_6d"),
            pl.lit(None, dtype=pl.Boolean).alias("is_3_in_4"),
            pl.lit(None, dtype=pl.Boolean).alias("is_4_in_6"),
            pl.lit(None, dtype=pl.Float64).alias("travel_km_prev_game"),
            pl.lit(None, dtype=pl.Float64).alias("travel_km_last_3d"),
            pl.lit(None, dtype=pl.Float64).alias("tz_shift_hours"),
            pl.lit(None, dtype=pl.Boolean).alias("is_b2b_with_travel"),
        )

    rows = sorted_df.to_dicts()
    n_rows = len(rows)

    rest_days_out = [max_rest_days] * n_rows
    is_b2b_out = [False] * n_rows
    is_dh_out = [False] * n_rows
    turnaround_out: list[float | None] = [None] * n_rows
    games_4d_out = [1] * n_rows
    games_6d_out = [1] * n_rows
    is_3_in_4_out = [False] * n_rows
    is_4_in_6_out = [False] * n_rows
    travel_prev_out = [0.0] * n_rows
    travel_3d_out = [0.0] * n_rows
    tz_shift_out = [0.0] * n_rows
    is_b2b_travel_out = [False] * n_rows

    # Per-(entity, season) state history: list of (event_date, start_time_utc, lat, lon, tz, leg_km)
    history: dict[
        tuple[str, int], list[tuple[date, datetime, float | None, float | None, str | None, float]]
    ] = {}

    for idx, row in enumerate(rows):
        entity = str(row[entity_col])
        season = int(row.get("season") or 0)
        start_utc: datetime = row["start_time_utc"]
        ev_date: date = (
            row[date_col] if date_col in row and row[date_col] is not None else start_utc.date()
        )
        lat = float(row["venue_lat"]) if row.get("venue_lat") is not None else None
        lon = float(row["venue_lon"]) if row.get("venue_lon") is not None else None
        tz = str(row["venue_timezone"]) if row.get("venue_timezone") is not None else None

        key = (entity, season)
        ent_hist = history.setdefault(key, [])

        if not ent_hist:
            rest_days_out[idx] = max_rest_days
            is_b2b_out[idx] = False
            is_dh_out[idx] = False
            turnaround_out[idx] = None
            leg_km = 0.0
            tz_delta = 0.0
        else:
            prev_date, prev_utc, prev_lat, prev_lon, prev_tz, _ = ent_hist[-1]
            raw_days = (ev_date - prev_date).days
            elapsed_hours = (start_utc - prev_utc).total_seconds() / 3600.0
            turnaround_out[idx] = float(elapsed_hours)

            if raw_days <= 0:
                rest_days_out[idx] = 0
                is_b2b_out[idx] = False
                is_dh_out[idx] = True
            else:
                rest_days_out[idx] = min(raw_days, max_rest_days)
                is_b2b_out[idx] = raw_days == 1
                is_dh_out[idx] = False

            if (
                lat is not None
                and lon is not None
                and prev_lat is not None
                and prev_lon is not None
            ):
                leg_km = haversine_distance_km(prev_lat, prev_lon, lat, lon)
            else:
                leg_km = 0.0

            if tz is not None and prev_tz is not None:
                tz_delta = timezone_shift_hours(prev_tz, tz, reference_utc=start_utc)
            else:
                tz_delta = 0.0

        travel_prev_out[idx] = leg_km
        tz_shift_out[idx] = tz_delta
        is_b2b_travel_out[idx] = bool(is_b2b_out[idx] and leg_km > 1.0)

        # Record current appearance in entity history (strictly <= t)
        ent_hist.append((ev_date, start_utc, lat, lon, tz, leg_km))

        # Count appearances in last 4 days [ev_date - 3d, ev_date] and 6 days [ev_date - 5d]
        g_4d = sum(1 for (d, _, _, _, _, _) in ent_hist if 0 <= (ev_date - d).days <= 3)
        g_6d = sum(1 for (d, _, _, _, _, _) in ent_hist if 0 <= (ev_date - d).days <= 5)
        km_3d = sum(km for (d, _, _, _, _, km) in ent_hist if 0 <= (ev_date - d).days <= 3)

        games_4d_out[idx] = g_4d
        games_6d_out[idx] = g_6d
        is_3_in_4_out[idx] = g_4d >= 3
        is_4_in_6_out[idx] = g_6d >= 4
        travel_3d_out[idx] = float(km_3d)

    return sorted_df.with_columns(
        pl.Series("rest_days", rest_days_out, dtype=pl.Int32),
        pl.Series("is_b2b", is_b2b_out, dtype=pl.Boolean),
        pl.Series("is_same_day_doubleheader", is_dh_out, dtype=pl.Boolean),
        pl.Series("turnaround_hours", turnaround_out, dtype=pl.Float64),
        pl.Series("games_last_4d", games_4d_out, dtype=pl.Int32),
        pl.Series("games_last_6d", games_6d_out, dtype=pl.Int32),
        pl.Series("is_3_in_4", is_3_in_4_out, dtype=pl.Boolean),
        pl.Series("is_4_in_6", is_4_in_6_out, dtype=pl.Boolean),
        pl.Series("travel_km_prev_game", travel_prev_out, dtype=pl.Float64),
        pl.Series("travel_km_last_3d", travel_3d_out, dtype=pl.Float64),
        pl.Series("tz_shift_hours", tz_shift_out, dtype=pl.Float64),
        pl.Series("is_b2b_with_travel", is_b2b_travel_out, dtype=pl.Boolean),
    )
