"""Schedule, rest, density, travel distance, and timezone shift calculators (`core/schedule.py`).

Handles back-to-backs (`is_b2b`), same-day doubleheader turnarounds (`event_seq`),
`k`-in-`m` night density, geodesic haversine distance, and timezone shifts.
Full implementation scheduled for Milestone 1.
"""

from __future__ import annotations

import polars as pl


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Compute geodesic great-circle distance in kilometers between two coordinates."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")


def timezone_shift_hours(tz_from: str, tz_to: str) -> float:
    """Compute UTC offset difference in hours between two IANA timezones."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")


def compute_schedule_features(df: pl.DataFrame) -> pl.DataFrame:
    """Annotate chronological entity logs with rest days, B2B, density, and travel features."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")
