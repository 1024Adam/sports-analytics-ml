"""Schedule, B2B, doubleheader, density, and travel tests (`tests/core/test_schedule.py`)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from sports_ml.core.schedule import (
    compute_schedule_features,
    haversine_distance_km,
    timezone_shift_hours,
)


def test_haversine_and_timezone_shift() -> None:
    """Verify geodesic haversine distance and IANA timezone shifts."""
    # Boston (TD Garden) to Toronto (Scotiabank Arena) ~ 690 km
    dist_bos_tor = haversine_distance_km(42.3662, -71.0621, 43.6435, -79.3791)
    assert 680.0 <= dist_bos_tor <= 705.0
    assert haversine_distance_km(42.3662, -71.0621, 42.3662, -71.0621) == pytest.approx(0.0)

    ref_utc = datetime(2025, 1, 15, 20, 0, tzinfo=UTC)
    assert timezone_shift_hours(
        "America/New_York", "America/Los_Angeles", ref_utc
    ) == pytest.approx(-3.0)
    assert timezone_shift_hours(
        "America/Los_Angeles", "America/New_York", ref_utc
    ) == pytest.approx(3.0)


def test_b2b_3in4_month_year_boundaries_and_doubleheaders() -> None:
    """Verify B2B across month/year boundaries, 3-in-4 density, and same-day doubleheaders."""
    schedule_df = pl.DataFrame(
        {
            "season": [20242025] * 6,
            "event_id": [1, 2, 3, 4, 5, 6],
            "event_seq": [1, 1, 1, 1, 1, 2],
            "team": ["BOS"] * 6,
            "game_date": [
                date(2024, 10, 31),  # Game 1 (Season opener -> rest_days=7)
                date(2024, 11, 1),  # Game 2 (Month boundary B2B -> rest_days=1, is_b2b=True)
                date(2024, 11, 3),  # Game 3 (3rd game in 4 days -> is_3_in_4=True)
                date(2024, 12, 31),  # Game 4 (New Year's Eve after long break -> rest_days=7)
                date(2025, 1, 1),  # Game 5 (Year boundary B2B -> rest_days=1, is_b2b=True)
                date(2025, 1, 1),  # Game 6 (Same-day doubleheader Game 2 -> rest_days=0)
            ],
            "start_time_utc": [
                datetime(2024, 10, 31, 23, 0, tzinfo=UTC),
                datetime(2024, 11, 1, 23, 0, tzinfo=UTC),
                datetime(2024, 11, 3, 23, 0, tzinfo=UTC),
                datetime(2024, 12, 31, 23, 0, tzinfo=UTC),
                datetime(2025, 1, 1, 18, 0, tzinfo=UTC),
                datetime(2025, 1, 1, 22, 30, tzinfo=UTC),
            ],
            "venue_lat": [42.3662, 43.6435, 45.4961, 42.3662, 40.7505, 40.7505],
            "venue_lon": [-71.0621, -79.3791, -73.5693, -71.0621, -73.9934, -73.9934],
            "venue_timezone": [
                "America/New_York",
                "America/Toronto",
                "America/Toronto",
                "America/New_York",
                "America/New_York",
                "America/New_York",
            ],
        }
    )

    feat = compute_schedule_features(schedule_df)

    # Opener
    assert feat["rest_days"][0] == 7
    assert feat["is_b2b"][0] is False

    # Oct 31 -> Nov 1 B2B with travel (BOS -> TOR)
    assert feat["rest_days"][1] == 1
    assert feat["is_b2b"][1] is True
    assert feat["is_b2b_with_travel"][1] is True
    assert feat["travel_km_prev_game"][1] > 600.0

    # Nov 3: 3rd game in 4 days (Oct 31, Nov 1, Nov 3)
    assert feat["games_last_4d"][2] == 3
    assert feat["is_3_in_4"][2] is True

    # Dec 31 -> Jan 1 year boundary B2B
    assert feat["rest_days"][4] == 1
    assert feat["is_b2b"][4] is True

    # Jan 1 Game 2 (same-day doubleheader)
    assert feat["rest_days"][5] == 0
    assert feat["is_b2b"][5] is False
    assert feat["is_same_day_doubleheader"][5] is True
    assert feat["turnaround_hours"][5] == pytest.approx(4.5)
    assert feat["travel_km_prev_game"][5] == pytest.approx(0.0)
