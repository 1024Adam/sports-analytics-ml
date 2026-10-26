"""Tests for EPL club constants, cross-source alias normalization, and `SoccerAdapter`."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from sports_ml.core.contracts import SportAdapter
from sports_ml.core.temporal import pregame_exposure_rate, sort_events_chronologically
from sports_ml.sports.soccer import (
    EPL_CLUBS,
    SoccerAdapter,
    get_stadium_metadata,
    normalize_soccer_club,
)


@pytest.mark.parametrize(
    ("raw_name", "expected_code"),
    [
        ("Arsenal", "ARS"),
        ("ARS", "ARS"),
        ("Man City", "MCI"),
        ("Manchester City", "MCI"),
        ("Man Utd", "MUN"),
        ("Manchester United", "MUN"),
        ("Spurs", "TOT"),
        ("Tottenham Hotspur", "TOT"),
        ("Nott'm Forest", "NFO"),
        ("Nottingham Forest", "NFO"),
        ("Wolves", "WOL"),
        ("Wolverhampton Wanderers", "WOL"),
        ("Brighton", "BHA"),
        ("Brighton & Hove Albion", "BHA"),
        ("Ipswich", "IPS"),
        ("Ipswich Town", "IPS"),
        ("Leicester", "LEI"),
        ("Southampton", "SOU"),
        ("Newcastle United", "NEW"),
        ("West Ham", "WHU"),
    ],
)
def test_normalize_soccer_club_aliases(raw_name: str, expected_code: str) -> None:
    """Verify FPL, Understat, and Football-Data club names map to canonical 3-letter codes."""
    code = normalize_soccer_club(raw_name)
    assert code == expected_code
    assert code in EPL_CLUBS
    meta = get_stadium_metadata(raw_name)
    assert meta is not None
    assert meta.team == expected_code
    assert meta.timezone == "Europe/London"


def test_soccer_adapter_protocol_and_schema_validation() -> None:
    """Verify SoccerAdapter satisfies SportAdapter and normalizes match and participant frames."""
    adapter = SoccerAdapter()
    assert isinstance(adapter, SportAdapter)
    assert adapter.sport_slug == "soccer"
    assert adapter.topology == "head_to_head"

    matches_df = pl.DataFrame(
        {
            "season": [20242025, 20242025],
            "match_id": [1002, 1001],
            "start_time_utc": [
                datetime(2024, 8, 17, 16, 30, tzinfo=UTC),
                datetime(2024, 8, 16, 19, 0, tzinfo=UTC),
            ],
            "home_team": ["West Ham", "Man Utd"],
            "away_team": ["Aston Villa", "Fulham"],
        }
    )
    validated_events = adapter.validate_event_schema(matches_df)
    assert validated_events["event_id"].to_list() == [1001, 1002]
    assert validated_events["sport"].to_list() == ["soccer", "soccer"]
    assert validated_events["home_team"].to_list() == ["MUN", "WHU"]
    assert validated_events["away_team"].to_list() == ["FUL", "AVL"]

    players_df = pl.DataFrame(
        {
            "match_id": [1001, 1001],
            "start_time_utc": [
                datetime(2024, 8, 16, 19, 0, tzinfo=UTC),
                datetime(2024, 8, 16, 19, 0, tzinfo=UTC),
            ],
            "team": ["Man Utd", "Fulham"],
            "player_id": [10, 20],
            "minutes": [90.0, 90.0],
            "npxg": [0.45, 0.20],
        }
    )
    validated_players = adapter.validate_participant_schema(players_df)
    assert validated_players["entity_id"].to_list() == ["FUL", "MUN"]
    assert validated_players["participant_id"].to_list() == [20, 10]


def test_per_90_pregame_exposure_rate_is_leak_free() -> None:
    """Verify `pregame_exposure_rate(..., scale_factor=90.0)` computes leak-free per-90 rates."""
    df = pl.DataFrame(
        {
            "event_id": [1, 2, 3],
            "event_seq": [1, 1, 1],
            "start_time_utc": [
                datetime(2024, 8, 17, 14, 0, tzinfo=UTC),
                datetime(2024, 8, 24, 14, 0, tzinfo=UTC),
                datetime(2024, 8, 31, 14, 0, tzinfo=UTC),
            ],
            "player_id": [9, 9, 9],
            "npxg": [0.50, 0.25, 5.00],
            "minutes": [90.0, 45.0, 90.0],
        }
    )
    sorted_df = sort_events_chronologically(df)
    with_rate = sorted_df.with_columns(
        pregame_exposure_rate(
            "npxg",
            "minutes",
            window=5,
            group_by="player_id",
            scale_factor=90.0,
        ).alias("npxg_per_90")
    )

    assert with_rate["npxg_per_90"][0] is None
    assert with_rate["npxg_per_90"][1] == pytest.approx(0.50)
    # Match 3 pre-match rate uses Match 1 (0.50 in 90m) + Match 2 (0.25 in 45m) = 0.75 in 135m
    assert with_rate["npxg_per_90"][2] == pytest.approx(90.0 * (0.75 / 135.0))
