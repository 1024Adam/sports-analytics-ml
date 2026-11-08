"""Tests for Football constants (32 teams, stadiums, aliases) and `FootballAdapter`."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from sports_ml.core.contracts import SchemaValidationError, SportAdapter
from sports_ml.sports.football import (
    ALL_FOOTBALL_TEAMS,
    FOOTBALL_ELO_CONFIG,
    FOOTBALL_STADIUMS,
    NFL_DIVISIONS,
    NFL_TEAMS_32,
    FootballAdapter,
    franchise_codes,
    get_football_team_metadata,
    get_league_priors,
    get_stadium_metadata,
    gsis_id_to_int,
    is_neutral_game,
    normalize_football_team,
)
from sports_ml.sports.football.constants import (
    ROOF_TYPES,
    SURFACE_TYPES,
)


def _games(rows: list[dict[str, object]] | None = None) -> pl.DataFrame:
    base: list[dict[str, object]] = [
        {
            "season": 2023,
            "season_type": "REG",
            "week": 1,
            "game_id": 2023091000,
            "start_time_utc": datetime(2023, 9, 10, 17, 0, tzinfo=UTC),
            "home_team": "BUF",
            "away_team": "NYJ",
        },
        {
            "season": 2023,
            "season_type": "REG",
            "week": 1,
            "game_id": 2023091001,
            "start_time_utc": datetime(2023, 9, 10, 20, 25, tzinfo=UTC),
            "home_team": "KC",
            "away_team": "DEN",
        },
        {
            "season": 2023,
            "season_type": "REG",
            "week": 2,
            "game_id": 2023091700,
            "start_time_utc": datetime(2023, 9, 17, 17, 0, tzinfo=UTC),
            "home_team": "NYJ",
            "away_team": "KC",
        },
    ]
    return pl.DataFrame(rows if rows is not None else base)


def test_registry_has_32_teams_in_8_divisions() -> None:
    assert len(NFL_TEAMS_32) == 32
    assert set(ALL_FOOTBALL_TEAMS) == set(NFL_TEAMS_32)
    assert len(NFL_DIVISIONS) == 8
    assert all(len(teams) == 4 for teams in NFL_DIVISIONS.values())
    assert {t.conference for t in NFL_TEAMS_32.values()} == {"AFC", "NFC"}
    assert sum(1 for t in NFL_TEAMS_32.values() if t.conference == "AFC") == 16
    for team in NFL_TEAMS_32.values():
        assert get_stadium_metadata(team.primary_venue_id) is not None
        assert team.team in get_stadium_metadata(team.primary_venue_id).primary_teams  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("OAK", "LV"),
        ("SD", "LAC"),
        ("STL", "LA"),
        ("LAR", "LA"),
        ("SL", "LA"),
        ("WSH", "WAS"),
        ("WAS", "WAS"),
        ("JAC", "JAX"),
        ("ARZ", "ARI"),
        ("BLT", "BAL"),
        ("CLV", "CLE"),
        ("HST", "HOU"),
        (" kc ", "KC"),
        ("Oakland Raiders", "LV"),
        ("Washington Redskins", "WAS"),
        ("Kansas City Chiefs", "KC"),
    ],
)
def test_historical_aliases_resolve_to_current_franchise(alias: str, canonical: str) -> None:
    assert normalize_football_team(alias) == canonical
    meta = get_football_team_metadata(alias)
    assert meta is not None
    assert meta.team == canonical


def test_franchise_continuity_keeps_relocated_history() -> None:
    assert franchise_codes("OAK") == ("OAK", "LV")
    assert franchise_codes("SD") == ("SD", "LAC")
    assert franchise_codes("STL") == ("STL", "LA")
    assert franchise_codes("KC") == ("KC",)
    assert get_football_team_metadata("NOT_A_TEAM") is None


def test_every_stadium_has_valid_geo_enums_and_altitude() -> None:
    for venue in FOOTBALL_STADIUMS.values():
        assert -90.0 <= venue.latitude <= 90.0
        assert -180.0 <= venue.longitude <= 180.0
        ZoneInfo(venue.timezone)
        assert venue.roof in ROOF_TYPES
        assert venue.surface in SURFACE_TYPES
        assert math.isfinite(venue.altitude_m)
        assert venue.altitude_m >= 0.0
        assert venue.is_international == (venue.country != "US")
        if venue.is_international:
            assert venue.is_neutral_site

    domestic_alts = {
        v.venue_id: v.altitude_m for v in FOOTBALL_STADIUMS.values() if not v.is_international
    }
    assert max(domestic_alts, key=domestic_alts.__getitem__) == "DEN_MILE_HIGH"
    assert domestic_alts["DEN_MILE_HIGH"] > 1500.0


def test_international_neutral_and_super_bowl_venues() -> None:
    for venue_id in (
        "LON_WEMBLEY",
        "LON_TOTTENHAM",
        "MUC_ALLIANZ",
        "FRA_DEUTSCHE_BANK",
        "MEX_AZTECA",
        "SAO_CORINTHIANS",
        "MAD_BERNABEU",
    ):
        venue = FOOTBALL_STADIUMS[venue_id]
        assert venue.is_neutral_site
        assert venue.is_international
        assert is_neutral_game("REG", venue_id)

    assert not is_neutral_game("REG", "BUF_HIGHMARK")
    assert is_neutral_game("SB", "LA_SOFI")
    assert not FOOTBALL_STADIUMS["BUF_HIGHMARK"].is_neutral_site

    host_seasons = [s for v in FOOTBALL_STADIUMS.values() for s in v.super_bowl_seasons]
    assert sorted(host_seasons) == list(range(1998, 2026))


def test_elo_config_and_league_priors() -> None:
    assert FOOTBALL_ELO_CONFIG.base_rating == 1500.0
    assert FOOTBALL_ELO_CONFIG.k_factor == pytest.approx(20.0)
    assert FOOTBALL_ELO_CONFIG.home_field_advantage == pytest.approx(48.0)
    assert 1.0 - FOOTBALL_ELO_CONFIG.offseason_retention == pytest.approx(1.0 / 3.0)
    assert FOOTBALL_ELO_CONFIG.draw_param == 0.0
    assert FOOTBALL_ELO_CONFIG.playoff_k_multiplier > 1.0
    assert (
        get_league_priors(2000).points_per_team_game != get_league_priors(2020).points_per_team_game
    )
    with pytest.raises(ValueError, match="Unsupported football league"):
        get_league_priors(2023, "NCAAF")


def test_adapter_implements_protocol_and_normalizes_games() -> None:
    adapter = FootballAdapter()
    assert isinstance(adapter, SportAdapter)
    assert adapter.sport_slug == "football"
    assert adapter.topology == "head_to_head"

    validated = adapter.validate_event_schema(
        _games().with_columns(pl.Series("away_team", ["NYJ", "DEN", "KAN"]))
    )
    assert validated["sport"].unique().to_list() == ["football"]
    assert validated["league"].unique().to_list() == ["NFL"]
    assert validated["event_seq"].to_list() == [1, 1, 1]
    assert validated["away_team"].to_list()[-1] == "KC"
    assert validated["start_time_utc"].is_sorted()


def test_adapter_rejects_invalid_games() -> None:
    adapter = FootballAdapter()

    with pytest.raises(SchemaValidationError, match="reserved"):
        adapter.validate_event_schema(_games().with_columns(pl.lit("NCAAF").alias("league")))
    with pytest.raises(SchemaValidationError, match="league"):
        adapter.validate_event_schema(_games().with_columns(pl.lit("CFL").alias("league")))
    with pytest.raises(SchemaValidationError, match="event_seq"):
        adapter.validate_event_schema(_games().with_columns(pl.lit(2).alias("event_seq")))
    with pytest.raises(SchemaValidationError, match="Duplicate game_id"):
        adapter.validate_event_schema(
            _games().with_columns(pl.Series("game_id", [2023091000, 2023091000, 2023091700]))
        )
    with pytest.raises(SchemaValidationError, match="Preseason"):
        adapter.validate_event_schema(
            _games().with_columns(pl.Series("season_type", ["PRE", "REG", "REG"]))
        )
    with pytest.raises(SchemaValidationError, match="season_type"):
        adapter.validate_event_schema(
            _games().with_columns(pl.Series("season_type", ["XYZ", "REG", "REG"]))
        )
    with pytest.raises(SchemaValidationError, match="missing columns"):
        adapter.validate_event_schema(_games().drop("week"))

    # KC plays twice in (2023, week 1)
    two_games_one_week = _games().with_columns(
        pl.Series("week", [1, 1, 1]), pl.Series("home_team", ["BUF", "KC", "NYJ"])
    )
    with pytest.raises(SchemaValidationError, match="at most one game per"):
        adapter.validate_event_schema(
            two_games_one_week.with_columns(pl.Series("away_team", ["NYJ", "DEN", "KC"]))
        )


def test_adapter_team_logs_and_participants() -> None:
    adapter = FootballAdapter()
    team_logs = pl.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "season": [2023] * 4,
            "season_type": ["REG"] * 4,
            "week": [1, 1, 2, 2],
            "start_time_utc": [
                datetime(2023, 9, 10, 17, 0, tzinfo=UTC),
                datetime(2023, 9, 10, 17, 0, tzinfo=UTC),
                datetime(2023, 9, 17, 17, 0, tzinfo=UTC),
                datetime(2023, 9, 17, 17, 0, tzinfo=UTC),
            ],
            "team": ["BUF", "NYJ", "BUF", "OAK"],
            "opponent": ["NYJ", "BUF", "OAK", "BUF"],
            "is_home": [True, False, True, False],
        }
    )
    validated = adapter.validate_team_game_logs(team_logs)
    assert "LV" in validated["team"].to_list()

    with pytest.raises(SchemaValidationError, match="at most one game per"):
        adapter.validate_team_game_logs(team_logs.with_columns(pl.Series("week", [1, 1, 1, 1])))
    with pytest.raises(SchemaValidationError, match="Duplicate"):
        adapter.validate_team_game_logs(
            pl.concat([team_logs, team_logs.head(1)]).with_columns(
                pl.Series("week", [1, 1, 2, 2, 3])
            )
        )

    players = pl.DataFrame(
        {
            "game_id": [1, 1],
            "team": ["BUF", "BUF"],
            "player_id": ["00-0034857", "00-0033873"],
        }
    )
    out = adapter.validate_participant_schema(players)
    assert sorted(out["participant_id"].to_list()) == [33873, 34857]
    with pytest.raises(SchemaValidationError, match="Duplicate"):
        adapter.validate_participant_schema(pl.concat([players, players.head(1)]))


def test_gsis_id_to_int() -> None:
    assert gsis_id_to_int("00-0033873") == 33873
    assert gsis_id_to_int(33873) == 33873
    assert gsis_id_to_int("33873") == 33873
    with pytest.raises(ValueError, match="Invalid GSIS"):
        gsis_id_to_int("abc")
