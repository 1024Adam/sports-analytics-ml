"""NFL Football plugin (`src/sports_ml/sports/football/`)."""

from __future__ import annotations

from sports_ml.sports.football.adapter import FootballAdapter, gsis_id_to_int
from sports_ml.sports.football.constants import (
    ALL_FOOTBALL_TEAMS,
    FOOTBALL_ELO_CONFIG,
    FOOTBALL_STADIUMS,
    NFL_DIVISIONS,
    NFL_TEAMS_32,
    FootballStadiumMetadata,
    FootballTeamMetadata,
    franchise_codes,
    get_football_team_metadata,
    get_league_priors,
    get_stadium_metadata,
    is_neutral_game,
    normalize_football_team,
    resolve_team_league,
)

__all__ = [
    "ALL_FOOTBALL_TEAMS",
    "FOOTBALL_ELO_CONFIG",
    "FOOTBALL_STADIUMS",
    "FootballAdapter",
    "FootballStadiumMetadata",
    "FootballTeamMetadata",
    "NFL_DIVISIONS",
    "NFL_TEAMS_32",
    "franchise_codes",
    "get_football_team_metadata",
    "get_league_priors",
    "get_stadium_metadata",
    "gsis_id_to_int",
    "is_neutral_game",
    "normalize_football_team",
    "resolve_team_league",
]
