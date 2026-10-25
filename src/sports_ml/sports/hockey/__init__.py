"""NHL Hockey plugin (`src/sports_ml/sports/hockey/`)."""

from sports_ml.sports.hockey.adapter import HockeyAdapter
from sports_ml.sports.hockey.constants import (
    NHL_TEAMS_32,
    SILVER_GAMES_SCHEMA,
    SILVER_GOALIE_GAME_LOGS_SCHEMA,
    SILVER_SKATER_GAME_LOGS_SCHEMA,
    SILVER_TEAM_GAME_LOGS_SCHEMA,
    TRICODE_ALIASES,
    ArenaMetadata,
    get_arena_metadata,
    normalize_tricode,
)

__all__ = [
    "ArenaMetadata",
    "HockeyAdapter",
    "NHL_TEAMS_32",
    "SILVER_GAMES_SCHEMA",
    "SILVER_GOALIE_GAME_LOGS_SCHEMA",
    "SILVER_SKATER_GAME_LOGS_SCHEMA",
    "SILVER_TEAM_GAME_LOGS_SCHEMA",
    "TRICODE_ALIASES",
    "get_arena_metadata",
    "normalize_tricode",
]
