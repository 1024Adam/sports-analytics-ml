"""EPL Soccer plugin (`src/sports_ml/sports/soccer/`)."""

from sports_ml.sports.soccer.adapter import SoccerAdapter
from sports_ml.sports.soccer.constants import (
    EPL_CLUBS,
    SILVER_GK_MATCH_LOGS_SCHEMA,
    SILVER_MATCHES_SCHEMA,
    SILVER_OUTFIELD_MATCH_LOGS_SCHEMA,
    SILVER_TEAM_MATCH_LOGS_SCHEMA,
    SOCCER_CLUB_ALIASES,
    StadiumMetadata,
    get_stadium_metadata,
    normalize_soccer_club,
)

__all__ = [
    "EPL_CLUBS",
    "SILVER_GK_MATCH_LOGS_SCHEMA",
    "SILVER_MATCHES_SCHEMA",
    "SILVER_OUTFIELD_MATCH_LOGS_SCHEMA",
    "SILVER_TEAM_MATCH_LOGS_SCHEMA",
    "SOCCER_CLUB_ALIASES",
    "SoccerAdapter",
    "StadiumMetadata",
    "get_stadium_metadata",
    "normalize_soccer_club",
]
