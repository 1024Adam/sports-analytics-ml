"""MLB & NPB Baseball plugin (`src/sports_ml/sports/baseball/`)."""

from __future__ import annotations

from sports_ml.sports.baseball.adapter import (
    BaseballAdapter,
    outs_to_true_innings,
    parse_baseball_ip_to_outs,
)
from sports_ml.sports.baseball.constants import (
    ALL_BASEBALL_CLUBS,
    BASEBALL_BALLPARKS,
    MLB_CLUBS_30,
    NPB_CLUBS_12,
    BallparkMetadata,
    BaseballClubMetadata,
    BaseballLeague,
    get_ballpark_metadata,
    get_baseball_club_metadata,
    normalize_baseball_team,
    resolve_team_league,
)

__all__ = [
    "ALL_BASEBALL_CLUBS",
    "BASEBALL_BALLPARKS",
    "BaseballAdapter",
    "BaseballClubMetadata",
    "BaseballLeague",
    "BallparkMetadata",
    "MLB_CLUBS_30",
    "NPB_CLUBS_12",
    "get_ballpark_metadata",
    "get_baseball_club_metadata",
    "normalize_baseball_team",
    "outs_to_true_innings",
    "parse_baseball_ip_to_outs",
    "resolve_team_league",
]
