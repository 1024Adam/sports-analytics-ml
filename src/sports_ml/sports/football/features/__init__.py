"""Football pre-game feature engineering package (`sports/football/features/`)."""

from __future__ import annotations

from sports_ml.sports.football.features.elo_hooks import (
    DEFAULT_KAPPA_QB,
    FootballQBEloHook,
    FootballVenueEloHook,
    build_rest_days_lookup,
    compute_football_elo_ratings,
    compute_qb_elo_delta,
    prepare_football_elo_games,
)
from sports_ml.sports.football.features.schedule_hooks import (
    FOOTBALL_SCHEDULE_FEATURE_COLUMNS,
    build_football_schedule_features,
)

__all__ = [
    "DEFAULT_KAPPA_QB",
    "FOOTBALL_SCHEDULE_FEATURE_COLUMNS",
    "FootballQBEloHook",
    "FootballVenueEloHook",
    "build_football_schedule_features",
    "build_rest_days_lookup",
    "compute_football_elo_ratings",
    "compute_qb_elo_delta",
    "prepare_football_elo_games",
]
