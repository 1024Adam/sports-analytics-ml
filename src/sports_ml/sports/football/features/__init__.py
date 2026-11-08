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
from sports_ml.sports.football.features.pipeline import (
    FOOTBALL_STAGE1_CANDIDATE_FEATURE_COLS,
    FOOTBALL_STAGE2_CANDIDATE_FEATURE_COLS,
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    assert_zero_firewall_contamination,
    build_and_persist_football_gold_features,
    build_football_stage1_features,
    filter_forbidden_feature_columns,
    is_forbidden_feature_column,
    strip_forbidden_columns,
)
from sports_ml.sports.football.features.qb import (
    QB_FEATURE_COLUMNS,
    build_qb_elo_value_maps,
    build_qb_features,
    compose_qb_value,
    draft_capital_factor,
    qb_prior_rates,
)
from sports_ml.sports.football.features.schedule_context import (
    build_elo_context,
    build_schedule_context,
)
from sports_ml.sports.football.features.schedule_hooks import (
    FOOTBALL_SCHEDULE_FEATURE_COLUMNS,
    build_football_schedule_features,
)
from sports_ml.sports.football.features.team import (
    TEAM_METRICS,
    TeamFeatureConfig,
    build_roster_continuity,
    build_team_rolling_features,
    compute_adjusted_epa_ratings,
    prior_season_weight_from_continuity,
    team_feature_column_names,
)

__all__ = [
    "DEFAULT_KAPPA_QB",
    "FOOTBALL_SCHEDULE_FEATURE_COLUMNS",
    "FOOTBALL_STAGE1_CANDIDATE_FEATURE_COLS",
    "FOOTBALL_STAGE2_CANDIDATE_FEATURE_COLS",
    "FootballQBEloHook",
    "FootballVenueEloHook",
    "QB_FEATURE_COLUMNS",
    "STAGE1_FEATURE_COLUMNS",
    "STAGE2_FEATURE_COLUMNS",
    "TEAM_METRICS",
    "TeamFeatureConfig",
    "assert_zero_firewall_contamination",
    "build_and_persist_football_gold_features",
    "build_elo_context",
    "build_football_schedule_features",
    "build_football_stage1_features",
    "build_qb_elo_value_maps",
    "build_qb_features",
    "build_rest_days_lookup",
    "build_roster_continuity",
    "build_schedule_context",
    "build_team_rolling_features",
    "compose_qb_value",
    "compute_adjusted_epa_ratings",
    "compute_football_elo_ratings",
    "compute_qb_elo_delta",
    "draft_capital_factor",
    "filter_forbidden_feature_columns",
    "is_forbidden_feature_column",
    "prepare_football_elo_games",
    "prior_season_weight_from_continuity",
    "qb_prior_rates",
    "strip_forbidden_columns",
    "team_feature_column_names",
]
