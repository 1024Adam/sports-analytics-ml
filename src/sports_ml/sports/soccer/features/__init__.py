"""EPL Soccer feature engineering package (`src/sports_ml/sports/soccer/features/`)."""

from sports_ml.sports.soccer.features.goalkeeper import (
    DEFAULT_GK_WINDOWS,
    GK_SHRINKAGE_K,
    PRIOR_GK_CLEAN_SHEET_RATE,
    PRIOR_GK_GSAA_PER_90,
    PRIOR_GK_SAVE_PCT,
    GoalkeeperStarterEloHook,
    build_goalkeeper_rolling_features,
    resolve_team_starting_gk_features,
)
from sports_ml.sports.soccer.features.pipeline import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    assert_zero_odds_contamination,
    filter_forbidden_market_columns,
    strip_market_columns_from_frame,
)
from sports_ml.sports.soccer.features.player import (
    DEFAULT_SOCCER_PLAYER_WINDOWS,
    build_soccer_stage2_player_features,
)
from sports_ml.sports.soccer.features.roster import (
    OUTFIELD_SHRINKAGE_K,
    build_outfield_rolling_features,
    build_soccer_roster_strength_features,
)
from sports_ml.sports.soccer.features.team import (
    DEFAULT_SOCCER_EWMA_ALPHA,
    DEFAULT_SOCCER_TEAM_WINDOWS,
    SOCCER_ELO_CONFIG,
    SOCCER_TEAM_SHRINKAGE_K,
    build_soccer_stage1_match_features,
    build_soccer_team_rolling_features,
)

__all__ = [
    "DEFAULT_GK_WINDOWS",
    "DEFAULT_SOCCER_EWMA_ALPHA",
    "DEFAULT_SOCCER_PLAYER_WINDOWS",
    "DEFAULT_SOCCER_TEAM_WINDOWS",
    "GK_SHRINKAGE_K",
    "GoalkeeperStarterEloHook",
    "OUTFIELD_SHRINKAGE_K",
    "PRIOR_GK_CLEAN_SHEET_RATE",
    "PRIOR_GK_GSAA_PER_90",
    "PRIOR_GK_SAVE_PCT",
    "SOCCER_ELO_CONFIG",
    "SOCCER_TEAM_SHRINKAGE_K",
    "STAGE1_FEATURE_COLUMNS",
    "STAGE2_FEATURE_COLUMNS",
    "assert_zero_odds_contamination",
    "build_goalkeeper_rolling_features",
    "build_outfield_rolling_features",
    "build_soccer_roster_strength_features",
    "build_soccer_stage1_match_features",
    "build_soccer_stage2_player_features",
    "build_soccer_team_rolling_features",
    "filter_forbidden_market_columns",
    "resolve_team_starting_gk_features",
    "strip_market_columns_from_frame",
]
