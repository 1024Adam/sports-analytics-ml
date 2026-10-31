"""Baseball pre-game feature engineering package (`sports/baseball/features/`)."""

from __future__ import annotations

from sports_ml.sports.baseball.features.bullpen import build_bullpen_pregame_features
from sports_ml.sports.baseball.features.elo_hooks import (
    BaseballStarterEloHook,
    BaseballVenueEloHook,
    compute_starter_xfip_elo_delta,
)
from sports_ml.sports.baseball.features.lineup import (
    build_batter_rolling_features,
    build_platoon_lineup_strength_features,
    compute_platoon_adjusted_woba,
)
from sports_ml.sports.baseball.features.pipeline import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_BATTER_FEATURE_COLUMNS,
    STAGE2_PITCHER_FEATURE_COLUMNS,
    assert_zero_odds_contamination,
    filter_forbidden_market_columns,
    strip_market_columns_from_frame,
)
from sports_ml.sports.baseball.features.pitcher import (
    build_starting_pitcher_rolling_features,
    resolve_game_starting_pitcher_features,
)
from sports_ml.sports.baseball.features.player import (
    build_baseball_stage2_batter_features,
    build_baseball_stage2_pitcher_features,
)
from sports_ml.sports.baseball.features.team import (
    build_baseball_stage1_game_features,
    build_baseball_team_rolling_features,
)
from sports_ml.sports.baseball.features.translation import (
    TranslatedBatterPrior,
    TranslatedPitcherPrior,
    get_rookie_batter_prior,
    get_rookie_pitcher_prior,
    resolve_batter_prior,
    resolve_pitcher_prior,
    translate_mlb_to_npb_batter_prior,
    translate_mlb_to_npb_pitcher_prior,
    translate_npb_to_mlb_batter_prior,
    translate_npb_to_mlb_pitcher_prior,
)

__all__ = [
    "BaseballStarterEloHook",
    "BaseballVenueEloHook",
    "STAGE1_FEATURE_COLUMNS",
    "STAGE2_BATTER_FEATURE_COLUMNS",
    "STAGE2_PITCHER_FEATURE_COLUMNS",
    "TranslatedBatterPrior",
    "TranslatedPitcherPrior",
    "assert_zero_odds_contamination",
    "build_baseball_stage1_game_features",
    "build_baseball_stage2_batter_features",
    "build_baseball_stage2_pitcher_features",
    "build_baseball_team_rolling_features",
    "build_batter_rolling_features",
    "build_bullpen_pregame_features",
    "build_platoon_lineup_strength_features",
    "build_starting_pitcher_rolling_features",
    "compute_platoon_adjusted_woba",
    "compute_starter_xfip_elo_delta",
    "filter_forbidden_market_columns",
    "get_rookie_batter_prior",
    "get_rookie_pitcher_prior",
    "resolve_batter_prior",
    "resolve_game_starting_pitcher_features",
    "resolve_pitcher_prior",
    "strip_market_columns_from_frame",
    "translate_mlb_to_npb_batter_prior",
    "translate_mlb_to_npb_pitcher_prior",
    "translate_npb_to_mlb_batter_prior",
    "translate_npb_to_mlb_pitcher_prior",
]
