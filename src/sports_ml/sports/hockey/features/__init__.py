"""NHL Hockey pre-game feature engineering (`sports/hockey/features/`)."""

from sports_ml.sports.hockey.features.goalie import (
    build_goalie_rolling_features,
    resolve_team_starting_goalie_features,
)
from sports_ml.sports.hockey.features.player import build_stage2_player_features
from sports_ml.sports.hockey.features.roster import (
    build_roster_strength_features,
    build_skater_rolling_features,
)
from sports_ml.sports.hockey.features.team import (
    build_stage1_game_features,
    build_team_rolling_features,
    compute_special_teams_net_edge,
)

__all__ = [
    "build_goalie_rolling_features",
    "build_roster_strength_features",
    "build_skater_rolling_features",
    "build_stage1_game_features",
    "build_stage2_player_features",
    "build_team_rolling_features",
    "compute_special_teams_net_edge",
    "resolve_team_starting_goalie_features",
]
