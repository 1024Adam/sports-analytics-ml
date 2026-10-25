"""Two-Stage NHL Hockey prediction models and matchup service (`sports/hockey/models/`)."""

from sports_ml.sports.hockey.models.predictor import (
    HockeyMatchupPredictor,
    MatchupPrediction,
    PlayerMatchupProjections,
    build_and_persist_gold_features,
    score_prediction_ledger,
)
from sports_ml.sports.hockey.models.stage1_game import (
    Stage1BacktestResult,
    Stage1GameOutcomeModel,
    run_stage1_walk_forward_backtest,
)
from sports_ml.sports.hockey.models.stage2_player import (
    Stage2PlayerProjectionModel,
    reconcile_skater_team_goals,
)

__all__ = [
    "HockeyMatchupPredictor",
    "MatchupPrediction",
    "PlayerMatchupProjections",
    "Stage1BacktestResult",
    "Stage1GameOutcomeModel",
    "Stage2PlayerProjectionModel",
    "build_and_persist_gold_features",
    "reconcile_skater_team_goals",
    "run_stage1_walk_forward_backtest",
    "score_prediction_ledger",
]
