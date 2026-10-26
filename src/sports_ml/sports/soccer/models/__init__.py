"""EPL Soccer models package (`src/sports_ml/sports/soccer/models/`)."""

from sports_ml.sports.soccer.models.predictor import (
    SoccerMatchupPrediction,
    SoccerMatchupPredictor,
    SoccerPlayerMatchupProjections,
    build_and_persist_soccer_gold_features,
    score_soccer_prediction_ledger,
)
from sports_ml.sports.soccer.models.stage1_match import (
    SOCCER_STAGE1_CANDIDATE_FEATURE_COLS,
    SoccerStage1BacktestResult,
    Stage1SoccerMatchModel,
    compute_dixon_coles_1x2_probabilities,
    run_soccer_stage1_walk_forward_backtest,
)
from sports_ml.sports.soccer.models.stage2_player import (
    SOCCER_STAGE2_OUTFIELD_FEATURE_COLS,
    Stage2SoccerPlayerModel,
    reconcile_outfield_team_goals,
)

__all__ = [
    "SOCCER_STAGE1_CANDIDATE_FEATURE_COLS",
    "SOCCER_STAGE2_OUTFIELD_FEATURE_COLS",
    "SoccerMatchupPrediction",
    "SoccerMatchupPredictor",
    "SoccerPlayerMatchupProjections",
    "SoccerStage1BacktestResult",
    "Stage1SoccerMatchModel",
    "Stage2SoccerPlayerModel",
    "build_and_persist_soccer_gold_features",
    "compute_dixon_coles_1x2_probabilities",
    "reconcile_outfield_team_goals",
    "run_soccer_stage1_walk_forward_backtest",
    "score_soccer_prediction_ledger",
]
