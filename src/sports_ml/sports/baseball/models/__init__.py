"""Baseball Stage 1 and Stage 2 models package (`sports/baseball/models/`)."""

from __future__ import annotations

from sports_ml.sports.baseball.models.predictor import (
    BaseballMatchupPrediction,
    BaseballMatchupPredictor,
    BaseballPlayerMatchupProjections,
    build_and_persist_baseball_gold_features,
    score_baseball_prediction_ledger,
)
from sports_ml.sports.baseball.models.stage1_game import (
    BaseballStage1BacktestResult,
    Stage1BaseballGameModel,
    compute_negative_binomial_game_probabilities,
    run_baseball_stage1_walk_forward_backtest,
)
from sports_ml.sports.baseball.models.stage2_player import (
    Stage2BaseballPlayerModel,
    reconcile_baseball_stage2_projections,
)

__all__ = [
    "BaseballMatchupPrediction",
    "BaseballMatchupPredictor",
    "BaseballPlayerMatchupProjections",
    "BaseballStage1BacktestResult",
    "Stage1BaseballGameModel",
    "Stage2BaseballPlayerModel",
    "build_and_persist_baseball_gold_features",
    "compute_negative_binomial_game_probabilities",
    "reconcile_baseball_stage2_projections",
    "run_baseball_stage1_walk_forward_backtest",
    "score_baseball_prediction_ledger",
]
