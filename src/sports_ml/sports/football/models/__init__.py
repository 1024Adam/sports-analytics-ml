"""Football Stage 1 models package (`sports/football/models/`)."""

from __future__ import annotations

from sports_ml.sports.football.models.predictor import (
    FootballLedgerScore,
    FootballMatchupPrediction,
    FootballMatchupPredictor,
    FootballQBCard,
    score_football_prediction_ledger,
)
from sports_ml.sports.football.models.stage1_game import (
    BootstrapInterval,
    FitBoundaryAudit,
    FootballStage1BacktestResult,
    ScoreMetrics,
    SliceReport,
    Stage1FootballGameModel,
    era_home_win_rates,
    run_football_stage1_walk_forward_backtest,
    swap_home_away_features,
)
from sports_ml.sports.football.models.stage2_player import (
    Stage2FootballPlayerModel,
    reconcile_football_stage2_projections,
)
from sports_ml.sports.football.models.training import (
    FootballTrainingResult,
    compute_validation_metrics,
    evaluate_football_stage2_holdout,
    load_football_stage1_bundle,
    train_football_stage1_bundle,
)

__all__ = [
    "BootstrapInterval",
    "FitBoundaryAudit",
    "FootballStage1BacktestResult",
    "FootballLedgerScore",
    "FootballMatchupPrediction",
    "FootballMatchupPredictor",
    "FootballQBCard",
    "FootballTrainingResult",
    "ScoreMetrics",
    "SliceReport",
    "Stage1FootballGameModel",
    "Stage2FootballPlayerModel",
    "compute_validation_metrics",
    "era_home_win_rates",
    "evaluate_football_stage2_holdout",
    "load_football_stage1_bundle",
    "reconcile_football_stage2_projections",
    "run_football_stage1_walk_forward_backtest",
    "score_football_prediction_ledger",
    "swap_home_away_features",
    "train_football_stage1_bundle",
]
