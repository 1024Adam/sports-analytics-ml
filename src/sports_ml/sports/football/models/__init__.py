"""Football Stage 1 models package (`sports/football/models/`)."""

from __future__ import annotations

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
from sports_ml.sports.football.models.training import (
    FootballTrainingResult,
    compute_validation_metrics,
    load_football_stage1_bundle,
    train_football_stage1_bundle,
)

__all__ = [
    "BootstrapInterval",
    "FitBoundaryAudit",
    "FootballStage1BacktestResult",
    "FootballTrainingResult",
    "ScoreMetrics",
    "SliceReport",
    "Stage1FootballGameModel",
    "compute_validation_metrics",
    "era_home_win_rates",
    "load_football_stage1_bundle",
    "run_football_stage1_walk_forward_backtest",
    "swap_home_away_features",
    "train_football_stage1_bundle",
]
