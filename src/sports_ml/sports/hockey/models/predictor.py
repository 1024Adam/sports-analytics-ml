"""Unified two-stage NHL matchup predictor service (`models/predictor.py`).

Exposes `HockeyMatchupPredictor.predict_matchup(..., include_players=...)` and
`predict_date(...)`, surfaces confidence warnings, and appends to `data/predictions/`.
Full implementation scheduled for Milestone 6.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict


class PlayerMatchupProjections(BaseModel):
    """Container for Stage 2 skater and goalie projection DataFrames."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    home_skaters: pl.DataFrame
    away_skaters: pl.DataFrame
    goalies: pl.DataFrame


class MatchupPrediction(BaseModel):
    """Structured Stage 1 (mandatory) and Stage 2 (optional) matchup prediction output."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    game_id: int | None = None
    game_date: date
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    warnings: list[str] = []
    player_projections: PlayerMatchupProjections | None = None


class HockeyMatchupPredictor:
    """Unified two-stage inference service for NHL matchups."""

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> HockeyMatchupPredictor:
        """Load a fingerprinted model bundle from `artifact_dir`."""
        raise NotImplementedError("Scheduled for Milestone 6 implementation.")

    def predict_matchup(
        self,
        home_team: str,
        away_team: str,
        game_date: date,
        include_players: bool = False,
    ) -> MatchupPrediction:
        """Predict Stage 1 calibrated win probability and optional Stage 2 player projections."""
        raise NotImplementedError("Scheduled for Milestone 6 implementation.")
