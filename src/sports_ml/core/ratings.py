"""Dynamic pairwise Elo and multi-entrant skill rating engines (`core/ratings.py`).

Records pre-event ratings strictly before applying post-event updates.
Full implementation scheduled for Milestone 1.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict


class EloConfig(BaseModel):
    """Configuration for pairwise Elo rating updates."""

    model_config = ConfigDict(frozen=True)

    base_rating: float = 1500.0
    home_field_advantage: float = 35.0
    k_factor: float = 6.0
    shootout_winner_score: float = 0.60
    offseason_retention: float = 0.67


class PairwiseEloEngine:
    """Dynamic pairwise Elo engine supporting HFA, MOV scaling, OT/SO dampening, and regression."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()

    def compute_ratings(self, games_df: pl.DataFrame) -> pl.DataFrame:
        """Compute pre-game Elo ratings, win probabilities, and post-game updates."""
        raise NotImplementedError("Scheduled for Milestone 1 implementation.")


class MultiEntrantSkillEngine:
    """Extensible field-adjusted skill rating engine for `multi_entrant_field` events."""

    def compute_ratings(self, field_df: pl.DataFrame) -> pl.DataFrame:
        """Compute pre-event skill ratings across multi-entrant tournament fields."""
        raise NotImplementedError("Scheduled for Milestone 1 implementation.")
