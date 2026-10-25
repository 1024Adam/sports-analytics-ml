"""Stage 2 Optional Player Matchup Stat Projections and coherence reconciler (`stage2_player.py`).

Predicts skater TOI, SOG, Goals, Assists, Points, milestone probabilities, and
goalie SA, SV, GA, SV% conditioned on Stage 1 game context.
Full implementation scheduled for Milestone 5.
"""

from __future__ import annotations

import polars as pl

from sports_ml.config import RANDOM_SEED


class Stage2PlayerProjectionModel:
    """Optional Stage 2 NHL skater and goalie matchup stat projection pipeline."""

    def __init__(
        self,
        reconciliation_gamma: float = 0.50,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.reconciliation_gamma = reconciliation_gamma
        self.random_seed = random_seed

    def fit(self, train_df: pl.DataFrame) -> Stage2PlayerProjectionModel:
        """Fit skater TOI, count/rate Poisson/Tweedie regressors, and goalie models."""
        raise NotImplementedError("Scheduled for Milestone 5 implementation.")

    def predict(self, player_features_df: pl.DataFrame) -> pl.DataFrame:
        """Generate Stage 2 player projections with cross-stage coherence scaling."""
        raise NotImplementedError("Scheduled for Milestone 5 implementation.")
