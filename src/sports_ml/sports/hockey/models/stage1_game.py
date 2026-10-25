"""Stage 1 Mandatory Calibrated Game Win Probability pipeline (`models/stage1_game.py`).

Combines dynamic Elo baseline, LightGBM classifier, out-of-fold probability
calibration, and auxiliary Poisson/Tweedie team-goals pace estimator.
Full implementation scheduled for Milestone 4.
"""

from __future__ import annotations

import polars as pl

from sports_ml.config import RANDOM_SEED


class Stage1GameOutcomeModel:
    """Mandatory Stage 1 NHL game win probability and expected team-goals estimator."""

    def __init__(self, random_seed: int = RANDOM_SEED) -> None:
        self.random_seed = random_seed

    def fit(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> Stage1GameOutcomeModel:
        """Fit GBDT classifier, team-goals regressor, and out-of-fold calibrator."""
        raise NotImplementedError("Scheduled for Milestone 4 implementation.")

    def predict(self, features_df: pl.DataFrame) -> pl.DataFrame:
        """Predict calibrated `home_win_prob`, `away_win_prob`, and expected team goals."""
        raise NotImplementedError("Scheduled for Milestone 4 implementation.")
