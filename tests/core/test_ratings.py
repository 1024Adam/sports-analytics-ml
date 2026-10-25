"""Dynamic Elo and multi-entrant rating engine tests (`tests/core/test_ratings.py`).

Full Elo zero-sum conservation, HFA, shootout dampening, and off-season
regression tests will be implemented in Milestone 1.
"""

from __future__ import annotations

from sports_ml.core.ratings import EloConfig, PairwiseEloEngine


def test_elo_config_defaults() -> None:
    """Verify default Elo hyperparameters match architecture specification."""
    engine = PairwiseEloEngine()
    assert isinstance(engine.config, EloConfig)
    assert engine.config.base_rating == 1500.0
    assert engine.config.home_field_advantage == 35.0
    assert engine.config.k_factor == 6.0
    assert engine.config.shootout_winner_score == 0.60
    assert engine.config.offseason_retention == 0.67
