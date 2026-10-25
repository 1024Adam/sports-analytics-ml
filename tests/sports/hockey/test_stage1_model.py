"""Stage 1 Mandatory Calibrated Game Win Probability model tests (`test_stage1_model.py`).

Full probability bounds, sum-to-one, and walk-forward calibration tests
will be implemented in Milestone 4.
"""

from __future__ import annotations

from sports_ml.config import RANDOM_SEED
from sports_ml.sports.hockey.models.stage1_game import Stage1GameOutcomeModel


def test_stage1_model_init() -> None:
    """Verify Stage 1 model initializes with central RANDOM_SEED."""
    model = Stage1GameOutcomeModel()
    assert model.random_seed == RANDOM_SEED
