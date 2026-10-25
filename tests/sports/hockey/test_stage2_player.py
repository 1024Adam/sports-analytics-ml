"""Stage 2 Optional Player Matchup Stat Projection tests (`test_stage2_player.py`).

Full non-negativity, Stage 1 monotonicity, and cross-stage reconciliation tests
will be implemented in Milestone 5.
"""

from __future__ import annotations

from sports_ml.config import RANDOM_SEED
from sports_ml.sports.hockey.models.stage2_player import Stage2PlayerProjectionModel


def test_stage2_model_init() -> None:
    """Verify Stage 2 model initializes with default reconciliation gamma and RANDOM_SEED."""
    model = Stage2PlayerProjectionModel()
    assert model.reconciliation_gamma == 0.50
    assert model.random_seed == RANDOM_SEED
