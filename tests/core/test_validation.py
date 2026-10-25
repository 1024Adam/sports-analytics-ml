"""Walk-forward chronological cross-validation tests (`tests/core/test_validation.py`).

Full expanding and sliding window split verification will be implemented in Milestone 1.
"""

from __future__ import annotations

from sports_ml.core.validation import WalkForwardValidator


def test_walk_forward_validator_init() -> None:
    """Verify default walk-forward validator configuration."""
    validator = WalkForwardValidator(n_splits=4, expanding=True)
    assert validator.n_splits == 4
    assert validator.expanding is True
    assert validator.timestamp_col == "start_time_utc"
