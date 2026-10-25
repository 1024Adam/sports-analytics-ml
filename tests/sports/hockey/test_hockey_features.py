"""Hockey pre-game team, goalie, roster, and player feature tests (`test_hockey_features.py`).

Full anti-leakage, cold-start, and TOI roster weighting tests will be implemented in Milestone 3.
"""

from __future__ import annotations

import polars as pl
import pytest

from sports_ml.sports.hockey.features.team import build_team_rolling_features


def test_hockey_features_stub(sample_chronological_logs: pl.DataFrame) -> None:
    """Verify feature stub signature prior to Milestone 3."""
    with pytest.raises(NotImplementedError):
        build_team_rolling_features(sample_chronological_logs)
