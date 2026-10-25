"""Temporal firewall and future-mutation invariance tests (`tests/core/test_temporal_leakage.py`).

Full Future-Mutation Invariance tests for `core/temporal.py` and `core/shrinkage.py`
will be implemented in Milestone 1.
"""

from __future__ import annotations

import polars as pl

from sports_ml.config import RANDOM_SEED, Settings
from sports_ml.core.contracts import (
    CANONICAL_EVENT_KEY_COLS,
    CANONICAL_PARTICIPANT_KEY_COLS,
    CANONICAL_TEMPORAL_SORT_COLS,
)


def test_framework_seed_and_canonical_keys(
    temp_settings: Settings,
    sample_chronological_logs: pl.DataFrame,
) -> None:
    """Verify deterministic seed configuration and canonical temporal sort columns."""
    assert RANDOM_SEED == 42
    assert temp_settings.random_seed == 42
    assert temp_settings.paths.silver_dir.exists()
    assert CANONICAL_TEMPORAL_SORT_COLS == ("start_time_utc", "event_seq", "event_id")
    assert "event_id" in CANONICAL_EVENT_KEY_COLS
    assert "participant_id" in CANONICAL_PARTICIPANT_KEY_COLS
    assert set(CANONICAL_TEMPORAL_SORT_COLS).issubset(set(sample_chronological_logs.columns))
