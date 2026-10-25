"""Walk-forward expanding and sliding window time-series cross-validator (`core/validation.py`).

Guarantees chronological separation (`max(train_time) < min(val_time)`) with
zero same-timestamp leakage across splits.
Full implementation scheduled for Milestone 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single chronological train/validation fold."""

    fold_index: int
    train_df: pl.DataFrame
    val_df: pl.DataFrame


class WalkForwardValidator:
    """Expanding or sliding window chronological cross-validator."""

    def __init__(
        self,
        n_splits: int = 3,
        expanding: bool = True,
        timestamp_col: str = "start_time_utc",
    ) -> None:
        self.n_splits = n_splits
        self.expanding = expanding
        self.timestamp_col = timestamp_col

    def split(self, df: pl.DataFrame) -> list[WalkForwardSplit]:
        """Generate chronological train/validation splits."""
        raise NotImplementedError("Scheduled for Milestone 1 implementation.")
