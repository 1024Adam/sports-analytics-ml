"""Walk-forward expanding and sliding window time-series cross-validator (`core/validation.py`).

Guarantees chronological separation (`max(train_time) < min(val_time)`) with
zero same-timestamp leakage across splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from sports_ml.core.temporal import sort_events_chronologically


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single chronological train/validation fold."""

    fold_index: int
    train_df: pl.DataFrame
    val_df: pl.DataFrame


class WalkForwardValidator:
    """Expanding or sliding window chronological cross-validator grouped by event timestamp."""

    def __init__(
        self,
        n_splits: int = 3,
        expanding: bool = True,
        timestamp_col: str = "start_time_utc",
        min_train_groups: int = 1,
        sliding_train_groups: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
        if min_train_groups < 1:
            raise ValueError(f"min_train_groups must be >= 1, got {min_train_groups}.")
        self.n_splits = n_splits
        self.expanding = expanding
        self.timestamp_col = timestamp_col
        self.min_train_groups = min_train_groups
        self.sliding_train_groups = sliding_train_groups

    def split(self, df: pl.DataFrame) -> list[WalkForwardSplit]:
        """Generate chronological train/validation splits with strict temporal separation."""
        sorted_df = sort_events_chronologically(df)
        unique_times = sorted_df[self.timestamp_col].unique().sort().to_list()
        n_times = len(unique_times)

        if n_times < self.min_train_groups + self.n_splits:
            raise ValueError(
                f"Insufficient unique timestamps ({n_times}) for n_splits={self.n_splits} "
                f"and min_train_groups={self.min_train_groups}."
            )

        val_group_size = max(1, (n_times - self.min_train_groups) // self.n_splits)
        initial_train_groups = n_times - val_group_size * self.n_splits
        sliding_width = self.sliding_train_groups or initial_train_groups

        splits: list[WalkForwardSplit] = []
        for fold_idx in range(self.n_splits):
            val_start_idx = initial_train_groups + fold_idx * val_group_size
            val_end_idx = (
                n_times if fold_idx == self.n_splits - 1 else val_start_idx + val_group_size
            )
            train_start_idx = 0 if self.expanding else max(0, val_start_idx - sliding_width)

            train_min_time = unique_times[train_start_idx]
            train_max_time = unique_times[val_start_idx - 1]
            val_min_time = unique_times[val_start_idx]
            val_max_time = unique_times[val_end_idx - 1]

            train_fold = sorted_df.filter(
                (pl.col(self.timestamp_col) >= train_min_time)
                & (pl.col(self.timestamp_col) <= train_max_time)
            )
            val_fold = sorted_df.filter(
                (pl.col(self.timestamp_col) >= val_min_time)
                & (pl.col(self.timestamp_col) <= val_max_time)
            )

            splits.append(
                WalkForwardSplit(
                    fold_index=fold_idx,
                    train_df=train_fold,
                    val_df=val_fold,
                )
            )

        return splits
