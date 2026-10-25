"""Walk-forward chronological cross-validation tests (`tests/core/test_validation.py`)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sports_ml.core.validation import WalkForwardValidator


def _build_timeline_df(n_timestamps: int = 10) -> pl.DataFrame:
    base_time = datetime(2024, 10, 1, 23, 0, tzinfo=UTC)
    rows = []
    event_id = 1
    for t_idx in range(n_timestamps):
        ts = base_time + timedelta(days=t_idx)
        for seq in (1, 2):
            rows.append(
                {
                    "event_id": event_id,
                    "event_seq": seq,
                    "start_time_utc": ts,
                    "home_win": event_id % 2,
                }
            )
            event_id += 1
    return pl.DataFrame(rows)


def test_walk_forward_expanding_and_sliding_splits() -> None:
    """Verify strict chronological separation and window sizing in walk-forward splits."""
    df = _build_timeline_df(n_timestamps=10)

    expanding_cv = WalkForwardValidator(n_splits=3, expanding=True, min_train_groups=4)
    exp_splits = expanding_cv.split(df)
    assert len(exp_splits) == 3

    prev_train_len = 0
    for split in exp_splits:
        max_train_ts = split.train_df["start_time_utc"].max()
        min_val_ts = split.val_df["start_time_utc"].min()
        assert isinstance(max_train_ts, datetime) and isinstance(min_val_ts, datetime)
        assert max_train_ts < min_val_ts
        assert split.train_df.height > prev_train_len
        prev_train_len = split.train_df.height

    sliding_cv = WalkForwardValidator(
        n_splits=3,
        expanding=False,
        min_train_groups=4,
        sliding_train_groups=4,
    )
    slid_splits = sliding_cv.split(df)
    assert len(slid_splits) == 3
    for split in slid_splits:
        max_train_ts = split.train_df["start_time_utc"].max()
        min_val_ts = split.val_df["start_time_utc"].min()
        assert isinstance(max_train_ts, datetime) and isinstance(min_val_ts, datetime)
        assert max_train_ts < min_val_ts
        assert split.train_df["start_time_utc"].n_unique() == 4

    with pytest.raises(ValueError, match="Insufficient unique timestamps"):
        WalkForwardValidator(n_splits=5, min_train_groups=8).split(df)
