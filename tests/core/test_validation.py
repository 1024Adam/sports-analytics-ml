"""Walk-forward chronological cross-validation tests (`tests/core/test_validation.py`)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from sports_ml.core.registry import (
    ModelRegistry,
    SchemaMismatchError,
)
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


def test_registry_schema_fingerprint_and_mismatch_errors(tmp_path: Path) -> None:
    """Verify registry raises SchemaMismatchError when columns are missing or reordered."""
    train_df = pl.DataFrame(
        {
            "game_date": ["2024-10-10", "2024-10-12"],
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
            "feat_c": [5, 6],
        }
    )
    registry = ModelRegistry(artifacts_dir=tmp_path / "artifacts")
    bundle_dir = registry.save_bundle(
        sport="hockey",
        model_version="v_test",
        stage1_model={"dummy": "model"},
        stage1_train_df=train_df,
        stage1_feature_columns=["feat_a", "feat_b", "feat_c"],
        validation_metrics={"log_loss": 0.65, "brier_score": 0.23},
    )

    loaded = registry.load_bundle(bundle_dir)
    assert loaded.metadata.sport == "hockey"
    assert loaded.metadata.stage1_feature_columns == ["feat_a", "feat_b", "feat_c"]

    # Exact schema passes cleanly
    registry.validate_features(train_df, loaded.metadata, stage=1)

    # 1. Missing feature column -> raises SchemaMismatchError
    missing_col_df = train_df.drop("feat_b")
    with pytest.raises(SchemaMismatchError, match="missing columns"):
        registry.validate_features(missing_col_df, loaded.metadata, stage=1)

    # 2. Reordered feature columns -> raises SchemaMismatchError
    reordered_df = train_df.select(["game_date", "feat_b", "feat_a", "feat_c"])
    with pytest.raises(SchemaMismatchError, match="column order mismatch"):
        registry.validate_features(reordered_df, loaded.metadata, stage=1)

    # 3. Changed dtype -> raises SchemaMismatchError
    retyped_df = train_df.with_columns(pl.col("feat_a").cast(pl.Utf8))
    with pytest.raises(SchemaMismatchError, match="dtype mismatch"):
        registry.validate_features(retyped_df, loaded.metadata, stage=1)
