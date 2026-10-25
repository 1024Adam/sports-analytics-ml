"""Model artifact registry and SHA-256 schema fingerprinting (`core/registry.py`).

Persists trained model bundles alongside `metadata.json` and validates feature
compatibility prior to inference.
Full implementation scheduled for Milestone 6.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict


class ModelMetadata(BaseModel):
    """Schema fingerprint, training coverage, and validation metrics for a saved model bundle."""

    model_config = ConfigDict(frozen=True)

    sport: str
    model_version: str
    stage1_feature_columns: list[str]
    stage1_feature_dtypes: dict[str, str]
    schema_sha256: str
    min_event_date: str
    max_event_date: str
    total_events: int
    git_commit_sha: str
    validation_metrics: dict[str, float]


class ModelRegistry:
    """Handles saving, loading, and validating fingerprinted model bundles."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

    def validate_features(self, df: pl.DataFrame, metadata: ModelMetadata) -> None:
        """Verify that `df` matches the ordered columns, dtypes, and SHA-256 fingerprint."""
        raise NotImplementedError("Scheduled for Milestone 6 implementation.")
