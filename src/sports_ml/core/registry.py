"""Model artifact registry and SHA-256 schema fingerprinting (`core/registry.py`).

Persists trained model bundles alongside `metadata.json` and validates feature
compatibility prior to inference so silent feature-schema drift is impossible.
"""

from __future__ import annotations

import hashlib
import pickle
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class SchemaMismatchError(ValueError):
    """Raised when live inference features do not match the saved model schema fingerprint."""


def compute_schema_fingerprint(
    feature_columns: Sequence[str],
    feature_dtypes: Mapping[str, str],
) -> str:
    """Compute a deterministic SHA-256 hex digest over ordered `(column, dtype)` pairs."""
    canonical_pairs = [f"{col}:{feature_dtypes.get(col, 'Unknown')}" for col in feature_columns]
    payload = "|".join(canonical_pairs).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_schema_signature(
    df: pl.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[list[str], dict[str, str], str]:
    """Extract ordered feature names, Polars dtype strings, and SHA-256 schema hash from `df`."""
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise SchemaMismatchError(f"Missing required feature columns in DataFrame: {missing}")

    ordered_cols = list(feature_columns)
    dtypes_map = {col: str(df.schema[col]) for col in ordered_cols}
    sha256_hex = compute_schema_fingerprint(ordered_cols, dtypes_map)
    return ordered_cols, dtypes_map, sha256_hex


def get_current_git_sha() -> str:
    """Return current short git commit SHA or `'unknown'` if unavailable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
        )
        sha = proc.stdout.strip()
        return sha if proc.returncode == 0 and sha else "unknown"
    except Exception:
        return "unknown"


class ModelMetadata(BaseModel):
    """Schema fingerprint, training coverage, and validation metrics for a saved model bundle."""

    model_config = ConfigDict(frozen=True)

    sport: str
    model_version: str
    stage1_feature_columns: list[str]
    stage1_feature_dtypes: dict[str, str]
    schema_sha256: str
    stage2_feature_columns: list[str] = Field(default_factory=list)
    stage2_feature_dtypes: dict[str, str] = Field(default_factory=dict)
    stage2_schema_sha256: str | None = None
    min_event_date: str
    max_event_date: str
    total_events: int
    git_commit_sha: str
    validation_metrics: dict[str, float]


@dataclass(frozen=True)
class LoadedModelBundle:
    """Container returned when loading a fingerprinted model bundle from disk."""

    metadata: ModelMetadata
    stage1_model: Any
    stage2_model: Any | None
    bundle_dir: Path


class ModelRegistry:
    """Handles saving, loading, and validating fingerprinted model bundles."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_bundle(
        self,
        *,
        sport: str,
        model_version: str,
        stage1_model: Any,
        stage1_train_df: pl.DataFrame,
        stage1_feature_columns: Sequence[str],
        validation_metrics: Mapping[str, float],
        stage2_model: Any | None = None,
        stage2_train_df: pl.DataFrame | None = None,
        stage2_feature_columns: Sequence[str] | None = None,
        update_latest: bool = True,
    ) -> Path:
        """Persist `stage1_model`, optional `stage2_model`, and `metadata.json` to `artifacts/`."""
        s1_cols, s1_dtypes, s1_sha = extract_schema_signature(
            stage1_train_df, stage1_feature_columns
        )

        if (
            stage2_model is not None
            and stage2_train_df is not None
            and stage2_feature_columns is not None
        ):
            s2_cols, s2_dtypes, s2_sha = extract_schema_signature(
                stage2_train_df, stage2_feature_columns
            )
        else:
            s2_cols, s2_dtypes, s2_sha = [], {}, None

        date_col = (
            "game_date"
            if "game_date" in stage1_train_df.columns
            else ("start_time_utc" if "start_time_utc" in stage1_train_df.columns else None)
        )
        if date_col is not None and not stage1_train_df.is_empty():
            min_date_str = str(stage1_train_df[date_col].min())[:10]
            max_date_str = str(stage1_train_df[date_col].max())[:10]
        else:
            min_date_str = "unknown"
            max_date_str = "unknown"

        metadata = ModelMetadata(
            sport=sport,
            model_version=model_version,
            stage1_feature_columns=s1_cols,
            stage1_feature_dtypes=s1_dtypes,
            schema_sha256=s1_sha,
            stage2_feature_columns=s2_cols,
            stage2_feature_dtypes=s2_dtypes,
            stage2_schema_sha256=s2_sha,
            min_event_date=min_date_str,
            max_event_date=max_date_str,
            total_events=stage1_train_df.height,
            git_commit_sha=get_current_git_sha(),
            validation_metrics={k: float(v) for k, v in validation_metrics.items()},
        )

        targets = [self.artifacts_dir / sport / model_version]
        if update_latest and model_version != "latest":
            targets.append(self.artifacts_dir / sport / "latest")

        for bundle_dir in targets:
            bundle_dir.mkdir(parents=True, exist_ok=True)
            (bundle_dir / "metadata.json").write_text(
                metadata.model_dump_json(indent=2), encoding="utf-8"
            )
            with (bundle_dir / "stage1_model.pkl").open("wb") as f1:
                pickle.dump(stage1_model, f1)
            if stage2_model is not None:
                with (bundle_dir / "stage2_model.pkl").open("wb") as f2:
                    pickle.dump(stage2_model, f2)

        return targets[0]

    @staticmethod
    def load_bundle(bundle_dir: str | Path) -> LoadedModelBundle:
        """Load a saved model bundle and its `metadata.json` from `bundle_dir`."""
        path = Path(bundle_dir)
        meta_path = path / "metadata.json"
        s1_path = path / "stage1_model.pkl"
        if not meta_path.exists() or not s1_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found in '{path}'. Run 'sports-ml train' first."
            )

        metadata = ModelMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
        with s1_path.open("rb") as f1:
            stage1_model = pickle.load(f1)

        s2_path = path / "stage2_model.pkl"
        if s2_path.exists():
            with s2_path.open("rb") as f2:
                stage2_model = pickle.load(f2)
        else:
            stage2_model = None

        return LoadedModelBundle(
            metadata=metadata,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            bundle_dir=path,
        )

    @staticmethod
    def validate_features(
        df: pl.DataFrame,
        metadata: ModelMetadata,
        *,
        stage: int = 1,
        enforce_column_order: bool = True,
    ) -> None:
        """Verify that `df` matches the ordered columns, dtypes, and SHA-256 fingerprint."""
        expected_cols = (
            metadata.stage1_feature_columns if stage == 1 else metadata.stage2_feature_columns
        )
        expected_dtypes = (
            metadata.stage1_feature_dtypes if stage == 1 else metadata.stage2_feature_dtypes
        )
        expected_sha = metadata.schema_sha256 if stage == 1 else metadata.stage2_schema_sha256

        if not expected_cols or expected_sha is None:
            return

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise SchemaMismatchError(
                f"Stage {stage} feature schema mismatch — missing columns: {missing}. "
                "Please rebuild features or retrain the model."
            )

        if enforce_column_order:
            present_feature_order = [col for col in df.columns if col in set(expected_cols)]
            if present_feature_order != expected_cols:
                raise SchemaMismatchError(
                    f"Stage {stage} feature column order mismatch: expected {expected_cols}, "
                    f"got {present_feature_order}."
                )

        actual_dtypes = {col: str(df.schema[col]) for col in expected_cols}
        mismatched_dtypes = {
            col: (expected_dtypes[col], actual_dtypes[col])
            for col in expected_cols
            if expected_dtypes.get(col) != actual_dtypes.get(col)
        }
        if mismatched_dtypes:
            raise SchemaMismatchError(f"Stage {stage} feature dtype mismatch: {mismatched_dtypes}.")

        actual_sha = compute_schema_fingerprint(expected_cols, actual_dtypes)
        if actual_sha != expected_sha:
            raise SchemaMismatchError(
                f"Stage {stage} SHA-256 schema fingerprint mismatch: "
                f"expected {expected_sha}, got {actual_sha}."
            )
