"""Centralized configuration, paths, and deterministic seed settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

RANDOM_SEED: Final[int] = 42


def _default_repo_root() -> Path:
    """Resolve the repository root directory (or `SPORTS_ML_ROOT_DIR` override)."""
    env_override = os.environ.get("SPORTS_ML_ROOT_DIR")
    if env_override:
        return Path(env_override).resolve()
    return Path(__file__).resolve().parents[2]


class DataLakePaths(BaseModel):
    """Medallion data lakehouse and artifact directory paths."""

    model_config = ConfigDict(frozen=True)

    root_dir: Path = Field(default_factory=_default_repo_root)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    @property
    def artifacts_dir(self) -> Path:
        return self.root_dir / "artifacts"

    def ensure_dirs(self, sport: str = "hockey") -> None:
        """Create local git-ignored lakehouse and artifact directories if missing."""
        for directory in (
            self.raw_dir / sport,
            self.bronze_dir / sport,
            self.silver_dir / sport,
            self.gold_dir / sport,
            self.predictions_dir,
            self.artifacts_dir / sport,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """Platform-wide configuration and default hyperparameters."""

    model_config = ConfigDict(frozen=True)

    random_seed: int = RANDOM_SEED
    paths: DataLakePaths = Field(default_factory=DataLakePaths)

    # Default temporal & shrinkage parameters (see docs/data-and-features.md)
    rolling_windows: tuple[int, ...] = (5, 10, 25)
    ewma_alpha: float = 0.12
    prior_season_weight: float = 0.65
    league_mean_weight: float = 0.35
    team_shrinkage_k: float = 8.0
    skater_shrinkage_k: float = 12.0
    goalie_shrinkage_k: float = 15.0

    # Default Elo parameters (see src/sports_ml/core/ratings.py)
    elo_base_rating: float = 1500.0
    elo_home_advantage: float = 24.0
    elo_k_factor: float = 8.0
    elo_shootout_winner_score: float = 0.60
    elo_offseason_retention: float = 0.72

    # Stage 2 cross-stage reconciliation damping factor
    stage2_reconciliation_gamma: float = 0.50


def get_settings(root_dir: Path | None = None) -> Settings:
    """Return an immutable Settings instance (optionally rooted at `root_dir` for tests)."""
    if root_dir is not None:
        return Settings(paths=DataLakePaths(root_dir=root_dir))
    return Settings()
