"""Shared deterministic synthetic Polars fixtures for offline unit tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from sports_ml.config import Settings, get_settings


@pytest.fixture
def temp_settings(tmp_path: Path) -> Settings:
    """Provide an isolated Settings instance rooted at `tmp_path`."""
    settings = get_settings(root_dir=tmp_path)
    settings.paths.ensure_dirs(sport="hockey")
    return settings


@pytest.fixture
def sample_chronological_logs() -> pl.DataFrame:
    """Synthetic multi-game chronological team appearance logs for anti-leakage testing."""
    return pl.DataFrame(
        {
            "sport": ["hockey"] * 6,
            "season": [20242025] * 6,
            "event_id": [2024020001, 2024020001, 2024020002, 2024020002, 2024020003, 2024020003],
            "event_seq": [1, 1, 1, 1, 1, 1],
            "game_date": [
                date(2024, 10, 10),
                date(2024, 10, 10),
                date(2024, 10, 12),
                date(2024, 10, 12),
                date(2024, 10, 15),
                date(2024, 10, 15),
            ],
            "start_time_utc": [
                datetime(2024, 10, 10, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 10, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 12, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 12, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 23, 0, tzinfo=UTC),
            ],
            "team": ["BOS", "TOR", "BOS", "MTL", "TOR", "BOS"],
            "opponent": ["TOR", "BOS", "MTL", "BOS", "BOS", "TOR"],
            "is_home": [True, False, True, False, True, False],
            "goals_for": [4, 2, 3, 1, 2, 5],
            "goals_against": [2, 4, 1, 3, 5, 2],
            "xgf_5v5": [2.8, 1.9, 2.5, 1.4, 2.1, 3.2],
            "xga_5v5": [1.9, 2.8, 1.4, 2.5, 3.2, 2.1],
            "toi_5v5_minutes": [48.0, 48.0, 50.0, 50.0, 49.0, 49.0],
        }
    )
