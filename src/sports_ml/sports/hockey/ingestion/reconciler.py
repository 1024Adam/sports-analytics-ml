"""Reconciler joining NHL Web API and MoneyPuck into Silver Parquet tables (`reconciler.py`).

Produces `games.parquet`, `team_game_logs.parquet`, `skater_game_logs.parquet`,
and `goalie_game_logs.parquet` with graceful fallback when MoneyPuck lags.
Full implementation scheduled for Milestone 2.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


class HockeySilverReconciler:
    """Reconciles raw NHL API and MoneyPuck records into canonical Silver Parquet tables."""

    def __init__(self, silver_dir: Path) -> None:
        self.silver_dir = silver_dir

    def reconcile_games(self, raw_games_df: pl.DataFrame) -> pl.DataFrame:
        """Normalize and upsert `games.parquet`."""
        raise NotImplementedError("Scheduled for Milestone 2 implementation.")
