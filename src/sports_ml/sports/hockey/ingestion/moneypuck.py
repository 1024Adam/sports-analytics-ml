"""MoneyPuck historical and active-season CSV downloader and parser (`ingestion/moneypuck.py`).

Parses `5on5`, `5on4`, `4on5`, and `all` situation metrics (`xGF`, `xGA`,
`Corsi`, `Fenwick`, `High-Danger xG`, and goalie `GSAx`).
Full implementation scheduled for Milestone 2.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


class MoneyPuckClient:
    """Downloader and Polars parser for MoneyPuck team, skater, and goalie CSV snapshots."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def load_team_game_stats(self, season_start_year: int) -> pl.DataFrame:
        """Load and normalize MoneyPuck team game-by-game situational stats."""
        raise NotImplementedError("Scheduled for Milestone 2 implementation.")

    def load_skater_game_stats(self, season_start_year: int) -> pl.DataFrame:
        """Load and normalize MoneyPuck skater game-by-game situational stats."""
        raise NotImplementedError("Scheduled for Milestone 2 implementation.")

    def load_goalie_game_stats(self, season_start_year: int) -> pl.DataFrame:
        """Load and normalize MoneyPuck goalie game-by-game situational stats."""
        raise NotImplementedError("Scheduled for Milestone 2 implementation.")
