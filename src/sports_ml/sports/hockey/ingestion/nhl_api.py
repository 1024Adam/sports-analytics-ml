"""Official NHL Web API (`https://api-web.nhle.com/v1/`) client (`ingestion/nhl_api.py`).

Provides deterministic disk caching and exponential backoff retries for
schedules, boxscores, landing pages, and team rosters.
Full implementation scheduled for Milestone 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

NHL_WEB_API_BASE_URL: Final[str] = "https://api-web.nhle.com/v1"


class NHLWebAPIClient:
    """Cached HTTP client for the Official NHL Web API (`api-web.nhle.com/v1`)."""

    def __init__(self, cache_dir: Path, base_url: str = NHL_WEB_API_BASE_URL) -> None:
        self.cache_dir = cache_dir
        self.base_url = base_url

    def fetch_schedule_by_date(self, date_str: str) -> dict[str, Any]:
        """Fetch 7-day schedule week starting on `YYYY-MM-DD`."""
        raise NotImplementedError("Scheduled for Milestone 2 implementation.")

    def fetch_game_boxscore(self, game_id: int) -> dict[str, Any]:
        """Fetch official skater and goalie boxscore payload for `game_id`."""
        raise NotImplementedError("Scheduled for Milestone 2 implementation.")

    def fetch_team_roster(self, team_tricode: str) -> dict[str, Any]:
        """Fetch current active roster for `team_tricode`."""
        raise NotImplementedError("Scheduled for Milestone 2 implementation.")
