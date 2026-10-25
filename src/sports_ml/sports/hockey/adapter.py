"""NHL Hockey implementation of the `SportAdapter` protocol (`sports/hockey/adapter.py`)."""

from __future__ import annotations

import polars as pl

from sports_ml.core.contracts import CompetitionTopology


class HockeyAdapter:
    """NHL Hockey adapter implementing `SportAdapter` for head-to-head matchups."""

    @property
    def sport_slug(self) -> str:
        return "hockey"

    @property
    def topology(self) -> CompetitionTopology:
        return "head_to_head"

    def validate_event_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize an NHL games DataFrame against canonical event schema rules."""
        raise NotImplementedError("Scheduled for Milestone 1/2 implementation.")

    def validate_participant_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize an NHL skater/goalie DataFrame against participant rules."""
        raise NotImplementedError("Scheduled for Milestone 1/2 implementation.")
