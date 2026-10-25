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
        from sports_ml.core.contracts import validate_canonical_event_frame

        working = df
        if "sport" not in working.columns:
            working = working.with_columns(pl.lit(self.sport_slug).alias("sport"))
        if "event_id" not in working.columns and "game_id" in working.columns:
            working = working.with_columns(pl.col("game_id").alias("event_id"))
        return validate_canonical_event_frame(working, allow_default_event_seq=True)

    def validate_participant_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize an NHL skater/goalie DataFrame against participant rules."""
        from sports_ml.core.contracts import validate_canonical_participant_frame

        working = df
        if "event_id" not in working.columns and "game_id" in working.columns:
            working = working.with_columns(pl.col("game_id").alias("event_id"))
        if "entity_id" not in working.columns and "team" in working.columns:
            working = working.with_columns(pl.col("team").alias("entity_id"))
        if "participant_id" not in working.columns and "player_id" in working.columns:
            working = working.with_columns(pl.col("player_id").alias("participant_id"))
        return validate_canonical_participant_frame(working)
