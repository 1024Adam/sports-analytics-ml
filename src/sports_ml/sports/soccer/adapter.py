"""EPL Soccer implementation of the `SportAdapter` protocol (`sports/soccer/adapter.py`)."""

from __future__ import annotations

import polars as pl

from sports_ml.core.contracts import CompetitionTopology
from sports_ml.sports.soccer.constants import normalize_soccer_club


class SoccerAdapter:
    """EPL Soccer adapter implementing `SportAdapter` for head-to-head (`1X2`) matches."""

    @property
    def sport_slug(self) -> str:
        return "soccer"

    @property
    def topology(self) -> CompetitionTopology:
        return "head_to_head"

    def validate_event_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize an EPL matches DataFrame against canonical event schema rules."""
        from sports_ml.core.contracts import validate_canonical_event_frame

        working = df
        if "sport" not in working.columns:
            working = working.with_columns(pl.lit(self.sport_slug).alias("sport"))
        if "event_id" not in working.columns:
            if "match_id" in working.columns:
                working = working.with_columns(pl.col("match_id").alias("event_id"))
            elif "game_id" in working.columns:
                working = working.with_columns(pl.col("game_id").alias("event_id"))
        if "home_team" in working.columns:
            working = working.with_columns(
                pl.col("home_team")
                .map_elements(normalize_soccer_club, return_dtype=pl.Utf8)
                .alias("home_team")
            )
        if "away_team" in working.columns:
            working = working.with_columns(
                pl.col("away_team")
                .map_elements(normalize_soccer_club, return_dtype=pl.Utf8)
                .alias("away_team")
            )
        return validate_canonical_event_frame(working, allow_default_event_seq=True)

    def validate_participant_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize an EPL outfield/goalkeeper DataFrame against participant rules."""
        from sports_ml.core.contracts import validate_canonical_participant_frame

        working = df
        if "event_id" not in working.columns:
            if "match_id" in working.columns:
                working = working.with_columns(pl.col("match_id").alias("event_id"))
            elif "game_id" in working.columns:
                working = working.with_columns(pl.col("game_id").alias("event_id"))
        if "entity_id" not in working.columns and "team" in working.columns:
            working = working.with_columns(
                pl.col("team")
                .map_elements(normalize_soccer_club, return_dtype=pl.Utf8)
                .alias("entity_id")
            )
        if "participant_id" not in working.columns and "player_id" in working.columns:
            working = working.with_columns(pl.col("player_id").alias("participant_id"))
        return validate_canonical_participant_frame(working)
