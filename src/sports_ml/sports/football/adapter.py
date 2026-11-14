"""NFL implementation of `SportAdapter` (`sports/football/adapter.py`)."""

from __future__ import annotations

import re

import polars as pl

from sports_ml.core.contracts import (
    CompetitionTopology,
    SchemaValidationError,
    validate_canonical_event_frame,
    validate_canonical_participant_frame,
)
from sports_ml.sports.football.constants import (
    GAMES_REQUIRED_COLUMNS,
    IMPLEMENTED_FOOTBALL_LEAGUES,
    MODEL_SEASON_TYPES,
    RESERVED_FOOTBALL_LEAGUES,
    TEAM_LOGS_REQUIRED_COLUMNS,
    normalize_football_team,
)

_GSIS_ID_PATTERN = re.compile(r"^\d{2}-(\d{7})$")


def gsis_id_to_int(gsis_id: str | int) -> int:
    """Convert an nflverse GSIS player id (`'00-0033873'`) to its `Int64` participant id."""
    if isinstance(gsis_id, int):
        if gsis_id < 0:
            raise ValueError(f"Player id cannot be negative, got {gsis_id}.")
        return gsis_id
    cleaned = gsis_id.strip()
    match = _GSIS_ID_PATTERN.match(cleaned)
    if match is not None:
        return int(match.group(1))
    if cleaned.isdigit():
        return int(cleaned)
    raise ValueError(f"Invalid GSIS player id: {gsis_id!r}.")


def _normalize_team_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(col).map_elements(normalize_football_team, return_dtype=pl.Utf8).alias(col)
            for col in columns
            if col in df.columns
        ]
    )


def _require_columns(df: pl.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"Football {label} frame is missing columns: {missing}.")


def _validate_season_types(df: pl.DataFrame) -> pl.DataFrame:
    working = df.with_columns(pl.col("season_type").cast(pl.Utf8).str.to_uppercase())
    if working.filter(pl.col("season_type") == "PRE").height > 0:
        raise SchemaValidationError(
            "Preseason ('PRE') rows are not allowed in model tables; filter them upstream."
        )
    invalid = working.filter(~pl.col("season_type").is_in(list(MODEL_SEASON_TYPES)))
    if invalid.height > 0:
        raise SchemaValidationError(
            f"Football 'season_type' must be one of {list(MODEL_SEASON_TYPES)}, "
            f"got {invalid['season_type'].unique().to_list()}."
        )
    return working


def _validate_league(df: pl.DataFrame) -> None:
    leagues = [str(v) for v in df["league"].unique().to_list()]
    reserved = [lg for lg in leagues if lg in RESERVED_FOOTBALL_LEAGUES]
    if reserved:
        raise SchemaValidationError(f"Football league(s) {reserved} are reserved/unimplemented.")
    unknown = [lg for lg in leagues if lg not in IMPLEMENTED_FOOTBALL_LEAGUES]
    if unknown:
        raise SchemaValidationError(
            f"Football 'league' must be one of {list(IMPLEMENTED_FOOTBALL_LEAGUES)}, got {unknown}."
        )


def _validate_event_seq_is_one(df: pl.DataFrame) -> None:
    bad = df.filter(pl.col("event_seq") != 1)
    if bad.height > 0:
        raise SchemaValidationError(
            f"Football 'event_seq' must always be 1, got {bad['event_seq'].unique().to_list()}."
        )


def _validate_one_game_per_team_week(df: pl.DataFrame, team_columns: tuple[str, ...]) -> None:
    long = pl.concat(
        [
            df.select("season", "week", pl.col(col).alias("_team"))
            for col in team_columns
            if col in df.columns
        ]
    )
    dupes = long.group_by("season", "week", "_team").len().filter(pl.col("len") > 1)
    if dupes.height > 0:
        first = dupes.row(0, named=True)
        raise SchemaValidationError(
            "Each team may play at most one game per (season, week); "
            f"found {dupes.height} violation(s), e.g. team={first['_team']}, "
            f"season={first['season']}, week={first['week']}."
        )


class FootballAdapter:
    """NFL (`league = 'NFL'`) adapter implementing `SportAdapter` for weekly head-to-head games."""

    @property
    def sport_slug(self) -> str:
        return "football"

    @property
    def topology(self) -> CompetitionTopology:
        return "head_to_head"

    def validate_event_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize a Silver `games.parquet` frame against canonical rules."""
        working = df
        if "sport" not in working.columns:
            working = working.with_columns(pl.lit(self.sport_slug).alias("sport"))
        if "event_id" not in working.columns and "game_id" in working.columns:
            working = working.with_columns(pl.col("game_id").alias("event_id"))
        if "league" not in working.columns:
            working = working.with_columns(pl.lit("NFL").alias("league"))
        _require_columns(working, GAMES_REQUIRED_COLUMNS, "games")
        if "event_id" not in working.columns:
            raise SchemaValidationError("Football games frame requires 'game_id' or 'event_id'.")

        working = _normalize_team_columns(working, ("home_team", "away_team"))
        _validate_league(working)
        working = _validate_season_types(working)

        validated = validate_canonical_event_frame(
            working,
            allow_default_event_seq=True,
            check_unique_event_key=False,
        )
        n_dupes = validated.height - validated["event_id"].n_unique()
        if n_dupes > 0:
            raise SchemaValidationError(f"Duplicate game_id keys detected: {n_dupes}.")
        _validate_event_seq_is_one(validated)
        _validate_one_game_per_team_week(validated, ("home_team", "away_team"))
        return validated

    def validate_team_game_logs(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize a Silver `team_game_logs.parquet` frame (one row per team)."""
        working = df
        if "event_seq" not in working.columns:
            working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
        _require_columns(working, TEAM_LOGS_REQUIRED_COLUMNS, "team log")
        working = _normalize_team_columns(working, ("team", "opponent"))
        working = _validate_season_types(working)
        if "league" in working.columns:
            _validate_league(working)
        _validate_event_seq_is_one(working)
        if working.select("game_id", "team").n_unique() != working.height:
            raise SchemaValidationError("Duplicate (game_id, team) rows detected in team logs.")
        _validate_one_game_per_team_week(working, ("team",))
        return working.sort("start_time_utc", "event_seq", "game_id", "team")

    def validate_participant_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize a `player_game_logs.parquet` frame."""
        working = df
        if "event_id" not in working.columns and "game_id" in working.columns:
            working = working.with_columns(pl.col("game_id").alias("event_id"))
        if "entity_id" not in working.columns and "team" in working.columns:
            working = working.with_columns(
                pl.col("team")
                .map_elements(normalize_football_team, return_dtype=pl.Utf8)
                .alias("entity_id")
            )
        if "participant_id" not in working.columns and "player_id" in working.columns:
            working = working.with_columns(
                pl.col("player_id")
                .map_elements(gsis_id_to_int, return_dtype=pl.Int64)
                .alias("participant_id")
            )
        if "event_seq" not in working.columns:
            working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
        _validate_event_seq_is_one(working)
        return validate_canonical_participant_frame(working, check_unique_participant_key=True)
