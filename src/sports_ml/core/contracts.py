"""Canonical multi-sport contracts and `SportAdapter` protocol (`core/contracts.py`).

Supports both `head_to_head` (NHL, MLB, PGA H2H) and `multi_entrant_field` (PGA
tournament field) competition topologies without sport-specific assumptions.
"""

from __future__ import annotations

from typing import Final, Literal, Protocol, runtime_checkable

import polars as pl

CompetitionTopology = Literal["head_to_head", "multi_entrant_field"]

CANONICAL_TEMPORAL_SORT_COLS: Final[tuple[str, str, str]] = (
    "start_time_utc",
    "event_seq",
    "event_id",
)

CANONICAL_EVENT_KEY_COLS: Final[tuple[str, ...]] = (
    "sport",
    "season",
    "event_id",
    "event_seq",
    "start_time_utc",
)

CANONICAL_PARTICIPANT_KEY_COLS: Final[tuple[str, ...]] = (
    "event_id",
    "entity_id",
    "participant_id",
)


class SchemaValidationError(ValueError):
    """Raised when a DataFrame violates canonical multi-sport schema contracts."""


def validate_canonical_event_frame(
    df: pl.DataFrame,
    *,
    allow_default_event_seq: bool = True,
    check_unique_event_key: bool = False,
) -> pl.DataFrame:
    """Validate, normalize dtypes, and chronologically sort a canonical event DataFrame."""
    working = df
    if allow_default_event_seq and "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))

    missing = [col for col in CANONICAL_EVENT_KEY_COLS if col not in working.columns]
    if missing:
        raise SchemaValidationError(f"Missing canonical event key columns: {missing}")

    for col in CANONICAL_EVENT_KEY_COLS:
        if working[col].null_count() > 0:
            raise SchemaValidationError(f"Canonical event key column '{col}' contains nulls.")

    start_dtype = working.schema["start_time_utc"]
    if not isinstance(start_dtype, pl.Datetime):
        raise SchemaValidationError(
            f"Column 'start_time_utc' must be pl.Datetime, got {start_dtype}."
        )
    if start_dtype.time_zone != "UTC":
        working = working.with_columns(
            pl.col("start_time_utc").dt.replace_time_zone("UTC").dt.cast_time_unit("us")
        )
    else:
        working = working.with_columns(pl.col("start_time_utc").dt.cast_time_unit("us"))

    working = working.with_columns(
        pl.col("sport").cast(pl.Utf8),
        pl.col("season").cast(pl.Int32),
        pl.col("event_id").cast(pl.Int64),
        pl.col("event_seq").cast(pl.Int32),
    )

    if check_unique_event_key:
        n_rows = working.height
        n_unique = working.select(["event_id", "event_seq"]).n_unique()
        if n_unique != n_rows:
            raise SchemaValidationError(
                f"Duplicate (event_id, event_seq) keys detected: {n_rows - n_unique} duplicates."
            )

    return working.sort(list(CANONICAL_TEMPORAL_SORT_COLS))


def validate_canonical_participant_frame(
    df: pl.DataFrame,
    *,
    check_unique_participant_key: bool = False,
) -> pl.DataFrame:
    """Validate and normalize a canonical participant appearance DataFrame."""
    missing = [col for col in CANONICAL_PARTICIPANT_KEY_COLS if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing canonical participant key columns: {missing}")

    for col in CANONICAL_PARTICIPANT_KEY_COLS:
        if df[col].null_count() > 0:
            raise SchemaValidationError(f"Canonical participant key column '{col}' contains nulls.")

    working = df.with_columns(
        pl.col("event_id").cast(pl.Int64),
        pl.col("entity_id").cast(pl.Utf8),
        pl.col("participant_id").cast(pl.Int64),
    )

    if check_unique_participant_key:
        n_rows = working.height
        n_unique = working.select(list(CANONICAL_PARTICIPANT_KEY_COLS)).n_unique()
        if n_unique != n_rows:
            raise SchemaValidationError(
                "Duplicate (event_id, entity_id, participant_id) keys detected: "
                f"{n_rows - n_unique} duplicates."
            )

    sort_cols = [
        col
        for col in (*CANONICAL_TEMPORAL_SORT_COLS, "entity_id", "participant_id")
        if col in working.columns
    ]
    return working.sort(sort_cols) if sort_cols else working


@runtime_checkable
class StarterModifierHook(Protocol):
    """Pluggable hook for starter-unit rating adjustments (e.g., NHL Goalie, MLB SP)."""

    def get_starter_elo_delta(self, event_id: int, entity_id: str) -> float:
        """Return pre-event Elo point adjustment for `entity_id`'s starting specialist."""
        ...


@runtime_checkable
class VenueModifierHook(Protocol):
    """Pluggable hook for venue, ballpark, or course environment modifiers."""

    def get_venue_elo_delta(self, event_id: int, home_entity: str, away_entity: str) -> float:
        """Return pre-event Elo point adjustment for the event venue."""
        ...


@runtime_checkable
class SportAdapter(Protocol):
    """Protocol that every sport plugin (`sports/<sport>/adapter.py`) must implement."""

    @property
    def sport_slug(self) -> str:
        """Canonical lowercase identifier (e.g., 'hockey', 'baseball', 'golf')."""
        ...

    @property
    def topology(self) -> CompetitionTopology:
        """Competition format ('head_to_head' or 'multi_entrant_field')."""
        ...

    def validate_event_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize a DataFrame against canonical event schema rules."""
        ...

    def validate_participant_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize a DataFrame against canonical participant schema rules."""
        ...
