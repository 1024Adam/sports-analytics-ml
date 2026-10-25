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


@runtime_checkable
class StarterModifierHook(Protocol):
    """Pluggable hook for starter-unit rating adjustments (e.g., NHL Goalie, MLB SP)."""

    def compute_starter_delta(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute pre-event rating adjustments attributable to the starting specialist."""
        ...


@runtime_checkable
class VenueModifierHook(Protocol):
    """Pluggable hook for venue, ballpark, or course environment modifiers."""

    def compute_venue_modifier(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute venue/park/course context modifiers for each event."""
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
