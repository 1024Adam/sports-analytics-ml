"""NHL 32-team registry, tricode normalization, and arena metadata (`sports/hockey/constants.py`).

Full arena coordinates and schema registries scheduled for Milestone 2.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict


class ArenaMetadata(BaseModel):
    """NHL team arena geographic and timezone metadata."""

    model_config = ConfigDict(frozen=True)

    team: str
    arena_name: str
    latitude: float
    longitude: float
    timezone: str


TRICODE_ALIASES: Final[dict[str, str]] = {
    "S.J": "SJS",
    "L.A": "LAK",
    "N.J": "NJD",
    "T.B": "TBL",
    "ARI": "UTA",
    "PHX": "UTA",
    "ATL": "WPG",
}


def normalize_tricode(tricode: str) -> str:
    """Normalize historical or punctuation NHL team tricodes to canonical 3-letter codes."""
    cleaned = tricode.strip().upper()
    return TRICODE_ALIASES.get(cleaned, cleaned)
