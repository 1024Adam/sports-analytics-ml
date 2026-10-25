"""2-Tier Live Pre-Game Lineup, Starting Goalie & Injury Resolver (`ingestion/lineups.py`).

Tier 1: Official NHL API warmup/pre-game lineup and starter (~30m pre-puck-drop).
Tier 2: Daytime morning-skate starting goalie, line combinations, and injuries.
Full implementation scheduled for Milestone 2.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

GoalieConfirmationStatus = Literal[
    "CONFIRMED_NHL_OFFICIAL",
    "CONFIRMED_MORNING_SKATE",
    "EXPECTED_MORNING_SKATE",
    "PROJECTED_HEURISTIC",
]


class ResolvedTeamLineup(BaseModel):
    """Pre-game resolved active skaters, starting goalie, and injury exclusions."""

    model_config = ConfigDict(frozen=True)

    team: str
    starting_goalie_id: int
    goalie_status: GoalieConfirmationStatus
    active_skater_ids: list[int]
    excluded_injured_ids: list[int]
