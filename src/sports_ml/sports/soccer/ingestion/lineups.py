"""2-Tier Live Pre-Match Lineup, Starting GK & Injury/Suspension Resolver (`lineups.py`).

Tier 1: Official pre-match starting XI (`11` starters + `9` bench substitutes) posted
        ~60–75 minutes before kickoff (`CONFIRMED_OFFICIAL_XI`).
Tier 2: Daytime FPL official injury/suspension status (`status in {'a', 'd', 'i', 's', 'u'}`,
        `chance_of_playing_next_round in {0, 25, 50, 75, 100}`) + projected starting
        goalkeeper and outfield XI based on recent starts (`FPL_PROJECTED_XI`).
Fallback: Historical appearance heuristic (`HEURISTIC_XI`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from sports_ml.sports.soccer.constants import normalize_soccer_club

SoccerLineupStatus = Literal[
    "CONFIRMED_OFFICIAL_XI",
    "FPL_PROJECTED_XI",
    "HEURISTIC_XI",
]


def normalize_soccer_player_name(name: str) -> str:
    """Normalize player name by stripping diacritics, punctuation, and lowercasing."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", ascii_only).lower()
    return " ".join(cleaned.split())


class ResolvedSoccerTeamLineup(BaseModel):
    """Pre-match resolved starting goalkeeper, outfield squad, and injury/suspension exclusions."""

    model_config = ConfigDict(frozen=True)

    team: str
    starting_gk_id: int
    lineup_status: SoccerLineupStatus
    starting_outfield_ids: list[int]
    bench_outfield_ids: list[int]
    doubtful_player_ids: list[int]
    excluded_unavailable_ids: list[int]

    @property
    def active_outfield_ids(self) -> list[int]:
        """All available outfield players (starters + bench) eligible for Stage 2 projections."""
        seen: set[int] = set()
        ordered: list[int] = []
        for pid in [*self.starting_outfield_ids, *self.bench_outfield_ids]:
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
        return ordered


class SoccerPreMatchLineupResolver:
    """2-Tier waterfall resolver for pre-match EPL starting XI, GK, and injuries/suspensions."""

    @staticmethod
    def _try_tier1_official_lineup(
        team_code: str,
        official_lineup_payload: Mapping[str, Any] | None,
        squad_df: pl.DataFrame | None = None,
    ) -> ResolvedSoccerTeamLineup | None:
        if not official_lineup_payload:
            return None

        team_entry = official_lineup_payload.get(team_code)
        if not isinstance(team_entry, Mapping):
            # Also check if payload is a single match dict with home/away keys
            for side in ("home", "away", "homeTeam", "awayTeam"):
                candidate = official_lineup_payload.get(side)
                if isinstance(candidate, Mapping):
                    c_team = normalize_soccer_club(
                        str(candidate.get("team") or candidate.get("short_name") or "")
                    )
                    if c_team == team_code:
                        team_entry = candidate
                        break

        if not isinstance(team_entry, Mapping):
            return None

        gk_raw = team_entry.get("starting_gk_id")
        starters_raw = team_entry.get("starting_outfield_ids") or team_entry.get("starters") or []
        bench_raw = team_entry.get("bench_outfield_ids") or team_entry.get("bench") or []

        if gk_raw is None or not starters_raw:
            return None

        starting_gk_id = int(gk_raw)
        starting_outfield = [int(pid) for pid in starters_raw if int(pid) != starting_gk_id][:10]
        bench_outfield = [
            int(pid)
            for pid in bench_raw
            if int(pid) != starting_gk_id and int(pid) not in starting_outfield
        ]

        active_set = {starting_gk_id, *starting_outfield, *bench_outfield}
        excluded_ids: list[int] = []
        if squad_df is not None and not squad_df.is_empty():
            team_squad = squad_df.filter(pl.col("team") == team_code)
            excluded_ids = [
                int(pid) for pid in team_squad["player_id"].to_list() if int(pid) not in active_set
            ]

        return ResolvedSoccerTeamLineup(
            team=team_code,
            starting_gk_id=starting_gk_id,
            lineup_status="CONFIRMED_OFFICIAL_XI",
            starting_outfield_ids=starting_outfield,
            bench_outfield_ids=bench_outfield,
            doubtful_player_ids=[],
            excluded_unavailable_ids=sorted(excluded_ids),
        )

    @staticmethod
    def _try_tier2_fpl_squad(
        team_code: str,
        squad_df: pl.DataFrame | None,
    ) -> ResolvedSoccerTeamLineup | None:
        if squad_df is None or squad_df.is_empty():
            return None

        team_squad = squad_df.filter(pl.col("team") == team_code)
        if team_squad.is_empty():
            return None

        unavailable_mask = pl.col("status").is_in(["i", "s", "u", "n"]) | (
            pl.col("chance_of_playing_next_round") <= 25
        )
        excluded_df = team_squad.filter(unavailable_mask)
        available_df = team_squad.filter(~unavailable_mask)

        if available_df.is_empty():
            return None

        excluded_ids = [int(pid) for pid in excluded_df["player_id"].to_list()]
        doubtful_df = available_df.filter(
            (pl.col("status") == "d") | (pl.col("chance_of_playing_next_round") < 100)
        )
        doubtful_ids = [int(pid) for pid in doubtful_df["player_id"].to_list()]

        # Resolve starting GK: highest starts & season_minutes among available GKP
        gks = available_df.filter(pl.col("position_group") == "GKP").sort(
            ["starts", "season_minutes", "chance_of_playing_next_round"],
            descending=[True, True, True],
        )
        if not gks.is_empty():
            starting_gk_id = int(gks["player_id"][0])
        else:
            all_gks = team_squad.filter(pl.col("position_group") == "GKP").sort(
                ["starts", "season_minutes"], descending=[True, True]
            )
            starting_gk_id = int(all_gks["player_id"][0]) if not all_gks.is_empty() else 1

        outfield_avail = available_df.filter(pl.col("position_group") != "GKP").sort(
            ["starts", "season_minutes", "chance_of_playing_next_round", "player_id"],
            descending=[True, True, True, False],
        )
        outfield_ids = [int(pid) for pid in outfield_avail["player_id"].to_list()]
        starting_outfield = outfield_ids[:10]
        bench_outfield = outfield_ids[10:18]

        return ResolvedSoccerTeamLineup(
            team=team_code,
            starting_gk_id=starting_gk_id,
            lineup_status="FPL_PROJECTED_XI",
            starting_outfield_ids=starting_outfield,
            bench_outfield_ids=bench_outfield,
            doubtful_player_ids=sorted(doubtful_ids),
            excluded_unavailable_ids=sorted(excluded_ids),
        )

    def resolve_team_lineup(
        self,
        team: str,
        *,
        official_lineup_payload: Mapping[str, Any] | None = None,
        squad_df: pl.DataFrame | None = None,
        historical_outfield_logs: pl.DataFrame | None = None,
        historical_gk_logs: pl.DataFrame | None = None,
    ) -> ResolvedSoccerTeamLineup:
        """Resolve a club's starting GK, starting XI, bench, and injury/suspension exclusions."""
        team_code = normalize_soccer_club(team)

        # Tier 1: Confirmed pre-match starting XI (~60-75m before kickoff)
        tier1 = self._try_tier1_official_lineup(team_code, official_lineup_payload, squad_df)
        if tier1 is not None:
            return tier1

        # Tier 2: Daytime FPL squad availability & starts ranking
        tier2 = self._try_tier2_fpl_squad(team_code, squad_df)
        if tier2 is not None:
            return tier2

        # Fallback: Historical match logs heuristic
        starting_gk_id = 1
        if historical_gk_logs is not None and not historical_gk_logs.is_empty():
            team_gks = historical_gk_logs.filter(pl.col("team") == team_code)
            if not team_gks.is_empty():
                top_gk = (
                    team_gks.group_by("player_id")
                    .agg(pl.col("minutes").sum().alias("total_min"))
                    .sort(["total_min", "player_id"], descending=[True, False])
                )
                starting_gk_id = int(top_gk["player_id"][0])

        starting_outfield: list[int] = []
        bench_outfield: list[int] = []
        if historical_outfield_logs is not None and not historical_outfield_logs.is_empty():
            team_out = historical_outfield_logs.filter(pl.col("team") == team_code)
            if not team_out.is_empty():
                ranked_out = (
                    team_out.group_by("player_id")
                    .agg(pl.col("minutes").sum().alias("total_min"))
                    .sort(["total_min", "player_id"], descending=[True, False])
                )
                all_ids = [int(pid) for pid in ranked_out["player_id"].to_list()]
                starting_outfield = all_ids[:10]
                bench_outfield = all_ids[10:18]

        return ResolvedSoccerTeamLineup(
            team=team_code,
            starting_gk_id=starting_gk_id,
            lineup_status="HEURISTIC_XI",
            starting_outfield_ids=starting_outfield,
            bench_outfield_ids=bench_outfield,
            doubtful_player_ids=[],
            excluded_unavailable_ids=[],
        )
