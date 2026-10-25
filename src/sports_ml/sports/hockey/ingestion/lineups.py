"""2-Tier Live Pre-Game Lineup, Starting Goalie & Injury Resolver (`ingestion/lineups.py`).

Tier 1: Official NHL API warmup/pre-game lineup and starter (~30m pre-puck-drop).
Tier 2: Daytime morning-skate starting goalie (`Confirmed` / `Expected`), projected
        line combinations, and injury exclusions reconciled to NHL 7-digit `player_id`s.
Fallback: Active roster most recent lineup + primary non-B2B starter (`PROJECTED_HEURISTIC`).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from datetime import date
from typing import Any, Final, Literal

import httpx
import polars as pl
from pydantic import BaseModel, ConfigDict

from sports_ml.sports.hockey.constants import NHL_TEAMS_32, normalize_tricode
from sports_ml.sports.hockey.ingestion.nhl_api import parse_roster_payload

GoalieConfirmationStatus = Literal[
    "CONFIRMED_NHL_OFFICIAL",
    "CONFIRMED_MORNING_SKATE",
    "EXPECTED_MORNING_SKATE",
    "PROJECTED_HEURISTIC",
]

DAILYFACEOFF_STARTING_GOALIES_URL: Final[str] = "https://www.dailyfaceoff.com/starting-goalies"
_TEAM_NAME_TO_TRICODE: Final[dict[str, str]] = {
    meta.team_name.lower(): code for code, meta in NHL_TEAMS_32.items()
} | {
    "utah hockey club": "UTA",
    "utah mammoth": "UTA",
    "arizona coyotes": "UTA",
    "montreal canadiens": "MTL",
    "st. louis blues": "STL",
    "st louis blues": "STL",
}


def fetch_dailyfaceoff_starting_goalies(
    target_date: date,
    *,
    timeout_seconds: float = 15.0,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch DailyFaceoff `__NEXT_DATA__` starting goalies for `target_date` keyed by tricode."""
    url = f"{DAILYFACEOFF_STARTING_GOALIES_URL}/{target_date.isoformat()}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        )
    }
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
            headers=headers,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return {}

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
    )
    if not match:
        return {}

    try:
        payload = json.loads(match.group(1))
        items = payload.get("props", {}).get("pageProps", {}).get("data", [])
    except Exception:
        return {}

    resolved: dict[str, dict[str, Any]] = {}
    for item in items:
        for side_prefix in ("home", "away"):
            raw_team_name = str(item.get(f"{side_prefix}TeamName") or "").strip().lower()
            team_code = _TEAM_NAME_TO_TRICODE.get(raw_team_name)
            if not team_code:
                continue
            goalie_name = str(item.get(f"{side_prefix}GoalieName") or "").strip()
            strength = str(item.get(f"{side_prefix}NewsStrengthName") or "Expected").strip()
            if goalie_name:
                resolved[team_code] = {
                    "starting_goalie_name": goalie_name,
                    "goalie_status": strength,
                }
    return resolved


class ResolvedTeamLineup(BaseModel):
    """Pre-game resolved active skaters, starting goalie, and injury exclusions."""

    model_config = ConfigDict(frozen=True)

    team: str
    starting_goalie_id: int
    goalie_status: GoalieConfirmationStatus
    active_skater_ids: list[int]
    excluded_injured_ids: list[int]


def normalize_player_name(name: str) -> str:
    """Normalize player name by stripping diacritics, punctuation, and lowercasing."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", ascii_only).lower()
    return " ".join(cleaned.split())


class PreGameLineupResolver:
    """2-Tier waterfall resolver for pre-game dressed skaters, starting goalies, and injuries."""

    @staticmethod
    def _try_tier1_official_boxscore(
        team_code: str,
        boxscore_payload: dict[str, Any] | None,
        roster_df: pl.DataFrame,
    ) -> ResolvedTeamLineup | None:
        if not boxscore_payload:
            return None

        home_code = normalize_tricode(
            str((boxscore_payload.get("homeTeam") or {}).get("abbrev", ""))
        )
        away_code = normalize_tricode(
            str((boxscore_payload.get("awayTeam") or {}).get("abbrev", ""))
        )
        if team_code == home_code:
            side_key = "homeTeam"
        elif team_code == away_code:
            side_key = "awayTeam"
        else:
            return None

        pbg = (boxscore_payload.get("playerByGameStats") or {}).get(side_key) or {}
        forwards = pbg.get("forwards", [])
        defense = pbg.get("defense", [])
        goalies = pbg.get("goalies", [])

        dressed_skaters = [int(p["playerId"]) for p in [*forwards, *defense] if "playerId" in p]
        starter_goalies = [
            int(g["playerId"]) for g in goalies if bool(g.get("starter")) and "playerId" in g
        ]

        if not dressed_skaters or not starter_goalies:
            return None

        roster_skater_ids = set(
            roster_df.filter(pl.col("position_group") != "G")["player_id"].to_list()
        )
        dressed_set = set(dressed_skaters)
        excluded_ids = sorted(roster_skater_ids - dressed_set)

        return ResolvedTeamLineup(
            team=team_code,
            starting_goalie_id=starter_goalies[0],
            goalie_status="CONFIRMED_NHL_OFFICIAL",
            active_skater_ids=dressed_skaters,
            excluded_injured_ids=excluded_ids,
        )

    @staticmethod
    def _try_tier2_morning_skate(
        team_code: str,
        morning_skate_payload: dict[str, Any] | None,
        roster_df: pl.DataFrame,
    ) -> ResolvedTeamLineup | None:
        if not morning_skate_payload or roster_df.is_empty():
            return None

        name_to_id: dict[str, int] = {}
        for row in roster_df.iter_rows(named=True):
            norm_name = normalize_player_name(str(row["player_name"]))
            name_to_id[norm_name] = int(row["player_id"])

        starter_name = str(morning_skate_payload.get("starting_goalie_name", ""))
        starter_id_raw = morning_skate_payload.get("starting_goalie_id")
        if starter_id_raw is not None:
            starter_id = int(starter_id_raw)
        else:
            starter_id = name_to_id.get(normalize_player_name(starter_name), -1)

        if starter_id <= 0:
            return None

        raw_status = str(morning_skate_payload.get("goalie_status", "EXPECTED")).upper()
        goalie_status: GoalieConfirmationStatus = (
            "CONFIRMED_MORNING_SKATE" if "CONFIRM" in raw_status else "EXPECTED_MORNING_SKATE"
        )

        injured_names: Sequence[str] = morning_skate_payload.get("injured_players", [])
        injured_ids_raw: Sequence[int] = morning_skate_payload.get("injured_player_ids", [])
        injured_set: set[int] = {int(pid) for pid in injured_ids_raw}
        for inj_name in injured_names:
            matched_id = name_to_id.get(normalize_player_name(str(inj_name)))
            if matched_id is not None:
                injured_set.add(matched_id)

        projected_names: Sequence[str] = morning_skate_payload.get("projected_skaters", [])
        all_skaters_df = roster_df.filter(pl.col("position_group") != "G")

        if projected_names:
            active_skater_ids: list[int] = []
            for p_name in projected_names:
                pid = name_to_id.get(normalize_player_name(str(p_name)))
                if pid is not None and pid not in injured_set and pid not in active_skater_ids:
                    active_skater_ids.append(pid)
        else:
            active_skater_ids = [
                int(pid)
                for pid in all_skaters_df["player_id"].to_list()
                if int(pid) not in injured_set
            ]

        return ResolvedTeamLineup(
            team=team_code,
            starting_goalie_id=starter_id,
            goalie_status=goalie_status,
            active_skater_ids=active_skater_ids,
            excluded_injured_ids=sorted(injured_set),
        )

    @staticmethod
    def _fallback_heuristic(
        team_code: str,
        roster_df: pl.DataFrame,
        recent_goalie_logs: pl.DataFrame | None = None,
        is_b2b: bool = False,
    ) -> ResolvedTeamLineup:
        skater_ids = [
            int(pid)
            for pid in roster_df.filter(pl.col("position_group") != "G")["player_id"].to_list()
        ]
        goalie_ids = [
            int(pid)
            for pid in roster_df.filter(pl.col("position_group") == "G")["player_id"].to_list()
        ]

        chosen_goalie = goalie_ids[0] if goalie_ids else -1
        if recent_goalie_logs is not None and not recent_goalie_logs.is_empty() and goalie_ids:
            team_g_logs = recent_goalie_logs.filter(
                (pl.col("team") == team_code) & pl.col("player_id").is_in(goalie_ids)
            ).sort(["start_time_utc", "game_id"])
            if not team_g_logs.is_empty():
                last_starter = int(team_g_logs["player_id"][-1])
                counts = (
                    team_g_logs.filter(pl.col("is_starter"))
                    .group_by("player_id")
                    .len()
                    .sort(["len", "player_id"], descending=[True, False])
                )
                ordered_starters = [int(pid) for pid in counts["player_id"].to_list()]
                if not ordered_starters:
                    ordered_starters = goalie_ids

                if is_b2b and len(ordered_starters) > 1 and ordered_starters[0] == last_starter:
                    chosen_goalie = ordered_starters[1]
                else:
                    chosen_goalie = ordered_starters[0]

        return ResolvedTeamLineup(
            team=team_code,
            starting_goalie_id=chosen_goalie,
            goalie_status="PROJECTED_HEURISTIC",
            active_skater_ids=skater_ids,
            excluded_injured_ids=[],
        )

    def resolve_team_lineup(
        self,
        team_tricode: str,
        roster_payload_or_df: dict[str, Any] | pl.DataFrame,
        *,
        boxscore_payload: dict[str, Any] | None = None,
        morning_skate_payload: dict[str, Any] | None = None,
        recent_goalie_logs: pl.DataFrame | None = None,
        is_b2b: bool = False,
    ) -> ResolvedTeamLineup:
        """Resolve pre-game starting goalie, active skaters, and injuries via 2-Tier waterfall."""
        team_code = normalize_tricode(team_tricode)
        roster_df = (
            parse_roster_payload(team_code, roster_payload_or_df)
            if isinstance(roster_payload_or_df, dict)
            else roster_payload_or_df
        )

        tier1 = self._try_tier1_official_boxscore(team_code, boxscore_payload, roster_df)
        if tier1 is not None:
            return tier1

        tier2 = self._try_tier2_morning_skate(team_code, morning_skate_payload, roster_df)
        if tier2 is not None:
            return tier2

        return self._fallback_heuristic(
            team_code,
            roster_df,
            recent_goalie_logs=recent_goalie_logs,
            is_b2b=is_b2b,
        )
