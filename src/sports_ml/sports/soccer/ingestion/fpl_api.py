"""Official Fantasy Premier League (FPL) / Premier League API client & parsers (`fpl_api.py`).

Provides deterministic disk caching (`data/raw/fpl_api/`), `tenacity` retry/backoff,
and Polars normalization for:
- `/api/bootstrap-static/`: 20 active EPL clubs, squad players, positions (`GKP`, `DEF`, `MID`,
  `FWD`), set-piece/penalty taker order, and injury/suspension flags (`status`,
  `chance_of_playing_next_round`).
- `/api/fixtures/`: Full 380-match season schedule, matchweek (`event`) assignments, UTC kickoffs,
  finished flags, and official scores.
- `/api/element-summary/{player_id}/` & `/api/event/{gw}/live/`: Match-by-match player logs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sports_ml.core.cache import read_cached_json, write_cached_json
from sports_ml.sports.soccer.constants import (
    SILVER_GK_MATCH_LOGS_SCHEMA,
    SILVER_MATCHES_SCHEMA,
    SILVER_OUTFIELD_MATCH_LOGS_SCHEMA,
    normalize_soccer_club,
)

FPL_API_BASE_URL: Final[str] = "https://fantasy.premierleague.com/api"
DEFAULT_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)

ELEMENT_TYPE_TO_POS_GROUP: Final[dict[int, str]] = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


def derive_active_epl_season(ref_date: date | None = None) -> int:
    """Derive 8-digit EPL season identifier (e.g. `20242025` or `20262027`) from date.

    The EPL season runs August through May; dates in July or later belong to the
    season starting that calendar year.
    """
    today = ref_date or datetime.now(tz=UTC).date()
    start_year = today.year if today.month >= 7 else today.year - 1
    return int(f"{start_year}{start_year + 1}")


def make_canonical_match_id(season: int, fixture_id: int) -> int:
    """Create a globally unique 10-digit canonical `match_id` from `season` and `fixture_id`.

    If `fixture_id` is already >= 1_000_000, returns it unchanged; otherwise prefixes
    with the 4-digit season start year (`YYYY * 100_000 + fixture_id`).
    """
    if fixture_id >= 1_000_000:
        return int(fixture_id)
    start_yr = int(str(season)[:4])
    return start_yr * 100_000 + int(fixture_id)


def parse_bootstrap_teams_map(bootstrap_payload: Mapping[str, Any]) -> dict[int, str]:
    """Extract mapping from FPL numeric `team_id` (`1..20`) to canonical 3-letter club code."""
    teams = bootstrap_payload.get("teams", [])
    mapping: dict[int, str] = {}
    for t in teams:
        t_id = int(t["id"])
        raw_short = str(t.get("short_name") or t.get("name") or "")
        mapping[t_id] = normalize_soccer_club(raw_short)
    return mapping


def parse_bootstrap_squad_df(bootstrap_payload: Mapping[str, Any]) -> pl.DataFrame:
    """Parse `/api/bootstrap-static/` elements into a canonical squad & availability DataFrame."""
    teams_map = parse_bootstrap_teams_map(bootstrap_payload)
    elements = bootstrap_payload.get("elements", [])

    rows: list[dict[str, Any]] = []
    for el in elements:
        player_id = int(el["id"])
        team_id = int(el.get("team", 0))
        team_code = teams_map.get(team_id, normalize_soccer_club(str(el.get("team_code", ""))))
        el_type = int(el.get("element_type", 3))
        pos_group = ELEMENT_TYPE_TO_POS_GROUP.get(el_type, "MID")

        first = str(el.get("first_name") or "").strip()
        second = str(el.get("second_name") or "").strip()
        web_name = str(el.get("web_name") or "").strip()
        full_name = f"{first} {second}".strip() or web_name

        status = str(el.get("status") or "a").strip().lower()
        raw_chance = el.get("chance_of_playing_next_round")
        if raw_chance is None:
            chance = 100 if status == "a" else (75 if status == "d" else 0)
        else:
            chance = int(raw_chance)

        pen_order = el.get("penalties_order")
        fk_order = el.get("direct_freekicks_order")
        corner_order = el.get("corners_and_indirect_freekicks_order")

        rows.append(
            {
                "player_id": player_id,
                "player_name": full_name,
                "web_name": web_name,
                "team": team_code,
                "position_group": pos_group,
                "status": status,
                "chance_of_playing_next_round": chance,
                "penalties_order": int(pen_order) if pen_order is not None else 99,
                "direct_freekicks_order": int(fk_order) if fk_order is not None else 99,
                "corners_order": int(corner_order) if corner_order is not None else 99,
                "season_minutes": float(el.get("minutes") or 0.0),
                "starts": int(el.get("starts") or 0),
            }
        )

    schema = {
        "player_id": pl.Int64(),
        "player_name": pl.Utf8(),
        "web_name": pl.Utf8(),
        "team": pl.Utf8(),
        "position_group": pl.Utf8(),
        "status": pl.Utf8(),
        "chance_of_playing_next_round": pl.Int16(),
        "penalties_order": pl.Int16(),
        "direct_freekicks_order": pl.Int16(),
        "corners_order": pl.Int16(),
        "season_minutes": pl.Float64(),
        "starts": pl.Int16(),
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(["team", "player_id"])


def parse_fixtures_payload(
    fixtures_payload: Sequence[Mapping[str, Any]],
    teams_map: Mapping[int, str],
    *,
    season: int,
) -> pl.DataFrame:
    """Parse `/api/fixtures/` list into canonical `SILVER_MATCHES_SCHEMA`."""
    rows: list[dict[str, Any]] = []
    for fx in fixtures_payload:
        kickoff_raw = fx.get("kickoff_time")
        if not kickoff_raw:
            continue
        dt_utc = datetime.fromisoformat(str(kickoff_raw).replace("Z", "+00:00")).astimezone(UTC)
        match_date = dt_utc.date()

        raw_id = int(fx["id"])
        match_id = make_canonical_match_id(season, raw_id)
        matchweek = int(fx.get("event") or 1)

        team_h_raw = fx.get("team_h")
        team_a_raw = fx.get("team_a")
        home_team = (
            teams_map[int(team_h_raw)]
            if isinstance(team_h_raw, int) and int(team_h_raw) in teams_map
            else normalize_soccer_club(str(fx.get("home_team") or team_h_raw or ""))
        )
        away_team = (
            teams_map[int(team_a_raw)]
            if isinstance(team_a_raw, int) and int(team_a_raw) in teams_map
            else normalize_soccer_club(str(fx.get("away_team") or team_a_raw or ""))
        )

        finished = bool(fx.get("finished") or fx.get("finished_provisional"))
        h_score_raw = fx.get("team_h_score")
        a_score_raw = fx.get("team_a_score")

        if finished and h_score_raw is not None and a_score_raw is not None:
            game_state = "FINAL"
            h_goals = int(h_score_raw)
            a_goals = int(a_score_raw)
            if h_goals > a_goals:
                outcome_1x2 = 0
                home_win = 1
                is_draw = 0
                away_win = 0
            elif h_goals == a_goals:
                outcome_1x2 = 1
                home_win = 0
                is_draw = 1
                away_win = 0
            else:
                outcome_1x2 = 2
                home_win = 0
                is_draw = 0
                away_win = 1
        else:
            game_state = "FUT"
            h_goals = None
            a_goals = None
            outcome_1x2 = None
            home_win = None
            is_draw = None
            away_win = None

        rows.append(
            {
                "sport": "soccer",
                "season": int(season),
                "matchweek": matchweek,
                "match_id": match_id,
                "event_seq": int(fx.get("event_seq") or 1),
                "match_date": match_date,
                "start_time_utc": dt_utc,
                "home_team": home_team,
                "away_team": away_team,
                "venue_timezone": "Europe/London",
                "game_state": game_state,
                "home_goals": h_goals,
                "away_goals": a_goals,
                "outcome_1x2": outcome_1x2,
                "home_win": home_win,
                "is_draw": is_draw,
                "away_win": away_win,
            }
        )

    if not rows:
        return pl.DataFrame(schema=SILVER_MATCHES_SCHEMA)
    return pl.DataFrame(rows, schema=SILVER_MATCHES_SCHEMA).sort(
        ["match_date", "start_time_utc", "event_seq", "match_id"]
    )


def parse_element_summary_payload(
    player_id: int,
    player_name: str,
    team_code: str,
    position_group: str,
    summary_payload: Mapping[str, Any],
    teams_map: Mapping[int, str],
    *,
    season: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse `/api/element-summary/{player_id}/` `history` into outfield and GK match logs."""
    history = summary_payload.get("history", [])
    outfield_rows: list[dict[str, Any]] = []
    gk_rows: list[dict[str, Any]] = []

    for h in history:
        minutes = float(h.get("minutes") or 0.0)
        if minutes <= 0.0:
            continue

        kickoff_raw = h.get("kickoff_time")
        if not kickoff_raw:
            continue
        dt_utc = datetime.fromisoformat(str(kickoff_raw).replace("Z", "+00:00")).astimezone(UTC)
        match_date = dt_utc.date()

        fixture_id = int(h["fixture"])
        match_id = make_canonical_match_id(season, fixture_id)
        opp_id = int(h.get("opponent_team") or 0)
        opponent = teams_map.get(opp_id, normalize_soccer_club(str(h.get("opponent") or opp_id)))
        is_home = bool(h.get("was_home", True))
        starts_val = h.get("starts")
        is_starter = bool(int(starts_val) >= 1) if starts_val is not None else (minutes >= 60.0)

        xg = float(h.get("expected_goals") or 0.0)
        xa = float(h.get("expected_assists") or 0.0)
        xgi = float(h.get("expected_goal_involvements") or (xg + xa))
        xgc = float(h.get("expected_goals_conceded") or 0.0)

        if position_group == "GKP":
            saves = int(h.get("saves") or 0)
            ga = int(h.get("goals_conceded") or 0)
            sot_against = int(h.get("shots_on_target_against") or (saves + ga))
            cs = int(h.get("clean_sheets") or (1 if (minutes >= 60.0 and ga == 0) else 0))
            xga_faced = xgc if xgc > 0.0 else float(ga)
            gk_rows.append(
                {
                    "match_id": match_id,
                    "match_date": match_date,
                    "start_time_utc": dt_utc,
                    "season": int(season),
                    "player_id": int(player_id),
                    "player_name": player_name,
                    "team": normalize_soccer_club(team_code),
                    "opponent": opponent,
                    "is_home": is_home,
                    "is_starter": is_starter,
                    "minutes": minutes,
                    "shots_on_target_against": sot_against,
                    "saves": saves,
                    "goals_against": ga,
                    "clean_sheet": cs,
                    "xga_faced": xga_faced,
                    "gsaa": xga_faced - float(ga),
                }
            )
        else:
            goals = int(h.get("goals_scored") or 0)
            assists = int(h.get("assists") or 0)
            shots = int(h.get("shots") or max(goals, round(xg / 0.11)))
            sot = int(h.get("sot") or max(goals, round(shots * 0.38)))
            key_passes = int(h.get("key_passes") or max(assists, round(xa / 0.10)))
            npxg = float(h.get("npxg") if h.get("npxg") is not None else xg)
            xg_chain = float(h.get("xg_chain") if h.get("xg_chain") is not None else xgi)
            xg_buildup = float(
                h.get("xg_buildup") if h.get("xg_buildup") is not None else max(0.0, xgi - xg)
            )
            outfield_rows.append(
                {
                    "match_id": match_id,
                    "match_date": match_date,
                    "start_time_utc": dt_utc,
                    "season": int(season),
                    "player_id": int(player_id),
                    "player_name": player_name,
                    "team": normalize_soccer_club(team_code),
                    "opponent": opponent,
                    "is_home": is_home,
                    "position_group": position_group,
                    "is_starter": is_starter,
                    "minutes": minutes,
                    "goals": goals,
                    "assists": assists,
                    "shots": shots,
                    "sot": sot,
                    "key_passes": key_passes,
                    "xg": xg,
                    "npxg": npxg,
                    "xa": xa,
                    "xg_chain": xg_chain,
                    "xg_buildup": xg_buildup,
                    "yellow_cards": int(h.get("yellow_cards") or 0),
                    "red_cards": int(h.get("red_cards") or 0),
                }
            )

    outfield_df = (
        pl.DataFrame(outfield_rows, schema=SILVER_OUTFIELD_MATCH_LOGS_SCHEMA)
        if outfield_rows
        else pl.DataFrame(schema=SILVER_OUTFIELD_MATCH_LOGS_SCHEMA)
    )
    gk_df = (
        pl.DataFrame(gk_rows, schema=SILVER_GK_MATCH_LOGS_SCHEMA)
        if gk_rows
        else pl.DataFrame(schema=SILVER_GK_MATCH_LOGS_SCHEMA)
    )
    return outfield_df, gk_df


class FPLAPIClient:
    """HTTP client with disk caching and retry/backoff for Official FPL / Premier League API."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        base_url: str = FPL_API_BASE_URL,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get_json(self, path: str) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            transport=self.transport,
            headers=headers,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()

    def _read_cached_json(self, cache_path: Path, ttl_seconds: float | None) -> Any | None:
        return read_cached_json(cache_path, ttl_seconds=ttl_seconds)

    def fetch_bootstrap_static(self, *, ttl_seconds: float | None = 3600.0) -> dict[str, Any]:
        """Fetch `/api/bootstrap-static/` with disk cache."""
        cache_path = self.cache_dir / "bootstrap_static.json.gz"
        cached = self._read_cached_json(cache_path, ttl_seconds)
        if isinstance(cached, dict):
            return cached
        payload = self._get_json("bootstrap-static/")
        write_cached_json(cache_path, payload)
        return dict(payload)

    def fetch_fixtures(self, *, ttl_seconds: float | None = 1800.0) -> list[dict[str, Any]]:
        """Fetch `/api/fixtures/` list with disk cache."""
        cache_path = self.cache_dir / "fixtures.json.gz"
        cached = self._read_cached_json(cache_path, ttl_seconds)
        if isinstance(cached, list):
            return cached
        payload = self._get_json("fixtures/")
        write_cached_json(cache_path, payload)
        return list(payload)

    def fetch_element_summary(
        self,
        player_id: int,
        *,
        ttl_seconds: float | None = 21600.0,
    ) -> dict[str, Any]:
        """Fetch `/api/element-summary/{player_id}/` with disk cache."""
        summaries_dir = self.cache_dir / "element_summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        cache_path = summaries_dir / f"{player_id}.json.gz"
        cached = self._read_cached_json(cache_path, ttl_seconds)
        if isinstance(cached, dict):
            return cached
        payload = self._get_json(f"element-summary/{player_id}/")
        write_cached_json(cache_path, payload)
        return dict(payload)
