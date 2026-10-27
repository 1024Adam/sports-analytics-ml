"""Official NHL Web API (`https://api-web.nhle.com/v1/`) client (`ingestion/nhl_api.py`).

Provides deterministic disk caching (`data/raw/nhl_api/`) and `tenacity` retry/backoff
for schedules, boxscores, landing pages, and team rosters.
Supports permanent caching for `FINAL`/`OFF` games and incremental delta fetching
(`--current-season` / `--start-date` / `--end-date`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sports_ml.core.cache import read_cached_json, write_cached_json
from sports_ml.sports.hockey.constants import (
    SILVER_GAMES_SCHEMA,
    SILVER_GOALIE_GAME_LOGS_SCHEMA,
    SILVER_SKATER_GAME_LOGS_SCHEMA,
    SILVER_TEAM_GAME_LOGS_SCHEMA,
    get_arena_metadata,
    normalize_tricode,
)

NHL_WEB_API_BASE_URL: Final[str] = "https://api-web.nhle.com/v1"
FINAL_GAME_STATES: Final[frozenset[str]] = frozenset({"FINAL", "OFF"})


def parse_toi_minutes(toi_str: str | float | int | None) -> float:
    """Convert NHL API `'MM:SS'` string (or numeric minutes) to decimal minutes."""
    if toi_str is None:
        return 0.0
    if isinstance(toi_str, (int, float)):
        return float(toi_str)
    cleaned = str(toi_str).strip()
    if not cleaned:
        return 0.0
    if ":" not in cleaned:
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    parts = cleaned.split(":")
    try:
        minutes = int(parts[0])
        seconds = int(parts[1]) if len(parts) > 1 else 0
        return float(minutes) + float(seconds) / 60.0
    except ValueError:
        return 0.0


def derive_active_nhl_season(ref_date: date | None = None) -> int:
    """Derive 8-digit NHL season ID (e.g., `20262027`) from a calendar date."""
    target = ref_date or datetime.now(tz=UTC).date()
    start_year = target.year if target.month >= 8 else target.year - 1
    return int(f"{start_year}{start_year + 1}")


def compute_single_game_score(
    goals: int,
    assists: int,
    sog: int,
    blocked_shots: int = 0,
    pim: int = 0,
) -> float:
    """Compute baseline boxscore Game Score approximation when MoneyPuck inputs are absent."""
    return (
        0.75 * float(goals)
        + 0.63 * float(assists)
        + 0.075 * float(sog)
        + 0.05 * float(blocked_shots)
        - 0.075 * float(pim)
    )


def parse_schedule_payload(
    payload: dict[str, Any],
    *,
    allowed_game_types: Sequence[int] = (2, 3),
) -> pl.DataFrame:
    """Parse an NHL Web API `/v1/schedule/{date}` or club-schedule payload into games."""
    rows: list[dict[str, Any]] = []
    allowed_set = set(allowed_game_types)

    game_entries: list[tuple[str | None, dict[str, Any]]] = []
    if "gameWeek" in payload and isinstance(payload["gameWeek"], list):
        for day_block in payload["gameWeek"]:
            day_str = day_block.get("date")
            for game in day_block.get("games", []):
                game_entries.append((day_str, game))
    elif "games" in payload and isinstance(payload["games"], list):
        for game in payload["games"]:
            game_entries.append((game.get("gameDate"), game))

    for day_str, game in game_entries:
        game_type = int(game.get("gameType", 2))
        if game_type not in allowed_set:
            continue

        game_id = int(game["id"])
        season = int(game.get("season") or f"{str(game_id)[:4]}{int(str(game_id)[:4]) + 1}")
        start_str = str(game.get("startTimeUTC") or f"{day_str}T23:00:00Z")
        start_utc = datetime.fromisoformat(start_str.replace("Z", "+00:00")).astimezone(UTC)
        game_date_val = date.fromisoformat(str(day_str)) if day_str else start_utc.date()

        home_obj = game.get("homeTeam", {})
        away_obj = game.get("awayTeam", {})
        home_team = normalize_tricode(str(home_obj.get("abbrev", "")))
        away_team = normalize_tricode(str(away_obj.get("abbrev", "")))

        arena_meta = get_arena_metadata(home_team)
        venue_tz = str(
            game.get("venueTimezone")
            or (arena_meta.timezone if arena_meta is not None else "America/New_York")
        )

        raw_state = str(game.get("gameState", "FUT")).upper()
        is_completed = raw_state in FINAL_GAME_STATES
        canonical_state = "FINAL" if is_completed else "FUT"

        outcome = game.get("gameOutcome") or {}
        period_desc = game.get("periodDescriptor") or {}
        raw_period_type = outcome.get("lastPeriodType") or period_desc.get("periodType")
        finish_period_type = (
            str(raw_period_type).upper() if (is_completed and raw_period_type) else None
        )

        home_score_raw = home_obj.get("score")
        away_score_raw = away_obj.get("score")

        if is_completed and home_score_raw is not None and away_score_raw is not None:
            home_goals = int(home_score_raw)
            away_goals = int(away_score_raw)
            home_win = 1 if home_goals > away_goals else 0
            if finish_period_type == "SO":
                home_reg_goals = home_goals - (1 if home_win == 1 else 0)
                away_reg_goals = away_goals - (1 if home_win == 0 else 0)
            else:
                home_reg_goals = home_goals
                away_reg_goals = away_goals
        else:
            home_goals = None
            away_goals = None
            home_reg_goals = None
            away_reg_goals = None
            home_win = None

        rows.append(
            {
                "sport": "hockey",
                "season": season,
                "game_type": game_type,
                "game_id": game_id,
                "game_date": game_date_val,
                "start_time_utc": start_utc,
                "home_team": home_team,
                "away_team": away_team,
                "venue_timezone": venue_tz,
                "game_state": canonical_state,
                "finish_period_type": finish_period_type,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_reg_goals": home_reg_goals,
                "away_reg_goals": away_reg_goals,
                "home_win": home_win,
            }
        )

    if not rows:
        return pl.DataFrame(schema=SILVER_GAMES_SCHEMA)

    df = pl.DataFrame(rows, schema=SILVER_GAMES_SCHEMA)
    return df.unique(subset=["game_id"], keep="last").sort(
        ["game_date", "start_time_utc", "game_id"]
    )


def parse_boxscore_payload(
    boxscore: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parse an NHL Web API `/v1/gamecenter/{GAME_ID}/boxscore` payload.

    Returns `(team_game_logs_df, skater_game_logs_df, goalie_game_logs_df)` adhering to
    Silver Parquet schemas (with MoneyPuck columns defaulted to fallback values).
    """
    game_id = int(boxscore["id"])
    season = int(boxscore.get("season") or f"{str(game_id)[:4]}{int(str(game_id)[:4]) + 1}")
    start_str = str(
        boxscore.get("startTimeUTC") or f"{boxscore.get('gameDate', '2024-10-10')}T23:00:00Z"
    )
    start_utc = datetime.fromisoformat(start_str.replace("Z", "+00:00")).astimezone(UTC)
    game_date_str = boxscore.get("gameDate")
    game_date_val = date.fromisoformat(str(game_date_str)) if game_date_str else start_utc.date()

    outcome = boxscore.get("gameOutcome") or {}
    period_desc = boxscore.get("periodDescriptor") or {}
    finish_type = str(
        outcome.get("lastPeriodType") or period_desc.get("periodType") or "REG"
    ).upper()

    home_meta = boxscore.get("homeTeam", {})
    away_meta = boxscore.get("awayTeam", {})
    home_team = normalize_tricode(str(home_meta.get("abbrev", "")))
    away_team = normalize_tricode(str(away_meta.get("abbrev", "")))

    home_score = int(home_meta.get("score", 0))
    away_score = int(away_meta.get("score", 0))
    if finish_type == "SO":
        if home_score > away_score:
            home_reg_goals = home_score - 1
            away_reg_goals = away_score
        else:
            home_reg_goals = home_score
            away_reg_goals = max(0, away_score - 1)
    else:
        home_reg_goals = home_score
        away_reg_goals = away_score

    player_by_game = boxscore.get("playerByGameStats") or {}

    skater_rows: list[dict[str, Any]] = []
    goalie_rows: list[dict[str, Any]] = []
    team_pp_goals: dict[str, float] = {home_team: 0.0, away_team: 0.0}
    team_sog: dict[str, int] = {
        home_team: int(home_meta.get("sog", 0)),
        away_team: int(away_meta.get("sog", 0)),
    }

    side_configs = [
        ("homeTeam", home_team, away_team, True),
        ("awayTeam", away_team, home_team, False),
    ]

    for side_key, team_code, opp_code, is_home in side_configs:
        side_stats = player_by_game.get(side_key) or {}
        forwards = side_stats.get("forwards", [])
        defense = side_stats.get("defense", [])
        goalies = side_stats.get("goalies", [])

        summed_sog = 0
        for pos_group, group_list in (("F", forwards), ("D", defense)):
            for p in group_list:
                player_id = int(p["playerId"])
                name_obj = p.get("name", {})
                player_name = (
                    str(name_obj.get("default"))
                    if isinstance(name_obj, dict) and "default" in name_obj
                    else str(p.get("name", f"Player {player_id}"))
                )
                toi_total = parse_toi_minutes(p.get("toi"))
                toi_pp = parse_toi_minutes(p.get("powerPlayToi"))
                toi_pk = parse_toi_minutes(p.get("shorthandedToi"))
                pp_g = int(p.get("powerPlayGoals", 0))
                if toi_pp == 0.0 and pp_g > 0 and toi_total > 0.0:
                    toi_pp = min(toi_total * 0.20, 3.5)
                toi_ev = max(0.0, toi_total - toi_pp - toi_pk)

                goals = int(p.get("goals", 0))
                assists = int(p.get("assists", 0))
                points = int(p.get("points", goals + assists))
                sog = int(p.get("sog", 0))
                pp_g = int(p.get("powerPlayGoals", 0))
                blocked = int(p.get("blockedShots", 0))
                pim = int(p.get("pim", 0))

                summed_sog += sog
                team_pp_goals[team_code] += float(pp_g)

                # Fallback individual xG approximation (~0.095 per SOG) until MoneyPuck joins
                approx_ixg_all = round(0.095 * float(sog), 4)
                ev_share = (toi_ev / toi_total) if toi_total > 0.0 else 0.85
                approx_ixg_5v5 = round(approx_ixg_all * ev_share, 4)
                gs_val = compute_single_game_score(goals, assists, sog, blocked, pim)

                skater_rows.append(
                    {
                        "game_id": game_id,
                        "game_date": game_date_val,
                        "start_time_utc": start_utc,
                        "season": season,
                        "player_id": player_id,
                        "player_name": player_name,
                        "team": team_code,
                        "opponent": opp_code,
                        "is_home": is_home,
                        "position_group": pos_group,
                        "toi_total_min": toi_total,
                        "toi_ev_min": toi_ev,
                        "toi_pp_min": toi_pp,
                        "toi_pk_min": toi_pk,
                        "goals": goals,
                        "assists": assists,
                        "points": points,
                        "sog": sog,
                        "ixg_all": approx_ixg_all,
                        "ixg_5v5": approx_ixg_5v5,
                        "icf_all": float(sog + blocked),
                        "on_ice_xgf_5v5": 0.0,
                        "on_ice_xga_5v5": 0.0,
                        "game_score": gs_val,
                    }
                )

        if team_sog[team_code] == 0 and summed_sog > 0:
            team_sog[team_code] = summed_sog

        # Determine starter goalie (check explicit `starter` flag, fallback to max TOI)
        has_explicit_starter = any(bool(g.get("starter")) for g in goalies)
        max_toi_idx = -1
        max_toi_val = -1.0
        for g_idx, g in enumerate(goalies):
            g_toi = parse_toi_minutes(g.get("toi"))
            if g_toi > max_toi_val:
                max_toi_val = g_toi
                max_toi_idx = g_idx

        for g_idx, g in enumerate(goalies):
            g_id = int(g["playerId"])
            g_toi = parse_toi_minutes(g.get("toi"))
            if g_toi <= 0.0 and not bool(g.get("starter")):
                # Backup goalie who did not play
                continue

            is_starter = bool(g.get("starter")) if has_explicit_starter else (g_idx == max_toi_idx)
            sa = int(g.get("shotsAgainst", 0))
            ga = int(g.get("goalsAgainst", 0))
            saves = int(g.get("saves", max(0, sa - ga)))
            approx_xga = round(0.095 * float(sa), 4)
            approx_gsax = round(approx_xga - float(ga), 4)

            goalie_rows.append(
                {
                    "game_id": game_id,
                    "game_date": game_date_val,
                    "start_time_utc": start_utc,
                    "season": season,
                    "player_id": g_id,
                    "team": team_code,
                    "opponent": opp_code,
                    "is_home": is_home,
                    "is_starter": is_starter,
                    "toi_min": g_toi,
                    "shots_against": sa,
                    "saves": saves,
                    "goals_against": ga,
                    "xga_all": approx_xga,
                    "gsax": approx_gsax,
                }
            )

    home_sog = team_sog[home_team]
    away_sog = team_sog[away_team]

    team_rows: list[dict[str, Any]] = [
        {
            "game_id": game_id,
            "game_date": game_date_val,
            "start_time_utc": start_utc,
            "season": season,
            "team": home_team,
            "opponent": away_team,
            "is_home": True,
            "goals_for": home_reg_goals,
            "goals_against": away_reg_goals,
            "sog_for": home_sog,
            "sog_against": away_sog,
            "xgf_5v5": round(0.08 * float(home_sog), 4),
            "xga_5v5": round(0.08 * float(away_sog), 4),
            "cf_5v5": home_sog,
            "ca_5v5": away_sog,
            "ff_5v5": home_sog,
            "fa_5v5": away_sog,
            "hd_xgf_5v5": round(0.03 * float(home_sog), 4),
            "hd_xga_5v5": round(0.03 * float(away_sog), 4),
            "toi_5v5_minutes": 50.0,
            "pp_goals": team_pp_goals[home_team],
            "pp_opportunities": max(1.0, team_pp_goals[home_team]),
            "pp_xgf": round(0.20 * max(1.0, team_pp_goals[home_team]), 4),
            "pp_toi_min": 4.0,
            "pk_goals_against": team_pp_goals[away_team],
            "pk_times_shorthanded": max(1.0, team_pp_goals[away_team]),
            "pk_xga": round(0.20 * max(1.0, team_pp_goals[away_team]), 4),
            "pk_toi_min": 4.0,
            "has_moneypuck_xg": False,
        },
        {
            "game_id": game_id,
            "game_date": game_date_val,
            "start_time_utc": start_utc,
            "season": season,
            "team": away_team,
            "opponent": home_team,
            "is_home": False,
            "goals_for": away_reg_goals,
            "goals_against": home_reg_goals,
            "sog_for": away_sog,
            "sog_against": home_sog,
            "xgf_5v5": round(0.08 * float(away_sog), 4),
            "xga_5v5": round(0.08 * float(home_sog), 4),
            "cf_5v5": away_sog,
            "ca_5v5": home_sog,
            "ff_5v5": away_sog,
            "fa_5v5": home_sog,
            "hd_xgf_5v5": round(0.03 * float(away_sog), 4),
            "hd_xga_5v5": round(0.03 * float(home_sog), 4),
            "toi_5v5_minutes": 50.0,
            "pp_goals": team_pp_goals[away_team],
            "pp_opportunities": max(1.0, team_pp_goals[away_team]),
            "pp_xgf": round(0.20 * max(1.0, team_pp_goals[away_team]), 4),
            "pp_toi_min": 4.0,
            "pk_goals_against": team_pp_goals[home_team],
            "pk_times_shorthanded": max(1.0, team_pp_goals[home_team]),
            "pk_xga": round(0.20 * max(1.0, team_pp_goals[home_team]), 4),
            "pk_toi_min": 4.0,
            "has_moneypuck_xg": False,
        },
    ]

    team_df = pl.DataFrame(team_rows, schema=SILVER_TEAM_GAME_LOGS_SCHEMA)
    skater_df = (
        pl.DataFrame(skater_rows, schema=SILVER_SKATER_GAME_LOGS_SCHEMA)
        if skater_rows
        else pl.DataFrame(schema=SILVER_SKATER_GAME_LOGS_SCHEMA)
    )
    goalie_df = (
        pl.DataFrame(goalie_rows, schema=SILVER_GOALIE_GAME_LOGS_SCHEMA)
        if goalie_rows
        else pl.DataFrame(schema=SILVER_GOALIE_GAME_LOGS_SCHEMA)
    )
    return team_df, skater_df, goalie_df


def parse_roster_payload(team_tricode: str, roster_payload: dict[str, Any]) -> pl.DataFrame:
    """Parse `/v1/roster/{TEAM}/current` JSON into a normalized active roster DataFrame."""
    team_code = normalize_tricode(team_tricode)
    rows: list[dict[str, Any]] = []

    groups = (
        ("forwards", "F"),
        ("defensemen", "D"),
        ("goalies", "G"),
    )
    for key_name, pos_group in groups:
        for player in roster_payload.get(key_name, []):
            player_id = int(player["id"])
            first_obj = player.get("firstName", {})
            last_obj = player.get("lastName", {})
            first_name = (
                str(first_obj.get("default", "")) if isinstance(first_obj, dict) else str(first_obj)
            )
            last_name = (
                str(last_obj.get("default", "")) if isinstance(last_obj, dict) else str(last_obj)
            )
            full_name = f"{first_name} {last_name}".strip()
            pos_code = str(player.get("positionCode", pos_group))

            rows.append(
                {
                    "team": team_code,
                    "player_id": player_id,
                    "player_name": full_name,
                    "position_code": pos_code,
                    "position_group": pos_group,
                }
            )

    schema = {
        "team": pl.Utf8(),
        "player_id": pl.Int64(),
        "player_name": pl.Utf8(),
        "position_code": pl.Utf8(),
        "position_group": pl.Utf8(),
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).unique(subset=["player_id"], keep="last")


class NHLWebAPIClient:
    """Cached HTTP client for the Official NHL Web API (`https://api-web.nhle.com/v1`)."""

    def __init__(
        self,
        cache_dir: Path,
        base_url: str = NHL_WEB_API_BASE_URL,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if "statsapi.web.nhl.com" in base_url:
            raise ValueError(
                "Deprecated NHL API 'statsapi.web.nhl.com' is shut down; "
                "use 'https://api-web.nhle.com/v1'."
            )
        self.cache_dir = cache_dir
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

        for sub in ("schedules", "boxscores", "landings", "rosters"):
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _get_json(self, endpoint: str) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            transport=self.transport,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data

    @staticmethod
    def _is_schedule_week_all_final(payload: dict[str, Any]) -> bool:
        game_week = payload.get("gameWeek", [])
        if not game_week:
            return False
        states: list[str] = []
        for day_block in game_week:
            for g in day_block.get("games", []):
                states.append(str(g.get("gameState", "FUT")).upper())
        return bool(states) and all(st in FINAL_GAME_STATES for st in states)

    def fetch_schedule_now(self, *, ttl_seconds: float = 900.0) -> dict[str, Any]:
        """Fetch `/v1/schedule/now` for active season detection and today's slate."""
        cache_path = self.cache_dir / "schedules" / "schedule_now.json.gz"
        cached = read_cached_json(cache_path, ttl_seconds=ttl_seconds)
        if isinstance(cached, dict):
            return cached

        payload = self._get_json("/schedule/now")
        write_cached_json(cache_path, payload)
        return payload

    def fetch_schedule_by_date(
        self,
        date_str: str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch 7-day schedule week starting on `YYYY-MM-DD`.

        Caches permanently only when all games in the week are `FINAL` or `OFF`.
        """
        cache_path = self.cache_dir / "schedules" / f"schedule_{date_str}.json.gz"
        if not force_refresh:
            cached = read_cached_json(cache_path)
            if isinstance(cached, dict) and self._is_schedule_week_all_final(cached):
                return cached

        payload = self._get_json(f"/schedule/{date_str}")
        if self._is_schedule_week_all_final(payload):
            write_cached_json(cache_path, payload)
        return payload

    def fetch_club_schedule_season(
        self,
        team_tricode: str,
        season: int,
        *,
        is_active_season: bool = False,
        ttl_seconds: float = 900.0,
    ) -> dict[str, Any]:
        """Fetch `/v1/club-schedule-season/{TEAM}/{SEASON}`."""
        team_code = normalize_tricode(team_tricode)
        cache_path = self.cache_dir / "schedules" / f"club_{team_code}_{season}.json.gz"
        effective_ttl = ttl_seconds if is_active_season else None
        cached = read_cached_json(cache_path, ttl_seconds=effective_ttl)
        if isinstance(cached, dict):
            return cached

        payload = self._get_json(f"/club-schedule-season/{team_code}/{season}")
        write_cached_json(cache_path, payload)
        return payload

    def fetch_game_boxscore(
        self,
        game_id: int,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch `/v1/gamecenter/{GAME_ID}/boxscore`.

        Once a game reaches `FINAL` or `OFF`, its raw JSON is cached permanently on disk
        and never re-downloaded.
        """
        cache_path = self.cache_dir / "boxscores" / f"{game_id}.json.gz"
        if not force_refresh:
            cached = read_cached_json(cache_path)
            if (
                isinstance(cached, dict)
                and str(cached.get("gameState", "")).upper() in FINAL_GAME_STATES
            ):
                return cached

        payload = self._get_json(f"/gamecenter/{game_id}/boxscore")
        if str(payload.get("gameState", "")).upper() in FINAL_GAME_STATES:
            write_cached_json(cache_path, payload)
        return payload

    def fetch_game_landing(
        self,
        game_id: int,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch `/v1/gamecenter/{GAME_ID}/landing`."""
        cache_path = self.cache_dir / "landings" / f"{game_id}.json.gz"
        if not force_refresh:
            cached = read_cached_json(cache_path)
            if (
                isinstance(cached, dict)
                and str(cached.get("gameState", "")).upper() in FINAL_GAME_STATES
            ):
                return cached

        payload = self._get_json(f"/gamecenter/{game_id}/landing")
        if str(payload.get("gameState", "")).upper() in FINAL_GAME_STATES:
            write_cached_json(cache_path, payload)
        return payload

    def fetch_team_roster(
        self,
        team_tricode: str,
        *,
        ttl_seconds: float = 3600.0,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch `/v1/roster/{TEAM}/current` with configurable TTL cache."""
        team_code = normalize_tricode(team_tricode)
        cache_path = self.cache_dir / "rosters" / f"{team_code}_current.json.gz"
        if not force_refresh:
            cached = read_cached_json(cache_path, ttl_seconds=ttl_seconds)
            if isinstance(cached, dict):
                return cached

        payload = self._get_json(f"/roster/{team_code}/current")
        write_cached_json(cache_path, payload)
        return payload

    def fetch_date_range_games(
        self,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Incrementally fetch schedule weeks covering `[start_date, end_date]`."""
        if end_date < start_date:
            raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date}).")

        frames: list[pl.DataFrame] = []
        cursor = start_date
        while cursor <= end_date:
            payload = self.fetch_schedule_by_date(cursor.isoformat())
            week_df = parse_schedule_payload(payload)
            if not week_df.is_empty():
                frames.append(week_df)
            cursor += timedelta(days=7)

        if not frames:
            return pl.DataFrame(schema=SILVER_GAMES_SCHEMA)

        combined = pl.concat(frames, how="vertical")
        filtered = combined.filter(
            (pl.col("game_date") >= start_date) & (pl.col("game_date") <= end_date)
        )
        return filtered.unique(subset=["game_id"], keep="last").sort(
            ["game_date", "start_time_utc", "game_id"]
        )
