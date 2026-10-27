"""Understat EPL match and player shot-quality client & parsers (`understat.py`).

Parses team match logs (`npxgf`, `npxga`, `xgf`, `xga`, `ppda`, `opp_ppda`,
`deep_completions`, `opp_deep_completions`) and player match logs (`npxg`, `xg`,
`xa`, `xg_chain`, `xg_buildup`, `shots`, `key_passes`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sports_ml.core.cache import read_cached_json, write_cached_json
from sports_ml.sports.soccer.constants import normalize_soccer_club

UNDERSTAT_BASE_URL: Final[str] = "https://understat.com"
DEFAULT_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)


def _to_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    return float(val)


def _compute_ppda(att_Allowed: Any, def_actions: Any) -> float:
    """Compute PPDA = opponent passes allowed / defensive actions (`tackles + interceptions`)."""
    att = _to_float(att_Allowed, 0.0)
    def_act = _to_float(def_actions, 0.0)
    if def_act <= 0.0:
        return 11.5
    return att / def_act


def parse_understat_team_matches_payload(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    season: int,
) -> pl.DataFrame:
    """Parse Understat league/team match records into normalized team-date advanced metrics.

    Accepts either:
    1. A list of match records containing `h`, `a`, `xG`, `npxG`, `ppda`, `deep`, `datetime`, or
    2. A mapping of `{team_id: {"title": ..., "history": [...]}}` from Understat's `teamsData`.
    """
    rows: list[dict[str, Any]] = []

    if isinstance(payload, Mapping):
        for team_entry in payload.values():
            if not isinstance(team_entry, Mapping):
                continue
            team_code = normalize_soccer_club(str(team_entry.get("title") or ""))
            for h in team_entry.get("history", []):
                dt_str = str(h.get("date") or "")
                if not dt_str:
                    continue
                m_date = date.fromisoformat(dt_str[:10])
                side = str(h.get("h_a") or "h").lower()
                is_home = side == "h"

                ppda_dict = h.get("ppda") or {}
                opp_ppda_dict = h.get("ppda_allowed") or {}
                ppda_val = (
                    _compute_ppda(ppda_dict.get("att"), ppda_dict.get("def"))
                    if isinstance(ppda_dict, Mapping)
                    else float(ppda_dict or 11.5)
                )
                opp_ppda_val = (
                    _compute_ppda(opp_ppda_dict.get("att"), opp_ppda_dict.get("def"))
                    if isinstance(opp_ppda_dict, Mapping)
                    else float(opp_ppda_dict or 11.5)
                )

                rows.append(
                    {
                        "season": int(season),
                        "match_date": m_date,
                        "team": team_code,
                        "is_home": is_home,
                        "xgf": float(h.get("xG") or 0.0),
                        "xga": float(h.get("xGA") or 0.0),
                        "npxgf": float(h.get("npxG") or h.get("xG") or 0.0),
                        "npxga": float(h.get("npxGA") or h.get("xGA") or 0.0),
                        "ppda": ppda_val,
                        "opp_ppda": opp_ppda_val,
                        "deep_completions": int(h.get("deep") or 0),
                        "opp_deep_completions": int(h.get("deep_allowed") or 0),
                    }
                )
    else:
        for item in payload:
            dt_str = str(item.get("datetime") or item.get("date") or "")
            if not dt_str:
                continue
            m_date = date.fromisoformat(dt_str[:10])
            h_info = item.get("h") or {}
            a_info = item.get("a") or {}
            home_team = normalize_soccer_club(
                str(
                    (h_info.get("title") or h_info.get("short_title"))
                    if isinstance(h_info, Mapping)
                    else item.get("home_team", "")
                )
            )
            away_team = normalize_soccer_club(
                str(
                    (a_info.get("title") or a_info.get("short_title"))
                    if isinstance(a_info, Mapping)
                    else item.get("away_team", "")
                )
            )
            xg_info = item.get("xG") or {}
            npxg_info = item.get("npxG") or {}
            h_xg = _to_float(
                xg_info.get("h") if isinstance(xg_info, Mapping) else item.get("home_xg")
            )
            a_xg = _to_float(
                xg_info.get("a") if isinstance(xg_info, Mapping) else item.get("away_xg")
            )
            h_npxg = _to_float(
                npxg_info.get("h")
                if isinstance(npxg_info, Mapping) and npxg_info.get("h") is not None
                else item.get("home_npxg"),
                default=h_xg,
            )
            a_npxg = _to_float(
                npxg_info.get("a")
                if isinstance(npxg_info, Mapping) and npxg_info.get("a") is not None
                else item.get("away_npxg"),
                default=a_xg,
            )
            h_ppda = float(item.get("home_ppda") or 11.0)
            a_ppda = float(item.get("away_ppda") or 11.0)
            h_deep = int(item.get("home_deep") or 6)
            a_deep = int(item.get("away_deep") or 6)

            rows.append(
                {
                    "season": int(season),
                    "match_date": m_date,
                    "team": home_team,
                    "is_home": True,
                    "xgf": h_xg,
                    "xga": a_xg,
                    "npxgf": h_npxg,
                    "npxga": a_npxg,
                    "ppda": h_ppda,
                    "opp_ppda": a_ppda,
                    "deep_completions": h_deep,
                    "opp_deep_completions": a_deep,
                }
            )
            rows.append(
                {
                    "season": int(season),
                    "match_date": m_date,
                    "team": away_team,
                    "is_home": False,
                    "xgf": a_xg,
                    "xga": h_xg,
                    "npxgf": a_npxg,
                    "npxga": h_npxg,
                    "ppda": a_ppda,
                    "opp_ppda": h_ppda,
                    "deep_completions": a_deep,
                    "opp_deep_completions": h_deep,
                }
            )

    schema = {
        "season": pl.Int32(),
        "match_date": pl.Date(),
        "team": pl.Utf8(),
        "is_home": pl.Boolean(),
        "xgf": pl.Float64(),
        "xga": pl.Float64(),
        "npxgf": pl.Float64(),
        "npxga": pl.Float64(),
        "ppda": pl.Float64(),
        "opp_ppda": pl.Float64(),
        "deep_completions": pl.Int16(),
        "opp_deep_completions": pl.Int16(),
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows, schema=schema)
        .unique(subset=["season", "match_date", "team"], keep="last")
        .sort(["match_date", "team"])
    )


def parse_understat_player_matches_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    season: int,
) -> pl.DataFrame:
    """Parse Understat player match records into per-match shot & possession-chain metrics."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        dt_str = str(rec.get("date") or rec.get("match_date") or "")
        if not dt_str:
            continue
        m_date = date.fromisoformat(dt_str[:10])
        team = normalize_soccer_club(str(rec.get("team") or rec.get("h_team") or ""))
        rows.append(
            {
                "season": int(season),
                "match_date": m_date,
                "player_id": int(rec["player_id"]) if "player_id" in rec else None,
                "player_name": str(rec.get("player_name") or rec.get("player") or "").strip(),
                "team": team,
                "shots": int(rec.get("shots") or 0),
                "sot": int(rec.get("sot") or rec.get("shots_on_target") or 0),
                "key_passes": int(rec.get("key_passes") or 0),
                "xg": _to_float(rec.get("xG") if "xG" in rec else rec.get("xg")),
                "npxg": _to_float(
                    rec.get("npxG")
                    if "npxG" in rec
                    else (rec.get("npxg") or rec.get("xG") or rec.get("xg"))
                ),
                "xa": _to_float(rec.get("xA") if "xA" in rec else rec.get("xa")),
                "xg_chain": _to_float(
                    rec.get("xGChain") if "xGChain" in rec else rec.get("xg_chain")
                ),
                "xg_buildup": _to_float(
                    rec.get("xGBuildup") if "xGBuildup" in rec else rec.get("xg_buildup")
                ),
            }
        )

    schema = {
        "season": pl.Int32(),
        "match_date": pl.Date(),
        "player_id": pl.Int64(),
        "player_name": pl.Utf8(),
        "team": pl.Utf8(),
        "shots": pl.Int16(),
        "sot": pl.Int16(),
        "key_passes": pl.Int16(),
        "xg": pl.Float64(),
        "npxg": pl.Float64(),
        "xa": pl.Float64(),
        "xg_chain": pl.Float64(),
        "xg_buildup": pl.Float64(),
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(["match_date", "team", "player_name"])


class UnderstatClient:
    """Cached client for Understat EPL team and player shot-quality datasets."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        base_url: str = UNDERSTAT_BASE_URL,
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
    def _fetch_league_json(self, season_start_year: int) -> dict[str, Any]:
        url = f"{self.base_url}/getLeagueData/EPL/{season_start_year}"
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "X-Requested-With": "XMLHttpRequest",
        }
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            transport=self.transport,
            headers=headers,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return dict(resp.json())

    def fetch_season_team_matches(
        self,
        season: int,
        *,
        ttl_seconds: float | None = 21600.0,
    ) -> pl.DataFrame:
        """Fetch or load cached Understat team match logs for `season` (e.g. `20242025`)."""
        start_yr = int(str(season)[:4])
        cache_path = self.cache_dir / f"epl_teams_{start_yr}.json.gz"
        payload = read_cached_json(cache_path, ttl_seconds=ttl_seconds)
        if payload is None:
            raw = self._fetch_league_json(start_yr)
            payload = raw.get("teams", raw)
            write_cached_json(cache_path, payload)
        return parse_understat_team_matches_payload(payload, season=season)

    def fetch_season_player_matches(
        self,
        season: int,
        *,
        ttl_seconds: float | None = 21600.0,
    ) -> pl.DataFrame:
        """Fetch or load cached Understat player match logs (`epl_players_{yr}.json.gz`)."""
        start_yr = int(str(season)[:4])
        cache_path = self.cache_dir / f"epl_players_{start_yr}.json.gz"
        payload = read_cached_json(cache_path, ttl_seconds=ttl_seconds)
        if payload is None:
            raw = self._fetch_league_json(start_yr)
            payload = raw.get("players", raw if isinstance(raw, list) else [])
            write_cached_json(cache_path, payload)
        records = payload if isinstance(payload, list) else []
        return parse_understat_player_matches_payload(records, season=season)
