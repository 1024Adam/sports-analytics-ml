"""Upstream data source connectivity & schema contract probes (`baseball/ingestion/health.py`).

Verifies reachability and Polars schema contracts across MLB StatsAPI, Baseball
Savant Statcast, and NPB schedules/rosters.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Literal

import httpx

from sports_ml.sports.baseball.constants import SILVER_BASEBALL_GAMES_SCHEMA
from sports_ml.sports.baseball.ingestion.mlb_api import (
    MLB_STATSAPI_BASE_URL,
    MLBStatsAPIClient,
    derive_active_baseball_season,
    parse_mlb_schedule_payload,
)
from sports_ml.sports.baseball.ingestion.npb_client import (
    NPB_BIS_BASE_URL,
    NPBClient,
    parse_npb_schedule_payload,
)
from sports_ml.sports.baseball.ingestion.statcast import (
    STATCAST_BASE_URL,
    StatcastClient,
)
from sports_ml.sports.hockey.ingestion.health import DataSourceHealthResult


def check_mlb_statsapi_source(
    cache_dir: Path,
    *,
    target_date: date | None = None,
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify Official MLB StatsAPI `/api/v1/schedule` reachability and schema contract."""
    probe_date = target_date or date(derive_active_baseball_season(), 6, 15)
    endpoint = f"{MLB_STATSAPI_BASE_URL}/schedule?sportId=1&startDate={probe_date.isoformat()}"
    t0 = time.perf_counter()
    try:
        client = MLBStatsAPIClient(cache_dir=cache_dir, transport=transport)
        payload = client.fetch_schedule(probe_date, probe_date, ttl_seconds=0.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        games_df = parse_mlb_schedule_payload(payload, season=probe_date.year)
        if dict(games_df.schema) != SILVER_BASEBALL_GAMES_SCHEMA:
            return DataSourceHealthResult(
                source="MLB StatsAPI (Schedule & Starters)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Parsed MLB schedule schema mismatch against SILVER_BASEBALL_GAMES_SCHEMA.",
            )
        return DataSourceHealthResult(
            source="MLB StatsAPI (Schedule & Starters)",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK ({games_df.height} MLB games parsed)",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="MLB StatsAPI (Schedule & Starters)",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_statcast_source(
    cache_dir: Path,
    *,
    season: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify Baseball Savant / Statcast CSV reachability and Polars parsing."""
    active_season = season or derive_active_baseball_season()
    endpoint = f"{STATCAST_BASE_URL}?type=teams&year={active_season}"
    t0 = time.perf_counter()
    try:
        client = StatcastClient(cache_dir=cache_dir, transport=transport)
        df = client.fetch_season_team_metrics(active_season, is_active_season=True, ttl_seconds=0.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        required = {"game_id", "team", "team_xwoba"}
        if df.is_empty() or not required.issubset(set(df.columns)):
            return DataSourceHealthResult(
                source="Baseball Savant Statcast (xwOBA/Stuff)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details=f"Empty or incomplete Statcast DataFrame (rows={df.height}).",
            )
        return DataSourceHealthResult(
            source="Baseball Savant Statcast (xwOBA/Stuff)",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK ({df.height} team-game Statcast records parsed)",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="Baseball Savant Statcast (xwOBA/Stuff)",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_npb_source(
    cache_dir: Path,
    *,
    season: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify NPB schedule & 12-inning boxscore feed reachability and schema contract."""
    active_season = season or derive_active_baseball_season()
    endpoint = f"{NPB_BIS_BASE_URL}/{active_season}/games/schedule.json"
    t0 = time.perf_counter()
    try:
        client = NPBClient(cache_dir=cache_dir, transport=transport)
        payload = client.fetch_season_schedule(active_season, ttl_seconds=0.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        games_df = parse_npb_schedule_payload(payload, season=active_season)
        if dict(games_df.schema) != SILVER_BASEBALL_GAMES_SCHEMA:
            return DataSourceHealthResult(
                source="NPB BIS (Schedule & Starters)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Parsed NPB schedule schema mismatch against SILVER_BASEBALL_GAMES_SCHEMA.",
            )
        return DataSourceHealthResult(
            source="NPB BIS (Schedule & Starters)",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK ({games_df.height} NPB games parsed)",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="NPB BIS (Schedule & Starters)",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_baseball_data_sources(
    raw_dir: Path,
    *,
    league: Literal["mlb", "npb", "all"] = "all",
    season: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> list[DataSourceHealthResult]:
    """Run upstream data source health probes for MLB, NPB, or both."""
    lg = league.strip().lower()
    results: list[DataSourceHealthResult] = []
    if lg in {"mlb", "all"}:
        results.append(
            check_mlb_statsapi_source(raw_dir / "baseball" / "mlb_api", transport=transport)
        )
        results.append(
            check_statcast_source(
                raw_dir / "baseball" / "statcast", season=season, transport=transport
            )
        )
    if lg in {"npb", "all"}:
        results.append(
            check_npb_source(raw_dir / "baseball" / "npb", season=season, transport=transport)
        )
    return results
