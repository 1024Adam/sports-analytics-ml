"""Upstream data source connectivity & schema contract probes (`soccer/ingestion/health.py`).

Verifies live reachability, HTTP response validity, and parser/schema contracts
across the three external EPL soccer data feeds:
1. Official FPL / Premier League API (`fantasy.premierleague.com/api`: bootstrap & fixtures)
2. Understat EPL advanced metrics (`understat.com`: team match xG/npxG/PPDA)
3. Football-Data.co.uk historical/active season CSV (`football-data.co.uk`: `E0.csv`)
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from sports_ml.sports.hockey.ingestion.health import DataSourceHealthResult
from sports_ml.sports.soccer.constants import SILVER_MATCHES_SCHEMA
from sports_ml.sports.soccer.ingestion.football_data import (
    FOOTBALL_DATA_BASE_URL,
    FootballDataClient,
    season_to_football_data_code,
)
from sports_ml.sports.soccer.ingestion.fpl_api import (
    FPL_API_BASE_URL,
    FPLAPIClient,
    derive_active_epl_season,
    parse_bootstrap_squad_df,
    parse_bootstrap_teams_map,
    parse_fixtures_payload,
)
from sports_ml.sports.soccer.ingestion.understat import (
    UNDERSTAT_BASE_URL,
    UnderstatClient,
)


def check_fpl_bootstrap_and_fixtures_source(
    cache_dir: Path,
    *,
    season: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify Official FPL API `/api/bootstrap-static/` and `/api/fixtures/` contracts."""
    endpoint = f"{FPL_API_BASE_URL}/bootstrap-static/ & /fixtures/"
    active_season = season or derive_active_epl_season()
    t0 = time.perf_counter()
    try:
        client = FPLAPIClient(cache_dir=cache_dir, transport=transport)
        bootstrap = client.fetch_bootstrap_static(ttl_seconds=0.0)
        fixtures = client.fetch_fixtures(ttl_seconds=0.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        teams_map = parse_bootstrap_teams_map(bootstrap)
        squad_df = parse_bootstrap_squad_df(bootstrap)
        matches_df = parse_fixtures_payload(fixtures, teams_map, season=active_season)

        if len(teams_map) == 0 or squad_df.is_empty():
            return DataSourceHealthResult(
                source="FPL API (Squads & Fixtures)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Empty teams or squad parsed from /api/bootstrap-static/.",
            )
        if dict(matches_df.schema) != SILVER_MATCHES_SCHEMA:
            return DataSourceHealthResult(
                source="FPL API (Squads & Fixtures)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Parsed fixtures schema mismatch against SILVER_MATCHES_SCHEMA.",
            )
        return DataSourceHealthResult(
            source="FPL API (Squads & Fixtures)",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=(
                f"OK ({len(teams_map)} clubs, {squad_df.height} players, "
                f"{matches_df.height} fixtures)"
            ),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="FPL API (Squads & Fixtures)",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_understat_epl_source(
    cache_dir: Path,
    *,
    season: int = 20242025,
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify Understat EPL team match xG/npxG/PPDA feed reachability and parsing."""
    start_yr = int(str(season)[:4])
    endpoint = f"{UNDERSTAT_BASE_URL}/getLeagueData/EPL/{start_yr}"
    t0 = time.perf_counter()
    try:
        client = UnderstatClient(cache_dir=cache_dir, transport=transport)
        df = client.fetch_season_team_matches(season, ttl_seconds=0.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        required_cols = {"team", "xgf", "xga", "npxgf", "npxga", "ppda"}
        if df.is_empty() or not required_cols.issubset(set(df.columns)):
            return DataSourceHealthResult(
                source="Understat (EPL xG/PPDA)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details=f"Empty or incomplete Understat DataFrame (rows={df.height}).",
            )
        return DataSourceHealthResult(
            source="Understat (EPL xG/PPDA)",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK ({df.height} team-match xG/npxG/PPDA records parsed)",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="Understat (EPL xG/PPDA)",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_football_data_epl_source(
    cache_dir: Path,
    *,
    season: int = 20242025,
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify Football-Data.co.uk `E0.csv` reachability and Polars parsing."""
    code = season_to_football_data_code(season)
    endpoint = f"{FOOTBALL_DATA_BASE_URL}/{code}/E0.csv"
    t0 = time.perf_counter()
    try:
        client = FootballDataClient(cache_dir=cache_dir, transport=transport)
        matches_df, box_df = client.fetch_season_csv(season, is_active_season=True, ttl_seconds=0.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if matches_df.is_empty() or box_df.is_empty():
            return DataSourceHealthResult(
                source="Football-Data.co.uk (E0.csv)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details=f"Empty matches ({matches_df.height}) or boxscores ({box_df.height}).",
            )
        return DataSourceHealthResult(
            source="Football-Data.co.uk (E0.csv)",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK ({matches_df.height} matches, {box_df.height} team boxscore rows)",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="Football-Data.co.uk (E0.csv)",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_soccer_data_sources(
    raw_dir: Path,
    *,
    season: int = 20242025,
    transport: httpx.BaseTransport | None = None,
) -> list[DataSourceHealthResult]:
    """Run all three EPL soccer upstream data source health probes."""
    return [
        check_fpl_bootstrap_and_fixtures_source(
            raw_dir / "fpl_api", season=season, transport=transport
        ),
        check_understat_epl_source(raw_dir / "understat", season=season, transport=transport),
        check_football_data_epl_source(
            raw_dir / "football_data", season=season, transport=transport
        ),
    ]
