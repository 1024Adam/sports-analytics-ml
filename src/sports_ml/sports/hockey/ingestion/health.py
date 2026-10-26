"""Upstream data source connectivity & schema contract probes (`ingestion/health.py`).

Verifies live reachability, HTTP response validity, and parser/schema contracts
across the three external hockey data feeds:
1. Official NHL Web API (`api-web.nhle.com/v1`: schedule & active team roster)
2. MoneyPuck situational xG feed (`moneypuck.com`: single-team game-by-game CSV)
3. DailyFaceoff morning-skate starting goalies (`dailyfaceoff.com`: `__NEXT_DATA__` JSON)
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from sports_ml.sports.hockey.constants import SILVER_GAMES_SCHEMA
from sports_ml.sports.hockey.ingestion.lineups import DAILYFACEOFF_STARTING_GOALIES_URL
from sports_ml.sports.hockey.ingestion.moneypuck import (
    DEFAULT_USER_AGENT,
    MONEYPUCK_BASE_URL,
    MoneyPuckClient,
)
from sports_ml.sports.hockey.ingestion.nhl_api import (
    NHL_WEB_API_BASE_URL,
    NHLWebAPIClient,
    parse_roster_payload,
    parse_schedule_payload,
)


class DataSourceHealthResult(BaseModel):
    """Health check result for a single upstream data source."""

    model_config = ConfigDict(frozen=True)

    source: str
    endpoint: str
    healthy: bool
    latency_ms: float
    details: str


def check_nhl_schedule_source(
    cache_dir: Path,
    *,
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify Official NHL Web API `/v1/schedule/now` reachability and schema parsing."""
    endpoint = f"{NHL_WEB_API_BASE_URL}/schedule/now"
    t0 = time.perf_counter()
    try:
        client = NHLWebAPIClient(cache_dir=cache_dir, transport=transport)
        payload = client.fetch_schedule_now(ttl_seconds=0.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if "gameWeek" not in payload or not isinstance(payload["gameWeek"], list):
            return DataSourceHealthResult(
                source="NHL API (Schedule)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Missing or invalid 'gameWeek' list in response JSON.",
            )
        games_df = parse_schedule_payload(payload)
        if dict(games_df.schema) != SILVER_GAMES_SCHEMA:
            return DataSourceHealthResult(
                source="NHL API (Schedule)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Parsed schedule DataFrame schema mismatch against SILVER_GAMES_SCHEMA.",
            )
        return DataSourceHealthResult(
            source="NHL API (Schedule)",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK (parsed {games_df.height} regular/playoff games in current week)",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="NHL API (Schedule)",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_nhl_roster_source(
    cache_dir: Path,
    *,
    team_code: str = "BOS",
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify Official NHL Web API `/v1/roster/{TEAM}/current` reachability and parsing."""
    endpoint = f"{NHL_WEB_API_BASE_URL}/roster/{team_code}/current"
    t0 = time.perf_counter()
    try:
        client = NHLWebAPIClient(cache_dir=cache_dir, transport=transport)
        payload = client.fetch_team_roster(team_code, force_refresh=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        roster_df = parse_roster_payload(team_code, payload)
        pos_groups = (
            set(roster_df["position_group"].to_list()) if not roster_df.is_empty() else set()
        )
        if roster_df.is_empty() or not {"F", "D", "G"}.issubset(pos_groups):
            return DataSourceHealthResult(
                source="NHL API (Roster)",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details=(
                    f"Incomplete roster parsed "
                    f"(rows={roster_df.height}, groups={sorted(pos_groups)})."
                ),
            )
        return DataSourceHealthResult(
            source="NHL API (Roster)",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK ({roster_df.height} active players across F/D/G for {team_code})",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="NHL API (Roster)",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_moneypuck_source(
    cache_dir: Path,
    *,
    team_code: str = "BOS",
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify MoneyPuck game-by-game CSV reachability and situational xG column parsing."""
    endpoint = f"{MONEYPUCK_BASE_URL}/regular/teams/{team_code}.csv"
    t0 = time.perf_counter()
    try:
        client = MoneyPuckClient(cache_dir=cache_dir, transport=transport)
        df = client.load_single_team_game_stats(team_code)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        required_cols = {"game_id", "team", "xgf_5v5", "xga_5v5", "cf_5v5", "pp_xgf", "pk_xga"}
        if df.is_empty() or not required_cols.issubset(set(df.columns)):
            return DataSourceHealthResult(
                source="MoneyPuck CSV",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details=f"Parsed team CSV is empty or missing columns (rows={df.height}).",
            )
        return DataSourceHealthResult(
            source="MoneyPuck CSV",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK (parsed {df.height} historical/current games for {team_code})",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="MoneyPuck CSV",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_dailyfaceoff_source(
    target_date: date | None = None,
    *,
    timeout_seconds: float = 15.0,
    transport: httpx.BaseTransport | None = None,
) -> DataSourceHealthResult:
    """Verify DailyFaceoff starting-goalies page reachability and `__NEXT_DATA__` JSON structure."""
    dt = target_date or datetime.now(tz=UTC).date()
    endpoint = f"{DAILYFACEOFF_STARTING_GOALIES_URL}/{dt.isoformat()}"
    t0 = time.perf_counter()
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            resp = client.get(endpoint)
            resp.raise_for_status()
            html = resp.text
        latency_ms = (time.perf_counter() - t0) * 1000.0

        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html,
        )
        if not match:
            return DataSourceHealthResult(
                source="DailyFaceoff Goalies",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Missing <script id='__NEXT_DATA__'> JSON block in HTML response.",
            )
        payload = json.loads(match.group(1))
        page_props = payload.get("props", {}).get("pageProps", {})
        if not isinstance(page_props, dict) or "data" not in page_props:
            return DataSourceHealthResult(
                source="DailyFaceoff Goalies",
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Missing 'props.pageProps.data' array in __NEXT_DATA__ payload.",
            )
        matchups = page_props.get("data") or []
        return DataSourceHealthResult(
            source="DailyFaceoff Goalies",
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=(
                f"OK (__NEXT_DATA__ schema valid; "
                f"{len(matchups)} matchups listed for {dt.isoformat()})"
            ),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source="DailyFaceoff Goalies",
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_hockey_data_sources(
    cache_dir: Path,
    *,
    target_date: date | None = None,
    team_code: str = "BOS",
    nhl_transport: httpx.BaseTransport | None = None,
    moneypuck_transport: httpx.BaseTransport | None = None,
    dailyfaceoff_transport: httpx.BaseTransport | None = None,
) -> list[DataSourceHealthResult]:
    """Run all connectivity and schema contract probes for the hockey plugin."""
    nhl_dir = cache_dir / "nhl_api"
    mp_dir = cache_dir / "moneypuck"
    return [
        check_nhl_schedule_source(nhl_dir, transport=nhl_transport),
        check_nhl_roster_source(nhl_dir, team_code=team_code, transport=nhl_transport),
        check_moneypuck_source(mp_dir, team_code=team_code, transport=moneypuck_transport),
        check_dailyfaceoff_source(target_date=target_date, transport=dailyfaceoff_transport),
    ]
