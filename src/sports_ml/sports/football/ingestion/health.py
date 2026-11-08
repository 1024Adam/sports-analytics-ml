"""Upstream data source connectivity & schema contract probes (`football/ingestion/health.py`).

Verifies that every nflverse release asset is reachable, parses, and still exposes the
columns the reconciler depends on (`sports-ml check-sources --sport football --league nfl`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Final, Literal

import httpx

from sports_ml.sports.football.ingestion.nflverse_client import (
    NFLVERSE_ASSETS,
    NflverseClient,
    derive_active_nfl_season,
)
from sports_ml.sports.hockey.ingestion.health import DataSourceHealthResult

# Each entry is a column name or a tuple of acceptable alternatives (schema drift tolerance).
NFLVERSE_REQUIRED_COLUMNS: Final[dict[str, tuple[str | tuple[str, ...], ...]]] = {
    "schedules": (
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "gametime",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "location",
        "overtime",
        "stadium_id",
        "home_qb_id",
        "away_qb_id",
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
    ),
    "pbp": (
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "play_type",
        "epa",
        "success",
        "yards_gained",
        "pass",
        "rush",
        "qb_dropback",
        "id",
        "sack",
        "interception",
        "fumble_lost",
        "penalty",
        "penalty_team",
        "penalty_yards",
        "drive",
        "drive_time_of_possession",
        "cpoe",
        "air_yards",
        "qb_scramble",
        "pass_attempt",
        "wp",
        "down",
        "qb_kneel",
        "qb_spike",
        "yardline_100",
        "fixed_drive_result",
        "fumble",
    ),
    "player_stats": (
        "player_id",
        "season",
        "week",
        "position",
        ("recent_team", "team"),
        "attempts",
        "carries",
        "targets",
    ),
    "rosters": ("season", "team", "gsis_id", "position"),
    "rosters_weekly": ("season", "week", "team", "gsis_id", "status"),
    "injuries": (
        "season",
        "week",
        "team",
        "gsis_id",
        "report_status",
        "practice_status",
        "date_modified",
    ),
    "depth_charts": ("season", "week", ("club_code", "team"), "gsis_id", "depth_team", "position"),
    "snap_counts": (
        "game_id",
        "season",
        "week",
        "team",
        "pfr_player_id",
        "offense_snaps",
        "offense_pct",
    ),
    "nextgen_passing": ("season", "week", "player_gsis_id", "team_abbr"),
    "nextgen_rushing": ("season", "week", "player_gsis_id", "team_abbr"),
    "nextgen_receiving": ("season", "week", "player_gsis_id", "team_abbr"),
    "draft_picks": ("season", "round", "pick", "gsis_id", "position"),
    "players": ("gsis_id", "pfr_id", "display_name", "position"),
}

_SOURCE_LABELS: Final[dict[str, str]] = {
    "schedules": "nflverse Schedules (games.csv)",
    "pbp": "nflverse Play-by-Play (EPA/WP)",
    "player_stats": "nflverse Weekly Player Stats",
    "rosters": "nflverse Rosters",
    "rosters_weekly": "nflverse Weekly Rosters (inactives)",
    "injuries": "nflverse Injury Reports",
    "depth_charts": "nflverse Depth Charts",
    "snap_counts": "nflverse Snap Counts",
    "nextgen_passing": "nflverse Next Gen Stats (Passing)",
    "nextgen_rushing": "nflverse Next Gen Stats (Rushing)",
    "nextgen_receiving": "nflverse Next Gen Stats (Receiving)",
    "draft_picks": "nflverse Draft Picks",
    "players": "nflverse Players (ID crosswalk)",
}


def missing_required_columns(asset: str, columns: list[str]) -> list[str]:
    """Return the required columns (or alternative groups) absent from `columns`."""
    present = set(columns)
    missing: list[str] = []
    for requirement in NFLVERSE_REQUIRED_COLUMNS[asset]:
        options = (requirement,) if isinstance(requirement, str) else requirement
        if not present.intersection(options):
            missing.append("|".join(options))
    return missing


def check_nflverse_asset(
    client: NflverseClient,
    asset: str,
    *,
    season: int,
) -> DataSourceHealthResult:
    """Probe one nflverse asset for reachability, non-emptiness, and its schema contract."""
    spec = NFLVERSE_ASSETS[asset]
    endpoint = client.asset_url(asset, season if spec.per_season else None)
    t0 = time.perf_counter()
    source = _SOURCE_LABELS[asset]
    try:
        frame = client.fetch_asset(asset, season if spec.per_season else None, ttl_seconds=86400.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if frame.is_empty():
            return DataSourceHealthResult(
                source=source,
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details="Empty frame returned.",
            )
        missing = missing_required_columns(asset, frame.columns)
        if missing:
            return DataSourceHealthResult(
                source=source,
                endpoint=endpoint,
                healthy=False,
                latency_ms=latency_ms,
                details=f"Schema contract violated; missing columns: {missing}.",
            )
        return DataSourceHealthResult(
            source=source,
            endpoint=endpoint,
            healthy=True,
            latency_ms=latency_ms,
            details=f"OK ({frame.height} rows, {frame.width} columns)",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DataSourceHealthResult(
            source=source,
            endpoint=endpoint,
            healthy=False,
            latency_ms=latency_ms,
            details=f"{type(exc).__name__}: {exc}",
        )


def check_football_data_sources(
    raw_dir: Path,
    *,
    league: Literal["nfl"] = "nfl",
    season: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> list[DataSourceHealthResult]:
    """Run reachability + schema-contract probes on every nflverse release asset."""
    if league.strip().lower() != "nfl":
        raise ValueError(f"Unsupported football league '{league}'. Only 'nfl' is implemented.")
    probe_season = season if season is not None else derive_active_nfl_season() - 1
    client = NflverseClient(raw_dir / "football" / "nflverse", transport=transport)
    return [check_nflverse_asset(client, asset, season=probe_season) for asset in NFLVERSE_ASSETS]
