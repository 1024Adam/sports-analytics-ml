"""Cached nflverse release client (`sports/football/ingestion/nflverse_client.py`).

Downloads nflverse release assets directly as parquet / csv (no `nfl_data_py` dependency),
caches them under `data/raw/football/nflverse/` (`*.parquet` natively compressed, schedule CSV
Gzip-compressed through `core/cache.py`), retries transient failures with exponential backoff,
and degrades gracefully (`None`) when an asset is missing, lags, or predates its availability
window.
"""

from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path
from typing import Final, Literal

import httpx
import polars as pl
from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from sports_ml.core.cache import read_cached_bytes, resolve_cache_read_path, write_gzip_bytes

NFLVERSE_RELEASE_BASE_URL: Final[str] = (
    "https://github.com/nflverse/nflverse-data/releases/download"
)
NFLDATA_GAMES_URL: Final[str] = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)
DEFAULT_USER_AGENT: Final[str] = (
    "sports-analytics-ml/0.1 (+https://github.com/1024Adam/sports-analytics-ml)"
)

NextGenStatType = Literal["passing", "rushing", "receiving"]


class NflverseAsset(BaseModel):
    """Download specification and availability window for one nflverse asset."""

    model_config = ConfigDict(frozen=True)

    name: str
    url_template: str
    per_season: bool
    fmt: Literal["parquet", "csv"]
    first_season: int | None = None


def _release(tag: str, file_template: str) -> str:
    return f"{NFLVERSE_RELEASE_BASE_URL}/{tag}/{file_template}"


def _asset(
    name: str,
    url: str,
    *,
    per_season: bool = False,
    fmt: Literal["parquet", "csv"] = "parquet",
    first_season: int | None = None,
) -> NflverseAsset:
    return NflverseAsset(
        name=name, url_template=url, per_season=per_season, fmt=fmt, first_season=first_season
    )


NFLVERSE_ASSETS: Final[dict[str, NflverseAsset]] = {
    a.name: a
    for a in (
        _asset("schedules", NFLDATA_GAMES_URL, fmt="csv", first_season=1999),
        _asset(
            "pbp",
            _release("pbp", "play_by_play_{season}.parquet"),
            per_season=True,
            first_season=1999,
        ),
        _asset(
            "player_stats",
            _release("player_stats", "player_stats_{season}.parquet"),
            per_season=True,
            first_season=1999,
        ),
        _asset(
            "rosters",
            _release("rosters", "roster_{season}.parquet"),
            per_season=True,
            first_season=1999,
        ),
        _asset(
            "rosters_weekly",
            _release("weekly_rosters", "roster_weekly_{season}.parquet"),
            per_season=True,
            first_season=2002,
        ),
        _asset(
            "injuries",
            _release("injuries", "injuries_{season}.parquet"),
            per_season=True,
            first_season=2009,
        ),
        _asset(
            "depth_charts",
            _release("depth_charts", "depth_charts_{season}.parquet"),
            per_season=True,
            first_season=2001,
        ),
        _asset(
            "snap_counts",
            _release("snap_counts", "snap_counts_{season}.parquet"),
            per_season=True,
            first_season=2012,
        ),
        _asset(
            "nextgen_passing",
            _release("nextgen_stats", "ngs_passing.parquet"),
            first_season=2016,
        ),
        _asset(
            "nextgen_rushing",
            _release("nextgen_stats", "ngs_rushing.parquet"),
            first_season=2016,
        ),
        _asset(
            "nextgen_receiving",
            _release("nextgen_stats", "ngs_receiving.parquet"),
            first_season=2016,
        ),
        _asset("draft_picks", _release("draft_picks", "draft_picks.parquet")),
        _asset("players", _release("players", "players.parquet")),
    )
}


class NflverseAssetUnavailableError(Exception):
    """Raised when an asset is not published (HTTP 404) or predates its availability window."""


def derive_active_nfl_season(today: date | None = None) -> int:
    """Return the NFL season (start year) containing `today` (seasons roll over in March)."""
    ref = today or date.today()
    return ref.year if ref.month >= 3 else ref.year - 1


def asset_available_for_season(asset: str, season: int) -> bool:
    """Return True when `season` is inside the documented availability window of `asset`."""
    first = NFLVERSE_ASSETS[asset].first_season
    return first is None or season >= first


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code != 404
    return False


class NflverseClient:
    """Cached, retrying nflverse downloader returning Polars DataFrames."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        retry_attempts: int = 3,
        retry_min_wait: float = 0.5,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.retry_attempts = retry_attempts
        self.retry_min_wait = retry_min_wait
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def asset_url(self, asset: str, season: int | None = None) -> str:
        """Return the download URL for `asset` (and `season` for per-season assets)."""
        spec = NFLVERSE_ASSETS[asset]
        if spec.per_season:
            if season is None:
                raise ValueError(f"Asset '{asset}' requires a season.")
            return spec.url_template.format(season=season)
        return spec.url_template

    def cache_path(self, asset: str, season: int | None = None) -> Path:
        """Return the preferred raw-cache path for `asset` (`.parquet` or Gzip `.csv.gz`)."""
        spec = NFLVERSE_ASSETS[asset]
        stem = f"{asset}_{season}" if spec.per_season else asset
        return self.cache_dir / (f"{stem}.parquet" if spec.fmt == "parquet" else f"{stem}.csv.gz")

    def _download_bytes(self, url: str) -> bytes:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=self.retry_min_wait, min=self.retry_min_wait, max=8.0),
            retry=retry_if_exception(_is_retryable),
        )
        def _get() -> bytes:
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=True,
                transport=self.transport,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.content

        try:
            return _get()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise NflverseAssetUnavailableError(f"Not published (404): {url}") from exc
            raise

    @staticmethod
    def _parse(raw: bytes, fmt: str) -> pl.DataFrame:
        if fmt == "parquet":
            return pl.read_parquet(io.BytesIO(raw))
        return pl.read_csv(io.BytesIO(raw), infer_schema_length=10000, null_values=["NA", ""])

    @staticmethod
    def _existing_path(path: Path, fmt: str) -> Path | None:
        if fmt == "csv":
            return resolve_cache_read_path(path)
        return path if path.exists() else None

    @staticmethod
    def _read_cache(path: Path, fmt: str) -> bytes:
        return read_cached_bytes(path) if fmt == "csv" else path.read_bytes()

    @staticmethod
    def _write_cache(path: Path, raw: bytes, fmt: str) -> None:
        if fmt == "csv":
            write_gzip_bytes(path, raw)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)

    def fetch_asset(
        self,
        asset: str,
        season: int | None = None,
        *,
        is_active_season: bool = False,
        ttl_seconds: float | None = None,
    ) -> pl.DataFrame:
        """Fetch `asset` from cache or network.

        Completed seasons are cached permanently; the active season (and non-season assets)
        refresh after `ttl_seconds` (default 6 h in-season, 24 h otherwise). If a refresh
        fails but a stale cache exists, the stale copy is returned.
        """
        spec = NFLVERSE_ASSETS[asset]
        if season is not None and not asset_available_for_season(asset, season):
            raise NflverseAssetUnavailableError(
                f"Asset '{asset}' is unavailable before {spec.first_season} (requested {season})."
            )
        path = self.cache_path(asset, season)
        existing = self._existing_path(path, spec.fmt)
        refreshable = is_active_season or not spec.per_season
        ttl = ttl_seconds if ttl_seconds is not None else (21600.0 if is_active_season else 86400.0)
        if existing is not None:
            fresh = (not refreshable) or (time.time() - existing.stat().st_mtime) <= ttl
            if fresh:
                return self._parse(self._read_cache(path, spec.fmt), spec.fmt)
        try:
            raw = self._download_bytes(self.asset_url(asset, season))
        except NflverseAssetUnavailableError:
            raise
        except (httpx.TransportError, httpx.HTTPStatusError):
            if existing is None:
                raise
            return self._parse(self._read_cache(path, spec.fmt), spec.fmt)
        frame = self._parse(raw, spec.fmt)
        self._write_cache(path, raw, spec.fmt)
        return frame

    def try_fetch_asset(
        self,
        asset: str,
        season: int | None = None,
        *,
        is_active_season: bool = False,
        ttl_seconds: float | None = None,
    ) -> pl.DataFrame | None:
        """Like `fetch_asset` but returns None when the asset is unpublished or out of window."""
        try:
            return self.fetch_asset(
                asset, season, is_active_season=is_active_season, ttl_seconds=ttl_seconds
            )
        except NflverseAssetUnavailableError:
            return None

    def fetch_schedule(self, *, ttl_seconds: float | None = None) -> pl.DataFrame:
        """Fetch the full nflverse `games.csv` schedule (all seasons)."""
        return self.fetch_asset("schedules", ttl_seconds=ttl_seconds)

    def fetch_nextgen_stats(
        self, stat_type: NextGenStatType, season: int | None = None
    ) -> pl.DataFrame | None:
        """Fetch Next Gen Stats (2016+), optionally filtered to one season."""
        asset = f"nextgen_{stat_type}"
        if season is not None and not asset_available_for_season(asset, season):
            return None
        frame = self.try_fetch_asset(asset)
        if frame is None or season is None:
            return frame
        return frame.filter(pl.col("season") == season)
