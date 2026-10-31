"""Baseball Savant / Statcast downloader and Polars parser (`baseball/ingestion/statcast.py`).

Caches `.csv.gz` archives for MLB team, pitcher, and batter contact-quality and
stuff metrics (`xwOBA`, `Barrel%`, `HardHit%`, `SwStr%`, `avg_fastball_velo`).
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Final, Literal

import httpx
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from sports_ml.core.cache import (
    read_cached_bytes,
    resolve_cache_read_path,
    write_gzip_bytes,
)
from sports_ml.sports.baseball.constants import normalize_baseball_team

STATCAST_BASE_URL: Final[str] = "https://baseballsavant.mlb.com/statcast_search/csv"

StatcastEntityKind = Literal["teams", "pitchers", "batters"]


def _read_polars_csv(source: Path | bytes) -> pl.DataFrame:
    if isinstance(source, bytes):
        return pl.read_csv(io.BytesIO(source))
    resolved = resolve_cache_read_path(source)
    if resolved is not None and resolved.suffix == ".gz":
        raw_bytes = read_cached_bytes(source)
        return pl.read_csv(io.BytesIO(raw_bytes))
    target = resolved if resolved is not None else source
    return pl.read_csv(target)


def parse_statcast_team_metrics_csv(source: Path | bytes) -> pl.DataFrame:
    """Parse Statcast team CSV into `(game_id, event_seq, team, team_xwoba, opp_xwoba)`."""
    df = _read_polars_csv(source)
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "event_seq": pl.Int32,
                "team": pl.Utf8,
                "team_xwoba": pl.Float64,
                "opp_xwoba": pl.Float64,
            }
        )

    game_col = "game_id" if "game_id" in df.columns else "game_pk"
    seq_expr = (
        pl.col("event_seq").cast(pl.Int32)
        if "event_seq" in df.columns
        else pl.lit(1, dtype=pl.Int32).alias("event_seq")
    )
    xwoba_col = "team_xwoba" if "team_xwoba" in df.columns else "xwoba"
    opp_xwoba_expr = (
        pl.col("opp_xwoba").cast(pl.Float64)
        if "opp_xwoba" in df.columns
        else pl.lit(None, dtype=pl.Float64).alias("opp_xwoba")
    )

    return df.select(
        pl.col(game_col).cast(pl.Int64).alias("game_id"),
        seq_expr,
        pl.col("team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8).alias("team"),
        pl.col(xwoba_col).cast(pl.Float64).alias("team_xwoba"),
        opp_xwoba_expr,
    )


def parse_statcast_pitcher_metrics_csv(source: Path | bytes) -> pl.DataFrame:
    """Parse Statcast pitcher game CSV into canonical pitcher micro-metric columns."""
    df = _read_polars_csv(source)
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "event_seq": pl.Int32,
                "player_id": pl.Int64,
                "xwoba_against": pl.Float64,
                "barrel_pct_against": pl.Float64,
                "swstr_pct": pl.Float64,
                "avg_fastball_velo": pl.Float64,
            }
        )

    game_col = "game_id" if "game_id" in df.columns else "game_pk"
    seq_expr = (
        pl.col("event_seq").cast(pl.Int32)
        if "event_seq" in df.columns
        else pl.lit(1, dtype=pl.Int32).alias("event_seq")
    )
    xwoba_col = "xwoba_against" if "xwoba_against" in df.columns else "xwoba"
    barrel_col = (
        "barrel_pct_against" if "barrel_pct_against" in df.columns else "barrel_batted_rate"
    )
    swstr_col = "swstr_pct" if "swstr_pct" in df.columns else "whiff_percent"
    velo_col = "avg_fastball_velo" if "avg_fastball_velo" in df.columns else "release_speed"

    return df.select(
        pl.col(game_col).cast(pl.Int64).alias("game_id"),
        seq_expr,
        pl.col("player_id").cast(pl.Int64),
        (
            pl.col(xwoba_col).cast(pl.Float64)
            if xwoba_col in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("xwoba_against"),
        (
            pl.col(barrel_col).cast(pl.Float64)
            if barrel_col in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("barrel_pct_against"),
        (
            pl.col(swstr_col).cast(pl.Float64)
            if swstr_col in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("swstr_pct"),
        (
            pl.col(velo_col).cast(pl.Float64)
            if velo_col in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("avg_fastball_velo"),
    )


def parse_statcast_batter_metrics_csv(source: Path | bytes) -> pl.DataFrame:
    """Parse Statcast batter CSV into `(game_id, event_seq, player_id, xwoba, ...)`."""
    df = _read_polars_csv(source)
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "event_seq": pl.Int32,
                "player_id": pl.Int64,
                "xwoba": pl.Float64,
                "hard_hit_rate": pl.Float64,
                "barrel_rate": pl.Float64,
            }
        )

    game_col = "game_id" if "game_id" in df.columns else "game_pk"
    seq_expr = (
        pl.col("event_seq").cast(pl.Int32)
        if "event_seq" in df.columns
        else pl.lit(1, dtype=pl.Int32).alias("event_seq")
    )
    hh_col = "hard_hit_rate" if "hard_hit_rate" in df.columns else "hard_hit_percent"
    brl_col = "barrel_rate" if "barrel_rate" in df.columns else "barrel_batted_rate"

    return df.select(
        pl.col(game_col).cast(pl.Int64).alias("game_id"),
        seq_expr,
        pl.col("player_id").cast(pl.Int64),
        (
            pl.col("xwoba").cast(pl.Float64)
            if "xwoba" in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("xwoba"),
        (
            pl.col(hh_col).cast(pl.Float64)
            if hh_col in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("hard_hit_rate"),
        (
            pl.col(brl_col).cast(pl.Float64)
            if brl_col in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("barrel_rate"),
    )


class StatcastClient:
    """Downloader and `.csv.gz` disk cache manager for MLB Statcast metrics."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        base_url: str = STATCAST_BASE_URL,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    def _download_bytes(self, kind: StatcastEntityKind, season: int) -> bytes:
        url = f"{self.base_url}?type={kind}&year={int(season)}"
        with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content

    def _fetch_cached_csv_path(
        self,
        kind: StatcastEntityKind,
        season: int,
        *,
        is_active_season: bool = False,
        ttl_seconds: float = 21600.0,
    ) -> Path:
        gz_path = self.cache_dir / f"statcast_{kind}_{int(season)}.csv.gz"
        existing = resolve_cache_read_path(gz_path)
        if existing is not None:
            if not is_active_season:
                return existing
            age = time.time() - existing.stat().st_mtime
            if age <= ttl_seconds:
                return existing

        raw_bytes = self._download_bytes(kind, season)
        return write_gzip_bytes(gz_path, raw_bytes)

    def fetch_season_team_metrics(
        self,
        season: int,
        *,
        is_active_season: bool = False,
        ttl_seconds: float = 21600.0,
    ) -> pl.DataFrame:
        """Fetch and parse team-game Statcast `xwOBA` metrics for `season`."""
        path = self._fetch_cached_csv_path(
            "teams",
            season,
            is_active_season=is_active_season,
            ttl_seconds=ttl_seconds,
        )
        return parse_statcast_team_metrics_csv(path)

    def fetch_season_pitcher_metrics(
        self,
        season: int,
        *,
        is_active_season: bool = False,
        ttl_seconds: float = 21600.0,
    ) -> pl.DataFrame:
        """Fetch and parse pitcher-game Statcast metrics for `season`."""
        path = self._fetch_cached_csv_path(
            "pitchers",
            season,
            is_active_season=is_active_season,
            ttl_seconds=ttl_seconds,
        )
        return parse_statcast_pitcher_metrics_csv(path)

    def fetch_season_batter_metrics(
        self,
        season: int,
        *,
        is_active_season: bool = False,
        ttl_seconds: float = 21600.0,
    ) -> pl.DataFrame:
        """Fetch and parse batter-game Statcast metrics for `season`."""
        path = self._fetch_cached_csv_path(
            "batters",
            season,
            is_active_season=is_active_season,
            ttl_seconds=ttl_seconds,
        )
        return parse_statcast_batter_metrics_csv(path)
