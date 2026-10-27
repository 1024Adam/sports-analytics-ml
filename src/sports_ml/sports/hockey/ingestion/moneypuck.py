"""MoneyPuck historical and active-season CSV downloader and parser (`ingestion/moneypuck.py`).

Parses `5on5`, `5on4`, `4on5`, and `all` situation metrics (`xGF`, `xGA`,
`Corsi`, `Fenwick`, `High-Danger xG`, and goalie `GSAx`) with ETag/TTL caching.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sports_ml.core.cache import resolve_cache_read_path, write_gzip_bytes
from sports_ml.sports.hockey.constants import normalize_tricode

MONEYPUCK_BASE_URL: Final[str] = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame"
MONEYPUCK_SEASON_URL: Final[str] = "https://moneypuck.com/moneypuck/playerData/seasonSummary"
MONEYPUCK_BULK_DOWNLOAD_URL: Final[str] = (
    "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary"
)
DEFAULT_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)


def _normalize_team_expr(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
        .map_elements(normalize_tricode, return_dtype=pl.Utf8)
    )


def parse_moneypuck_teams_csv(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Pivot and normalize MoneyPuck team game-by-game CSV across `5on5`, `5on4`, and `4on5`."""
    if raw_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64(),
                "team": pl.Utf8(),
                "xgf_5v5": pl.Float64(),
                "xga_5v5": pl.Float64(),
                "cf_5v5": pl.Int16(),
                "ca_5v5": pl.Int16(),
                "ff_5v5": pl.Int16(),
                "fa_5v5": pl.Int16(),
                "hd_xgf_5v5": pl.Float64(),
                "hd_xga_5v5": pl.Float64(),
                "toi_5v5_minutes": pl.Float64(),
                "pp_goals": pl.Float64(),
                "pp_xgf": pl.Float64(),
                "pp_toi_min": pl.Float64(),
                "pk_goals_against": pl.Float64(),
                "pk_xga": pl.Float64(),
                "pk_toi_min": pl.Float64(),
            }
        )

    game_col = "gameId" if "gameId" in raw_df.columns else "game_id"
    team_col = "team" if "team" in raw_df.columns else "playerTeam"

    cleaned = raw_df.with_columns(
        pl.col(game_col).cast(pl.Int64).alias("game_id"),
        _normalize_team_expr(team_col).alias("team"),
        pl.col("situation").cast(pl.Utf8).str.strip_chars(),
    )

    df_5v5 = (
        cleaned.filter(pl.col("situation") == "5on5")
        .select(
            pl.col("game_id"),
            pl.col("team"),
            pl.col("xGoalsFor").cast(pl.Float64).alias("xgf_5v5"),
            pl.col("xGoalsAgainst").cast(pl.Float64).alias("xga_5v5"),
            pl.col("shotAttemptsFor").cast(pl.Float64).round(0).cast(pl.Int16).alias("cf_5v5"),
            pl.col("shotAttemptsAgainst").cast(pl.Float64).round(0).cast(pl.Int16).alias("ca_5v5"),
            pl.col("unblockedShotAttemptsFor")
            .cast(pl.Float64)
            .round(0)
            .cast(pl.Int16)
            .alias("ff_5v5"),
            pl.col("unblockedShotAttemptsAgainst")
            .cast(pl.Float64)
            .round(0)
            .cast(pl.Int16)
            .alias("fa_5v5"),
            pl.col("highDangerxGoalsFor").cast(pl.Float64).alias("hd_xgf_5v5"),
            pl.col("highDangerxGoalsAgainst").cast(pl.Float64).alias("hd_xga_5v5"),
            (pl.col("iceTime").cast(pl.Float64) / 60.0).alias("toi_5v5_minutes"),
        )
        .unique(subset=["game_id", "team"], keep="last")
    )

    df_5v4 = (
        cleaned.filter(pl.col("situation") == "5on4")
        .select(
            pl.col("game_id"),
            pl.col("team"),
            pl.col("goalsFor").cast(pl.Float64).alias("pp_goals"),
            pl.col("xGoalsFor").cast(pl.Float64).alias("pp_xgf"),
            (pl.col("iceTime").cast(pl.Float64) / 60.0).alias("pp_toi_min"),
        )
        .unique(subset=["game_id", "team"], keep="last")
    )

    df_4v5 = (
        cleaned.filter(pl.col("situation") == "4on5")
        .select(
            pl.col("game_id"),
            pl.col("team"),
            pl.col("goalsAgainst").cast(pl.Float64).alias("pk_goals_against"),
            pl.col("xGoalsAgainst").cast(pl.Float64).alias("pk_xga"),
            (pl.col("iceTime").cast(pl.Float64) / 60.0).alias("pk_toi_min"),
        )
        .unique(subset=["game_id", "team"], keep="last")
    )

    joined = df_5v5.join(df_5v4, on=["game_id", "team"], how="left").join(
        df_4v5, on=["game_id", "team"], how="left"
    )
    return joined.with_columns(
        pl.col("pp_goals").fill_null(0.0),
        pl.col("pp_xgf").fill_null(0.0),
        pl.col("pp_toi_min").fill_null(0.0),
        pl.col("pk_goals_against").fill_null(0.0),
        pl.col("pk_xga").fill_null(0.0),
        pl.col("pk_toi_min").fill_null(0.0),
    )


def parse_moneypuck_skaters_csv(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Pivot and normalize MoneyPuck skater game-by-game CSV (`all` and `5on5` situations)."""
    if raw_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64(),
                "player_id": pl.Int64(),
                "team": pl.Utf8(),
                "ixg_all": pl.Float64(),
                "ixg_5v5": pl.Float64(),
                "icf_all": pl.Float64(),
                "on_ice_xgf_5v5": pl.Float64(),
                "on_ice_xga_5v5": pl.Float64(),
                "game_score": pl.Float64(),
            }
        )

    game_col = "gameId" if "gameId" in raw_df.columns else "game_id"
    player_col = "playerId" if "playerId" in raw_df.columns else "player_id"
    team_col = "playerTeam" if "playerTeam" in raw_df.columns else "team"

    cleaned = raw_df.with_columns(
        pl.col(game_col).cast(pl.Int64).alias("game_id"),
        pl.col(player_col).cast(pl.Int64).alias("player_id"),
        _normalize_team_expr(team_col).alias("team"),
        pl.col("situation").cast(pl.Utf8).str.strip_chars(),
    )

    gs_expr = (
        pl.col("gameScore").cast(pl.Float64)
        if "gameScore" in cleaned.columns
        else (0.75 * pl.col("I_F_xGoals").cast(pl.Float64))
    )

    df_all = (
        cleaned.filter(pl.col("situation") == "all")
        .select(
            pl.col("game_id"),
            pl.col("player_id"),
            pl.col("team"),
            pl.col("I_F_xGoals").cast(pl.Float64).alias("ixg_all"),
            pl.col("I_F_shotAttempts").cast(pl.Float64).alias("icf_all"),
            gs_expr.alias("game_score"),
        )
        .unique(subset=["game_id", "player_id"], keep="last")
    )

    df_5v5 = (
        cleaned.filter(pl.col("situation") == "5on5")
        .select(
            pl.col("game_id"),
            pl.col("player_id"),
            pl.col("I_F_xGoals").cast(pl.Float64).alias("ixg_5v5"),
            pl.col("OnIce_F_xGoals").cast(pl.Float64).alias("on_ice_xgf_5v5"),
            pl.col("OnIce_A_xGoals").cast(pl.Float64).alias("on_ice_xga_5v5"),
        )
        .unique(subset=["game_id", "player_id"], keep="last")
    )

    return df_all.join(df_5v5, on=["game_id", "player_id"], how="left").with_columns(
        pl.col("ixg_5v5").fill_null(0.0),
        pl.col("on_ice_xgf_5v5").fill_null(0.0),
        pl.col("on_ice_xga_5v5").fill_null(0.0),
    )


def parse_moneypuck_goalies_csv(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Normalize MoneyPuck goalie game-by-game CSV (`all` situation) and compute GSAx."""
    if raw_df.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64(),
                "player_id": pl.Int64(),
                "team": pl.Utf8(),
                "xga_all": pl.Float64(),
                "gsax": pl.Float64(),
            }
        )

    game_col = "gameId" if "gameId" in raw_df.columns else "game_id"
    player_col = "playerId" if "playerId" in raw_df.columns else "player_id"
    team_col = "playerTeam" if "playerTeam" in raw_df.columns else "team"

    cleaned = raw_df.with_columns(
        pl.col(game_col).cast(pl.Int64).alias("game_id"),
        pl.col(player_col).cast(pl.Int64).alias("player_id"),
        _normalize_team_expr(team_col).alias("team"),
        pl.col("situation").cast(pl.Utf8).str.strip_chars(),
    )

    return (
        cleaned.filter(pl.col("situation") == "all")
        .select(
            pl.col("game_id"),
            pl.col("player_id"),
            pl.col("team"),
            pl.col("xGoals").cast(pl.Float64).alias("xga_all"),
            (pl.col("xGoals").cast(pl.Float64) - pl.col("goals").cast(pl.Float64)).alias("gsax"),
        )
        .unique(subset=["game_id", "player_id"], keep="last")
    )


class MoneyPuckClient:
    """Downloader and Polars parser for MoneyPuck team, skater, and goalie CSV snapshots."""

    def __init__(
        self,
        cache_dir: Path,
        base_url: str = MONEYPUCK_SEASON_URL,
        ttl_seconds: float = 21600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.base_url = base_url.rstrip("/")
        self.ttl_seconds = ttl_seconds
        self.transport = transport
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_download_url(self, entity_kind: str, season_start_year: int) -> str:
        """Resolve live MoneyPuck game-by-game URL (or custom base_url override)."""
        if self.base_url != MONEYPUCK_SEASON_URL.rstrip("/"):
            return f"{self.base_url}/{season_start_year}/{entity_kind}.csv"
        if entity_kind in {"skaters", "goalies"}:
            return f"{MONEYPUCK_BULK_DOWNLOAD_URL}/{entity_kind}/{season_start_year}.zip"
        return f"{MONEYPUCK_BASE_URL}/all_teams.csv"

    @staticmethod
    def _extract_csv_bytes(raw_bytes: bytes) -> bytes:
        """Extract CSV payload from ZIP archive bytes or return raw CSV bytes directly."""
        if raw_bytes[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                target_name = csv_names[0] if csv_names else zf.namelist()[0]
                return zf.read(target_name)
        return raw_bytes

    def _is_shared_all_teams_endpoint(self, entity_kind: str) -> bool:
        """Return True when fetching cumulative `all_teams.csv` from the default endpoint."""
        return entity_kind == "teams" and self.base_url == MONEYPUCK_SEASON_URL.rstrip("/")

    def _cache_stem_for_entity(self, entity_kind: str, season_start_year: int) -> str:
        """Resolve cache file stem (`teams_all` for default cumulative teams, else per-season)."""
        if self._is_shared_all_teams_endpoint(entity_kind):
            return "teams_all"
        return f"{entity_kind}_{season_start_year}"

    def _resolve_existing_teams_cache_path(
        self,
        season_start_year: int,
    ) -> tuple[Path | None, Path]:
        """Resolve `teams_all.csv.gz` (or legacy `teams_{YYYY}.csv[.gz]`) and meta path."""
        stem = self._cache_stem_for_entity("teams", season_start_year)
        preferred_gz = self.cache_dir / f"{stem}.csv.gz"
        meta_path = self.cache_dir / f"{stem}.meta.json"
        resolved = resolve_cache_read_path(preferred_gz)
        if resolved is not None:
            return resolved, meta_path

        if self._is_shared_all_teams_endpoint("teams"):
            legacy_season = resolve_cache_read_path(
                self.cache_dir / f"teams_{season_start_year}.csv.gz"
            )
            if legacy_season is not None:
                legacy_meta = self.cache_dir / f"teams_{season_start_year}.meta.json"
                return legacy_season, (legacy_meta if legacy_meta.exists() else meta_path)
        return None, meta_path

    @staticmethod
    def _cached_csv_contains_season(csv_path: Path, season_start_year: int) -> bool:
        """Check whether a cached cumulative CSV contains rows for `season_start_year`."""
        try:
            schema = pl.read_csv(csv_path, n_rows=1).schema
            if "season" not in schema:
                return True
            matched = (
                pl.scan_csv(csv_path)
                .filter(pl.col("season").cast(pl.Int32) == season_start_year)
                .limit(1)
                .collect()
            )
            return not matched.is_empty()
        except Exception:
            return False

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _download_with_etag_or_ttl(
        self,
        entity_kind: str,
        season_start_year: int,
        *,
        is_active_season: bool = False,
    ) -> Path:
        """Download or refresh a MoneyPuck CSV/ZIP using ETag / Last-Modified and TTL rules."""
        stem = self._cache_stem_for_entity(entity_kind, season_start_year)
        preferred_gz = self.cache_dir / f"{stem}.csv.gz"
        meta_path = self.cache_dir / f"{stem}.meta.json"

        if entity_kind == "teams":
            existing_path, existing_meta_path = self._resolve_existing_teams_cache_path(
                season_start_year
            )
        else:
            existing_path = resolve_cache_read_path(preferred_gz)
            existing_meta_path = meta_path

        if existing_path is not None:
            has_requested_season = (
                self._cached_csv_contains_season(existing_path, season_start_year)
                if self._is_shared_all_teams_endpoint(entity_kind)
                else True
            )
            if has_requested_season:
                if not is_active_season:
                    return existing_path
                age = time.time() - existing_path.stat().st_mtime
                if age <= self.ttl_seconds:
                    return existing_path

        headers: dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}
        if existing_path is not None and existing_meta_path.exists():
            meta: dict[str, Any] = json.loads(existing_meta_path.read_text(encoding="utf-8"))
            if "etag" in meta and meta["etag"]:
                headers["If-None-Match"] = str(meta["etag"])
            if "last_modified" in meta and meta["last_modified"]:
                headers["If-Modified-Since"] = str(meta["last_modified"])

        url = self._resolve_download_url(entity_kind, season_start_year)
        with httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            transport=self.transport,
        ) as client:
            response = client.get(url, headers=headers)
            if response.status_code == 304 and existing_path is not None:
                existing_path.touch()
                return existing_path
            response.raise_for_status()
            csv_bytes = self._extract_csv_bytes(response.content)
            written_gz = write_gzip_bytes(preferred_gz, csv_bytes)
            meta_out = {
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "source_url": url,
            }
            meta_path.write_text(json.dumps(meta_out), encoding="utf-8")

        return written_gz

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def load_single_team_game_stats(
        self,
        team_tricode: str,
        season_start_year: int | None = None,
    ) -> pl.DataFrame:
        """Download and parse a single team's game-by-game CSV (`regular/teams/{TEAM}.csv`)."""
        team_code = normalize_tricode(team_tricode)
        url = f"{MONEYPUCK_BASE_URL}/regular/teams/{team_code}.csv"
        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            transport=self.transport,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            raw_df = pl.read_csv(io.BytesIO(response.content))
        if season_start_year is not None and "season" in raw_df.columns:
            raw_df = raw_df.filter(pl.col("season").cast(pl.Int32) == season_start_year)
        return parse_moneypuck_teams_csv(raw_df)

    def load_team_game_stats(
        self,
        season_start_year: int,
        *,
        is_active_season: bool = False,
        csv_bytes: bytes | None = None,
    ) -> pl.DataFrame:
        """Load and normalize MoneyPuck team game-by-game situational stats."""
        if csv_bytes is not None:
            raw_df = pl.read_csv(io.BytesIO(self._extract_csv_bytes(csv_bytes)))
        else:
            csv_path = self._download_with_etag_or_ttl(
                "teams", season_start_year, is_active_season=is_active_season
            )
            raw_df = pl.read_csv(csv_path)
        if "season" in raw_df.columns:
            raw_df = raw_df.filter(pl.col("season").cast(pl.Int32) == season_start_year)
        return parse_moneypuck_teams_csv(raw_df)

    def load_skater_game_stats(
        self,
        season_start_year: int,
        *,
        is_active_season: bool = False,
        csv_bytes: bytes | None = None,
    ) -> pl.DataFrame:
        """Load and normalize MoneyPuck skater game-by-game situational stats."""
        if csv_bytes is not None:
            raw_df = pl.read_csv(io.BytesIO(self._extract_csv_bytes(csv_bytes)))
        else:
            csv_path = self._download_with_etag_or_ttl(
                "skaters", season_start_year, is_active_season=is_active_season
            )
            raw_df = pl.read_csv(csv_path)
        if "season" in raw_df.columns:
            raw_df = raw_df.filter(pl.col("season").cast(pl.Int32) == season_start_year)
        return parse_moneypuck_skaters_csv(raw_df)

    def load_goalie_game_stats(
        self,
        season_start_year: int,
        *,
        is_active_season: bool = False,
        csv_bytes: bytes | None = None,
    ) -> pl.DataFrame:
        """Load and normalize MoneyPuck goalie game-by-game situational stats."""
        if csv_bytes is not None:
            raw_df = pl.read_csv(io.BytesIO(self._extract_csv_bytes(csv_bytes)))
        else:
            csv_path = self._download_with_etag_or_ttl(
                "goalies", season_start_year, is_active_season=is_active_season
            )
            raw_df = pl.read_csv(csv_path)
        if "season" in raw_df.columns:
            raw_df = raw_df.filter(pl.col("season").cast(pl.Int32) == season_start_year)
        return parse_moneypuck_goalies_csv(raw_df)
