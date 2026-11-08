"""Pre-game forecast weather for NFL games (`football/ingestion/weather.py`).

Historical games use the Open-Meteo *previous runs* archive (`*_previous_day1` hourly variables,
i.e. the forecast issued about 24 h before the valid time); upcoming games use the live Open-Meteo
forecast. Rows are cached under `data/raw/football/weather/` as Parquet, requests retry with
exponential backoff, and an offline mode serves the cache only.

`build_game_weather` turns those hourly forecasts into one row per game:

* Domes and closed-roof venues get neutral indoor values (`weather_is_indoor = 1`).
* Outdoor / open-roof games use the forecast. A season is a *forecast season* only when at least
  `min_forecast_coverage` of its outdoor games have a complete (temperature + wind) forecast.
* Seasons without that coverage fall back to the observed nflverse `temp`/`wind` (quarantined as
  `obs_weather_*` in Silver) for **every** outdoor game of that season and set
  `weather_is_observed_proxy = 1`; forecasts and observations are never mixed within a season.
* Anything still missing becomes a neutral outdoor value with `weather_is_imputed = 1`.

Observed roof state and post-kickoff weather are never read: the roof comes from the stadium
registry and only forecast rows are used for forecast seasons.
"""

from __future__ import annotations

import io
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from sports_ml.sports.football.constants import (
    COVERED_ROOF_TYPES,
    FOOTBALL_STADIUMS,
    FootballStadiumMetadata,
    get_stadium_metadata,
)

OPEN_METEO_PREVIOUS_RUNS_URL: Final[str] = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
DEFAULT_USER_AGENT: Final[str] = (
    "sports-analytics-ml/0.1 (+https://github.com/1024Adam/sports-analytics-ml)"
)

FORECAST_FIRST_SEASON: Final[int] = 2021
PREVIOUS_RUNS_LEAD_HOURS: Final[float] = 24.0
UPCOMING_HORIZON_DAYS: Final[int] = 14
UPCOMING_TTL_SECONDS: Final[float] = 3 * 3600.0
HISTORICAL_LAG: Final[timedelta] = timedelta(days=2)
GAME_WINDOW_HOURS: Final[int] = 4
MIN_FORECAST_COVERAGE: Final[float] = 0.9

INDOOR_TEMP_F: Final[float] = 70.0
INDOOR_WIND_MPH: Final[float] = 0.0
INDOOR_PRECIP_PROB: Final[float] = 0.0
NEUTRAL_OUTDOOR_TEMP_F: Final[float] = 55.0
NEUTRAL_OUTDOOR_WIND_MPH: Final[float] = 7.0
NEUTRAL_OUTDOOR_PRECIP_PROB: Final[float] = 0.15

FORECAST_SCHEMA: Final[dict[str, pl.DataType]] = {
    "venue_id": pl.Utf8(),
    "valid_time_utc": pl.Datetime("us", "UTC"),
    "forecast_temp_f": pl.Float64(),
    "forecast_wind_mph": pl.Float64(),
    "forecast_precip_prob": pl.Float64(),
    "forecast_lead_hours": pl.Float64(),
    "source": pl.Utf8(),
}

GAME_WEATHER_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "forecast_temp_f": pl.Float64(),
    "forecast_wind_mph": pl.Float64(),
    "forecast_precip_prob": pl.Float64(),
    "forecast_lead_hours": pl.Float64(),
    "weather_temp_f": pl.Float64(),
    "weather_wind_mph": pl.Float64(),
    "weather_precip_prob": pl.Float64(),
    "weather_is_indoor": pl.Int8(),
    "weather_is_observed_proxy": pl.Int8(),
    "weather_is_imputed": pl.Int8(),
    "weather_precip_is_imputed": pl.Int8(),
}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class ForecastUnavailableError(RuntimeError):
    """Raised in offline mode (or on a permanent API error) when a forecast cannot be served."""


@dataclass
class ForecastSyncReport:
    """Outcome of `sync_football_forecasts`."""

    windows_requested: int = 0
    windows_from_cache: int = 0
    windows_failed: int = 0
    failures: list[str] = field(default_factory=list)


class OpenMeteoForecastClient:
    """Cached, retrying Open-Meteo forecast client returning hourly Polars frames."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        retry_attempts: int = 3,
        retry_min_wait: float = 0.5,
        offline: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.retry_attempts = retry_attempts
        self.retry_min_wait = retry_min_wait
        self.offline = offline
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------- HTTP

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=self.retry_min_wait, min=self.retry_min_wait, max=8.0),
            retry=retry_if_exception(_is_retryable),
        )
        def _get() -> dict[str, Any]:
            with httpx.Client(
                timeout=self.timeout_seconds,
                transport=self.transport,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            ) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                payload: dict[str, Any] = resp.json()
                return payload

        return _get()

    @staticmethod
    def _hourly_frame(
        venue_id: str,
        payload: dict[str, Any],
        *,
        suffix: str,
        source: str,
        lead_hours: float | None,
        issued_at: datetime | None,
    ) -> pl.DataFrame:
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return pl.DataFrame(schema=FORECAST_SCHEMA)

        def col(name: str) -> list[float | None]:
            values = hourly.get(f"{name}{suffix}")
            return (
                [None if v is None else float(v) for v in values] if values else [None] * len(times)
            )

        valid = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in times]
        precip = [None if p is None else p / 100.0 for p in col("precipitation_probability")]
        if lead_hours is not None:
            leads = [lead_hours] * len(valid)
        else:
            assert issued_at is not None
            leads = [max(0.0, (t - issued_at).total_seconds() / 3600.0) for t in valid]
        return pl.DataFrame(
            {
                "venue_id": [venue_id] * len(valid),
                "valid_time_utc": valid,
                "forecast_temp_f": col("temperature_2m"),
                "forecast_wind_mph": col("wind_speed_10m"),
                "forecast_precip_prob": precip,
                "forecast_lead_hours": leads,
                "source": [source] * len(valid),
            },
            schema=FORECAST_SCHEMA,
        )

    # ------------------------------------------------------------------- cache

    def _historical_path(self, venue_id: str, season: int) -> Path:
        return self.cache_dir / f"previous_day1_{venue_id}_{season}.parquet"

    def _upcoming_path(self, venue_id: str) -> Path:
        return self.cache_dir / f"upcoming_{venue_id}.parquet"

    def fetch_historical(
        self, venue_id: str, lat: float, lon: float, season: int, start: date, end: date
    ) -> tuple[pl.DataFrame, bool]:
        """Return `(previous_day1 forecast for [start, end], served_from_cache)`."""
        path = self._historical_path(venue_id, season)
        if path.exists():
            cached = pl.read_parquet(path)
            if not cached.is_empty():
                lo, hi = cached["valid_time_utc"].min(), cached["valid_time_utc"].max()
                assert isinstance(lo, datetime) and isinstance(hi, datetime)
                if lo.date() <= start and hi.date() >= end:
                    return cached, True
            if self.offline:
                return cached, True
        if self.offline:
            raise ForecastUnavailableError(f"Offline and not cached: {path.name}")
        payload = self._get_json(
            OPEN_METEO_PREVIOUS_RUNS_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": (
                    "temperature_2m_previous_day1,wind_speed_10m_previous_day1,"
                    "precipitation_probability_previous_day1"
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
            },
        )
        frame = self._hourly_frame(
            venue_id,
            payload,
            suffix="_previous_day1",
            source="previous_day1",
            lead_hours=PREVIOUS_RUNS_LEAD_HOURS,
            issued_at=None,
        )
        self._write(path, frame)
        return frame, False

    def fetch_upcoming(
        self,
        venue_id: str,
        lat: float,
        lon: float,
        start: date,
        end: date,
        *,
        now: datetime | None = None,
    ) -> tuple[pl.DataFrame, bool]:
        """Return the live forecast for `[start, end]` (cached for `UPCOMING_TTL_SECONDS`)."""
        path = self._upcoming_path(venue_id)
        if path.exists() and (
            self.offline or time.time() - path.stat().st_mtime <= UPCOMING_TTL_SECONDS
        ):
            return pl.read_parquet(path), True
        if self.offline:
            raise ForecastUnavailableError(f"Offline and not cached: {path.name}")
        issued = now or datetime.now(tz=UTC)
        payload = self._get_json(
            OPEN_METEO_FORECAST_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
            },
        )
        frame = self._hourly_frame(
            venue_id, payload, suffix="", source="live", lead_hours=None, issued_at=issued
        )
        self._write(path, frame)
        return frame, False

    @staticmethod
    def _write(path: Path, frame: pl.DataFrame) -> None:
        buf = io.BytesIO()
        frame.write_parquet(buf, compression="zstd")
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)

    def load_cached(self) -> pl.DataFrame:
        """Return every cached hourly forecast row (empty frame if none)."""
        return load_cached_forecasts(self.cache_dir)


def load_cached_forecasts(cache_dir: Path) -> pl.DataFrame:
    """Read all cached forecast parquet files under `cache_dir` (no network, no mkdir)."""
    files = sorted(cache_dir.glob("*.parquet")) if cache_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=FORECAST_SCHEMA)
    frames = [pl.read_parquet(f).select(list(FORECAST_SCHEMA)) for f in files]
    return pl.concat(frames, how="vertical_relaxed")


def venue_for_game(row: dict[str, Any]) -> FootballStadiumMetadata | None:
    """Resolve the stadium for a Silver game row (venue id first, then the home franchise)."""
    venue_id = row.get("venue_id")
    if venue_id is not None and str(venue_id) in FOOTBALL_STADIUMS:
        return FOOTBALL_STADIUMS[str(venue_id)]
    return get_stadium_metadata(str(row["home_team"]))


def sync_football_forecasts(
    games_df: pl.DataFrame,
    client: OpenMeteoForecastClient,
    *,
    now: datetime | None = None,
    first_season: int = FORECAST_FIRST_SEASON,
) -> ForecastSyncReport:
    """Fetch or load cached forecasts per outdoor venue-season; API errors are recorded."""
    now = now or datetime.now(tz=UTC)
    report = ForecastSyncReport()
    historical: dict[tuple[str, int], list[date]] = defaultdict(list)
    upcoming: dict[str, list[date]] = defaultdict(list)
    venues: dict[str, FootballStadiumMetadata] = {}
    for g in games_df.iter_rows(named=True):
        venue = venue_for_game(g)
        if venue is None or venue.roof in COVERED_ROOF_TYPES:
            continue
        kickoff: datetime = g["start_time_utc"]
        venues[venue.venue_id] = venue
        if kickoff <= now - HISTORICAL_LAG:
            if int(g["season"]) >= first_season:
                historical[(venue.venue_id, int(g["season"]))].append(kickoff.date())
        elif kickoff <= now + timedelta(days=UPCOMING_HORIZON_DAYS):
            upcoming[venue.venue_id].append(kickoff.date())

    def record(label: str, call: Any) -> None:
        report.windows_requested += 1
        try:
            _frame, cached = call()
            report.windows_from_cache += int(cached)
        except (httpx.HTTPError, ForecastUnavailableError, ValueError) as exc:
            report.windows_failed += 1
            report.failures.append(f"{label}: {exc}")

    for (venue_id, season), days in sorted(historical.items()):
        v = venues[venue_id]
        record(
            f"{venue_id}/{season}",
            lambda v=v, season=season, days=days: client.fetch_historical(
                v.venue_id, v.latitude, v.longitude, season, min(days), max(days)
            ),
        )
    for venue_id, days in sorted(upcoming.items()):
        v = venues[venue_id]
        record(
            f"{venue_id}/upcoming",
            lambda v=v, days=days: client.fetch_upcoming(
                v.venue_id, v.latitude, v.longitude, min(days), max(days), now=now
            ),
        )
    return report


def _forecast_lookup(
    forecasts: pl.DataFrame | None,
) -> dict[tuple[str, datetime], tuple[float | None, float | None, float | None, float | None]]:
    out: dict[
        tuple[str, datetime], tuple[float | None, float | None, float | None, float | None]
    ] = {}
    if forecasts is None or forecasts.is_empty():
        return out
    # Archived previous-day runs outrank the live forecast for the same valid time.
    ranked = forecasts.with_columns((pl.col("source") == "previous_day1").alias("_hist")).sort(
        "_hist"
    )
    for r in ranked.iter_rows(named=True):
        out[(str(r["venue_id"]), r["valid_time_utc"])] = (
            r["forecast_temp_f"],
            r["forecast_wind_mph"],
            r["forecast_precip_prob"],
            r["forecast_lead_hours"],
        )
    return out


def _window_forecast(
    lookup: dict[
        tuple[str, datetime], tuple[float | None, float | None, float | None, float | None]
    ],
    venue_id: str,
    kickoff: datetime,
) -> tuple[float | None, float | None, float | None, float | None]:
    start = kickoff.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    temps: list[float] = []
    winds: list[float] = []
    precs: list[float] = []
    leads: list[float] = []
    for h in range(GAME_WINDOW_HOURS):
        hit = lookup.get((venue_id, start + timedelta(hours=h)))
        if hit is None:
            continue
        t, w, p, lead = hit
        if t is not None:
            temps.append(t)
        if w is not None:
            winds.append(w)
        if p is not None:
            precs.append(p)
        if lead is not None:
            leads.append(lead)
    return (
        sum(temps) / len(temps) if temps else None,
        sum(winds) / len(winds) if winds else None,
        max(precs) if precs else None,
        sum(leads) / len(leads) if leads else None,
    )


def build_game_weather(
    games_df: pl.DataFrame,
    forecasts: pl.DataFrame | None = None,
    *,
    min_forecast_coverage: float = MIN_FORECAST_COVERAGE,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Return one `GAME_WEATHER_SCHEMA` row per game (forecast, or a flagged observed proxy)."""
    lookup = _forecast_lookup(forecasts)
    horizon = (now or datetime.now(tz=UTC)) + timedelta(days=UPCOMING_HORIZON_DAYS)
    has_obs = "obs_weather_temp" in games_df.columns and "obs_weather_wind" in games_df.columns
    raw: list[dict[str, Any]] = []
    for g in games_df.iter_rows(named=True):
        venue = venue_for_game(g)
        indoor = venue is not None and venue.roof in COVERED_ROOF_TYPES
        fc: tuple[float | None, ...] = (None, None, None, None)
        if not indoor and venue is not None:
            fc = _window_forecast(lookup, venue.venue_id, g["start_time_utc"])
        raw.append(
            {
                "game_id": int(g["game_id"]),
                "season": int(g["season"]),
                "indoor": indoor,
                "in_horizon": g["start_time_utc"] <= horizon,
                "fc": fc,
                "obs": (g.get("obs_weather_temp"), g.get("obs_weather_wind"))
                if has_obs
                else (None, None),
            }
        )

    outdoor = [r for r in raw if not r["indoor"]]
    by_season: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in outdoor:
        # Games beyond the live-forecast horizon cannot have a forecast yet: don't count them
        if r["in_horizon"]:
            by_season[r["season"]].append(r)
    forecast_seasons = {
        season: sum(1 for r in rows if r["fc"][0] is not None and r["fc"][1] is not None)
        / len(rows)
        >= min_forecast_coverage
        for season, rows in by_season.items()
    }

    out: list[dict[str, Any]] = []
    for r in raw:
        t_fc, w_fc, p_fc, lead = r["fc"] if not r["indoor"] else (None, None, None, None)
        row: dict[str, Any] = {
            "game_id": r["game_id"],
            "forecast_temp_f": t_fc,
            "forecast_wind_mph": w_fc,
            "forecast_precip_prob": p_fc,
            "forecast_lead_hours": lead,
            "weather_is_indoor": int(r["indoor"]),
            "weather_is_observed_proxy": 0,
            "weather_is_imputed": 0,
            "weather_precip_is_imputed": 0,
        }
        if r["indoor"]:
            row |= {
                "weather_temp_f": INDOOR_TEMP_F,
                "weather_wind_mph": INDOOR_WIND_MPH,
                "weather_precip_prob": INDOOR_PRECIP_PROB,
            }
        elif forecast_seasons.get(r["season"], True):
            temp, wind, prec = t_fc, w_fc, p_fc
            imputed = temp is None or wind is None
            row["weather_temp_f"] = NEUTRAL_OUTDOOR_TEMP_F if temp is None else temp
            row["weather_wind_mph"] = NEUTRAL_OUTDOOR_WIND_MPH if wind is None else wind
            row["weather_precip_prob"] = NEUTRAL_OUTDOOR_PRECIP_PROB if prec is None else prec
            row["weather_is_imputed"] = int(imputed)
            row["weather_precip_is_imputed"] = int(prec is None)
        else:
            obs_t, obs_w = r["obs"]
            row["weather_is_observed_proxy"] = 1
            row["weather_temp_f"] = NEUTRAL_OUTDOOR_TEMP_F if obs_t is None else float(obs_t)
            row["weather_wind_mph"] = NEUTRAL_OUTDOOR_WIND_MPH if obs_w is None else float(obs_w)
            row["weather_precip_prob"] = NEUTRAL_OUTDOOR_PRECIP_PROB
            row["weather_is_imputed"] = int(obs_t is None or obs_w is None)
            row["weather_precip_is_imputed"] = 1
        out.append(row)
    return pl.DataFrame(out, schema=GAME_WEATHER_SCHEMA)


__all__ = [
    "FORECAST_SCHEMA",
    "GAME_WEATHER_SCHEMA",
    "ForecastSyncReport",
    "ForecastUnavailableError",
    "OpenMeteoForecastClient",
    "build_game_weather",
    "load_cached_forecasts",
    "sync_football_forecasts",
    "venue_for_game",
]
