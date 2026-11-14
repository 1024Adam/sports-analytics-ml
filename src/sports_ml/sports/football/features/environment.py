"""Game environment features: weather penalties, roof/surface/altitude interactions.

Consumes the per-game `build_game_weather` frame (forecast or flagged observed proxy) and the
stadium registry. Every column is a function of pre-game information only: forecast (or
flagged proxy) weather, the *default* roof configuration of the venue (never the observed
game-day roof), the surface, and altitude. Domes/closed roofs carry neutral indoor weather, so
every penalty is exactly zero there.

The `env_*` shift coefficients are fixed, documented priors (not fit on evaluation data); the
Stage 1 GBDT can still learn from the underlying penalties.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from sports_ml.sports.football.constants import COVERED_ROOF_TYPES, get_stadium_metadata
from sports_ml.sports.football.ingestion.weather import (
    GAME_WEATHER_SCHEMA,
    build_game_weather,
    venue_for_game,
)

WIND_THRESHOLD_MPH: Final[float] = 15.0
WIND_SCALE_MPH: Final[float] = 10.0
WIND_EXPONENT: Final[float] = 1.5
COLD_THRESHOLD_F: Final[float] = 40.0
COLD_SCALE_F: Final[float] = 10.0
PRECIP_RISK_PROB: Final[float] = 0.5
WARM_TEAM_MAX_LATITUDE: Final[float] = 35.3
OPEN_ROOF_EXPOSURE: Final[float] = 0.5

# Expected-points shifts per unit of penalty (priors from typical NFL wind/cold/rain effects).
TOTAL_SHIFT_PER_WIND: Final[float] = 1.2
TOTAL_SHIFT_PER_COLD: Final[float] = 0.3
TOTAL_SHIFT_PER_PRECIP: Final[float] = 0.8
PASSING_WEIGHTS: Final[tuple[float, float, float]] = (0.60, 0.25, 0.30)
KICKING_WEIGHTS: Final[tuple[float, float, float]] = (0.80, 0.30, 0.20)
ALTITUDE_KICK_BOOST_PER_KM: Final[float] = 1.0

ENVIRONMENT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "weather_temp_f",
    "weather_wind_mph",
    "weather_precip_prob",
    "weather_is_indoor",
    "weather_is_observed_proxy",
    "weather_is_imputed",
    "weather_precip_is_imputed",
    "roof_exposure",
    "is_turf",
    "altitude_km",
    "wind_penalty",
    "cold_penalty",
    "is_precipitation_risk",
    "home_warm_team_in_cold",
    "away_warm_team_in_cold",
    "diff_warm_team_in_cold",
    "env_passing_penalty",
    "env_kicking_penalty",
    "env_total_shift",
    "cold_penalty_x_turf",
    "precip_risk_x_grass",
    "altitude_kicking_boost",
)


def wind_penalty(wind_mph: NDArray[np.float64]) -> NDArray[np.float64]:
    """Zero up to 15 mph, then `((w - 15) / 10) ** 1.5` (accelerating)."""
    excess = np.clip(np.asarray(wind_mph, dtype=np.float64) - WIND_THRESHOLD_MPH, 0.0, None)
    return np.asarray((excess / WIND_SCALE_MPH) ** WIND_EXPONENT, dtype=np.float64)


def cold_penalty(temp_f: NDArray[np.float64]) -> NDArray[np.float64]:
    """Zero at or above 40 F, then one unit per 10 F below."""
    deficit = np.clip(COLD_THRESHOLD_F - np.asarray(temp_f, dtype=np.float64), 0.0, None)
    return np.asarray(deficit / COLD_SCALE_F, dtype=np.float64)


def _is_warm_or_covered(team: str) -> bool:
    stadium = get_stadium_metadata(team)
    if stadium is None:
        return False
    return stadium.roof in COVERED_ROOF_TYPES or stadium.latitude <= WARM_TEAM_MAX_LATITUDE


def build_environment_features(
    games_df: pl.DataFrame, weather_df: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Return `game_id` + `ENVIRONMENT_FEATURE_COLUMNS` for every game in `games_df`.

    `weather_df` defaults to `build_game_weather(games_df)` without forecasts, i.e. neutral
    imputed outdoor weather flagged `weather_is_imputed = 1` (indoor games stay exact).
    """
    weather = weather_df if weather_df is not None else build_game_weather(games_df)
    games = games_df.select("game_id", "home_team", "away_team", "venue_id").join(
        weather.select(list(GAME_WEATHER_SCHEMA)), on="game_id", how="left"
    )
    n = games.height
    rows = games.to_dicts()
    roof_exposure = np.ones(n)
    is_turf = np.zeros(n)
    altitude_km = np.zeros(n)
    home_warm = np.zeros(n)
    away_warm = np.zeros(n)
    home_is_host = np.zeros(n)
    for i, g in enumerate(rows):
        venue = venue_for_game(g)
        if venue is not None:
            if venue.roof in COVERED_ROOF_TYPES:
                roof_exposure[i] = 0.0
            elif venue.roof == "open_retractable":
                roof_exposure[i] = OPEN_ROOF_EXPOSURE
            is_turf[i] = float(venue.surface == "turf")
            altitude_km[i] = venue.altitude_m / 1000.0
            home_is_host[i] = float(str(g["home_team"]) in venue.primary_teams)
        home_warm[i] = float(_is_warm_or_covered(str(g["home_team"])))
        away_warm[i] = float(_is_warm_or_covered(str(g["away_team"])))

    def col(name: str) -> NDArray[np.float64]:
        return np.asarray(games[name].cast(pl.Float64).fill_null(0.0).to_numpy(), dtype=np.float64)

    temp, wind, precip = col("weather_temp_f"), col("weather_wind_mph"), col("weather_precip_prob")
    indoor = col("weather_is_indoor")
    wp = wind_penalty(wind) * roof_exposure
    cp = cold_penalty(temp) * roof_exposure
    precip_risk = ((precip >= PRECIP_RISK_PROB) & (roof_exposure > 0.0)).astype(np.float64)
    cold_game = (cp > 0.0).astype(np.float64)
    # A warm/dome team playing outdoors in the cold, away from its own venue
    home_cold_visitor = home_warm * cold_game * (1.0 - home_is_host)
    away_cold_visitor = away_warm * cold_game

    pw, cw, rw = PASSING_WEIGHTS
    kw, kc, kr = KICKING_WEIGHTS
    out = pl.DataFrame(
        {
            "game_id": games["game_id"],
            "weather_temp_f": temp,
            "weather_wind_mph": wind,
            "weather_precip_prob": precip,
            "weather_is_indoor": indoor.astype(np.int8),
            "weather_is_observed_proxy": col("weather_is_observed_proxy").astype(np.int8),
            "weather_is_imputed": col("weather_is_imputed").astype(np.int8),
            "weather_precip_is_imputed": col("weather_precip_is_imputed").astype(np.int8),
            "roof_exposure": roof_exposure,
            "is_turf": is_turf,
            "altitude_km": altitude_km,
            "wind_penalty": wp,
            "cold_penalty": cp,
            "is_precipitation_risk": precip_risk,
            "home_warm_team_in_cold": home_cold_visitor,
            "away_warm_team_in_cold": away_cold_visitor,
            "diff_warm_team_in_cold": home_cold_visitor - away_cold_visitor,
            "env_passing_penalty": pw * wp + cw * cp + rw * precip_risk,
            "env_kicking_penalty": kw * wp + kc * cp + kr * precip_risk,
            "env_total_shift": -(
                TOTAL_SHIFT_PER_WIND * wp
                + TOTAL_SHIFT_PER_COLD * cp
                + TOTAL_SHIFT_PER_PRECIP * precip_risk
            ),
            "cold_penalty_x_turf": cp * is_turf,
            "precip_risk_x_grass": precip_risk * (1.0 - is_turf),
            "altitude_kicking_boost": ALTITUDE_KICK_BOOST_PER_KM * altitude_km * (1.0 - indoor),
        }
    )
    return out.select("game_id", *ENVIRONMENT_FEATURE_COLUMNS)


__all__ = [
    "ENVIRONMENT_FEATURE_COLUMNS",
    "build_environment_features",
    "cold_penalty",
    "wind_penalty",
]
