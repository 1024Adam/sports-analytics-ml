"""NHL 32-team registry, tricode normalization, arena metadata, and Silver schemas."""

from __future__ import annotations

from typing import Final

import polars as pl
from pydantic import BaseModel, ConfigDict


class ArenaMetadata(BaseModel):
    """NHL team arena geographic and timezone metadata."""

    model_config = ConfigDict(frozen=True)

    team: str
    team_name: str
    arena_name: str
    latitude: float
    longitude: float
    timezone: str


TRICODE_ALIASES: Final[dict[str, str]] = {
    "S.J": "SJS",
    "L.A": "LAK",
    "N.J": "NJD",
    "T.B": "TBL",
    "ARI": "UTA",
    "PHX": "UTA",
    "ATL": "WPG",
    "MON": "MTL",
    "CAL": "CGY",
    "WAS": "WSH",
    "NAS": "NSH",
    "CLB": "CBJ",
    "VEG": "VGK",
}

NHL_TEAMS_32: Final[dict[str, ArenaMetadata]] = {
    "ANA": ArenaMetadata(
        team="ANA",
        team_name="Anaheim Ducks",
        arena_name="Honda Center",
        latitude=33.8078,
        longitude=-117.8765,
        timezone="America/Los_Angeles",
    ),
    "BOS": ArenaMetadata(
        team="BOS",
        team_name="Boston Bruins",
        arena_name="TD Garden",
        latitude=42.3662,
        longitude=-71.0621,
        timezone="America/New_York",
    ),
    "BUF": ArenaMetadata(
        team="BUF",
        team_name="Buffalo Sabres",
        arena_name="KeyBank Center",
        latitude=42.8750,
        longitude=-78.8764,
        timezone="America/New_York",
    ),
    "CAR": ArenaMetadata(
        team="CAR",
        team_name="Carolina Hurricanes",
        arena_name="Lenovo Center",
        latitude=35.8033,
        longitude=-78.7219,
        timezone="America/New_York",
    ),
    "CBJ": ArenaMetadata(
        team="CBJ",
        team_name="Columbus Blue Jackets",
        arena_name="Nationwide Arena",
        latitude=39.9693,
        longitude=-83.0061,
        timezone="America/New_York",
    ),
    "CGY": ArenaMetadata(
        team="CGY",
        team_name="Calgary Flames",
        arena_name="Scotiabank Saddledome",
        latitude=51.0375,
        longitude=-114.0519,
        timezone="America/Edmonton",
    ),
    "CHI": ArenaMetadata(
        team="CHI",
        team_name="Chicago Blackhawks",
        arena_name="United Center",
        latitude=41.8807,
        longitude=-87.6742,
        timezone="America/Chicago",
    ),
    "COL": ArenaMetadata(
        team="COL",
        team_name="Colorado Avalanche",
        arena_name="Ball Arena",
        latitude=39.7487,
        longitude=-105.0077,
        timezone="America/Denver",
    ),
    "DAL": ArenaMetadata(
        team="DAL",
        team_name="Dallas Stars",
        arena_name="American Airlines Center",
        latitude=32.7905,
        longitude=-96.8103,
        timezone="America/Chicago",
    ),
    "DET": ArenaMetadata(
        team="DET",
        team_name="Detroit Red Wings",
        arena_name="Little Caesars Arena",
        latitude=42.3411,
        longitude=-83.0553,
        timezone="America/Detroit",
    ),
    "EDM": ArenaMetadata(
        team="EDM",
        team_name="Edmonton Oilers",
        arena_name="Rogers Place",
        latitude=53.5469,
        longitude=-113.4978,
        timezone="America/Edmonton",
    ),
    "FLA": ArenaMetadata(
        team="FLA",
        team_name="Florida Panthers",
        arena_name="Amerant Bank Arena",
        latitude=26.1584,
        longitude=-80.3256,
        timezone="America/New_York",
    ),
    "LAK": ArenaMetadata(
        team="LAK",
        team_name="Los Angeles Kings",
        arena_name="Crypto.com Arena",
        latitude=34.0430,
        longitude=-118.2673,
        timezone="America/Los_Angeles",
    ),
    "MIN": ArenaMetadata(
        team="MIN",
        team_name="Minnesota Wild",
        arena_name="Xcel Energy Center",
        latitude=44.9448,
        longitude=-93.1011,
        timezone="America/Chicago",
    ),
    "MTL": ArenaMetadata(
        team="MTL",
        team_name="Montreal Canadiens",
        arena_name="Bell Centre",
        latitude=45.4961,
        longitude=-73.5693,
        timezone="America/Toronto",
    ),
    "NJD": ArenaMetadata(
        team="NJD",
        team_name="New Jersey Devils",
        arena_name="Prudential Center",
        latitude=40.7336,
        longitude=-74.1711,
        timezone="America/New_York",
    ),
    "NSH": ArenaMetadata(
        team="NSH",
        team_name="Nashville Predators",
        arena_name="Bridgestone Arena",
        latitude=36.1592,
        longitude=-86.7785,
        timezone="America/Chicago",
    ),
    "NYI": ArenaMetadata(
        team="NYI",
        team_name="New York Islanders",
        arena_name="UBS Arena",
        latitude=40.7118,
        longitude=-73.7260,
        timezone="America/New_York",
    ),
    "NYR": ArenaMetadata(
        team="NYR",
        team_name="New York Rangers",
        arena_name="Madison Square Garden",
        latitude=40.7505,
        longitude=-73.9934,
        timezone="America/New_York",
    ),
    "OTT": ArenaMetadata(
        team="OTT",
        team_name="Ottawa Senators",
        arena_name="Canadian Tire Centre",
        latitude=45.2969,
        longitude=-75.9272,
        timezone="America/Toronto",
    ),
    "PHI": ArenaMetadata(
        team="PHI",
        team_name="Philadelphia Flyers",
        arena_name="Wells Fargo Center",
        latitude=39.9012,
        longitude=-75.1720,
        timezone="America/New_York",
    ),
    "PIT": ArenaMetadata(
        team="PIT",
        team_name="Pittsburgh Penguins",
        arena_name="PPG Paints Arena",
        latitude=40.4396,
        longitude=-79.9892,
        timezone="America/New_York",
    ),
    "SEA": ArenaMetadata(
        team="SEA",
        team_name="Seattle Kraken",
        arena_name="Climate Pledge Arena",
        latitude=47.6221,
        longitude=-122.3540,
        timezone="America/Los_Angeles",
    ),
    "SJS": ArenaMetadata(
        team="SJS",
        team_name="San Jose Sharks",
        arena_name="SAP Center",
        latitude=37.3328,
        longitude=-121.9012,
        timezone="America/Los_Angeles",
    ),
    "STL": ArenaMetadata(
        team="STL",
        team_name="St. Louis Blues",
        arena_name="Enterprise Center",
        latitude=38.6268,
        longitude=-90.2026,
        timezone="America/Chicago",
    ),
    "TBL": ArenaMetadata(
        team="TBL",
        team_name="Tampa Bay Lightning",
        arena_name="Amalie Arena",
        latitude=27.9427,
        longitude=-82.4519,
        timezone="America/New_York",
    ),
    "TOR": ArenaMetadata(
        team="TOR",
        team_name="Toronto Maple Leafs",
        arena_name="Scotiabank Arena",
        latitude=43.6435,
        longitude=-79.3791,
        timezone="America/Toronto",
    ),
    "UTA": ArenaMetadata(
        team="UTA",
        team_name="Utah Hockey Club",
        arena_name="Delta Center",
        latitude=40.7683,
        longitude=-111.9011,
        timezone="America/Denver",
    ),
    "VAN": ArenaMetadata(
        team="VAN",
        team_name="Vancouver Canucks",
        arena_name="Rogers Arena",
        latitude=49.2778,
        longitude=-123.1089,
        timezone="America/Vancouver",
    ),
    "VGK": ArenaMetadata(
        team="VGK",
        team_name="Vegas Golden Knights",
        arena_name="T-Mobile Arena",
        latitude=36.1029,
        longitude=-115.1784,
        timezone="America/Los_Angeles",
    ),
    "WPG": ArenaMetadata(
        team="WPG",
        team_name="Winnipeg Jets",
        arena_name="Canada Life Centre",
        latitude=49.8928,
        longitude=-97.1436,
        timezone="America/Winnipeg",
    ),
    "WSH": ArenaMetadata(
        team="WSH",
        team_name="Washington Capitals",
        arena_name="Capital One Arena",
        latitude=38.8981,
        longitude=-77.0209,
        timezone="America/New_York",
    ),
}


def normalize_tricode(tricode: str) -> str:
    """Normalize historical or punctuation NHL team tricodes to canonical 3-letter codes."""
    cleaned = tricode.strip().upper()
    return TRICODE_ALIASES.get(cleaned, cleaned)


def get_arena_metadata(tricode: str) -> ArenaMetadata | None:
    """Return arena metadata for a canonical or aliased NHL team tricode."""
    canonical = normalize_tricode(tricode)
    return NHL_TEAMS_32.get(canonical)


SILVER_GAMES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "sport": pl.Utf8(),
    "season": pl.Int32(),
    "game_type": pl.Int8(),
    "game_id": pl.Int64(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "home_team": pl.Utf8(),
    "away_team": pl.Utf8(),
    "venue_timezone": pl.Utf8(),
    "game_state": pl.Utf8(),
    "finish_period_type": pl.Utf8(),
    "home_goals": pl.Int16(),
    "away_goals": pl.Int16(),
    "home_reg_goals": pl.Int16(),
    "away_reg_goals": pl.Int16(),
    "home_win": pl.Int8(),
    "market_home_decimal_odds": pl.Float64(),
    "market_away_decimal_odds": pl.Float64(),
    "market_prob_home": pl.Float64(),
    "market_prob_away": pl.Float64(),
    "market_overround": pl.Float64(),
}

SILVER_TEAM_GAME_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "goals_for": pl.Int16(),
    "goals_against": pl.Int16(),
    "sog_for": pl.Int16(),
    "sog_against": pl.Int16(),
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
    "pp_opportunities": pl.Float64(),
    "pp_xgf": pl.Float64(),
    "pp_toi_min": pl.Float64(),
    "pk_goals_against": pl.Float64(),
    "pk_times_shorthanded": pl.Float64(),
    "pk_xga": pl.Float64(),
    "pk_toi_min": pl.Float64(),
    "has_moneypuck_xg": pl.Boolean(),
}

SILVER_SKATER_GAME_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "player_id": pl.Int64(),
    "player_name": pl.Utf8(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "position_group": pl.Utf8(),
    "toi_total_min": pl.Float64(),
    "toi_ev_min": pl.Float64(),
    "toi_pp_min": pl.Float64(),
    "toi_pk_min": pl.Float64(),
    "goals": pl.Int16(),
    "assists": pl.Int16(),
    "points": pl.Int16(),
    "sog": pl.Int16(),
    "ixg_all": pl.Float64(),
    "ixg_5v5": pl.Float64(),
    "icf_all": pl.Float64(),
    "on_ice_xgf_5v5": pl.Float64(),
    "on_ice_xga_5v5": pl.Float64(),
    "game_score": pl.Float64(),
}

SILVER_GOALIE_GAME_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "player_id": pl.Int64(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "is_starter": pl.Boolean(),
    "toi_min": pl.Float64(),
    "shots_against": pl.Int16(),
    "saves": pl.Int16(),
    "goals_against": pl.Int16(),
    "xga_all": pl.Float64(),
    "gsax": pl.Float64(),
}

SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "score_adj_xgf_5v5": pl.Float64(),
    "score_adj_xga_5v5": pl.Float64(),
    "score_adj_ff_5v5": pl.Float64(),
    "score_adj_fa_5v5": pl.Float64(),
    "rush_xgf_5v5": pl.Float64(),
    "rebound_xgf_5v5": pl.Float64(),
    "rebound_xga_5v5": pl.Float64(),
    "avg_shot_distance_5v5": pl.Float64(),
    "inner_slot_xg_share": pl.Float64(),
}

SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "player_id": pl.Int64(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "rush_ixg_5v5": pl.Float64(),
    "rebound_ixg_5v5": pl.Float64(),
    "inner_slot_ixg_5v5": pl.Float64(),
    "qoc_opponent_toi_share": pl.Float64(),
    "qot_teammate_ixg_per_60": pl.Float64(),
}
