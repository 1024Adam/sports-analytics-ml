"""EPL club registry, cross-source tricode normalization, stadium metadata, and Silver schemas."""

from __future__ import annotations

from typing import Final

import polars as pl
from pydantic import BaseModel, ConfigDict


class StadiumMetadata(BaseModel):
    """EPL club stadium geographic and timezone metadata."""

    model_config = ConfigDict(frozen=True)

    team: str
    team_name: str
    stadium_name: str
    latitude: float
    longitude: float
    timezone: str = "Europe/London"


EPL_CLUBS: Final[dict[str, StadiumMetadata]] = {
    "ARS": StadiumMetadata(
        team="ARS",
        team_name="Arsenal",
        stadium_name="Emirates Stadium",
        latitude=51.5549,
        longitude=-0.1084,
    ),
    "AVL": StadiumMetadata(
        team="AVL",
        team_name="Aston Villa",
        stadium_name="Villa Park",
        latitude=52.5092,
        longitude=-1.8847,
    ),
    "BHA": StadiumMetadata(
        team="BHA",
        team_name="Brighton & Hove Albion",
        stadium_name="American Express Stadium",
        latitude=50.8616,
        longitude=-0.0837,
    ),
    "BOU": StadiumMetadata(
        team="BOU",
        team_name="AFC Bournemouth",
        stadium_name="Vitality Stadium",
        latitude=50.7352,
        longitude=-1.8384,
    ),
    "BRE": StadiumMetadata(
        team="BRE",
        team_name="Brentford",
        stadium_name="Gtech Community Stadium",
        latitude=51.4907,
        longitude=-0.2886,
    ),
    "BUR": StadiumMetadata(
        team="BUR",
        team_name="Burnley",
        stadium_name="Turf Moor",
        latitude=53.7890,
        longitude=-2.2302,
    ),
    "CHE": StadiumMetadata(
        team="CHE",
        team_name="Chelsea",
        stadium_name="Stamford Bridge",
        latitude=51.4817,
        longitude=-0.1910,
    ),
    "CRY": StadiumMetadata(
        team="CRY",
        team_name="Crystal Palace",
        stadium_name="Selhurst Park",
        latitude=51.3983,
        longitude=-0.0856,
    ),
    "EVE": StadiumMetadata(
        team="EVE",
        team_name="Everton",
        stadium_name="Goodison Park",
        latitude=53.4388,
        longitude=-2.9663,
    ),
    "FUL": StadiumMetadata(
        team="FUL",
        team_name="Fulham",
        stadium_name="Craven Cottage",
        latitude=51.4749,
        longitude=-0.2216,
    ),
    "IPS": StadiumMetadata(
        team="IPS",
        team_name="Ipswich Town",
        stadium_name="Portman Road",
        latitude=52.0544,
        longitude=1.1456,
    ),
    "LEE": StadiumMetadata(
        team="LEE",
        team_name="Leeds United",
        stadium_name="Elland Road",
        latitude=53.7778,
        longitude=-1.5721,
    ),
    "LEI": StadiumMetadata(
        team="LEI",
        team_name="Leicester City",
        stadium_name="King Power Stadium",
        latitude=52.6204,
        longitude=-1.1422,
    ),
    "LIV": StadiumMetadata(
        team="LIV",
        team_name="Liverpool",
        stadium_name="Anfield",
        latitude=53.4308,
        longitude=-2.9608,
    ),
    "LUT": StadiumMetadata(
        team="LUT",
        team_name="Luton Town",
        stadium_name="Kenilworth Road",
        latitude=51.8842,
        longitude=-0.4316,
    ),
    "MCI": StadiumMetadata(
        team="MCI",
        team_name="Manchester City",
        stadium_name="Etihad Stadium",
        latitude=53.4831,
        longitude=-2.2004,
    ),
    "MUN": StadiumMetadata(
        team="MUN",
        team_name="Manchester United",
        stadium_name="Old Trafford",
        latitude=53.4631,
        longitude=-2.2913,
    ),
    "NEW": StadiumMetadata(
        team="NEW",
        team_name="Newcastle United",
        stadium_name="St James' Park",
        latitude=54.9756,
        longitude=-1.6217,
    ),
    "NFO": StadiumMetadata(
        team="NFO",
        team_name="Nottingham Forest",
        stadium_name="City Ground",
        latitude=52.9400,
        longitude=-1.1329,
    ),
    "NOR": StadiumMetadata(
        team="NOR",
        team_name="Norwich City",
        stadium_name="Carrow Road",
        latitude=52.6221,
        longitude=1.3091,
    ),
    "SHU": StadiumMetadata(
        team="SHU",
        team_name="Sheffield United",
        stadium_name="Bramall Lane",
        latitude=53.3703,
        longitude=-1.4708,
    ),
    "SOU": StadiumMetadata(
        team="SOU",
        team_name="Southampton",
        stadium_name="St Mary's Stadium",
        latitude=50.9058,
        longitude=-1.3910,
    ),
    "SUN": StadiumMetadata(
        team="SUN",
        team_name="Sunderland",
        stadium_name="Stadium of Light",
        latitude=54.9146,
        longitude=-1.3884,
    ),
    "TOT": StadiumMetadata(
        team="TOT",
        team_name="Tottenham Hotspur",
        stadium_name="Tottenham Hotspur Stadium",
        latitude=51.6042,
        longitude=-0.0662,
    ),
    "WAT": StadiumMetadata(
        team="WAT",
        team_name="Watford",
        stadium_name="Vicarage Road",
        latitude=51.6499,
        longitude=-0.4015,
    ),
    "WBA": StadiumMetadata(
        team="WBA",
        team_name="West Bromwich Albion",
        stadium_name="The Hawthorns",
        latitude=52.5090,
        longitude=-1.9639,
    ),
    "WHU": StadiumMetadata(
        team="WHU",
        team_name="West Ham United",
        stadium_name="London Stadium",
        latitude=51.5387,
        longitude=-0.0166,
    ),
    "WOL": StadiumMetadata(
        team="WOL",
        team_name="Wolverhampton Wanderers",
        stadium_name="Molineux Stadium",
        latitude=52.5902,
        longitude=-2.1304,
    ),
}

SOCCER_CLUB_ALIASES: Final[dict[str, str]] = {
    "ARSENAL": "ARS",
    "ARSENAL FC": "ARS",
    "ASTON VILLA": "AVL",
    "VILLA": "AVL",
    "BRIGHTON": "BHA",
    "BRIGHTON AND HOVE ALBION": "BHA",
    "BRIGHTON & HOVE ALBION": "BHA",
    "BOURNEMOUTH": "BOU",
    "AFC BOURNEMOUTH": "BOU",
    "BRENTFORD": "BRE",
    "BRENTFORD FC": "BRE",
    "BURNLEY": "BUR",
    "BURNLEY FC": "BUR",
    "CHELSEA": "CHE",
    "CHELSEA FC": "CHE",
    "CRYSTAL PALACE": "CRY",
    "PALACE": "CRY",
    "EVERTON": "EVE",
    "EVERTON FC": "EVE",
    "FULHAM": "FUL",
    "FULHAM FC": "FUL",
    "IPSWICH": "IPS",
    "IPSWICH TOWN": "IPS",
    "LEEDS": "LEE",
    "LEEDS UNITED": "LEE",
    "LDS": "LEE",
    "LEICESTER": "LEI",
    "LEICESTER CITY": "LEI",
    "LIVERPOOL": "LIV",
    "LIVERPOOL FC": "LIV",
    "LUTON": "LUT",
    "LUTON TOWN": "LUT",
    "MAN CITY": "MCI",
    "MANCHESTER CITY": "MCI",
    "MCFC": "MCI",
    "MAN UTD": "MUN",
    "MAN UNITED": "MUN",
    "MANCHESTER UNITED": "MUN",
    "MUFC": "MUN",
    "NEWCASTLE": "NEW",
    "NEWCASTLE UNITED": "NEW",
    "NEWCASTLE UTD": "NEW",
    "NUFC": "NEW",
    "NOTT'M FOREST": "NFO",
    "NOTTM FOREST": "NFO",
    "NOTTINGHAM FOREST": "NFO",
    "NOTTS FOREST": "NFO",
    "FOREST": "NFO",
    "NOT": "NFO",
    "NORWICH": "NOR",
    "NORWICH CITY": "NOR",
    "SHEFFIELD UNITED": "SHU",
    "SHEFFIELD UTD": "SHU",
    "SHEFF UTD": "SHU",
    "SOUTHAMPTON": "SOU",
    "SOUTHAMPTON FC": "SOU",
    "SUNDERLAND": "SUN",
    "SUNDERLAND AFC": "SUN",
    "SPURS": "TOT",
    "TOTTENHAM": "TOT",
    "TOTTENHAM HOTSPUR": "TOT",
    "THFC": "TOT",
    "WATFORD": "WAT",
    "WATFORD FC": "WAT",
    "WEST BROM": "WBA",
    "WEST BROMWICH ALBION": "WBA",
    "WEST BROMWICH": "WBA",
    "WEST HAM": "WHU",
    "WEST HAM UNITED": "WHU",
    "WEST HAM UTD": "WHU",
    "WOLVES": "WOL",
    "WOLVERHAMPTON": "WOL",
    "WOLVERHAMPTON WANDERERS": "WOL",
}


def normalize_soccer_club(club: str) -> str:
    """Normalize FPL, Understat, or Football-Data club names/codes to canonical 3-letter code."""
    cleaned = " ".join(club.strip().upper().split())
    if cleaned in EPL_CLUBS:
        return cleaned
    return SOCCER_CLUB_ALIASES.get(cleaned, cleaned)


def get_stadium_metadata(club: str) -> StadiumMetadata | None:
    """Return stadium geographic and timezone metadata for an EPL club."""
    canonical = normalize_soccer_club(club)
    return EPL_CLUBS.get(canonical)


SILVER_MATCHES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "sport": pl.Utf8(),
    "season": pl.Int32(),
    "matchweek": pl.Int16(),
    "match_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "match_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "home_team": pl.Utf8(),
    "away_team": pl.Utf8(),
    "venue_timezone": pl.Utf8(),
    "game_state": pl.Utf8(),
    "home_goals": pl.Int16(),
    "away_goals": pl.Int16(),
    "outcome_1x2": pl.Int8(),
    "home_win": pl.Int8(),
    "is_draw": pl.Int8(),
    "away_win": pl.Int8(),
    "market_home_decimal_odds": pl.Float64(),
    "market_draw_decimal_odds": pl.Float64(),
    "market_away_decimal_odds": pl.Float64(),
    "market_prob_home": pl.Float64(),
    "market_prob_draw": pl.Float64(),
    "market_prob_away": pl.Float64(),
    "market_overround": pl.Float64(),
}

SILVER_TEAM_MATCH_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "match_id": pl.Int64(),
    "match_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "goals_for": pl.Int16(),
    "goals_against": pl.Int16(),
    "shots_for": pl.Int16(),
    "shots_against": pl.Int16(),
    "sot_for": pl.Int16(),
    "sot_against": pl.Int16(),
    "xgf": pl.Float64(),
    "xga": pl.Float64(),
    "npxgf": pl.Float64(),
    "npxga": pl.Float64(),
    "ppda": pl.Float64(),
    "opp_ppda": pl.Float64(),
    "deep_completions": pl.Int16(),
    "opp_deep_completions": pl.Int16(),
    "corners_for": pl.Int16(),
    "corners_against": pl.Int16(),
    "fouls_for": pl.Int16(),
    "fouls_against": pl.Int16(),
    "yellow_cards": pl.Int16(),
    "red_cards": pl.Int16(),
    "has_understat_xg": pl.Boolean(),
}

SILVER_OUTFIELD_MATCH_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "match_id": pl.Int64(),
    "match_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "player_id": pl.Int64(),
    "player_name": pl.Utf8(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "position_group": pl.Utf8(),
    "is_starter": pl.Boolean(),
    "minutes": pl.Float64(),
    "goals": pl.Int16(),
    "assists": pl.Int16(),
    "shots": pl.Int16(),
    "sot": pl.Int16(),
    "key_passes": pl.Int16(),
    "xg": pl.Float64(),
    "npxg": pl.Float64(),
    "xa": pl.Float64(),
    "xg_chain": pl.Float64(),
    "xg_buildup": pl.Float64(),
    "yellow_cards": pl.Int16(),
    "red_cards": pl.Int16(),
}

SILVER_GK_MATCH_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "match_id": pl.Int64(),
    "match_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "player_id": pl.Int64(),
    "player_name": pl.Utf8(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "is_starter": pl.Boolean(),
    "minutes": pl.Float64(),
    "shots_on_target_against": pl.Int16(),
    "saves": pl.Int16(),
    "goals_against": pl.Int16(),
    "clean_sheet": pl.Int8(),
    "xga_faced": pl.Float64(),
    "gsaa": pl.Float64(),
}
