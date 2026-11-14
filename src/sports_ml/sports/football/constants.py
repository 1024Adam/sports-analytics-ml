"""NFL 32-team registry, stadium database, Elo/prior parameters, and Silver schemas.

Registries are partitioned by `league` (`"NFL"` implemented; `"NCAAF"` reserved) so a
college plugin can be added later without restructuring. Team codes follow nflverse
tricodes (`LA` Rams, `LAC` Chargers, `LV` Raiders, `WAS`, `JAX`, `ARI`, `BAL`, `CLE`, `HOU`).
"""

from __future__ import annotations

import re
from typing import Final, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from sports_ml.core.ratings import EloConfig

FootballLeague = Literal["NFL", "NCAAF"]
Conference = Literal["AFC", "NFC"]
DivisionName = Literal["East", "North", "South", "West"]
RoofType = Literal["dome", "closed_retractable", "open_retractable", "outdoors"]
SurfaceType = Literal["grass", "turf"]

IMPLEMENTED_FOOTBALL_LEAGUES: Final[tuple[str, ...]] = ("NFL",)
RESERVED_FOOTBALL_LEAGUES: Final[tuple[str, ...]] = ("NCAAF",)
MODEL_SEASON_TYPES: Final[tuple[str, ...]] = ("REG", "WC", "DIV", "CON", "SB")
PLAYOFF_SEASON_TYPES: Final[frozenset[str]] = frozenset({"WC", "DIV", "CON", "SB"})
ROOF_TYPES: Final[tuple[str, ...]] = (
    "dome",
    "closed_retractable",
    "open_retractable",
    "outdoors",
)
SURFACE_TYPES: Final[tuple[str, ...]] = ("grass", "turf")
COVERED_ROOF_TYPES: Final[frozenset[str]] = frozenset({"dome", "closed_retractable"})
HOME_COUNTRY: Final[str] = "US"


class FootballTeamMetadata(BaseModel):
    """Canonical NFL franchise metadata."""

    model_config = ConfigDict(frozen=True)

    team: str
    team_name: str
    league: FootballLeague = "NFL"
    conference: Conference
    division: DivisionName
    primary_venue_id: str

    @property
    def division_label(self) -> str:
        """Return the division label (e.g., `AFC East`)."""
        return f"{self.conference} {self.division}"


class FootballStadiumMetadata(BaseModel):
    """Geographic, roof, surface, and altitude metadata for a football venue.

    `roof` is the venue's default configuration; observed game-day roof state is handled
    by the weather/availability features in Milestone 27.
    """

    model_config = ConfigDict(frozen=True)

    venue_id: str
    venue_name: str
    primary_teams: tuple[str, ...] = ()
    latitude: float
    longitude: float
    timezone: str
    country: str = "US"
    roof: RoofType = "outdoors"
    surface: SurfaceType = "grass"
    altitude_m: float = 0.0
    is_neutral_site: bool = False
    is_international: bool = False
    is_historical: bool = False
    super_bowl_seasons: tuple[int, ...] = ()

    @property
    def is_covered(self) -> bool:
        """Return True for fixed domes and retractable roofs that default to closed."""
        return self.roof in COVERED_ROOF_TYPES


class FranchiseEra(BaseModel):
    """Maps a historical team code to its continuous franchise (rating-history key)."""

    model_config = ConfigDict(frozen=True)

    code: str
    franchise: str
    first_season: int
    last_season: int | None = None


class FootballLeaguePriors(BaseModel):
    """League-average scoring/EPA seed priors for a season."""

    model_config = ConfigDict(frozen=True)

    points_per_team_game: float
    epa_per_play: float = 0.0


# ---------------------------------------------------------------------------
# 32-Team Registry (AFC/NFC x East/North/South/West)
# ---------------------------------------------------------------------------

_TEAM_ROWS: Final[tuple[tuple[str, str, Conference, DivisionName, str], ...]] = (
    ("BUF", "Buffalo Bills", "AFC", "East", "BUF_HIGHMARK"),
    ("MIA", "Miami Dolphins", "AFC", "East", "MIA_HARD_ROCK"),
    ("NE", "New England Patriots", "AFC", "East", "NE_GILLETTE"),
    ("NYJ", "New York Jets", "AFC", "East", "NYC_METLIFE"),
    ("BAL", "Baltimore Ravens", "AFC", "North", "BAL_M_AND_T"),
    ("CIN", "Cincinnati Bengals", "AFC", "North", "CIN_PAYCOR"),
    ("CLE", "Cleveland Browns", "AFC", "North", "CLE_HUNTINGTON"),
    ("PIT", "Pittsburgh Steelers", "AFC", "North", "PIT_ACRISURE"),
    ("HOU", "Houston Texans", "AFC", "South", "HOU_NRG"),
    ("IND", "Indianapolis Colts", "AFC", "South", "IND_LUCAS_OIL"),
    ("JAX", "Jacksonville Jaguars", "AFC", "South", "JAX_EVERBANK"),
    ("TEN", "Tennessee Titans", "AFC", "South", "TEN_NISSAN"),
    ("DEN", "Denver Broncos", "AFC", "West", "DEN_MILE_HIGH"),
    ("KC", "Kansas City Chiefs", "AFC", "West", "KC_ARROWHEAD"),
    ("LV", "Las Vegas Raiders", "AFC", "West", "LV_ALLEGIANT"),
    ("LAC", "Los Angeles Chargers", "AFC", "West", "LA_SOFI"),
    ("DAL", "Dallas Cowboys", "NFC", "East", "DAL_ATT"),
    ("NYG", "New York Giants", "NFC", "East", "NYC_METLIFE"),
    ("PHI", "Philadelphia Eagles", "NFC", "East", "PHI_LINCOLN"),
    ("WAS", "Washington Commanders", "NFC", "East", "WAS_NORTHWEST"),
    ("CHI", "Chicago Bears", "NFC", "North", "CHI_SOLDIER"),
    ("DET", "Detroit Lions", "NFC", "North", "DET_FORD"),
    ("GB", "Green Bay Packers", "NFC", "North", "GB_LAMBEAU"),
    ("MIN", "Minnesota Vikings", "NFC", "North", "MIN_US_BANK"),
    ("ATL", "Atlanta Falcons", "NFC", "South", "ATL_MERCEDES_BENZ"),
    ("CAR", "Carolina Panthers", "NFC", "South", "CAR_BOA"),
    ("NO", "New Orleans Saints", "NFC", "South", "NO_SUPERDOME"),
    ("TB", "Tampa Bay Buccaneers", "NFC", "South", "TB_RAYMOND_JAMES"),
    ("ARI", "Arizona Cardinals", "NFC", "West", "ARI_STATE_FARM"),
    ("LA", "Los Angeles Rams", "NFC", "West", "LA_SOFI"),
    ("SF", "San Francisco 49ers", "NFC", "West", "SF_LEVIS"),
    ("SEA", "Seattle Seahawks", "NFC", "West", "SEA_LUMEN"),
)

NFL_TEAMS_32: Final[dict[str, FootballTeamMetadata]] = {
    code: FootballTeamMetadata(
        team=code,
        team_name=name,
        conference=conf,
        division=div,
        primary_venue_id=venue,
    )
    for code, name, conf, div, venue in _TEAM_ROWS
}
ALL_FOOTBALL_TEAMS: Final[dict[str, FootballTeamMetadata]] = dict(NFL_TEAMS_32)

NFL_DIVISIONS: Final[dict[str, tuple[str, ...]]] = {
    label: tuple(sorted(t.team for t in NFL_TEAMS_32.values() if t.division_label == label))
    for label in sorted({t.division_label for t in NFL_TEAMS_32.values()})
}

# Historical team codes within the nflverse window (1999+) mapped to their continuous franchise.
FRANCHISE_ERAS: Final[tuple[FranchiseEra, ...]] = (
    FranchiseEra(code="OAK", franchise="LV", first_season=1999, last_season=2019),
    FranchiseEra(code="LV", franchise="LV", first_season=2020),
    FranchiseEra(code="SD", franchise="LAC", first_season=1999, last_season=2016),
    FranchiseEra(code="LAC", franchise="LAC", first_season=2017),
    FranchiseEra(code="STL", franchise="LA", first_season=1999, last_season=2015),
    FranchiseEra(code="LA", franchise="LA", first_season=2016),
)

_CODE_ALIASES: Final[dict[str, str]] = {
    "OAK": "LV",
    "LVR": "LV",
    "SD": "LAC",
    "SDG": "LAC",
    "STL": "LA",
    "SL": "LA",
    "LAR": "LA",
    "WSH": "WAS",
    "WFT": "WAS",
    "JAC": "JAX",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "GNB": "GB",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
}

_HISTORICAL_NAME_ALIASES: Final[dict[str, str]] = {
    "OAKLAND RAIDERS": "LV",
    "LOS ANGELES RAIDERS": "LV",
    "SAN DIEGO CHARGERS": "LAC",
    "ST. LOUIS RAMS": "LA",
    "ST LOUIS RAMS": "LA",
    "WASHINGTON REDSKINS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
    "WASHINGTON": "WAS",
}

FOOTBALL_TEAM_ALIASES: Final[dict[str, str]] = {
    **{t.team_name.upper(): t.team for t in NFL_TEAMS_32.values()},
    **_HISTORICAL_NAME_ALIASES,
    **_CODE_ALIASES,
}


def normalize_football_team(team: str) -> str:
    """Normalize any nflverse/PFR/historical code or team name to its canonical tricode.

    Relocated franchises resolve to their current code (`OAK` -> `LV`, `SD` -> `LAC`,
    `STL`/`LAR` -> `LA`), so rating history follows the franchise.
    """
    cleaned = " ".join(team.strip().split()).upper()
    if cleaned in NFL_TEAMS_32:
        return cleaned
    return FOOTBALL_TEAM_ALIASES.get(cleaned, cleaned)


def get_football_team_metadata(team: str) -> FootballTeamMetadata | None:
    """Return canonical team metadata or None for an unknown team."""
    return ALL_FOOTBALL_TEAMS.get(normalize_football_team(team))


def franchise_codes(team: str) -> tuple[str, ...]:
    """Return every historical code (oldest first) belonging to a team's franchise."""
    franchise = normalize_football_team(team)
    codes = [era.code for era in FRANCHISE_ERAS if era.franchise == franchise]
    return tuple(codes) if codes else (franchise,)


def resolve_team_league(team: str) -> FootballLeague | None:
    """Return the league (`NFL`) for a team code/name, or None if unknown."""
    meta = get_football_team_metadata(team)
    return meta.league if meta is not None else None


# ---------------------------------------------------------------------------
# Stadium Registry (30 active home venues + relocation-era venues + neutral/international)
# ---------------------------------------------------------------------------

_NY = "America/New_York"
_CHI = "America/Chicago"
_LA = "America/Los_Angeles"


def _venue(
    venue_id: str,
    name: str,
    lat: float,
    lon: float,
    tz: str,
    *,
    teams: tuple[str, ...] = (),
    roof: RoofType = "outdoors",
    surface: SurfaceType = "grass",
    alt: float = 0.0,
    country: str = "US",
    neutral: bool = False,
    historical: bool = False,
    super_bowls: tuple[int, ...] = (),
) -> FootballStadiumMetadata:
    return FootballStadiumMetadata(
        venue_id=venue_id,
        venue_name=name,
        primary_teams=teams,
        latitude=lat,
        longitude=lon,
        timezone=tz,
        country=country,
        roof=roof,
        surface=surface,
        altitude_m=alt,
        is_neutral_site=neutral,
        is_international=country != HOME_COUNTRY,
        is_historical=historical,
        super_bowl_seasons=super_bowls,
    )


_STADIUMS: Final[tuple[FootballStadiumMetadata, ...]] = (
    # --- Active NFL home venues ---
    _venue(
        "ARI_STATE_FARM",
        "State Farm Stadium",
        33.5276,
        -112.2626,
        "America/Phoenix",
        teams=("ARI",),
        roof="closed_retractable",
        alt=335.0,
        super_bowls=(2007, 2014, 2022),
    ),
    _venue(
        "ATL_MERCEDES_BENZ",
        "Mercedes-Benz Stadium",
        33.7554,
        -84.4010,
        _NY,
        teams=("ATL",),
        roof="closed_retractable",
        surface="turf",
        alt=320.0,
        super_bowls=(2018,),
    ),
    _venue("BAL_M_AND_T", "M&T Bank Stadium", 39.2780, -76.6227, _NY, teams=("BAL",), alt=10.0),
    _venue(
        "BUF_HIGHMARK",
        "Highmark Stadium",
        42.7738,
        -78.7870,
        _NY,
        teams=("BUF",),
        surface="turf",
        alt=200.0,
    ),
    _venue("CAR_BOA", "Bank of America Stadium", 35.2258, -80.8528, _NY, teams=("CAR",), alt=220.0),
    _venue("CHI_SOLDIER", "Soldier Field", 41.8623, -87.6167, _CHI, teams=("CHI",), alt=180.0),
    _venue(
        "CIN_PAYCOR",
        "Paycor Stadium",
        39.0955,
        -84.5161,
        _NY,
        teams=("CIN",),
        surface="turf",
        alt=150.0,
    ),
    _venue(
        "CLE_HUNTINGTON",
        "Huntington Bank Field",
        41.5061,
        -81.6995,
        _NY,
        teams=("CLE",),
        alt=180.0,
    ),
    _venue(
        "DAL_ATT",
        "AT&T Stadium",
        32.7473,
        -97.0945,
        _CHI,
        teams=("DAL",),
        roof="closed_retractable",
        surface="turf",
        alt=180.0,
        super_bowls=(2010,),
    ),
    _venue(
        "DEN_MILE_HIGH",
        "Empower Field at Mile High",
        39.7439,
        -105.0201,
        "America/Denver",
        teams=("DEN",),
        alt=1609.0,
    ),
    _venue(
        "DET_FORD",
        "Ford Field",
        42.3400,
        -83.0456,
        "America/Detroit",
        teams=("DET",),
        roof="dome",
        surface="turf",
        alt=180.0,
        super_bowls=(2005,),
    ),
    _venue("GB_LAMBEAU", "Lambeau Field", 44.5013, -88.0622, _CHI, teams=("GB",), alt=200.0),
    _venue(
        "HOU_NRG",
        "NRG Stadium",
        29.6847,
        -95.4107,
        _CHI,
        teams=("HOU",),
        roof="closed_retractable",
        surface="turf",
        alt=15.0,
        super_bowls=(2003, 2016),
    ),
    _venue(
        "IND_LUCAS_OIL",
        "Lucas Oil Stadium",
        39.7601,
        -86.1639,
        "America/Indiana/Indianapolis",
        teams=("IND",),
        roof="closed_retractable",
        surface="turf",
        alt=220.0,
        super_bowls=(2011,),
    ),
    _venue(
        "JAX_EVERBANK",
        "EverBank Stadium",
        30.3239,
        -81.6373,
        _NY,
        teams=("JAX",),
        alt=5.0,
        super_bowls=(2004,),
    ),
    _venue(
        "KC_ARROWHEAD",
        "GEHA Field at Arrowhead Stadium",
        39.0489,
        -94.4839,
        _CHI,
        teams=("KC",),
        alt=275.0,
    ),
    _venue(
        "LV_ALLEGIANT",
        "Allegiant Stadium",
        36.0909,
        -115.1833,
        _LA,
        teams=("LV",),
        roof="dome",
        alt=620.0,
        super_bowls=(2023,),
    ),
    _venue(
        "LA_SOFI",
        "SoFi Stadium",
        33.9535,
        -118.3392,
        _LA,
        teams=("LA", "LAC"),
        roof="dome",
        surface="turf",
        alt=30.0,
        super_bowls=(2021,),
    ),
    _venue(
        "MIA_HARD_ROCK",
        "Hard Rock Stadium",
        25.9580,
        -80.2389,
        _NY,
        teams=("MIA",),
        alt=3.0,
        super_bowls=(1998, 2006, 2009, 2019),
    ),
    _venue(
        "MIN_US_BANK",
        "U.S. Bank Stadium",
        44.9735,
        -93.2575,
        _CHI,
        teams=("MIN",),
        roof="dome",
        surface="turf",
        alt=255.0,
        super_bowls=(2017,),
    ),
    _venue(
        "NE_GILLETTE",
        "Gillette Stadium",
        42.0909,
        -71.2643,
        _NY,
        teams=("NE",),
        surface="turf",
        alt=90.0,
    ),
    _venue(
        "NO_SUPERDOME",
        "Caesars Superdome",
        29.9511,
        -90.0812,
        _CHI,
        teams=("NO",),
        roof="dome",
        surface="turf",
        alt=1.0,
        super_bowls=(2001, 2012, 2024),
    ),
    _venue(
        "NYC_METLIFE",
        "MetLife Stadium",
        40.8135,
        -74.0745,
        _NY,
        teams=("NYG", "NYJ"),
        surface="turf",
        alt=2.0,
        super_bowls=(2013,),
    ),
    _venue(
        "PHI_LINCOLN",
        "Lincoln Financial Field",
        39.9008,
        -75.1675,
        _NY,
        teams=("PHI",),
        alt=5.0,
    ),
    _venue("PIT_ACRISURE", "Acrisure Stadium", 40.4468, -80.0158, _NY, teams=("PIT",), alt=230.0),
    _venue(
        "SF_LEVIS",
        "Levi's Stadium",
        37.4033,
        -121.9694,
        _LA,
        teams=("SF",),
        alt=5.0,
        super_bowls=(2015, 2025),
    ),
    _venue(
        "SEA_LUMEN",
        "Lumen Field",
        47.5952,
        -122.3316,
        _LA,
        teams=("SEA",),
        surface="turf",
        alt=5.0,
    ),
    _venue(
        "TB_RAYMOND_JAMES",
        "Raymond James Stadium",
        27.9759,
        -82.5033,
        _NY,
        teams=("TB",),
        alt=10.0,
        super_bowls=(2000, 2008, 2020),
    ),
    _venue("TEN_NISSAN", "Nissan Stadium", 36.1665, -86.7713, _CHI, teams=("TEN",), alt=130.0),
    _venue("WAS_NORTHWEST", "Northwest Stadium", 38.9076, -76.8645, _NY, teams=("WAS",), alt=60.0),
    # --- Relocation-era and historical venues (1999+ backfills) ---
    _venue(
        "OAK_COLISEUM",
        "Oakland-Alameda County Coliseum",
        37.7516,
        -122.2005,
        _LA,
        teams=("LV",),
        alt=5.0,
        historical=True,
    ),
    _venue(
        "SD_QUALCOMM",
        "Qualcomm Stadium",
        32.7831,
        -117.1196,
        _LA,
        teams=("LAC",),
        alt=30.0,
        historical=True,
        super_bowls=(2002,),
    ),
    _venue(
        "STL_AMERICAS_CENTER",
        "Dome at America's Center",
        38.6328,
        -90.1885,
        _CHI,
        teams=("LA",),
        roof="dome",
        surface="turf",
        alt=140.0,
        historical=True,
    ),
    _venue(
        "LA_COLISEUM",
        "Los Angeles Memorial Coliseum",
        34.0141,
        -118.2879,
        _LA,
        teams=("LA",),
        alt=60.0,
        historical=True,
    ),
    _venue(
        "LAC_DIGNITY_HEALTH",
        "Dignity Health Sports Park",
        33.8644,
        -118.2611,
        _LA,
        teams=("LAC",),
        alt=15.0,
        historical=True,
    ),
    _venue(
        "ATL_GEORGIA_DOME",
        "Georgia Dome",
        33.7577,
        -84.4008,
        _NY,
        teams=("ATL",),
        roof="dome",
        surface="turf",
        alt=320.0,
        historical=True,
        super_bowls=(1999,),
    ),
    # --- Neutral-site international venues ---
    _venue(
        "LON_WEMBLEY",
        "Wembley Stadium",
        51.5560,
        -0.2795,
        "Europe/London",
        country="GB",
        alt=30.0,
        neutral=True,
    ),
    _venue(
        "LON_TOTTENHAM",
        "Tottenham Hotspur Stadium",
        51.6043,
        -0.0664,
        "Europe/London",
        country="GB",
        alt=30.0,
        neutral=True,
    ),
    _venue(
        "LON_TWICKENHAM",
        "Twickenham Stadium",
        51.4559,
        -0.3415,
        "Europe/London",
        country="GB",
        alt=15.0,
        neutral=True,
    ),
    _venue(
        "MUC_ALLIANZ",
        "Allianz Arena",
        48.2188,
        11.6247,
        "Europe/Berlin",
        country="DE",
        alt=520.0,
        neutral=True,
    ),
    _venue(
        "FRA_DEUTSCHE_BANK",
        "Deutsche Bank Park",
        50.0686,
        8.6455,
        "Europe/Berlin",
        country="DE",
        alt=110.0,
        neutral=True,
    ),
    _venue(
        "MEX_AZTECA",
        "Estadio Azteca",
        19.3029,
        -99.1505,
        "America/Mexico_City",
        country="MX",
        alt=2240.0,
        neutral=True,
    ),
    _venue(
        "SAO_CORINTHIANS",
        "Arena Corinthians",
        -23.5453,
        -46.4742,
        "America/Sao_Paulo",
        country="BR",
        alt=760.0,
        neutral=True,
    ),
    _venue(
        "MAD_BERNABEU",
        "Santiago Bernabeu Stadium",
        40.4530,
        -3.6883,
        "Europe/Madrid",
        country="ES",
        roof="open_retractable",
        alt=650.0,
        neutral=True,
    ),
    _venue(
        "DUB_CROKE_PARK",
        "Croke Park",
        53.3609,
        -6.2512,
        "Europe/Dublin",
        country="IE",
        alt=10.0,
        neutral=True,
    ),
    _venue(
        "TOR_ROGERS_CENTRE",
        "Rogers Centre",
        43.6414,
        -79.3894,
        "America/Toronto",
        country="CA",
        roof="closed_retractable",
        surface="turf",
        alt=76.0,
        neutral=True,
    ),
    _venue(
        "MEL_MCG",
        "Melbourne Cricket Ground",
        -37.8199,
        144.9834,
        "Australia/Melbourne",
        country="AU",
        alt=5.0,
        neutral=True,
    ),
    _venue(
        "PAR_STADE_DE_FRANCE",
        "Stade de France",
        48.9245,
        2.3602,
        "Europe/Paris",
        country="FR",
        alt=35.0,
        neutral=True,
    ),
    _venue(
        "RIO_MARACANA",
        "Maracana Stadium",
        -22.9122,
        -43.2302,
        "America/Sao_Paulo",
        country="BR",
        alt=10.0,
        neutral=True,
    ),
)

FOOTBALL_STADIUMS: Final[dict[str, FootballStadiumMetadata]] = {v.venue_id: v for v in _STADIUMS}


# nflverse `stadium_id` -> registry `venue_id` (unlisted legacy venues fall back to the host
# team's primary venue in the reconciler).
NFLVERSE_STADIUM_ID_TO_VENUE: Final[dict[str, str]] = {
    "ATL00": "ATL_GEORGIA_DOME",
    "ATL97": "ATL_MERCEDES_BENZ",
    "BAL00": "BAL_M_AND_T",
    "BOS00": "NE_GILLETTE",
    "BUF00": "BUF_HIGHMARK",
    "BUF01": "TOR_ROGERS_CENTRE",
    "CAR00": "CAR_BOA",
    "CHI98": "CHI_SOLDIER",
    "CIN00": "CIN_PAYCOR",
    "CLE00": "CLE_HUNTINGTON",
    "DAL00": "DAL_ATT",
    "DEN00": "DEN_MILE_HIGH",
    "DET00": "DET_FORD",
    "FRA00": "FRA_DEUTSCHE_BANK",
    "GER00": "MUC_ALLIANZ",
    "GNB00": "GB_LAMBEAU",
    "HOU00": "HOU_NRG",
    "IND00": "IND_LUCAS_OIL",
    "JAX00": "JAX_EVERBANK",
    "KAN00": "KC_ARROWHEAD",
    "LAX01": "LA_SOFI",
    "LAX97": "LAC_DIGNITY_HEALTH",
    "LAX99": "LA_COLISEUM",
    "LON00": "LON_WEMBLEY",
    "LON01": "LON_TWICKENHAM",
    "LON02": "LON_TOTTENHAM",
    "MAD01": "MAD_BERNABEU",
    "MEL00": "MEL_MCG",
    "MEX00": "MEX_AZTECA",
    "MIA00": "MIA_HARD_ROCK",
    "MIN01": "MIN_US_BANK",
    "MUN01": "MUC_ALLIANZ",
    "NAS00": "TEN_NISSAN",
    "NOR00": "NO_SUPERDOME",
    "NYC01": "NYC_METLIFE",
    "OAK00": "OAK_COLISEUM",
    "PAR00": "PAR_STADE_DE_FRANCE",
    "PHI00": "PHI_LINCOLN",
    "PHO00": "ARI_STATE_FARM",
    "PIT00": "PIT_ACRISURE",
    "RIO00": "RIO_MARACANA",
    "SAO00": "SAO_CORINTHIANS",
    "SDG00": "SD_QUALCOMM",
    "SEA00": "SEA_LUMEN",
    "SFO01": "SF_LEVIS",
    "STL00": "STL_AMERICAS_CENTER",
    "TAM00": "TB_RAYMOND_JAMES",
    "VEG00": "LV_ALLEGIANT",
    "WAS00": "WAS_NORTHWEST",
}

_TEAM_INDEX: Final[dict[str, int]] = {t: i + 1 for i, t in enumerate(sorted(NFL_TEAMS_32))}
_GAME_ID_PATTERN = re.compile(r"^(\d{4})_(\d{1,2})_([A-Z]{2,3})_([A-Z]{2,3})$")


def make_football_game_id(season: int, week: int, away_team: str, home_team: str) -> int:
    """Return the deterministic canonical `Int64` game id `SSSSWWAAHH` (season, week, away, home).

    Team codes are franchise-normalized first, so the id is stable across relocations and can
    be rebuilt from any nflverse table (schedule, play-by-play, snap counts) without a lookup.
    """
    away = _TEAM_INDEX.get(normalize_football_team(away_team))
    home = _TEAM_INDEX.get(normalize_football_team(home_team))
    if away is None or home is None:
        raise ValueError(f"Unknown team in game key: away={away_team!r}, home={home_team!r}.")
    if not (0 <= week <= 99):
        raise ValueError(f"Week must be in [0, 99], got {week}.")
    return int(season) * 1_000_000 + int(week) * 10_000 + away * 100 + home


def parse_nflverse_game_id(nflverse_game_id: str) -> tuple[int, int, str, str]:
    """Parse `'2023_01_BUF_NYJ'` into `(season, week, away_team, home_team)` (normalized)."""
    match = _GAME_ID_PATTERN.match(nflverse_game_id.strip().upper())
    if match is None:
        raise ValueError(f"Invalid nflverse game id: {nflverse_game_id!r}.")
    season, week, away, home = match.groups()
    return int(season), int(week), normalize_football_team(away), normalize_football_team(home)


def canonical_game_id_from_nflverse(nflverse_game_id: str) -> int:
    """Map an nflverse string game id to the canonical `Int64` game id."""
    season, week, away, home = parse_nflverse_game_id(nflverse_game_id)
    return make_football_game_id(season, week, away, home)


def get_stadium_metadata(venue_or_team: str) -> FootballStadiumMetadata | None:
    """Return stadium metadata by `venue_id`, falling back to a team's primary venue."""
    key = venue_or_team.strip().upper()
    if key in FOOTBALL_STADIUMS:
        return FOOTBALL_STADIUMS[key]
    team = get_football_team_metadata(key)
    if team is not None:
        return FOOTBALL_STADIUMS.get(team.primary_venue_id)
    return None


def is_neutral_game(season_type: str, venue_id: str | None) -> bool:
    """Return True for Super Bowls and neutral-site/international venues (HFA = 0)."""
    if season_type.upper() == "SB":
        return True
    if venue_id is None:
        return False
    venue = FOOTBALL_STADIUMS.get(venue_id.strip().upper())
    return venue is not None and venue.is_neutral_site


# ---------------------------------------------------------------------------
# Elo & Prior Parameters
# ---------------------------------------------------------------------------

FOOTBALL_ELO_CONFIG: Final[EloConfig] = EloConfig(
    base_rating=1500.0,
    home_field_advantage=48.0,
    k_factor=20.0,
    offseason_retention=2.0 / 3.0,  # regress ~1/3 of the way to the mean each offseason
    use_mov_multiplier=True,
    draw_param=0.0,  # ties are a 2-way 0.5 update, never a third class
    playoff_k_multiplier=1.2,
    mov_winner_perspective=True,
)

# Elo points per day of rest differential (capped) and venue-environment edges.
REST_ELO_PER_DAY: Final[float] = 3.0
MAX_REST_DIFF_DAYS: Final[float] = 7.0
ALTITUDE_THRESHOLD_M: Final[float] = 1000.0
ALTITUDE_ELO_PER_KM: Final[float] = 12.0
DOME_HOME_ELO_EDGE: Final[float] = 2.0

# Play-level filters for pre-game efficiency features (garbage-time WP band, early downs).
WP_BAND_MIN: Final[float] = 0.05
WP_BAND_MAX: Final[float] = 0.95
EARLY_DOWNS: Final[tuple[int, ...]] = (1, 2)
EXPLOSIVE_PASS_YARDS: Final[float] = 20.0
EXPLOSIVE_RUSH_YARDS: Final[float] = 10.0
RED_ZONE_YARDLINE_100: Final[float] = 20.0

# Cadence thresholds (days between kickoffs).
SHORT_WEEK_MAX_DAYS: Final[float] = 5.0
MINI_BYE_MIN_DAYS: Final[float] = 9.0
POST_BYE_MIN_DAYS: Final[float] = 13.0

# Approximate era seed priors (points per team-game); Milestone 24 replaces these with
# empirical prior-season league means once Silver data exists.
_LEAGUE_PRIOR_ERAS: Final[tuple[tuple[int, int, float], ...]] = (
    (1999, 2005, 21.4),
    (2006, 2010, 22.0),
    (2011, 2017, 22.6),
    (2018, 2021, 23.4),
    (2022, 9999, 22.0),
)


def get_league_priors(season: int, league: str = "NFL") -> FootballLeaguePriors:
    """Return seed league-average priors for `season` (passing-era drift by era table)."""
    if league.upper() not in IMPLEMENTED_FOOTBALL_LEAGUES:
        raise ValueError(f"Unsupported football league '{league}'.")
    for first, last, ppg in _LEAGUE_PRIOR_ERAS:
        if first <= season <= last:
            return FootballLeaguePriors(points_per_team_game=ppg)
    return FootballLeaguePriors(points_per_team_game=_LEAGUE_PRIOR_ERAS[0][2])


# ---------------------------------------------------------------------------
# Canonical Silver Parquet Schemas (`data/silver/football/`)
# ---------------------------------------------------------------------------
# Post-game nflverse schedule fields (`spread_line`, `total_line`, moneylines, `temp`, `wind`,
# post-game inactives) are never part of the feature-eligible columns: odds map to the
# evaluation-only `market_*` columns and weather is handled by Milestone 27.

SILVER_FOOTBALL_GAMES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "sport": pl.Utf8(),
    "league": pl.Utf8(),
    "season": pl.Int32(),
    "season_type": pl.Utf8(),
    "week": pl.Int16(),
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "home_team": pl.Utf8(),
    "away_team": pl.Utf8(),
    "venue_id": pl.Utf8(),
    "venue_timezone": pl.Utf8(),
    "is_neutral_site": pl.Boolean(),
    "is_international": pl.Boolean(),
    "game_state": pl.Utf8(),
    "home_qb_id": pl.Int64(),
    "away_qb_id": pl.Int64(),
    "overtime": pl.Boolean(),
    "home_score": pl.Int16(),
    "away_score": pl.Int16(),
    "home_win": pl.Int8(),
    "is_tie": pl.Int8(),
    "market_home_decimal_odds": pl.Float64(),
    "market_away_decimal_odds": pl.Float64(),
    "market_prob_home": pl.Float64(),
    "market_prob_away": pl.Float64(),
    "market_spread_home": pl.Float64(),
    "market_total_points": pl.Float64(),
    "obs_weather_temp": pl.Float64(),
    "obs_weather_wind": pl.Float64(),
    "obs_weather_roof": pl.Utf8(),
    "obs_weather_surface": pl.Utf8(),
    "nflverse_game_id": pl.Utf8(),
}

SILVER_FOOTBALL_TEAM_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "season_type": pl.Utf8(),
    "week": pl.Int16(),
    "league": pl.Utf8(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "venue_id": pl.Utf8(),
    "overtime": pl.Boolean(),
    "points_for": pl.Int16(),
    "points_against": pl.Int16(),
    "off_plays": pl.Int16(),
    "off_epa_total": pl.Float64(),
    "off_dropbacks": pl.Int16(),
    "off_rushes": pl.Int16(),
    "off_pass_epa_total": pl.Float64(),
    "off_rush_epa_total": pl.Float64(),
    "off_success_plays": pl.Int16(),
    "off_yards": pl.Float64(),
    "off_turnovers": pl.Int8(),
    "off_sacks_taken": pl.Int8(),
    "off_fumbles": pl.Int8(),
    "off_fumbles_lost": pl.Int8(),
    "off_rz_drives": pl.Int8(),
    "off_rz_td_drives": pl.Int8(),
    "f_off_plays": pl.Int16(),
    "f_off_epa": pl.Float64(),
    "f_off_dropbacks": pl.Int16(),
    "f_off_pass_epa": pl.Float64(),
    "f_off_rushes": pl.Int16(),
    "f_off_rush_epa": pl.Float64(),
    "f_off_success": pl.Int16(),
    "f_off_early_plays": pl.Int16(),
    "f_off_early_success": pl.Int16(),
    "f_off_explosive": pl.Int16(),
    "f_off_drives": pl.Int8(),
    "f_off_drive_points": pl.Float64(),
    "def_plays": pl.Int16(),
    "def_epa_total": pl.Float64(),
    "def_success_plays": pl.Int16(),
    "def_yards": pl.Float64(),
    "def_takeaways": pl.Int8(),
    "def_sacks": pl.Int8(),
    "penalties": pl.Int8(),
    "penalty_yards": pl.Int16(),
    "time_of_possession_s": pl.Int32(),
}

SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "season_type": pl.Utf8(),
    "week": pl.Int16(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "player_id": pl.Int64(),
    "player_name": pl.Utf8(),
    "position": pl.Utf8(),
    "is_starter": pl.Boolean(),
    "snaps": pl.Int16(),
    "snap_share": pl.Float64(),
    "pass_attempts": pl.Int16(),
    "completions": pl.Int16(),
    "pass_air_yards": pl.Float64(),
    "pass_yards": pl.Float64(),
    "pass_tds": pl.Int8(),
    "interceptions": pl.Int8(),
    "carries": pl.Int16(),
    "rush_yards": pl.Float64(),
    "rush_tds": pl.Int8(),
    "targets": pl.Int16(),
    "receptions": pl.Int16(),
    "rec_yards": pl.Float64(),
    "rec_tds": pl.Int8(),
    "rec_air_yards": pl.Float64(),
    "rz_targets": pl.Int16(),
    "rz_carries": pl.Int16(),
    "gl_carries": pl.Int16(),
}

SILVER_FOOTBALL_QB_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "season_type": pl.Utf8(),
    "week": pl.Int16(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
    "qb_id": pl.Int64(),
    "is_starter": pl.Boolean(),
    "dropbacks": pl.Int16(),
    "pass_attempts": pl.Int16(),
    "qb_epa_total": pl.Float64(),
    "cpoe_total": pl.Float64(),
    "cpoe_plays": pl.Int16(),
    "sacks": pl.Int8(),
    "air_yards_total": pl.Float64(),
    "scrambles": pl.Int8(),
    "rush_attempts": pl.Int8(),
    "rush_epa_total": pl.Float64(),
}

# As-of availability snapshots (injury report, depth chart, official inactives). A row is
# usable for a prediction only when `as_of_time_utc` is at or before the prediction cutoff.
SILVER_FOOTBALL_AVAILABILITY_SCHEMA: Final[dict[str, pl.DataType]] = {
    "as_of_time_utc": pl.Datetime("us", "UTC"),
    "game_id": pl.Int64(),
    "season": pl.Int32(),
    "week": pl.Int16(),
    "team": pl.Utf8(),
    "player_id": pl.Int64(),
    "position": pl.Utf8(),
    "source": pl.Utf8(),
    "report_status": pl.Utf8(),
    "practice_status": pl.Utf8(),
    "depth_rank": pl.Int8(),
    "is_inactive": pl.Boolean(),
}

SILVER_FOOTBALL_SNAP_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "week": pl.Int16(),
    "team": pl.Utf8(),
    "player_id": pl.Int64(),
    "position": pl.Utf8(),
    "offense_snaps": pl.Int16(),
    "offense_pct": pl.Float64(),
    "defense_snaps": pl.Int16(),
    "defense_pct": pl.Float64(),
}

# ---------------------------------------------------------------------------
# Milestone 28: play-by-play micro-feature Silver tables
# ---------------------------------------------------------------------------
TEAM_MICRO_KEY_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "week": pl.Int16(),
    "team": pl.Utf8(),
    "opponent": pl.Utf8(),
}
# Offense-side numerators/denominators (defense is the opponent's row). `epa_total` / `plays`
# reconcile exactly to `team_game_logs.off_epa_total` / `off_plays`.
TEAM_MICRO_PBP_COLUMNS: Final[tuple[str, ...]] = (
    "epa_total",
    "plays",
    "neutral_plays",
    "neutral_epa",
    "neutral_epa_plays",
    "neutral_seconds",
    "neutral_seconds_plays",
    "proe_num",
    "proe_den",
    "fourth_go_num",
    "fourth_go_den",
    "two_min_plays",
    "two_min_epa",
    "dropbacks",
    "cpoe_sum",
    "cpoe_plays",
    "air_yards_sum",
    "air_yards_plays",
    "adot_sum",
    "adot_plays",
    "yac_oe_sum",
    "yac_oe_plays",
    "pressure_num",
    "sacks",
    "rz_plays",
    "rz_epa",
    "third_down_plays",
    "third_down_epa",
    "explosive_pass",
    "pass_plays",
    "run_stuffs",
    "rush_plays",
    "nopen_epa",
    "nopen_plays",
)
TEAM_MICRO_NGS_COLUMNS: Final[tuple[str, ...]] = (
    "ngs_pass_attempts",
    "ngs_time_to_throw_sum",
    "ngs_cpoe_sum",
    "ngs_rush_attempts",
    "ngs_ryoe_sum",
    "ngs_targets",
    "ngs_separation_sum",
    "ngs_receptions",
    "ngs_yac_oe_sum",
)
SILVER_FOOTBALL_TEAM_MICRO_SCHEMA: Final[dict[str, pl.DataType]] = {
    **TEAM_MICRO_KEY_SCHEMA,
    **{c: pl.Float64() for c in (*TEAM_MICRO_PBP_COLUMNS, *TEAM_MICRO_NGS_COLUMNS)},
    "has_ngs": pl.Int8(),
}
PLAYER_MICRO_PBP_COLUMNS: Final[tuple[str, ...]] = (
    "targets",
    "target_air_yards",
    "receptions",
    "rec_yards",
    "yac_oe_sum",
    "yac_oe_plays",
    "rush_attempts",
    "rush_yards",
)
PLAYER_MICRO_NGS_COLUMNS: Final[tuple[str, ...]] = (
    "ngs_rush_attempts",
    "ngs_ryoe_sum",
    "ngs_targets",
    "ngs_separation_sum",
)
SILVER_FOOTBALL_PLAYER_MICRO_SCHEMA: Final[dict[str, pl.DataType]] = {
    **TEAM_MICRO_KEY_SCHEMA,
    "player_id": pl.Int64(),
    **{c: pl.Float64() for c in (*PLAYER_MICRO_PBP_COLUMNS, *PLAYER_MICRO_NGS_COLUMNS)},
    "has_ngs": pl.Int8(),
}
NGS_FIRST_SEASON: Final[int] = 2016
# Receiving yards per (estimated) route run priors by position; routes = snaps x team dropback rate.
YPRR_PRIORS: Final[dict[str, float]] = {"WR": 1.50, "TE": 1.20, "RB": 0.80}

GAMES_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "season_type",
    "week",
    "start_time_utc",
    "home_team",
    "away_team",
)
TEAM_LOGS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "season_type",
    "week",
    "start_time_utc",
    "team",
    "opponent",
    "is_home",
)


# ---------------------------------------------------------------------------
# Stage 2 player-model priors (approximate NFL replacement/league levels per position)
# ---------------------------------------------------------------------------
# Position priors for shrunk, per-opportunity rates and role shares. Fixed constants (not
# estimated from the evaluation data) so cold-start rows never depend on future games.
POSITION_PRIORS: Final[dict[str, dict[str, float]]] = {
    "QB": {
        "pass_att_share": 0.95,
        "comp_rate": 0.645,
        "yds_per_att": 7.0,
        "td_per_att": 0.044,
        "int_per_att": 0.025,
        "rush_att_pg": 3.0,
        "rush_ypc": 5.0,
        "rush_td_per_carry": 0.035,
        "snap_share": 0.98,
    },
    "RB": {
        "carry_share": 0.30,
        "gl_carry_share": 0.30,
        "rz_carry_share": 0.30,
        "target_share": 0.08,
        "rz_target_share": 0.07,
        "air_yards_share": 0.02,
        "rush_ypc": 4.2,
        "rush_td_per_carry": 0.04,
        "catch_rate": 0.77,
        "rec_yds_per_target": 5.9,
        "rec_td_per_target": 0.025,
        "snap_share": 0.45,
    },
    "WR": {
        "target_share": 0.14,
        "rz_target_share": 0.12,
        "air_yards_share": 0.15,
        "catch_rate": 0.64,
        "rec_yds_per_target": 8.2,
        "rec_td_per_target": 0.050,
        "snap_share": 0.75,
    },
    "TE": {
        "target_share": 0.09,
        "rz_target_share": 0.10,
        "air_yards_share": 0.08,
        "catch_rate": 0.68,
        "rec_yds_per_target": 7.4,
        "rec_td_per_target": 0.060,
        "snap_share": 0.65,
    },
}
# Draft-capital bonus (pick 1 = full bonus) added to the share priors of drafted players.
DRAFT_SHARE_BONUS: Final[dict[str, dict[str, float]]] = {
    "QB": {"pass_att_share": 0.03},
    "RB": {"carry_share": 0.20, "target_share": 0.04},
    "WR": {"target_share": 0.10},
    "TE": {"target_share": 0.05},
}
STAGE2_SHRINK_K: Final[dict[str, float]] = {
    "share": 60.0,
    "yards": 40.0,
    "td": 120.0,
    "qb_att": 200.0,
    "qb_td": 250.0,
    "defense_games": 6.0,
    "team_games": 6.0,
}
