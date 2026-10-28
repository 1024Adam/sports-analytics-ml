"""Non-league & European club fixture ingestion for EPL clubs (`non_league_fixtures.py`).

Ingests historical and active-season non-EPL club fixtures (`competition_type` in
`{"UCL", "UEL", "UECL", "FA_CUP", "EFL_CUP"}`) alongside domestic `"EPL"` fixtures into
`data/silver/soccer/club_all_competitions_schedule.parquet` (`season`, `match_date`,
`start_time_utc`, `team`, `opponent`, `competition_type`, `is_home`, `venue_lat`,
`venue_lon`, `extra_time_played`, `squad_minutes_load`).

These fixtures are used strictly to compute true multi-competition schedule rest,
European travel distance, 14/28-day match congestion, and squad rotation/minutes load
without polluting domestic EPL tactical xG rates or Elo ratings.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Final

import polars as pl

from sports_ml.core.cache import list_cached_files, read_cached_bytes, read_cached_json
from sports_ml.sports.soccer.constants import (
    EPL_CLUBS,
    SILVER_CLUB_ALL_COMPETITIONS_SCHEDULE_SCHEMA,
    get_stadium_metadata,
    normalize_soccer_club,
)

EUROPEAN_COMPETITION_TYPES: Final[frozenset[str]] = frozenset({"UCL", "UEL", "UECL"})
CUP_COMPETITION_TYPES: Final[frozenset[str]] = frozenset(
    {"UCL", "UEL", "UECL", "FA_CUP", "EFL_CUP"}
)
VALID_COMPETITION_TYPES: Final[frozenset[str]] = frozenset(
    {"EPL", "UCL", "UEL", "UECL", "FA_CUP", "EFL_CUP"}
)

DEFAULT_REGULATION_SQUAD_MINUTES: Final[float] = 990.0
DEFAULT_EXTRA_TIME_SQUAD_MINUTES: Final[float] = 1320.0

# Representative European away stadium coordinates for common UEFA opponents
EUROPEAN_OPPONENT_COORDS: Final[dict[str, tuple[float, float]]] = {
    "REAL MADRID": (40.4531, -3.6883),
    "RMA": (40.4531, -3.6883),
    "BARCELONA": (41.3809, 2.1228),
    "BAR": (41.3809, 2.1228),
    "ATLETICO MADRID": (40.4362, -3.5995),
    "ATM": (40.4362, -3.5995),
    "BAYERN MUNICH": (48.2188, 11.6247),
    "BAY": (48.2188, 11.6247),
    "BORUSSIA DORTMUND": (51.4926, 7.4519),
    "BVB": (51.4926, 7.4519),
    "BAYER LEVERKUSEN": (51.0383, 7.0022),
    "B04": (51.0383, 7.0022),
    "RB LEIPZIG": (51.3458, 12.3483),
    "RBL": (51.3458, 12.3483),
    "PARIS SAINT-GERMAIN": (48.8414, 2.2530),
    "PSG": (48.8414, 2.2530),
    "INTER": (45.4781, 9.1240),
    "INTER MILAN": (45.4781, 9.1240),
    "INT": (45.4781, 9.1240),
    "AC MILAN": (45.4781, 9.1240),
    "MIL": (45.4781, 9.1240),
    "JUVENTUS": (45.1096, 7.6413),
    "JUV": (45.1096, 7.6413),
    "ATALANTA": (45.7089, 9.6808),
    "NAPOLI": (40.8280, 14.1931),
    "ROMA": (41.9341, 12.4547),
    "BENFICA": (38.7527, -9.1847),
    "PORTO": (41.1618, -8.5839),
    "SPORTING CP": (38.7612, -9.1608),
    "AJAX": (52.3144, 4.9419),
    "PSV": (51.4417, 5.4675),
    "FEYENOORD": (51.8939, 4.5231),
    "CELTIC": (55.8497, -4.2055),
    "RANGERS": (55.8532, -4.3093),
    "GALATASARAY": (41.1034, 28.9910),
    "FENERBAHCE": (40.9876, 29.0369),
}

DEFAULT_EUROPEAN_AWAY_COORDS: Final[tuple[float, float]] = (48.8566, 9.1829)
DEFAULT_UK_CUP_AWAY_COORDS: Final[tuple[float, float]] = (52.4862, -1.8904)


def normalize_competition_type(raw_comp: str | None) -> str:
    """Normalize raw competition strings into canonical codes (`EPL`, `UCL`, `UEL`, `UECL`)."""
    if not raw_comp:
        return "EPL"
    cleaned = raw_comp.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "PREMIER_LEAGUE": "EPL",
        "E0": "EPL",
        "CHAMPIONS_LEAGUE": "UCL",
        "UEFA_CHAMPIONS_LEAGUE": "UCL",
        "EUROPA_LEAGUE": "UEL",
        "UEFA_EUROPA_LEAGUE": "UEL",
        "CONFERENCE_LEAGUE": "UECL",
        "UEFA_CONFERENCE_LEAGUE": "UECL",
        "UEFA_EUROPA_CONFERENCE_LEAGUE": "UECL",
        "FACUP": "FA_CUP",
        "FA": "FA_CUP",
        "EFLCUP": "EFL_CUP",
        "CARABAO_CUP": "EFL_CUP",
        "LEAGUE_CUP": "EFL_CUP",
    }
    mapped = aliases.get(cleaned, cleaned)
    return mapped if mapped in VALID_COMPETITION_TYPES else "EPL"


def resolve_fixture_venue_coords(
    team: str,
    opponent: str,
    competition_type: str,
    is_home: bool,
    explicit_lat: float | None = None,
    explicit_lon: float | None = None,
) -> tuple[float, float]:
    """Resolve stadium coordinates (`venue_lat`, `venue_lon`) for domestic or European fixtures."""
    if explicit_lat is not None and explicit_lon is not None:
        return float(explicit_lat), float(explicit_lon)

    venue_club = normalize_soccer_club(team if is_home else opponent)
    meta = get_stadium_metadata(venue_club)
    if meta is not None:
        return meta.latitude, meta.longitude

    opp_upper = opponent.strip().upper()
    if opp_upper in EUROPEAN_OPPONENT_COORDS:
        return EUROPEAN_OPPONENT_COORDS[opp_upper]

    if competition_type in EUROPEAN_COMPETITION_TYPES and not is_home:
        return DEFAULT_EUROPEAN_AWAY_COORDS

    team_meta = get_stadium_metadata(team)
    if is_home and team_meta is not None:
        return team_meta.latitude, team_meta.longitude
    return DEFAULT_UK_CUP_AWAY_COORDS


def _parse_fixture_datetime(
    raw_dt: Any,
    raw_date: Any = None,
    raw_time: Any = None,
) -> tuple[date, datetime] | None:
    if isinstance(raw_dt, datetime):
        dt_utc = raw_dt.astimezone(UTC) if raw_dt.tzinfo else raw_dt.replace(tzinfo=UTC)
        return dt_utc.date(), dt_utc
    if raw_dt:
        cleaned_dt = str(raw_dt).strip()
        try:
            dt_utc = datetime.fromisoformat(cleaned_dt.replace("Z", "+00:00")).astimezone(UTC)
            return dt_utc.date(), dt_utc
        except ValueError:
            pass

    d_str = str(raw_date or raw_dt or "").strip()
    if not d_str:
        return None
    parsed_date: date | None = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            parsed_date = datetime.strptime(d_str[:10], fmt).date()
            break
        except ValueError:
            continue
    if parsed_date is None:
        return None

    parsed_time = dt_time(19, 45)
    if raw_time:
        t_str = str(raw_time).strip()
        for tfmt in ("%H:%M", "%H:%M:%S"):
            try:
                parsed_time = datetime.strptime(t_str, tfmt).time()
                break
            except ValueError:
                continue
    return parsed_date, datetime.combine(parsed_date, parsed_time, tzinfo=UTC)


def parse_non_league_fixtures_records(
    records: Sequence[Mapping[str, Any]],
    *,
    default_season: int | None = None,
) -> pl.DataFrame:
    """Parse structured non-league or all-competition fixture dicts into canonical Silver schema."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        dt_pair = _parse_fixture_datetime(
            rec.get("start_time_utc") or rec.get("kickoff_time") or rec.get("datetime"),
            raw_date=rec.get("match_date") or rec.get("date"),
            raw_time=rec.get("time"),
        )
        if dt_pair is None:
            continue
        m_date, dt_utc = dt_pair

        raw_season = rec.get("season") or default_season
        if raw_season is not None:
            season = int(raw_season)
        else:
            start_yr = m_date.year if m_date.month >= 7 else m_date.year - 1
            season = int(f"{start_yr}{start_yr + 1}")

        comp_type = normalize_competition_type(
            str(rec.get("competition_type") or rec.get("competition") or "UCL")
        )

        # Support both team-centric records (`team`, `opponent`, `is_home`) and
        # match-centric records (`home_team`, `away_team`)
        if "team" in rec and rec.get("team"):
            team_code = normalize_soccer_club(str(rec["team"]))
            opp_code = normalize_soccer_club(str(rec.get("opponent") or "OPP"))
            raw_is_home = rec.get("is_home")
            if isinstance(raw_is_home, str):
                is_home = raw_is_home.strip().lower() in {"true", "1", "h", "home", "yes"}
            else:
                is_home = bool(raw_is_home) if raw_is_home is not None else True

            raw_et = rec.get("extra_time_played")
            if isinstance(raw_et, str):
                extra_time = raw_et.strip().lower() in {"true", "1", "yes", "et", "aet"}
            else:
                extra_time = bool(raw_et) if raw_et is not None else False

            raw_load = rec.get("squad_minutes_load")
            squad_load = (
                float(raw_load)
                if raw_load is not None and str(raw_load).strip() != ""
                else (
                    DEFAULT_EXTRA_TIME_SQUAD_MINUTES
                    if extra_time
                    else DEFAULT_REGULATION_SQUAD_MINUTES
                )
            )
            raw_lat = rec.get("venue_lat")
            raw_lon = rec.get("venue_lon")
            lat_val = float(raw_lat) if raw_lat is not None and str(raw_lat).strip() != "" else None
            lon_val = float(raw_lon) if raw_lon is not None and str(raw_lon).strip() != "" else None
            v_lat, v_lon = resolve_fixture_venue_coords(
                team_code, opp_code, comp_type, is_home, lat_val, lon_val
            )
            rows.append(
                {
                    "season": season,
                    "match_date": m_date,
                    "start_time_utc": dt_utc,
                    "team": team_code,
                    "opponent": opp_code,
                    "competition_type": comp_type,
                    "is_home": is_home,
                    "venue_lat": v_lat,
                    "venue_lon": v_lon,
                    "extra_time_played": extra_time,
                    "squad_minutes_load": squad_load,
                }
            )
        elif "home_team" in rec and "away_team" in rec:
            h_code = normalize_soccer_club(str(rec["home_team"]))
            a_code = normalize_soccer_club(str(rec["away_team"]))
            raw_et = rec.get("extra_time_played")
            extra_time = bool(raw_et) if raw_et is not None else False
            squad_load = float(
                rec.get("squad_minutes_load")
                or (
                    DEFAULT_EXTRA_TIME_SQUAD_MINUTES
                    if extra_time
                    else DEFAULT_REGULATION_SQUAD_MINUTES
                )
            )
            raw_lat = rec.get("venue_lat")
            raw_lon = rec.get("venue_lon")
            lat_val = float(raw_lat) if raw_lat is not None else None
            lon_val = float(raw_lon) if raw_lon is not None else None

            for side_team, side_opp, side_home in ((h_code, a_code, True), (a_code, h_code, False)):
                if side_team in EPL_CLUBS or comp_type == "EPL":
                    v_lat, v_lon = resolve_fixture_venue_coords(
                        side_team, side_opp, comp_type, side_home, lat_val, lon_val
                    )
                    rows.append(
                        {
                            "season": season,
                            "match_date": m_date,
                            "start_time_utc": dt_utc,
                            "team": side_team,
                            "opponent": side_opp,
                            "competition_type": comp_type,
                            "is_home": side_home,
                            "venue_lat": v_lat,
                            "venue_lon": v_lon,
                            "extra_time_played": extra_time,
                            "squad_minutes_load": squad_load,
                        }
                    )

    if not rows:
        return pl.DataFrame(schema=SILVER_CLUB_ALL_COMPETITIONS_SCHEDULE_SCHEMA)
    return (
        pl.DataFrame(rows, schema=SILVER_CLUB_ALL_COMPETITIONS_SCHEDULE_SCHEMA)
        .unique(subset=["start_time_utc", "team", "competition_type"], keep="last")
        .sort(["match_date", "start_time_utc", "team"])
    )


def parse_non_league_fixtures_csv(
    csv_text: str,
    *,
    default_season: int | None = None,
) -> pl.DataFrame:
    """Parse a CSV string of non-league club fixtures into canonical Silver schedule schema."""
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    records = [{str(k).strip(): v for k, v in row.items() if k is not None} for row in reader]
    return parse_non_league_fixtures_records(records, default_season=default_season)


def build_epl_schedule_rows_from_matches(matches_df: pl.DataFrame) -> pl.DataFrame:
    """Convert EPL `matches_df` (2 sides per match) into canonical all-competition schedule rows."""
    if matches_df.is_empty():
        return pl.DataFrame(schema=SILVER_CLUB_ALL_COMPETITIONS_SCHEDULE_SCHEMA)

    rows: list[dict[str, Any]] = []
    for r in matches_df.iter_rows(named=True):
        season = int(r["season"])
        dt_utc: datetime = r["start_time_utc"]
        m_date: date = r.get("match_date") or dt_utc.date()
        h_team = normalize_soccer_club(str(r["home_team"]))
        a_team = normalize_soccer_club(str(r["away_team"]))
        v_lat, v_lon = resolve_fixture_venue_coords(h_team, a_team, "EPL", True)

        rows.append(
            {
                "season": season,
                "match_date": m_date,
                "start_time_utc": dt_utc,
                "team": h_team,
                "opponent": a_team,
                "competition_type": "EPL",
                "is_home": True,
                "venue_lat": v_lat,
                "venue_lon": v_lon,
                "extra_time_played": False,
                "squad_minutes_load": DEFAULT_REGULATION_SQUAD_MINUTES,
            }
        )
        rows.append(
            {
                "season": season,
                "match_date": m_date,
                "start_time_utc": dt_utc,
                "team": a_team,
                "opponent": h_team,
                "competition_type": "EPL",
                "is_home": False,
                "venue_lat": v_lat,
                "venue_lon": v_lon,
                "extra_time_played": False,
                "squad_minutes_load": DEFAULT_REGULATION_SQUAD_MINUTES,
            }
        )

    return (
        pl.DataFrame(rows, schema=SILVER_CLUB_ALL_COMPETITIONS_SCHEDULE_SCHEMA)
        .unique(subset=["start_time_utc", "team", "competition_type"], keep="last")
        .sort(["match_date", "start_time_utc", "team"])
    )


class NonLeagueFixturesIngestor:
    """Ingests cached CSV/JSON non-EPL club fixtures (`UCL`, `UEL`, `UECL`, `FA_CUP`, `EFL_CUP`)."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_cached_non_league_fixtures(self) -> pl.DataFrame:
        """Load all cached `.csv`/`.csv.gz` and `.json`/`.json.gz` non-league fixture files."""
        frames: list[pl.DataFrame] = []
        if not self.cache_dir.exists():
            return pl.DataFrame(schema=SILVER_CLUB_ALL_COMPETITIONS_SCHEDULE_SCHEMA)

        for csv_path in list_cached_files(self.cache_dir, "*.csv"):
            raw_text = read_cached_bytes(csv_path).decode("utf-8", errors="replace")
            df = parse_non_league_fixtures_csv(raw_text)
            if not df.is_empty():
                frames.append(df)

        for json_path in list_cached_files(self.cache_dir, "*.json"):
            payload = read_cached_json(json_path)
            if isinstance(payload, list):
                df = parse_non_league_fixtures_records(payload)
                if not df.is_empty():
                    frames.append(df)
            elif isinstance(payload, Mapping) and isinstance(payload.get("fixtures"), list):
                df = parse_non_league_fixtures_records(payload["fixtures"])
                if not df.is_empty():
                    frames.append(df)

        if not frames:
            return pl.DataFrame(schema=SILVER_CLUB_ALL_COMPETITIONS_SCHEDULE_SCHEMA)
        return (
            pl.concat(frames, how="vertical")
            .unique(subset=["start_time_utc", "team", "competition_type"], keep="last")
            .sort(["match_date", "start_time_utc", "team"])
        )
