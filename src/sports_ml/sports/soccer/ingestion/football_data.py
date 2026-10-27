"""Football-Data.co.uk historical & active-season EPL CSV downloader/parser (`football_data.py`).

Parses `E0.csv` match results and team boxscore statistics (`FTHG`, `FTAG`, `FTR`,
`HS`, `AS`, `HST`, `AST`, `HC`, `AC`, `HF`, `AF`, `HY`, `AY`, `HR`, `AR`).
"""

from __future__ import annotations

import csv
import io
import time
from datetime import UTC, date, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sports_ml.core.cache import read_cached_bytes, resolve_cache_read_path, write_gzip_bytes
from sports_ml.sports.soccer.constants import SILVER_MATCHES_SCHEMA, normalize_soccer_club
from sports_ml.sports.soccer.ingestion.fpl_api import make_canonical_match_id

FOOTBALL_DATA_BASE_URL: Final[str] = "https://www.football-data.co.uk/mmz4281"
DEFAULT_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)


def season_to_football_data_code(season: int) -> str:
    """Convert 8-digit season (`20242025`) to 4-digit Football-Data code (`'2425'`)."""
    s = str(season)
    if len(s) == 8:
        return f"{s[2:4]}{s[6:8]}"
    start_yr = int(s[:4])
    end_yr = start_yr + 1
    return f"{str(start_yr)[2:]}{str(end_yr)[2:]}"


def _parse_uk_date(raw_date: str) -> date | None:
    cleaned = raw_date.strip()
    if not cleaned:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _parse_uk_time(raw_time: str | None) -> dt_time:
    if not raw_time or not raw_time.strip():
        return dt_time(15, 0)
    cleaned = raw_time.strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt).time()
        except ValueError:
            continue
    return dt_time(15, 0)


def parse_football_data_csv(
    csv_text: str,
    *,
    season: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse Football-Data `E0.csv` text into `(matches_df, team_boxscore_df)`."""
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    match_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []

    idx = 0
    for row in reader:
        raw_home = str(row.get("HomeTeam") or "").strip()
        raw_away = str(row.get("AwayTeam") or "").strip()
        m_date = _parse_uk_date(str(row.get("Date") or ""))
        if not raw_home or not raw_away or m_date is None:
            continue

        idx += 1
        m_time = _parse_uk_time(row.get("Time"))
        dt_utc = datetime.combine(m_date, m_time, tzinfo=UTC)
        match_id = make_canonical_match_id(season, idx)
        matchweek = ((idx - 1) // 10) + 1

        home_team = normalize_soccer_club(raw_home)
        away_team = normalize_soccer_club(raw_away)

        fthg_raw = str(row.get("FTHG") or "").strip()
        ftag_raw = str(row.get("FTAG") or "").strip()
        if fthg_raw != "" and ftag_raw != "":
            h_goals = int(float(fthg_raw))
            a_goals = int(float(ftag_raw))
            game_state = "FINAL"
            if h_goals > a_goals:
                outcome_1x2, home_win, is_draw, away_win = 0, 1, 0, 0
            elif h_goals == a_goals:
                outcome_1x2, home_win, is_draw, away_win = 1, 0, 1, 0
            else:
                outcome_1x2, home_win, is_draw, away_win = 2, 0, 0, 1
        else:
            h_goals = None
            a_goals = None
            game_state = "FUT"
            outcome_1x2 = None
            home_win = None
            is_draw = None
            away_win = None

        match_rows.append(
            {
                "sport": "soccer",
                "season": int(season),
                "matchweek": matchweek,
                "match_id": match_id,
                "event_seq": 1,
                "match_date": m_date,
                "start_time_utc": dt_utc,
                "home_team": home_team,
                "away_team": away_team,
                "venue_timezone": "Europe/London",
                "game_state": game_state,
                "home_goals": h_goals,
                "away_goals": a_goals,
                "outcome_1x2": outcome_1x2,
                "home_win": home_win,
                "is_draw": is_draw,
                "away_win": away_win,
            }
        )

        if game_state == "FINAL" and h_goals is not None and a_goals is not None:
            hs = int(float(row.get("HS") or 12))
            as_ = int(float(row.get("AS") or 10))
            hst = int(float(row.get("HST") or 4))
            ast = int(float(row.get("AST") or 4))
            hc = int(float(row.get("HC") or 5))
            ac = int(float(row.get("AC") or 4))
            hf = int(float(row.get("HF") or 10))
            af = int(float(row.get("AF") or 11))
            hy = int(float(row.get("HY") or 2))
            ay = int(float(row.get("AY") or 2))
            hr = int(float(row.get("HR") or 0))
            ar = int(float(row.get("AR") or 0))

            box_rows.append(
                {
                    "season": int(season),
                    "match_date": m_date,
                    "team": home_team,
                    "opponent": away_team,
                    "is_home": True,
                    "goals_for": h_goals,
                    "goals_against": a_goals,
                    "shots_for": hs,
                    "shots_against": as_,
                    "sot_for": hst,
                    "sot_against": ast,
                    "corners_for": hc,
                    "corners_against": ac,
                    "fouls_for": hf,
                    "fouls_against": af,
                    "yellow_cards": hy,
                    "red_cards": hr,
                }
            )
            box_rows.append(
                {
                    "season": int(season),
                    "match_date": m_date,
                    "team": away_team,
                    "opponent": home_team,
                    "is_home": False,
                    "goals_for": a_goals,
                    "goals_against": h_goals,
                    "shots_for": as_,
                    "shots_against": hs,
                    "sot_for": ast,
                    "sot_against": hst,
                    "corners_for": ac,
                    "corners_against": hc,
                    "fouls_for": af,
                    "fouls_against": hf,
                    "yellow_cards": ay,
                    "red_cards": ar,
                }
            )

    box_schema = {
        "season": pl.Int32(),
        "match_date": pl.Date(),
        "team": pl.Utf8(),
        "opponent": pl.Utf8(),
        "is_home": pl.Boolean(),
        "goals_for": pl.Int16(),
        "goals_against": pl.Int16(),
        "shots_for": pl.Int16(),
        "shots_against": pl.Int16(),
        "sot_for": pl.Int16(),
        "sot_against": pl.Int16(),
        "corners_for": pl.Int16(),
        "corners_against": pl.Int16(),
        "fouls_for": pl.Int16(),
        "fouls_against": pl.Int16(),
        "yellow_cards": pl.Int16(),
        "red_cards": pl.Int16(),
    }

    matches_df = (
        pl.DataFrame(match_rows, schema=SILVER_MATCHES_SCHEMA).sort(
            ["match_date", "start_time_utc", "event_seq", "match_id"]
        )
        if match_rows
        else pl.DataFrame(schema=SILVER_MATCHES_SCHEMA)
    )
    box_df = (
        pl.DataFrame(box_rows, schema=box_schema).sort(["match_date", "team"])
        if box_rows
        else pl.DataFrame(schema=box_schema)
    )
    return matches_df, box_df


class FootballDataClient:
    """Cached downloader for Football-Data.co.uk `E0.csv` historical and active-season files."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        base_url: str = FOOTBALL_DATA_BASE_URL,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _download_csv_text(self, season_code: str) -> str:
        url = f"{self.base_url}/{season_code}/E0.csv"
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            transport=self.transport,
            headers=headers,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text

    def fetch_season_csv(
        self,
        season: int,
        *,
        is_active_season: bool = False,
        ttl_seconds: float = 21600.0,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fetch or load cached Football-Data `E0.csv` for `season`."""
        code = season_to_football_data_code(season)
        preferred_gz = self.cache_dir / f"E0_{code}.csv.gz"
        existing = resolve_cache_read_path(preferred_gz)
        csv_text: str | None = None
        if existing is not None:
            if not is_active_season:
                csv_text = read_cached_bytes(preferred_gz).decode("utf-8", errors="replace")
            else:
                age = time.time() - existing.stat().st_mtime
                if age <= ttl_seconds:
                    csv_text = read_cached_bytes(preferred_gz).decode("utf-8", errors="replace")
        if csv_text is None:
            csv_text = self._download_csv_text(code)
            write_gzip_bytes(preferred_gz, csv_text.encode("utf-8"))
        return parse_football_data_csv(csv_text, season=season)
