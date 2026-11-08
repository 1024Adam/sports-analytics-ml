"""Evaluation-only NFL closing-odds ingestion (`football/ingestion/odds.py`).

`FootballOddsIngestor` merges three market sources into the nullable `market_*` columns of
Silver `games.parquet`:

1. The quarantined nflverse schedule columns (already reconciled by Milestone 23). They always
   win: a value the schedule provides is never overwritten.
2. An optional local historical CSV (`data/raw/odds/nfl_historical_odds.csv.gz`) that fills
   closing values the schedule lacks. Columns: either `nflverse_game_id` or
   `game_date,home_team,away_team`; moneylines (`home_moneyline`, `away_moneyline`, American)
   or `home_decimal_odds`/`away_decimal_odds`; `spread_home` (betting convention, negative when
   the home team is favored) or nflverse-style `spread_line` (positive when home is favored);
   and `total_points` (alias `total_line`).
3. An optional `THE_ODDS_API_KEY` pre-game snapshot (moneyline, spread, total averaged across
   bookmakers). It only fills not-yet-final games and is skipped silently without a key, when
   offline, or on any API failure.

Two-way moneyline probabilities are power de-vigged through `core.market.devig_probabilities`.
Everything produced here is strictly evaluation-only; `market_*` columns are stripped from every
feature builder by the Milestone 24 firewall.
"""

from __future__ import annotations

import csv
import io
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Final

import httpx
import polars as pl

from sports_ml.core.cache import read_cached_bytes, resolve_cache_read_path, write_cached_json
from sports_ml.core.market import american_to_decimal, devig_probabilities
from sports_ml.sports.football.constants import (
    normalize_football_team,
)

THE_ODDS_API_URL: Final[str] = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
HISTORICAL_ARCHIVE_NAME: Final[str] = "nfl_historical_odds.csv.gz"
LIVE_SNAPSHOT_NAME: Final[str] = "nfl_live_odds_latest.json.gz"

MARKET_COLUMNS: Final[tuple[str, ...]] = (
    "market_home_decimal_odds",
    "market_away_decimal_odds",
    "market_prob_home",
    "market_prob_away",
    "market_spread_home",
    "market_total_points",
)
TWO_SIDED_COLUMNS: Final[tuple[str, ...]] = MARKET_COLUMNS[:4]
ODDS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "nflverse_game_id": pl.Utf8(),
    "game_date": pl.Date(),
    "home_team": pl.Utf8(),
    "away_team": pl.Utf8(),
    **{c: pl.Float64() for c in MARKET_COLUMNS},
}
MATCH_DATE_TOLERANCE_DAYS: Final[int] = 1


def _float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number == number else None


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _team(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return normalize_football_team(text)
    except ValueError:
        return None


def _decimal_pair(
    dec_home: float | None, dec_away: float | None
) -> tuple[float | None, float | None, float | None, float | None]:
    if dec_home is None or dec_away is None or dec_home <= 1.0 or dec_away <= 1.0:
        return None, None, None, None
    probs = devig_probabilities([dec_home, dec_away], method="power")
    return dec_home, dec_away, float(probs[0]), float(probs[1])


def odds_row(
    *,
    home_ml: float | None = None,
    away_ml: float | None = None,
    home_decimal: float | None = None,
    away_decimal: float | None = None,
    spread_home: float | None = None,
    total: float | None = None,
) -> dict[str, float | None]:
    """Normalize one game's raw quotes into the `market_*` columns (power de-vigged)."""
    if home_decimal is None and home_ml is not None and abs(home_ml) >= 100:
        home_decimal = float(american_to_decimal(home_ml))
    if away_decimal is None and away_ml is not None and abs(away_ml) >= 100:
        away_decimal = float(american_to_decimal(away_ml))
    d_home, d_away, p_home, p_away = _decimal_pair(home_decimal, away_decimal)
    return {
        "market_home_decimal_odds": d_home,
        "market_away_decimal_odds": d_away,
        "market_prob_home": p_home,
        "market_prob_away": p_away,
        "market_spread_home": spread_home,
        "market_total_points": total,
    }


def parse_historical_odds_csv(csv_text: str) -> pl.DataFrame:
    """Parse a historical NFL odds CSV into `ODDS_SCHEMA` rows (unmatched keys are skipped)."""
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(io.StringIO(csv_text)):
        rec = {str(k).strip().lower(): v for k, v in raw.items() if k is not None}
        game_id = (rec.get("nflverse_game_id") or "").strip() or None
        home, away = _team(rec.get("home_team")), _team(rec.get("away_team"))
        game_date = _parse_date(rec.get("game_date") or rec.get("gameday"))
        if game_id is None and (home is None or away is None or game_date is None):
            continue
        spread = _float(rec.get("spread_home"))
        if spread is None:
            line = _float(rec.get("spread_line"))
            spread = -line if line is not None else None
        quote = odds_row(
            home_ml=_float(rec.get("home_moneyline")),
            away_ml=_float(rec.get("away_moneyline")),
            home_decimal=_float(rec.get("home_decimal_odds")),
            away_decimal=_float(rec.get("away_decimal_odds")),
            spread_home=spread,
            total=_float(
                rec.get("total_points") if "total_points" in rec else rec.get("total_line")
            ),
        )
        rows.append(
            {
                "nflverse_game_id": game_id,
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                **quote,
            }
        )
    return pl.DataFrame(rows, schema=ODDS_SCHEMA)


def parse_the_odds_api_payload(payload: list[dict[str, Any]]) -> pl.DataFrame:
    """Parse The Odds API v4 `h2h` / `spreads` / `totals` events (decimal odds, bookmaker mean)."""
    rows: list[dict[str, Any]] = []
    for event in payload:
        home, away = _team(event.get("home_team")), _team(event.get("away_team"))
        kickoff = str(event.get("commence_time") or "")
        try:
            game_date = (
                datetime.fromisoformat(kickoff.replace("Z", "+00:00")).astimezone(UTC).date()
            )
        except ValueError:
            continue
        if home is None or away is None:
            continue
        h2h_home: list[float] = []
        h2h_away: list[float] = []
        spreads: list[float] = []
        totals: list[float] = []
        for book in event.get("bookmakers") or []:
            for market in book.get("markets") or []:
                key = market.get("key")
                for out in market.get("outcomes") or []:
                    name = str(out.get("name") or "")
                    price, point = _float(out.get("price")), _float(out.get("point"))
                    if key == "h2h" and price is not None:
                        (h2h_home if _team(name) == home else h2h_away).append(price)
                    elif key == "spreads" and point is not None and _team(name) == home:
                        spreads.append(point)
                    elif key == "totals" and point is not None and name.lower() == "over":
                        totals.append(point)
        quote = odds_row(
            home_decimal=mean(h2h_home) if h2h_home else None,
            away_decimal=mean(h2h_away) if h2h_away else None,
            spread_home=mean(spreads) if spreads else None,
            total=mean(totals) if totals else None,
        )
        rows.append(
            {
                "nflverse_game_id": None,
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                **quote,
            }
        )
    return pl.DataFrame(rows, schema=ODDS_SCHEMA)


def merge_odds_into_games(
    games: pl.DataFrame,
    odds: pl.DataFrame,
    *,
    only_unfinished: bool = False,
) -> pl.DataFrame:
    """Fill null `market_*` columns of `games` from `odds` (existing values are never replaced).

    Rows match on `nflverse_game_id`, else on `(home_team, away_team)` with a `game_date` within
    one day (UTC kickoffs can fall on the next calendar day). `only_unfinished` restricts the fill
    to games that are not FINAL (the live pre-game snapshot).
    """
    if odds.is_empty() or games.is_empty():
        return games
    by_id = {
        str(r["nflverse_game_id"]): r for r in odds.iter_rows(named=True) if r["nflverse_game_id"]
    }
    by_teams: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in odds.iter_rows(named=True):
        if r["home_team"] and r["away_team"] and r["game_date"]:
            by_teams.setdefault((r["home_team"], r["away_team"]), []).append(r)

    tol = timedelta(days=MATCH_DATE_TOLERANCE_DAYS)
    fills: dict[str, list[float | None]] = {c: [] for c in MARKET_COLUMNS}
    for g in games.iter_rows(named=True):
        match: dict[str, Any] | None = None
        if not (only_unfinished and str(g["game_state"]).upper() == "FINAL"):
            match = by_id.get(str(g.get("nflverse_game_id") or ""))
            if match is None:
                candidates = by_teams.get((str(g["home_team"]), str(g["away_team"])), [])
                gd: date = g["game_date"]
                near = [c for c in candidates if abs(c["game_date"] - gd) <= tol]
                match = min(near, key=lambda c: abs(c["game_date"] - gd)) if near else None
        # Odds and probabilities form one two-sided quote: take them together or not at all.
        has_quote = any(g.get(c) is not None for c in TWO_SIDED_COLUMNS)
        for col in MARKET_COLUMNS:
            current = g.get(col)
            if (
                current is None
                and match is not None
                and not (col in TWO_SIDED_COLUMNS and has_quote)
            ):
                current = match[col]
            fills[col].append(current)
    return games.with_columns(*[pl.Series(c, v, dtype=pl.Float64) for c, v in fills.items()])


class FootballOddsIngestor:
    """Loads NFL closing odds (CSV archive, optional live snapshot) and merges them into Silver."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.api_key = api_key if api_key is not None else os.environ.get("THE_ODDS_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def historical_archive_path(self) -> Path:
        return self.cache_dir / HISTORICAL_ARCHIVE_NAME

    def load_historical_odds(self, archive_path: Path | None = None) -> pl.DataFrame:
        """Load the local historical CSV (`.csv.gz` or plain `.csv`); empty frame when absent."""
        target = archive_path or self.historical_archive_path
        resolved = resolve_cache_read_path(target)
        if resolved is None:
            return pl.DataFrame(schema=ODDS_SCHEMA)
        text = read_cached_bytes(resolved).decode("utf-8", errors="replace")
        return parse_historical_odds_csv(text)

    def fetch_live_odds_snapshot(self, *, offline: bool = False) -> pl.DataFrame:
        """Fetch a pre-game snapshot; empty without `THE_ODDS_API_KEY`, offline, or on failure."""
        if offline or not self.api_key:
            return pl.DataFrame(schema=ODDS_SCHEMA)
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
        }
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, follow_redirects=True, transport=self.transport
            ) as client:
                resp = client.get(THE_ODDS_API_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except (httpx.HTTPError, ValueError):
            return pl.DataFrame(schema=ODDS_SCHEMA)
        if not isinstance(payload, list):
            return pl.DataFrame(schema=ODDS_SCHEMA)
        write_cached_json(self.cache_dir / LIVE_SNAPSHOT_NAME, payload)
        return parse_the_odds_api_payload(payload)

    def apply_to_games(self, games: pl.DataFrame, *, offline: bool = False) -> pl.DataFrame:
        """Return `games` with archive closing odds and live pre-game quotes filled in."""
        merged = merge_odds_into_games(games, self.load_historical_odds())
        live = self.fetch_live_odds_snapshot(offline=offline)
        return merge_odds_into_games(merged, live, only_unfinished=True)

    def persist_to_silver(self, silver_dir: Path, *, offline: bool = False) -> int:
        """Merge odds into `games.parquet` in place; returns the number of games with a quote."""
        path = silver_dir / "games.parquet"
        if not path.exists():
            return 0
        games = pl.read_parquet(path)
        merged = self.apply_to_games(games, offline=offline)
        if not merged.equals(games):
            merged.write_parquet(path, compression="zstd")
        return int(merged["market_prob_home"].is_not_null().sum())


__all__ = [
    "FootballOddsIngestor",
    "MARKET_COLUMNS",
    "merge_odds_into_games",
    "odds_row",
    "parse_historical_odds_csv",
    "parse_the_odds_api_payload",
]
