"""Historical and optional pre-game NHL closing odds ingestion (`sports/hockey/ingestion/odds.py`).

Supports:
1. Local/public historical NHL closing moneyline & total CSV archives
   (`data/raw/odds/nhl_historical_odds.csv.gz`), converting American moneylines
   (e.g. `-145` / `+125`) or decimal odds into decimal and power-de-vigged
   probabilities (`market_home_decimal_odds`, `market_away_decimal_odds`,
   `market_prob_home`, `market_prob_away`, `market_overround`).
2. Optional pre-game odds snapshot hook via `THE_ODDS_API_KEY` environment variable
   (gracefully skipped when absent).

All columns produced by this module are strictly evaluation-only (`market_*`) and are
forbidden from entering Stage 1 or Stage 2 feature matrices (`X`).
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl

from sports_ml.core.cache import (
    read_cached_bytes,
    resolve_cache_read_path,
    write_cached_json,
    write_gzip_bytes,
)
from sports_ml.core.market import american_to_decimal, devig_probabilities
from sports_ml.sports.hockey.constants import NHL_TEAMS_32, normalize_tricode

THE_ODDS_API_BASE_URL: Final[str] = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
ESPN_NHL_SCOREBOARD_URL: Final[str] = (
    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
)
ESPN_NHL_CORE_EVENTS_URL: Final[str] = (
    "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/events"
)

DEFAULT_ESPN_BACKFILL_MONTHS: Final[tuple[str, ...]] = (
    "202310",
    "202311",
    "202312",
    "202401",
    "202402",
    "202403",
    "202404",
    "202410",
    "202411",
    "202412",
    "202501",
    "202502",
    "202503",
    "202504",
    "202510",
    "202511",
    "202512",
    "202601",
    "202602",
    "202603",
    "202604",
    "202610",
)

NHL_ODDS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "game_date": pl.Date(),
    "home_team": pl.Utf8(),
    "away_team": pl.Utf8(),
    "market_home_decimal_odds": pl.Float64(),
    "market_away_decimal_odds": pl.Float64(),
    "market_prob_home": pl.Float64(),
    "market_prob_away": pl.Float64(),
    "market_overround": pl.Float64(),
}

_FULL_NAME_TO_TRICODE: Final[dict[str, str]] = {
    **{meta.team_name.upper(): code for code, meta in NHL_TEAMS_32.items()},
    "MONTREAL CANADIENS": "MTL",
    "MONTRÉAL CANADIENS": "MTL",
    "UTAH MAMMOTH": "UTA",
    "UTAH HOCKEY CLUB": "UTA",
    "ARIZONA COYOTES": "UTA",
    "PHOENIX COYOTES": "UTA",
    "ATLANTA THRASHERS": "WPG",
    "ST LOUIS BLUES": "STL",
    "TB": "TBL",
    "NJ": "NJD",
    "LA": "LAK",
    "SJ": "SJS",
}


def _normalize_nhl_odds_team(raw_team: str) -> str:
    ascii_str = (
        unicodedata.normalize("NFKD", raw_team.strip()).encode("ascii", "ignore").decode("ascii")
    )
    cleaned = " ".join(ascii_str.upper().split())
    if cleaned in _FULL_NAME_TO_TRICODE:
        return _FULL_NAME_TO_TRICODE[cleaned]
    return normalize_tricode(cleaned)


def _coerce_odd_to_decimal(raw_val: Any) -> float | None:
    """Convert an American moneyline (`-145`, `+125`) or decimal odd (`1.69`) to decimal."""
    if raw_val is None:
        return None
    cleaned = str(raw_val).strip().replace("+", "")
    if not cleaned:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None

    if abs(val) >= 100.0:
        return float(american_to_decimal(val))
    if val > 1.0:
        return val
    return None


def parse_espn_core_odds_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[float, float] | None:
    """Extract consensus 2-way closing decimal odds `(home_dec, away_dec)` from ESPN Core API items.

    Filters out live in-play lines (`"live"` in provider name) and 3-way 60-minute regulation
    lines (`overround < 0.98`), preferring closing lines (`close.moneyLine`) when present and
    falling back to pre-game `moneyLine` across ESPN BET, Caesars, DraftKings, MGM, Westgate,
    and Bet365.
    """
    valid_pairs: list[tuple[float, float]] = []
    for item in items:
        prov_name = str((item.get("provider") or {}).get("name") or "").strip().lower()
        if "live" in prov_name:
            continue

        h_obj = item.get("homeTeamOdds") or {}
        a_obj = item.get("awayTeamOdds") or {}
        if not isinstance(h_obj, Mapping) or not isinstance(a_obj, Mapping):
            continue

        h_close = (h_obj.get("close") or {}).get("moneyLine") or {}
        a_close = (a_obj.get("close") or {}).get("moneyLine") or {}
        h_dec = _coerce_odd_to_decimal(
            h_close.get("decimal") if isinstance(h_close, Mapping) else None
        ) or _coerce_odd_to_decimal(h_obj.get("moneyLine"))
        a_dec = _coerce_odd_to_decimal(
            a_close.get("decimal") if isinstance(a_close, Mapping) else None
        ) or _coerce_odd_to_decimal(a_obj.get("moneyLine"))

        if h_dec is None or a_dec is None:
            continue

        inv_sum = (1.0 / h_dec) + (1.0 / a_dec)
        if 0.99 <= inv_sum <= 1.12:
            valid_pairs.append((h_dec, a_dec))

    if not valid_pairs:
        return None

    mean_h = float(sum(p[0] for p in valid_pairs) / len(valid_pairs))
    mean_a = float(sum(p[1] for p in valid_pairs) / len(valid_pairs))
    return (mean_h, mean_a)


def parse_nhl_historical_odds_csv(csv_text: str) -> pl.DataFrame:
    """Parse an NHL historical closing odds CSV text into `NHL_ODDS_SCHEMA`.

    Accepts flexible column headers:
    - `game_id` (optional)
    - `game_date` or `date`
    - `home_team` / `home` and `away_team` / `away`
    - `home_moneyline` / `home_ml` / `home_odds` / `market_home_decimal_odds`
    - `away_moneyline` / `away_ml` / `away_odds` / `market_away_decimal_odds`
    """
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    rows: list[dict[str, Any]] = []

    for raw_row in reader:
        row = {str(k).strip().lower(): v for k, v in raw_row.items() if k is not None}

        league_val = str(row.get("league") or "").strip().upper()
        if league_val and "NHL" not in league_val:
            continue

        raw_gid = row.get("game_id") or row.get("gameid")
        game_id: int | None = None
        if raw_gid is not None and str(raw_gid).strip().isdigit():
            game_id = int(str(raw_gid).strip())

        raw_date = str(row.get("game_date") or row.get("date") or "").strip()
        game_date_val: date | None = None
        if raw_date:
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
                try:
                    game_date_val = datetime.strptime(raw_date[:10], fmt).date()
                    break
                except ValueError:
                    continue

        raw_home = str(row.get("home_team") or row.get("home") or "").strip()
        raw_away = str(row.get("away_team") or row.get("away") or "").strip()
        home_team = _normalize_nhl_odds_team(raw_home) if raw_home else ""
        away_team = _normalize_nhl_odds_team(raw_away) if raw_away else ""

        if game_id is None and (game_date_val is None or not home_team or not away_team):
            continue
        if (
            home_team
            and away_team
            and (home_team not in NHL_TEAMS_32 or away_team not in NHL_TEAMS_32)
        ):
            continue

        h_dec = _coerce_odd_to_decimal(
            row.get("market_home_decimal_odds")
            or row.get("home_decimal_odds")
            or row.get("home_moneyline")
            or row.get("home_ml")
            or row.get("home_odds")
            or row.get("odds_1")
            or row.get("close_home_ml")
        )
        a_dec = _coerce_odd_to_decimal(
            row.get("market_away_decimal_odds")
            or row.get("away_decimal_odds")
            or row.get("away_moneyline")
            or row.get("away_ml")
            or row.get("away_odds")
            or row.get("odds_2")
            or row.get("close_away_ml")
        )
        if h_dec is None or a_dec is None:
            continue

        devigged = devig_probabilities([h_dec, a_dec], method="power")
        overround = (1.0 / h_dec) + (1.0 / a_dec) - 1.0

        rows.append(
            {
                "game_id": game_id,
                "game_date": game_date_val,
                "home_team": home_team or None,
                "away_team": away_team or None,
                "market_home_decimal_odds": float(h_dec),
                "market_away_decimal_odds": float(a_dec),
                "market_prob_home": float(devigged[0]),
                "market_prob_away": float(devigged[1]),
                "market_overround": float(overround),
            }
        )

    if not rows:
        return pl.DataFrame(schema=NHL_ODDS_SCHEMA)
    return pl.DataFrame(rows, schema=NHL_ODDS_SCHEMA)


def parse_the_odds_api_payload(events: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """Parse a response payload from The-Odds-API (`/v4/sports/icehockey_nhl/odds`)."""
    rows: list[dict[str, Any]] = []
    for ev in events:
        commence_raw = ev.get("commence_time")
        if not commence_raw:
            continue
        dt_utc = datetime.fromisoformat(str(commence_raw).replace("Z", "+00:00")).astimezone(UTC)
        g_date = dt_utc.date()

        home_team = _normalize_nhl_odds_team(str(ev.get("home_team") or ""))
        away_team = _normalize_nhl_odds_team(str(ev.get("away_team") or ""))
        if not home_team or not away_team:
            continue

        bookmakers = ev.get("bookmakers") or []
        h_odds_list: list[float] = []
        a_odds_list: list[float] = []

        for book in bookmakers:
            for mkt in book.get("markets") or []:
                if mkt.get("key") != "h2h":
                    continue
                outcomes = mkt.get("outcomes") or []
                h_price: float | None = None
                a_price: float | None = None
                for out in outcomes:
                    name_code = _normalize_nhl_odds_team(str(out.get("name") or ""))
                    price_dec = _coerce_odd_to_decimal(out.get("price"))
                    if price_dec is None:
                        continue
                    if name_code == home_team:
                        h_price = price_dec
                    elif name_code == away_team:
                        a_price = price_dec
                if h_price is not None and a_price is not None:
                    h_odds_list.append(h_price)
                    a_odds_list.append(a_price)

        if not h_odds_list or not a_odds_list:
            continue

        h_dec = float(sum(h_odds_list) / len(h_odds_list))
        a_dec = float(sum(a_odds_list) / len(a_odds_list))
        devigged = devig_probabilities([h_dec, a_dec], method="power")
        overround = (1.0 / h_dec) + (1.0 / a_dec) - 1.0

        rows.append(
            {
                "game_id": None,
                "game_date": g_date,
                "home_team": home_team,
                "away_team": away_team,
                "market_home_decimal_odds": h_dec,
                "market_away_decimal_odds": a_dec,
                "market_prob_home": float(devigged[0]),
                "market_prob_away": float(devigged[1]),
                "market_overround": float(overround),
            }
        )

    if not rows:
        return pl.DataFrame(schema=NHL_ODDS_SCHEMA)
    return pl.DataFrame(rows, schema=NHL_ODDS_SCHEMA)


class NHLOddsIngestor:
    """Ingests evaluation-only NHL historical closing moneylines and optional live snapshots."""

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
        return self.cache_dir / "nhl_historical_odds.csv.gz"

    async def _backfill_espn_historical_odds_async(
        self,
        months: Sequence[str] = DEFAULT_ESPN_BACKFILL_MONTHS,
        *,
        max_concurrency: int = 40,
    ) -> pl.DataFrame:
        """Fetch historical NHL closing moneylines concurrently from ESPN Public Core API."""
        sem = asyncio.Semaphore(max_concurrency)
        limits = httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=20)

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            limits=limits,
        ) as client:

            async def _fetch_month(ym: str) -> list[tuple[str, date, str, str]]:
                async with sem:
                    try:
                        resp = await client.get(
                            ESPN_NHL_SCOREBOARD_URL,
                            params={"dates": ym, "limit": 1000},
                        )
                        if resp.status_code != 200:
                            return []
                        payload = resp.json()
                    except Exception:
                        return []

                out_events: list[tuple[str, date, str, str]] = []
                for ev in payload.get("events") or []:
                    ev_id = str(ev.get("id") or "").strip()
                    raw_dt = str(ev.get("date") or "").strip()
                    if not ev_id or not raw_dt:
                        continue
                    try:
                        dt_utc = datetime.fromisoformat(raw_dt.replace("Z", "+00:00")).astimezone(
                            UTC
                        )
                    except ValueError:
                        continue
                    comps = ev.get("competitions") or []
                    if not comps:
                        continue
                    competitors = comps[0].get("competitors") or []
                    h_code = ""
                    a_code = ""
                    for comp in competitors:
                        side = str(comp.get("homeAway") or "").lower()
                        team_obj = comp.get("team") or {}
                        abbr = _normalize_nhl_odds_team(
                            str(team_obj.get("abbreviation") or team_obj.get("displayName") or "")
                        )
                        if side == "home":
                            h_code = abbr
                        elif side == "away":
                            a_code = abbr
                    if h_code in NHL_TEAMS_32 and a_code in NHL_TEAMS_32:
                        out_events.append((ev_id, dt_utc.date(), h_code, a_code))
                return out_events

            month_lists = await asyncio.gather(*(_fetch_month(ym) for ym in months))
            unique_events: dict[str, tuple[date, str, str]] = {}
            for m_list in month_lists:
                for ev_id, g_date, h_code, a_code in m_list:
                    unique_events[ev_id] = (g_date, h_code, a_code)

            if not unique_events:
                return pl.DataFrame(schema=NHL_ODDS_SCHEMA)

            async def _fetch_event_odds(
                ev_id: str, g_date: date, h_code: str, a_code: str
            ) -> dict[str, Any] | None:
                url = f"{ESPN_NHL_CORE_EVENTS_URL}/{ev_id}/competitions/{ev_id}/odds"
                async with sem:
                    try:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            return None
                        data = resp.json()
                    except Exception:
                        return None

                items = data.get("items") or []
                parsed = parse_espn_core_odds_items(items)
                if parsed is None:
                    return None
                h_dec, a_dec = parsed
                devigged = devig_probabilities([h_dec, a_dec], method="power")
                overround = (1.0 / h_dec) + (1.0 / a_dec) - 1.0
                return {
                    "game_id": None,
                    "game_date": g_date,
                    "home_team": h_code,
                    "away_team": a_code,
                    "market_home_decimal_odds": h_dec,
                    "market_away_decimal_odds": a_dec,
                    "market_prob_home": float(devigged[0]),
                    "market_prob_away": float(devigged[1]),
                    "market_overround": float(overround),
                }

            results = await asyncio.gather(
                *(
                    _fetch_event_odds(ev_id, g_date, h_code, a_code)
                    for ev_id, (g_date, h_code, a_code) in unique_events.items()
                )
            )
            valid_rows = [r for r in results if r is not None]
            if not valid_rows:
                return pl.DataFrame(schema=NHL_ODDS_SCHEMA)
            return pl.DataFrame(valid_rows, schema=NHL_ODDS_SCHEMA).sort(["game_date", "home_team"])

    def fetch_espn_historical_odds(
        self,
        months: Sequence[str] = DEFAULT_ESPN_BACKFILL_MONTHS,
    ) -> pl.DataFrame:
        """Synchronously execute ESPN Public Core API historical NHL odds backfill and cache."""
        try:
            df = asyncio.run(self._backfill_espn_historical_odds_async(months=months))
        except Exception:
            return pl.DataFrame(schema=NHL_ODDS_SCHEMA)
        if not df.is_empty():
            buf = io.BytesIO()
            df.write_csv(buf)
            write_gzip_bytes(self.historical_archive_path, buf.getvalue())
        return df

    def load_historical_odds_archive(
        self,
        archive_path: Path | None = None,
        *,
        offline: bool = False,
    ) -> pl.DataFrame:
        """Load `nhl_historical_odds.csv.gz`, backfilling from ESPN Public API when online."""
        target = archive_path or self.historical_archive_path
        resolved = resolve_cache_read_path(target)
        if resolved is None and not offline and self.transport is None:
            return self.fetch_espn_historical_odds()

        if resolved is None:
            return pl.DataFrame(schema=NHL_ODDS_SCHEMA)
        csv_text = read_cached_bytes(resolved).decode("utf-8", errors="replace")
        return parse_nhl_historical_odds_csv(csv_text)

    def fetch_live_odds_snapshot(self, *, offline: bool = False) -> pl.DataFrame:
        """Fetch pre-game NHL moneyline snapshot when `THE_ODDS_API_KEY` is configured.

        Gracefully returns an empty DataFrame when `THE_ODDS_API_KEY` is absent or `offline=True`.
        """
        if offline or not self.api_key:
            return pl.DataFrame(schema=NHL_ODDS_SCHEMA)

        params = {
            "apiKey": self.api_key,
            "regions": "us,eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=True,
                transport=self.transport,
            ) as client:
                resp = client.get(THE_ODDS_API_BASE_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except Exception:
            return pl.DataFrame(schema=NHL_ODDS_SCHEMA)

        if not isinstance(payload, list):
            return pl.DataFrame(schema=NHL_ODDS_SCHEMA)

        snap_path = self.cache_dir / "nhl_live_odds_latest.json.gz"
        write_cached_json(snap_path, payload)
        return parse_the_odds_api_payload(payload)

    def load_all_odds(self, *, offline: bool = False) -> pl.DataFrame:
        """Combine historical CSV archive and optional live snapshot into a single DataFrame."""
        hist_df = self.load_historical_odds_archive(offline=offline)
        live_df = self.fetch_live_odds_snapshot(offline=offline)
        if hist_df.is_empty():
            return live_df
        if live_df.is_empty():
            return hist_df
        return pl.concat([hist_df, live_df], how="vertical")
