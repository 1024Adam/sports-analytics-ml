"""Evaluation-only MLB & NPB historical/closing odds ingestion (`baseball/ingestion/odds.py`).

Supports:
- MLB 2-Way Moneyline (American moneylines like `-150` / `+130` or decimal odds),
  power-de-vigged into `market_prob_home`, `market_prob_away` (`market_prob_draw = None`).
- NPB 3-Way (`1X2`: Home, 12-Inning Draw, Away) and 2-Way Draw-No-Bet (`DNB`) closing
  decimal odds, power-de-vigged into `market_prob_home`, `market_prob_draw`, `market_prob_away`.

All columns produced by this module are strictly evaluation-only (`market_*`) and are
forbidden from entering Stage 1 or Stage 2 feature matrices (`X`).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from sports_ml.core.cache import read_cached_bytes, resolve_cache_read_path
from sports_ml.core.market import american_to_decimal, devig_probabilities
from sports_ml.sports.baseball.constants import (
    normalize_baseball_team,
    resolve_team_league,
)

BASEBALL_ODDS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "league": pl.Utf8(),
    "game_date": pl.Date(),
    "home_team": pl.Utf8(),
    "away_team": pl.Utf8(),
    "market_home_decimal_odds": pl.Float64(),
    "market_draw_decimal_odds": pl.Float64(),
    "market_away_decimal_odds": pl.Float64(),
    "market_prob_home": pl.Float64(),
    "market_prob_draw": pl.Float64(),
    "market_prob_away": pl.Float64(),
}


def _to_decimal_odds(val: Any) -> float | None:
    if val is None or str(val).strip() == "":
        return None
    num = float(val)
    if abs(num) >= 100.0:
        return float(american_to_decimal(num))
    if num > 1.0:
        return num
    return None


def parse_baseball_historical_odds_csv(csv_text_or_bytes: str | bytes) -> pl.DataFrame:
    """Parse MLB (2-way) and NPB (3-way or 2-way) historical closing odds CSV."""
    text = (
        csv_text_or_bytes.decode("utf-8", errors="replace")
        if isinstance(csv_text_or_bytes, bytes)
        else csv_text_or_bytes
    )
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []

    for raw in reader:
        home_team = normalize_baseball_team(str(raw.get("home_team") or ""))
        away_team = normalize_baseball_team(str(raw.get("away_team") or ""))
        if not home_team or not away_team:
            continue

        gid = int(raw["game_id"]) if raw.get("game_id") else -1
        seq = int(raw.get("event_seq") or 1)
        lg = str(raw.get("league") or resolve_team_league(home_team) or "MLB").strip().upper()
        date_str = str(raw.get("game_date") or raw.get("date") or "2025-05-01")[:10]

        h_dec = _to_decimal_odds(
            raw.get("market_home_decimal_odds") or raw.get("home_odds") or raw.get("home_moneyline")
        )
        d_dec = _to_decimal_odds(raw.get("market_draw_decimal_odds") or raw.get("draw_odds"))
        a_dec = _to_decimal_odds(
            raw.get("market_away_decimal_odds") or raw.get("away_odds") or raw.get("away_moneyline")
        )
        if h_dec is None or a_dec is None:
            continue

        if d_dec is not None and d_dec > 1.0:
            arr_3 = np.array([[h_dec, d_dec, a_dec]], dtype=np.float64)
            devig_3 = devig_probabilities(arr_3, method="power")[0]
            p_h = float(devig_3[0])
            p_d: float | None = float(devig_3[1])
            p_a = float(devig_3[2])
        else:
            arr_2 = np.array([[h_dec, a_dec]], dtype=np.float64)
            devig_2 = devig_probabilities(arr_2, method="power")[0]
            p_h = float(devig_2[0])
            p_d = None
            p_a = float(devig_2[1])

        rows.append(
            {
                "game_id": gid,
                "event_seq": seq,
                "league": lg,
                "game_date": pl.Series([date_str]).str.to_date()[0],
                "home_team": home_team,
                "away_team": away_team,
                "market_home_decimal_odds": h_dec,
                "market_draw_decimal_odds": d_dec,
                "market_away_decimal_odds": a_dec,
                "market_prob_home": p_h,
                "market_prob_draw": p_d,
                "market_prob_away": p_a,
            }
        )

    if not rows:
        return pl.DataFrame(schema=BASEBALL_ODDS_SCHEMA)
    df = pl.DataFrame(rows)
    return df.select([pl.col(c).cast(dt) for c, dt in BASEBALL_ODDS_SCHEMA.items()])


class BaseballOddsIngestor:
    """Loader for evaluation-only MLB and NPB closing odds (`data/raw/baseball/odds/`)."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_cached_odds(self) -> pl.DataFrame:
        """Load and de-vig any cached MLB/NPB historical odds CSVs (`*.csv` / `*.csv.gz`)."""
        frames: list[pl.DataFrame] = []
        for stem in (
            "baseball_historical_odds.csv.gz",
            "mlb_historical_odds.csv.gz",
            "npb_historical_odds.csv.gz",
        ):
            cand = resolve_cache_read_path(self.cache_dir / stem)
            if cand is not None:
                raw_bytes = read_cached_bytes(self.cache_dir / stem)
                parsed = parse_baseball_historical_odds_csv(raw_bytes)
                if not parsed.is_empty():
                    frames.append(parsed)
        if not frames:
            return pl.DataFrame(schema=BASEBALL_ODDS_SCHEMA)
        return pl.concat(frames, how="vertical").unique(
            subset=["game_date", "home_team", "away_team", "event_seq"], keep="last"
        )
