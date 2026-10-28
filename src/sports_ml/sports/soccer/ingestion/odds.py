"""Real-time & historical EPL 3-way (`1X2`) market odds via ESPN Public API (`odds.py`).

Fetches pre-match and closing 3-way (`1X2`) moneylines (`homeTeamOdds`, `drawOdds`,
`awayTeamOdds`) from ESPN's official public soccer endpoints
(`site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard` and
`sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/events/{ID}/competitions/{ID}/odds`)
and converts them into decimal odds and power-de-vigged `1X2` probabilities
(`market_prob_home`, `market_prob_draw`, `market_prob_away`).

All columns produced by this module are strictly evaluation-only (`market_*`) and are
forbidden from entering Stage 1 or Stage 2 feature matrices (`X`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, Final

import httpx
import polars as pl

from sports_ml.core.market import american_to_decimal, devig_probabilities
from sports_ml.sports.soccer.constants import EPL_CLUBS, normalize_soccer_club

ESPN_EPL_SCOREBOARD_URL: Final[str] = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
)
ESPN_EPL_CORE_EVENTS_URL: Final[str] = (
    "https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/events"
)

SOCCER_ODDS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "match_date": pl.Date(),
    "home_team": pl.Utf8(),
    "away_team": pl.Utf8(),
    "market_home_decimal_odds": pl.Float64(),
    "market_draw_decimal_odds": pl.Float64(),
    "market_away_decimal_odds": pl.Float64(),
    "market_prob_home": pl.Float64(),
    "market_prob_draw": pl.Float64(),
    "market_prob_away": pl.Float64(),
    "market_overround": pl.Float64(),
}


def _coerce_soccer_odd_to_decimal(raw_val: Any) -> float | None:
    if raw_val is None:
        return None
    if isinstance(raw_val, Mapping):
        for k in ("decimal", "value", "moneyLine", "american"):
            if raw_val.get(k) is not None:
                res = _coerce_soccer_odd_to_decimal(raw_val[k])
                if res is not None:
                    return res
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


def parse_espn_epl_1x2_odds_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float] | None:
    """Extract consensus 3-way (`1X2`) decimal odds `(home_dec, draw_dec, away_dec)`."""
    valid_triplets: list[tuple[float, float, float]] = []
    for item in items:
        prov_name = str((item.get("provider") or {}).get("name") or "").strip().lower()
        if "live" in prov_name:
            continue

        h_obj = item.get("homeTeamOdds") or {}
        a_obj = item.get("awayTeamOdds") or {}
        d_obj = item.get("drawOdds") or {}
        if not isinstance(h_obj, Mapping) or not isinstance(a_obj, Mapping):
            continue

        h_dec = _coerce_soccer_odd_to_decimal(
            (h_obj.get("close") or {}).get("moneyLine")
            if isinstance(h_obj.get("close"), Mapping)
            else None
        ) or _coerce_soccer_odd_to_decimal(h_obj.get("moneyLine"))
        a_dec = _coerce_soccer_odd_to_decimal(
            (a_obj.get("close") or {}).get("moneyLine")
            if isinstance(a_obj.get("close"), Mapping)
            else None
        ) or _coerce_soccer_odd_to_decimal(a_obj.get("moneyLine"))
        d_dec = _coerce_soccer_odd_to_decimal(
            (d_obj.get("close") or {}).get("moneyLine")
            if isinstance(d_obj, Mapping) and isinstance(d_obj.get("close"), Mapping)
            else (d_obj.get("moneyLine") if isinstance(d_obj, Mapping) else item.get("drawOdds"))
        )

        if h_dec is None or d_dec is None or a_dec is None:
            continue

        inv_sum = (1.0 / h_dec) + (1.0 / d_dec) + (1.0 / a_dec)
        if 0.99 <= inv_sum <= 1.16:
            valid_triplets.append((h_dec, d_dec, a_dec))

    if not valid_triplets:
        return None

    n = len(valid_triplets)
    return (
        float(sum(t[0] for t in valid_triplets) / n),
        float(sum(t[1] for t in valid_triplets) / n),
        float(sum(t[2] for t in valid_triplets) / n),
    )


async def _fetch_espn_epl_odds_async(
    date_tokens: Sequence[str | None],
    *,
    timeout_seconds: float = 12.0,
) -> pl.DataFrame:
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:

        async def _fetch_board(token: str | None) -> list[tuple[str, date, str, str]]:
            params: dict[str, Any] = {"limit": 200}
            if token:
                params["dates"] = token
            async with sem:
                try:
                    resp = await client.get(ESPN_EPL_SCOREBOARD_URL, params=params)
                    if resp.status_code != 200:
                        return []
                    payload = resp.json()
                except Exception:
                    return []

            out_evs: list[tuple[str, date, str, str]] = []
            for ev in payload.get("events") or []:
                ev_id = str(ev.get("id") or "").strip()
                raw_dt = str(ev.get("date") or "").strip()
                if not ev_id or not raw_dt:
                    continue
                try:
                    dt_utc = datetime.fromisoformat(raw_dt.replace("Z", "+00:00")).astimezone(UTC)
                except ValueError:
                    continue
                comps = ev.get("competitions") or []
                if not comps:
                    continue
                h_code = ""
                a_code = ""
                for comp in comps[0].get("competitors") or []:
                    side = str(comp.get("homeAway") or "").lower()
                    t_obj = comp.get("team") or {}
                    code = normalize_soccer_club(
                        str(
                            t_obj.get("displayName")
                            or t_obj.get("shortDisplayName")
                            or t_obj.get("abbreviation")
                            or ""
                        )
                    )
                    if side == "home":
                        h_code = code
                    elif side == "away":
                        a_code = code
                if h_code in EPL_CLUBS and a_code in EPL_CLUBS:
                    out_evs.append((ev_id, dt_utc.date(), h_code, a_code))
            return out_evs

        boards = await asyncio.gather(*(_fetch_board(tok) for tok in date_tokens))
        unique_evs: dict[str, tuple[date, str, str]] = {}
        for b_list in boards:
            for ev_id, m_date, h_code, a_code in b_list:
                unique_evs[ev_id] = (m_date, h_code, a_code)

        if not unique_evs:
            return pl.DataFrame(schema=SOCCER_ODDS_SCHEMA)

        async def _fetch_ev_odds(
            ev_id: str, m_date: date, h_code: str, a_code: str
        ) -> dict[str, Any] | None:
            url = f"{ESPN_EPL_CORE_EVENTS_URL}/{ev_id}/competitions/{ev_id}/odds"
            async with sem:
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        return None
                    data = resp.json()
                except Exception:
                    return None

            parsed = parse_espn_epl_1x2_odds_items(data.get("items") or [])
            if parsed is None:
                return None
            dh, dd, da = parsed
            devigged = devig_probabilities([dh, dd, da], method="power")
            overround = (1.0 / dh) + (1.0 / dd) + (1.0 / da) - 1.0
            return {
                "match_date": m_date,
                "home_team": h_code,
                "away_team": a_code,
                "market_home_decimal_odds": dh,
                "market_draw_decimal_odds": dd,
                "market_away_decimal_odds": da,
                "market_prob_home": float(devigged[0]),
                "market_prob_draw": float(devigged[1]),
                "market_prob_away": float(devigged[2]),
                "market_overround": float(overround),
            }

        results = await asyncio.gather(
            *(
                _fetch_ev_odds(ev_id, m_date, h_code, a_code)
                for ev_id, (m_date, h_code, a_code) in unique_evs.items()
            )
        )
        valid_rows = [r for r in results if r is not None]
        if not valid_rows:
            return pl.DataFrame(schema=SOCCER_ODDS_SCHEMA)
        return pl.DataFrame(valid_rows, schema=SOCCER_ODDS_SCHEMA).sort(["match_date", "home_team"])


def fetch_espn_epl_date_odds(
    target_date: date | None = None,
    *,
    offline: bool = False,
) -> pl.DataFrame:
    """Fetch real-time ESPN EPL `1X2` odds for `target_date` (and active matchweek scoreboard)."""
    if offline:
        return pl.DataFrame(schema=SOCCER_ODDS_SCHEMA)
    tokens: list[str | None] = [None]
    if target_date is not None:
        tokens.append(target_date.strftime("%Y%m%d"))
        tokens.append(target_date.strftime("%Y%m"))
    try:
        return asyncio.run(_fetch_espn_epl_odds_async(tokens))
    except Exception:
        return pl.DataFrame(schema=SOCCER_ODDS_SCHEMA)
