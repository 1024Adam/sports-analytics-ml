"""Cached HTTP client and parsers for Nippon Professional Baseball (`npb_client.py`).

Handles Central League (`CL`), Pacific League (`PL`), and Interleague (`KORYUSEN`)
schedules, 12-inning Tie games (`is_tie = 1`, `outcome_1x2 = 1`), provincial/regional
neutral stadiums (Matsuyama, Miyazaki, Toyama, etc.), August Koshien displacement,
probable starter announcements (`予告先発`), and active `ichi-gun` (first-team) rosters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from sports_ml.core.cache import read_cached_json, write_cached_json
from sports_ml.sports.baseball.adapter import (
    outs_to_true_innings,
    parse_baseball_ip_to_outs,
)
from sports_ml.sports.baseball.constants import (
    BASEBALL_BALLPARKS,
    NPB_SABERMETRIC_PRIORS,
    SILVER_BASEBALL_BATTER_LOGS_SCHEMA,
    SILVER_BASEBALL_GAMES_SCHEMA,
    SILVER_BASEBALL_PITCHER_LOGS_SCHEMA,
    SILVER_BASEBALL_TEAM_LOGS_SCHEMA,
    get_ballpark_metadata,
    get_baseball_club_metadata,
    normalize_baseball_team,
)

NPB_BIS_BASE_URL: Final[str] = "https://npb.jp/bis/eng"
NPB_GAME_ID_OFFSET: Final[int] = 9_000_000_000

_NPB_VENUE_KEYWORDS: Final[list[tuple[tuple[str, ...], str]]] = [
    (("MATSUYAMA", "BOTCHAN", "松山", "坊っちゃん"), "NPB_MATSUYAMA_BOTCHAN"),
    (("MIYAZAKI", "SUN MARINE", "宮崎", "サンマリン"), "NPB_MIYAZAKI"),
    (("TOYAMA", "ALPEN", "富山", "アルペン"), "NPB_TOYAMA"),
    (("NAGANO", "OLYMPIC", "長野"), "NPB_NAGANO"),
    (("NIIGATA", "HARD OFF", "新潟", "ハードオフ"), "NPB_NIIGATA_ECO"),
    (("KITAKYUSHU", "北九州"), "NPB_KITAKYUSHU"),
    (("AKITA", "KOMACHI", "秋田", "こまち"), "NPB_AKITA"),
    (("KORIYAMA", "KAISEIYAMA", "郡山", "開成山"), "NPB_KORIYAMA"),
    (("HOTTO MOTTO", "KOBE", "ほっともっと", "神戸"), "NPB_HOTTO_MOTTO_KOBE"),
    (("TOKYO DOME", "東京ドーム"), "NPB_TOKYO_DOME"),
    (("KOSHIEN", "甲子園"), "NPB_KOSHIEN"),
    (("YOKOHAMA", "横浜"), "NPB_YOKOHAMA"),
    (("MAZDA", "HIROSHIMA", "マツダ", "広島"), "NPB_MAZDA"),
    (("JINGU", "明治神宮", "神宮"), "NPB_JINGU"),
    (("VANTELIN", "NAGOYA DOME", "バンテリン", "ナゴヤドーム"), "NPB_VANTELIN_DOME"),
    (("PAYPAY", "MIZUHO", "FUKUOKA DOME", "みずほ", "ペイペイドーム"), "NPB_PAYPAY_DOME"),
    (("KYOCERA", "OSAKA DOME", "京セラ", "大阪ドーム"), "NPB_KYOCERA_DOME"),
    (("ZOZO", "CHIBA MARINE", "マリン"), "NPB_ZOZO_MARINE"),
    (("RAKUTEN MOBILE", "MIYAGI", "楽天モバイル", "宮城"), "NPB_RAKUTEN_MOBILE"),
    (("BELLUNA", "METLIFE", "SEIBU DOME", "ベルーナ", "西武ドーム"), "NPB_BELLUNA_DOME"),
    (("ES CON", "HOKKAIDO", "エスコン"), "NPB_ES_CON_FIELD"),
]


def _cast_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema=schema)
    exprs = [
        pl.col(col).cast(dtype) if col in df.columns else pl.lit(None, dtype=dtype).alias(col)
        for col, dtype in schema.items()
    ]
    return df.select(exprs)


def namespace_npb_game_id(raw_id: int | str) -> int:
    """Ensure NPB game IDs are namespaced in the `9_000_000_000+` range to avoid MLB collisions."""
    val = int(raw_id)
    if val < NPB_GAME_ID_OFFSET:
        return NPB_GAME_ID_OFFSET + val
    return val


def resolve_npb_venue(
    venue_id_or_name: str | None,
    home_team: str,
) -> tuple[str, str, str, bool, bool]:
    """Resolve `(venue_id, venue_name, timezone, is_dome, is_neutral_or_regional)` for an NPB game.

    Automatically detects provincial neutral stadiums (Matsuyama, Miyazaki, Toyama, etc.)
    as well as displaced home games (e.g., Hanshin Tigers playing home games at Kyocera Dome
    during the August Summer Koshien High School Baseball Tournament).
    """
    norm_home = normalize_baseball_team(home_team)
    club = get_baseball_club_metadata(norm_home)
    raw = (venue_id_or_name or "").strip()
    upper = raw.upper()

    resolved_park = None
    if upper in BASEBALL_BALLPARKS:
        resolved_park = BASEBALL_BALLPARKS[upper]
    else:
        for keywords, vid in _NPB_VENUE_KEYWORDS:
            if any(kw in upper or kw in raw for kw in keywords):
                resolved_park = BASEBALL_BALLPARKS[vid]
                break

    if resolved_park is None:
        resolved_park = get_ballpark_metadata(norm_home) or BASEBALL_BALLPARKS["NPB_TOKYO_DOME"]

    is_displaced_home = club is not None and resolved_park.venue_id != club.primary_venue_id
    is_neutral_or_regional = resolved_park.is_neutral_or_regional or is_displaced_home
    venue_name = raw if raw and upper not in BASEBALL_BALLPARKS else resolved_park.venue_name

    return (
        resolved_park.venue_id,
        venue_name,
        resolved_park.timezone,
        resolved_park.is_dome_or_retractable,
        is_neutral_or_regional,
    )


def resolve_npb_sub_league_and_dh(
    home_team: str,
    away_team: str,
    explicit_uses_dh: bool | None = None,
) -> tuple[str, bool]:
    """Resolve NPB `sub_league` (`CL`, `PL`, or `KORYUSEN`) and DH (`uses_dh`) rule."""
    h_club = get_baseball_club_metadata(home_team)
    a_club = get_baseball_club_metadata(away_team)
    h_sub = h_club.sub_league if h_club is not None else "CL"
    a_sub = a_club.sub_league if a_club is not None else "CL"

    sub_league = "KORYUSEN" if h_sub != a_sub else h_sub

    # In NPB, Pacific League home games always use DH; Central League home games do not
    # unless explicitly overridden (e.g., special interleague or future rule years).
    uses_dh = explicit_uses_dh if explicit_uses_dh is not None else (h_sub == "PL")
    return (sub_league, uses_dh)


def _compute_npb_woba_num(
    *,
    singles: int,
    doubles: int,
    triples: int,
    hr: int,
    bb: int,
    ibb: int,
    hbp: int,
) -> float:
    priors = NPB_SABERMETRIC_PRIORS
    ubb = max(0, bb - ibb)
    return float(
        priors.w_ubb * ubb
        + priors.w_hbp * hbp
        + priors.w_1b * singles
        + priors.w_2b * doubles
        + priors.w_3b * triples
        + priors.w_hr * hr
    )


def _parse_utc_timestamp(ts_str: str | None, fallback_date: date) -> datetime:
    if ts_str:
        cleaned = ts_str.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
            return dt.astimezone(UTC) if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
        except ValueError:
            pass
    # Typical NPB night game starts at 18:00 JST = 09:00 UTC
    return datetime(fallback_date.year, fallback_date.month, fallback_date.day, 9, 0, tzinfo=UTC)


def parse_npb_schedule_payload(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    season: int | None = None,
) -> pl.DataFrame:
    """Parse NPB schedule JSON/records into `SILVER_BASEBALL_GAMES_SCHEMA` (12-inning ties)."""
    if isinstance(payload, Mapping):
        games_iter: Sequence[Mapping[str, Any]] = payload.get("games") or []
    else:
        games_iter = payload

    rows: list[dict[str, Any]] = []
    for g in games_iter:
        if not isinstance(g, Mapping):
            continue
        raw_id = g.get("game_id") or g.get("id")
        if raw_id is None:
            continue
        game_id = namespace_npb_game_id(raw_id)
        raw_seq = int(g.get("event_seq") or g.get("game_number") or 1)
        event_seq = 2 if raw_seq >= 2 else 1

        date_str = str(g.get("game_date") or g.get("date") or "2025-04-01")
        game_date = date.fromisoformat(date_str[:10])
        season_val = int(season or g.get("season") or game_date.year)
        start_utc = _parse_utc_timestamp(g.get("start_time_utc") or g.get("start_time"), game_date)

        home_team = normalize_baseball_team(str(g.get("home_team") or g.get("home") or "YOM"))
        away_team = normalize_baseball_team(str(g.get("away_team") or g.get("away") or "HAN"))

        raw_venue = str(g.get("venue_id") or g.get("venue_name") or g.get("venue") or "")
        venue_id, venue_name, venue_tz, is_dome, is_neutral = resolve_npb_venue(
            raw_venue, home_team
        )
        if bool(g.get("is_neutral_or_regional_venue", False)):
            is_neutral = True

        explicit_dh = g.get("uses_dh") if isinstance(g.get("uses_dh"), bool) else None
        sub_league, uses_dh = resolve_npb_sub_league_and_dh(
            home_team, away_team, explicit_uses_dh=explicit_dh
        )

        state_str = str(g.get("game_state") or g.get("status") or "FUT").strip().upper()
        is_final = state_str in {"FINAL", "F", "COMPLETED", "FINISHED"}
        game_state = "FINAL" if is_final else "FUT"

        home_sp_id = int(g["home_sp_id"]) if g.get("home_sp_id") is not None else None
        away_sp_id = int(g["away_sp_id"]) if g.get("away_sp_id") is not None else None
        home_sp_hand = str(g.get("home_sp_hand") or "R").upper() if home_sp_id is not None else None
        away_sp_hand = str(g.get("away_sp_hand") or "R").upper() if away_sp_id is not None else None

        sched_innings = int(g.get("scheduled_innings") or 9)
        innings_played = int(g.get("innings_played") or sched_innings) if is_final else None

        h_runs_raw = g.get("home_runs") if g.get("home_runs") is not None else g.get("home_score")
        a_runs_raw = g.get("away_runs") if g.get("away_runs") is not None else g.get("away_score")
        home_runs: int | None
        away_runs: int | None
        outcome_1x2: int | None
        home_win: int | None
        is_tie: int | None

        if is_final and h_runs_raw is not None and a_runs_raw is not None:
            h_val = int(h_runs_raw)
            a_val = int(a_runs_raw)
            home_runs = h_val
            away_runs = a_val
            if h_val > a_val:
                outcome_1x2 = 0
                home_win = 1
                is_tie = 0
            elif h_val < a_val:
                outcome_1x2 = 2
                home_win = 0
                is_tie = 0
            else:
                # NPB 12-inning tie rule
                outcome_1x2 = 1
                home_win = 0
                is_tie = 1
                if innings_played is None or innings_played < 12:
                    innings_played = int(g.get("innings_played") or 12)
        else:
            home_runs = None
            away_runs = None
            outcome_1x2 = None
            home_win = None
            is_tie = None

        rows.append(
            {
                "sport": "baseball",
                "league": "NPB",
                "sub_league": sub_league,
                "season": season_val,
                "game_id": game_id,
                "event_seq": event_seq,
                "game_date": game_date,
                "start_time_utc": start_utc,
                "home_team": home_team,
                "away_team": away_team,
                "venue_id": venue_id,
                "venue_name": venue_name,
                "venue_timezone": venue_tz,
                "is_dome_or_retractable": is_dome,
                "is_neutral_or_regional_venue": is_neutral,
                "uses_dh": uses_dh,
                "scheduled_innings": sched_innings,
                "innings_played": innings_played,
                "game_state": game_state,
                "home_sp_id": home_sp_id,
                "away_sp_id": away_sp_id,
                "home_sp_hand": home_sp_hand,
                "away_sp_hand": away_sp_hand,
                "home_runs": home_runs,
                "away_runs": away_runs,
                "outcome_1x2": outcome_1x2,
                "home_win": home_win,
                "is_tie": is_tie,
                "market_home_decimal_odds": g.get("market_home_decimal_odds"),
                "market_draw_decimal_odds": g.get("market_draw_decimal_odds"),
                "market_away_decimal_odds": g.get("market_away_decimal_odds"),
                "market_prob_home": g.get("market_prob_home"),
                "market_prob_draw": g.get("market_prob_draw"),
                "market_prob_away": g.get("market_prob_away"),
            }
        )

    if not rows:
        return pl.DataFrame(schema=SILVER_BASEBALL_GAMES_SCHEMA)
    return _cast_to_schema(pl.DataFrame(rows), SILVER_BASEBALL_GAMES_SCHEMA)


def parse_npb_boxscore_payload(
    payload: Mapping[str, Any],
    *,
    game_id: int,
    event_seq: int = 1,
    game_date: date,
    start_time_utc: datetime,
    season: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parse an NPB game boxscore payload into team, pitcher, and batter Silver DataFrames."""
    ns_game_id = namespace_npb_game_id(game_id)
    home_team = normalize_baseball_team(str(payload.get("home_team") or "YOM"))
    away_team = normalize_baseball_team(str(payload.get("away_team") or "HAN"))

    team_rows: list[dict[str, Any]] = []
    pitcher_rows: list[dict[str, Any]] = []
    batter_rows: list[dict[str, Any]] = []

    sides = [
        (True, home_team, away_team, payload.get("home") or {}),
        (False, away_team, home_team, payload.get("away") or {}),
    ]

    parsed_side_summaries: dict[bool, dict[str, Any]] = {}

    for is_home, team_code, opp_code, side_node in sides:
        if not isinstance(side_node, Mapping):
            side_node = {}
        pitchers_list = side_node.get("pitchers") or []
        batters_list = side_node.get("batters") or []

        bp_outs = 0
        bp_pitches = 0
        bp_er = 0
        total_outs_pitched = 0
        hits_allowed_sum = 0
        runs_allowed_sum = 0
        hr_allowed_sum = 0
        bb_allowed_sum = 0
        k_recorded_sum = 0

        for idx, p in enumerate(pitchers_list):
            if not isinstance(p, Mapping) or p.get("player_id") is None:
                continue
            pid = int(p["player_id"])
            pname = str(p.get("player_name") or f"NPB Pitcher {pid}")
            phand = str(p.get("pitch_hand") or "R").upper()
            is_starter = bool(p.get("is_starter", idx == 0))
            outs = (
                int(p["outs_pitched"])
                if p.get("outs_pitched") is not None
                else parse_baseball_ip_to_outs(p.get("ip") or p.get("innings_pitched") or "0.0")
            )
            true_ip = outs_to_true_innings(outs)
            bf = int(p.get("batters_faced") or max(outs, 0))
            pitches = int(p.get("pitches_thrown") or p.get("pitches") or 0)
            strikes = int(p.get("strikes_thrown") or int(round(pitches * 0.64)))
            h_all = int(p.get("hits_allowed") or p.get("h") or 0)
            r_all = int(p.get("runs_allowed") or p.get("r") or 0)
            er_all = int(p.get("er_allowed") or p.get("er") or 0)
            hr_all = int(p.get("hr_allowed") or p.get("hr") or 0)
            bb_all = int(p.get("bb_allowed") or p.get("bb") or 0)
            ibb_all = int(p.get("ibb_allowed") or p.get("ibb") or 0)
            hbp_all = int(p.get("hbp_allowed") or p.get("hbp") or 0)
            k_rec = int(p.get("k_recorded") or p.get("k") or p.get("so") or 0)
            gb_cnt = int(p.get("gb_count") or 0)
            fb_cnt = int(p.get("fb_count") or 0)
            ld_cnt = int(p.get("ld_count") or 0)

            total_outs_pitched += outs
            hits_allowed_sum += h_all
            runs_allowed_sum += r_all
            hr_allowed_sum += hr_all
            bb_allowed_sum += bb_all
            k_recorded_sum += k_rec
            if not is_starter:
                bp_outs += outs
                bp_pitches += pitches
                bp_er += er_all

            qs = 1 if (is_starter and outs >= 18 and er_all <= 3) else 0
            pitcher_rows.append(
                {
                    "game_id": ns_game_id,
                    "event_seq": event_seq,
                    "game_date": game_date,
                    "start_time_utc": start_time_utc,
                    "season": season,
                    "league": "NPB",
                    "player_id": pid,
                    "player_name": pname,
                    "team": team_code,
                    "opponent": opp_code,
                    "is_home": is_home,
                    "pitch_hand": phand if phand in {"L", "R"} else "R",
                    "is_starter": is_starter,
                    "outs_pitched": outs,
                    "innings_pitched": true_ip,
                    "batters_faced": bf,
                    "pitches_thrown": pitches,
                    "strikes_thrown": strikes,
                    "hits_allowed": h_all,
                    "runs_allowed": r_all,
                    "er_allowed": er_all,
                    "hr_allowed": hr_all,
                    "bb_allowed": bb_all,
                    "ibb_allowed": ibb_all,
                    "hbp_allowed": hbp_all,
                    "k_recorded": k_rec,
                    "gb_count": gb_cnt,
                    "fb_count": fb_cnt,
                    "ld_count": ld_cnt,
                    "quality_start": qs,
                    "xwoba_against": p.get("xwoba_against"),
                    "barrel_pct_against": p.get("barrel_pct_against"),
                    "swstr_pct": p.get("swstr_pct"),
                    "avg_fastball_velo": p.get("avg_fastball_velo"),
                }
            )

        runs_for = 0
        hits_for = 0
        doubles_for = 0
        triples_for = 0
        hr_for = 0
        bb_for = 0
        ibb_for = 0
        hbp_for = 0
        k_for = 0
        sf_for = 0
        pa_for = 0
        ab_for = 0

        for idx, b in enumerate(batters_list):
            if not isinstance(b, Mapping) or b.get("player_id") is None:
                continue
            bid = int(b["player_id"])
            bname = str(b.get("player_name") or f"NPB Batter {bid}")
            bside = str(b.get("bat_side") or "R").upper()
            slot = int(b.get("batting_order_slot") or (idx + 1 if idx < 9 else 0))
            is_starter = bool(b.get("is_starter", 1 <= slot <= 9))
            pos_grp = str(b.get("position_group") or "INF").upper()

            ab = int(b.get("ab") or 0)
            hits = int(b.get("hits") or b.get("h") or 0)
            doubles = int(b.get("doubles") or b.get("2b") or 0)
            triples = int(b.get("triples") or b.get("3b") or 0)
            hr = int(b.get("hr") or 0)
            singles = max(0, hits - doubles - triples - hr)
            tb = int(b.get("total_bases") or (singles + 2 * doubles + 3 * triples + 4 * hr))
            runs = int(b.get("runs") or b.get("r") or 0)
            rbi = int(b.get("rbi") or 0)
            bb = int(b.get("bb") or 0)
            ibb = int(b.get("ibb") or 0)
            hbp = int(b.get("hbp") or 0)
            k = int(b.get("k") or b.get("so") or 0)
            sf = int(b.get("sf") or 0)
            sb = int(b.get("sb") or 0)
            cs = int(b.get("cs") or 0)
            pa = int(b.get("pa") or (ab + bb + hbp + sf))

            runs_for += runs
            hits_for += hits
            doubles_for += doubles
            triples_for += triples
            hr_for += hr
            bb_for += bb
            ibb_for += ibb
            hbp_for += hbp
            k_for += k
            sf_for += sf
            pa_for += pa
            ab_for += ab

            woba_num = _compute_npb_woba_num(
                singles=singles,
                doubles=doubles,
                triples=triples,
                hr=hr,
                bb=bb,
                ibb=ibb,
                hbp=hbp,
            )

            batter_rows.append(
                {
                    "game_id": ns_game_id,
                    "event_seq": event_seq,
                    "game_date": game_date,
                    "start_time_utc": start_time_utc,
                    "season": season,
                    "league": "NPB",
                    "player_id": bid,
                    "player_name": bname,
                    "team": team_code,
                    "opponent": opp_code,
                    "is_home": is_home,
                    "bat_side": bside if bside in {"L", "R", "S"} else "R",
                    "batting_order_slot": slot,
                    "position_group": pos_grp if pos_grp in {"C", "INF", "OF", "DH"} else "INF",
                    "is_starter": is_starter,
                    "pa": pa,
                    "ab": ab,
                    "hits": hits,
                    "singles": singles,
                    "doubles": doubles,
                    "triples": triples,
                    "hr": hr,
                    "total_bases": tb,
                    "runs": runs,
                    "rbi": rbi,
                    "bb": bb,
                    "ibb": ibb,
                    "hbp": hbp,
                    "k": k,
                    "sf": sf,
                    "sb": sb,
                    "cs": cs,
                    "woba_num": woba_num,
                    "xwoba": b.get("xwoba"),
                    "hard_hit_rate": b.get("hard_hit_rate"),
                    "barrel_rate": b.get("barrel_rate"),
                }
            )

        singles_for = max(0, hits_for - doubles_for - triples_for - hr_for)
        woba_num_team = _compute_npb_woba_num(
            singles=singles_for,
            doubles=doubles_for,
            triples=triples_for,
            hr=hr_for,
            bb=bb_for,
            ibb=ibb_for,
            hbp=hbp_for,
        )
        woba_denom = max(1, ab_for + bb_for - ibb_for + sf_for + hbp_for)
        team_woba = woba_num_team / float(woba_denom)

        side_runs_raw = side_node.get("runs")
        parsed_side_summaries[is_home] = {
            "team_code": team_code,
            "opp_code": opp_code,
            "runs_for": int(side_runs_raw) if side_runs_raw is not None else runs_for,
            "hits_for": hits_for,
            "doubles_for": doubles_for,
            "triples_for": triples_for,
            "hr_for": hr_for,
            "bb_for": bb_for,
            "k_for": k_for,
            "hbp_for": hbp_for,
            "sf_for": sf_for,
            "pa_for": pa_for,
            "outs_pitched": total_outs_pitched if total_outs_pitched > 0 else 27,
            "lob_for": int(side_node.get("lob") or 0),
            "errors_committed": int(side_node.get("errors") or 0),
            "bullpen_outs_pitched": bp_outs,
            "bullpen_pitches": bp_pitches,
            "bullpen_er": bp_er,
            "team_woba": team_woba,
            "hits_against": hits_allowed_sum,
            "runs_against": runs_allowed_sum,
            "hr_against": hr_allowed_sum,
            "bb_against": bb_allowed_sum,
            "k_against": k_recorded_sum,
        }

    for is_home in (True, False):
        mine = parsed_side_summaries[is_home]
        theirs = parsed_side_summaries[not is_home]
        team_rows.append(
            {
                "game_id": ns_game_id,
                "event_seq": event_seq,
                "game_date": game_date,
                "start_time_utc": start_time_utc,
                "season": season,
                "league": "NPB",
                "team": mine["team_code"],
                "opponent": mine["opp_code"],
                "is_home": is_home,
                "runs_for": mine["runs_for"],
                "runs_against": theirs["runs_for"],
                "hits_for": mine["hits_for"],
                "hits_against": theirs["hits_for"],
                "doubles_for": mine["doubles_for"],
                "triples_for": mine["triples_for"],
                "hr_for": mine["hr_for"],
                "hr_against": theirs["hr_for"],
                "bb_for": mine["bb_for"],
                "bb_against": theirs["bb_for"],
                "k_for": mine["k_for"],
                "k_against": theirs["k_for"],
                "hbp_for": mine["hbp_for"],
                "sf_for": mine["sf_for"],
                "pa_for": mine["pa_for"],
                "outs_pitched": mine["outs_pitched"],
                "lob_for": mine["lob_for"],
                "errors_committed": mine["errors_committed"],
                "bullpen_outs_pitched": mine["bullpen_outs_pitched"],
                "bullpen_pitches": mine["bullpen_pitches"],
                "bullpen_er": mine["bullpen_er"],
                "team_woba": mine["team_woba"],
                "opp_woba": theirs["team_woba"],
                "team_xwoba": None,
                "opp_xwoba": None,
                "has_statcast_metrics": False,
            }
        )

    return (
        _cast_to_schema(pl.DataFrame(team_rows), SILVER_BASEBALL_TEAM_LOGS_SCHEMA),
        _cast_to_schema(pl.DataFrame(pitcher_rows), SILVER_BASEBALL_PITCHER_LOGS_SCHEMA),
        _cast_to_schema(pl.DataFrame(batter_rows), SILVER_BASEBALL_BATTER_LOGS_SCHEMA),
    )


def parse_npb_roster_payload(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    team: str | None = None,
) -> pl.DataFrame:
    """Parse NPB active first-team (`ichi-gun`) registrations into a roster DataFrame."""
    entries: Sequence[Mapping[str, Any]] = (
        payload.get("roster") or [] if isinstance(payload, Mapping) else payload
    )
    rows: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, Mapping) or item.get("player_id") is None:
            continue
        pid = int(item["player_id"])
        pname = str(item.get("player_name") or f"NPB Player {pid}")
        raw_team = str(item.get("team") or team or "YOM")
        norm_team = normalize_baseball_team(raw_team)
        pos_grp = str(item.get("position_group") or "INF").upper()
        is_pitcher = bool(item.get("is_pitcher", pos_grp == "P"))
        bat_side = str(item.get("bat_side") or "R").upper()
        pitch_hand = str(item.get("pitch_hand") or "R").upper()
        status_code = str(item.get("status_code") or "ICHI_GUN").upper()
        is_available = status_code in {"ICHI_GUN", "ACTIVE", "A", "1ST"}

        rows.append(
            {
                "player_id": pid,
                "player_name": pname,
                "team": norm_team,
                "league": "NPB",
                "position_group": "P" if is_pitcher else pos_grp,
                "is_pitcher": is_pitcher,
                "bat_side": bat_side if bat_side in {"L", "R", "S"} else "R",
                "pitch_hand": pitch_hand if pitch_hand in {"L", "R"} else "R",
                "status_code": status_code,
                "is_available": is_available,
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "player_id": pl.Int64,
                "player_name": pl.Utf8,
                "team": pl.Utf8,
                "league": pl.Utf8,
                "position_group": pl.Utf8,
                "is_pitcher": pl.Boolean,
                "bat_side": pl.Utf8,
                "pitch_hand": pl.Utf8,
                "status_code": pl.Utf8,
                "is_available": pl.Boolean,
            }
        )
    return pl.DataFrame(rows)


class NPBClient:
    """Cached HTTP client for NPB schedules, 12-inning boxscores, and `ichi-gun` rosters."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        base_url: str = NPB_BIS_BASE_URL,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        (self.cache_dir / "schedules").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "boxscores").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "rosters").mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    def _get_json(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return dict(data) if isinstance(data, Mapping) else {"games": data}

    def fetch_season_schedule(
        self,
        season: int,
        *,
        ttl_seconds: float | None = 3600.0,
    ) -> dict[str, Any]:
        """Fetch and `.json.gz`-cache NPB season schedule."""
        cache_path = self.cache_dir / "schedules" / f"npb_schedule_{int(season)}.json.gz"
        cached = read_cached_json(cache_path, ttl_seconds=ttl_seconds)
        if isinstance(cached, dict):
            return cached

        payload = self._get_json(f"{int(season)}/games/schedule.json")
        write_cached_json(cache_path, payload)
        return payload

    def fetch_boxscore(
        self,
        game_id: int,
        *,
        is_final: bool = True,
    ) -> dict[str, Any]:
        """Fetch NPB game boxscore; permanently cached once `is_final=True`."""
        ns_id = namespace_npb_game_id(game_id)
        cache_path = self.cache_dir / "boxscores" / f"{ns_id}.json.gz"
        ttl = None if is_final else 900.0
        cached = read_cached_json(cache_path, ttl_seconds=ttl)
        if isinstance(cached, dict):
            return cached

        payload = self._get_json(f"games/boxscore/{ns_id}.json")
        write_cached_json(cache_path, payload)
        return payload

    def fetch_active_rosters(
        self,
        season: int,
        *,
        ttl_seconds: float | None = 3600.0,
    ) -> dict[str, Any]:
        """Fetch NPB active `ichi-gun` (first-team) registrations."""
        cache_path = self.cache_dir / "rosters" / f"npb_rosters_{int(season)}.json.gz"
        cached = read_cached_json(cache_path, ttl_seconds=ttl_seconds)
        if isinstance(cached, dict):
            return cached

        payload = self._get_json(f"{int(season)}/rosters/active.json")
        write_cached_json(cache_path, payload)
        return payload
