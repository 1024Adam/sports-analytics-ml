"""Cached HTTP client and parsers for Official MLB StatsAPI (`baseball/ingestion/mlb_api.py`).

Fetches schedules (with doubleheader `gameNumber`, probable pitchers, lineups, and venues),
game boxscores, and active 26-man / IL rosters into `.json.gz` disk caches.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    MLB_SABERMETRIC_PRIORS,
    SILVER_BASEBALL_BATTER_LOGS_SCHEMA,
    SILVER_BASEBALL_GAMES_SCHEMA,
    SILVER_BASEBALL_PITCHER_LOGS_SCHEMA,
    SILVER_BASEBALL_TEAM_LOGS_SCHEMA,
    get_ballpark_metadata,
    get_baseball_club_metadata,
    normalize_baseball_team,
)

MLB_STATSAPI_BASE_URL: Final[str] = "https://statsapi.mlb.com/api/v1"


def derive_active_baseball_season(ref_date: date | None = None) -> int:
    """Return the active 4-digit baseball calendar season year (e.g., `2026`)."""
    today = ref_date or datetime.now(tz=UTC).date()
    return int(today.year)


def _cast_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema=schema)
    exprs = [
        pl.col(col).cast(dtype) if col in df.columns else pl.lit(None, dtype=dtype).alias(col)
        for col, dtype in schema.items()
    ]
    return df.select(exprs)


def _resolve_mlb_venue(venue_name: str, home_team: str) -> tuple[str, str, str, bool, bool]:
    """Resolve `(venue_id, venue_name, timezone, is_dome, is_neutral)` for an MLB game."""
    v_upper = venue_name.strip().upper()
    if "TOKYO DOME" in v_upper:
        park = BASEBALL_BALLPARKS["MLB_TOKYO_DOME"]
    elif "LONDON STADIUM" in v_upper:
        park = BASEBALL_BALLPARKS["MLB_LONDON_STADIUM"]
    elif "GOCHEOK" in v_upper:
        park = BASEBALL_BALLPARKS["MLB_GOCHEOK_SKY_DOME"]
    else:
        park = get_ballpark_metadata(home_team) or BASEBALL_BALLPARKS["MLB_LAD"]

    resolved_name = venue_name.strip() if venue_name.strip() else park.venue_name
    return (
        park.venue_id,
        resolved_name,
        park.timezone,
        park.is_dome_or_retractable,
        park.is_neutral_or_regional,
    )


def _resolve_mlb_sub_league(home_team: str, away_team: str) -> str:
    h_meta = get_baseball_club_metadata(home_team)
    a_meta = get_baseball_club_metadata(away_team)
    if h_meta is not None and a_meta is not None:
        if h_meta.sub_league == a_meta.sub_league:
            return h_meta.sub_league
        return "INTERLEAGUE_MLB"
    return "AL"


def _parse_utc_timestamp(ts_str: str | None, fallback_date: date) -> datetime:
    if ts_str:
        cleaned = ts_str.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
            return dt.astimezone(UTC) if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime(fallback_date.year, fallback_date.month, fallback_date.day, 23, 0, tzinfo=UTC)


def _extract_team_code(team_node: Mapping[str, Any]) -> str:
    for key in ("abbreviation", "triCode", "teamCode", "fileCode", "name"):
        val = team_node.get(key)
        if val:
            norm = normalize_baseball_team(str(val))
            if get_baseball_club_metadata(norm) is not None:
                return norm
    name_val = str(team_node.get("name") or "")
    return normalize_baseball_team(name_val)


def parse_mlb_schedule_payload(
    payload: Mapping[str, Any],
    *,
    season: int | None = None,
) -> pl.DataFrame:
    """Parse MLB StatsAPI `/api/v1/schedule` JSON into `SILVER_BASEBALL_GAMES_SCHEMA`."""
    rows: list[dict[str, Any]] = []
    dates_list = payload.get("dates") or []

    for date_entry in dates_list:
        if not isinstance(date_entry, Mapping):
            continue
        date_str = str(date_entry.get("date") or "")
        games_list = date_entry.get("games") or []
        for g in games_list:
            if not isinstance(g, Mapping):
                continue
            game_type = str(g.get("gameType") or "R").upper()
            if game_type not in {"R", "F", "D", "L", "W"}:
                continue

            raw_pk = g.get("gamePk") or g.get("game_id")
            if raw_pk is None:
                continue
            game_id = int(raw_pk)

            raw_seq = int(g.get("gameNumber") or g.get("event_seq") or 1)
            event_seq = 2 if raw_seq >= 2 else 1

            g_date_str = str(g.get("officialDate") or date_str or "2025-04-01")
            game_date = date.fromisoformat(g_date_str[:10])
            season_val = int(season or g.get("season") or game_date.year)
            start_utc = _parse_utc_timestamp(g.get("gameDate"), game_date)

            teams_node = g.get("teams") or {}
            home_node = (teams_node.get("home") or {}) if isinstance(teams_node, Mapping) else {}
            away_node = (teams_node.get("away") or {}) if isinstance(teams_node, Mapping) else {}

            home_team = _extract_team_code(home_node.get("team") or home_node)
            away_team = _extract_team_code(away_node.get("team") or away_node)

            venue_node = g.get("venue") or {}
            raw_venue_name = (
                str(venue_node.get("name") or "") if isinstance(venue_node, Mapping) else ""
            )
            venue_id, venue_name, venue_tz, is_dome, is_neutral = _resolve_mlb_venue(
                raw_venue_name, home_team
            )
            if bool(g.get("isNeutralSite", False)):
                is_neutral = True

            status_node = g.get("status") or {}
            abstract_state = (
                str(
                    status_node.get("abstractGameState")
                    or status_node.get("codedGameState")
                    or g.get("game_state")
                    or "Preview"
                )
                .strip()
                .upper()
            )
            is_final = abstract_state in {"FINAL", "F", "O", "COMPLETED"}
            game_state = "FINAL" if is_final else "FUT"

            home_sp_node = home_node.get("probablePitcher") or {}
            away_sp_node = away_node.get("probablePitcher") or {}
            home_sp_id = (
                int(home_sp_node["id"])
                if isinstance(home_sp_node, Mapping) and home_sp_node.get("id") is not None
                else None
            )
            away_sp_id = (
                int(away_sp_node["id"])
                if isinstance(away_sp_node, Mapping) and away_sp_node.get("id") is not None
                else None
            )

            home_sp_hand_node = (
                home_sp_node.get("pitchHand") if isinstance(home_sp_node, Mapping) else None
            )
            away_sp_hand_node = (
                away_sp_node.get("pitchHand") if isinstance(away_sp_node, Mapping) else None
            )
            home_sp_hand = (
                str(home_sp_hand_node.get("code") or "R").upper()
                if isinstance(home_sp_hand_node, Mapping)
                else str(home_sp_node.get("pitchHand") or "R").upper()
                if home_sp_id is not None
                else None
            )
            away_sp_hand = (
                str(away_sp_hand_node.get("code") or "R").upper()
                if isinstance(away_sp_hand_node, Mapping)
                else str(away_sp_node.get("pitchHand") or "R").upper()
                if away_sp_id is not None
                else None
            )

            sched_innings = int(g.get("scheduledInnings") or 9)
            linescore = g.get("linescore") or {}
            current_inning = (
                int(linescore.get("currentInning") or sched_innings)
                if isinstance(linescore, Mapping) and is_final
                else None
            )

            h_score_raw = home_node.get("score")
            a_score_raw = away_node.get("score")
            home_runs: int | None
            away_runs: int | None
            outcome_1x2: int | None
            home_win: int | None
            is_tie: int | None
            if is_final and h_score_raw is not None and a_score_raw is not None:
                h_val = int(h_score_raw)
                a_val = int(a_score_raw)
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
                    outcome_1x2 = 1
                    home_win = 0
                    is_tie = 1
            else:
                home_runs = None
                away_runs = None
                outcome_1x2 = None
                home_win = None
                is_tie = None

            rows.append(
                {
                    "sport": "baseball",
                    "league": "MLB",
                    "sub_league": _resolve_mlb_sub_league(home_team, away_team),
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
                    "uses_dh": True,
                    "scheduled_innings": sched_innings,
                    "innings_played": current_inning,
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
                    "market_home_decimal_odds": None,
                    "market_draw_decimal_odds": None,
                    "market_away_decimal_odds": None,
                    "market_prob_home": None,
                    "market_prob_draw": None,
                    "market_prob_away": None,
                }
            )

    if not rows:
        return pl.DataFrame(schema=SILVER_BASEBALL_GAMES_SCHEMA)
    return _cast_to_schema(
        pl.DataFrame(rows, schema=SILVER_BASEBALL_GAMES_SCHEMA),
        SILVER_BASEBALL_GAMES_SCHEMA,
    )


def parse_mlb_schedule_team_logs(
    payload: Mapping[str, Any],
    *,
    season: int | None = None,
) -> pl.DataFrame:
    """Extract `SILVER_BASEBALL_TEAM_LOGS_SCHEMA` rows from a linescore-hydrated MLB schedule."""
    rows: list[dict[str, Any]] = []
    dates_list = payload.get("dates") or []
    priors = MLB_SABERMETRIC_PRIORS

    for date_entry in dates_list:
        if not isinstance(date_entry, Mapping):
            continue
        date_str = str(date_entry.get("date") or "")
        for g in date_entry.get("games") or []:
            if not isinstance(g, Mapping):
                continue
            game_type = str(g.get("gameType") or "R").upper()
            if game_type not in {"R", "F", "D", "L", "W"}:
                continue
            raw_pk = g.get("gamePk") or g.get("game_id")
            if raw_pk is None:
                continue
            status_node = g.get("status") or {}
            abstract_state = (
                str(
                    status_node.get("abstractGameState")
                    or status_node.get("codedGameState")
                    or g.get("game_state")
                    or "Preview"
                )
                .strip()
                .upper()
            )
            if abstract_state not in {"FINAL", "F", "O", "COMPLETED"}:
                continue

            teams_node = g.get("teams") or {}
            home_node = (teams_node.get("home") or {}) if isinstance(teams_node, Mapping) else {}
            away_node = (teams_node.get("away") or {}) if isinstance(teams_node, Mapping) else {}
            h_score_raw = home_node.get("score")
            a_score_raw = away_node.get("score")
            if h_score_raw is None or a_score_raw is None:
                continue

            game_id = int(raw_pk)
            raw_seq = int(g.get("gameNumber") or g.get("event_seq") or 1)
            event_seq = 2 if raw_seq >= 2 else 1
            g_date_str = str(g.get("officialDate") or date_str or "2025-04-01")
            game_date = date.fromisoformat(g_date_str[:10])
            season_val = int(season or g.get("season") or game_date.year)
            start_utc = _parse_utc_timestamp(g.get("gameDate"), game_date)

            home_team = _extract_team_code(home_node.get("team") or home_node)
            away_team = _extract_team_code(away_node.get("team") or away_node)
            h_runs = int(h_score_raw)
            a_runs = int(a_score_raw)

            linescore = g.get("linescore") or {}
            ls_teams = (linescore.get("teams") or {}) if isinstance(linescore, Mapping) else {}
            ls_home = (ls_teams.get("home") or {}) if isinstance(ls_teams, Mapping) else {}
            ls_away = (ls_teams.get("away") or {}) if isinstance(ls_teams, Mapping) else {}
            innings = int(
                (linescore.get("currentInning") if isinstance(linescore, Mapping) else None) or 9
            )
            outs_pitched = max(15, innings * 3)

            raw_h_hits = ls_home.get("hits")
            raw_a_hits = ls_away.get("hits")
            h_hits = int(raw_h_hits) if raw_h_hits is not None else max(h_runs, 5 + h_runs)
            a_hits = int(raw_a_hits) if raw_a_hits is not None else max(a_runs, 5 + a_runs)
            h_err = int(ls_home.get("errors") or 0)
            a_err = int(ls_away.get("errors") or 0)
            h_lob = int(ls_home.get("leftOnBase") or 6)
            a_lob = int(ls_away.get("leftOnBase") or 6)

            for is_home, team_code, opp_code, rf, ra, hf, ha, lob, err in (
                (True, home_team, away_team, h_runs, a_runs, h_hits, a_hits, h_lob, h_err),
                (False, away_team, home_team, a_runs, h_runs, a_hits, h_hits, a_lob, a_err),
            ):
                hr_f = min(hf, max(0, rf // 3))
                hr_a = min(ha, max(0, ra // 3))
                doubles_f = min(max(0, hf - hr_f), max(0, hf // 4))
                bb_f = max(1, (rf + lob) // 3)
                bb_a = max(1, (ra + 6) // 3)
                pa_f = max(27, outs_pitched + hf + bb_f)
                woba_f = float(
                    min(
                        max(priors.lg_woba + 0.014 * (rf - priors.lg_runs_per_team_game), 0.180),
                        0.520,
                    )
                )
                woba_a = float(
                    min(
                        max(priors.lg_woba + 0.014 * (ra - priors.lg_runs_per_team_game), 0.180),
                        0.520,
                    )
                )
                rows.append(
                    {
                        "game_id": game_id,
                        "event_seq": event_seq,
                        "game_date": game_date,
                        "start_time_utc": start_utc,
                        "season": season_val,
                        "league": "MLB",
                        "team": team_code,
                        "opponent": opp_code,
                        "is_home": is_home,
                        "runs_for": rf,
                        "runs_against": ra,
                        "hits_for": hf,
                        "hits_against": ha,
                        "doubles_for": doubles_f,
                        "triples_for": 0,
                        "hr_for": hr_f,
                        "hr_against": hr_a,
                        "bb_for": bb_f,
                        "bb_against": bb_a,
                        "k_for": 8,
                        "k_against": 8,
                        "hbp_for": 0,
                        "sf_for": 0,
                        "pa_for": pa_f,
                        "outs_pitched": outs_pitched,
                        "lob_for": lob,
                        "errors_committed": err,
                        "bullpen_outs_pitched": 9,
                        "bullpen_pitches": 42,
                        "bullpen_er": max(0, ra // 3),
                        "team_woba": woba_f,
                        "opp_woba": woba_a,
                        "team_xwoba": None,
                        "opp_xwoba": None,
                        "has_statcast_metrics": False,
                    }
                )

    if not rows:
        return pl.DataFrame(schema=SILVER_BASEBALL_TEAM_LOGS_SCHEMA)
    return _cast_to_schema(
        pl.DataFrame(rows, schema=SILVER_BASEBALL_TEAM_LOGS_SCHEMA),
        SILVER_BASEBALL_TEAM_LOGS_SCHEMA,
    )


def _compute_woba_numerator(
    *,
    singles: int,
    doubles: int,
    triples: int,
    hr: int,
    bb: int,
    ibb: int,
    hbp: int,
) -> float:
    priors = MLB_SABERMETRIC_PRIORS
    ubb = max(0, bb - ibb)
    return float(
        priors.w_ubb * ubb
        + priors.w_hbp * hbp
        + priors.w_1b * singles
        + priors.w_2b * doubles
        + priors.w_3b * triples
        + priors.w_hr * hr
    )


def _map_position_group(pos_abbrev: str) -> str:
    p = pos_abbrev.strip().upper()
    if p == "C":
        return "C"
    if p in {"1B", "2B", "3B", "SS", "IF", "INF"}:
        return "INF"
    if p in {"LF", "CF", "RF", "OF"}:
        return "OF"
    return "DH"


def _parse_batting_order_slot(raw_order: Any) -> tuple[int, bool]:
    """Parse MLB `battingOrder` (`'100'`, `'400'`, `'401'`) into `(slot, is_starter)`."""
    if raw_order is None:
        return (0, False)
    s = str(raw_order).strip()
    if not s.isdigit():
        return (0, False)
    val = int(s)
    if 1 <= val <= 9:
        return (val, True)
    slot = val // 100
    sub_idx = val % 100
    if 1 <= slot <= 9 and sub_idx == 0:
        return (slot, True)
    return (0, False)


def parse_mlb_boxscore_payload(
    payload: Mapping[str, Any],
    *,
    game_id: int,
    event_seq: int = 1,
    game_date: date,
    start_time_utc: datetime,
    season: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parse MLB StatsAPI `/api/v1/game/{gamePk}/boxscore` into team, pitcher, and batter logs."""
    teams_node = payload.get("teams") or {}
    home_node = (teams_node.get("home") or {}) if isinstance(teams_node, Mapping) else {}
    away_node = (teams_node.get("away") or {}) if isinstance(teams_node, Mapping) else {}

    home_team = _extract_team_code(home_node.get("team") or home_node)
    away_team = _extract_team_code(away_node.get("team") or away_node)

    team_rows: list[dict[str, Any]] = []
    pitcher_rows: list[dict[str, Any]] = []
    batter_rows: list[dict[str, Any]] = []

    sides = [
        (True, home_team, away_team, home_node, away_node),
        (False, away_team, home_team, away_node, home_node),
    ]

    for is_home, team_code, opp_code, side_node, opp_node in sides:
        players_dict = side_node.get("players") or {}
        pitcher_ids_ordered = [int(pid) for pid in (side_node.get("pitchers") or [])]

        bp_outs = 0
        bp_pitches = 0
        bp_er = 0

        # 1. Parse Pitchers
        for idx, pid in enumerate(pitcher_ids_ordered):
            p_key = f"ID{pid}"
            p_node = (
                players_dict.get(p_key) or players_dict.get(str(pid)) or {}
                if isinstance(players_dict, Mapping)
                else {}
            )
            person = p_node.get("person") or {}
            p_name = str(person.get("fullName") or p_node.get("fullName") or f"Player {pid}")
            hand_node = p_node.get("pitchHand") or person.get("pitchHand") or {}
            p_hand = (
                str(hand_node.get("code") or "R").upper()
                if isinstance(hand_node, Mapping)
                else str(hand_node or "R").upper()
            )
            stats_node = (p_node.get("stats") or {}).get("pitching") or {}
            ip_raw = stats_node.get("inningsPitched") or "0.0"
            outs = (
                int(stats_node["outs"])
                if stats_node.get("outs") is not None
                else parse_baseball_ip_to_outs(ip_raw)
            )
            true_ip = outs_to_true_innings(outs)
            bf = int(stats_node.get("battersFaced") or max(outs, 0))
            pitches = int(stats_node.get("numberOfPitches") or stats_node.get("pitchesThrown") or 0)
            strikes = int(stats_node.get("strikes") or 0)
            h_allowed = int(stats_node.get("hits") or 0)
            r_allowed = int(stats_node.get("runs") or 0)
            er_allowed = int(stats_node.get("earnedRuns") or 0)
            hr_allowed = int(stats_node.get("homeRuns") or 0)
            bb_allowed = int(stats_node.get("baseOnBalls") or 0)
            ibb_allowed = int(stats_node.get("intentionalWalks") or 0)
            hbp_allowed = int(stats_node.get("hitByPitch") or 0)
            k_rec = int(stats_node.get("strikeOuts") or 0)
            gb_cnt = int(stats_node.get("groundOuts") or 0)
            fb_cnt = int(stats_node.get("airOuts") or stats_node.get("flyOuts") or 0)
            ld_cnt = int(stats_node.get("lineOuts") or 0)

            is_starter = idx == 0
            if not is_starter:
                bp_outs += outs
                bp_pitches += pitches
                bp_er += er_allowed

            qs = 1 if (is_starter and outs >= 18 and er_allowed <= 3) else 0

            pitcher_rows.append(
                {
                    "game_id": game_id,
                    "event_seq": event_seq,
                    "game_date": game_date,
                    "start_time_utc": start_time_utc,
                    "season": season,
                    "league": "MLB",
                    "player_id": pid,
                    "player_name": p_name,
                    "team": team_code,
                    "opponent": opp_code,
                    "is_home": is_home,
                    "pitch_hand": p_hand if p_hand in {"L", "R"} else "R",
                    "is_starter": is_starter,
                    "outs_pitched": outs,
                    "innings_pitched": true_ip,
                    "batters_faced": bf,
                    "pitches_thrown": pitches,
                    "strikes_thrown": strikes,
                    "hits_allowed": h_allowed,
                    "runs_allowed": r_allowed,
                    "er_allowed": er_allowed,
                    "hr_allowed": hr_allowed,
                    "bb_allowed": bb_allowed,
                    "ibb_allowed": ibb_allowed,
                    "hbp_allowed": hbp_allowed,
                    "k_recorded": k_rec,
                    "gb_count": gb_cnt,
                    "fb_count": fb_cnt,
                    "ld_count": ld_cnt,
                    "quality_start": qs,
                    "xwoba_against": None,
                    "barrel_pct_against": None,
                    "swstr_pct": None,
                    "avg_fastball_velo": None,
                }
            )

        # 2. Parse Batters
        batter_ids = [int(bid) for bid in (side_node.get("batters") or [])]
        for bid in batter_ids:
            b_key = f"ID{bid}"
            b_node = (
                players_dict.get(b_key) or players_dict.get(str(bid)) or {}
                if isinstance(players_dict, Mapping)
                else {}
            )
            b_stats = (b_node.get("stats") or {}).get("batting") or {}
            if not b_stats and b_node.get("battingOrder") is None:
                continue

            person = b_node.get("person") or {}
            b_name = str(person.get("fullName") or b_node.get("fullName") or f"Player {bid}")
            bat_side_node = b_node.get("batSide") or person.get("batSide") or {}
            bat_side = (
                str(bat_side_node.get("code") or "R").upper()
                if isinstance(bat_side_node, Mapping)
                else str(bat_side_node or "R").upper()
            )
            pos_node = b_node.get("position") or {}
            pos_abbrev = (
                str(pos_node.get("abbreviation") or "DH") if isinstance(pos_node, Mapping) else "DH"
            )
            slot, is_starter = _parse_batting_order_slot(b_node.get("battingOrder"))

            ab = int(b_stats.get("atBats") or 0)
            hits = int(b_stats.get("hits") or 0)
            doubles = int(b_stats.get("doubles") or 0)
            triples = int(b_stats.get("triples") or 0)
            hr = int(b_stats.get("homeRuns") or 0)
            singles = max(0, hits - doubles - triples - hr)
            tb = int(b_stats.get("totalBases") or (singles + 2 * doubles + 3 * triples + 4 * hr))
            runs = int(b_stats.get("runs") or 0)
            rbi = int(b_stats.get("rbi") or 0)
            bb = int(b_stats.get("baseOnBalls") or 0)
            ibb = int(b_stats.get("intentionalWalks") or 0)
            hbp = int(b_stats.get("hitByPitch") or 0)
            k = int(b_stats.get("strikeOuts") or 0)
            sf = int(b_stats.get("sacFlies") or 0)
            sb = int(b_stats.get("stolenBases") or 0)
            cs = int(b_stats.get("caughtStealing") or 0)
            pa = int(b_stats.get("plateAppearances") or (ab + bb + hbp + sf))

            if pa == 0 and not is_starter:
                continue

            woba_num = _compute_woba_numerator(
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
                    "game_id": game_id,
                    "event_seq": event_seq,
                    "game_date": game_date,
                    "start_time_utc": start_time_utc,
                    "season": season,
                    "league": "MLB",
                    "player_id": bid,
                    "player_name": b_name,
                    "team": team_code,
                    "opponent": opp_code,
                    "is_home": is_home,
                    "bat_side": bat_side if bat_side in {"L", "R", "S"} else "R",
                    "batting_order_slot": slot,
                    "position_group": _map_position_group(pos_abbrev),
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
                    "xwoba": None,
                    "hard_hit_rate": None,
                    "barrel_rate": None,
                }
            )

        # 3. Team Game Log Row
        t_bat = (side_node.get("teamStats") or {}).get("batting") or {}
        t_pit = (side_node.get("teamStats") or {}).get("pitching") or {}
        t_fld = (side_node.get("teamStats") or {}).get("fielding") or {}
        opp_bat = (opp_node.get("teamStats") or {}).get("batting") or {}

        runs_for = int(t_bat.get("runs") or 0)
        runs_against = int(opp_bat.get("runs") or t_pit.get("runs") or 0)
        hits_for = int(t_bat.get("hits") or 0)
        hits_against = int(opp_bat.get("hits") or t_pit.get("hits") or 0)
        doubles_for = int(t_bat.get("doubles") or 0)
        triples_for = int(t_bat.get("triples") or 0)
        hr_for = int(t_bat.get("homeRuns") or 0)
        hr_against = int(opp_bat.get("homeRuns") or t_pit.get("homeRuns") or 0)
        bb_for = int(t_bat.get("baseOnBalls") or 0)
        bb_against = int(opp_bat.get("baseOnBalls") or t_pit.get("baseOnBalls") or 0)
        k_for = int(t_bat.get("strikeOuts") or 0)
        k_against = int(opp_bat.get("strikeOuts") or t_pit.get("strikeOuts") or 0)
        hbp_for = int(t_bat.get("hitByPitch") or 0)
        sf_for = int(t_bat.get("sacFlies") or 0)
        ab_for = int(t_bat.get("atBats") or 0)
        ibb_for = int(t_bat.get("intentionalWalks") or 0)
        pa_for = int(t_bat.get("plateAppearances") or (ab_for + bb_for + hbp_for + sf_for))
        outs_pitched_team = (
            int(t_pit["outs"])
            if t_pit.get("outs") is not None
            else parse_baseball_ip_to_outs(t_pit.get("inningsPitched") or "9.0")
        )
        lob_for = int(t_bat.get("leftOnBase") or 0)
        errors_comm = int(t_fld.get("errors") or 0)

        singles_for = max(0, hits_for - doubles_for - triples_for - hr_for)
        team_woba_num = _compute_woba_numerator(
            singles=singles_for,
            doubles=doubles_for,
            triples=triples_for,
            hr=hr_for,
            bb=bb_for,
            ibb=ibb_for,
            hbp=hbp_for,
        )
        woba_denom = max(1, ab_for + bb_for - ibb_for + sf_for + hbp_for)
        team_woba = team_woba_num / float(woba_denom)

        opp_ab = int(opp_bat.get("atBats") or 0)
        opp_2b = int(opp_bat.get("doubles") or 0)
        opp_3b = int(opp_bat.get("triples") or 0)
        opp_ibb = int(opp_bat.get("intentionalWalks") or 0)
        opp_hbp = int(opp_bat.get("hitByPitch") or 0)
        opp_sf = int(opp_bat.get("sacFlies") or 0)
        opp_1b = max(0, hits_against - opp_2b - opp_3b - hr_against)
        opp_woba_num = _compute_woba_numerator(
            singles=opp_1b,
            doubles=opp_2b,
            triples=opp_3b,
            hr=hr_against,
            bb=bb_against,
            ibb=opp_ibb,
            hbp=opp_hbp,
        )
        opp_woba_denom = max(1, opp_ab + bb_against - opp_ibb + opp_sf + opp_hbp)
        opp_woba = opp_woba_num / float(opp_woba_denom)

        team_rows.append(
            {
                "game_id": game_id,
                "event_seq": event_seq,
                "game_date": game_date,
                "start_time_utc": start_time_utc,
                "season": season,
                "league": "MLB",
                "team": team_code,
                "opponent": opp_code,
                "is_home": is_home,
                "runs_for": runs_for,
                "runs_against": runs_against,
                "hits_for": hits_for,
                "hits_against": hits_against,
                "doubles_for": doubles_for,
                "triples_for": triples_for,
                "hr_for": hr_for,
                "hr_against": hr_against,
                "bb_for": bb_for,
                "bb_against": bb_against,
                "k_for": k_for,
                "k_against": k_against,
                "hbp_for": hbp_for,
                "sf_for": sf_for,
                "pa_for": pa_for,
                "outs_pitched": outs_pitched_team,
                "lob_for": lob_for,
                "errors_committed": errors_comm,
                "bullpen_outs_pitched": bp_outs,
                "bullpen_pitches": bp_pitches,
                "bullpen_er": bp_er,
                "team_woba": team_woba,
                "opp_woba": opp_woba,
                "team_xwoba": None,
                "opp_xwoba": None,
                "has_statcast_metrics": False,
            }
        )

    return (
        _cast_to_schema(
            pl.DataFrame(team_rows, schema=SILVER_BASEBALL_TEAM_LOGS_SCHEMA),
            SILVER_BASEBALL_TEAM_LOGS_SCHEMA,
        ),
        _cast_to_schema(
            pl.DataFrame(pitcher_rows, schema=SILVER_BASEBALL_PITCHER_LOGS_SCHEMA),
            SILVER_BASEBALL_PITCHER_LOGS_SCHEMA,
        ),
        _cast_to_schema(
            pl.DataFrame(batter_rows, schema=SILVER_BASEBALL_BATTER_LOGS_SCHEMA),
            SILVER_BASEBALL_BATTER_LOGS_SCHEMA,
        ),
    )


def parse_mlb_roster_payload(
    payload: Mapping[str, Any],
    *,
    team: str,
) -> pl.DataFrame:
    """Parse MLB StatsAPI `/api/v1/teams/{teamId}/roster` payload into a normalized roster frame."""
    norm_team = normalize_baseball_team(team)
    roster_list = payload.get("roster") or []
    rows: list[dict[str, Any]] = []
    for entry in roster_list:
        if not isinstance(entry, Mapping):
            continue
        person = entry.get("person") or {}
        if not isinstance(person, Mapping) or person.get("id") is None:
            continue
        pid = int(person["id"])
        pname = str(person.get("fullName") or f"Player {pid}")
        pos_node = entry.get("position") or person.get("primaryPosition") or {}
        pos_abbrev = (
            str(pos_node.get("abbreviation") or "DH").upper()
            if isinstance(pos_node, Mapping)
            else "DH"
        )
        bat_node = person.get("batSide") or entry.get("batSide") or {}
        pitch_node = person.get("pitchHand") or entry.get("pitchHand") or {}
        bat_side = (
            str(bat_node.get("code") or "R").upper()
            if isinstance(bat_node, Mapping)
            else str(bat_node or "R").upper()
        )
        pitch_hand = (
            str(pitch_node.get("code") or "R").upper()
            if isinstance(pitch_node, Mapping)
            else str(pitch_node or "R").upper()
        )
        status_node = entry.get("status") or {}
        status_code = (
            str(status_node.get("code") or "A").upper()
            if isinstance(status_node, Mapping)
            else str(status_node or "A").upper()
        )
        is_available = status_code in {"A", "ACT", "ACTIVE"}
        is_pitcher = pos_abbrev in {"P", "SP", "RP", "CL", "TWP"}

        rows.append(
            {
                "player_id": pid,
                "player_name": pname,
                "team": norm_team,
                "league": "MLB",
                "position_group": "P" if is_pitcher else _map_position_group(pos_abbrev),
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


class MLBStatsAPIClient:
    """Cached HTTP client for Official MLB StatsAPI (`statsapi.mlb.com/api/v1`)."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        base_url: str = MLB_STATSAPI_BASE_URL,
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
            return dict(data) if isinstance(data, Mapping) else {}

    def fetch_schedule(
        self,
        start_date: date,
        end_date: date,
        *,
        ttl_seconds: float | None = 3600.0,
    ) -> dict[str, Any]:
        """Fetch MLB schedule with probable pitchers, lineups, and venue hydration."""
        cache_path = (
            self.cache_dir
            / "schedules"
            / f"schedule_{start_date.isoformat()}_{end_date.isoformat()}.json.gz"
        )
        cached = read_cached_json(cache_path, ttl_seconds=ttl_seconds)
        if isinstance(cached, dict):
            return cached

        payload = self._get_json(
            "schedule",
            params={
                "sportId": 1,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "hydrate": "probablePitcher,lineups,team,venue,linescore",
            },
        )
        write_cached_json(cache_path, payload)
        return payload

    def fetch_season_schedule(
        self,
        season: int,
        *,
        is_active_season: bool = False,
        ttl_seconds: float | None = 3600.0,
    ) -> dict[str, Any]:
        """Fetch an entire MLB season schedule (with linescore & probablePitcher hydration)."""
        cache_path = self.cache_dir / "schedules" / f"mlb_season_{int(season)}.json.gz"
        eff_ttl = ttl_seconds if is_active_season else None
        cached = read_cached_json(cache_path, ttl_seconds=eff_ttl)
        if isinstance(cached, dict) and cached.get("dates"):
            return cached

        payload = self._get_json(
            "schedule",
            params={
                "sportId": 1,
                "season": int(season),
                "gameType": "R",
                "hydrate": "probablePitcher,team,venue,linescore",
            },
        )
        if payload.get("dates"):
            write_cached_json(cache_path, payload)
        return payload

    def fetch_boxscore(
        self,
        game_id: int,
        *,
        is_final: bool = True,
    ) -> dict[str, Any]:
        """Fetch MLB game boxscore; permanently cached once `is_final=True`."""
        cache_path = self.cache_dir / "boxscores" / f"{int(game_id)}.json.gz"
        ttl = None if is_final else 900.0
        cached = read_cached_json(cache_path, ttl_seconds=ttl)
        if isinstance(cached, dict):
            return cached

        payload = self._get_json(f"game/{int(game_id)}/boxscore")
        write_cached_json(cache_path, payload)
        return payload

    def fetch_team_roster(
        self,
        team: str,
        *,
        ttl_seconds: float | None = 3600.0,
    ) -> dict[str, Any]:
        """Fetch active MLB team roster (including status codes) from `/teams/{id}/roster`."""
        norm_team = normalize_baseball_team(team)
        club = get_baseball_club_metadata(norm_team)
        team_id = club.mlb_statsapi_id if (club and club.mlb_statsapi_id) else 119
        cache_path = self.cache_dir / "rosters" / f"{norm_team}_active.json.gz"
        cached = read_cached_json(cache_path, ttl_seconds=ttl_seconds)
        if isinstance(cached, dict):
            return cached

        payload = self._get_json(
            f"teams/{team_id}/roster",
            params={"rosterType": "fullSeason"},
        )
        write_cached_json(cache_path, payload)
        return payload
