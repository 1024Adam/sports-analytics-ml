"""NHL shot-coordinate and play-by-play micro-event ingestion (`ingestion/pbp_shots.py`).

Ingests MoneyPuck bulk shot-level archives (`shots_{YYYY}.csv.gz`) and NHL API
play-by-play payloads (`/v1/gamecenter/{GAME_ID}/play-by-play`) and aggregates
them into compact game-team (`SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA`) and game-skater
(`SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA`) micro-feature tables:
- Score-Venue Adjusted 5v5 xG & Fenwick (`score_adj_xgf_5v5`, `score_adj_xga_5v5`,
  `score_adj_ff_5v5`, `score_adj_fa_5v5`): Discounts trailing-team garbage-time shot
  volume (`score_diff <= -2`) and boosts shots generated while leading (`score_diff >= +1`).
- Rush & Rebound xG Generation (`rush_xgf_5v5`, `rebound_xgf_5v5`, `rebound_xga_5v5`).
- Spatial Shot Quality (`avg_shot_distance_5v5`, `inner_slot_xg_share`) from `(x, y)`
  ice coordinates relative to the offensive net at `(89.0, 0.0)`.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl

from sports_ml.core.cache import list_cached_files, read_cached_bytes, read_cached_json
from sports_ml.sports.hockey.constants import (
    SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA,
    SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA,
    normalize_tricode,
)

NHL_NET_X_FEET: Final[float] = 89.0
INNER_SLOT_MAX_DISTANCE_FT: Final[float] = 28.0
INNER_SLOT_MAX_ABS_Y_FT: Final[float] = 18.0
INNER_SLOT_MAX_ANGLE_DEG: Final[float] = 42.0

# Empirical NHL 5v5 score-venue adjustment weights (Micah Blake McCurdy / Evolving-Hockey style):
# Trailing teams shoot more frequently from the perimeter against prevent shells, so trailing
# shot volume/xG is discounted (< 1.0) while leading teams' 5v5 shots are boosted (> 1.0).
SCORE_STATE_5V5_WEIGHTS: Final[dict[int, tuple[float, float]]] = {
    # score_diff (shooter_goals - goalie_goals): (home_shooter_weight, away_shooter_weight)
    -3: (0.830, 0.860),
    -2: (0.865, 0.895),
    -1: (0.915, 0.945),
    0: (0.985, 1.015),
    1: (1.065, 1.095),
    2: (1.125, 1.155),
    3: (1.170, 1.200),
}


def compute_5v5_score_venue_weight(
    shooter_score_diff: int,
    *,
    is_home_shooter: bool = True,
) -> float:
    """Return strictly positive 5v5 score-venue adjustment weight for a shot attempt.

    Invariants:
    1. Strictly positive (`> 0.0`) for all score states.
    2. Monotonically increasing with `shooter_score_diff`: a 5v5 shot taken while leading
       by `+2` receives a strictly higher weight than an identical shot taken while
       trailing by `-2`.
    """
    clamped = max(-3, min(3, int(shooter_score_diff)))
    h_w, a_w = SCORE_STATE_5V5_WEIGHTS[clamped]
    return float(h_w if is_home_shooter else a_w)


def compute_shot_geometry(x_coord: float, y_coord: float) -> tuple[float, float, bool]:
    """Compute `(distance_ft, angle_deg, is_inner_slot)` from NHL ice coordinates `(x, y)`.

    Normalizes `x` to the offensive zone (`abs(x)`) relative to the goal line at `x = 89.0`.
    """
    abs_x = abs(float(x_coord))
    abs_y = abs(float(y_coord))
    dx = max(1.0, NHL_NET_X_FEET - abs_x)
    dist = math.hypot(dx, abs_y)
    angle_deg = math.degrees(math.atan2(abs_y, dx))
    is_slot = (
        abs_x >= 60.0
        and dist <= INNER_SLOT_MAX_DISTANCE_FT
        and abs_y <= INNER_SLOT_MAX_ABS_Y_FT
        and angle_deg <= INNER_SLOT_MAX_ANGLE_DEG
    )
    return float(dist), float(angle_deg), bool(is_slot)


def estimate_shot_xg_from_coords(
    distance_ft: float,
    angle_deg: float,
    *,
    is_rush: bool = False,
    is_rebound: bool = False,
    is_inner_slot: bool = False,
) -> float:
    """Estimate single-shot 5v5 `xG` when explicit `xGoal` is not present (e.g. raw NHL PBP)."""
    log_odds = -1.45 - 0.052 * distance_ft - 0.018 * angle_deg
    if is_inner_slot:
        log_odds += 0.55
    if is_rush:
        log_odds += 0.45
    if is_rebound:
        log_odds += 0.65
    prob = 1.0 / (1.0 + math.exp(-max(-12.0, min(6.0, log_odds))))
    return float(max(0.002, min(0.85, prob)))


def _to_bool_flag(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val) > 0.0
    return str(val).strip().lower() in {"1", "true", "yes", "y", "t"}


def _normalize_game_id(raw_gid: Any, season: int) -> int:
    gid = int(float(str(raw_gid).strip()))
    if gid < 1_000_000_000:
        start_yr = int(str(season)[:4])
        return int(f"{start_yr}0{gid}") if gid >= 20_000 else int(f"{start_yr}02{gid:04d}")
    return gid


def aggregate_shot_events_to_micro_logs(
    shot_records: Sequence[Mapping[str, Any]],
    *,
    default_season: int = 20242025,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate normalized 5v5 shot records into `(team_micro_df, skater_micro_df)`."""
    # Group per game -> metadata and per-team / per-skater accumulators
    game_meta: dict[int, tuple[int, date, datetime, str, str]] = {}
    team_acc: dict[tuple[int, str], dict[str, float]] = {}
    skater_acc: dict[tuple[int, int], dict[str, Any]] = {}

    for rec in shot_records:
        # Filter strictly to 5v5 / even-strength unblocked shot attempts (Fenwick)
        skaters_h = int(float(rec.get("homeSkatersOnIce") or rec.get("home_skaters") or 5))
        skaters_a = int(float(rec.get("awaySkatersOnIce") or rec.get("away_skaters") or 5))
        strength = str(rec.get("situation") or rec.get("strength") or "5v5").strip().lower()
        if (skaters_h != 5 or skaters_a != 5) and strength not in {"5v5", "5on5", "ev"}:
            continue

        event_type = str(rec.get("event") or rec.get("typeDescKey") or "SHOT").strip().upper()
        if event_type in {"BLOCK", "BLOCKED_SHOT", "BLOCKED-SHOT"}:
            continue

        raw_season = rec.get("season") or default_season
        season_str = str(raw_season).strip()
        season = (
            int(f"{season_str[:4]}{int(season_str[:4]) + 1}")
            if len(season_str) == 4
            else int(season_str)
        )
        gid = _normalize_game_id(rec.get("game_id") or rec.get("gameId") or 2024020001, season)

        raw_dt = rec.get("start_time_utc") or rec.get("game_date") or "2024-10-10"
        if isinstance(raw_dt, datetime):
            dt_utc = raw_dt.astimezone(UTC) if raw_dt.tzinfo else raw_dt.replace(tzinfo=UTC)
            g_date = dt_utc.date()
        elif isinstance(raw_dt, date):
            g_date = raw_dt
            dt_utc = datetime(g_date.year, g_date.month, g_date.day, 23, 0, tzinfo=UTC)
        else:
            d_str = str(raw_dt).strip()
            g_date = date.fromisoformat(d_str[:10])
            dt_utc = datetime(g_date.year, g_date.month, g_date.day, 23, 0, tzinfo=UTC)

        h_team = normalize_tricode(str(rec.get("homeTeamCode") or rec.get("home_team") or "BOS"))
        a_team = normalize_tricode(str(rec.get("awayTeamCode") or rec.get("away_team") or "TOR"))
        game_meta[gid] = (season, g_date, dt_utc, h_team, a_team)

        for side_t in (h_team, a_team):
            team_acc.setdefault(
                (gid, side_t),
                {
                    "score_adj_xgf_5v5": 0.0,
                    "score_adj_xga_5v5": 0.0,
                    "score_adj_ff_5v5": 0.0,
                    "score_adj_fa_5v5": 0.0,
                    "rush_xgf_5v5": 0.0,
                    "rebound_xgf_5v5": 0.0,
                    "rebound_xga_5v5": 0.0,
                    "raw_xgf_5v5": 0.0,
                    "inner_slot_xgf_5v5": 0.0,
                    "dist_sum_5v5": 0.0,
                    "ff_count_5v5": 0.0,
                },
            )

        raw_is_home = rec.get("isHomeTeam")
        if raw_is_home is not None and str(raw_is_home).strip() != "":
            is_home_shooter = _to_bool_flag(raw_is_home)
            shooter_team = h_team if is_home_shooter else a_team
        else:
            shooter_team = normalize_tricode(str(rec.get("team") or rec.get("teamCode") or h_team))
            is_home_shooter = shooter_team == h_team
        defending_team = a_team if is_home_shooter else h_team

        h_goals = int(float(rec.get("homeTeamGoals") or rec.get("home_score") or 0))
        a_goals = int(float(rec.get("awayTeamGoals") or rec.get("away_score") or 0))
        if "score_diff" in rec and rec.get("score_diff") is not None:
            score_diff = int(float(rec["score_diff"]))
        else:
            score_diff = (h_goals - a_goals) if is_home_shooter else (a_goals - h_goals)

        raw_x = rec.get("xCordAdjusted")
        if raw_x is None:
            raw_x = rec.get("xCord")
        if raw_x is None:
            raw_x = rec.get("x")
        x_c = float(raw_x) if raw_x is not None and str(raw_x).strip() != "" else 65.0

        raw_y = rec.get("yCordAdjusted")
        if raw_y is None:
            raw_y = rec.get("yCord")
        if raw_y is None:
            raw_y = rec.get("y")
        y_c = float(raw_y) if raw_y is not None and str(raw_y).strip() != "" else 0.0
        dist_ft, angle_deg, is_slot = compute_shot_geometry(x_c, y_c)
        if "shotDistance" in rec and rec.get("shotDistance") is not None:
            dist_ft = max(1.0, float(rec["shotDistance"]))

        is_rush = _to_bool_flag(rec.get("shotRush") or rec.get("is_rush"))
        is_rebound = _to_bool_flag(rec.get("shotRebound") or rec.get("is_rebound"))

        raw_xg_val = rec.get("xGoal") if rec.get("xGoal") is not None else rec.get("xg")
        if raw_xg_val is not None and str(raw_xg_val).strip() != "":
            xg = max(0.0, float(raw_xg_val))
        else:
            xg = estimate_shot_xg_from_coords(
                dist_ft,
                angle_deg,
                is_rush=is_rush,
                is_rebound=is_rebound,
                is_inner_slot=is_slot,
            )

        w_adj = compute_5v5_score_venue_weight(score_diff, is_home_shooter=is_home_shooter)
        adj_xg = xg * w_adj

        sh_bucket = team_acc[(gid, shooter_team)]
        df_bucket = team_acc[(gid, defending_team)]

        sh_bucket["score_adj_xgf_5v5"] += adj_xg
        df_bucket["score_adj_xga_5v5"] += adj_xg
        sh_bucket["score_adj_ff_5v5"] += w_adj
        df_bucket["score_adj_fa_5v5"] += w_adj
        sh_bucket["raw_xgf_5v5"] += xg
        sh_bucket["dist_sum_5v5"] += dist_ft
        sh_bucket["ff_count_5v5"] += 1.0

        if is_rush:
            sh_bucket["rush_xgf_5v5"] += xg
        if is_rebound:
            sh_bucket["rebound_xgf_5v5"] += xg
            df_bucket["rebound_xga_5v5"] += xg
        if is_slot:
            sh_bucket["inner_slot_xgf_5v5"] += xg

        raw_pid = (
            rec.get("shooterPlayerId")
            or rec.get("shootingPlayerId")
            or rec.get("player_id")
            or rec.get("playerId")
        )
        if raw_pid is not None and str(raw_pid).strip() != "":
            try:
                pid = int(float(str(raw_pid).strip()))
            except ValueError:
                pid = 0
            if pid > 0:
                sk_b = skater_acc.setdefault(
                    (gid, pid),
                    {
                        "season": season,
                        "game_date": g_date,
                        "start_time_utc": dt_utc,
                        "team": shooter_team,
                        "opponent": defending_team,
                        "is_home": is_home_shooter,
                        "rush_ixg_5v5": 0.0,
                        "rebound_ixg_5v5": 0.0,
                        "inner_slot_ixg_5v5": 0.0,
                    },
                )
                if is_rush:
                    sk_b["rush_ixg_5v5"] += xg
                if is_rebound:
                    sk_b["rebound_ixg_5v5"] += xg
                if is_slot:
                    sk_b["inner_slot_ixg_5v5"] += xg

    team_rows: list[dict[str, Any]] = []
    for (gid, team_code), b in team_acc.items():
        season, g_date, dt_utc, h_team, a_team = game_meta[gid]
        is_home = team_code == h_team
        opp_code = a_team if is_home else h_team
        ff_n = b["ff_count_5v5"]
        raw_xgf = b["raw_xgf_5v5"]
        avg_dist = (b["dist_sum_5v5"] / ff_n) if ff_n > 0.0 else 34.0
        slot_share = (b["inner_slot_xgf_5v5"] / raw_xgf) if raw_xgf > 1e-9 else 0.42

        team_rows.append(
            {
                "game_id": gid,
                "game_date": g_date,
                "start_time_utc": dt_utc,
                "season": season,
                "team": team_code,
                "opponent": opp_code,
                "is_home": is_home,
                "score_adj_xgf_5v5": max(0.0, float(b["score_adj_xgf_5v5"])),
                "score_adj_xga_5v5": max(0.0, float(b["score_adj_xga_5v5"])),
                "score_adj_ff_5v5": max(0.0, float(b["score_adj_ff_5v5"])),
                "score_adj_fa_5v5": max(0.0, float(b["score_adj_fa_5v5"])),
                "rush_xgf_5v5": max(0.0, float(b["rush_xgf_5v5"])),
                "rebound_xgf_5v5": max(0.0, float(b["rebound_xgf_5v5"])),
                "rebound_xga_5v5": max(0.0, float(b["rebound_xga_5v5"])),
                "avg_shot_distance_5v5": float(avg_dist),
                "inner_slot_xg_share": float(max(0.0, min(1.0, slot_share))),
            }
        )

    skater_rows: list[dict[str, Any]] = []
    for (gid, pid), sb in skater_acc.items():
        skater_rows.append(
            {
                "game_id": gid,
                "game_date": sb["game_date"],
                "start_time_utc": sb["start_time_utc"],
                "season": sb["season"],
                "player_id": pid,
                "team": sb["team"],
                "opponent": sb["opponent"],
                "is_home": sb["is_home"],
                "rush_ixg_5v5": float(sb["rush_ixg_5v5"]),
                "rebound_ixg_5v5": float(sb["rebound_ixg_5v5"]),
                "inner_slot_ixg_5v5": float(sb["inner_slot_ixg_5v5"]),
                "qoc_opponent_toi_share": 0.30,
                "qot_teammate_ixg_per_60": 0.72,
            }
        )

    team_df = (
        pl.DataFrame(team_rows, schema=SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA).sort(
            ["game_date", "start_time_utc", "game_id", "team"]
        )
        if team_rows
        else pl.DataFrame(schema=SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA)
    )
    skater_df = (
        pl.DataFrame(skater_rows, schema=SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA).sort(
            ["game_date", "start_time_utc", "game_id", "player_id"]
        )
        if skater_rows
        else pl.DataFrame(schema=SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA)
    )
    return team_df, skater_df


def parse_moneypuck_shots_csv(
    csv_text: str,
    *,
    default_season: int = 20242025,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse MoneyPuck `shots_{YYYY}.csv` text into `(team_pbp_micro_df, skater_pbp_micro_df)`."""
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    records = [{str(k).strip(): v for k, v in row.items() if k is not None} for row in reader]
    return aggregate_shot_events_to_micro_logs(records, default_season=default_season)


def parse_nhl_pbp_payload(
    pbp_payload: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse an Official NHL Web API `/v1/gamecenter/{GAME_ID}/play-by-play` JSON payload."""
    gid = int(pbp_payload.get("id") or 2024020001)
    season = int(pbp_payload.get("season") or f"{str(gid)[:4]}{int(str(gid)[:4]) + 1}")
    start_str = str(
        pbp_payload.get("startTimeUTC") or f"{pbp_payload.get('gameDate', '2024-10-10')}T23:00:00Z"
    )
    dt_utc = datetime.fromisoformat(start_str.replace("Z", "+00:00")).astimezone(UTC)

    home_meta = pbp_payload.get("homeTeam") or {}
    away_meta = pbp_payload.get("awayTeam") or {}
    home_id = int(home_meta.get("id") or 1)
    h_team = normalize_tricode(str(home_meta.get("abbrev") or "BOS"))
    a_team = normalize_tricode(str(away_meta.get("abbrev") or "TOR"))

    plays = pbp_payload.get("plays") or []
    shot_records: list[dict[str, Any]] = []
    h_score = 0
    a_score = 0
    prev_shot_sec: float | None = None
    prev_shot_team: str | None = None

    for play in plays:
        type_key = str(play.get("typeDescKey") or "").strip().lower()
        if type_key not in {"shot-on-goal", "missed-shot", "goal"}:
            continue

        # NHL situationCode is 4 digits: AwayGoalie + AwaySkaters + HomeSkaters + HomeGoalie
        sit_code = str(play.get("situationCode") or "1551").strip()
        is_5v5 = len(sit_code) == 4 and sit_code[1] == "5" and sit_code[2] == "5"

        details = play.get("details") or {}
        owner_id = int(details.get("eventOwnerTeamId") or home_id)
        is_home_sh = owner_id == home_id
        sh_team = h_team if is_home_sh else a_team

        # Parse period time to detect rebounds (<= 3s after prior shot by same team)
        time_str = str(play.get("timeInPeriod") or "00:00")
        parts = time_str.split(":")
        sec_in_p = float(int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0))
        period_num = int((play.get("periodDescriptor") or {}).get("number") or 1)
        abs_sec = (period_num - 1) * 1200.0 + sec_in_p

        is_rebound = bool(
            prev_shot_sec is not None
            and prev_shot_team == sh_team
            and 0.0 <= (abs_sec - prev_shot_sec) <= 3.0
        )
        is_rush = bool(details.get("isRush")) or str(details.get("shotType") or "").lower() in {
            "breakaway",
            "wrap-around",
        }
        prev_shot_sec = abs_sec
        prev_shot_team = sh_team

        if is_5v5:
            shooter_pid = details.get("shootingPlayerId") or details.get("scoringPlayerId")
            shot_records.append(
                {
                    "game_id": gid,
                    "season": season,
                    "start_time_utc": dt_utc,
                    "homeTeamCode": h_team,
                    "awayTeamCode": a_team,
                    "isHomeTeam": is_home_sh,
                    "homeTeamGoals": h_score,
                    "awayTeamGoals": a_score,
                    "xCord": float(details.get("xCoord") or 65.0),
                    "yCord": float(details.get("yCoord") or 0.0),
                    "shotRush": is_rush,
                    "shotRebound": is_rebound,
                    "shooterPlayerId": shooter_pid,
                    "xGoal": details.get("xGoal"),
                }
            )

        if type_key == "goal":
            if is_home_sh:
                h_score += 1
            else:
                a_score += 1

    return aggregate_shot_events_to_micro_logs(shot_records, default_season=season)


class NHLPbpShotsIngestor:
    """Loads cached MoneyPuck shot archives (`shots_*.csv.gz`) and NHL PBP payloads."""

    def __init__(self, moneypuck_dir: Path, nhl_api_dir: Path | None = None) -> None:
        self.moneypuck_dir = moneypuck_dir
        self.nhl_api_dir = nhl_api_dir

    def load_cached_micro_logs(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Scan cached `shots_*.csv(.gz)` and `pbp/*.json(.gz)` into team & skater micro-frames."""
        team_frames: list[pl.DataFrame] = []
        skater_frames: list[pl.DataFrame] = []

        if self.moneypuck_dir.exists():
            for s_csv in list_cached_files(self.moneypuck_dir, "shots_*.csv"):
                csv_text = read_cached_bytes(s_csv).decode("utf-8", errors="replace")
                t_df, s_df = parse_moneypuck_shots_csv(csv_text)
                if not t_df.is_empty():
                    team_frames.append(t_df)
                if not s_df.is_empty():
                    skater_frames.append(s_df)

        if self.nhl_api_dir is not None and (self.nhl_api_dir / "pbp").exists():
            for p_file in list_cached_files(self.nhl_api_dir / "pbp", "*.json"):
                payload = read_cached_json(p_file)
                if isinstance(payload, Mapping):
                    t_df, s_df = parse_nhl_pbp_payload(payload)
                    if not t_df.is_empty():
                        team_frames.append(t_df)
                    if not s_df.is_empty():
                        skater_frames.append(s_df)

        t_out = (
            pl.concat(team_frames, how="vertical")
            .unique(subset=["game_id", "team"], keep="last")
            .sort(["game_date", "start_time_utc", "game_id", "team"])
            if team_frames
            else pl.DataFrame(schema=SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA)
        )
        s_out = (
            pl.concat(skater_frames, how="vertical")
            .unique(subset=["game_id", "player_id"], keep="last")
            .sort(["game_date", "start_time_utc", "game_id", "player_id"])
            if skater_frames
            else pl.DataFrame(schema=SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA)
        )
        return t_out, s_out
