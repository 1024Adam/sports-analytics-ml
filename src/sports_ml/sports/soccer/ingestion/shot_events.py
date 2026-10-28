"""Understat EPL shot-level micro-event ingestion & spatial parser (`shot_events.py`).

Extracts shot-level event records (`situation` in `{"OpenPlay", "FromCorner", "SetPiece",
"DirectFreekick", "Penalty"}`, `shotType`, `(X, Y)` pitch coordinates in `[0, 1]`,
`lastAction` in `{"Throughball", "Cross", "Rebound", "Standard"}`, and minute/score-state
at shot timestamp) from Understat match payloads into `SILVER_TEAM_SHOT_MICRO_LOGS_SCHEMA`:
- Open-Play vs. Set-Piece xG Split (`open_play_xgf`, `open_play_xga`, `set_piece_xgf`,
  `set_piece_xga`, `penalty_xgf`, `penalty_xga`, `total_xgf`, `total_xga`), satisfying
  `open_play_xgf + set_piece_xgf + penalty_xgf == total_xgf` (within `1e-6`).
- Game-State Neutral / Tied npxG Rate (`tied_state_npxgd_per_90`).
- Box-Entry & Throughball Danger (`box_shot_share`, `six_yard_shot_share`, `throughball_xgf`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl

from sports_ml.core.cache import list_cached_files, read_cached_json
from sports_ml.sports.soccer.constants import (
    SILVER_TEAM_SHOT_MICRO_LOGS_SCHEMA,
    normalize_soccer_club,
)
from sports_ml.sports.soccer.ingestion.fpl_api import make_canonical_match_id

SET_PIECE_SITUATIONS: Final[frozenset[str]] = frozenset(
    {"FROMCORNER", "SETPIECE", "DIRECTFREEKICK"}
)
PENALTY_SITUATIONS: Final[frozenset[str]] = frozenset({"PENALTY"})


def classify_shot_situation(raw_situation: str | None) -> str:
    """Classify an Understat shot `situation` into `'open_play'`, `'set_piece'`, or `'penalty'`."""
    if not raw_situation:
        return "open_play"
    cleaned = raw_situation.strip().upper().replace("_", "").replace(" ", "")
    if cleaned in PENALTY_SITUATIONS:
        return "penalty"
    if cleaned in SET_PIECE_SITUATIONS:
        return "set_piece"
    return "open_play"


def is_18_yard_box_shot(x_norm: float, y_norm: float) -> tuple[bool, bool]:
    """Return `(in_18_yard_box, in_6_yard_box)` from normalized Understat pitch coords `(X, Y)`.

    Understat normalizes attacking pitch coordinates to `[0, 1]` where `(1.0, 0.5)` is
    the center of the opponent's goal line:
    - 18-yard penalty box: `X >= 0.83` and `0.21 <= Y <= 0.79`
    - 6-yard goal area: `X >= 0.94` and `0.37 <= Y <= 0.63`
    Also supports Opta `[0, 100]` coordinates by dividing by `100.0` if `X > 1.5`.
    """
    x = float(x_norm) / 100.0 if float(x_norm) > 1.5 else float(x_norm)
    y = float(y_norm) / 100.0 if float(y_norm) > 1.5 else float(y_norm)
    in_box = x >= 0.83 and 0.21 <= y <= 0.79
    in_six = x >= 0.94 and 0.37 <= y <= 0.63
    return bool(in_box), bool(in_six)


def parse_understat_match_shots_payload(
    shots_payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    default_season: int = 20242025,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse Understat match shot records into `(team_shot_micro_df, player_shot_micro_df)`.

    Accepts either:
    - A flat list of shot dicts, or
    - An Understat match dict `{"h": [...], "a": [...]}` or `{"shots": [...]}`.
    """
    flat_shots: list[Mapping[str, Any]] = []
    if isinstance(shots_payload, Mapping):
        if "h" in shots_payload or "a" in shots_payload:
            for s in shots_payload.get("h") or []:
                if isinstance(s, Mapping):
                    flat_shots.append({**s, "h_a": s.get("h_a", "h")})
            for s in shots_payload.get("a") or []:
                if isinstance(s, Mapping):
                    flat_shots.append({**s, "h_a": s.get("h_a", "a")})
        elif isinstance(shots_payload.get("shots"), list):
            for s in shots_payload["shots"]:
                if isinstance(s, Mapping):
                    flat_shots.append(s)
    else:
        flat_shots = list(shots_payload)

    # Sort shots chronologically within each match so running score-state is tracked causally
    def _shot_sort_key(s: Mapping[str, Any]) -> tuple[int, int, int]:
        mid = int(float(str(s.get("match_id") or 1)))
        minute = int(float(str(s.get("minute") or 0)))
        sid = int(float(str(s.get("id") or 0)))
        return (mid, minute, sid)

    sorted_shots = sorted(flat_shots, key=_shot_sort_key)

    match_meta: dict[int, tuple[int, date, datetime, str, str]] = {}
    team_acc: dict[tuple[int, str], dict[str, float]] = {}
    match_score: dict[int, tuple[int, int]] = {}
    match_tied_minutes: dict[int, float] = {}
    match_last_min: dict[int, float] = {}
    player_acc: dict[tuple[int, int, str], dict[str, Any]] = {}

    for s in sorted_shots:
        raw_season = s.get("season") or default_season
        s_str = str(raw_season).strip()
        season = int(f"{s_str[:4]}{int(s_str[:4]) + 1}") if len(s_str) == 4 else int(s_str)

        raw_mid = int(float(str(s.get("match_id") or 1)))
        mid = make_canonical_match_id(season, raw_mid)

        raw_dt = s.get("start_time_utc") or s.get("date") or s.get("match_date") or "2024-08-17"
        if isinstance(raw_dt, datetime):
            dt_utc = raw_dt.astimezone(UTC) if raw_dt.tzinfo else raw_dt.replace(tzinfo=UTC)
            m_date = dt_utc.date()
        elif isinstance(raw_dt, date):
            m_date = raw_dt
            dt_utc = datetime(m_date.year, m_date.month, m_date.day, 15, 0, tzinfo=UTC)
        else:
            d_str = str(raw_dt).strip()
            m_date = date.fromisoformat(d_str[:10])
            dt_utc = datetime(m_date.year, m_date.month, m_date.day, 15, 0, tzinfo=UTC)

        h_team = normalize_soccer_club(str(s.get("h_team") or s.get("home_team") or "ARS"))
        a_team = normalize_soccer_club(str(s.get("a_team") or s.get("away_team") or "MCI"))
        match_meta[mid] = (season, m_date, dt_utc, h_team, a_team)

        for t_code in (h_team, a_team):
            team_acc.setdefault(
                (mid, t_code),
                {
                    "open_play_xgf": 0.0,
                    "open_play_xga": 0.0,
                    "set_piece_xgf": 0.0,
                    "set_piece_xga": 0.0,
                    "penalty_xgf": 0.0,
                    "penalty_xga": 0.0,
                    "tied_npxgf": 0.0,
                    "tied_npxga": 0.0,
                    "shot_count": 0.0,
                    "box_shots": 0.0,
                    "six_yard_shots": 0.0,
                    "throughball_xgf": 0.0,
                },
            )

        curr_h_g, curr_a_g = match_score.get(mid, (0, 0))
        prev_min = match_last_min.get(mid, 0.0)
        minute_val = max(prev_min, min(95.0, float(s.get("minute") or 45.0)))
        if curr_h_g == curr_a_g:
            match_tied_minutes[mid] = match_tied_minutes.get(mid, 0.0) + max(
                0.0, minute_val - prev_min
            )
        match_last_min[mid] = minute_val

        h_a_raw = str(s.get("h_a") or "").strip().lower()
        if h_a_raw in {"h", "a"}:
            is_home_sh = h_a_raw == "h"
            sh_team = h_team if is_home_sh else a_team
        else:
            sh_team = normalize_soccer_club(str(s.get("team") or h_team))
            is_home_sh = sh_team == h_team
        df_team = a_team if is_home_sh else h_team

        if "score_diff" in s and s.get("score_diff") is not None:
            score_diff = int(float(s["score_diff"]))
        else:
            score_diff = (curr_h_g - curr_a_g) if is_home_sh else (curr_a_g - curr_h_g)

        raw_xg = s.get("xG") if s.get("xG") is not None else s.get("xg")
        xg = max(0.0, float(raw_xg) if raw_xg is not None and str(raw_xg).strip() != "" else 0.08)
        sit_cat = classify_shot_situation(str(s.get("situation") or "OpenPlay"))
        last_action = str(s.get("lastAction") or s.get("last_action") or "Standard").strip().lower()
        raw_x = s.get("X") if s.get("X") is not None else s.get("x")
        raw_y = s.get("Y") if s.get("Y") is not None else s.get("y")
        x_val = float(raw_x) if raw_x is not None and str(raw_x).strip() != "" else 0.86
        y_val = float(raw_y) if raw_y is not None and str(raw_y).strip() != "" else 0.50
        in_box, in_six = is_18_yard_box_shot(x_val, y_val)

        sh_b = team_acc[(mid, sh_team)]
        df_b = team_acc[(mid, df_team)]

        if sit_cat == "open_play":
            sh_b["open_play_xgf"] += xg
            df_b["open_play_xga"] += xg
        elif sit_cat == "set_piece":
            sh_b["set_piece_xgf"] += xg
            df_b["set_piece_xga"] += xg
        else:
            sh_b["penalty_xgf"] += xg
            df_b["penalty_xga"] += xg

        # Game-state neutral / tied or within 1 goal npxG (excludes trailing/leading by 2+ goals)
        if sit_cat != "penalty" and abs(score_diff) <= 1:
            sh_b["tied_npxgf"] += xg
            df_b["tied_npxga"] += xg

        sh_b["shot_count"] += 1.0
        if in_box:
            sh_b["box_shots"] += 1.0
        if in_six:
            sh_b["six_yard_shots"] += 1.0
        if last_action in {"throughball", "through_ball", "chipped"}:
            sh_b["throughball_xgf"] += xg

        # Player-level micro accumulation for Stage 2 outfield features
        raw_pid = s.get("player_id")
        p_name = str(s.get("player") or s.get("player_name") or "").strip()
        pid = int(float(str(raw_pid))) if raw_pid is not None and str(raw_pid).isdigit() else 0
        if pid > 0 or p_name:
            pk_key = (mid, pid, p_name or f"P_{pid}")
            pb = player_acc.setdefault(
                pk_key,
                {
                    "match_id": mid,
                    "season": season,
                    "match_date": m_date,
                    "player_id": pid,
                    "player_name": p_name,
                    "team": sh_team,
                    "open_play_npxg": 0.0,
                    "set_piece_xg": 0.0,
                    "penalty_xg": 0.0,
                    "total_xg": 0.0,
                },
            )
            pb["total_xg"] += xg
            if sit_cat == "open_play":
                pb["open_play_npxg"] += xg
            elif sit_cat == "set_piece":
                pb["set_piece_xg"] += xg
            else:
                pb["penalty_xg"] += xg

        # Update running match score strictly AFTER evaluating shot pre-event score state
        result_str = str(s.get("result") or "").strip().lower()
        if result_str == "goal":
            if is_home_sh:
                curr_h_g += 1
            else:
                curr_a_g += 1
        elif result_str in {"owngoal", "own_goal"}:
            if is_home_sh:
                curr_a_g += 1
            else:
                curr_h_g += 1
        match_score[mid] = (curr_h_g, curr_a_g)

    team_rows: list[dict[str, Any]] = []
    for (mid, t_code), b in team_acc.items():
        season, m_date, dt_utc, h_team, a_team = match_meta[mid]
        is_home = t_code == h_team
        opp_code = a_team if is_home else h_team

        # Account for remaining match minutes through 90' if score ended tied
        end_h, end_a = match_score.get(mid, (0, 0))
        last_m = match_last_min.get(mid, 0.0)
        tied_m = match_tied_minutes.get(mid, 0.0) + (
            max(0.0, 90.0 - last_m) if end_h == end_a else 0.0
        )
        eff_minutes = max(30.0, min(90.0, tied_m if tied_m > 0.0 else 90.0))

        op_xgf = float(b["open_play_xgf"])
        sp_xgf = float(b["set_piece_xgf"])
        pen_xgf = float(b["penalty_xgf"])
        tot_xgf = op_xgf + sp_xgf + pen_xgf

        op_xga = float(b["open_play_xga"])
        sp_xga = float(b["set_piece_xga"])
        pen_xga = float(b["penalty_xga"])
        tot_xga = op_xga + sp_xga + pen_xga

        shots_n = b["shot_count"]
        box_share = (b["box_shots"] / shots_n) if shots_n > 0.0 else 0.62
        six_share = (b["six_yard_shots"] / shots_n) if shots_n > 0.0 else 0.12
        tied_npxgd_90 = 90.0 * (b["tied_npxgf"] - b["tied_npxga"]) / eff_minutes

        team_rows.append(
            {
                "match_id": mid,
                "match_date": m_date,
                "start_time_utc": dt_utc,
                "season": season,
                "team": t_code,
                "opponent": opp_code,
                "is_home": is_home,
                "open_play_xgf": op_xgf,
                "open_play_xga": op_xga,
                "set_piece_xgf": sp_xgf,
                "set_piece_xga": sp_xga,
                "penalty_xgf": pen_xgf,
                "penalty_xga": pen_xga,
                "total_xgf": tot_xgf,
                "total_xga": tot_xga,
                "tied_state_npxgd_per_90": float(tied_npxgd_90),
                "box_shot_share": float(max(0.0, min(1.0, box_share))),
                "six_yard_shot_share": float(max(0.0, min(1.0, six_share))),
                "throughball_xgf": float(b["throughball_xgf"]),
            }
        )

    player_schema = {
        "match_id": pl.Int64(),
        "season": pl.Int32(),
        "match_date": pl.Date(),
        "player_id": pl.Int64(),
        "player_name": pl.Utf8(),
        "team": pl.Utf8(),
        "open_play_npxg": pl.Float64(),
        "set_piece_xg": pl.Float64(),
        "penalty_xg": pl.Float64(),
        "total_xg": pl.Float64(),
    }
    team_df = (
        pl.DataFrame(team_rows, schema=SILVER_TEAM_SHOT_MICRO_LOGS_SCHEMA).sort(
            ["match_date", "start_time_utc", "match_id", "team"]
        )
        if team_rows
        else pl.DataFrame(schema=SILVER_TEAM_SHOT_MICRO_LOGS_SCHEMA)
    )
    player_df = (
        pl.DataFrame(list(player_acc.values()), schema=player_schema).sort(
            ["match_date", "match_id", "team", "player_id"]
        )
        if player_acc
        else pl.DataFrame(schema=player_schema)
    )
    return team_df, player_df


class SoccerShotEventsIngestor:
    """Loads cached Understat match shot payloads (`epl_shots_*.json.gz`) into Silver micro logs."""

    def __init__(self, understat_dir: Path) -> None:
        self.understat_dir = understat_dir

    def load_cached_shot_micro_logs(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        team_frames: list[pl.DataFrame] = []
        player_frames: list[pl.DataFrame] = []

        if self.understat_dir.exists():
            for sh_path in list_cached_files(self.understat_dir, "epl_shots_*.json"):
                payload = read_cached_json(sh_path)
                if payload is not None:
                    t_df, p_df = parse_understat_match_shots_payload(payload)
                    if not t_df.is_empty():
                        team_frames.append(t_df)
                    if not p_df.is_empty():
                        player_frames.append(p_df)

        t_out = (
            pl.concat(team_frames, how="vertical")
            .unique(subset=["match_id", "team"], keep="last")
            .sort(["match_date", "start_time_utc", "match_id", "team"])
            if team_frames
            else pl.DataFrame(schema=SILVER_TEAM_SHOT_MICRO_LOGS_SCHEMA)
        )
        p_out = (
            pl.concat(player_frames, how="vertical").unique(
                subset=["match_id", "team", "player_id", "player_name"], keep="last"
            )
            if player_frames
            else pl.DataFrame(
                schema={
                    "match_id": pl.Int64(),
                    "season": pl.Int32(),
                    "match_date": pl.Date(),
                    "player_id": pl.Int64(),
                    "player_name": pl.Utf8(),
                    "team": pl.Utf8(),
                    "open_play_npxg": pl.Float64(),
                    "set_piece_xg": pl.Float64(),
                    "penalty_xg": pl.Float64(),
                    "total_xg": pl.Float64(),
                }
            )
        )
        return t_out, p_out
