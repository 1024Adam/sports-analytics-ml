"""NHL PBP shot-coordinate & shift micro-feature tests (`test_pbp_micro_features.py`)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sports_ml.sports.hockey.features.player import build_stage2_player_features
from sports_ml.sports.hockey.features.team import (
    build_stage1_game_features,
    build_team_rolling_features,
)
from sports_ml.sports.hockey.ingestion.pbp_shots import (
    NHLPbpShotsIngestor,
    aggregate_shot_events_to_micro_logs,
    compute_5v5_score_venue_weight,
    compute_shot_geometry,
    parse_moneypuck_shots_csv,
    parse_nhl_pbp_payload,
)
from sports_ml.sports.hockey.ingestion.reconciler import HockeySilverReconciler
from sports_ml.sports.hockey.ingestion.shifts import (
    compute_shift_overlap_qoc_qot,
    enrich_skater_micro_logs_with_shifts,
)


def test_score_adjustment_invariant_and_non_negativity() -> None:
    """Verify +2 leading 5v5 shot weight > -2 trailing weight and score_adj_xgf_5v5 >= 0."""
    # 1. Direct weight invariant across home and away shooters
    for is_home in (True, False):
        w_lead_2 = compute_5v5_score_venue_weight(2, is_home_shooter=is_home)
        w_tied = compute_5v5_score_venue_weight(0, is_home_shooter=is_home)
        w_trail_2 = compute_5v5_score_venue_weight(-2, is_home_shooter=is_home)
        assert w_lead_2 > w_tied > w_trail_2 > 0.0

    # 2. Identical 5v5 shot taken while leading by +2 vs trailing by -2 in two games
    shot_lead = {
        "game_id": 2024020001,
        "season": 20242025,
        "game_date": "2024-10-10",
        "homeTeamCode": "BOS",
        "awayTeamCode": "TOR",
        "isHomeTeam": True,
        "homeTeamGoals": 3,
        "awayTeamGoals": 1,  # leading by +2
        "xCordAdjusted": 78.0,
        "yCordAdjusted": 5.0,
        "xGoal": 0.15,
        "shotRush": 1,
        "shotRebound": 0,
        "shooterPlayerId": 8477956,
    }
    shot_trail = {
        **shot_lead,
        "game_id": 2024020002,
        "game_date": "2024-10-12",
        "homeTeamGoals": 1,
        "awayTeamGoals": 3,  # trailing by -2
    }

    team_micro_df, skater_micro_df = aggregate_shot_events_to_micro_logs([shot_lead, shot_trail])
    bos_lead = team_micro_df.filter(
        (pl.col("game_id") == 2024020001) & (pl.col("team") == "BOS")
    ).row(0, named=True)
    bos_trail = team_micro_df.filter(
        (pl.col("game_id") == 2024020002) & (pl.col("team") == "BOS")
    ).row(0, named=True)

    assert bos_lead["score_adj_xgf_5v5"] > bos_trail["score_adj_xgf_5v5"]
    assert bos_lead["score_adj_ff_5v5"] > bos_trail["score_adj_ff_5v5"]
    assert np.all(team_micro_df["score_adj_xgf_5v5"].to_numpy() >= 0.0)
    assert np.all(team_micro_df["score_adj_xga_5v5"].to_numpy() >= 0.0)
    assert skater_micro_df.height == 2
    assert skater_micro_df["rush_ixg_5v5"][0] == pytest.approx(0.15)


def test_shot_geometry_and_shift_overlap_qoc_qot() -> None:
    """Verify spatial slot detection and shift-interval overlap QoC/QoT computation."""
    dist_slot, ang_slot, in_slot = compute_shot_geometry(78.0, 6.0)
    dist_point, ang_point, in_point = compute_shot_geometry(35.0, 32.0)
    assert in_slot is True
    assert in_point is False
    assert dist_slot < dist_point
    assert ang_slot < ang_point

    # Shift intervals: BOS top defenseman (pid=101) overlaps heavily with TOR top forward (pid=201)
    # while BOS 3rd-pair defenseman (pid=102) overlaps only with TOR 4th-line forward (pid=202)
    shifts = [
        # Period 1: TOR top forward (201) plays 900s; TOR 4th-line forward (202) plays 180s
        {
            "game_id": 2024020001,
            "player_id": 201,
            "team": "TOR",
            "period": 1,
            "start_sec": 0,
            "end_sec": 900,
        },
        {
            "game_id": 2024020001,
            "player_id": 202,
            "team": "TOR",
            "period": 1,
            "start_sec": 900,
            "end_sec": 1080,
        },
        # BOS shutdown D (101) matches up against 201 for 800s alongside elite teammate (103)
        {
            "game_id": 2024020001,
            "player_id": 101,
            "team": "BOS",
            "period": 1,
            "start_sec": 0,
            "end_sec": 800,
        },
        {
            "game_id": 2024020001,
            "player_id": 103,
            "team": "BOS",
            "period": 1,
            "start_sec": 0,
            "end_sec": 800,
        },
        # BOS sheltered D (102) matches up against 202 for 180s alongside low-ixG teammate (104)
        {
            "game_id": 2024020001,
            "player_id": 102,
            "team": "BOS",
            "period": 1,
            "start_sec": 900,
            "end_sec": 1080,
        },
        {
            "game_id": 2024020001,
            "player_id": 104,
            "team": "BOS",
            "period": 1,
            "start_sec": 900,
            "end_sec": 1080,
        },
    ]
    qoc_df = compute_shift_overlap_qoc_qot(
        shifts,
        skater_pregame_ixg_per_60={101: 0.35, 102: 0.25, 103: 1.15, 104: 0.30},
    )
    r_101 = qoc_df.filter(pl.col("player_id") == 101).row(0, named=True)
    r_102 = qoc_df.filter(pl.col("player_id") == 102).row(0, named=True)

    # Shutdown D (101) faced higher-TOI opponent (201) and played with higher-ixG teammate (103)
    assert r_101["qoc_opponent_toi_share"] > r_102["qoc_opponent_toi_share"]
    assert r_101["qot_teammate_ixg_per_60"] > r_102["qot_teammate_ixg_per_60"]


def test_strict_temporal_anti_leakage_on_hockey_pbp_micro_features() -> None:
    """Verify mutating shot coords, score states, or shifts at >= t leaves < t unchanged."""
    base_dt = datetime(2024, 10, 10, 23, 0, tzinfo=UTC)
    games_rows = []
    team_log_rows = []
    skater_log_rows = []
    shot_events_base = []
    shift_records_base = []

    for i in range(6):
        gid = 2024020001 + i
        dt_utc = base_dt + timedelta(days=i * 2)
        g_date = dt_utc.date()
        games_rows.append(
            {
                "sport": "hockey",
                "season": 20242025,
                "game_type": 2,
                "game_id": gid,
                "game_date": g_date,
                "start_time_utc": dt_utc,
                "home_team": "BOS",
                "away_team": "TOR",
                "venue_timezone": "America/New_York",
                "game_state": "FINAL",
                "finish_period_type": "REG",
                "home_goals": 3,
                "away_goals": 2,
                "home_reg_goals": 3,
                "away_reg_goals": 2,
                "home_win": 1,
            }
        )
        for team_c, opp_c, is_h in (("BOS", "TOR", True), ("TOR", "BOS", False)):
            team_log_rows.append(
                {
                    "game_id": gid,
                    "game_date": g_date,
                    "start_time_utc": dt_utc,
                    "season": 20242025,
                    "team": team_c,
                    "opponent": opp_c,
                    "is_home": is_h,
                    "goals_for": 3 if is_h else 2,
                    "goals_against": 2 if is_h else 3,
                    "xgf_5v5": 2.3 if is_h else 1.9,
                    "xga_5v5": 1.9 if is_h else 2.3,
                }
            )
        skater_log_rows.append(
            {
                "game_id": gid,
                "game_date": g_date,
                "start_time_utc": dt_utc,
                "season": 20242025,
                "player_id": 8477956,
                "player_name": "David Pastrnak",
                "team": "BOS",
                "opponent": "TOR",
                "is_home": True,
                "position_group": "F",
                "toi_total_min": 20.0,
                "toi_ev_min": 16.0,
                "toi_pp_min": 4.0,
                "toi_pk_min": 0.0,
                "goals": 1,
                "assists": 1,
                "points": 2,
                "sog": 4,
                "ixg_all": 0.45,
                "ixg_5v5": 0.32,
                "icf_all": 7.0,
                "on_ice_xgf_5v5": 1.1,
                "on_ice_xga_5v5": 0.6,
                "game_score": 1.5,
            }
        )
        shot_events_base.append(
            {
                "game_id": gid,
                "season": 20242025,
                "start_time_utc": dt_utc,
                "homeTeamCode": "BOS",
                "awayTeamCode": "TOR",
                "isHomeTeam": True,
                "homeTeamGoals": 1,
                "awayTeamGoals": 1,
                "xCordAdjusted": 76.0,
                "yCordAdjusted": 4.0,
                "xGoal": 0.18,
                "shotRush": 1,
                "shotRebound": 0,
                "shooterPlayerId": 8477956,
            }
        )
        shift_records_base.append(
            {
                "game_id": gid,
                "player_id": 8477956,
                "team": "BOS",
                "qoc_opponent_toi_share": 0.32,
                "qot_teammate_ixg_per_60": 0.85,
            }
        )

    games_df = pl.DataFrame(games_rows)
    team_logs_df = pl.DataFrame(team_log_rows)
    skater_logs_df = pl.DataFrame(skater_log_rows)

    # Mutate shot coordinates, xGoal, score states, rush/rebound flags, and QoC at games >= index 3
    shot_events_mut = [dict(s) for s in shot_events_base]
    shift_records_mut = [dict(sh) for sh in shift_records_base]
    for idx in range(3, 6):
        shot_events_mut[idx]["homeTeamGoals"] = 0
        shot_events_mut[idx]["awayTeamGoals"] = 5
        shot_events_mut[idx]["xCordAdjusted"] = 32.0
        shot_events_mut[idx]["yCordAdjusted"] = 35.0
        shot_events_mut[idx]["xGoal"] = 0.85
        shot_events_mut[idx]["shotRush"] = 1
        shot_events_mut[idx]["shotRebound"] = 1
        shift_records_mut[idx]["qoc_opponent_toi_share"] = 0.58
        shift_records_mut[idx]["qot_teammate_ixg_per_60"] = 1.95

    t_micro_base, s_micro_base = aggregate_shot_events_to_micro_logs(shot_events_base)
    s_micro_base = enrich_skater_micro_logs_with_shifts(
        s_micro_base, compute_shift_overlap_qoc_qot(shift_records_base)
    )

    t_micro_mut, s_micro_mut = aggregate_shot_events_to_micro_logs(shot_events_mut)
    s_micro_mut = enrich_skater_micro_logs_with_shifts(
        s_micro_mut, compute_shift_overlap_qoc_qot(shift_records_mut)
    )

    s1_base = build_stage1_game_features(
        games_df, team_logs_df, skater_logs_df, team_pbp_micro_logs=t_micro_base
    )
    s1_mut = build_stage1_game_features(
        games_df, team_logs_df, skater_logs_df, team_pbp_micro_logs=t_micro_mut
    )

    # All pre-game Stage 1 features for games 0, 1, 2, AND game 3 (since game 3's pre-game features
    # only depend on games 0..2!) must be bit-for-bit identical
    s1_pre_base = s1_base.filter(pl.col("game_id") <= 2024020004)
    s1_pre_mut = s1_mut.filter(pl.col("game_id") <= 2024020004)
    for col in s1_pre_base.columns:
        assert s1_pre_base[col].to_list() == s1_pre_mut[col].to_list(), f"Leakage in Stage 1 {col}"

    # Verify Stage 2 skater rolling micro-features at games <= index 3 are bit-for-bit identical
    tf_base = build_team_rolling_features(team_logs_df, team_pbp_micro_logs=t_micro_base)
    tf_mut = build_team_rolling_features(team_logs_df, team_pbp_micro_logs=t_micro_mut)
    s2_base = build_stage2_player_features(
        skater_logs_df, tf_base, pl.DataFrame(), skater_pbp_micro_logs=s_micro_base
    )
    s2_mut = build_stage2_player_features(
        skater_logs_df, tf_mut, pl.DataFrame(), skater_pbp_micro_logs=s_micro_mut
    )
    s2_pre_base = s2_base.filter(pl.col("game_id") <= 2024020004)
    s2_pre_mut = s2_mut.filter(pl.col("game_id") <= 2024020004)
    for col in (
        "rolling_rush_rebound_ixg_per_60",
        "rolling_qoc_index",
        "rolling_qot_teammate_ixg_per_60",
    ):
        assert s2_pre_base[col].to_list() == s2_pre_mut[col].to_list(), f"Leakage in Stage 2 {col}"


def test_pbp_ingestor_and_silver_reconciler_persistence(tmp_path: Path) -> None:
    """Verify NHLPbpShotsIngestor, parse_nhl_pbp_payload, and HockeySilverReconciler round-trip."""
    mp_dir = tmp_path / "raw" / "moneypuck"
    nhl_dir = tmp_path / "raw" / "nhl_api"
    silver_dir = tmp_path / "silver" / "hockey"
    mp_dir.mkdir(parents=True, exist_ok=True)
    (nhl_dir / "pbp").mkdir(parents=True, exist_ok=True)

    csv_text = (
        "game_id,season,homeTeamCode,awayTeamCode,isHomeTeam,homeTeamGoals,awayTeamGoals,"
        "xCordAdjusted,yCordAdjusted,xGoal,shotRush,shotRebound,shooterPlayerId,homeSkatersOnIce,awaySkatersOnIce\n"
        "2024020001,2024,BOS,TOR,1,2,0,80,4,0.22,1,0,8477956,5,5\n"
    )
    (mp_dir / "shots_2024.csv").write_text(csv_text, encoding="utf-8")

    pbp_json = {
        "id": 2024020002,
        "season": 20242025,
        "gameDate": "2024-10-12",
        "startTimeUTC": "2024-10-12T23:00:00Z",
        "homeTeam": {"id": 6, "abbrev": "BOS"},
        "awayTeam": {"id": 10, "abbrev": "TOR"},
        "plays": [
            {
                "typeDescKey": "shot-on-goal",
                "situationCode": "1551",
                "timeInPeriod": "05:10",
                "periodDescriptor": {"number": 1},
                "details": {
                    "eventOwnerTeamId": 6,
                    "shootingPlayerId": 8477956,
                    "xCoord": 79.0,
                    "yCoord": 3.0,
                },
            },
            {
                "typeDescKey": "goal",
                "situationCode": "1551",
                "timeInPeriod": "05:12",
                "periodDescriptor": {"number": 1},
                "details": {
                    "eventOwnerTeamId": 6,
                    "scoringPlayerId": 8477956,
                    "xCoord": 84.0,
                    "yCoord": 1.0,
                },
            },
        ],
    }
    t_pbp, s_pbp = parse_nhl_pbp_payload(pbp_json)
    assert t_pbp.height == 2
    assert s_pbp.height == 1

    ingestor = NHLPbpShotsIngestor(moneypuck_dir=mp_dir, nhl_api_dir=nhl_dir)
    t_df, s_df = ingestor.load_cached_micro_logs()
    reconciler = HockeySilverReconciler(silver_dir=silver_dir)
    merged_t = reconciler.reconcile_team_pbp_micro_logs(t_df)
    merged_s = reconciler.reconcile_skater_pbp_micro_logs(s_df)

    assert reconciler.team_pbp_micro_logs_path.exists()
    assert reconciler.skater_pbp_micro_logs_path.exists()
    assert merged_t.height == 2
    assert merged_s.height == 1
    assert parse_moneypuck_shots_csv("")[0].is_empty()
    assert date(2024, 10, 12) == t_pbp["game_date"][0]
