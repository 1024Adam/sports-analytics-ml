"""EPL Understat shot-event micro-feature & conservation tests (`test_shot_micro_features.py`)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sports_ml.sports.soccer.features.player import build_soccer_stage2_player_features
from sports_ml.sports.soccer.features.team import (
    build_soccer_stage1_match_features,
    build_soccer_team_rolling_features,
)
from sports_ml.sports.soccer.ingestion.reconciler import SoccerSilverReconciler
from sports_ml.sports.soccer.ingestion.shot_events import (
    SoccerShotEventsIngestor,
    classify_shot_situation,
    is_18_yard_box_shot,
    parse_understat_match_shots_payload,
)


def test_open_play_set_piece_penalty_xg_conservation_and_spatial_box_shares() -> None:
    """Verify open_play_xgf + set_piece_xgf + penalty_xgf == total_xgf (within 1e-6)."""
    shots_payload = {
        "h": [
            {
                "id": 1,
                "match_id": 101,
                "season": 2024,
                "date": "2024-08-17",
                "minute": 12,
                "h_team": "Arsenal",
                "a_team": "Wolves",
                "h_a": "h",
                "situation": "OpenPlay",
                "lastAction": "Throughball",
                "X": 0.88,
                "Y": 0.48,
                "xG": 0.34,
                "result": "Goal",
                "player_id": 1001,
                "player": "Bukayo Saka",
            },
            {
                "id": 2,
                "match_id": 101,
                "season": 2024,
                "date": "2024-08-17",
                "minute": 38,
                "h_team": "Arsenal",
                "a_team": "Wolves",
                "h_a": "h",
                "situation": "FromCorner",
                "lastAction": "Cross",
                "X": 0.95,
                "Y": 0.50,
                "xG": 0.28,
                "result": "SavedShot",
                "player_id": 1002,
                "player": "Gabriel",
            },
            {
                "id": 3,
                "match_id": 101,
                "season": 2024,
                "date": "2024-08-17",
                "minute": 75,
                "h_team": "Arsenal",
                "a_team": "Wolves",
                "h_a": "h",
                "situation": "Penalty",
                "lastAction": "Standard",
                "X": 0.885,
                "Y": 0.50,
                "xG": 0.76,
                "result": "Goal",
                "player_id": 1001,
                "player": "Bukayo Saka",
            },
        ],
        "a": [
            {
                "id": 4,
                "match_id": 101,
                "season": 2024,
                "date": "2024-08-17",
                "minute": 55,
                "h_team": "Arsenal",
                "a_team": "Wolves",
                "h_a": "a",
                "situation": "DirectFreekick",
                "lastAction": "Standard",
                "X": 0.76,
                "Y": 0.52,
                "xG": 0.07,
                "result": "MissedShots",
                "player_id": 2001,
                "player": "Matheus Cunha",
            },
            {
                "id": 5,
                "match_id": 101,
                "season": 2024,
                "date": "2024-08-17",
                "minute": 84,
                "h_team": "Arsenal",
                "a_team": "Wolves",
                "h_a": "a",
                "situation": "OpenPlay",
                "lastAction": "Standard",
                "X": 0.78,
                "Y": 0.30,
                "xG": 0.04,
                "result": "BlockedShot",
                "player_id": 2001,
                "player": "Matheus Cunha",
            },
        ],
    }

    team_micro_df, player_micro_df = parse_understat_match_shots_payload(shots_payload)
    assert team_micro_df.height == 2
    assert player_micro_df.height == 3

    for row in team_micro_df.iter_rows(named=True):
        sum_for = row["open_play_xgf"] + row["set_piece_xgf"] + row["penalty_xgf"]
        sum_against = row["open_play_xga"] + row["set_piece_xga"] + row["penalty_xga"]
        assert sum_for == pytest.approx(row["total_xgf"], abs=1e-6)
        assert sum_against == pytest.approx(row["total_xga"], abs=1e-6)

    ars_row = team_micro_df.filter(pl.col("team") == "ARS").row(0, named=True)
    assert ars_row["open_play_xgf"] == pytest.approx(0.34, abs=1e-6)
    assert ars_row["set_piece_xgf"] == pytest.approx(0.28, abs=1e-6)
    assert ars_row["penalty_xgf"] == pytest.approx(0.76, abs=1e-6)
    assert ars_row["throughball_xgf"] == pytest.approx(0.34, abs=1e-6)
    assert ars_row["box_shot_share"] == pytest.approx(1.0, abs=1e-6)
    assert ars_row["six_yard_shot_share"] == pytest.approx(1.0 / 3.0, abs=1e-6)

    # Wolves shot #5 at minute 84 occurred while trailing 0-2 -> excluded from tied_state_npxgd
    assert classify_shot_situation("FromCorner") == "set_piece"
    assert classify_shot_situation("Penalty") == "penalty"
    assert classify_shot_situation("OpenPlay") == "open_play"
    assert is_18_yard_box_shot(0.95, 0.50) == (True, True)
    assert is_18_yard_box_shot(0.75, 0.50) == (False, False)


def test_strict_temporal_anti_leakage_on_soccer_shot_micro_features() -> None:
    """Verify mutating shot coords, situations, or score states at >= t leaves < t unchanged."""
    base_dt = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
    match_rows = []
    team_log_rows = []
    outfield_rows = []
    shots_base = []

    for i in range(6):
        mid = 202400001 + i
        dt_utc = base_dt + timedelta(days=i * 7)
        m_date = dt_utc.date()
        match_rows.append(
            {
                "sport": "soccer",
                "season": 20242025,
                "matchweek": i + 1,
                "match_id": mid,
                "event_seq": 1,
                "match_date": m_date,
                "start_time_utc": dt_utc,
                "home_team": "ARS",
                "away_team": "MCI",
                "venue_timezone": "Europe/London",
                "game_state": "FINAL",
                "home_goals": 2,
                "away_goals": 1,
                "outcome_1x2": 0,
                "home_win": 1,
                "is_draw": 0,
                "away_win": 0,
            }
        )
        for t_c, o_c, is_h in (("ARS", "MCI", True), ("MCI", "ARS", False)):
            team_log_rows.append(
                {
                    "match_id": mid,
                    "match_date": m_date,
                    "start_time_utc": dt_utc,
                    "season": 20242025,
                    "team": t_c,
                    "opponent": o_c,
                    "is_home": is_h,
                    "goals_for": 2 if is_h else 1,
                    "goals_against": 1 if is_h else 2,
                    "xgf": 1.6 if is_h else 1.1,
                    "xga": 1.1 if is_h else 1.6,
                    "npxgf": 1.6 if is_h else 1.1,
                    "npxga": 1.1 if is_h else 1.6,
                }
            )
        outfield_rows.append(
            {
                "match_id": mid,
                "match_date": m_date,
                "start_time_utc": dt_utc,
                "season": 20242025,
                "player_id": 1001,
                "player_name": "Bukayo Saka",
                "team": "ARS",
                "opponent": "MCI",
                "is_home": True,
                "position_group": "FWD",
                "is_starter": True,
                "minutes": 90.0,
                "goals": 1,
                "assists": 0,
                "shots": 3,
                "sot": 2,
                "key_passes": 2,
                "xg": 0.45,
                "npxg": 0.45,
                "xa": 0.25,
                "xg_chain": 0.65,
                "xg_buildup": 0.20,
            }
        )
        shots_base.append(
            {
                "id": i * 10 + 1,
                "match_id": mid,
                "season": 20242025,
                "start_time_utc": dt_utc,
                "minute": 25,
                "h_team": "ARS",
                "a_team": "MCI",
                "h_a": "h",
                "situation": "OpenPlay",
                "lastAction": "Throughball",
                "X": 0.89,
                "Y": 0.49,
                "xG": 0.30,
                "result": "Goal",
                "player_id": 1001,
                "player": "Bukayo Saka",
            }
        )

    matches_df = pl.DataFrame(match_rows)
    team_logs_df = pl.DataFrame(team_log_rows)
    outfield_df = pl.DataFrame(outfield_rows)

    # Mutate shot events for matches >= index 3
    shots_mut = [dict(s) for s in shots_base]
    for idx in range(3, 6):
        shots_mut[idx]["situation"] = "FromCorner"
        shots_mut[idx]["lastAction"] = "Cross"
        shots_mut[idx]["X"] = 0.68
        shots_mut[idx]["Y"] = 0.15
        shots_mut[idx]["xG"] = 0.92
        shots_mut[idx]["score_diff"] = -3

    t_micro_base, p_micro_base = parse_understat_match_shots_payload(shots_base)
    t_micro_mut, p_micro_mut = parse_understat_match_shots_payload(shots_mut)

    s1_base = build_soccer_stage1_match_features(
        matches_df, team_logs_df, outfield_df, team_shot_micro_logs=t_micro_base
    )
    s1_mut = build_soccer_stage1_match_features(
        matches_df, team_logs_df, outfield_df, team_shot_micro_logs=t_micro_mut
    )

    # Matches 0, 1, 2, AND match 3 (whose pre-match features only depend on 0..2) are unchanged
    s1_pre_base = s1_base.filter(pl.col("match_id") <= 202400004)
    s1_pre_mut = s1_mut.filter(pl.col("match_id") <= 202400004)
    for col in s1_pre_base.columns:
        assert s1_pre_base[col].to_list() == s1_pre_mut[col].to_list(), f"Leakage in Stage 1 {col}"

    tf_base = build_soccer_team_rolling_features(team_logs_df, team_shot_micro_logs=t_micro_base)
    tf_mut = build_soccer_team_rolling_features(team_logs_df, team_shot_micro_logs=t_micro_mut)
    s2_base = build_soccer_stage2_player_features(
        outfield_df, tf_base, pl.DataFrame(), player_shot_micro_logs=p_micro_base
    )
    s2_mut = build_soccer_stage2_player_features(
        outfield_df, tf_mut, pl.DataFrame(), player_shot_micro_logs=p_micro_mut
    )
    s2_pre_base = s2_base.filter(pl.col("match_id") <= 202400004)
    s2_pre_mut = s2_mut.filter(pl.col("match_id") <= 202400004)
    for col in ("rolling_open_play_npxg_per_90", "rolling_set_piece_xg_share"):
        assert s2_pre_base[col].to_list() == s2_pre_mut[col].to_list(), f"Leakage in Stage 2 {col}"


def test_soccer_shot_ingestor_and_silver_reconciler_persistence(tmp_path: Path) -> None:
    """Verify SoccerShotEventsIngestor and SoccerSilverReconciler persist shot micro logs."""
    from sports_ml.core.cache import write_cached_json

    us_dir = tmp_path / "raw" / "understat"
    silver_dir = tmp_path / "silver" / "soccer"
    us_dir.mkdir(parents=True, exist_ok=True)

    write_cached_json(
        us_dir / "epl_shots_2024.json.gz",
        [
            {
                "id": 1,
                "match_id": 202400001,
                "season": 20242025,
                "date": "2024-08-17",
                "minute": 10,
                "h_team": "ARS",
                "a_team": "WOL",
                "h_a": "h",
                "situation": "OpenPlay",
                "lastAction": "Throughball",
                "X": 0.90,
                "Y": 0.50,
                "xG": 0.35,
                "result": "Goal",
                "player_id": 1001,
                "player": "Bukayo Saka",
            }
        ],
    )

    ingestor = SoccerShotEventsIngestor(understat_dir=us_dir)
    t_df, p_df = ingestor.load_cached_shot_micro_logs()
    assert t_df.height == 2
    assert p_df.height == 1

    reconciler = SoccerSilverReconciler(silver_dir=silver_dir)
    merged = reconciler.reconcile_team_shot_micro_logs(t_df)
    assert reconciler.team_shot_micro_logs_path.exists()
    assert merged.height == 2
    assert date(2024, 8, 17) == merged["match_date"][0]
    assert np.all(merged["total_xgf"].to_numpy() >= 0.0)
