"""Offline unit and anti-leakage tests for MLB Statcast & NPB DIPS micro-features."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from sports_ml.sports.baseball.ingestion import (
    aggregate_statcast_pitches_to_micro_logs,
    compute_npb_dips_and_tactical_micro_features,
    compute_npb_venue_park_interaction,
    compute_pregame_batter_woba_minus_xwoba_luck,
    compute_pregame_pitcher_velo_delta,
)


def test_statcast_pitch_aggregation_velo_delta_and_anti_leakage() -> None:
    """Verify -1.8 mph velocity drop detection and strict future-mutation invariance."""
    # 1. Test pitch-level aggregation into pitcher & batter micro-logs
    pitches_df = pl.DataFrame(
        {
            "game_id": [101, 101, 101, 101],
            "event_seq": [1, 1, 1, 1],
            "game_date": [date(2025, 5, 1)] * 4,
            "start_time_utc": [datetime(2025, 5, 1, 20, 0, tzinfo=UTC)] * 4,
            "season": [2025] * 4,
            "league": ["MLB"] * 4,
            "pitcher_id": [501] * 4,
            "pitcher_team": ["LAD"] * 4,
            "batter_id": [1001, 1001, 1002, 1002],
            "batter_team": ["SDP"] * 4,
            "pitch_type": ["FF", "SL", "FF", "CU"],
            "release_speed": [96.2, 87.5, 95.8, 81.0],
            "is_whiff": [False, True, False, False],
            "is_out_of_zone": [False, True, True, False],
            "is_swing": [True, True, False, True],
            "is_first_pitch": [True, False, True, False],
            "is_strike": [True, True, False, True],
            "launch_speed": [104.5, None, None, 98.2],
            "launch_angle": [18.0, None, None, 12.0],
            "is_barrel": [True, False, False, False],
            "estimated_woba_using_speedangle": [0.680, 0.0, 0.0, 0.420],
        }
    )

    p_micro, b_micro = aggregate_statcast_pitches_to_micro_logs(pitches_df)
    assert p_micro.height == 1
    assert b_micro.height == 2
    assert pytest.approx(float(p_micro["avg_fastball_velo"][0]), abs=1e-6) == 96.0
    assert pytest.approx(float(p_micro["swstr_pct"][0]), abs=1e-6) == 0.25

    # 2. Test -1.8 mph velocity drop over last 2 starts before Start 7 (t = 7)
    # Starts 0..3: 96.0 mph; Starts 4..5: 94.2 mph (-1.8 mph drop!); Start 6 (t=7): current start
    start_dt = datetime(2025, 4, 1, 20, 0, tzinfo=UTC)
    velos = [96.0, 96.0, 96.0, 96.0, 94.2, 94.2, 95.5]
    sp_starts = pl.DataFrame(
        {
            "game_id": [1000 + i for i in range(7)],
            "event_seq": [1] * 7,
            "game_date": [(start_dt + timedelta(days=5 * i)).date() for i in range(7)],
            "start_time_utc": [start_dt + timedelta(days=5 * i) for i in range(7)],
            "season": [2025] * 7,
            "player_id": [501] * 7,
            "avg_fastball_velo": velos,
        }
    )

    base_res = compute_pregame_pitcher_velo_delta(sp_starts, short_window=2, baseline_window=12)
    # At Start index 6 (after 4 starts at 96.0 and 2 starts at 94.2):
    # short_velo (last 2 starts: 4, 5) = 94.2 mph
    # base_velo (all 6 prior starts: 4*96.0 + 2*94.2)/6 = 95.4 mph
    # sp_velo_delta_mph = 94.2 - 95.4 = -1.20 mph (< -1.0 mph drop!)
    assert float(base_res["sp_velo_delta_mph"][6]) < -1.0
    assert pytest.approx(float(base_res["sp_velo_delta_mph"][6]), abs=1e-6) == -1.20

    # Strict Anti-Leakage Test: Mutate Start index 6's velocity to 80.0 mph
    # and verify Starts 0..6 pre-game deltas are bit-for-bit identical!
    mut_starts = sp_starts.with_columns(
        pl.when(pl.col("game_id") == 1006)
        .then(pl.lit(80.0))
        .otherwise(pl.col("avg_fastball_velo"))
        .alias("avg_fastball_velo")
    )
    mut_res = compute_pregame_pitcher_velo_delta(mut_starts, short_window=2, baseline_window=12)
    assert base_res["sp_velo_delta_mph"].to_list() == mut_res["sp_velo_delta_mph"].to_list()


def test_batter_woba_minus_xwoba_luck_and_npb_dips_micro_features() -> None:
    """Verify batter BABIP luck regression signal and NPB DIPS/tactical micro-features."""
    start_dt = datetime(2025, 5, 1, 20, 0, tzinfo=UTC)
    batter_logs = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "event_seq": [1, 1, 1],
            "game_date": [(start_dt + timedelta(days=i)).date() for i in range(3)],
            "start_time_utc": [start_dt + timedelta(days=i) for i in range(3)],
            "season": [2025, 2025, 2025],
            "player_id": [10, 10, 10],
            "pa": [4, 4, 4],
            "woba_num": [2.0, 2.0, 0.0],  # 0.500 wOBA in games 1 & 2
            "xwoba": [0.280, 0.280, 0.450],  # Only 0.280 xwOBA in games 1 & 2 -> +0.220 luck!
        }
    )
    luck_df = compute_pregame_batter_woba_minus_xwoba_luck(batter_logs, window=5)
    # Game 1 (idx 0): 0.0; Game 3 (idx 2): sees Games 1 & 2 (wOBA=0.500, xwOBA=0.280 -> +0.220)
    assert pytest.approx(float(luck_df["batter_woba_minus_xwoba_luck"][0]), abs=1e-9) == 0.0
    assert pytest.approx(float(luck_df["batter_woba_minus_xwoba_luck"][2]), abs=1e-6) == 0.220

    # NPB DIPS & tactical micro-features
    npb_team_logs = pl.DataFrame(
        {
            "game_id": [9001, 9002],
            "event_seq": [1, 1],
            "game_date": [date(2025, 6, 1), date(2025, 6, 2)],
            "start_time_utc": [
                datetime(2025, 6, 1, 9, 0, tzinfo=UTC),
                datetime(2025, 6, 2, 9, 0, tzinfo=UTC),
            ],
            "season": [2025, 2025],
            "team": ["HAN", "HAN"],
            "pa_for": [38, 36],
            "k_for": [7, 8],
            "bb_for": [4, 3],
            "hr_for": [1, 0],
            "sac_bunts": [2, 1],
            "team_woba": [0.320, 0.280],
            "team_xwoba": [None, None],
        }
    )
    npb_micro = compute_npb_dips_and_tactical_micro_features(npb_team_logs)
    assert "pregame_team_xwoba" in npb_micro.columns
    assert "pregame_sac_bunt_rate" in npb_micro.columns
    assert float(npb_micro["pregame_team_xwoba"][1]) > 0.200

    # High-HR Meiji Jingu vs low-HR Koshien park interaction for a power lineup
    jingu_edge = compute_npb_venue_park_interaction("YAK", lineup_iso=0.185, sp_gb_pct=0.40)
    koshien_edge = compute_npb_venue_park_interaction("HAN", lineup_iso=0.185, sp_gb_pct=0.52)
    assert jingu_edge > koshien_edge
