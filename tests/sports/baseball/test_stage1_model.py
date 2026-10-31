"""Offline unit tests for Stage 1 MLB (2-Way) & NPB (3-Way 12-Inning) outcome models."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from sports_ml.sports.baseball.models import (
    Stage1BaseballGameModel,
    compute_negative_binomial_game_probabilities,
    run_baseball_stage1_walk_forward_backtest,
)


def _generate_synthetic_directional_baseball_features(
    n_mlb: int = 80,
    n_npb: int = 60,
) -> pl.DataFrame:
    """Create a synthetic chronological MLB + NPB Stage 1 feature DataFrame with strong signal."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    start_dt = datetime(2025, 4, 1, 18, 0, tzinfo=UTC)

    total = n_mlb + n_npb
    for idx in range(total):
        is_npb = idx % 2 == 1 and (idx // 2) < n_npb
        league = "NPB" if is_npb else "MLB"
        ev_time = start_dt + timedelta(hours=12 * idx)
        ev_date: date = ev_time.date()

        # Strong true underlying talent differential
        true_strength = float(rng.uniform(-1.5, 1.5))
        elo_diff = 24.0 + 95.0 * true_strength
        sp_xfip_edge = 0.85 * true_strength
        lineup_woba_edge = 0.032 * true_strength
        bp_xfip_edge = 0.45 * true_strength

        home_sp_xfip = (3.35 if is_npb else 4.15) - 0.5 * sp_xfip_edge
        away_sp_xfip = (3.35 if is_npb else 4.15) + 0.5 * sp_xfip_edge
        home_woba = (0.298 if is_npb else 0.315) + 0.5 * lineup_woba_edge
        away_woba = (0.298 if is_npb else 0.315) - 0.5 * lineup_woba_edge

        if is_npb:
            # In NPB, balanced low-scoring games have a chance of a 12-inning tie
            if abs(true_strength) < 0.15 and idx % 7 == 0:
                home_runs = 2
                away_runs = 2
                outcome_1x2 = 1
                home_win = 0
                is_tie = 1
            elif true_strength >= 0.0:
                home_runs = 5
                away_runs = 1
                outcome_1x2 = 0
                home_win = 1
                is_tie = 0
            else:
                home_runs = 1
                away_runs = 5
                outcome_1x2 = 2
                home_win = 0
                is_tie = 0
        else:
            if true_strength >= -0.05:
                home_runs = 6
                away_runs = 2
                outcome_1x2 = 0
                home_win = 1
                is_tie = 0
            else:
                home_runs = 2
                away_runs = 6
                outcome_1x2 = 2
                home_win = 0
                is_tie = 0

        p_elo_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        p_elo_d = 0.032 * 4.0 * p_elo_h * (1.0 - p_elo_h) if is_npb else 0.0
        p_elo_a = 1.0 - p_elo_h - p_elo_d
        if is_npb:
            p_elo_h -= 0.5 * p_elo_d

        rows.append(
            {
                "sport": "baseball",
                "league": league,
                "season": 2025,
                "game_id": 9_000_000_000 + idx if is_npb else 700_000 + idx,
                "event_id": 9_000_000_000 + idx if is_npb else 700_000 + idx,
                "event_seq": 1,
                "game_date": ev_date,
                "start_time_utc": ev_time,
                "home_team": "HAN" if is_npb else "LAD",
                "away_team": "YOM" if is_npb else "SDP",
                "game_state": "FINAL",
                "home_runs": home_runs,
                "away_runs": away_runs,
                "outcome_1x2": outcome_1x2,
                "home_win": home_win,
                "is_tie": is_tie,
                "elo_home_pre": 1500.0 + 0.5 * elo_diff,
                "elo_away_pre": 1500.0 - 0.5 * elo_diff,
                "diff_elo_pre": elo_diff,
                "elo_home_win_prob": p_elo_h,
                "elo_draw_prob": p_elo_d,
                "elo_away_win_prob": p_elo_a,
                "home_sp_xfip_shrunk": home_sp_xfip,
                "away_sp_xfip_shrunk": away_sp_xfip,
                "diff_sp_xfip_edge": sp_xfip_edge,
                "diff_sp_fip_edge": sp_xfip_edge,
                "home_sp_k_minus_bb_pct_shrunk": 0.16 + 0.04 * true_strength,
                "away_sp_k_minus_bb_pct_shrunk": 0.16 - 0.04 * true_strength,
                "diff_sp_k_minus_bb_pct": 0.08 * true_strength,
                "home_sp_gb_pct_shrunk": 0.45,
                "away_sp_gb_pct_shrunk": 0.44,
                "home_sp_expected_outs_ceiling": 18.0,
                "away_sp_expected_outs_ceiling": 17.5,
                "diff_sp_expected_outs": 0.5,
                "home_sp_days_rest": 5.0,
                "away_sp_days_rest": 5.0,
                "home_sp_xwoba_against_shrunk": 0.305 - 0.02 * true_strength,
                "away_sp_xwoba_against_shrunk": 0.305 + 0.02 * true_strength,
                "home_sp_swstr_pct_shrunk": 0.12,
                "away_sp_swstr_pct_shrunk": 0.11,
                "home_sp_velo_delta_mph": 0.25 * true_strength,
                "away_sp_velo_delta_mph": -0.25 * true_strength,
                "diff_sp_velo_delta": 0.50 * true_strength,
                "diff_sp_swstr_pct": 0.01 * true_strength,
                "home_bp_xfip_shrunk": 3.80 - 0.5 * bp_xfip_edge,
                "away_bp_xfip_shrunk": 3.80 + 0.5 * bp_xfip_edge,
                "diff_bp_xfip_edge": bp_xfip_edge,
                "home_bp_k_minus_bb_pct_shrunk": 0.16,
                "away_bp_k_minus_bb_pct_shrunk": 0.15,
                "home_bp_pitches_last_1d": 20.0,
                "away_bp_pitches_last_1d": 25.0,
                "home_bp_pitches_last_3d": 70.0,
                "away_bp_pitches_last_3d": 80.0,
                "diff_bp_pitches_last_3d": -10.0,
                "home_high_leverage_arms_unavailable_count": 0.0,
                "away_high_leverage_arms_unavailable_count": 1.0 if true_strength > 0.5 else 0.0,
                "diff_hl_arms_unavailable": -1.0 if true_strength > 0.5 else 0.0,
                "home_lineup_platoon_woba": home_woba,
                "away_lineup_platoon_woba": away_woba,
                "diff_lineup_platoon_woba": lineup_woba_edge,
                "home_lineup_platoon_iso": 0.160 + 0.02 * true_strength,
                "away_lineup_platoon_iso": 0.160 - 0.02 * true_strength,
                "diff_lineup_platoon_iso": 0.04 * true_strength,
                "home_lineup_platoon_k_pct": 0.21,
                "away_lineup_platoon_k_pct": 0.22,
                "diff_lineup_platoon_k_pct": -0.01,
                "home_lineup_same_hand_share": 0.35,
                "away_lineup_same_hand_share": 0.45,
                "home_missing_regular_pa_woba": 0.0,
                "away_missing_regular_pa_woba": 0.0,
                "diff_missing_regular_woba": 0.0,
                "diff_lineup_xwoba": lineup_woba_edge,
                "diff_woba_minus_xwoba_regression": 0.0,
                "home_team_runs_for_per_g_shrunk": 4.5 + 0.6 * true_strength,
                "away_team_runs_for_per_g_shrunk": 4.5 - 0.6 * true_strength,
                "home_team_runs_against_per_g_shrunk": 4.2 - 0.5 * true_strength,
                "away_team_runs_against_per_g_shrunk": 4.2 + 0.5 * true_strength,
                "diff_team_run_diff": 2.2 * true_strength,
                "diff_team_woba": lineup_woba_edge,
                "diff_pythag_luck": 0.0,
                "diff_baseruns_luck": 0.0,
                "venue_run_factor": 0.91 if is_npb else 1.00,
                "venue_hr_factor_matchup": 0.85 if is_npb else 1.05,
                "is_dome_flag": 0.0,
                "is_neutral_or_regional_flag": 0.0,
                "is_koshien_summer_road_trip_flag": 0.0,
                "is_interleague_flag": 0.0,
                "uses_dh_flag": 0.0 if is_npb else 1.0,
                "is_doubleheader_g2": 0.0,
                "diff_rest_days": 0.0,
                "away_travel_km": 450.0,
            }
        )

    return pl.DataFrame(rows)


def test_negative_binomial_mlb_2way_and_npb_12_inning_tie_physics() -> None:
    """Verify MLB has draw_prob==0 and NPB draw_prob rises in low-run pitching duels."""
    # 1. MLB unlimited extra innings: draw_prob == 0.0, p_home + p_away == 1.0
    mlb_ph, mlb_pd, mlb_pa = compute_negative_binomial_game_probabilities(4.6, 4.2, league="MLB")
    assert mlb_pd == 0.0
    assert 0.0 < mlb_ph < 1.0
    assert 0.0 < mlb_pa < 1.0
    assert pytest.approx(mlb_ph + mlb_pa, abs=1e-9) == 1.0
    assert mlb_ph > 0.50

    # 2. NPB 12-inning cap: low-scoring pitcher's duel (2.8 vs 2.8) vs slugfest (5.5 vs 5.5)
    npb_low_ph, npb_low_pd, npb_low_pa = compute_negative_binomial_game_probabilities(
        2.8, 2.8, league="NPB"
    )
    npb_high_ph, npb_high_pd, npb_high_pa = compute_negative_binomial_game_probabilities(
        5.5, 5.5, league="NPB"
    )
    npb_mismatch_ph, npb_mismatch_pd, npb_mismatch_pa = (
        compute_negative_binomial_game_probabilities(5.8, 2.2, league="NPB")
    )

    assert pytest.approx(npb_low_ph + npb_low_pd + npb_low_pa, abs=1e-9) == 1.0
    assert pytest.approx(npb_high_ph + npb_high_pd + npb_high_pa, abs=1e-9) == 1.0
    assert pytest.approx(npb_mismatch_ph + npb_mismatch_pd + npb_mismatch_pa, abs=1e-9) == 1.0

    # Low-run pitching duel must have strictly higher 12-inning tie probability than slugfest
    assert npb_low_pd > npb_high_pd > npb_mismatch_pd > 0.0


def test_stage1_baseball_model_probability_invariants_and_leak_guard() -> None:
    """Verify MLB 2-way and NPB 3-way probability sum-to-one invariants and temporal guard."""
    df = _generate_synthetic_directional_baseball_features(n_mlb=60, n_npb=50)
    model = Stage1BaseballGameModel()
    model.fit(df.head(85), df.slice(85, 25))

    preds = model.predict(df.tail(25))
    for r in preds.iter_rows(named=True):
        ph = float(r["home_win_prob"])
        pd = float(r["draw_prob"])
        pa = float(r["away_win_prob"])
        ph_dnb = float(r["home_dnb_prob"])
        pa_dnb = float(r["away_dnb_prob"])
        assert 0.0 < ph < 1.0
        assert 0.0 < pa < 1.0
        assert pytest.approx(ph_dnb + pa_dnb, abs=1e-9) == 1.0
        assert 1.0 <= float(r["expected_home_runs"]) <= 11.0
        assert 1.0 <= float(r["expected_away_runs"]) <= 11.0

        if r["league"] == "MLB":
            assert pd == 0.0
            assert pytest.approx(ph + pa, abs=1e-9) == 1.0
            assert pytest.approx(ph_dnb, abs=1e-9) == ph
        else:
            assert 0.0 < pd < 1.0
            assert pytest.approx(ph + pd + pa, abs=1e-9) == 1.0
            assert pytest.approx(ph_dnb, abs=1e-9) == ph / (ph + pa)

    # Verify explicit temporal overlap rejection between train_df and val_df
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        Stage1BaseballGameModel().fit(df.tail(30), df.head(30))


def test_walk_forward_backtest_temporal_audit_and_directional_skill() -> None:
    """Verify walk-forward fold boundary audit and benchmark skill across MLB & NPB."""
    df = _generate_synthetic_directional_baseball_features(n_mlb=80, n_npb=60)
    res = run_baseball_stage1_walk_forward_backtest(
        df,
        n_splits=3,
        min_train_groups=20,
    )

    assert len(res.fold_audits) >= 2
    for audit in res.fold_audits:
        assert audit.gbdt_train_max_time < audit.calibrator_min_time
        assert audit.calibrator_min_time <= audit.calibrator_max_time
        assert audit.calibrator_max_time < audit.val_min_time

    assert res.mlb_benchmark_report is not None
    assert res.npb_benchmark_report is not None
    assert res.npb_dnb_log_loss is not None

    # Calibrated MLB model beats constant home-win-rate baseline on Brier Score and Log Loss
    assert (
        res.mlb_benchmark_report.model.brier_score
        < res.mlb_benchmark_report.home_win_rate_baseline.brier_score
    )
    assert (
        res.mlb_benchmark_report.model.log_loss
        < res.mlb_benchmark_report.home_win_rate_baseline.log_loss
    )

    # Calibrated NPB 3-way model beats empirical 1X2 prior on RPS, Brier Score, and Log Loss
    assert (
        res.npb_benchmark_report.model.rps < res.npb_benchmark_report.empirical_prior_baseline.rps
    )
    assert (
        res.npb_benchmark_report.model.brier_score
        < res.npb_benchmark_report.empirical_prior_baseline.brier_score
    )
    assert (
        res.npb_benchmark_report.model.log_loss
        < res.npb_benchmark_report.empirical_prior_baseline.log_loss
    )
