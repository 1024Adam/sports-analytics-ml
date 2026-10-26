"""Stage 1 Mandatory Calibrated 3-Way (`1X2`) & Goals Environment model tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from sports_ml.config import RANDOM_SEED
from sports_ml.core.ratings import compute_elo_1x2_probabilities
from sports_ml.sports.soccer.models.stage1_match import (
    Stage1SoccerMatchModel,
    compute_dixon_coles_1x2_probabilities,
    run_soccer_stage1_walk_forward_backtest,
)


def _synthetic_directional_soccer_stage1_df(n_matches: int = 150) -> pl.DataFrame:
    """Create a deterministic chronological EPL Stage 1 dataset with directional 1X2 signal."""
    rng = np.random.default_rng(RANDOM_SEED)
    base_time = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)

    rows = []
    for i in range(n_matches):
        day_offset = i // 2
        ts = base_time + timedelta(days=day_offset, hours=(i % 2) * 2)
        strength_diff = float(rng.uniform(-1.6, 1.6))
        elo_diff = 48.0 + 135.0 * strength_diff
        p_h_elo, p_d_elo, p_a_elo = compute_elo_1x2_probabilities(elo_diff, draw_param=0.26)

        npxgd_home = 0.55 * strength_diff
        npxgd_away = -0.55 * strength_diff
        gk_diff = 0.35 * strength_diff
        fwd_diff = 0.22 * strength_diff

        # True outcome generated from strong directional 1X2 probabilities
        p_h_true, p_d_true, p_a_true = compute_elo_1x2_probabilities(
            48.0 + 240.0 * strength_diff, draw_param=0.24
        )
        outcome_1x2 = int(rng.choice(3, p=[p_h_true, p_d_true, p_a_true]))
        if outcome_1x2 == 0:
            home_goals = max(1, int(round(1.9 + 0.6 * max(0.0, strength_diff))))
            away_goals = max(0, home_goals - int(rng.choice([1, 2])))
        elif outcome_1x2 == 1:
            draw_g = int(rng.choice([0, 1, 2], p=[0.30, 0.50, 0.20]))
            home_goals = draw_g
            away_goals = draw_g
        else:
            away_goals = max(1, int(round(1.8 + 0.6 * max(0.0, -strength_diff))))
            home_goals = max(0, away_goals - int(rng.choice([1, 2])))

        rows.append(
            {
                "sport": "soccer",
                "season": 20242025,
                "matchweek": (i // 10) + 1,
                "match_id": 202400001 + i,
                "event_id": 202400001 + i,
                "event_seq": 1,
                "start_time_utc": ts,
                "home_team": "ARS" if i % 2 == 0 else "LIV",
                "away_team": "MCI" if i % 2 == 0 else "CHE",
                "elo_home_pre": 1500.0 + 67.5 * strength_diff,
                "elo_away_pre": 1500.0 - 67.5 * strength_diff,
                "elo_diff_pre": elo_diff,
                "elo_home_win_prob": p_h_elo,
                "elo_draw_prob": p_d_elo,
                "elo_away_win_prob": p_a_elo,
                "npxgd_per_90_w10_home": npxgd_home,
                "npxgd_per_90_w10_away": npxgd_away,
                "npxgd_per_90_w10_diff": npxgd_home - npxgd_away,
                "npxgf_pct_w10_diff": float(np.clip(0.12 * strength_diff, -0.35, 0.35)),
                "pressing_intensity_diff": 1.2 * strength_diff,
                "deep_completions_per_90_w10_diff": 2.0 * strength_diff,
                "rest_advantage": 1 if strength_diff > 0 else 0,
                "gk_gsaa_per_90_diff": gk_diff,
                "fwd_npxg_per_90_diff": fwd_diff,
                "mid_xg_chain_per_90_diff": 0.20 * strength_diff,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome_1x2": outcome_1x2,
                "home_win": 1 if outcome_1x2 == 0 else 0,
                "is_draw": 1 if outcome_1x2 == 1 else 0,
                "away_win": 1 if outcome_1x2 == 2 else 0,
            }
        )
    return pl.DataFrame(rows)


def test_dixon_coles_and_stage1_probability_bounds_and_sum_to_one() -> None:
    """Verify Dixon-Coles low-score draw adjustment and Stage 1 P_H + P_D + P_A == 1.0."""
    # Negative rho increases draw probability relative to independent Poisson (rho == 0)
    ph_ind, pd_ind, pa_ind = compute_dixon_coles_1x2_probabilities(1.35, 1.35, rho=0.0)
    ph_dc, pd_dc, pa_dc = compute_dixon_coles_1x2_probabilities(1.35, 1.35, rho=-0.12)
    assert ph_ind + pd_ind + pa_ind == pytest.approx(1.0)
    assert ph_dc + pd_dc + pa_dc == pytest.approx(1.0)
    assert pd_dc > pd_ind

    df = _synthetic_directional_soccer_stage1_df(n_matches=120)
    train_df = df.slice(0, 80)
    val_df = df.slice(80, 20)
    test_df = df.slice(100, 20)

    for method in ("sigmoid", "isotonic"):
        model = Stage1SoccerMatchModel(calibration_method=method, random_seed=RANDOM_SEED)
        model.fit(train_df, val_df)
        preds = model.predict(test_df)

        p_h = preds["home_win_prob"].to_numpy()
        p_d = preds["draw_prob"].to_numpy()
        p_a = preds["away_win_prob"].to_numpy()
        exp_h = preds["expected_home_goals"].to_numpy()
        exp_a = preds["expected_away_goals"].to_numpy()

        assert np.all((p_h > 0.0) & (p_h < 1.0))
        assert np.all((p_d > 0.0) & (p_d < 1.0))
        assert np.all((p_a > 0.0) & (p_a < 1.0))
        np.testing.assert_allclose(p_h + p_d + p_a, 1.0, rtol=0.0, atol=1e-9)
        assert np.all(exp_h > 0.0)
        assert np.all(exp_a > 0.0)


def test_soccer_walk_forward_never_fits_on_validation_dates() -> None:
    """Verify walk-forward training strictly isolates GBDT train, calibrator val, and test folds."""
    df = _synthetic_directional_soccer_stage1_df(n_matches=140)

    overlapping_train = df.slice(0, 90)
    overlapping_val = df.slice(70, 40)
    model = Stage1SoccerMatchModel()
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        model.fit(overlapping_train, overlapping_val)

    bt_result = run_soccer_stage1_walk_forward_backtest(
        df,
        n_splits=3,
        min_train_groups=15,
        calibration_method="sigmoid",
    )
    assert len(bt_result.fold_audits) == 3

    for audit in bt_result.fold_audits:
        assert audit.gbdt_train_max_time < audit.calibrator_min_time
        assert audit.calibrator_min_time <= audit.calibrator_max_time
        assert audit.calibrator_max_time < audit.val_min_time
        assert audit.val_min_time <= audit.val_max_time


def test_calibrated_soccer_stage1_beats_epl_baselines() -> None:
    """Verify calibrated 1X2 model achieves lower RPS, Brier Score, and Log Loss than baselines."""
    df = _synthetic_directional_soccer_stage1_df(n_matches=160)
    bt_result = run_soccer_stage1_walk_forward_backtest(
        df,
        n_splits=3,
        min_train_groups=20,
        calibration_method="sigmoid",
    )

    report = bt_result.benchmark_report
    assert report.beats_empirical_prior_rps is True
    assert report.beats_empirical_prior_brier is True
    assert report.beats_empirical_prior_log_loss is True
    assert report.model.rps < report.uniform_baseline.rps
    assert report.model.brier_score < report.uniform_baseline.brier_score
    assert report.model.log_loss < report.uniform_baseline.log_loss
    assert 0.0 <= report.model.ece < 0.25
