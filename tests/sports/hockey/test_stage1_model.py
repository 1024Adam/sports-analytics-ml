"""Stage 1 Mandatory Calibrated Game Win Probability model tests (`test_stage1_model.py`)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from sports_ml.config import RANDOM_SEED
from sports_ml.core.calibration import ProbabilityCalibrator, calibrate_multi_entrant_group
from sports_ml.core.evaluation import compare_against_baselines, expected_calibration_error
from sports_ml.sports.hockey.models.stage1_game import (
    Stage1GameOutcomeModel,
    run_stage1_walk_forward_backtest,
)


def _synthetic_directional_stage1_df(n_games: int = 120) -> pl.DataFrame:
    """Create a deterministic chronological Stage 1 dataset with a clear predictive signal."""
    rng = np.random.default_rng(RANDOM_SEED)
    base_time = datetime(2024, 10, 1, 23, 0, tzinfo=UTC)

    rows = []
    for i in range(n_games):
        # 2 games per day across 60 days
        day_offset = i // 2
        ts = base_time + timedelta(days=day_offset, hours=i % 2)
        strength_diff = float(rng.uniform(-1.5, 1.5))
        elo_diff = 35.0 + 120.0 * strength_diff
        elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        xgf_pct_home = float(np.clip(0.50 + 0.08 * strength_diff, 0.35, 0.65))
        xgf_pct_away = float(np.clip(0.50 - 0.08 * strength_diff, 0.35, 0.65))
        gsax_diff = 0.60 * strength_diff

        true_logit = 1.8 * strength_diff
        true_prob = 1.0 / (1.0 + np.exp(-true_logit))
        home_win = 1 if float(rng.uniform(0.0, 1.0)) < true_prob else 0
        home_reg_goals = max(1, int(round(3.2 + 1.1 * strength_diff)))
        away_reg_goals = max(1, int(round(2.9 - 1.0 * strength_diff)))

        rows.append(
            {
                "sport": "hockey",
                "season": 20242025,
                "game_id": 2024020001 + i,
                "event_id": 2024020001 + i,
                "event_seq": 1,
                "start_time_utc": ts,
                "home_team": "BOS" if i % 2 == 0 else "COL",
                "away_team": "TOR" if i % 2 == 0 else "MTL",
                "elo_home_pre": 1500.0 + 60.0 * strength_diff,
                "elo_away_pre": 1500.0 - 60.0 * strength_diff,
                "elo_diff_pre": elo_diff,
                "elo_home_win_prob": elo_prob,
                "xgf_pct_5v5_w10_home": xgf_pct_home,
                "xgf_pct_5v5_w10_away": xgf_pct_away,
                "xgf_pct_5v5_w10_diff": xgf_pct_home - xgf_pct_away,
                "goalie_gsax_per_60_diff": gsax_diff,
                "forward_ixg_per_60_diff": 0.25 * strength_diff,
                "st_net_edge_home_vs_away": 0.10 * strength_diff,
                "rest_advantage": 1 if strength_diff > 0 else 0,
                "home_reg_goals": home_reg_goals,
                "away_reg_goals": away_reg_goals,
                "home_goals": home_reg_goals,
                "away_goals": away_reg_goals,
                "home_win": home_win,
            }
        )
    return pl.DataFrame(rows)


def test_stage1_probability_bounds_and_sum_to_one() -> None:
    """Verify predicted probabilities lie strictly in (0, 1) and P(Home) + P(Away) == 1.0."""
    df = _synthetic_directional_stage1_df(n_games=100)
    train_df = df.slice(0, 70)
    val_df = df.slice(70, 15)
    test_df = df.slice(85, 15)

    for method in ("sigmoid", "isotonic"):
        model = Stage1GameOutcomeModel(calibration_method=method, random_seed=RANDOM_SEED)
        model.fit(train_df, val_df)
        preds = model.predict(test_df)

        home_p = preds["home_win_prob"].to_numpy()
        away_p = preds["away_win_prob"].to_numpy()
        exp_h_goals = preds["expected_home_goals"].to_numpy()
        exp_a_goals = preds["expected_away_goals"].to_numpy()

        assert np.all((home_p > 0.0) & (home_p < 1.0))
        assert np.all((away_p > 0.0) & (away_p < 1.0))
        np.testing.assert_allclose(home_p + away_p, 1.0, rtol=0.0, atol=1e-12)
        assert np.all(exp_h_goals > 0.0)
        assert np.all(exp_a_goals > 0.0)


def test_walk_forward_never_fits_gbdt_or_calibrator_on_validation_dates() -> None:
    """Verify walk-forward training strictly isolates GBDT train, calibrator val, and test folds."""
    df = _synthetic_directional_stage1_df(n_games=120)

    # 1. Explicit temporal leakage guard when train_df overlaps val_df
    overlapping_train = df.slice(0, 80)
    overlapping_val = df.slice(60, 40)
    model = Stage1GameOutcomeModel()
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        model.fit(overlapping_train, overlapping_val)

    # 2. Walk-forward backtest chronological audit trail
    bt_result = run_stage1_walk_forward_backtest(
        df,
        n_splits=3,
        min_train_groups=15,
        calibration_method="sigmoid",
    )
    assert len(bt_result.fold_audits) == 3

    for audit in bt_result.fold_audits:
        # GBDT training timestamps are strictly < Calibrator timestamps < Validation timestamps
        assert audit.gbdt_train_max_time < audit.calibrator_min_time
        assert audit.calibrator_min_time <= audit.calibrator_max_time
        assert audit.calibrator_max_time < audit.val_min_time
        assert audit.val_min_time <= audit.val_max_time


def test_calibrated_model_beats_constant_home_win_rate_baseline() -> None:
    """Verify calibrated model achieves lower Brier Score and Log Loss than constant baseline."""
    df = _synthetic_directional_stage1_df(n_games=140)
    bt_result = run_stage1_walk_forward_backtest(
        df,
        n_splits=3,
        min_train_groups=20,
        calibration_method="sigmoid",
    )

    report = bt_result.benchmark_report
    assert report.beats_home_baseline_brier is True
    assert report.beats_home_baseline_log_loss is True
    assert report.model.brier_score < report.coin_flip_baseline.brier_score
    assert report.model.log_loss < report.coin_flip_baseline.log_loss
    assert 0.0 <= report.model.ece < 0.25
    assert report.model.accuracy > 0.60


def test_calibration_and_evaluation_edge_cases() -> None:
    """Verify multi-entrant softmax calibration and ECE calculation."""
    group_probs = calibrate_multi_entrant_group(np.array([2.0, 1.0, 0.0], dtype=np.float64))
    assert float(np.sum(group_probs)) == pytest.approx(1.0)
    assert group_probs[0] > group_probs[1] > group_probs[2]

    cal = ProbabilityCalibrator(method="sigmoid")
    with pytest.raises(RuntimeError, match="must be fitted"):
        cal.predict_proba(np.array([0.5], dtype=np.float64))

    ece = expected_calibration_error(
        np.array([0.8, 0.8, 0.2, 0.2], dtype=np.float64),
        np.array([1, 1, 0, 0], dtype=np.int64),
        n_bins=5,
    )
    assert 0.0 <= ece <= 0.25

    comp = compare_against_baselines(
        np.array([0.85, 0.80, 0.15, 0.20], dtype=np.float64),
        np.array([1, 1, 0, 0], dtype=np.int64),
        train_home_win_rate=0.55,
    )
    assert comp.beats_home_baseline_brier is True
