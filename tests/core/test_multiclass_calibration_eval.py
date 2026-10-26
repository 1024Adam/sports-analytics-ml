"""Multiclass (`1X2`) calibration, RPS evaluation, and promotion/relegation Elo tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.core.calibration import (
    MulticlassProbabilityCalibrator,
    normalize_multiclass_probabilities,
)
from sports_ml.core.evaluation import (
    compare_multiclass_against_baselines,
    evaluate_multiclass_probabilities,
    multiclass_brier_score,
    ranked_probability_score,
)
from sports_ml.core.ratings import (
    EloConfig,
    PairwiseEloEngine,
    compute_elo_1x2_probabilities,
)


def test_compute_elo_1x2_probabilities_simplex_and_symmetry() -> None:
    """Verify analytical 3-way Elo probabilities sum to 1.0 and peak draw at 0."""
    p_h0, p_d0, p_a0 = compute_elo_1x2_probabilities(0.0, draw_param=0.26)
    assert p_h0 + p_d0 + p_a0 == pytest.approx(1.0)
    assert p_d0 == pytest.approx(0.26)
    assert p_h0 == pytest.approx(p_a0)

    p_h_fav, p_d_fav, p_a_fav = compute_elo_1x2_probabilities(180.0, draw_param=0.26)
    assert p_h_fav + p_d_fav + p_a_fav == pytest.approx(1.0)
    assert p_h_fav > p_h0
    assert p_a_fav < p_a0
    assert 0.0 < p_d_fav < p_d0


def test_pairwise_elo_1x2_and_promoted_team_rating_with_anti_leakage() -> None:
    """Verify 1X2 Elo columns, promoted club rating init, and future-mutation invariance."""
    matches = pl.DataFrame(
        {
            "season": [20232024, 20232024, 20242025, 20242025],
            "event_id": [101, 102, 201, 202],
            "event_seq": [1, 1, 1, 1],
            "start_time_utc": [
                datetime(2023, 8, 12, 14, 0, tzinfo=UTC),
                datetime(2023, 8, 19, 14, 0, tzinfo=UTC),
                datetime(2024, 8, 17, 14, 0, tzinfo=UTC),
                datetime(2024, 8, 24, 14, 0, tzinfo=UTC),
            ],
            "home_team": ["ARS", "MCI", "ARS", "IPS"],
            "away_team": ["MCI", "ARS", "IPS", "MCI"],
            "home_goals": [2, 1, 3, 0],
            "away_goals": [1, 1, 0, 2],
            "game_state": ["FINAL", "FINAL", "FINAL", "FINAL"],
        }
    )

    config = EloConfig(
        base_rating=1500.0,
        home_field_advantage=48.0,
        k_factor=18.0,
        offseason_retention=0.75,
        draw_param=0.26,
        promoted_team_rating=1390.0,
    )
    engine = PairwiseEloEngine(config=config)
    rated = engine.compute_ratings(matches)

    assert "elo_draw_prob" in rated.columns
    for row in rated.iter_rows(named=True):
        prob_sum = (
            float(row["elo_home_win_prob"])
            + float(row["elo_draw_prob"])
            + float(row["elo_away_win_prob"])
        )
        assert prob_sum == pytest.approx(1.0)
        assert 0.0 < float(row["elo_home_win_prob"]) < 1.0
        assert 0.0 < float(row["elo_draw_prob"]) < 1.0
        assert 0.0 < float(row["elo_away_win_prob"]) < 1.0

    # Returning club ARS regresses toward 1500.0, while newly promoted IPS starts at 1390.0
    ars_end_2324 = float(rated["elo_away_post"][1])
    ars_start_2425 = float(rated["elo_home_pre"][2])
    ips_start_2425 = float(rated["elo_away_pre"][2])
    assert ars_start_2425 == pytest.approx(0.75 * ars_end_2324 + 0.25 * 1500.0)
    assert ips_start_2425 == pytest.approx(1390.0)

    # Future-Mutation Invariance: corrupting 20242025 outcomes does not change pre-match ratings
    cutoff = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
    corrupted = matches.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff)
        .then(pl.lit(9))
        .otherwise(pl.col("home_goals"))
        .alias("home_goals"),
        pl.when(pl.col("start_time_utc") >= cutoff)
        .then(pl.lit(8))
        .otherwise(pl.col("away_goals"))
        .alias("away_goals"),
    )
    mutated = engine.compute_ratings(corrupted)
    pre_cols = [
        "event_id",
        "elo_home_pre",
        "elo_away_pre",
        "elo_diff_pre",
        "elo_home_win_prob",
        "elo_draw_prob",
        "elo_away_win_prob",
    ]
    assert_frame_equal(
        rated.filter(pl.col("start_time_utc") <= cutoff).select(pre_cols),
        mutated.filter(pl.col("start_time_utc") <= cutoff).select(pre_cols),
    )


def test_ranked_probability_score_ordinal_sensitivity() -> None:
    """Verify RPS penalizes distant errors (Home vs Away) more than adjacent errors."""
    y_true = np.array([0], dtype=np.int64)  # Actual outcome: Home Win (0)

    # Prediction A puts 80% mass on adjacent outcome Draw (1)
    pred_adjacent = np.array([[0.10, 0.80, 0.10]], dtype=np.float64)
    # Prediction B puts 80% mass on distant outcome Away Win (2)
    pred_distant = np.array([[0.10, 0.10, 0.80]], dtype=np.float64)

    # Multi-class Brier score treats both non-true classes identically, whereas RPS is ordinal
    assert multiclass_brier_score(pred_adjacent, y_true) == pytest.approx(
        multiclass_brier_score(pred_distant, y_true)
    )
    rps_adjacent = ranked_probability_score(pred_adjacent, y_true)
    rps_distant = ranked_probability_score(pred_distant, y_true)
    assert rps_adjacent < rps_distant


def test_multiclass_calibrator_and_baseline_comparison() -> None:
    """Verify MulticlassProbabilityCalibrator outputs valid simplex probs and beats baselines."""
    rng = np.random.default_rng(42)
    n = 360
    logits = rng.normal(loc=0.0, scale=1.2, size=(n, 3))
    logits[:, 0] += 0.30  # Home advantage
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    true_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    y_true = np.array([rng.choice(3, p=true_probs[i]) for i in range(n)], dtype=np.int64)

    # Overconfident raw probabilities
    overconf = normalize_multiclass_probabilities(np.power(true_probs, 2.2))

    for method in ("sigmoid", "isotonic"):
        cal = MulticlassProbabilityCalibrator(method=method)
        cal.fit(overconf[:240], y_true[:240])
        calibrated = cal.predict_proba(overconf[240:])

        assert calibrated.shape == (120, 3)
        assert np.all((calibrated > 0.0) & (calibrated < 1.0))
        assert np.allclose(np.sum(calibrated, axis=1), 1.0, atol=1e-9)

        report = compare_multiclass_against_baselines(calibrated, y_true[240:])
        metrics = evaluate_multiclass_probabilities(calibrated, y_true[240:])
        assert metrics.rps == pytest.approx(report.model.rps)
        assert report.beats_empirical_prior_rps
        assert report.beats_empirical_prior_brier
        assert report.beats_empirical_prior_log_loss
