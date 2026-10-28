"""Unit tests for sport-agnostic market odds math and edge evaluation (`test_market.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from sports_ml.core.market import (
    american_to_decimal,
    decimal_to_raw_prob,
    devig_probabilities,
    evaluate_market_edge,
    is_forbidden_market_column,
)


def test_american_to_decimal_and_raw_prob_conversions() -> None:
    """Verify American-to-Decimal and Decimal-to-Raw-Prob conversions for scalars and arrays."""
    # Scalar conversions
    assert american_to_decimal(-200) == pytest.approx(1.50, abs=1e-9)
    assert american_to_decimal(150) == pytest.approx(2.50, abs=1e-9)
    assert american_to_decimal(-125) == pytest.approx(1.80, abs=1e-9)
    assert american_to_decimal(100) == pytest.approx(2.00, abs=1e-9)

    assert decimal_to_raw_prob(2.00) == pytest.approx(0.50, abs=1e-9)
    assert decimal_to_raw_prob(1.50) == pytest.approx(2.0 / 3.0, abs=1e-9)

    # Vectorized conversions
    amer_arr = np.array([-200.0, 150.0, -145.0, 125.0], dtype=np.float64)
    dec_arr = american_to_decimal(amer_arr)
    assert isinstance(dec_arr, np.ndarray)
    np.testing.assert_allclose(
        dec_arr,
        np.array([1.50, 2.50, 1.0 + 100.0 / 145.0, 2.25], dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    )

    raw_p = decimal_to_raw_prob(dec_arr)
    np.testing.assert_allclose(raw_p, 1.0 / dec_arr, rtol=0.0, atol=1e-12)

    # Invalid inputs raise ValueError
    with pytest.raises(ValueError, match="magnitude must be >= 100"):
        american_to_decimal(-50.0)
    with pytest.raises(ValueError, match="Decimal odds must be > 1.0"):
        decimal_to_raw_prob(1.0)


def test_devig_probabilities_2way_and_3way_favorite_longshot_bias() -> None:
    """Verify multiplicative, power, and shin de-vigging on 2-way NHL and 3-way EPL odds."""
    # 1. 2-way NHL moneyline (-250 favorite vs +205 underdog) -> decimal [1.40, 3.05]
    nhl_odds = np.array([1.40, 3.05], dtype=np.float64)
    p_mult_2 = devig_probabilities(nhl_odds, method="multiplicative")
    p_pow_2 = devig_probabilities(nhl_odds, method="power")
    p_shin_2 = devig_probabilities(nhl_odds, method="shin")

    for probs in (p_mult_2, p_pow_2, p_shin_2):
        assert probs.shape == (2,)
        assert np.all((probs > 0.0) & (probs < 1.0))
        assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-9)

    # Power method shrinks the longshot (index 1) MORE than multiplicative (favorite-longshot bias)
    assert p_pow_2[1] < p_mult_2[1]
    assert p_pow_2[0] > p_mult_2[0]

    # 2. 3-way EPL 1X2 odds (Heavy Home Favorite 1.36, Draw 5.25, Longshot Away 8.50)
    epl_odds = np.array(
        [
            [1.36, 5.25, 8.50],
            [2.10, 3.40, 3.50],
        ],
        dtype=np.float64,
    )
    p_mult_3 = devig_probabilities(epl_odds, method="multiplicative")
    p_pow_3 = devig_probabilities(epl_odds, method="power")
    p_shin_3 = devig_probabilities(epl_odds, method="shin")

    for mat in (p_mult_3, p_pow_3, p_shin_3):
        assert mat.shape == (2, 3)
        assert np.all((mat > 0.0) & (mat < 1.0))
        np.testing.assert_allclose(np.sum(mat, axis=1), 1.0, rtol=0.0, atol=1e-9)

    # In match 0, Away (8.50) is the longshot and Home (1.36) is heavy favorite:
    # Power method shrinks longshot probability more than proportional multiplicative method
    raw_longshot = 1.0 / 8.50
    raw_fav = 1.0 / 1.36
    mult_longshot_ratio = p_mult_3[0, 2] / raw_longshot
    pow_longshot_ratio = p_pow_3[0, 2] / raw_longshot
    pow_fav_ratio = p_pow_3[0, 0] / raw_fav
    assert pow_longshot_ratio < mult_longshot_ratio
    assert pow_longshot_ratio < pow_fav_ratio


def test_evaluate_market_edge_binary_and_multiclass_roi_and_slices() -> None:
    """Verify evaluate_market_edge accurately computes ROI, Kelly edge, and disagreement slices."""
    # Construct 6 binary NHL games where the model has a +12% edge on games 0..3
    market_probs = np.array([0.50, 0.50, 0.50, 0.50, 0.55, 0.45], dtype=np.float64)
    model_probs = np.array([0.62, 0.62, 0.62, 0.38, 0.56, 0.44], dtype=np.float64)
    # Outcomes: games 0, 1, 2 are Home Wins (1); game 3 is Away Win (0); games 4, 5 are 1, 0
    y_true = np.array([1, 1, 1, 0, 1, 0], dtype=np.int64)
    # Fair 2.00 decimal odds on the 4 disagreement games
    dec_odds = np.array(
        [
            [2.00, 2.00],
            [2.00, 2.00],
            [2.00, 2.00],
            [2.00, 2.00],
            [1.80, 2.10],
            [2.20, 1.72],
        ],
        dtype=np.float64,
    )

    report = evaluate_market_edge(
        model_probs,
        market_probs,
        y_true,
        decimal_odds=dec_odds,
        thresholds=(0.03, 0.05, 0.08),
    )

    assert report.n_games_with_odds == 6
    assert report.model_brier_score < report.market_brier_score
    assert report.model_log_loss < report.market_log_loss

    slice_5pct = report.get_slice(0.05)
    assert slice_5pct is not None
    # Games 0, 1, 2, 3 have |P_model - P_market| == 0.12 >= 0.05; games 4, 5 have 0.01 < 0.05
    assert slice_5pct.sample_count == 4
    assert slice_5pct.bet_count == 4
    assert slice_5pct.model_accuracy == pytest.approx(1.0)
    assert slice_5pct.flat_stake_roi == pytest.approx(1.0)
    assert slice_5pct.fractional_kelly_roi == pytest.approx(1.0)

    # Also verify 3-way multiclass EPL market edge report
    mkt_1x2 = np.array(
        [
            [0.45, 0.25, 0.30],
            [0.40, 0.30, 0.30],
            [0.50, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    mod_1x2 = np.array(
        [
            [0.55, 0.20, 0.25],
            [0.32, 0.28, 0.40],
            [0.51, 0.25, 0.24],
        ],
        dtype=np.float64,
    )
    y_1x2 = np.array([0, 2, 0], dtype=np.int64)
    odds_1x2 = np.array(
        [
            [2.10, 3.80, 3.20],
            [2.40, 3.20, 3.10],
            [1.95, 3.80, 3.80],
        ],
        dtype=np.float64,
    )

    mc_report = evaluate_market_edge(
        mod_1x2,
        mkt_1x2,
        y_1x2,
        decimal_odds=odds_1x2,
    )
    assert mc_report.n_games_with_odds == 3
    assert mc_report.model_rps is not None
    assert mc_report.market_rps is not None
    assert mc_report.model_rps < mc_report.market_rps
    mc_slice_5 = mc_report.get_slice(0.05)
    assert mc_slice_5 is not None
    assert mc_slice_5.sample_count == 2
    assert mc_slice_5.flat_stake_roi > 0.0


def test_is_forbidden_market_column() -> None:
    """Verify the market/odds column name detector catches all market_* and *odds* names."""
    assert is_forbidden_market_column("market_prob_home") is True
    assert is_forbidden_market_column("market_home_decimal_odds") is True
    assert is_forbidden_market_column("closing_odds_diff") is True
    assert is_forbidden_market_column("PINNACLE_ODDS") is True
    assert is_forbidden_market_column("xgf_pct_5v5_w10_diff") is False
    assert is_forbidden_market_column("elo_home_win_prob") is False
