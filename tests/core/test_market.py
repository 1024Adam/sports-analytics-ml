"""Unit tests for sport-agnostic market odds math and edge evaluation (`test_market.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from sports_ml.core.evaluation import evaluate_line_markets
from sports_ml.core.market import (
    BREAK_EVEN_WIN_RATE,
    STANDARD_JUICE_DECIMAL,
    american_to_decimal,
    ats_break_even,
    closing_line_value,
    decimal_to_raw_prob,
    devig_probabilities,
    evaluate_market_edge,
    evaluate_spread_betting,
    evaluate_total_betting,
    is_forbidden_market_column,
    is_key_number_line,
    is_whole_point_line,
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


def test_two_way_devig_stays_in_open_unit_interval_across_nfl_price_ranges() -> None:
    """Power de-vig of NFL moneylines (heavy favorite to pick'em) stays in (0, 1) and sums to 1."""
    pairs = [(-1500, 800), (-110, -110), (-250, 205), (100, -120), (-3000, 1200), (-105, -115)]
    dec = np.array([[american_to_decimal(a), american_to_decimal(b)] for a, b in pairs])
    for method in ("power", "multiplicative", "shin"):
        probs = devig_probabilities(dec, method=method)
        assert np.all((probs > 0.0) & (probs < 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-9)
    pick = devig_probabilities(dec[1], method="power")
    assert pick[0] == pytest.approx(0.5, abs=1e-9)


def test_ats_break_even_and_roi_math_at_minus_110() -> None:
    assert pytest.approx(110.0 / 210.0) == BREAK_EVEN_WIN_RATE
    assert ats_break_even() == pytest.approx(0.5238095, abs=1e-6)
    assert pytest.approx(1.9090909, abs=1e-6) == STANDARD_JUICE_DECIMAL
    with pytest.raises(ValueError, match="decimal_odds"):
        ats_break_even(1.0)
    # 11 wins and 10 losses at -110 is exactly break-even: 11 * (100/110) - 10 = 0
    n = 21
    spread = np.full(n, -2.5)
    model = np.full(n, 6.0)  # edge +3.5 on the home side every game
    actual = np.where(np.arange(n) < 11, 7.0, 1.0)  # home covers 11 of 21
    rep = evaluate_spread_betting(model, spread, actual)
    b = rep.get_bucket(3.0)
    assert b is not None
    assert (b.wins, b.losses, b.pushes) == (11, 10, 0)
    assert b.hit_rate == pytest.approx(11 / 21)
    assert b.hit_rate == pytest.approx(rep.break_even_win_rate)
    assert b.roi == pytest.approx(0.0, abs=1e-12)


def test_ats_handles_pushes_whole_vs_half_point_and_side_selection() -> None:
    # spread_home: -3 (home favored by 3), +3 (home 3-pt dog), -2.5 (half point), -7
    spread = np.array([-3.0, 3.0, -2.5, -7.0, -3.0])
    model = np.array([6.0, -6.0, 6.0, 1.0, 6.0])  # edges +3, -3, +3.5, -6, +3
    actual = np.array([3.0, -3.0, 3.0, 10.0, 7.0])  # pushes in games 0 and 1
    rep = evaluate_spread_betting(model, spread, actual)
    assert rep.n_whole_point_lines == 4 and rep.n_half_point_lines == 1
    b = rep.get_bucket(3.0)
    assert b is not None
    # game 0: home bet, margin+spread = 0 -> push; game 1: away bet, -3+3 = 0 -> push
    # game 2: home bet, 3-2.5 > 0 win; game 3: away bet, home covers (10-7>0) -> loss;
    # game 4: home bet, 7-3 > 0 win
    assert (b.wins, b.losses, b.pushes) == (2, 1, 2)
    assert b.hit_rate == pytest.approx(2 / 3)  # pushes excluded from the hit rate
    assert b.roi == pytest.approx((2 * (100 / 110) - 1) / 5)  # ... and refunded in the ROI base
    assert b.n_bets == 5
    # half-point lines can never push
    half = evaluate_spread_betting(np.array([5.0]), np.array([-2.5]), np.array([3.0]))
    hb = half.get_bucket(1.0)
    assert hb is not None and hb.pushes == 0
    assert hb.wins + hb.losses == 1
    assert is_whole_point_line(-3.0) and not is_whole_point_line(-2.5)


def test_key_numbers_are_flagged_on_closing_lines() -> None:
    assert all(is_key_number_line(x) for x in (-3.0, 3.0, 6.0, -7.0, 10.0))
    assert not any(is_key_number_line(x) for x in (-3.5, 4.0, 5.0, 14.0))
    spread = np.array([-3.0, -3.0, -4.0, -7.0])
    model = np.array([6.0, 6.0, 8.0, 12.0])
    actual = np.array([3.0, 10.0, 4.0, 7.0])  # pushes at -3 (game 0), -4 (game 2), -7 (game 3)
    b = evaluate_spread_betting(model, spread, actual).get_bucket(1.0)
    assert b is not None
    assert b.key_bets == 3 and b.key_pushes == 2  # the push at -4 is not a key-number push


def test_thresholds_bucket_by_disagreement_and_empty_buckets_are_null() -> None:
    spread = np.zeros(4) - 1.0
    model = np.array([0.5, 1.5, 2.5, 3.5]) - 1.0 + 1.0  # edges -0.5? computed below
    edges = model + spread
    rep = evaluate_spread_betting(model, spread, np.array([2.0, 2.0, 2.0, 2.0]))
    counts = [rep.get_bucket(t).n_bets for t in (1.0, 2.0, 3.0)]  # type: ignore[union-attr]
    assert counts == [int(np.sum(np.abs(edges) >= t)) for t in (1.0, 2.0, 3.0)]
    assert counts == sorted(counts, reverse=True)
    none = evaluate_spread_betting(
        np.array([0.0]), np.array([-1.0]), np.array([3.0]), thresholds=(5.0,)
    )
    nb = none.get_bucket(5.0)
    assert nb is not None and nb.n_bets == 0 and nb.hit_rate is None and nb.roi is None
    with pytest.raises(ValueError, match="at least one game"):
        evaluate_spread_betting(np.array([np.nan]), np.array([-1.0]), np.array([3.0]))


def test_totals_over_under_pushes_and_missing_lines() -> None:
    closing = np.array([44.0, 44.0, 44.5, 50.0, np.nan])
    model = np.array([47.0, 40.0, 48.0, 46.0, 45.0])  # over, under, over, under, (no line)
    actual = np.array([44.0, 41.0, 45.0, 52.0, 40.0])  # push, under wins, over wins, under loses
    rep = evaluate_total_betting(model, closing, actual)
    assert rep.n_games == 4  # NaN closing total dropped
    b = rep.get_bucket(3.0)
    assert b is not None
    assert (b.wins, b.losses, b.pushes) == (2, 1, 1)
    assert b.key_bets == 0
    assert rep.mean_abs_error_model == pytest.approx(float(np.mean(np.abs(model[:4] - actual[:4]))))


def test_closing_line_value_signs_for_spread_and_total_bets() -> None:
    one, neg = np.array([1.0]), np.array([-1.0])
    # home bet at +3 closing +1 -> +2; away bet on the same prices -> -2
    assert closing_line_value(one, np.array([3.0]), np.array([1.0]), "spread")[0] == 2.0
    assert closing_line_value(neg, np.array([3.0]), np.array([1.0]), "spread")[0] == -2.0
    # over taken at 44 closing 46 -> +2; under taken at 46 closing 44 -> +2
    assert closing_line_value(one, np.array([44.0]), np.array([46.0]), "total")[0] == 2.0
    assert closing_line_value(neg, np.array([46.0]), np.array([44.0]), "total")[0] == 2.0
    rep = evaluate_spread_betting(
        np.array([6.0, -6.0]),
        np.array([-3.0, -3.0]),
        np.array([4.0, -4.0]),
        taken_spread_home=np.array([-2.0, -4.0]),  # home taken -2 (close -3): home side +1
    )
    b = rep.get_bucket(3.0)
    assert b is not None and b.n_bets == 2
    # game 0: home bet, taken -2 vs close -3 -> CLV +1; game 1: away bet, taken -4 vs close -3 -> +1
    assert b.mean_clv == pytest.approx(1.0) and b.clv_positive_rate == 1.0
    assert (
        evaluate_spread_betting(np.array([6.0]), np.array([-3.0]), np.array([4.0]))
        .buckets[0]
        .mean_clv
        is None
    )


def test_evaluate_line_markets_skips_markets_without_closing_lines() -> None:
    rep = evaluate_line_markets(
        model_margin=np.array([3.0, 1.0]),
        actual_margin=np.array([7.0, -2.0]),
        closing_spread_home=np.array([-3.0, np.nan]),
    )
    assert rep.spread is not None and rep.spread.n_games == 1
    assert rep.total is None
    empty = evaluate_line_markets(
        model_margin=np.array([3.0]),
        actual_margin=np.array([7.0]),
        closing_spread_home=np.array([np.nan]),
    )
    assert empty.spread is None and empty.total is None
