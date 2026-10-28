"""Probability and count-regression evaluation metrics (`core/evaluation.py`).

Computes Log Loss, Brier Score, Expected Calibration Error (ECE), accuracy, and
benchmark comparisons against coin-flip (`0.50`), home-team baseline, Elo-only
baseline, and de-vigged market closing lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import brier_score_loss, log_loss

from sports_ml.core.calibration import (
    clip_probabilities,
    normalize_multiclass_probabilities,
)

if TYPE_CHECKING:
    from sports_ml.core.market import MarketEdgeReport


class EvaluationMetrics(BaseModel):
    """Summary metrics for binary probability predictions."""

    model_config = ConfigDict(frozen=True)

    log_loss: float
    brier_score: float
    ece: float
    accuracy: float


class BenchmarkComparisonReport(BaseModel):
    """Comparison of Stage 1 calibrated model metrics against canonical baselines."""

    model_config = ConfigDict(frozen=True)

    model: EvaluationMetrics
    coin_flip_baseline: EvaluationMetrics
    home_win_rate_baseline: EvaluationMetrics
    elo_baseline: EvaluationMetrics | None = None
    market_closing_baseline: EvaluationMetrics | None = None
    market_edge_report: MarketEdgeReport | None = None

    @property
    def beats_home_baseline_brier(self) -> bool:
        return self.model.brier_score < self.home_win_rate_baseline.brier_score

    @property
    def beats_home_baseline_log_loss(self) -> bool:
        return self.model.log_loss < self.home_win_rate_baseline.log_loss


def expected_calibration_error(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across `n_bins` uniform probability buckets."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

    p = clip_probabilities(np.asarray(probs, dtype=np.float64).ravel())
    y = np.asarray(y_true, dtype=np.float64).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError("probs and y_true must be non-empty arrays of equal length.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = float(p.size)

    for b_idx in range(n_bins):
        low = bin_edges[b_idx]
        high = bin_edges[b_idx + 1]
        mask = (p >= low) & (p <= high) if b_idx == n_bins - 1 else (p >= low) & (p < high)

        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(p[mask]))
            avg_acc = float(np.mean(y[mask]))
            ece += (float(count) / n_total) * abs(avg_acc - avg_conf)

    return float(ece)


def evaluate_probabilities(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 10,
) -> EvaluationMetrics:
    """Compute Log Loss, Brier Score, ECE, and accuracy for predicted probabilities."""
    p = clip_probabilities(np.asarray(probs, dtype=np.float64).ravel())
    y = np.asarray(y_true, dtype=np.int64).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError("probs and y_true must be non-empty arrays of equal length.")

    ll = float(log_loss(y, p, labels=[0, 1]))
    bs = float(brier_score_loss(y, p))
    ece_val = expected_calibration_error(p, y, n_bins=n_bins)
    preds = (p >= 0.5).astype(np.int64)
    acc = float(np.mean(preds == y))

    return EvaluationMetrics(
        log_loss=ll,
        brier_score=bs,
        ece=ece_val,
        accuracy=acc,
    )


def compare_against_baselines(
    model_probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    *,
    train_home_win_rate: float = 0.535,
    elo_probs: NDArray[np.float64] | None = None,
    market_probs: NDArray[np.float64] | None = None,
    market_decimal_odds: NDArray[np.float64] | None = None,
    n_bins: int = 10,
) -> BenchmarkComparisonReport:
    """Evaluate `model_probs` alongside coin-flip, empirical home-win rate, Elo, and market."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    m_arr = np.asarray(model_probs, dtype=np.float64).ravel()
    n = y.size

    model_metrics = evaluate_probabilities(m_arr, y, n_bins=n_bins)
    coin_probs = np.full(n, 0.50, dtype=np.float64)
    home_probs = np.full(n, float(train_home_win_rate), dtype=np.float64)

    coin_metrics = evaluate_probabilities(coin_probs, y, n_bins=n_bins)
    home_metrics = evaluate_probabilities(home_probs, y, n_bins=n_bins)
    elo_metrics = (
        evaluate_probabilities(elo_probs, y, n_bins=n_bins) if elo_probs is not None else None
    )

    market_metrics: EvaluationMetrics | None = None
    market_edge: MarketEdgeReport | None = None
    if market_probs is not None:
        from sports_ml.core.market import evaluate_market_edge

        mkt_arr = np.asarray(market_probs, dtype=np.float64).ravel()
        valid_mask = np.isfinite(mkt_arr) & (mkt_arr > 0.0) & (mkt_arr < 1.0)
        if int(np.sum(valid_mask)) > 0:
            sub_mod = m_arr[valid_mask]
            sub_mkt = mkt_arr[valid_mask]
            sub_y = y[valid_mask]
            sub_odds: NDArray[np.float64] | None = None
            if market_decimal_odds is not None:
                odds_arr = np.asarray(market_decimal_odds, dtype=np.float64)
                if odds_arr.shape[0] == n:
                    cand_odds = odds_arr[valid_mask]
                    if np.all(np.isfinite(cand_odds)) and np.all(cand_odds > 1.0):
                        sub_odds = cand_odds

            market_metrics = evaluate_probabilities(sub_mkt, sub_y, n_bins=n_bins)
            market_edge = evaluate_market_edge(
                sub_mod,
                sub_mkt,
                sub_y,
                decimal_odds=sub_odds,
                n_bins=n_bins,
            )

    return BenchmarkComparisonReport(
        model=model_metrics,
        coin_flip_baseline=coin_metrics,
        home_win_rate_baseline=home_metrics,
        elo_baseline=elo_metrics,
        market_closing_baseline=market_metrics,
        market_edge_report=market_edge,
    )


class MulticlassEvaluationMetrics(BaseModel):
    """Summary metrics for 3-way (`1X2`) or ordinal multiclass probability predictions."""

    model_config = ConfigDict(frozen=True)

    rps: float
    log_loss: float
    brier_score: float
    ece: float
    accuracy: float


class MulticlassBenchmarkReport(BaseModel):
    """Comparison of Stage 1 multiclass (`1X2`) model metrics against canonical baselines."""

    model_config = ConfigDict(frozen=True)

    model: MulticlassEvaluationMetrics
    uniform_baseline: MulticlassEvaluationMetrics
    empirical_prior_baseline: MulticlassEvaluationMetrics
    elo_baseline: MulticlassEvaluationMetrics | None = None
    market_closing_baseline: MulticlassEvaluationMetrics | None = None
    market_edge_report: MarketEdgeReport | None = None

    @property
    def beats_empirical_prior_rps(self) -> bool:
        return self.model.rps < self.empirical_prior_baseline.rps

    @property
    def beats_empirical_prior_brier(self) -> bool:
        return self.model.brier_score < self.empirical_prior_baseline.brier_score

    @property
    def beats_empirical_prior_log_loss(self) -> bool:
        return self.model.log_loss < self.empirical_prior_baseline.log_loss


def ranked_probability_score(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
) -> float:
    """Compute mean Ranked Probability Score (RPS) for ordered multiclass outcomes.

    For `K` ordered outcome classes (e.g., `0 = Home Win`, `1 = Draw`, `2 = Away Win`):
        `RPS = (1 / (K - 1)) * sum_{k=1}^{K-1} (F_k - O_k) ** 2`
    where `F_k` is the cumulative predicted probability through class `k - 1` and
    `O_k` is the cumulative indicator `1[y_true <= k - 1]`.
    """
    p = normalize_multiclass_probabilities(np.asarray(probs, dtype=np.float64))
    y = np.asarray(y_true, dtype=np.int64).ravel()
    n_samples, n_classes = p.shape
    if y.size != n_samples or n_samples == 0:
        raise ValueError("probs and y_true must be non-empty and have matching sample counts.")
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError(f"y_true entries must lie in [0, {n_classes - 1}].")

    one_hot = np.zeros_like(p, dtype=np.float64)
    one_hot[np.arange(n_samples), y] = 1.0

    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_o = np.cumsum(one_hot, axis=1)[:, :-1]
    per_match_rps = np.sum((cum_p - cum_o) ** 2, axis=1) / float(n_classes - 1)
    return float(np.mean(per_match_rps))


def multiclass_brier_score(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
) -> float:
    """Compute mean multi-class Brier score `(1 / N) * sum_i sum_c (p_{i,c} - y_{i,c}) ** 2`."""
    p = normalize_multiclass_probabilities(np.asarray(probs, dtype=np.float64))
    y = np.asarray(y_true, dtype=np.int64).ravel()
    n_samples, n_classes = p.shape
    if y.size != n_samples or n_samples == 0:
        raise ValueError("probs and y_true must be non-empty and have matching sample counts.")

    one_hot = np.zeros_like(p, dtype=np.float64)
    one_hot[np.arange(n_samples), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def multiclass_expected_calibration_error(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 10,
) -> float:
    """Compute class-averaged Expected Calibration Error (ECE) across all `K` outcome classes."""
    p = normalize_multiclass_probabilities(np.asarray(probs, dtype=np.float64))
    y = np.asarray(y_true, dtype=np.int64).ravel()
    n_classes = p.shape[1]
    class_eces = [
        expected_calibration_error(p[:, c_idx], (y == c_idx).astype(np.int64), n_bins=n_bins)
        for c_idx in range(n_classes)
    ]
    return float(np.mean(class_eces))


def evaluate_multiclass_probabilities(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 10,
) -> MulticlassEvaluationMetrics:
    """Compute RPS, 3-class Log Loss, multi-class Brier Score, ECE, and accuracy."""
    p = normalize_multiclass_probabilities(np.asarray(probs, dtype=np.float64))
    y = np.asarray(y_true, dtype=np.int64).ravel()
    n_samples, n_classes = p.shape
    if y.size != n_samples or n_samples == 0:
        raise ValueError("probs and y_true must be non-empty and have matching sample counts.")

    rps_val = ranked_probability_score(p, y)
    ll_val = float(log_loss(y, p, labels=list(range(n_classes))))
    bs_val = multiclass_brier_score(p, y)
    ece_val = multiclass_expected_calibration_error(p, y, n_bins=n_bins)
    preds = np.argmax(p, axis=1)
    acc_val = float(np.mean(preds == y))

    return MulticlassEvaluationMetrics(
        rps=rps_val,
        log_loss=ll_val,
        brier_score=bs_val,
        ece=ece_val,
        accuracy=acc_val,
    )


def compare_multiclass_against_baselines(
    model_probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    *,
    empirical_prior: tuple[float, float, float] = (0.45, 0.24, 0.31),
    elo_probs: NDArray[np.float64] | None = None,
    market_probs: NDArray[np.float64] | None = None,
    market_decimal_odds: NDArray[np.float64] | None = None,
    n_bins: int = 10,
) -> MulticlassBenchmarkReport:
    """Evaluate `1X2` model probabilities against uniform, empirical EPL prior, Elo, and market."""
    p = normalize_multiclass_probabilities(np.asarray(model_probs, dtype=np.float64))
    y = np.asarray(y_true, dtype=np.int64).ravel()
    n_samples, n_classes = p.shape

    model_metrics = evaluate_multiclass_probabilities(p, y, n_bins=n_bins)
    uniform_probs = np.full((n_samples, n_classes), 1.0 / float(n_classes), dtype=np.float64)
    prior_vec = np.asarray(empirical_prior, dtype=np.float64).ravel()
    if prior_vec.size != n_classes:
        raise ValueError(f"empirical_prior must have length {n_classes}, got {prior_vec.size}.")
    prior_probs = np.tile(prior_vec / float(np.sum(prior_vec)), (n_samples, 1))

    uniform_metrics = evaluate_multiclass_probabilities(uniform_probs, y, n_bins=n_bins)
    prior_metrics = evaluate_multiclass_probabilities(prior_probs, y, n_bins=n_bins)
    elo_metrics = (
        evaluate_multiclass_probabilities(elo_probs, y, n_bins=n_bins)
        if elo_probs is not None
        else None
    )

    market_metrics: MulticlassEvaluationMetrics | None = None
    market_edge: MarketEdgeReport | None = None
    if market_probs is not None:
        from sports_ml.core.market import evaluate_market_edge

        mkt_mat = np.asarray(market_probs, dtype=np.float64)
        if mkt_mat.ndim == 2 and mkt_mat.shape == p.shape:
            valid_mask = np.all(np.isfinite(mkt_mat) & (mkt_mat > 0.0) & (mkt_mat < 1.0), axis=1)
            if int(np.sum(valid_mask)) > 0:
                sub_mod = p[valid_mask]
                sub_mkt = normalize_multiclass_probabilities(mkt_mat[valid_mask])
                sub_y = y[valid_mask]
                sub_odds: NDArray[np.float64] | None = None
                if market_decimal_odds is not None:
                    odds_arr = np.asarray(market_decimal_odds, dtype=np.float64)
                    if odds_arr.shape == p.shape:
                        cand_odds = odds_arr[valid_mask]
                        if np.all(np.isfinite(cand_odds)) and np.all(cand_odds > 1.0):
                            sub_odds = cand_odds

                market_metrics = evaluate_multiclass_probabilities(sub_mkt, sub_y, n_bins=n_bins)
                market_edge = evaluate_market_edge(
                    sub_mod,
                    sub_mkt,
                    sub_y,
                    decimal_odds=sub_odds,
                    n_bins=n_bins,
                )

    return MulticlassBenchmarkReport(
        model=model_metrics,
        uniform_baseline=uniform_metrics,
        empirical_prior_baseline=prior_metrics,
        elo_baseline=elo_metrics,
        market_closing_baseline=market_metrics,
        market_edge_report=market_edge,
    )


# Rebuild models referencing forward-declared MarketEdgeReport at import time
from sports_ml.core.market import MarketEdgeReport as _MarketEdgeReport  # noqa: E402

BenchmarkComparisonReport.model_rebuild(_types_namespace={"MarketEdgeReport": _MarketEdgeReport})
MulticlassBenchmarkReport.model_rebuild(_types_namespace={"MarketEdgeReport": _MarketEdgeReport})
