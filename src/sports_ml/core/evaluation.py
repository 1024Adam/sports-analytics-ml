"""Probability and count-regression evaluation metrics (`core/evaluation.py`).

Computes Log Loss, Brier Score, Expected Calibration Error (ECE), accuracy, and
benchmark comparisons against coin-flip (`0.50`), home-team baseline, and
Elo-only baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import brier_score_loss, log_loss

from sports_ml.core.calibration import clip_probabilities


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
    n_bins: int = 10,
) -> BenchmarkComparisonReport:
    """Evaluate `model_probs` alongside coin-flip, empirical home-win rate, and Elo baselines."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    n = y.size

    model_metrics = evaluate_probabilities(model_probs, y, n_bins=n_bins)
    coin_probs = np.full(n, 0.50, dtype=np.float64)
    home_probs = np.full(n, float(train_home_win_rate), dtype=np.float64)

    coin_metrics = evaluate_probabilities(coin_probs, y, n_bins=n_bins)
    home_metrics = evaluate_probabilities(home_probs, y, n_bins=n_bins)
    elo_metrics = (
        evaluate_probabilities(elo_probs, y, n_bins=n_bins) if elo_probs is not None else None
    )

    return BenchmarkComparisonReport(
        model=model_metrics,
        coin_flip_baseline=coin_metrics,
        home_win_rate_baseline=home_metrics,
        elo_baseline=elo_metrics,
    )
