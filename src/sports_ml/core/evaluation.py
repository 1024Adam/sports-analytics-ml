"""Probability and count-regression evaluation metrics (`core/evaluation.py`).

Computes Log Loss, Brier Score, Expected Calibration Error (ECE), accuracy, and
benchmark comparisons against coin-flip, home-win-rate, and Elo-only baselines.
Full implementation scheduled for Milestone 4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class EvaluationMetrics(BaseModel):
    """Summary metrics for binary probability predictions."""

    model_config = ConfigDict(frozen=True)

    log_loss: float
    brier_score: float
    ece: float
    accuracy: float


def expected_calibration_error(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across `n_bins` probability buckets."""
    raise NotImplementedError("Scheduled for Milestone 4 implementation.")


def evaluate_probabilities(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 10,
) -> EvaluationMetrics:
    """Compute Log Loss, Brier Score, ECE, and accuracy for predicted probabilities."""
    raise NotImplementedError("Scheduled for Milestone 4 implementation.")
