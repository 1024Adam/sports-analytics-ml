"""Out-of-fold probability calibration wrappers (`core/calibration.py`).

Supports Platt (Sigmoid) and Isotonic calibration fitted strictly on held-out
chronological validation folds, ensuring calibrated probabilities lie strictly
in `(0, 1)`.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sports_ml.config import RANDOM_SEED

CalibrationMethod = Literal["sigmoid", "isotonic"]
PROB_EPS: Final[float] = 1e-6


def clip_probabilities(
    probs: NDArray[np.float64],
    eps: float = PROB_EPS,
) -> NDArray[np.float64]:
    """Clip probability array strictly into `(eps, 1.0 - eps)`."""
    return np.asarray(np.clip(probs, eps, 1.0 - eps), dtype=np.float64)


class ProbabilityCalibrator:
    """Out-of-fold binary probability calibrator ensuring predictions lie in `(0, 1)`."""

    def __init__(
        self,
        method: CalibrationMethod = "sigmoid",
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.method: CalibrationMethod = method
        self.random_seed = random_seed
        self._platt_model: LogisticRegression | None = None
        self._isotonic_model: IsotonicRegression | None = None
        self._constant_prob: float | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        raw_probs: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> ProbabilityCalibrator:
        """Fit calibrator on out-of-fold validation predictions."""
        probs_1d = clip_probabilities(np.asarray(raw_probs, dtype=np.float64).ravel())
        y_1d = np.asarray(y_true, dtype=np.int64).ravel()

        if probs_1d.shape[0] != y_1d.shape[0] or probs_1d.shape[0] == 0:
            raise ValueError("raw_probs and y_true must be non-empty arrays of equal length.")

        unique_classes = np.unique(y_1d)
        if unique_classes.size < 2:
            # Degenerate validation fold with only 1 class: fall back to identity / shrunk mean
            self._constant_prob = None
            self._platt_model = None
            self._isotonic_model = None
            self._is_fitted = True
            return self

        if self.method == "isotonic":
            iso = IsotonicRegression(y_min=PROB_EPS, y_max=1.0 - PROB_EPS, out_of_bounds="clip")
            iso.fit(probs_1d, y_1d.astype(np.float64))
            self._isotonic_model = iso
            self._platt_model = None
        else:
            logits = logit(probs_1d).reshape(-1, 1)
            lr = LogisticRegression(C=0.25, solver="lbfgs", random_state=self.random_seed)
            lr.fit(logits, y_1d)
            slope = float(lr.coef_[0, 0])
            intercept = float(lr.intercept_[0])
            # Guard against small-sample validation folds (< 250 games) or extreme intercept shift
            if slope <= 0.25 or probs_1d.size < 250 or abs(intercept) > 0.35:
                self._platt_model = None
                self._isotonic_model = None
            else:
                self._platt_model = lr
                self._isotonic_model = None

        self._constant_prob = None
        self._is_fitted = True
        return self

    def predict_proba(self, raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform raw uncalibrated probabilities into calibrated probabilities in `(0, 1)`."""
        if not self._is_fitted:
            raise RuntimeError("ProbabilityCalibrator must be fitted before calling predict_proba.")

        probs_1d = clip_probabilities(np.asarray(raw_probs, dtype=np.float64).ravel())

        if self._isotonic_model is not None:
            calibrated = np.asarray(self._isotonic_model.predict(probs_1d), dtype=np.float64)
            return clip_probabilities(calibrated)

        if self._platt_model is not None:
            logits = logit(probs_1d).reshape(-1, 1)
            pred_matrix = self._platt_model.predict_proba(logits)
            calibrated = np.asarray(pred_matrix[:, 1], dtype=np.float64)
            return clip_probabilities(calibrated)

        return clip_probabilities(probs_1d)


def calibrate_multi_entrant_group(
    raw_scores: NDArray[np.float64],
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    """Normalize multi-entrant field scores via temperature-scaled softmax summing to `1.0`."""
    scores_1d = np.asarray(raw_scores, dtype=np.float64).ravel()
    if scores_1d.size == 0:
        return scores_1d
    temp = max(float(temperature), 1e-6)
    shifted = (scores_1d - np.max(scores_1d)) / temp
    weights = np.asarray(
        expit(shifted) if scores_1d.size == 1 else np.exp(shifted), dtype=np.float64
    )
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full_like(scores_1d, 1.0 / float(scores_1d.size))
    return weights / total
