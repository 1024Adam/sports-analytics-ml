"""Out-of-fold probability calibration wrappers (`core/calibration.py`).

Supports Platt (Sigmoid) and Isotonic calibration fitted strictly on held-out
chronological validation folds, ensuring calibrated probabilities lie strictly
in `(0, 1)`.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sports_ml.config import RANDOM_SEED

CalibrationMethod = Literal["sigmoid", "isotonic"]
BinaryCalibrationMethod = Literal["platt", "temperature"]
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


class BinaryProbabilityCalibrator:
    """Small-sample binary calibrator using Platt or temperature scaling (never isotonic).

    Designed for low-volume leagues (e.g., ~270 NFL games/season) where isotonic regression
    overfits. Falls back to the identity map when fewer than `min_samples` pooled
    out-of-fold predictions are available, when only one class is present, or when the fitted
    slope is not positive (`is_identity` / `fallback_reason` record why).

    * `"platt"`: logistic regression on `logit(p)` (slope + intercept).
    * `"temperature"`: `p' = sigmoid(logit(p) / T)` with `T` minimizing log loss.
    """

    def __init__(
        self,
        method: BinaryCalibrationMethod = "platt",
        min_samples: int = 400,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2, got {min_samples}.")
        self.method: BinaryCalibrationMethod = method
        self.min_samples = min_samples
        self.random_seed = random_seed
        self.slope: float = 1.0
        self.intercept: float = 0.0
        self.temperature: float = 1.0
        self.n_fit_samples: int = 0
        self.is_identity: bool = True
        self.fallback_reason: str | None = "not_fitted"
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _set_identity(self, reason: str) -> BinaryProbabilityCalibrator:
        self.slope, self.intercept, self.temperature = 1.0, 0.0, 1.0
        self.is_identity = True
        self.fallback_reason = reason
        self._is_fitted = True
        return self

    def fit(
        self,
        raw_probs: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> BinaryProbabilityCalibrator:
        """Fit on pooled out-of-fold predictions (identity fallback below `min_samples`)."""
        p = clip_probabilities(np.asarray(raw_probs, dtype=np.float64).ravel())
        y = np.asarray(y_true, dtype=np.int64).ravel()
        if p.size != y.size or p.size == 0:
            raise ValueError("raw_probs and y_true must be non-empty arrays of equal length.")
        self.n_fit_samples = int(p.size)
        if p.size < self.min_samples:
            return self._set_identity("insufficient_samples")
        if np.unique(y).size < 2:
            return self._set_identity("single_class")

        z = np.asarray(logit(p), dtype=np.float64)
        if self.method == "temperature":

            def nll(log_t: float) -> float:
                q = clip_probabilities(np.asarray(expit(z / np.exp(log_t)), dtype=np.float64))
                return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1.0 - q)))

            res = minimize_scalar(nll, bounds=(-2.0, 2.0), method="bounded")
            self.temperature = float(np.exp(res.x))
            self.slope, self.intercept = 1.0 / self.temperature, 0.0
        else:
            lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=self.random_seed)
            lr.fit(z.reshape(-1, 1), y)
            slope = float(lr.coef_[0, 0])
            if slope <= 0.0:
                return self._set_identity("non_positive_slope")
            self.slope, self.intercept = slope, float(lr.intercept_[0])
            self.temperature = 1.0 / slope
        self.is_identity = False
        self.fallback_reason = None
        self._is_fitted = True
        return self

    def predict_proba(self, raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibrated probabilities strictly inside `(0, 1)`."""
        if not self._is_fitted:
            raise RuntimeError("BinaryProbabilityCalibrator must be fitted before predict_proba.")
        p = clip_probabilities(np.asarray(raw_probs, dtype=np.float64).ravel())
        if self.is_identity:
            return p
        calibrated = np.asarray(expit(self.slope * logit(p) + self.intercept), dtype=np.float64)
        return clip_probabilities(calibrated)


def normalize_multiclass_probabilities(
    probs: NDArray[np.float64],
    eps: float = PROB_EPS,
) -> NDArray[np.float64]:
    """Clip and row-normalize a `(N, K)` probability matrix onto the open simplex.

    Guarantees every entry lies strictly in `(eps, 1.0 - eps)` and every row sums
    to `1.0` within floating-point precision.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"probs must be a 2D array with at least 2 classes, got {arr.shape}.")
    clipped = np.clip(arr, eps, 1.0 - eps)
    row_sums = np.sum(clipped, axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    normalized = clipped / safe_sums
    # Final guard to keep entries strictly inside (eps, 1 - eps) while preserving sum = 1
    normalized = np.clip(normalized, eps, 1.0 - eps)
    normalized = normalized / np.sum(normalized, axis=1, keepdims=True)
    return np.asarray(normalized, dtype=np.float64)


class MulticlassProbabilityCalibrator:
    """Out-of-fold multiclass (`1X2`) probability calibrator with simplex normalization.

    Supports both `"sigmoid"` (Dirichlet / multinomial log-probability Platt scaling)
    and `"isotonic"` (per-class one-vs-rest isotonic regression followed by simplex
    renormalization) fitted strictly on held-out validation folds.
    """

    def __init__(
        self,
        method: CalibrationMethod = "sigmoid",
        n_classes: int = 3,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}.")
        self.method: CalibrationMethod = method
        self.n_classes = n_classes
        self.random_seed = random_seed
        self._multinomial_model: LogisticRegression | None = None
        self._isotonic_models: list[IsotonicRegression] | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        raw_probs: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> MulticlassProbabilityCalibrator:
        """Fit multiclass calibrator on out-of-fold validation predictions `(N, K)`."""
        probs_2d = normalize_multiclass_probabilities(np.asarray(raw_probs, dtype=np.float64))
        y_1d = np.asarray(y_true, dtype=np.int64).ravel()

        if probs_2d.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected raw_probs with {self.n_classes} columns, got {probs_2d.shape[1]}."
            )
        if probs_2d.shape[0] != y_1d.shape[0] or probs_2d.shape[0] == 0:
            raise ValueError("raw_probs and y_true must be non-empty and have matching row counts.")

        unique_classes = np.unique(y_1d)
        if unique_classes.size < self.n_classes:
            self._multinomial_model = None
            self._isotonic_models = None
            self._is_fitted = True
            return self

        if self.method == "isotonic":
            iso_list: list[IsotonicRegression] = []
            for c_idx in range(self.n_classes):
                iso = IsotonicRegression(
                    y_min=PROB_EPS,
                    y_max=1.0 - PROB_EPS,
                    out_of_bounds="clip",
                )
                y_bin = (y_1d == c_idx).astype(np.float64)
                iso.fit(probs_2d[:, c_idx], y_bin)
                iso_list.append(iso)
            cols = [
                np.asarray(iso_list[c_idx].predict(probs_2d[:, c_idx]), dtype=np.float64)
                for c_idx in range(self.n_classes)
            ]
            # Blend 50% isotonic + 50% raw prior to prevent step-function boundary log-loss spikes
            iso_blended = normalize_multiclass_probabilities(
                0.50 * np.column_stack(cols) + 0.50 * probs_2d
            )
            self._isotonic_models = iso_list
            self._multinomial_model = None
        else:
            # Guard against tiny validation folds where multinomial regression can overfit
            if probs_2d.shape[0] < 120:
                self._multinomial_model = None
                self._isotonic_models = None
            else:
                log_p = np.log(probs_2d)
                lr = LogisticRegression(
                    C=0.35,
                    solver="lbfgs",
                    random_state=self.random_seed,
                )
                lr.fit(log_p, y_1d)
                # Ensure diagonal class weights remain positive (monotonic preserving)
                diag_weights = np.diag(lr.coef_)
                if bool(np.all(diag_weights > 0.10)):
                    self._multinomial_model = lr
                else:
                    self._multinomial_model = None
                self._isotonic_models = None
            iso_blended = probs_2d
        _ = iso_blended

        self._is_fitted = True
        return self

    def predict_proba(self, raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform raw `(N, K)` probabilities into calibrated simplex probabilities."""
        if not self._is_fitted:
            raise RuntimeError(
                "MulticlassProbabilityCalibrator must be fitted before calling predict_proba."
            )

        probs_2d = normalize_multiclass_probabilities(np.asarray(raw_probs, dtype=np.float64))
        if probs_2d.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected raw_probs with {self.n_classes} columns, got {probs_2d.shape[1]}."
            )

        if self._isotonic_models is not None:
            cols = [
                np.asarray(
                    self._isotonic_models[c_idx].predict(probs_2d[:, c_idx]),
                    dtype=np.float64,
                )
                for c_idx in range(self.n_classes)
            ]
            iso_raw = normalize_multiclass_probabilities(np.column_stack(cols))
            return normalize_multiclass_probabilities(0.65 * iso_raw + 0.35 * probs_2d)

        if self._multinomial_model is not None:
            log_p = np.log(probs_2d)
            pred_matrix = np.asarray(self._multinomial_model.predict_proba(log_p), dtype=np.float64)
            # Damped blend (50% multinomial Platt + 50% structural prior) prevents small-fold shift
            return normalize_multiclass_probabilities(0.50 * pred_matrix + 0.50 * probs_2d)

        return probs_2d


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
