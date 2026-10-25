"""Out-of-fold probability calibration wrappers (`core/calibration.py`).

Supports Platt (Sigmoid) and Isotonic calibration fitted strictly on held-out
chronological validation folds.
Full implementation scheduled for Milestone 4.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

CalibrationMethod = Literal["sigmoid", "isotonic"]


class ProbabilityCalibrator:
    """Out-of-fold binary probability calibrator ensuring predictions lie in `(0, 1)`."""

    def __init__(self, method: CalibrationMethod = "sigmoid") -> None:
        self.method: CalibrationMethod = method

    def fit(
        self,
        raw_probs: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> ProbabilityCalibrator:
        """Fit calibrator on out-of-fold validation predictions."""
        raise NotImplementedError("Scheduled for Milestone 4 implementation.")

    def predict_proba(self, raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform raw uncalibrated probabilities into calibrated probabilities in `(0, 1)`."""
        raise NotImplementedError("Scheduled for Milestone 4 implementation.")
