"""Sport-agnostic market odds math, de-vigging, and edge/ROI evaluation (`core/market.py`).

Provides conversions between American and Decimal odds, raw implied probabilities,
vig removal (`"power"`, `"multiplicative"`, and `"shin"`), and head-to-head model vs.
closing-market evaluation (`evaluate_market_edge` -> `MarketEdgeReport`).

All market columns are strictly evaluation-only (`market_*`) and are forbidden from
entering Stage 1 or Stage 2 feature matrices (`X`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from sports_ml.core.calibration import (
    clip_probabilities,
    normalize_multiclass_probabilities,
)
from sports_ml.core.evaluation import (
    EvaluationMetrics,
    MulticlassEvaluationMetrics,
    evaluate_multiclass_probabilities,
    evaluate_probabilities,
)

DevigMethod = Literal["power", "multiplicative", "shin"]

FORBIDDEN_MARKET_TOKENS: tuple[str, ...] = ("market_", "odds")


def is_forbidden_market_column(column_name: str) -> bool:
    """Return `True` if `column_name` belongs to the evaluation-only market/odds namespace."""
    lower = column_name.strip().lower()
    return lower.startswith("market_") or "odds" in lower


@overload
def american_to_decimal(odds: int | float) -> float: ...


@overload
def american_to_decimal(odds: NDArray[np.float64] | Sequence[float]) -> NDArray[np.float64]: ...


def american_to_decimal(
    odds: int | float | NDArray[np.float64] | Sequence[float],
) -> float | NDArray[np.float64]:
    """Convert American moneyline odds (e.g. `-145`, `+125`) to decimal odds (`> 1.0`).

    - For positive American odds (`>= +100`): `decimal = 1.0 + (american / 100.0)`
    - For negative American odds (`<= -100`): `decimal = 1.0 + (100.0 / abs(american))`
    """
    if isinstance(odds, (int, float, np.number)):
        val = float(odds)
        if abs(val) < 100.0:
            raise ValueError(f"American odds magnitude must be >= 100, got {val}.")
        if val > 0.0:
            return 1.0 + (val / 100.0)
        return 1.0 + (100.0 / abs(val))

    arr = np.asarray(odds, dtype=np.float64)
    if np.any(np.abs(arr) < 100.0):
        raise ValueError("All American odds entries must have magnitude >= 100.")
    out = np.where(arr > 0.0, 1.0 + (arr / 100.0), 1.0 + (100.0 / np.abs(arr)))
    return np.asarray(out, dtype=np.float64)


@overload
def decimal_to_raw_prob(decimal_odds: int | float) -> float: ...


@overload
def decimal_to_raw_prob(
    decimal_odds: NDArray[np.float64] | Sequence[float],
) -> NDArray[np.float64]: ...


def decimal_to_raw_prob(
    decimal_odds: int | float | NDArray[np.float64] | Sequence[float],
) -> float | NDArray[np.float64]:
    """Convert decimal odds (`d > 1.0`) to raw unnormalized implied probability (`1.0 / d`)."""
    if isinstance(decimal_odds, (int, float, np.number)):
        val = float(decimal_odds)
        if val <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0, got {val}.")
        return 1.0 / val

    arr = np.asarray(decimal_odds, dtype=np.float64)
    if np.any(arr <= 1.0):
        raise ValueError("All decimal odds entries must be > 1.0.")
    return np.asarray(1.0 / arr, dtype=np.float64)


def _solve_power_devig_row(raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve for exponent `k > 0` such that `sum_i (raw_probs_i ** k) == 1.0` via bisection."""
    clipped = np.clip(raw_probs, 1e-9, 1.0 - 1e-9)
    overround = float(np.sum(clipped))
    if abs(overround - 1.0) <= 1e-12:
        return clipped / overround

    # Monotonically decreasing f(k) = sum(q_i ** k) - 1.0 for q_i in (0, 1)
    low = 0.05
    high = 10.0
    while float(np.sum(np.power(clipped, high))) > 1.0 and high < 1000.0:
        high *= 2.0

    for _ in range(80):
        mid = 0.5 * (low + high)
        val = float(np.sum(np.power(clipped, mid))) - 1.0
        if abs(val) < 1e-12:
            low = high = mid
            break
        if val > 0.0:
            low = mid
        else:
            high = mid

    k = 0.5 * (low + high)
    powered = np.power(clipped, k)
    return np.asarray(powered / float(np.sum(powered)), dtype=np.float64)


def _solve_shin_devig_row(raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve Shin (1993) insider-trading de-vigging model for a single row of raw probabilities."""
    q = np.clip(raw_probs, 1e-9, 1.0 - 1e-9)
    s = float(np.sum(q))
    n = q.size
    if s <= 1.0 + 1e-12 or n < 2:
        return q / s

    low = 0.0
    high = min(0.99, (s - 1.0) / float(n - 1) * 1.5 + 0.05)

    for _ in range(80):
        z = 0.5 * (low + high)
        disc = np.maximum(1e-15, z * z + 4.0 * (1.0 - z) * (q * q) / s)
        p = (np.sqrt(disc) - z) / (2.0 * (1.0 - z))
        total = float(np.sum(p))
        if abs(total - 1.0) < 1e-12:
            return np.asarray(p / total, dtype=np.float64)
        if total > 1.0:
            low = z
        else:
            high = z

    z = 0.5 * (low + high)
    disc = np.maximum(1e-15, z * z + 4.0 * (1.0 - z) * (q * q) / s)
    p = (np.sqrt(disc) - z) / (2.0 * (1.0 - z))
    return np.asarray(p / float(np.sum(p)), dtype=np.float64)


def devig_probabilities(
    decimal_odds: NDArray[np.float64] | Sequence[float] | Sequence[Sequence[float]],
    method: DevigMethod = "power",
) -> NDArray[np.float64]:
    """Remove bookmaker overround (vig) from 1D or 2D decimal odds arrays.

    Parameters
    ----------
    decimal_odds:
        1D array of shape `(K,)` for a single event or 2D array of shape `(N, K)`
        across `N` events and `K >= 2` mutually exclusive outcomes.
    method:
        - `"multiplicative"`: Proportional normalization `P_i = (1/d_i) / sum_j (1/d_j)`.
        - `"power"` (default): Solves `sum_i (1/d_i)^k = 1.0` for `k > 1`, shrinking
          longshot implied probabilities more than favorites to account for the
          favorite-longshot bias.
        - `"shin"`: Solves Shin's (1993) insider-trading parameter `z`.
    """
    arr = np.asarray(decimal_odds, dtype=np.float64)
    is_1d = arr.ndim == 1
    if is_1d:
        mat = arr.reshape(1, -1)
    elif arr.ndim == 2:
        mat = arr
    else:
        raise ValueError(f"decimal_odds must be 1D or 2D, got {arr.ndim}D.")

    if mat.shape[0] == 0 or mat.shape[1] < 2:
        raise ValueError("decimal_odds must have at least 1 row and >= 2 outcome columns.")
    if np.any(~np.isfinite(mat)) or np.any(mat <= 1.0):
        raise ValueError("All decimal_odds entries must be finite and strictly > 1.0.")

    raw_probs = 1.0 / mat

    if method == "multiplicative":
        row_sums = np.sum(raw_probs, axis=1, keepdims=True)
        out = raw_probs / row_sums
    elif method == "power":
        out = np.vstack([_solve_power_devig_row(row) for row in raw_probs])
    elif method == "shin":
        out = np.vstack([_solve_shin_devig_row(row) for row in raw_probs])
    else:
        raise ValueError(
            f"Unsupported de-vig method '{method}'. Expected 'power', 'multiplicative', or 'shin'."
        )

    out = np.clip(out, 1e-9, 1.0 - 1e-9)
    out = out / np.sum(out, axis=1, keepdims=True)
    return np.asarray(out[0] if is_1d else out, dtype=np.float64)


class MarketDisagreementSlice(BaseModel):
    """Performance and simulated ROI on events where `|P_model - P_market| >= threshold`."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    sample_count: int
    bet_count: int
    model_brier_score: float
    market_brier_score: float
    model_accuracy: float
    market_accuracy: float
    mean_abs_edge: float
    flat_stake_roi: float
    fractional_kelly_roi: float


class MarketEdgeReport(BaseModel):
    """Head-to-head model vs. de-vigged closing market evaluation report."""

    model_config = ConfigDict(frozen=True)

    n_games_with_odds: int
    mean_overround: float
    model_log_loss: float
    market_log_loss: float
    model_brier_score: float
    market_brier_score: float
    model_ece: float
    market_ece: float
    model_accuracy: float
    market_accuracy: float
    model_rps: float | None = None
    market_rps: float | None = None
    disagreement_slices: tuple[MarketDisagreementSlice, ...] = ()

    def get_slice(self, threshold: float = 0.05) -> MarketDisagreementSlice | None:
        """Return the disagreement slice matching `threshold` (within `1e-4`), if present."""
        for sl in self.disagreement_slices:
            if abs(sl.threshold - threshold) < 1e-4:
                return sl
        return None


def _ensure_2d_probs(probs: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim == 1:
        p1 = clip_probabilities(arr.ravel())
        return np.column_stack([p1, 1.0 - p1])
    if arr.ndim == 2 and arr.shape[1] == 2:
        p1 = clip_probabilities(arr[:, 0])
        return np.column_stack([p1, 1.0 - p1])
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return normalize_multiclass_probabilities(arr)
    raise ValueError(f"Unsupported probability array shape {arr.shape}.")


def evaluate_market_edge(
    model_probs: NDArray[np.float64],
    market_probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    *,
    decimal_odds: NDArray[np.float64] | None = None,
    thresholds: Sequence[float] = (0.03, 0.05, 0.08),
    kelly_fraction: float = 0.25,
    max_kelly_stake: float = 0.10,
    n_bins: int = 10,
) -> MarketEdgeReport:
    """Evaluate fundamental model probabilities head-to-head against de-vigged market odds.

    Supports both binary (`1D` or `(N, 2)` where `y_true in {0, 1}` is `home_win`) and
    3-way multiclass (`(N, 3)` where `y_true in {0, 1, 2}` is `outcome_1x2`).
    """
    m_raw = np.asarray(model_probs, dtype=np.float64)
    k_raw = np.asarray(market_probs, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.int64).ravel()

    is_binary = m_raw.ndim == 1 or (m_raw.ndim == 2 and m_raw.shape[1] == 2)

    if is_binary:
        m_1d = clip_probabilities(m_raw[:, 0] if m_raw.ndim == 2 else m_raw.ravel())
        k_1d = clip_probabilities(k_raw[:, 0] if k_raw.ndim == 2 else k_raw.ravel())
        if m_1d.size != y_arr.size or k_1d.size != y_arr.size or y_arr.size == 0:
            raise ValueError("model_probs, market_probs, and y_true must have equal non-zero size.")

        mod_eval: EvaluationMetrics = evaluate_probabilities(m_1d, y_arr, n_bins=n_bins)
        mkt_eval: EvaluationMetrics = evaluate_probabilities(k_1d, y_arr, n_bins=n_bins)

        # Convert to 2-column [Home Win (y==1), Away Win (y==0)] for unified betting simulation
        p_mod_mat = np.column_stack([m_1d, 1.0 - m_1d])
        p_mkt_mat = np.column_stack([k_1d, 1.0 - k_1d])
        y_col_idx = np.where(y_arr == 1, 0, 1)

        mod_ll = mod_eval.log_loss
        mkt_ll = mkt_eval.log_loss
        mod_bs = mod_eval.brier_score
        mkt_bs = mkt_eval.brier_score
        mod_ece = mod_eval.ece
        mkt_ece = mkt_eval.ece
        mod_acc = mod_eval.accuracy
        mkt_acc = mkt_eval.accuracy
        mod_rps: float | None = None
        mkt_rps: float | None = None
    else:
        p_mod_mat = _ensure_2d_probs(m_raw)
        p_mkt_mat = _ensure_2d_probs(k_raw)
        if (
            p_mod_mat.shape != p_mkt_mat.shape
            or p_mod_mat.shape[0] != y_arr.size
            or y_arr.size == 0
        ):
            raise ValueError("Multiclass model_probs, market_probs, and y_true shapes must match.")

        mod_mc: MulticlassEvaluationMetrics = evaluate_multiclass_probabilities(
            p_mod_mat, y_arr, n_bins=n_bins
        )
        mkt_mc: MulticlassEvaluationMetrics = evaluate_multiclass_probabilities(
            p_mkt_mat, y_arr, n_bins=n_bins
        )
        y_col_idx = y_arr
        mod_ll = mod_mc.log_loss
        mkt_ll = mkt_mc.log_loss
        mod_bs = mod_mc.brier_score
        mkt_bs = mkt_mc.brier_score
        mod_ece = mod_mc.ece
        mkt_ece = mkt_mc.ece
        mod_acc = mod_mc.accuracy
        mkt_acc = mkt_mc.accuracy
        mod_rps = mod_mc.rps
        mkt_rps = mkt_mc.rps

    n_games, n_outcomes = p_mod_mat.shape
    if decimal_odds is not None:
        d_mat = np.asarray(decimal_odds, dtype=np.float64)
        if d_mat.shape != p_mkt_mat.shape:
            raise ValueError(
                f"decimal_odds shape {d_mat.shape} must match market_probs {p_mkt_mat.shape}."
            )
        mean_overround = float(np.mean(np.sum(1.0 / np.clip(d_mat, 1.0001, None), axis=1) - 1.0))
    else:
        # Synthesize fair decimal odds from market_probs when raw bookmaker odds aren't supplied
        d_mat = 1.0 / np.clip(p_mkt_mat, 1e-6, 1.0 - 1e-6)
        mean_overround = 0.0

    one_hot = np.zeros_like(p_mod_mat, dtype=np.float64)
    one_hot[np.arange(n_games), y_col_idx] = 1.0

    abs_diff_mat = np.abs(p_mod_mat - p_mkt_mat)
    max_abs_diff = np.max(abs_diff_mat, axis=1)
    edge_mat = p_mod_mat - p_mkt_mat

    slices: list[MarketDisagreementSlice] = []
    for tau in thresholds:
        mask = max_abs_diff >= float(tau)
        sample_count = int(np.sum(mask))
        if sample_count == 0:
            slices.append(
                MarketDisagreementSlice(
                    threshold=float(tau),
                    sample_count=0,
                    bet_count=0,
                    model_brier_score=0.0,
                    market_brier_score=0.0,
                    model_accuracy=0.0,
                    market_accuracy=0.0,
                    mean_abs_edge=0.0,
                    flat_stake_roi=0.0,
                    fractional_kelly_roi=0.0,
                )
            )
            continue

        sub_mod = p_mod_mat[mask]
        sub_mkt = p_mkt_mat[mask]
        sub_oh = one_hot[mask]
        sub_y = y_col_idx[mask]
        sub_d = d_mat[mask]
        sub_edge = edge_mat[mask]

        if is_binary:
            s_mod_bs = float(np.mean((sub_mod[:, 0] - sub_oh[:, 0]) ** 2))
            s_mkt_bs = float(np.mean((sub_mkt[:, 0] - sub_oh[:, 0]) ** 2))
        else:
            s_mod_bs = float(np.mean(np.sum((sub_mod - sub_oh) ** 2, axis=1)))
            s_mkt_bs = float(np.mean(np.sum((sub_mkt - sub_oh) ** 2, axis=1)))

        s_mod_acc = float(np.mean(np.argmax(sub_mod, axis=1) == sub_y))
        s_mkt_acc = float(np.mean(np.argmax(sub_mkt, axis=1) == sub_y))
        s_mean_edge = float(np.mean(max_abs_diff[mask]))

        # Select the outcome with the highest positive probability edge (P_model - P_market >= tau)
        best_side = np.argmax(sub_edge, axis=1)
        best_edge = sub_edge[np.arange(sample_count), best_side]
        bet_mask = best_edge >= float(tau)
        bet_count = int(np.sum(bet_mask))

        if bet_count > 0:
            chosen_side = best_side[bet_mask]
            chosen_odds = np.clip(sub_d[bet_mask, chosen_side], 1.01, None)
            chosen_mod_p = sub_mod[bet_mask, chosen_side]
            won = (sub_y[bet_mask] == chosen_side).astype(np.float64)

            flat_profits = np.where(won > 0.5, chosen_odds - 1.0, -1.0)
            flat_roi = float(np.mean(flat_profits))

            b_net = chosen_odds - 1.0
            raw_kelly = np.maximum(
                0.0, (chosen_mod_p * chosen_odds - 1.0) / np.maximum(b_net, 1e-6)
            )
            stakes = np.clip(kelly_fraction * raw_kelly, 0.0, max_kelly_stake)
            total_staked = float(np.sum(stakes))
            if total_staked > 1e-9:
                kelly_profits = stakes * np.where(won > 0.5, b_net, -1.0)
                kelly_roi = float(np.sum(kelly_profits) / total_staked)
            else:
                kelly_roi = flat_roi
        else:
            flat_roi = 0.0
            kelly_roi = 0.0

        slices.append(
            MarketDisagreementSlice(
                threshold=float(tau),
                sample_count=sample_count,
                bet_count=bet_count,
                model_brier_score=s_mod_bs,
                market_brier_score=s_mkt_bs,
                model_accuracy=s_mod_acc,
                market_accuracy=s_mkt_acc,
                mean_abs_edge=s_mean_edge,
                flat_stake_roi=flat_roi,
                fractional_kelly_roi=kelly_roi,
            )
        )

    return MarketEdgeReport(
        n_games_with_odds=n_games,
        mean_overround=mean_overround,
        model_log_loss=mod_ll,
        market_log_loss=mkt_ll,
        model_brier_score=mod_bs,
        market_brier_score=mkt_bs,
        model_ece=mod_ece,
        market_ece=mkt_ece,
        model_accuracy=mod_acc,
        market_accuracy=mkt_acc,
        model_rps=mod_rps,
        market_rps=mkt_rps,
        disagreement_slices=tuple(slices),
    )
