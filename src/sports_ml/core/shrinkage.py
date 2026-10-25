"""Early-season and low-sample Bayesian prior shrinkage (`core/shrinkage.py`).

Implements `(n * curr + k * prior) / (n + k)` so opening-night teams and
debuting rookies evaluate cleanly to the prior with zero `NaN` values.
"""

from __future__ import annotations

import polars as pl


def _to_float_expr(val: str | int | float | pl.Expr) -> pl.Expr:
    if isinstance(val, str):
        return pl.col(val).cast(pl.Float64)
    if isinstance(val, (int, float)):
        return pl.lit(float(val), dtype=pl.Float64)
    return val.cast(pl.Float64)


def compute_regressed_prior(
    prev_season_col: str | float | pl.Expr,
    league_mean_col: str | float | pl.Expr,
    prev_season_weight: float = 0.65,
) -> pl.Expr:
    """Blend prior-season entity rate with causal league-average baseline.

    Formula: `w * prev_season + (1 - w) * league_mean`.
    When `prev_season` is null (e.g., rookie debut or expansion team), falls back
    cleanly to `league_mean`.
    """
    if not (0.0 <= prev_season_weight <= 1.0):
        raise ValueError(f"prev_season_weight must be in [0.0, 1.0], got {prev_season_weight}.")

    prev_expr = _to_float_expr(prev_season_col)
    league_expr = _to_float_expr(league_mean_col)
    league_weight = 1.0 - prev_season_weight

    return (
        pl.when(prev_expr.is_not_null() & prev_expr.is_finite())
        .then(prev_season_weight * prev_expr + league_weight * league_expr)
        .otherwise(league_expr)
    )


def bayesian_shrinkage(
    current_rate_col: str | float | pl.Expr,
    sample_size_col: str | int | float | pl.Expr,
    prior_col: str | float | pl.Expr,
    pseudo_count_k: float,
) -> pl.Expr:
    """Apply Bayesian shrinkage `(n * curr + k * prior) / (n + k)`.

    When `n == 0` or `curr` is null/NaN (e.g., Game 1 of a season or rookie debut),
    the expression evaluates strictly to `prior` with zero `NaN` or `null` values.
    """
    if pseudo_count_k <= 0.0:
        raise ValueError(f"pseudo_count_k must be > 0.0, got {pseudo_count_k}.")

    curr_expr = _to_float_expr(current_rate_col)
    n_raw_expr = _to_float_expr(sample_size_col).fill_null(0.0).clip(lower_bound=0.0)
    prior_expr = _to_float_expr(prior_col)
    k_lit = pl.lit(float(pseudo_count_k), dtype=pl.Float64)

    has_valid_sample = (n_raw_expr > 0.0) & curr_expr.is_not_null() & curr_expr.is_finite()
    effective_n = pl.when(has_valid_sample).then(n_raw_expr).otherwise(0.0)
    safe_curr = pl.when(has_valid_sample).then(curr_expr).otherwise(0.0)

    return (effective_n * safe_curr + k_lit * prior_expr) / (effective_n + k_lit)


def shrink_scalar(
    current_rate: float | None,
    sample_size_n: int | float,
    prior_rate: float,
    pseudo_count_k: float,
) -> float:
    """Scalar equivalent of `bayesian_shrinkage` for single-value shrinkage."""
    if pseudo_count_k <= 0.0:
        raise ValueError(f"pseudo_count_k must be > 0.0, got {pseudo_count_k}.")
    if current_rate is None or sample_size_n <= 0:
        return float(prior_rate)
    n = float(sample_size_n)
    return float((n * current_rate + pseudo_count_k * prior_rate) / (n + pseudo_count_k))
