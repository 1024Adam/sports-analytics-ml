"""Early-season and low-sample Bayesian prior shrinkage (`core/shrinkage.py`).

Implements `(n * curr + k * prior) / (n + k)` so opening-night teams and
debuting rookies evaluate cleanly to the prior with zero `NaN` values.
Full implementation scheduled for Milestone 1.
"""

from __future__ import annotations

import polars as pl


def compute_regressed_prior(
    prev_season_col: str | pl.Expr,
    league_mean_col: str | float | pl.Expr,
    prev_season_weight: float = 0.65,
) -> pl.Expr:
    """Blend prior-season entity rate with causal league-average baseline."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")


def bayesian_shrinkage(
    current_rate_col: str | pl.Expr,
    sample_size_col: str | pl.Expr,
    prior_col: str | float | pl.Expr,
    pseudo_count_k: float,
) -> pl.Expr:
    """Apply Bayesian shrinkage `(n * curr + k * prior) / (n + k)`."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")
