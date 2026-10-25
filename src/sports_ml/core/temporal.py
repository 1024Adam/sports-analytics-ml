"""Leak-free Polars temporal window expressions (`core/temporal.py`).

All expressions enforce a strict left-closed `.shift(1)` within each entity
partition ordered by `(start_time_utc, event_seq, event_id)`.
Full implementation scheduled for Milestone 1.
"""

from __future__ import annotations

import polars as pl


def sort_events_chronologically(df: pl.DataFrame) -> pl.DataFrame:
    """Sort an event or appearance DataFrame strictly by `(start_time_utc, event_seq, event_id)`."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")


def pregame_rolling_mean(
    col_name: str,
    window: int,
    group_by: str | list[str],
    min_periods: int = 1,
) -> pl.Expr:
    """Compute a strictly shifted (`.shift(1)`) rolling mean over `group_by`."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")


def pregame_rolling_sum(
    col_name: str,
    window: int,
    group_by: str | list[str],
    min_periods: int = 1,
) -> pl.Expr:
    """Compute a strictly shifted (`.shift(1)`) rolling sum over `group_by`."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")


def pregame_ewma(
    col_name: str,
    alpha: float,
    group_by: str | list[str],
    min_periods: int = 1,
) -> pl.Expr:
    """Compute a strictly shifted (`.shift(1)`) exponentially weighted moving average."""
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")


def pregame_exposure_rate(
    numerator_col: str,
    exposure_col: str,
    window: int,
    group_by: str | list[str],
    scale_factor: float = 1.0,
    min_periods: int = 1,
) -> pl.Expr:
    """Compute a strictly shifted exposure-normalized rate (`scale * sum(num) / sum(exp)`).

    Supports per-60 TOI (`scale_factor=60.0` in NHL), per-9 IP (`scale_factor=9.0` in MLB),
    per-PA (`scale_factor=1.0` in MLB), and per-round Strokes Gained (`scale_factor=1.0` in PGA).
    """
    raise NotImplementedError("Scheduled for Milestone 1 implementation.")
