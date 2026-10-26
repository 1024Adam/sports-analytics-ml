"""Leak-free Polars temporal window expressions (`core/temporal.py`).

All expressions enforce a strict left-closed `.shift(1)` within each entity
partition ordered by `(start_time_utc, event_seq, event_id)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sports_ml.core.contracts import SchemaValidationError


def _normalize_group_by(group_by: str | Sequence[str]) -> list[str]:
    if isinstance(group_by, str):
        return [group_by]
    return list(group_by)


def sort_events_chronologically(
    df: pl.DataFrame,
    *,
    extra_tiebreakers: Sequence[str] = (),
) -> pl.DataFrame:
    """Sort an event or appearance DataFrame strictly by `(start_time_utc, event_seq, event_id)`.

    If `event_seq` is absent, a default `event_seq = 1` column is added.
    If `event_id` is absent but `game_id` is present, `game_id` is aliased to `event_id`.
    """
    working = df
    if "start_time_utc" not in working.columns:
        raise SchemaValidationError("DataFrame must contain 'start_time_utc' for temporal sorting.")

    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))

    if "event_id" not in working.columns:
        if "game_id" in working.columns:
            working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
        else:
            raise SchemaValidationError(
                "DataFrame must contain 'event_id' (or 'game_id') for temporal sorting."
            )

    sort_cols = ["start_time_utc", "event_seq", "event_id", *extra_tiebreakers]
    return working.sort(sort_cols)


def pregame_rolling_mean(
    col_name: str,
    window: int,
    group_by: str | Sequence[str],
    min_periods: int = 1,
) -> pl.Expr:
    """Compute a strictly shifted (`.shift(1)`) rolling mean over `group_by`."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if min_periods < 1 or min_periods > window:
        raise ValueError(f"min_periods must be in [1, {window}], got {min_periods}.")

    partition_cols = _normalize_group_by(group_by)
    return (
        pl.col(col_name)
        .cast(pl.Float64)
        .shift(1)
        .rolling_mean(window_size=window, min_samples=min_periods)
        .over(partition_cols)
    )


def pregame_rolling_sum(
    col_name: str,
    window: int,
    group_by: str | Sequence[str],
    min_periods: int = 1,
) -> pl.Expr:
    """Compute a strictly shifted (`.shift(1)`) rolling sum over `group_by`."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if min_periods < 1 or min_periods > window:
        raise ValueError(f"min_periods must be in [1, {window}], got {min_periods}.")

    partition_cols = _normalize_group_by(group_by)
    return (
        pl.col(col_name)
        .cast(pl.Float64)
        .shift(1)
        .rolling_sum(window_size=window, min_samples=min_periods)
        .over(partition_cols)
    )


def pregame_ewma(
    col_name: str,
    alpha: float,
    group_by: str | Sequence[str],
    min_periods: int = 1,
) -> pl.Expr:
    """Compute a strictly shifted (`.shift(1)`) exponentially weighted moving average."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}.")
    if min_periods < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods}.")

    partition_cols = _normalize_group_by(group_by)
    return (
        pl.col(col_name)
        .cast(pl.Float64)
        .shift(1)
        .ewm_mean(alpha=alpha, min_samples=min_periods, ignore_nulls=True)
        .over(partition_cols)
    )


def pregame_exposure_rate(
    numerator_col: str,
    exposure_col: str,
    window: int,
    group_by: str | Sequence[str],
    scale_factor: float = 1.0,
    min_periods: int = 1,
) -> pl.Expr:
    """Compute a strictly shifted exposure-normalized rate (`scale * sum(num) / sum(exp)`).

    Supports per-90 match minutes (`scale_factor=90.0` in EPL Soccer), per-60 TOI
    (`scale_factor=60.0` in NHL), per-9 IP (`scale_factor=9.0` in MLB), per-PA
    (`scale_factor=1.0` in MLB), and per-round Strokes Gained (`scale_factor=1.0` in PGA).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if min_periods < 1 or min_periods > window:
        raise ValueError(f"min_periods must be in [1, {window}], got {min_periods}.")
    if scale_factor <= 0.0:
        raise ValueError(f"scale_factor must be > 0.0, got {scale_factor}.")

    partition_cols = _normalize_group_by(group_by)
    shifted_num = (
        pl.col(numerator_col)
        .cast(pl.Float64)
        .shift(1)
        .rolling_sum(window_size=window, min_samples=min_periods)
        .over(partition_cols)
    )
    shifted_exp = (
        pl.col(exposure_col)
        .cast(pl.Float64)
        .shift(1)
        .rolling_sum(window_size=window, min_samples=min_periods)
        .over(partition_cols)
    )
    return scale_factor * (
        shifted_num / pl.when(shifted_exp > 0.0).then(shifted_exp).otherwise(None)
    )


def pregame_sample_count(
    col_name: str,
    group_by: str | Sequence[str],
    window: int | None = None,
) -> pl.Expr:
    """Count strictly prior non-null appearances for `col_name` within `group_by`."""
    partition_cols = _normalize_group_by(group_by)
    valid_flag = pl.col(col_name).is_not_null().cast(pl.Int32).shift(1, fill_value=0)
    if window is None:
        return valid_flag.cum_sum().over(partition_cols)
    if window < 1:
        raise ValueError(f"window must be >= 1 when specified, got {window}.")
    return valid_flag.rolling_sum(window_size=window, min_samples=1).over(partition_cols)
