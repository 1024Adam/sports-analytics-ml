"""Starting goalie resolver and rolling shrunk GSAx/60 features (`features/goalie.py`).

Computes leak-free rolling 10-start and 25-start shrunk GSAx/60 (`k = 15.0`),
save percentage (`SV%`), and starter workload/fatigue (`goalie_starts_last_7d`,
`goalie_days_rest`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Final

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import (
    pregame_exposure_rate,
    pregame_sample_count,
    sort_events_chronologically,
)

GOALIE_SHRINKAGE_K: Final[float] = 15.0
DEFAULT_GOALIE_WINDOWS: Final[tuple[int, ...]] = (10, 25)
PRIOR_GOALIE_GSAx_PER_60: Final[float] = 0.0
PRIOR_GOALIE_SAVE_PCT: Final[float] = 0.905
MAX_GOALIE_REST_DAYS: Final[int] = 14


def _compute_goalie_fatigue_columns(
    sorted_df: pl.DataFrame,
    *,
    max_rest_days: int = MAX_GOALIE_REST_DAYS,
) -> pl.DataFrame:
    """Compute strictly pre-game `goalie_days_rest` and `goalie_starts_last_7d` per goalie."""
    if sorted_df.is_empty():
        return sorted_df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("goalie_days_rest"),
            pl.lit(None, dtype=pl.Int32).alias("goalie_starts_last_7d"),
        )

    rows = sorted_df.to_dicts()
    n_rows = len(rows)
    days_rest_out = [max_rest_days] * n_rows
    starts_7d_out = [0] * n_rows

    # Per-(player_id, season) prior appearance dates
    history: dict[tuple[int, int], list[tuple[date, bool]]] = {}

    for idx, row in enumerate(rows):
        pid = int(row["player_id"])
        season = int(row.get("season") or 0)
        start_utc: datetime = row["start_time_utc"]
        ev_date: date = (
            row["game_date"]
            if "game_date" in row and row["game_date"] is not None
            else start_utc.date()
        )
        is_starter = bool(row.get("is_starter", True))

        key = (pid, season)
        g_hist = history.setdefault(key, [])

        # Pre-game metrics use strictly prior appearances (< idx)
        if not g_hist:
            days_rest_out[idx] = max_rest_days
            starts_7d_out[idx] = 0
        else:
            prev_date, _ = g_hist[-1]
            raw_days = max(0, (ev_date - prev_date).days)
            days_rest_out[idx] = min(raw_days, max_rest_days)
            starts_7d_out[idx] = sum(
                1 for (d, started) in g_hist if started and 1 <= (ev_date - d).days <= 7
            )

        # Record current appearance after extracting pre-game fatigue
        g_hist.append((ev_date, is_starter))

    return sorted_df.with_columns(
        pl.Series("goalie_days_rest", days_rest_out, dtype=pl.Int32),
        pl.Series("goalie_starts_last_7d", starts_7d_out, dtype=pl.Int32),
    )


def build_goalie_rolling_features(
    goalie_game_logs: pl.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_GOALIE_WINDOWS,
    shrinkage_k: float = GOALIE_SHRINKAGE_K,
) -> pl.DataFrame:
    """Compute leak-free rolling shrunk GSAx/60, SV%, and workload fatigue features."""
    working = goalie_game_logs
    if "gsax" not in working.columns and {"xga_all", "goals_against"}.issubset(working.columns):
        working = working.with_columns(
            (pl.col("xga_all").cast(pl.Float64) - pl.col("goals_against").cast(pl.Float64)).alias(
                "gsax"
            )
        )
    if "saves" not in working.columns and {"shots_against", "goals_against"}.issubset(
        working.columns
    ):
        working = working.with_columns(
            (pl.col("shots_against") - pl.col("goals_against")).cast(pl.Int16).alias("saves")
        )

    sorted_df = sort_events_chronologically(working, extra_tiebreakers=["player_id"])
    with_fatigue = _compute_goalie_fatigue_columns(sorted_df)

    exprs: list[pl.Expr] = [
        pregame_sample_count("gsax", group_by="player_id").alias("goalie_prior_games_n"),
    ]

    for w in windows:
        n_w = pregame_sample_count("gsax", group_by="player_id", window=w)
        raw_gsax_60 = pregame_exposure_rate(
            "gsax", "toi_min", window=w, group_by="player_id", scale_factor=60.0
        )
        raw_sv_pct = pregame_exposure_rate(
            "saves", "shots_against", window=w, group_by="player_id", scale_factor=1.0
        )
        exprs.extend(
            [
                bayesian_shrinkage(raw_gsax_60, n_w, PRIOR_GOALIE_GSAx_PER_60, shrinkage_k).alias(
                    f"gsax_per_60_shrunk_w{w}"
                ),
                bayesian_shrinkage(raw_sv_pct, n_w, PRIOR_GOALIE_SAVE_PCT, shrinkage_k).alias(
                    f"save_pct_shrunk_w{w}"
                ),
            ]
        )

    result = with_fatigue.with_columns(exprs)
    ref_w = 10 if 10 in windows else windows[0]
    return result.with_columns(
        pl.col(f"gsax_per_60_shrunk_w{ref_w}").alias("gsax_per_60_shrunk"),
        pl.col(f"save_pct_shrunk_w{ref_w}").alias("save_pct_shrunk"),
        (pl.col("goalie_prior_games_n") < shrinkage_k).alias("is_low_sample_goalie"),
    )


def resolve_team_starting_goalie_features(
    goalie_features_df: pl.DataFrame,
) -> pl.DataFrame:
    """Select one starting goalie row per `(game_id, team)` for Stage 1 & Stage 2 joins."""
    if goalie_features_df.is_empty():
        return goalie_features_df

    event_col = "game_id" if "game_id" in goalie_features_df.columns else "event_id"
    sorted_starters = goalie_features_df.sort(
        [event_col, "team", "is_starter", "toi_min", "player_id"],
        descending=[False, False, True, True, False],
    )
    deduped = sorted_starters.unique(
        subset=[event_col, "team"],
        keep="first",
        maintain_order=True,
    )
    sort_cols = (
        ["start_time_utc", event_col, "team"]
        if "start_time_utc" in deduped.columns
        else [event_col, "team"]
    )
    return deduped.sort(sort_cols)
