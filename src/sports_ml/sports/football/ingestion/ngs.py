"""Next Gen Stats aggregation (`football/ingestion/ngs.py`).

nflverse publishes weekly NGS player rows for passing, rushing and receiving from 2016 on
(`week == 0` is the season aggregate and is ignored). Rows are mapped onto Silver games through
`(season, week, team)` and summed into additive numerators (value x its own exposure) so that
the downstream rolling features can form exposure-weighted rates. Seasons before
`NGS_FIRST_SEASON` simply have no rows; callers flag that with `has_ngs = 0`.
"""

from __future__ import annotations

from typing import Any, Final

import polars as pl

from sports_ml.sports.football.adapter import gsis_id_to_int
from sports_ml.sports.football.constants import (
    NGS_FIRST_SEASON,
    PLAYER_MICRO_NGS_COLUMNS,
    TEAM_MICRO_NGS_COLUMNS,
    normalize_football_team,
)


def _lookup(games_df: pl.DataFrame) -> pl.DataFrame:
    home = games_df.select(
        "game_id",
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int16),
        pl.col("home_team").alias("team"),
    )
    away = games_df.select(
        "game_id",
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int16),
        pl.col("away_team").alias("team"),
    )
    return pl.concat([home, away])


def _weekly(frame: pl.DataFrame | None, required: tuple[str, ...]) -> pl.DataFrame | None:
    if frame is None or frame.is_empty():
        return None
    missing = [c for c in ("season", "week", "team_abbr", *required) if c not in frame.columns]
    if missing:
        raise ValueError(f"NGS frame is missing columns: {missing}.")
    return frame.filter(pl.col("week") > 0).with_columns(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int16),
        pl.col("team_abbr")
        .map_elements(normalize_football_team, return_dtype=pl.Utf8)
        .alias("team"),
    )


def _gsis(value: Any) -> int | None:
    try:
        return gsis_id_to_int(value)
    except (ValueError, TypeError):
        return None


def aggregate_team_ngs(
    games_df: pl.DataFrame,
    *,
    passing: pl.DataFrame | None = None,
    rushing: pl.DataFrame | None = None,
    receiving: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return `(game_id, team) + TEAM_MICRO_NGS_COLUMNS` for every team-week with NGS rows."""
    lookup = _lookup(games_df)
    parts: list[pl.DataFrame] = []
    p = _weekly(
        passing, ("attempts", "avg_time_to_throw", "completion_percentage_above_expectation")
    )
    if p is not None:
        parts.append(
            p.group_by("season", "week", "team").agg(
                pl.col("attempts").cast(pl.Float64).sum().alias("ngs_pass_attempts"),
                (pl.col("avg_time_to_throw") * pl.col("attempts"))
                .sum()
                .alias("ngs_time_to_throw_sum"),
                (pl.col("completion_percentage_above_expectation") * pl.col("attempts"))
                .sum()
                .alias("ngs_cpoe_sum"),
            )
        )
    r = _weekly(rushing, ("rush_attempts", "rush_yards_over_expected"))
    if r is not None:
        parts.append(
            r.group_by("season", "week", "team").agg(
                pl.col("rush_attempts").cast(pl.Float64).sum().alias("ngs_rush_attempts"),
                pl.col("rush_yards_over_expected").sum().alias("ngs_ryoe_sum"),
            )
        )
    c = _weekly(receiving, ("targets", "receptions", "avg_separation", "avg_yac_above_expectation"))
    if c is not None:
        parts.append(
            c.group_by("season", "week", "team").agg(
                pl.col("targets").cast(pl.Float64).sum().alias("ngs_targets"),
                pl.col("receptions").cast(pl.Float64).sum().alias("ngs_receptions"),
                (pl.col("avg_separation") * pl.col("targets")).sum().alias("ngs_separation_sum"),
                (pl.col("avg_yac_above_expectation") * pl.col("receptions"))
                .sum()
                .alias("ngs_yac_oe_sum"),
            )
        )
    if not parts:
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "team": pl.Utf8,
                **{c: pl.Float64 for c in TEAM_MICRO_NGS_COLUMNS},
            }
        )
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.join(part, on=["season", "week", "team"], how="full", coalesce=True)
    joined = merged.join(lookup, on=["season", "week", "team"], how="inner")
    for col in TEAM_MICRO_NGS_COLUMNS:
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(0.0).alias(col))
    return joined.select(
        "game_id",
        "team",
        *[pl.col(c).fill_null(0.0).cast(pl.Float64) for c in TEAM_MICRO_NGS_COLUMNS],
    )


def aggregate_player_ngs(
    games_df: pl.DataFrame,
    *,
    rushing: pl.DataFrame | None = None,
    receiving: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return `(game_id, team, player_id) + PLAYER_MICRO_NGS_COLUMNS` for NGS rushers/receivers."""
    lookup = _lookup(games_df)
    frames: list[pl.DataFrame] = []
    r = _weekly(rushing, ("player_gsis_id", "rush_attempts", "rush_yards_over_expected"))
    if r is not None:
        frames.append(
            r.select(
                "season",
                "week",
                "team",
                pl.col("player_gsis_id")
                .map_elements(_gsis, return_dtype=pl.Int64)
                .alias("player_id"),
                pl.col("rush_attempts").cast(pl.Float64).alias("ngs_rush_attempts"),
                pl.col("rush_yards_over_expected").cast(pl.Float64).alias("ngs_ryoe_sum"),
                pl.lit(0.0).alias("ngs_targets"),
                pl.lit(0.0).alias("ngs_separation_sum"),
            )
        )
    c = _weekly(receiving, ("player_gsis_id", "targets", "avg_separation"))
    if c is not None:
        frames.append(
            c.select(
                "season",
                "week",
                "team",
                pl.col("player_gsis_id")
                .map_elements(_gsis, return_dtype=pl.Int64)
                .alias("player_id"),
                pl.lit(0.0).alias("ngs_rush_attempts"),
                pl.lit(0.0).alias("ngs_ryoe_sum"),
                pl.col("targets").cast(pl.Float64).alias("ngs_targets"),
                (pl.col("avg_separation") * pl.col("targets")).alias("ngs_separation_sum"),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "team": pl.Utf8,
                "player_id": pl.Int64,
                **{c: pl.Float64 for c in PLAYER_MICRO_NGS_COLUMNS},
            }
        )
    stacked = pl.concat(frames).filter(pl.col("player_id").is_not_null())
    grouped = stacked.group_by("season", "week", "team", "player_id").agg(
        *[pl.col(c).sum() for c in PLAYER_MICRO_NGS_COLUMNS]
    )
    return grouped.join(lookup, on=["season", "week", "team"], how="inner").select(
        "game_id", "team", "player_id", *PLAYER_MICRO_NGS_COLUMNS
    )


NGS_ASSET_TYPES: Final[tuple[str, ...]] = ("passing", "rushing", "receiving")

__all__ = [
    "NGS_ASSET_TYPES",
    "NGS_FIRST_SEASON",
    "aggregate_player_ngs",
    "aggregate_team_ngs",
]
