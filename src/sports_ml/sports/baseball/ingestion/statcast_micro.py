"""Pitch-level & batted-ball Statcast micro-feature aggregators (`statcast_micro.py`).

Compresses pitch-by-pitch and batted-ball tracking streams into compact per-game
micro-logs and computes strictly pre-game (`.shift(1)`) micro-features:
- `sp_velo_delta_mph`: Recent 2-start fastball velocity minus long-term baseline
  velocity (detecting acute fatigue or arm health decline before ERA spikes).
- `sp_swstr_pct`, `sp_o_swing_pct`, `sp_f_strike_pct`, `sp_xwoba_against`,
  `sp_barrel_pct_against`.
- `batter_xwoba`, `batter_barrel_per_pa`, `batter_hard_hit_pct`,
  `batter_sweet_spot_pct`, `batter_chase_pct`, and `batter_woba_minus_xwoba_luck`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Final

import polars as pl

from sports_ml.core.cache import read_cached_bytes, resolve_cache_read_path
from sports_ml.core.temporal import (
    pregame_rolling_mean,
    sort_events_chronologically,
)
from sports_ml.sports.baseball.constants import normalize_baseball_team

SILVER_PITCHER_MICRO_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "league": pl.Utf8(),
    "player_id": pl.Int64(),
    "team": pl.Utf8(),
    "avg_fastball_velo": pl.Float64(),
    "swstr_pct": pl.Float64(),
    "o_swing_pct": pl.Float64(),
    "f_strike_pct": pl.Float64(),
    "xwoba_against": pl.Float64(),
    "barrel_pct_against": pl.Float64(),
}

SILVER_BATTER_MICRO_LOGS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "event_seq": pl.Int32(),
    "game_date": pl.Date(),
    "start_time_utc": pl.Datetime("us", "UTC"),
    "season": pl.Int32(),
    "league": pl.Utf8(),
    "player_id": pl.Int64(),
    "team": pl.Utf8(),
    "batter_xwoba": pl.Float64(),
    "batter_barrel_per_pa": pl.Float64(),
    "batter_hard_hit_pct": pl.Float64(),
    "batter_sweet_spot_pct": pl.Float64(),
    "batter_chase_pct": pl.Float64(),
}


def _read_pitch_frame(source: pl.DataFrame | Path | bytes) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        return source
    if isinstance(source, bytes):
        return pl.read_csv(io.BytesIO(source))
    resolved = resolve_cache_read_path(source)
    if resolved is not None and resolved.suffix == ".gz":
        return pl.read_csv(io.BytesIO(read_cached_bytes(source)))
    return pl.read_csv(resolved if resolved is not None else source)


def aggregate_statcast_pitches_to_micro_logs(
    source: pl.DataFrame | Path | bytes,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate pitch-by-pitch Statcast records into pitcher and batter game micro-logs."""
    df = _read_pitch_frame(source)
    if df.is_empty():
        return (
            pl.DataFrame(schema=SILVER_PITCHER_MICRO_LOGS_SCHEMA),
            pl.DataFrame(schema=SILVER_BATTER_MICRO_LOGS_SCHEMA),
        )

    working = df
    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
    if "league" not in working.columns:
        working = working.with_columns(pl.lit("MLB", dtype=pl.Utf8).alias("league"))

    is_fb = pl.col("pitch_type").is_in(["FF", "SI", "FC", "FA"])
    is_whiff = pl.col("is_whiff").cast(pl.Float64)
    is_out_zone = pl.col("is_out_of_zone").cast(pl.Float64)
    is_chase = (pl.col("is_out_of_zone") & pl.col("is_swing")).cast(pl.Float64)
    is_first_p = pl.col("is_first_pitch").cast(pl.Float64)
    is_fp_strike = (pl.col("is_first_pitch") & pl.col("is_strike")).cast(pl.Float64)
    is_bip = pl.col("launch_speed").is_not_null()
    is_hard_hit = (pl.col("launch_speed") >= 95.0).cast(pl.Float64)
    is_sweet_spot = ((pl.col("launch_angle") >= 8.0) & (pl.col("launch_angle") <= 32.0)).cast(
        pl.Float64
    )
    is_barrel = pl.col("is_barrel").cast(pl.Float64)

    pitcher_micro = (
        working.group_by(
            [
                "game_id",
                "event_seq",
                "game_date",
                "start_time_utc",
                "season",
                "league",
                "pitcher_id",
                "pitcher_team",
            ]
        )
        .agg(
            pl.when(is_fb)
            .then(pl.col("release_speed"))
            .otherwise(None)
            .mean()
            .alias("avg_fastball_velo"),
            is_whiff.mean().fill_null(0.110).alias("swstr_pct"),
            (is_chase.sum() / is_out_zone.sum().clip(lower_bound=1.0))
            .fill_null(0.285)
            .alias("o_swing_pct"),
            (is_fp_strike.sum() / is_first_p.sum().clip(lower_bound=1.0))
            .fill_null(0.610)
            .alias("f_strike_pct"),
            pl.col("estimated_woba_using_speedangle")
            .mean()
            .fill_null(0.312)
            .alias("xwoba_against"),
            pl.when(is_bip)
            .then(is_barrel)
            .otherwise(None)
            .mean()
            .fill_null(0.068)
            .alias("barrel_pct_against"),
        )
        .rename({"pitcher_id": "player_id", "pitcher_team": "team"})
        .with_columns(
            pl.col("team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
            pl.col("avg_fastball_velo").fill_null(93.8),
        )
        .sort(["game_date", "start_time_utc", "event_seq", "game_id", "player_id"])
    )

    batter_micro = (
        working.group_by(
            [
                "game_id",
                "event_seq",
                "game_date",
                "start_time_utc",
                "season",
                "league",
                "batter_id",
                "batter_team",
            ]
        )
        .agg(
            pl.col("estimated_woba_using_speedangle").mean().fill_null(0.315).alias("batter_xwoba"),
            (is_barrel.sum() / is_first_p.sum().clip(lower_bound=1.0))
            .fill_null(0.045)
            .alias("batter_barrel_per_pa"),
            pl.when(is_bip)
            .then(is_hard_hit)
            .otherwise(None)
            .mean()
            .fill_null(0.390)
            .alias("batter_hard_hit_pct"),
            pl.when(is_bip)
            .then(is_sweet_spot)
            .otherwise(None)
            .mean()
            .fill_null(0.335)
            .alias("batter_sweet_spot_pct"),
            (is_chase.sum() / is_out_zone.sum().clip(lower_bound=1.0))
            .fill_null(0.285)
            .alias("batter_chase_pct"),
        )
        .rename({"batter_id": "player_id", "batter_team": "team"})
        .with_columns(pl.col("team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8))
        .sort(["game_date", "start_time_utc", "event_seq", "game_id", "player_id"])
    )

    return (
        pitcher_micro.select(
            [pl.col(c).cast(dt) for c, dt in SILVER_PITCHER_MICRO_LOGS_SCHEMA.items()]
        ),
        batter_micro.select(
            [pl.col(c).cast(dt) for c, dt in SILVER_BATTER_MICRO_LOGS_SCHEMA.items()]
        ),
    )


def compute_pregame_pitcher_velo_delta(
    pitcher_logs_df: pl.DataFrame,
    *,
    short_window: int = 2,
    baseline_window: int = 12,
) -> pl.DataFrame:
    """Compute strictly pre-game `.shift(1)` `sp_velo_delta_mph` (`velo_w2 - velo_w12`).

    A negative value (e.g. `-1.8 mph`) indicates an acute fastball velocity drop over
    the pitcher's last `short_window` starts relative to their `baseline_window` average,
    strictly before start `t` begins.
    """
    working = pitcher_logs_df
    if "game_id" in working.columns and "event_id" not in working.columns:
        working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
    if "avg_fastball_velo" not in working.columns:
        working = working.with_columns(pl.lit(93.8, dtype=pl.Float64).alias("avg_fastball_velo"))

    sorted_df = sort_events_chronologically(working, extra_tiebreakers=["player_id"])
    short_velo = pregame_rolling_mean(
        "avg_fastball_velo", window=short_window, group_by="player_id"
    )
    base_velo = pregame_rolling_mean(
        "avg_fastball_velo", window=baseline_window, group_by="player_id"
    )

    return sorted_df.with_columns(
        short_velo.fill_null(93.8).alias("sp_rolling_velo_short"),
        base_velo.fill_null(93.8).alias("sp_rolling_velo_baseline"),
        (short_velo - base_velo).fill_null(0.0).alias("sp_velo_delta_mph"),
    )


def compute_pregame_batter_woba_minus_xwoba_luck(
    batter_logs_df: pl.DataFrame,
    *,
    window: int = 20,
) -> pl.DataFrame:
    """Compute strictly pre-game `.shift(1)` `batter_woba_minus_xwoba_luck` (`wOBA - xwOBA`)."""
    working = batter_logs_df
    if "game_id" in working.columns and "event_id" not in working.columns:
        working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))

    sorted_df = sort_events_chronologically(working, extra_tiebreakers=["player_id"])
    single_woba = pl.col("woba_num").cast(pl.Float64) / pl.col("pa").cast(pl.Float64).clip(
        lower_bound=1.0
    )
    sorted_df = sorted_df.with_columns(
        single_woba.alias("_game_woba"),
        pl.col("xwoba").cast(pl.Float64).fill_null(single_woba).alias("_game_xwoba"),
    )

    roll_woba = pregame_rolling_mean("_game_woba", window=window, group_by="player_id")
    roll_xwoba = pregame_rolling_mean("_game_xwoba", window=window, group_by="player_id")

    return sorted_df.with_columns(
        (roll_woba - roll_xwoba).fill_null(0.0).alias("batter_woba_minus_xwoba_luck")
    ).drop(["_game_woba", "_game_xwoba"])
