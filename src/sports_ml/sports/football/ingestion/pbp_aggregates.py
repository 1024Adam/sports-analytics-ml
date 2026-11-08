"""Play-level aggregation with garbage-time / kneel / spike filtering (`pbp_aggregates.py`).

Produces the per-`(game, team)` offensive numerators/denominators stored in Silver
`team_game_logs.parquet`. The `f_off_*` columns only use scrimmage `pass`/`run` plays whose
offense win probability is inside `[WP_BAND_MIN, WP_BAND_MAX]` (garbage time removed) and
exclude QB kneels and spikes, so they never enter success-rate or EPA/play denominators.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from sports_ml.sports.football.constants import (
    EARLY_DOWNS,
    EXPLOSIVE_PASS_YARDS,
    EXPLOSIVE_RUSH_YARDS,
    RED_ZONE_YARDLINE_100,
    WP_BAND_MAX,
    WP_BAND_MIN,
)

REQUIRED_PBP_COLUMNS: Final[tuple[str, ...]] = (
    "play_id",
    "posteam",
    "play_type",
    "epa",
    "success",
    "yards_gained",
    "pass",
    "rush",
    "qb_dropback",
    "wp",
    "down",
    "qb_kneel",
    "qb_spike",
    "yardline_100",
    "fixed_drive_result",
    "drive",
    "fumble",
    "fumble_lost",
    "_gid",
)

DRIVE_RESULT_POINTS: Final[dict[str, float]] = {"Touchdown": 7.0, "Field goal": 3.0}


def _flag(name: str) -> pl.Expr:
    return (pl.col(name).fill_null(0) == 1) if name != "down" else pl.col(name)


def scrimmage_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return offensive `pass`/`run` plays excluding QB kneels and spikes."""
    missing = [c for c in REQUIRED_PBP_COLUMNS if c not in pbp.columns]
    if missing:
        raise ValueError(f"Play-by-play frame is missing columns: {missing}.")
    return pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & ~_flag("qb_kneel")
        & ~_flag("qb_spike")
    )


def in_win_probability_band(wp: pl.Expr) -> pl.Expr:
    """Return the non-garbage-time predicate `WP_BAND_MIN <= wp <= WP_BAND_MAX` (null -> False)."""
    return (wp >= WP_BAND_MIN) & (wp <= WP_BAND_MAX)


def aggregate_filtered_offense(pbp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate play-by-play into one offensive row per `(_gid, posteam)`.

    Returned columns: `f_off_*` (garbage-time filtered plays, EPA, dropbacks, rushes, success,
    early-down, explosive plays, drives and drive points), plus unfiltered `off_fumbles`,
    `off_fumbles_lost`, `off_rz_drives`, and `off_rz_td_drives`.
    """
    scrim = scrimmage_plays(pbp)
    filt = scrim.filter(in_win_probability_band(pl.col("wp")) & pl.col("epa").is_not_null())
    is_pass = pl.col("pass").fill_null(0) == 1
    is_rush = pl.col("rush").fill_null(0) == 1
    success = pl.col("success").fill_null(0) == 1
    early = pl.col("down").cast(pl.Int64, strict=False).is_in(list(EARLY_DOWNS))
    explosive = (is_pass & (pl.col("yards_gained") >= EXPLOSIVE_PASS_YARDS)) | (
        is_rush & (pl.col("yards_gained") >= EXPLOSIVE_RUSH_YARDS)
    )
    plays = filt.group_by("_gid", "posteam").agg(
        pl.len().alias("f_off_plays"),
        pl.col("epa").sum().alias("f_off_epa"),
        (pl.col("qb_dropback").fill_null(0) == 1).sum().alias("f_off_dropbacks"),
        pl.col("epa").filter(is_pass).sum().alias("f_off_pass_epa"),
        is_rush.sum().alias("f_off_rushes"),
        pl.col("epa").filter(is_rush).sum().alias("f_off_rush_epa"),
        success.sum().alias("f_off_success"),
        early.sum().alias("f_off_early_plays"),
        (early & success).sum().alias("f_off_early_success"),
        explosive.sum().alias("f_off_explosive"),
    )

    raw = scrim.group_by("_gid", "posteam").agg(
        (pl.col("fumble").fill_null(0) == 1).sum().alias("off_fumbles"),
        (pl.col("fumble_lost").fill_null(0) == 1).sum().alias("off_fumbles_lost"),
    )

    drives = (
        scrim.filter(pl.col("drive").is_not_null())
        .sort("_gid", "posteam", "drive", "play_id")
        .group_by("_gid", "posteam", "drive", maintain_order=True)
        .agg(
            pl.col("wp").first().alias("_first_wp"),
            pl.col("fixed_drive_result").drop_nulls().first().alias("_result"),
            pl.col("yardline_100").min().alias("_min_yardline"),
        )
        .with_columns(
            in_win_probability_band(pl.col("_first_wp")).alias("_in_band"),
            pl.col("_result").replace_strict(DRIVE_RESULT_POINTS, default=0.0).alias("_points"),
            (pl.col("_min_yardline") <= RED_ZONE_YARDLINE_100).fill_null(False).alias("_rz"),
        )
        .group_by("_gid", "posteam")
        .agg(
            pl.col("_in_band").sum().alias("f_off_drives"),
            pl.col("_points").filter(pl.col("_in_band")).sum().alias("f_off_drive_points"),
            pl.col("_rz").sum().alias("off_rz_drives"),
            (pl.col("_rz") & (pl.col("_result") == "Touchdown")).sum().alias("off_rz_td_drives"),
        )
    )
    return (
        raw.join(plays, on=["_gid", "posteam"], how="left")
        .join(drives, on=["_gid", "posteam"], how="left")
        .with_columns(
            pl.col("f_off_plays", "f_off_dropbacks", "f_off_rushes", "f_off_success").fill_null(0),
            pl.col("f_off_early_plays", "f_off_early_success", "f_off_explosive").fill_null(0),
            pl.col("f_off_epa", "f_off_pass_epa", "f_off_rush_epa").fill_null(0.0),
            pl.col("f_off_drives", "off_rz_drives", "off_rz_td_drives").fill_null(0),
            pl.col("f_off_drive_points").fill_null(0.0),
        )
    )


GOAL_LINE_YARDLINE_100: Final[float] = 5.0


def aggregate_player_pbp_usage(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return per `(_gid, gsis_id)` red-zone targets/carries and goal-line carries.

    Needs `receiver_player_id`, `rusher_player_id`, `yardline_100`, `pass_attempt`, `play_type`;
    returns an empty frame when those play-level columns are unavailable.
    """
    needed = {
        "_gid",
        "receiver_player_id",
        "rusher_player_id",
        "yardline_100",
        "pass_attempt",
        "play_type",
        "qb_kneel",
        "qb_spike",
    }
    schema = {
        "_gid": pl.Int64,
        "gsis_id": pl.Utf8,
        "rz_targets": pl.Int64,
        "rz_carries": pl.Int64,
        "gl_carries": pl.Int64,
    }
    if not needed.issubset(pbp.columns):
        return pl.DataFrame(schema=schema)
    base = pbp.filter(
        pl.col("yardline_100").is_not_null()
        & ~_flag("qb_kneel")
        & ~_flag("qb_spike")
        & (pl.col("yardline_100") <= RED_ZONE_YARDLINE_100)
    )
    targets = (
        base.filter(
            (pl.col("pass_attempt").fill_null(0) == 1) & pl.col("receiver_player_id").is_not_null()
        )
        .group_by("_gid", pl.col("receiver_player_id").alias("gsis_id"))
        .agg(pl.len().alias("rz_targets"))
    )
    rushes = (
        base.filter((pl.col("play_type") == "run") & pl.col("rusher_player_id").is_not_null())
        .group_by("_gid", pl.col("rusher_player_id").alias("gsis_id"))
        .agg(
            pl.len().alias("rz_carries"),
            (pl.col("yardline_100") <= GOAL_LINE_YARDLINE_100).sum().alias("gl_carries"),
        )
    )
    return (
        targets.join(rushes, on=["_gid", "gsis_id"], how="full", coalesce=True)
        .with_columns(pl.col("rz_targets", "rz_carries", "gl_carries").fill_null(0))
        .select(list(schema))
        .cast(schema)  # type: ignore[arg-type]
    )
