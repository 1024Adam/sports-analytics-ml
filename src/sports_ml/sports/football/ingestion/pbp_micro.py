"""Play-by-play micro aggregation (`football/ingestion/pbp_micro.py`).

Compresses nflverse play-by-play (plus Next Gen Stats from `ngs.py`) into two Silver tables:

* `team_pbp_micro_logs.parquet`: one offense-side row per `(game_id, team)` of additive
  numerators/denominators (`TEAM_MICRO_PBP_COLUMNS`, `TEAM_MICRO_NGS_COLUMNS`). The defensive
  view of any metric is the opponent's row.
* `player_pbp_micro_logs.parquet`: one row per `(game_id, team, player_id)` with receiving and
  rushing micro usage plus NGS rushing/receiving totals.

`epa_total` and `plays` use exactly the `team_game_logs` scrimmage definition (offense `pass`/`run`
plays, kneels and spikes included), so they reconcile to `off_epa_total` / `off_plays`. Every other
metric excludes QB kneels and spikes. **Neutral situations** (`NeutralSituationConfig`) require the
win probability inside `[wp_min, wp_max]`, a score margin within `max_abs_score_diff`, more than
`min_half_seconds` left in the half, regulation, and down `<= max_down`.

Pressure is a play-by-play *proxy* (`sack or qb_hit` per dropback): nflverse play-by-play has no
charted pressure, so `pressure_num` is flagged as a proxy in the docs and never mixed with a
charted source.
"""

from __future__ import annotations

from typing import Any, Final

import polars as pl
from pydantic import BaseModel, ConfigDict

from sports_ml.sports.football.adapter import gsis_id_to_int
from sports_ml.sports.football.constants import (
    EXPLOSIVE_PASS_YARDS,
    PLAYER_MICRO_NGS_COLUMNS,
    PLAYER_MICRO_PBP_COLUMNS,
    RED_ZONE_YARDLINE_100,
    SILVER_FOOTBALL_PLAYER_MICRO_SCHEMA,
    SILVER_FOOTBALL_TEAM_MICRO_SCHEMA,
    TEAM_MICRO_NGS_COLUMNS,
    TEAM_MICRO_PBP_COLUMNS,
    WP_BAND_MAX,
    WP_BAND_MIN,
    canonical_game_id_from_nflverse,
    normalize_football_team,
)
from sports_ml.sports.football.ingestion.ngs import aggregate_player_ngs, aggregate_team_ngs

PACE_MAX_SECONDS: Final[float] = 60.0
FOURTH_DOWN_MAX_YDSTOGO: Final[float] = 4.0
FOURTH_DOWN_MAX_YARDLINE_100: Final[float] = 65.0
TWO_MINUTE_SECONDS: Final[float] = 120.0

REQUIRED_MICRO_PBP_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "play_id",
    "posteam",
    "play_type",
    "epa",
    "pass",
    "rush",
    "qb_dropback",
    "wp",
    "down",
    "yards_gained",
)
_OPTIONAL_NUMERIC: Final[tuple[str, ...]] = (
    "xpass",
    "cpoe",
    "air_yards",
    "yards_after_catch",
    "xyac_mean_yardage",
    "qb_hit",
    "sack",
    "score_differential",
    "qtr",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "penalty",
    "complete_pass",
    "pass_attempt",
    "ydstogo",
    "yardline_100",
    "drive",
    "qb_kneel",
    "qb_spike",
    "receiving_yards",
)
_OPTIONAL_TEXT: Final[tuple[str, ...]] = ("receiver_player_id", "rusher_player_id")


class NeutralSituationConfig(BaseModel):
    """Score/time band that defines a neutral game situation."""

    model_config = ConfigDict(frozen=True)

    wp_min: float = 0.20
    wp_max: float = 0.80
    max_abs_score_diff: float = 14.0
    min_half_seconds: float = 120.0
    max_down: int = 3


def neutral_situation_mask(config: NeutralSituationConfig | None = None) -> pl.Expr:
    """Boolean expression: the play is in a neutral score/time situation (null -> False)."""
    cfg = config or NeutralSituationConfig()
    return (
        (pl.col("wp") >= cfg.wp_min)
        & (pl.col("wp") <= cfg.wp_max)
        & (pl.col("score_differential").abs() <= cfg.max_abs_score_diff)
        & (pl.col("half_seconds_remaining") > cfg.min_half_seconds)
        & (pl.col("qtr") <= 4)
        & (pl.col("down") >= 1)
        & (pl.col("down") <= cfg.max_down)
    ).fill_null(False)


def _gsis(value: Any) -> int | None:
    try:
        return gsis_id_to_int(value)
    except (ValueError, TypeError):
        return None


def _flag(name: str) -> pl.Expr:
    return pl.col(name).fill_null(0) == 1


def prepare_micro_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Canonical `_gid`, normalized `posteam`, float casts, and the next-snap clock delta `_dt`."""
    missing = [c for c in REQUIRED_MICRO_PBP_COLUMNS if c not in pbp_df.columns]
    if missing:
        raise ValueError(f"Play-by-play frame is missing columns: {missing}.")
    ids = pbp_df["game_id"].unique().drop_nulls().to_list()
    mapping = {gid: canonical_game_id_from_nflverse(gid) for gid in ids}
    frame = pbp_df.with_columns(
        *[
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in _OPTIONAL_NUMERIC
            if c not in pbp_df.columns
        ],
        *[pl.lit(None, dtype=pl.Utf8).alias(c) for c in _OPTIONAL_TEXT if c not in pbp_df.columns],
    )
    frame = frame.with_columns(
        pl.col("game_id")
        .replace_strict(mapping, default=None, return_dtype=pl.Int64)
        .alias("_gid"),
        pl.col("posteam").map_elements(
            normalize_football_team, return_dtype=pl.Utf8, skip_nulls=True
        ),
        *[
            pl.col(c).cast(pl.Float64, strict=False)
            for c in (
                "epa",
                "wp",
                "yards_gained",
                "down",
                *[c for c in _OPTIONAL_NUMERIC if c != "drive"],
                "pass",
                "rush",
                "qb_dropback",
            )
        ],
    ).filter(pl.col("_gid").is_not_null())
    return (
        frame.sort("_gid", "play_id")
        .with_columns(
            (
                pl.col("game_seconds_remaining")
                - pl.col("game_seconds_remaining").shift(-1).over(["_gid", "drive"])
            ).alias("_dt")
        )
        .with_columns(
            pl.when((pl.col("_dt") > 0.0) & (pl.col("_dt") <= PACE_MAX_SECONDS))
            .then(pl.col("_dt"))
            .alias("_dt")
        )
    )


def _team_lookup(games_df: pl.DataFrame) -> pl.DataFrame:
    base = ["game_id", "event_seq", "start_time_utc", "season", "week"]
    home = games_df.select(
        *base, pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent")
    )
    away = games_df.select(
        *base, pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent")
    )
    return pl.concat([home, away]).with_columns(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int16),
        pl.col("event_seq").cast(pl.Int32),
    )


def aggregate_team_micro(
    prepared: pl.DataFrame, config: NeutralSituationConfig | None = None
) -> pl.DataFrame:
    """Aggregate prepared play-by-play into one offensive row per `(_gid, posteam)`."""
    neutral = neutral_situation_mask(config)
    is_pass, is_rush = _flag("pass"), _flag("rush")
    scrimmage_all = prepared.filter(
        pl.col("posteam").is_not_null() & pl.col("play_type").is_in(["pass", "run"])
    )
    totals = scrimmage_all.group_by("_gid", "posteam").agg(
        pl.col("epa").sum().alias("epa_total"),
        pl.len().cast(pl.Float64).alias("plays"),
    )

    scrim = scrimmage_all.filter(~_flag("qb_kneel") & ~_flag("qb_spike"))
    epa_ok = pl.col("epa").is_not_null()
    dropback = _flag("qb_dropback")
    attempt = _flag("pass_attempt")
    complete = _flag("complete_pass")
    yac_ok = (
        complete
        & pl.col("yards_after_catch").is_not_null()
        & pl.col("xyac_mean_yardage").is_not_null()
    )
    two_min = (
        (pl.col("half_seconds_remaining") <= TWO_MINUTE_SECONDS)
        & pl.col("qtr").is_in([2.0, 4.0])
        & (pl.col("score_differential") <= 0)
    ).fill_null(False)
    adot_ok = pl.col("air_yards").is_not_null() & pl.col("receiver_player_id").is_not_null()
    pass_oe_ok = neutral & pl.col("xpass").is_not_null()
    seconds_ok = neutral & pl.col("_dt").is_not_null()
    in_rz = (pl.col("yardline_100") <= RED_ZONE_YARDLINE_100).fill_null(False)
    no_pen = ~_flag("penalty")

    def f(expr: pl.Expr) -> pl.Expr:
        return expr.cast(pl.Float64)

    plays = scrim.group_by("_gid", "posteam").agg(
        f(neutral.sum()).alias("neutral_plays"),
        pl.col("epa").filter(neutral).sum().alias("neutral_epa"),
        f((neutral & epa_ok).sum()).alias("neutral_epa_plays"),
        pl.col("_dt").filter(seconds_ok).sum().alias("neutral_seconds"),
        f(seconds_ok.sum()).alias("neutral_seconds_plays"),
        (is_pass.cast(pl.Float64) - pl.col("xpass")).filter(pass_oe_ok).sum().alias("proe_num"),
        f(pass_oe_ok.sum()).alias("proe_den"),
        f(two_min.sum()).alias("two_min_plays"),
        pl.col("epa").filter(two_min).sum().alias("two_min_epa"),
        f(dropback.sum()).alias("dropbacks"),
        pl.col("cpoe").sum().alias("cpoe_sum"),
        f(pl.col("cpoe").is_not_null().sum()).alias("cpoe_plays"),
        pl.col("air_yards").filter(attempt).sum().alias("air_yards_sum"),
        f((attempt & pl.col("air_yards").is_not_null()).sum()).alias("air_yards_plays"),
        pl.col("air_yards").filter(adot_ok).sum().alias("adot_sum"),
        f(adot_ok.sum()).alias("adot_plays"),
        (pl.col("yards_after_catch") - pl.col("xyac_mean_yardage"))
        .filter(yac_ok)
        .sum()
        .alias("yac_oe_sum"),
        f(yac_ok.sum()).alias("yac_oe_plays"),
        f(((_flag("qb_hit") | _flag("sack")) & dropback).sum()).alias("pressure_num"),
        f((_flag("sack") & dropback).sum()).alias("sacks"),
        pl.col("epa").filter(in_rz & epa_ok).sum().alias("rz_epa"),
        f((in_rz & epa_ok).sum()).alias("rz_plays"),
        pl.col("epa").filter((pl.col("down") == 3) & epa_ok).sum().alias("third_down_epa"),
        f(((pl.col("down") == 3) & epa_ok).sum()).alias("third_down_plays"),
        f((is_pass & (pl.col("yards_gained") >= EXPLOSIVE_PASS_YARDS)).sum()).alias(
            "explosive_pass"
        ),
        f(is_pass.sum()).alias("pass_plays"),
        f((is_rush & (pl.col("yards_gained") <= 0)).sum()).alias("run_stuffs"),
        f(is_rush.sum()).alias("rush_plays"),
        pl.col("epa").filter(no_pen & epa_ok).sum().alias("nopen_epa"),
        f((no_pen & epa_ok).sum()).alias("nopen_plays"),
    )

    fourth = (
        prepared.filter(
            pl.col("posteam").is_not_null()
            & (pl.col("down") == 4)
            & pl.col("play_type").is_in(["pass", "run", "punt", "field_goal"])
            & ~_flag("qb_kneel")
            & ~_flag("qb_spike")
            & (pl.col("ydstogo") <= FOURTH_DOWN_MAX_YDSTOGO)
            & (pl.col("yardline_100") <= FOURTH_DOWN_MAX_YARDLINE_100)
            & (pl.col("wp") >= WP_BAND_MIN)
            & (pl.col("wp") <= WP_BAND_MAX)
        )
        .group_by("_gid", "posteam")
        .agg(
            f(pl.col("play_type").is_in(["pass", "run"]).sum()).alias("fourth_go_num"),
            pl.len().cast(pl.Float64).alias("fourth_go_den"),
        )
    )
    return (
        totals.join(plays, on=["_gid", "posteam"], how="left")
        .join(fourth, on=["_gid", "posteam"], how="left")
        .rename({"_gid": "game_id", "posteam": "team"})
    )


def build_team_pbp_micro_logs(
    pbp_df: pl.DataFrame,
    games_df: pl.DataFrame,
    *,
    ngs_passing: pl.DataFrame | None = None,
    ngs_rushing: pl.DataFrame | None = None,
    ngs_receiving: pl.DataFrame | None = None,
    config: NeutralSituationConfig | None = None,
) -> pl.DataFrame:
    """Return the Silver `team_pbp_micro_logs` rows for every game present in `pbp_df`."""
    if pbp_df.is_empty() or games_df.is_empty():
        return pl.DataFrame(schema=SILVER_FOOTBALL_TEAM_MICRO_SCHEMA)
    agg = aggregate_team_micro(prepare_micro_pbp(pbp_df), config)
    ngs = aggregate_team_ngs(
        games_df, passing=ngs_passing, rushing=ngs_rushing, receiving=ngs_receiving
    )
    out = (
        _team_lookup(games_df)
        .join(agg, on=["game_id", "team"], how="inner")
        .join(ngs, on=["game_id", "team"], how="left")
    )
    numeric = [*TEAM_MICRO_PBP_COLUMNS, *TEAM_MICRO_NGS_COLUMNS]
    out = out.with_columns(
        *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in numeric],
    ).with_columns(
        ((pl.col("ngs_pass_attempts") + pl.col("ngs_rush_attempts") + pl.col("ngs_targets")) > 0)
        .cast(pl.Int8)
        .alias("has_ngs")
    )
    return out.select(list(SILVER_FOOTBALL_TEAM_MICRO_SCHEMA)).sort(
        "start_time_utc", "event_seq", "game_id", "team"
    )


def build_player_pbp_micro_logs(
    pbp_df: pl.DataFrame,
    games_df: pl.DataFrame,
    *,
    ngs_rushing: pl.DataFrame | None = None,
    ngs_receiving: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return the Silver `player_pbp_micro_logs` rows (receivers and rushers)."""
    if pbp_df.is_empty() or games_df.is_empty():
        return pl.DataFrame(schema=SILVER_FOOTBALL_PLAYER_MICRO_SCHEMA)
    prepared = prepare_micro_pbp(pbp_df)
    scrim = prepared.filter(
        pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & ~_flag("qb_kneel")
        & ~_flag("qb_spike")
    )
    to_id = pl.col("receiver_player_id").map_elements(_gsis, return_dtype=pl.Int64, skip_nulls=True)
    complete = _flag("complete_pass")
    rec_yards = pl.coalesce(pl.col("receiving_yards"), pl.col("yards_gained"))
    yac_ok = (
        complete
        & pl.col("yards_after_catch").is_not_null()
        & pl.col("xyac_mean_yardage").is_not_null()
    )
    receivers = (
        scrim.filter(
            _flag("pass_attempt") & pl.col("receiver_player_id").is_not_null() & ~_flag("sack")
        )
        .with_columns(to_id.alias("player_id"))
        .filter(pl.col("player_id").is_not_null())
        .group_by("_gid", "posteam", "player_id")
        .agg(
            pl.len().cast(pl.Float64).alias("targets"),
            pl.col("air_yards").sum().alias("target_air_yards"),
            complete.sum().cast(pl.Float64).alias("receptions"),
            rec_yards.filter(complete).sum().alias("rec_yards"),
            (pl.col("yards_after_catch") - pl.col("xyac_mean_yardage"))
            .filter(yac_ok)
            .sum()
            .alias("yac_oe_sum"),
            yac_ok.sum().cast(pl.Float64).alias("yac_oe_plays"),
        )
    )
    rushers = (
        scrim.filter(_flag("rush") & pl.col("rusher_player_id").is_not_null())
        .with_columns(
            pl.col("rusher_player_id")
            .map_elements(_gsis, return_dtype=pl.Int64, skip_nulls=True)
            .alias("player_id")
        )
        .filter(pl.col("player_id").is_not_null())
        .group_by("_gid", "posteam", "player_id")
        .agg(
            pl.len().cast(pl.Float64).alias("rush_attempts"),
            pl.col("yards_gained").sum().alias("rush_yards"),
        )
    )
    merged = receivers.join(rushers, on=["_gid", "posteam", "player_id"], how="full", coalesce=True)
    merged = merged.rename({"_gid": "game_id", "posteam": "team"})
    ngs = aggregate_player_ngs(games_df, rushing=ngs_rushing, receiving=ngs_receiving)
    merged = merged.join(ngs, on=["game_id", "team", "player_id"], how="full", coalesce=True)
    lookup = _team_lookup(games_df)
    out = merged.join(lookup, on=["game_id", "team"], how="inner")
    numeric = [*PLAYER_MICRO_PBP_COLUMNS, *PLAYER_MICRO_NGS_COLUMNS]
    out = out.with_columns(
        *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in numeric]
    ).with_columns(
        ((pl.col("ngs_rush_attempts") + pl.col("ngs_targets")) > 0).cast(pl.Int8).alias("has_ngs")
    )
    return out.select(list(SILVER_FOOTBALL_PLAYER_MICRO_SCHEMA)).sort(
        "start_time_utc", "event_seq", "game_id", "team", "player_id"
    )


__all__ = [
    "NeutralSituationConfig",
    "aggregate_team_micro",
    "build_player_pbp_micro_logs",
    "build_team_pbp_micro_logs",
    "neutral_situation_mask",
    "prepare_micro_pbp",
]
