"""Leak-free pre-game micro features from `team_pbp_micro_logs` (Milestone 28).

Each metric is an exposure-weighted rate `sum(num) / sum(den)` over the team's previous
`ROLL_WINDOW` games of the season (`shift(1)` inside `[team, season]`, via
`core.temporal.pregame_exposure_rate`), shrunk toward a prior that blends last season's team rate
with last season's league rate (`PRIOR_TEAM_WEIGHT`) using `core.shrinkage.bayesian_shrinkage`
with the number of games that actually carry the metric's exposure. Metrics without any exposure
(for example Next Gen Stats before 2016) therefore resolve to the league prior with no NaN, and
`ngs_is_imputed` / `micro_is_imputed` flag the rows that have no history at all.

Matchup columns follow the Stage 1 convention: `diff_<name> = home - away` of a per-team value
that is either offense-only or an `off - def` / `def - off` edge, so a higher value is always
better for the team (except pace, a raw seconds-per-play value, and the over-expected rates).
`pace_total_neutral` and `proe_total` are symmetric level features for the expected total.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import pregame_exposure_rate
from sports_ml.sports.football.constants import TEAM_MICRO_NGS_COLUMNS, TEAM_MICRO_PBP_COLUMNS

K_MICRO_GAMES: Final[float] = 8.0
ROLL_WINDOW: Final[int] = 8
PRIOR_TEAM_WEIGHT: Final[float] = 0.5


class MicroMetric(NamedTuple):
    """A two-sided exposure-normalized rate over micro-log numerator/denominator columns."""

    name: str
    num: str
    den: str
    default: float
    two_sided: bool = True


MICRO_METRICS: Final[tuple[MicroMetric, ...]] = (
    MicroMetric("proe", "proe_num", "proe_den", 0.0, two_sided=False),
    MicroMetric("neutral_pace", "neutral_seconds", "neutral_seconds_plays", 29.5, two_sided=False),
    MicroMetric("pressure_rate", "pressure_num", "dropbacks", 0.145),
    MicroMetric("red_zone_epa", "rz_epa", "rz_plays", 0.0),
    MicroMetric("cpoe", "cpoe_sum", "cpoe_plays", 0.0),
    MicroMetric("fourth_go_rate", "fourth_go_num", "fourth_go_den", 0.40, two_sided=False),
    MicroMetric("yac_over_expected", "yac_oe_sum", "yac_oe_plays", 0.0),
    MicroMetric("third_down_epa", "third_down_epa", "third_down_plays", 0.0),
    MicroMetric("run_stuff_rate", "run_stuffs", "rush_plays", 0.17),
    MicroMetric("explosive_pass_rate", "explosive_pass", "pass_plays", 0.08),
    MicroMetric("neutral_epa", "neutral_epa", "neutral_epa_plays", 0.0),
    MicroMetric("penalty_adj_epa", "nopen_epa", "nopen_plays", 0.0),
    MicroMetric("sack_rate", "sacks", "dropbacks", 0.065),
    MicroMetric("two_minute_epa", "two_min_epa", "two_min_plays", 0.0, two_sided=False),
    MicroMetric(
        "time_to_throw", "ngs_time_to_throw_sum", "ngs_pass_attempts", 2.75, two_sided=False
    ),
    MicroMetric("ryoe_per_att", "ngs_ryoe_sum", "ngs_rush_attempts", 0.0),
    MicroMetric("separation", "ngs_separation_sum", "ngs_targets", 3.0, two_sided=False),
)

# Per-team values derived from the rolled shrunk `off`/`def` rates of `MICRO_METRICS`.
MICRO_TEAM_FEATURES: Final[tuple[str, ...]] = (
    "proe",
    "neutral_pace",
    "pressure_rate_edge",
    "red_zone_epa",
    "cpoe_vs_coverage",
    "fourth_down_go_over_expected",
    "yac_over_expected",
    "third_down_epa",
    "run_stuff_rate_edge",
    "explosive_pass_rate_edge",
    "neutral_epa",
    "penalty_adj_epa",
    "sack_rate_edge",
    "two_minute_epa",
    "time_to_throw",
    "ryoe_per_att",
    "separation",
)
MICRO_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *[f"diff_{n}" for n in MICRO_TEAM_FEATURES],
    "pace_total_neutral",
    "proe_total",
    "home_micro_is_imputed",
    "away_micro_is_imputed",
    "home_ngs_is_imputed",
    "away_ngs_is_imputed",
)
# Stage 1 ships only the five roadmap matchup diffs (+ imputation flags): on 2012-2026 nflverse data
# the wider set (NGS, pace/PROE totals, ...) did not improve out-of-sample Log Loss or score MAE.
STAGE1_MICRO_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "diff_proe",
    "diff_pressure_rate_edge",
    "diff_neutral_pace",
    "diff_red_zone_epa",
    "diff_cpoe_vs_coverage",
    "home_micro_is_imputed",
    "away_micro_is_imputed",
)
_REQUIRED_LOG_COLUMNS: Final[tuple[str, ...]] = (
    *TEAM_MICRO_PBP_COLUMNS,
    *TEAM_MICRO_NGS_COLUMNS,
    "has_ngs",
)


def _safe_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return num / pl.when(den > 0.0).then(den).otherwise(None)


def _team_rows(games_df: pl.DataFrame) -> pl.DataFrame:
    base = ["game_id", "event_seq", "season", "start_time_utc"]
    home = games_df.select(
        *base, pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent")
    )
    away = games_df.select(
        *base, pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent")
    )
    return pl.concat([home, away]).with_columns(pl.col("season").cast(pl.Int32))


def build_micro_team_features(
    games_df: pl.DataFrame,
    micro_logs: pl.DataFrame | None,
    *,
    k_games: float = K_MICRO_GAMES,
    window: int = ROLL_WINDOW,
) -> pl.DataFrame:
    """Per-`(game_id, team)` pre-game micro values (`MICRO_TEAM_FEATURES` + imputation flags)."""
    own_cols = [*TEAM_MICRO_PBP_COLUMNS, *TEAM_MICRO_NGS_COLUMNS, "has_ngs"]
    long = _team_rows(games_df)
    if micro_logs is None or micro_logs.is_empty():
        micro = pl.DataFrame(
            schema={"game_id": pl.Int64, "team": pl.Utf8, **{c: pl.Float64 for c in own_cols}}
        )
    else:
        missing = [c for c in _REQUIRED_LOG_COLUMNS if c not in micro_logs.columns]
        if missing:
            raise ValueError(f"Micro logs are missing required columns: {missing}.")
        micro = micro_logs.select(
            "game_id", "team", *[pl.col(c).cast(pl.Float64) for c in own_cols]
        )
    own = micro.with_columns(pl.lit(True).alias("_played")).rename(
        {c: f"s_{c}" for c in [*own_cols, "_played"]}
    )
    opp = (
        micro.with_columns(pl.lit(True).alias("_played"))
        .rename({c: f"o_{c}" for c in [*own_cols, "_played"]})
        .rename({"team": "opponent"})
    )
    long = (
        long.join(own, on=["game_id", "team"], how="left")
        .join(opp, on=["game_id", "opponent"], how="left")
        .with_columns(pl.col("s__played").fill_null(False))
        .sort("start_time_utc", "event_seq", "game_id", "team")
    )

    group = ["team", "season"]
    prefixes: list[tuple[MicroMetric, str]] = []
    exprs: list[pl.Expr] = []
    for m in MICRO_METRICS:
        for side, src in (("off", "s"), ("def", "o")):
            if side == "def" and not m.two_sided:
                continue
            p = f"{side}_{m.name}"
            prefixes.append((m, p))
            num = pl.col(f"{src}_{m.num}").cast(pl.Float64).fill_null(0.0)
            den = pl.col(f"{src}_{m.den}").cast(pl.Float64).fill_null(0.0)
            exprs.extend([num.alias(f"{p}_num"), den.alias(f"{p}_den")])
    long = long.with_columns(*exprs)
    long = long.with_columns(
        *[(pl.col(f"{p}_den") > 0.0).cast(pl.Float64).alias(f"{p}_g") for _m, p in prefixes],
        pl.col("s__played").cast(pl.Float64).alias("_played_f"),
        pl.col("s_has_ngs").fill_null(0.0).alias("_ngs_f"),
    )
    long = long.with_columns(
        *[
            pl.col(f"{p}_g").cum_sum().shift(1).over(group).fill_null(0.0).alias(f"{p}_gcum")
            for _m, p in prefixes
        ],
        pl.col("_played_f").cum_sum().shift(1).over("team").fill_null(0.0).alias("_hist_all"),
        pl.col("_ngs_f").cum_sum().shift(1).over("team").fill_null(0.0).alias("_ngs_hist_all"),
    )

    played = long.filter(pl.col("s__played"))
    team_prev = (
        played.group_by("team", "season")
        .agg(
            *[
                pl.col(f"{p}_{k}").sum().alias(f"pt_{p}_{k}")
                for _m, p in prefixes
                for k in ("num", "den")
            ]
        )
        .with_columns((pl.col("season") + 1).alias("season"))
    )
    league_prev = (
        played.group_by("season")
        .agg(
            *[
                pl.col(f"{p}_{k}").sum().alias(f"pl_{p}_{k}")
                for _m, p in prefixes
                for k in ("num", "den")
            ]
        )
        .with_columns((pl.col("season") + 1).alias("season"))
    )
    long = long.join(team_prev, on=["team", "season"], how="left").join(
        league_prev, on="season", how="left"
    )

    feats: list[pl.Expr] = []
    league_rates: list[pl.Expr] = []
    for m, p in prefixes:
        team_rate = _safe_ratio(pl.col(f"pt_{p}_num"), pl.col(f"pt_{p}_den"))
        league_rate = pl.coalesce(
            _safe_ratio(pl.col(f"pl_{p}_num"), pl.col(f"pl_{p}_den")), pl.lit(m.default)
        )
        prior = (
            pl.when(team_rate.is_not_null())
            .then(PRIOR_TEAM_WEIGHT * team_rate + (1.0 - PRIOR_TEAM_WEIGHT) * league_rate)
            .otherwise(league_rate)
        )
        roll = pregame_exposure_rate(f"{p}_num", f"{p}_den", window, group)
        n_w = pl.min_horizontal(pl.col(f"{p}_gcum"), pl.lit(float(window)))
        feats.append(bayesian_shrinkage(roll, n_w, prior, k_games).alias(f"_v_{p}"))
        league_rates.append(league_rate.alias(f"_lg_{p}"))
    long = long.with_columns(*feats, *league_rates)

    def v(name: str, side: str) -> pl.Expr:
        return pl.col(f"_v_{side}_{name}")

    long = long.with_columns(
        v("proe", "off").alias("proe"),
        v("neutral_pace", "off").alias("neutral_pace"),
        (v("pressure_rate", "def") - v("pressure_rate", "off")).alias("pressure_rate_edge"),
        (v("red_zone_epa", "off") - v("red_zone_epa", "def")).alias("red_zone_epa"),
        (v("cpoe", "off") - v("cpoe", "def")).alias("cpoe_vs_coverage"),
        (v("fourth_go_rate", "off") - pl.col("_lg_off_fourth_go_rate")).alias(
            "fourth_down_go_over_expected"
        ),
        (v("yac_over_expected", "off") - v("yac_over_expected", "def")).alias("yac_over_expected"),
        (v("third_down_epa", "off") - v("third_down_epa", "def")).alias("third_down_epa"),
        (v("run_stuff_rate", "def") - v("run_stuff_rate", "off")).alias("run_stuff_rate_edge"),
        (v("explosive_pass_rate", "off") - v("explosive_pass_rate", "def")).alias(
            "explosive_pass_rate_edge"
        ),
        (v("neutral_epa", "off") - v("neutral_epa", "def")).alias("neutral_epa"),
        (v("penalty_adj_epa", "off") - v("penalty_adj_epa", "def")).alias("penalty_adj_epa"),
        (v("sack_rate", "def") - v("sack_rate", "off")).alias("sack_rate_edge"),
        v("two_minute_epa", "off").alias("two_minute_epa"),
        v("time_to_throw", "off").alias("time_to_throw"),
        (v("ryoe_per_att", "off") - v("ryoe_per_att", "def")).alias("ryoe_per_att"),
        v("separation", "off").alias("separation"),
        (pl.col("_hist_all") == 0.0).cast(pl.Int8).alias("micro_is_imputed"),
        (pl.col("_ngs_hist_all") == 0.0).cast(pl.Int8).alias("ngs_is_imputed"),
    )
    return long.select(
        "game_id", "event_seq", "team", *MICRO_TEAM_FEATURES, "micro_is_imputed", "ngs_is_imputed"
    ).sort("game_id", "team")


def attach_micro_features(base: pl.DataFrame, team_features: pl.DataFrame) -> pl.DataFrame:
    """Join per-team micro values onto a per-game frame and form `diff_*`, totals, and flags."""
    cols = [*MICRO_TEAM_FEATURES, "micro_is_imputed", "ngs_is_imputed"]
    out = base
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        part = team_features.select("game_id", "team", *cols).rename(
            {"team": team_col, **{c: f"{side}_{c}" for c in cols}}
        )
        out = out.join(part, on=["game_id", team_col], how="left")
    out = out.with_columns(
        *[
            (pl.col(f"home_{n}") - pl.col(f"away_{n}")).alias(f"diff_{n}")
            for n in MICRO_TEAM_FEATURES
        ],
        (pl.col("home_neutral_pace") + pl.col("away_neutral_pace")).alias("pace_total_neutral"),
        (pl.col("home_proe") + pl.col("away_proe")).alias("proe_total"),
    )
    fills: list[pl.Expr] = [pl.col(f"diff_{n}").fill_null(0.0) for n in MICRO_TEAM_FEATURES]
    fills.extend(
        [
            pl.col("pace_total_neutral").fill_null(2.0 * 29.5),
            pl.col("proe_total").fill_null(0.0),
            *[
                pl.col(f"{s}_{f}").fill_null(1).cast(pl.Int8)
                for s in ("home", "away")
                for f in ("micro_is_imputed", "ngs_is_imputed")
            ],
        ]
    )
    return out.with_columns(fills).drop(
        [f"{s}_{n}" for s in ("home", "away") for n in MICRO_TEAM_FEATURES]
    )


__all__ = [
    "STAGE1_MICRO_FEATURE_COLUMNS",
    "MICRO_FEATURE_COLUMNS",
    "MICRO_METRICS",
    "MICRO_TEAM_FEATURES",
    "attach_micro_features",
    "build_micro_team_features",
]
