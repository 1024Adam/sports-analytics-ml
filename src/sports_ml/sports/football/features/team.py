"""Leak-free NFL team features: rolling EPA/efficiency, SoS adjustment, luck & cold-start priors.

Every feature for game `t` uses only the team's *previous* games (`shift(1)` inside
`[team, season]`), prior-season aggregates, and ridge ratings fit on games from earlier weeks.
Early-season cold starts are handled by Bayesian shrinkage in *games played*
(`(n * current + k * prior) / (n + k)`, `k = K_TEAM_GAMES`), so the weight on current-season
data is `n / (n + k)`: it rises monotonically with games played and tends to 1 with no
hard-coded week cutoff. The prior blends the franchise's prior-season rate with the previous
league mean, weighted by roster continuity (returning production, head-coach change).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Final, NamedTuple

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import pregame_ewma, pregame_exposure_rate
from sports_ml.sports.football.constants import normalize_football_team

K_TEAM_GAMES: Final[float] = 8.0
EWMA_ALPHA: Final[float] = 0.30
ROLLING_WINDOWS: Final[tuple[int, ...]] = (4, 8)
PRIOR_WEIGHT_MIN: Final[float] = 0.35
PRIOR_WEIGHT_MAX: Final[float] = 0.85
DEFAULT_ROSTER_CONTINUITY: Final[float] = 0.60
COACH_CHANGE_PRIOR_MULTIPLIER: Final[float] = 0.85
PYTHAGOREAN_EXPONENT: Final[float] = 2.37
FUMBLE_LOSS_RATE: Final[float] = 0.50
ONE_SCORE_MARGIN: Final[int] = 8
TURNOVER_DIFF_K: Final[float] = 16.0
CLOSE_GAME_K: Final[float] = 4.0
RED_ZONE_K_DRIVES: Final[float] = 20.0
LEAGUE_RED_ZONE_TD_RATE: Final[float] = 0.57
RIDGE_PLAYS_SCALE: Final[float] = 50.0
CONTINUITY_CUTOFF_MINUTES: Final[int] = 90
STARTER_DEPTH_RANK_MAX: Final[int] = 2


class TeamMetric(NamedTuple):
    """An exposure-normalized team metric `sum(num) / sum(den)` and its league default."""

    name: str
    num: str
    den: str
    default: float
    defensive: bool = True


TEAM_METRICS: Final[tuple[TeamMetric, ...]] = (
    TeamMetric("epa_per_play", "f_off_epa", "f_off_plays", 0.0),
    TeamMetric("pass_epa_per_dropback", "f_off_pass_epa", "f_off_dropbacks", 0.0),
    TeamMetric("rush_epa_per_rush", "f_off_rush_epa", "f_off_rushes", -0.07),
    TeamMetric("success_rate", "f_off_success", "f_off_plays", 0.45),
    TeamMetric("early_down_success_rate", "f_off_early_success", "f_off_early_plays", 0.46),
    TeamMetric("explosive_play_rate", "f_off_explosive", "f_off_plays", 0.09),
    TeamMetric("sack_rate", "off_sacks_taken", "off_dropbacks", 0.07),
    TeamMetric("turnover_rate", "off_turnovers", "off_plays", 0.017),
    TeamMetric("points_per_drive", "f_off_drive_points", "f_off_drives", 1.95),
    TeamMetric("penalty_yards_per_game", "penalty_yards", "_one", 50.0, defensive=False),
)

VARIANTS: Final[tuple[str, ...]] = ("std", *(f"roll{w}" for w in ROLLING_WINDOWS), "ewma")


class TeamFeatureConfig(BaseModel):
    """Tunable cold-start and rolling parameters (defaults match the module constants)."""

    model_config = ConfigDict(frozen=True)

    k_team_games: float = K_TEAM_GAMES
    ewma_alpha: float = EWMA_ALPHA
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS
    prior_weight_min: float = PRIOR_WEIGHT_MIN
    prior_weight_max: float = PRIOR_WEIGHT_MAX
    ridge_lambda: float = K_TEAM_GAMES


def metric_sides(metric: TeamMetric) -> tuple[str, ...]:
    """Return `('off', 'def')` for two-sided metrics and `('off',)` otherwise."""
    return ("off", "def") if metric.defensive else ("off",)


def team_feature_column_names() -> tuple[str, ...]:
    """Return the ordered per-team feature column names produced by the builder."""
    cols: list[str] = []
    for metric in TEAM_METRICS:
        for side in metric_sides(metric):
            cols.extend(f"{side}_{metric.name}_{variant}" for variant in VARIANTS)
    cols.extend(
        [
            "adj_off_epa",
            "adj_def_epa",
            "adj_epa_net",
            "turnover_diff_pg",
            "turnover_regression_signal",
            "fumble_luck_pg",
            "close_game_win_pct",
            "pythag_win_pct",
            "pythag_luck",
            "rz_td_rate",
            "games_played_pre",
            "roster_continuity",
            "prior_season_weight",
        ]
    )
    return tuple(cols)


# ---------------------------------------------------------------------------------------
# Roster continuity (returning production, head-coach change)
# ---------------------------------------------------------------------------------------


def prior_season_weight_from_continuity(
    continuity: float,
    *,
    coach_changed: bool = False,
    config: TeamFeatureConfig | None = None,
) -> float:
    """Map roster continuity `[0, 1]` (and a head-coach change) to the prior-season weight."""
    cfg = config or TeamFeatureConfig()
    c = min(max(float(continuity), 0.0), 1.0)
    weight = cfg.prior_weight_min + (cfg.prior_weight_max - cfg.prior_weight_min) * c
    return weight * (COACH_CHANGE_PRIOR_MULTIPLIER if coach_changed else 1.0)


def _team_game_rows(games_df: pl.DataFrame) -> pl.DataFrame:
    base = ["game_id", "event_seq", "season", "season_type", "week", "start_time_utc"]
    home = games_df.select(
        *base,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
    )
    away = games_df.select(
        *base,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
    )
    return pl.concat([home, away], how="vertical")


def build_roster_continuity(
    games_df: pl.DataFrame,
    player_logs: pl.DataFrame | None,
    availability: pl.DataFrame | None,
    *,
    head_coach_changed: pl.DataFrame | None = None,
    config: TeamFeatureConfig | None = None,
) -> pl.DataFrame:
    """Return `(game_id, team, roster_continuity, prior_season_weight, continuity_is_imputed)`.

    Continuity is the share of the team's *prior-season* offensive usage (snaps, else
    attempts + carries + targets) belonging to players on the latest depth-chart snapshot
    stamped at or before `kickoff - 90 min`. It therefore depends only on the prior season
    and pre-game information. Missing inputs fall back to `DEFAULT_ROSTER_CONTINUITY`.
    `head_coach_changed` (`team`, `season`, `head_coach_changed`) discounts the prior further.
    """
    cfg = config or TeamFeatureConfig()
    rows = _team_game_rows(games_df).select("game_id", "team", "season", "start_time_utc")
    coach: dict[tuple[str, int], bool] = {}
    if head_coach_changed is not None and not head_coach_changed.is_empty():
        coach = {
            (normalize_football_team(str(r["team"])), int(r["season"])): bool(
                r["head_coach_changed"]
            )
            for r in head_coach_changed.iter_rows(named=True)
        }

    prev_usage: dict[tuple[str, int], dict[int, float]] = {}
    if player_logs is not None and not player_logs.is_empty():
        usage = player_logs.select(
            "team",
            "season",
            "player_id",
            pl.coalesce(
                pl.col("snaps").cast(pl.Float64),
                pl.col("pass_attempts").fill_null(0).cast(pl.Float64)
                + pl.col("carries").fill_null(0).cast(pl.Float64)
                + pl.col("targets").fill_null(0).cast(pl.Float64),
            ).alias("usage"),
        )
        grouped = usage.group_by("team", "season", "player_id").agg(pl.col("usage").sum())
        for r in grouped.iter_rows(named=True):
            season_key = (str(r["team"]), int(r["season"]))
            prev_usage.setdefault(season_key, {})[int(r["player_id"])] = float(r["usage"])

    depth_by_game: dict[tuple[int, str], set[int]] = {}
    cutoffs = {
        (int(r["game_id"]), str(r["team"])): r["start_time_utc"]
        - timedelta(minutes=CONTINUITY_CUTOFF_MINUTES)
        for r in rows.iter_rows(named=True)
    }
    if availability is not None and not availability.is_empty():
        depth = availability.filter(
            (pl.col("source") == "depth_chart")
            & pl.col("depth_rank").is_not_null()
            & (pl.col("depth_rank") <= STARTER_DEPTH_RANK_MAX)
        )
        latest: dict[tuple[int, str], datetime] = {}
        for r in depth.select("game_id", "team", "as_of_time_utc").iter_rows(named=True):
            game_key = (int(r["game_id"]), str(r["team"]))
            stamp: datetime = r["as_of_time_utc"]
            cutoff = cutoffs.get(game_key)
            if cutoff is None or stamp > cutoff:
                continue
            if game_key not in latest or stamp > latest[game_key]:
                latest[game_key] = stamp
        for r in depth.iter_rows(named=True):
            game_key = (int(r["game_id"]), str(r["team"]))
            if game_key in latest and r["as_of_time_utc"] == latest[game_key]:
                depth_by_game.setdefault(game_key, set()).add(int(r["player_id"]))

    out: list[dict[str, object]] = []
    for r in rows.iter_rows(named=True):
        gid, team, season = int(r["game_id"]), str(r["team"]), int(r["season"])
        prev = prev_usage.get((team, season - 1))
        current = depth_by_game.get((gid, team))
        total = sum(prev.values()) if prev else 0.0
        if prev and current and total > 0.0:
            share = sum(u for p, u in prev.items() if p in current) / total
            imputed = False
        else:
            share, imputed = DEFAULT_ROSTER_CONTINUITY, True
        out.append(
            {
                "game_id": gid,
                "team": team,
                "roster_continuity": float(share),
                "prior_season_weight": prior_season_weight_from_continuity(
                    share, coach_changed=coach.get((team, season), False), config=cfg
                ),
                "continuity_is_imputed": imputed,
            }
        )
    schema = {
        "game_id": pl.Int64,
        "team": pl.Utf8,
        "roster_continuity": pl.Float64,
        "prior_season_weight": pl.Float64,
        "continuity_is_imputed": pl.Boolean,
    }
    return pl.DataFrame(out, schema=schema)


# ---------------------------------------------------------------------------------------
# Ridge strength-of-schedule adjustment
# ---------------------------------------------------------------------------------------


def _ridge_deviation(
    obs: Sequence[tuple[int, int, float, float]],
    n_teams: int,
    prior_off: np.ndarray,
    prior_def: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-shrink offense/defense strengths toward `(prior_off, prior_def)`.

    Model: `y = mu + off[team] + def[opp]` where `def` is EPA allowed (higher is worse).
    Solves `(X'WX + lam I) delta = X'W r` with `r = y - mu - prior terms`.
    """
    if not obs:
        return prior_off.copy(), prior_def.copy()
    idx_team = np.array([o[0] for o in obs])
    idx_opp = np.array([o[1] for o in obs])
    y = np.array([o[2] for o in obs], dtype=np.float64)
    w = np.array([o[3] for o in obs], dtype=np.float64)
    prior_term = prior_off[idx_team] + prior_def[idx_opp]
    mu = float(np.average(y - prior_term, weights=w))
    resid = y - mu - prior_term
    x = np.zeros((len(obs), 2 * n_teams), dtype=np.float64)
    x[np.arange(len(obs)), idx_team] = 1.0
    x[np.arange(len(obs)), n_teams + idx_opp] = 1.0
    xtw = x.T * w
    delta = np.linalg.solve(xtw @ x + lam * np.eye(2 * n_teams), xtw @ resid)
    return prior_off + delta[:n_teams], prior_def + delta[n_teams:]


def compute_adjusted_epa_ratings(
    long_df: pl.DataFrame,
    prior_weights: Mapping[tuple[str, int, int], float],
    *,
    config: TeamFeatureConfig | None = None,
    default_prior_weight: float = 0.65,
) -> pl.DataFrame:
    """Return `(game_id, team, adj_off_epa, adj_def_epa)` using only earlier-week games.

    For a game in `(season, week)` the ratings are ridge-fit on that season's completed games
    with `week' < week`, shrunk toward the prior-season ratings (themselves ridge-fit on the
    whole previous season) scaled by the team's prior-season weight.
    """
    cfg = config or TeamFeatureConfig()
    teams = sorted(set(long_df["team"].to_list()))
    t_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    zeros = np.zeros(n_teams)

    completed = long_df.filter(pl.col("_played")).select(
        "season", "week", "team", "opponent", "_epa_y", "_epa_w"
    )
    by_season: dict[int, list[tuple[int, int, int, float, float]]] = {}
    for r in completed.iter_rows(named=True):
        by_season.setdefault(int(r["season"]), []).append(
            (
                int(r["week"]),
                t_idx[str(r["team"])],
                t_idx[str(r["opponent"])],
                float(r["_epa_y"]),
                float(r["_epa_w"]),
            )
        )

    season_final: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for season, items in by_season.items():
        obs = [(a, b, y, w) for _, a, b, y, w in items]
        season_final[season] = _ridge_deviation(obs, n_teams, zeros, zeros, cfg.ridge_lambda)

    targets = long_df.select("game_id", "team", "season", "week").unique()
    out: list[dict[str, object]] = []
    for (season, week), grp in targets.group_by(["season", "week"], maintain_order=True):
        s, w = int(season), int(week)
        prev_off, prev_def = season_final.get(s - 1, (zeros, zeros))
        pw = np.array(
            [prior_weights.get((t, s, w), default_prior_weight) for t in teams], dtype=np.float64
        )
        obs = [(a, b, y, wt) for wk, a, b, y, wt in by_season.get(s, []) if wk < w]
        off, dfn = _ridge_deviation(obs, n_teams, pw * prev_off, pw * prev_def, cfg.ridge_lambda)
        for r in grp.iter_rows(named=True):
            i = t_idx[str(r["team"])]
            out.append(
                {
                    "game_id": int(r["game_id"]),
                    "team": str(r["team"]),
                    "adj_off_epa": float(off[i]),
                    "adj_def_epa": float(dfn[i]),
                }
            )
    return pl.DataFrame(
        out,
        schema={
            "game_id": pl.Int64,
            "team": pl.Utf8,
            "adj_off_epa": pl.Float64,
            "adj_def_epa": pl.Float64,
        },
    )


# ---------------------------------------------------------------------------------------
# Rolling team features
# ---------------------------------------------------------------------------------------

_LOG_COLUMNS: Final[tuple[str, ...]] = (
    "points_for",
    "points_against",
    "penalty_yards",
    "off_plays",
    "off_dropbacks",
    "off_sacks_taken",
    "off_turnovers",
    "off_fumbles",
    "off_fumbles_lost",
    "off_rz_drives",
    "off_rz_td_drives",
    "f_off_plays",
    "f_off_epa",
    "f_off_dropbacks",
    "f_off_pass_epa",
    "f_off_rushes",
    "f_off_rush_epa",
    "f_off_success",
    "f_off_early_plays",
    "f_off_early_success",
    "f_off_explosive",
    "f_off_drives",
    "f_off_drive_points",
)


def _assemble_long_frame(games_df: pl.DataFrame, team_logs: pl.DataFrame) -> pl.DataFrame:
    """Long `(game, team)` frame (incl. future games) with own `s_*` and opponent `o_*` stats."""
    missing = [c for c in _LOG_COLUMNS if c not in team_logs.columns]
    if missing:
        raise ValueError(f"Team logs are missing required columns: {missing}.")
    long = _team_game_rows(games_df)
    cols = list(_LOG_COLUMNS)
    stats = team_logs.select("game_id", "team", *cols).with_columns(
        pl.lit(True).alias("_played"), pl.lit(1.0).alias("_one")
    )
    own = stats.rename({c: f"s_{c}" for c in [*cols, "_played", "_one"]})
    opp = stats.rename({c: f"o_{c}" for c in [*cols, "_played", "_one"]}).rename(
        {"team": "opponent"}
    )
    out = (
        long.join(own, on=["game_id", "team"], how="left")
        .join(opp, on=["game_id", "opponent"], how="left")
        .with_columns(pl.col("s_f_off_plays").is_not_null().alias("_played"))
        .sort("start_time_utc", "event_seq", "game_id", "team")
    )
    return out


def _num_den_exprs() -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for m in TEAM_METRICS:
        own_num = f"s_{m.num}"
        own_den = f"s_{m.den}" if m.den != "_one" else "s__one"
        opp_num = f"o_{m.num}"
        opp_den = f"o_{m.den}" if m.den != "_one" else "o__one"
        exprs.append(pl.col(own_num).cast(pl.Float64).alias(f"off_{m.name}_num"))
        exprs.append(pl.col(own_den).cast(pl.Float64).alias(f"off_{m.name}_den"))
        if m.defensive:
            exprs.append(pl.col(opp_num).cast(pl.Float64).alias(f"def_{m.name}_num"))
            exprs.append(pl.col(opp_den).cast(pl.Float64).alias(f"def_{m.name}_den"))
    return exprs


def _safe_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return num / pl.when(den > 0.0).then(den).otherwise(None)


def build_team_rolling_features(
    games_df: pl.DataFrame,
    team_logs: pl.DataFrame,
    *,
    roster_continuity: pl.DataFrame | None = None,
    config: TeamFeatureConfig | None = None,
) -> pl.DataFrame:
    """Build per-`(game_id, team)` pre-game team features for every game (incl. upcoming).

    `games_df` supplies the schedule (completed and `FUT` games); `team_logs` the completed
    Silver team-game logs. Output columns are `game_id, event_seq, team` plus
    `team_feature_column_names()`.
    """
    cfg = config or TeamFeatureConfig()
    long = _assemble_long_frame(games_df, team_logs)
    group = ["team", "season"]
    default_w = cfg.prior_weight_min + 0.6 * (cfg.prior_weight_max - cfg.prior_weight_min)

    if roster_continuity is not None and not roster_continuity.is_empty():
        long = long.join(
            roster_continuity.select("game_id", "team", "roster_continuity", "prior_season_weight"),
            on=["game_id", "team"],
            how="left",
        )
    else:
        long = long.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("roster_continuity"),
            pl.lit(None, dtype=pl.Float64).alias("prior_season_weight"),
        )
    long = long.with_columns(
        pl.col("roster_continuity").fill_null(DEFAULT_ROSTER_CONTINUITY),
        pl.col("prior_season_weight").fill_null(default_w),
    )

    long = long.with_columns(*_num_den_exprs())
    nd_cols = [c for c in long.columns if c.endswith(("_num", "_den"))]

    # Extra to-date counters for luck features
    long = long.with_columns(
        (pl.col("s_points_for") - pl.col("s_points_against")).alias("_pd"),
        (pl.col("o_off_turnovers") - pl.col("s_off_turnovers")).alias("_to_diff"),
    ).with_columns(
        (pl.col("_pd").abs() <= ONE_SCORE_MARGIN).cast(pl.Float64).alias("_close"),
        pl.when(pl.col("_pd") > 0)
        .then(1.0)
        .when(pl.col("_pd") == 0)
        .then(0.5)
        .otherwise(0.0)
        .alias("_win"),
    )
    long = long.with_columns(
        pl.when(pl.col("_pd").is_not_null()).then(pl.col("_close")).alias("_close"),
        pl.when(pl.col("_pd").is_not_null()).then(pl.col("_win")).alias("_win"),
    )
    long = long.with_columns(
        (pl.col("_close") * pl.col("_win")).alias("_close_win"),
        (pl.col("s_points_for").cast(pl.Float64)).alias("_pf"),
        (pl.col("s_points_against").cast(pl.Float64)).alias("_pa"),
        pl.col("s_off_fumbles").cast(pl.Float64).alias("_fum"),
        pl.col("s_off_fumbles_lost").cast(pl.Float64).alias("_fum_lost"),
        pl.col("s_off_rz_drives").cast(pl.Float64).alias("_rz"),
        pl.col("s_off_rz_td_drives").cast(pl.Float64).alias("_rz_td"),
        pl.col("_played").cast(pl.Float64).alias("_played_f"),
    )
    extra_cols = [
        "_close",
        "_win",
        "_close_win",
        "_pf",
        "_pa",
        "_fum",
        "_fum_lost",
        "_rz",
        "_rz_td",
        "_to_diff",
        "_played_f",
    ]

    def to_date(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .cast(pl.Float64)
            .fill_null(0.0)
            .cum_sum()
            .shift(1)
            .over(group)
            .fill_null(0.0)
            .alias(f"{col}_cum")
        )

    long = long.with_columns(
        *[to_date(c) for c in [*nd_cols, *extra_cols]],
        *[
            _safe_ratio(pl.col(f"{p}_num"), pl.col(f"{p}_den")).alias(f"{p}_pg")
            for p in sorted({c.rsplit("_", 1)[0] for c in nd_cols})
        ],
    ).with_columns(pl.col("_played_f_cum").alias("games_played_pre"))

    # Prior-season (team) and previous-league aggregates joined on season + 1
    played = long.filter(pl.col("_played"))
    agg_cols = [*nd_cols, *extra_cols]
    team_prev = (
        played.group_by("team", "season")
        .agg([pl.col(c).cast(pl.Float64).sum().alias(f"pt_{c}") for c in agg_cols])
        .with_columns((pl.col("season") + 1).alias("season"))
    )
    league_prev = (
        played.group_by("season")
        .agg([pl.col(c).cast(pl.Float64).sum().alias(f"pl_{c}") for c in agg_cols])
        .with_columns((pl.col("season") + 1).alias("season"))
    )
    long = long.join(team_prev, on=["team", "season"], how="left").join(
        league_prev, on="season", how="left"
    )

    w = pl.col("prior_season_weight")
    n = pl.col("games_played_pre")

    def prior_expr(prefix: str, default: float) -> pl.Expr:
        team_rate = _safe_ratio(pl.col(f"pt_{prefix}_num"), pl.col(f"pt_{prefix}_den"))
        league_rate = pl.coalesce(
            _safe_ratio(pl.col(f"pl_{prefix}_num"), pl.col(f"pl_{prefix}_den")),
            pl.lit(default),
        )
        return (
            pl.when(team_rate.is_not_null())
            .then(w * team_rate + (1.0 - w) * league_rate)
            .otherwise(league_rate)
        )

    feature_exprs: list[pl.Expr] = []
    for m in TEAM_METRICS:
        for side in metric_sides(m):
            p = f"{side}_{m.name}"
            prior = prior_expr(p, m.default)
            std_rate = _safe_ratio(pl.col(f"{p}_num_cum"), pl.col(f"{p}_den_cum"))
            feature_exprs.append(
                bayesian_shrinkage(std_rate, n, prior, cfg.k_team_games).alias(f"{p}_std")
            )
            for window in cfg.rolling_windows:
                roll = pregame_exposure_rate(f"{p}_num", f"{p}_den", window, group)
                n_w = pl.min_horizontal(n, pl.lit(float(window)))
                feature_exprs.append(
                    bayesian_shrinkage(roll, n_w, prior, cfg.k_team_games).alias(
                        f"{p}_roll{window}"
                    )
                )
            feature_exprs.append(
                bayesian_shrinkage(
                    pregame_ewma(f"{p}_pg", cfg.ewma_alpha, group), n, prior, cfg.k_team_games
                ).alias(f"{p}_ewma")
            )
    long = long.with_columns(*feature_exprs)

    # Luck / regression signals
    games_td = pl.col("_played_f_cum")
    to_rate = _safe_ratio(pl.col("_to_diff_cum"), games_td)
    pyth_num = pl.col("_pf_cum") ** PYTHAGOREAN_EXPONENT
    pyth_den = pyth_num + pl.col("_pa_cum") ** PYTHAGOREAN_EXPONENT
    pyth_now = pl.when(pyth_den > 0.0).then(pyth_num / pyth_den).otherwise(0.5)
    win_pct = pl.when(games_td > 0).then(pl.col("_win_cum") / games_td).otherwise(0.5)
    prev_pyth_num = pl.col("pt__pf") ** PYTHAGOREAN_EXPONENT
    prev_pyth_den = prev_pyth_num + pl.col("pt__pa") ** PYTHAGOREAN_EXPONENT
    prev_pyth = pl.when(prev_pyth_den > 0.0).then(prev_pyth_num / prev_pyth_den).otherwise(None)
    pyth_prior = pl.coalesce(w * prev_pyth + (1.0 - w) * 0.5, pl.lit(0.5))
    rz_prev = _safe_ratio(pl.col("pt__rz_td"), pl.col("pt__rz"))
    rz_prior = pl.coalesce(
        w * rz_prev + (1.0 - w) * LEAGUE_RED_ZONE_TD_RATE, pl.lit(LEAGUE_RED_ZONE_TD_RATE)
    )
    long = long.with_columns(
        bayesian_shrinkage(to_rate, games_td, 0.0, TURNOVER_DIFF_K).alias("_to_shrunk"),
        to_rate.fill_null(0.0).alias("turnover_diff_pg"),
        (
            (FUMBLE_LOSS_RATE * pl.col("_fum_cum") - pl.col("_fum_lost_cum"))
            / pl.max_horizontal(games_td, pl.lit(1.0))
        ).alias("fumble_luck_pg"),
        bayesian_shrinkage(
            _safe_ratio(pl.col("_close_win_cum"), pl.col("_close_cum")),
            pl.col("_close_cum"),
            0.5,
            CLOSE_GAME_K,
        ).alias("close_game_win_pct"),
        bayesian_shrinkage(pyth_now, games_td, pyth_prior, cfg.k_team_games).alias(
            "pythag_win_pct"
        ),
        bayesian_shrinkage(win_pct - pyth_now, games_td, 0.0, cfg.k_team_games).alias(
            "pythag_luck"
        ),
        bayesian_shrinkage(
            _safe_ratio(pl.col("_rz_td_cum"), pl.col("_rz_cum")),
            pl.col("_rz_cum"),
            rz_prior,
            RED_ZONE_K_DRIVES,
        ).alias("rz_td_rate"),
    ).with_columns(
        (pl.col("_to_shrunk") - pl.col("turnover_diff_pg")).alias("turnover_regression_signal")
    )

    # Ridge strength-of-schedule (earlier-week games only)
    epa_pg = pl.col("s_f_off_epa") / pl.when(pl.col("s_f_off_plays") > 0).then(
        pl.col("s_f_off_plays")
    )
    ridge_in = long.with_columns(
        epa_pg.alias("_epa_y"),
        (pl.col("s_f_off_plays").cast(pl.Float64) / RIDGE_PLAYS_SCALE).alias("_epa_w"),
    ).with_columns(
        (pl.col("_played") & pl.col("_epa_y").is_not_null() & (pl.col("_epa_w") > 0)).alias(
            "_played"
        )
    )
    pw_lookup = {
        (str(r["team"]), int(r["season"]), int(r["week"])): float(r["prior_season_weight"])
        for r in long.select("team", "season", "week", "prior_season_weight").iter_rows(named=True)
    }
    adj = compute_adjusted_epa_ratings(
        ridge_in,
        pw_lookup,
        config=cfg,
        default_prior_weight=default_w,
    )
    long = (
        long.join(adj, on=["game_id", "team"], how="left")
        .with_columns(
            pl.col("adj_off_epa").fill_null(0.0),
            pl.col("adj_def_epa").fill_null(0.0),
        )
        .with_columns((pl.col("adj_off_epa") - pl.col("adj_def_epa")).alias("adj_epa_net"))
    )

    keep = ["game_id", "event_seq", "team", *team_feature_column_names()]
    return long.select(keep).sort("game_id", "team")
