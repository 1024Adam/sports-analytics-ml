"""NFL Stage 2: QB / RB / WR-TE projections, anytime-TD probabilities (`models/stage2_player.py`).

Structural, leak-free model conditioned on Stage 1:

1. **Team Opportunity Model**: positive-coefficient ridge regressions project team pass attempts
   and rush attempts from Stage 1 expected margin (game script: trailing teams pass more),
   expected total, pace, and pass rate (a pass-rate-over-expected proxy). Coefficient signs are
   constrained so a bigger lead never raises passing volume or lowers rushing volume.
2. **Player opportunity**: team volume x shrunk role share x `p_active`.
3. **Per-opportunity rates** (already shrunk in the Gold features) are adjusted by the opposing
   defense index and, for touchdowns, by team expected points and red-zone / goal-line usage.
4. **Cross-stage reconciler** (`reconcile_football_stage2_projections`): opportunity
   conservation, then soft-scaling (`gamma = 0.5`) of receiving yards and touchdowns toward team
   totals implied by Stage 1.
5. Threshold probabilities use zero-inflated gamma (yards) / Poisson (counts) distributions with
   dispersion estimated from training residuals; `P(anytime TD) = p_active * (1 - exp(-lambda))`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import gamma as gamma_dist
from scipy.stats import poisson
from sklearn.linear_model import Ridge

from sports_ml.config import RANDOM_SEED
from sports_ml.sports.football.constants import POSITION_PRIORS
from sports_ml.sports.football.features.player_columns import (
    STAGE2_PLAYER_FEATURE_COLUMNS,
    STAGE2_TARGET_COLUMNS,
)

LEAGUE_MEAN_POINTS: Final[float] = 22.5
TD_POINT_FRACTION: Final[float] = 0.78
POINTS_PER_TD: Final[float] = 6.95
TEAM_TD_COVERAGE: Final[float] = 0.95
DEFENSE_SENSITIVITY: Final[float] = 0.5
GAMMA_RECONCILE: Final[float] = 0.50
SCRIPT_FLOOR_PASS: Final[float] = 0.10
SCRIPT_FLOOR_RUSH: Final[float] = 0.08
MAX_REDISTRIBUTION: Final[float] = 1.6
SCALE_BOUNDS: Final[tuple[float, float]] = (0.6, 1.6)
TD_EXPOSURE_POWER: Final[float] = 1.0
DEFAULT_CV: Final[dict[str, float]] = {"pass_yds": 0.30, "rush_yds": 0.85, "rec_yds": 0.90}
CV_BOUNDS: Final[tuple[float, float]] = (0.2, 1.4)

PROJECTION_COLUMNS: Final[tuple[str, ...]] = (
    "proj_pass_att",
    "proj_completions",
    "proj_pass_yds",
    "proj_pass_td",
    "proj_int",
    "proj_carries",
    "proj_rush_yds",
    "proj_rush_td",
    "proj_targets",
    "proj_rec",
    "proj_rec_yds",
    "proj_rec_td",
)
PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "prob_250plus_pass_yds",
    "prob_300plus_pass_yds",
    "prob_2plus_pass_td",
    "prob_75plus_rush_yds",
    "prob_100plus_rush_yds",
    "prob_5plus_rec",
    "prob_75plus_rec_yds",
    "prob_100plus_rec_yds",
    "prob_anytime_td",
)
_RATE_COLUMNS: Final[tuple[str, ...]] = (
    "r_comp",
    "r_ypa",
    "r_td_att",
    "r_int_att",
    "r_ypc",
    "r_td_carry",
    "r_catch",
    "r_ypt",
    "r_td_tgt",
)
_ACTUAL_TO_PROJ: Final[dict[str, str]] = {
    "actual_pass_att": "proj_pass_att",
    "actual_completions": "proj_completions",
    "actual_pass_yds": "proj_pass_yds",
    "actual_pass_td": "proj_pass_td",
    "actual_int": "proj_int",
    "actual_carries": "proj_carries",
    "actual_rush_yds": "proj_rush_yds",
    "actual_rush_td": "proj_rush_td",
    "actual_targets": "proj_targets",
    "actual_rec": "proj_rec",
    "actual_rec_yds": "proj_rec_yds",
    "actual_rec_td": "proj_rec_td",
}


def _f64(df: pl.DataFrame, col: str, fill: float = 0.0) -> NDArray[np.float64]:
    return np.asarray(df[col].cast(pl.Float64).fill_null(fill).to_numpy(), dtype=np.float64)


def team_context_from_stage1(stage1_preds: pl.DataFrame) -> pl.DataFrame:
    """Return `(game_id, team, team_mu, opp_mu, margin_own, total)` from Stage 1 predictions."""
    home = stage1_preds.select(
        "game_id",
        pl.col("home_team").alias("team"),
        pl.col("expected_home_score").alias("team_mu"),
        pl.col("expected_away_score").alias("opp_mu"),
    )
    away = stage1_preds.select(
        "game_id",
        pl.col("away_team").alias("team"),
        pl.col("expected_away_score").alias("team_mu"),
        pl.col("expected_home_score").alias("opp_mu"),
    )
    return pl.concat([home, away]).with_columns(
        (pl.col("team_mu") - pl.col("opp_mu")).alias("margin_own"),
        (pl.col("team_mu") + pl.col("opp_mu")).alias("total"),
    )


def _gamma_sf(threshold: float, mean: NDArray[np.float64], cv: float) -> NDArray[np.float64]:
    out = np.zeros_like(mean)
    ok = mean > 1e-9
    if ok.any():
        shape = 1.0 / cv**2
        out[ok] = gamma_dist.sf(threshold, a=shape, scale=mean[ok] * cv**2)
    return out


def _zero_inflated(p: NDArray[np.float64], full: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.clip(p * full, 0.0, 1.0), dtype=np.float64)


def _safe_div(num: NDArray[np.float64], den: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)


def add_threshold_probabilities(
    proj: pl.DataFrame, cv: dict[str, float] | None = None
) -> pl.DataFrame:
    """Append zero-inflated threshold probabilities and `prob_anytime_td` to projections."""
    cvs = {**DEFAULT_CV, **(cv or {})}
    p = np.clip(_f64(proj, "p_active", 1.0), 0.0, 1.0)
    active = p > 1e-12

    def full(col: str) -> NDArray[np.float64]:
        return _safe_div(_f64(proj, col), p) * active

    pass_yds, rush_yds, rec_yds = full("proj_pass_yds"), full("proj_rush_yds"), full("proj_rec_yds")
    pass_td, rec = full("proj_pass_td"), full("proj_rec")
    rush_td, rec_td = full("proj_rush_td"), full("proj_rec_td")
    pass_sf_250 = _gamma_sf(250.0, pass_yds, cvs["pass_yds"])
    pass_sf_300 = np.minimum(_gamma_sf(300.0, pass_yds, cvs["pass_yds"]), pass_sf_250)
    rush_sf_75 = _gamma_sf(75.0, rush_yds, cvs["rush_yds"])
    rush_sf_100 = np.minimum(_gamma_sf(100.0, rush_yds, cvs["rush_yds"]), rush_sf_75)
    rec_sf_75 = _gamma_sf(75.0, rec_yds, cvs["rec_yds"])
    rec_sf_100 = np.minimum(_gamma_sf(100.0, rec_yds, cvs["rec_yds"]), rec_sf_75)
    lam_td = rush_td + rec_td
    return proj.with_columns(
        pl.Series("prob_250plus_pass_yds", _zero_inflated(p, pass_sf_250)),
        pl.Series("prob_300plus_pass_yds", _zero_inflated(p, pass_sf_300)),
        pl.Series("prob_2plus_pass_td", _zero_inflated(p, poisson.sf(1, np.maximum(pass_td, 0.0)))),
        pl.Series("prob_75plus_rush_yds", _zero_inflated(p, rush_sf_75)),
        pl.Series("prob_100plus_rush_yds", _zero_inflated(p, rush_sf_100)),
        pl.Series("prob_5plus_rec", _zero_inflated(p, poisson.sf(4, np.maximum(rec, 0.0)))),
        pl.Series("prob_75plus_rec_yds", _zero_inflated(p, rec_sf_75)),
        pl.Series("prob_100plus_rec_yds", _zero_inflated(p, rec_sf_100)),
        pl.Series("prob_anytime_td", _zero_inflated(p, 1.0 - np.exp(-np.maximum(lam_td, 0.0)))),
    )


def _scale(target: float, total: float, gamma: float) -> float:
    if total <= 1e-12 or target <= 0.0:
        return 1.0
    raw = (1.0 - gamma) + gamma * target / total
    return float(np.clip(raw, *SCALE_BOUNDS))


def _conserve(
    frame: pl.DataFrame,
    share_col: str,
    opp_col: str,
    mask: pl.Expr,
    pool: str,
    *,
    full_coverage: bool = False,
) -> pl.DataFrame:
    """Rescale `opp_col` within `(game_id, team)` groups so the pool sums to `pool x coverage`.

    `coverage = min(1, sum of shares)`; players with zero raw opportunity stay at zero, and the
    redistribution factor is capped at `MAX_REDISTRIBUTION`.
    """
    grouped = (
        frame.filter(mask)
        .group_by("game_id", "team")
        .agg(
            pl.col(opp_col).sum().alias("_raw_sum"),
            pl.col(share_col).sum().alias("_share_sum"),
            pl.col(pool).first().alias("_pool"),
        )
        .with_columns(
            (
                pl.col("_pool")
                if full_coverage
                else pl.col("_pool") * pl.min_horizontal(pl.col("_share_sum"), pl.lit(1.0))
            ).alias("_target")
        )
        .with_columns(
            pl.when(pl.col("_raw_sum") > 1e-12)
            .then(
                pl.min_horizontal(
                    pl.col("_target") / pl.col("_raw_sum"), pl.lit(MAX_REDISTRIBUTION)
                )
            )
            .otherwise(1.0)
            .alias("_factor")
        )
        .select("game_id", "team", "_factor")
    )
    joined = frame.join(grouped, on=["game_id", "team"], how="left").with_columns(
        pl.col("_factor").fill_null(1.0)
    )
    return joined.with_columns(
        pl.when(mask)
        .then(pl.col(opp_col) * pl.col("_factor"))
        .otherwise(pl.col(opp_col))
        .alias(opp_col)
    ).drop("_factor")


def reconcile_football_stage2_projections(
    raw: pl.DataFrame,
    team_context: pl.DataFrame,
    *,
    gamma: float = GAMMA_RECONCILE,
) -> pl.DataFrame:
    """Cross-stage coherence: conserve opportunities, then soft-scale yards/TDs to team totals.

    `raw` needs `game_id, team, position, p_active`, shares (`pass_att_share`, `carry_share`,
    `target_share`, `rz`/`gl` multipliers), raw opportunities (`proj_pass_att`, `proj_carries`,
    `proj_targets`), and per-opportunity rate columns (`r_*`). `team_context` supplies
    `game_id, team, team_mu, proj_pass_att_team, proj_rush_att_team`.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}.")
    ctx_cols = ["team_mu", "team_pass_att", "team_rush_att"]
    df = (
        raw.drop([c for c in ctx_cols if c in raw.columns])
        .join(
            team_context.select("game_id", "team", *ctx_cols),
            on=["game_id", "team"],
            how="left",
        )
        .with_columns(
            pl.col("team_mu").fill_null(LEAGUE_MEAN_POINTS),
            pl.col("team_pass_att").fill_null(0.0),
            pl.col("team_rush_att").fill_null(0.0),
        )
    )
    is_qb = pl.col("position") == "QB"
    df = _conserve(
        df, "pass_att_share", "proj_pass_att", is_qb, "team_pass_att", full_coverage=True
    )
    df = _conserve(
        df, "carry_share", "proj_carries", pl.col("position").is_in(["QB", "RB"]), "team_rush_att"
    )
    df = _conserve(
        df,
        "target_share",
        "proj_targets",
        pl.col("position").is_in(["RB", "WR", "TE"]),
        "team_pass_att",
    )

    df = df.with_columns(
        (pl.col("proj_pass_att") * pl.col("r_comp")).alias("proj_completions"),
        (pl.col("proj_pass_att") * pl.col("r_ypa")).alias("proj_pass_yds"),
        (pl.col("proj_pass_att") * pl.col("r_td_att")).alias("proj_pass_td"),
        (pl.col("proj_pass_att") * pl.col("r_int_att")).alias("proj_int"),
        (pl.col("proj_carries") * pl.col("r_ypc")).alias("proj_rush_yds"),
        (pl.col("proj_carries") * pl.col("r_td_carry")).alias("proj_rush_td"),
        (pl.col("proj_targets") * pl.col("r_catch")).alias("proj_rec"),
        (pl.col("proj_targets") * pl.col("r_ypt")).alias("proj_rec_yds"),
        (pl.col("proj_targets") * pl.col("r_td_tgt")).alias("proj_rec_td"),
    )

    # --- soft-scale receiving yards and TDs toward Stage 1 team totals (per game-team)
    team = (
        df.group_by("game_id", "team")
        .agg(
            pl.col("team_mu").first().alias("_mu"),
            pl.col("proj_pass_yds").sum().alias("_pass_yds"),
            pl.col("proj_rec_yds").sum().alias("_rec_yds"),
            (pl.col("proj_rush_td").sum() + pl.col("proj_rec_td").sum()).alias("_td"),
            pl.col("proj_pass_td").sum().alias("_pass_td"),
            pl.col("proj_rec_td").sum().alias("_rec_td"),
            pl.col("target_share")
            .filter(pl.col("position").is_in(["RB", "WR", "TE"]))
            .sum()
            .alias("_cov"),
        )
        .with_columns(
            pl.min_horizontal(pl.col("_cov"), pl.lit(1.0)).alias("_cov"),
        )
    )
    f_rec: list[float] = []
    f_td: list[float] = []
    for r in team.iter_rows(named=True):
        td_target = r["_mu"] * TD_POINT_FRACTION / POINTS_PER_TD * TEAM_TD_COVERAGE
        f_rec.append(_scale(r["_pass_yds"] * r["_cov"], r["_rec_yds"], gamma))
        f_td.append(_scale(td_target, r["_td"], gamma))
    team = team.with_columns(
        pl.Series("_f_rec", f_rec),
        pl.Series("_f_td", f_td),
    )
    df = df.join(team.select("game_id", "team", "_f_rec", "_f_td", "_cov"), on=["game_id", "team"])
    df = df.with_columns(
        (pl.col("proj_rec_yds") * pl.col("_f_rec")).alias("proj_rec_yds"),
        (pl.col("proj_rush_td") * pl.col("_f_td")).alias("proj_rush_td"),
        (pl.col("proj_rec_td") * pl.col("_f_td")).alias("proj_rec_td"),
    )
    # Passing TDs track the (scaled) receiving TDs the team's receivers are projected to score
    rec_td_team = df.group_by("game_id", "team").agg(pl.col("proj_rec_td").sum().alias("_rec_td_s"))
    qb_td = (
        df.filter(is_qb)
        .group_by("game_id", "team")
        .agg(pl.col("proj_pass_td").sum().alias("_qb_td"))
    )
    df = df.join(rec_td_team, on=["game_id", "team"]).join(
        qb_td, on=["game_id", "team"], how="left"
    )
    df = df.with_columns(
        pl.when(is_qb & (pl.col("_qb_td") > 1e-12))
        .then(
            (1.0 - gamma) * pl.col("proj_pass_td")
            + gamma
            * pl.col("_rec_td_s")
            * pl.col("_cov")
            * pl.col("proj_pass_td")
            / pl.col("_qb_td")
        )
        .otherwise(pl.col("proj_pass_td"))
        .alias("proj_pass_td")
    )
    drop = [c for c in df.columns if c.startswith("_")]
    return df.drop(drop)


@dataclass
class Stage2FootballPlayerModel:
    """Optional NFL Stage 2 player-projection model (conditioned on Stage 1 expected scores)."""

    gamma: float = GAMMA_RECONCILE
    ridge_alpha: float = 5.0
    random_seed: int = RANDOM_SEED
    feature_columns: list[str] = field(default_factory=list)
    league_mu: float = LEAGUE_MEAN_POINTS
    cv: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CV))
    _pass_model: Ridge | None = None
    _rush_model: Ridge | None = None
    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ----------------------------------------------------------- team opportunity

    @staticmethod
    def _opp_features(ctx: pl.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        margin = _f64(ctx, "margin_own")
        total = _f64(ctx, "total")
        pace = _f64(ctx, "team_pace_plays_pg", 63.0)
        pass_rate = _f64(ctx, "team_pass_rate", 0.58)
        x_pass = np.column_stack([-margin, total, pace * pass_rate])
        x_rush = np.column_stack([margin, total, pace * (1.0 - pass_rate)])
        return x_pass, x_rush

    def _team_frame(self, rows: pl.DataFrame, stage1_preds: pl.DataFrame) -> pl.DataFrame:
        ctx = team_context_from_stage1(stage1_preds)
        team_style = rows.group_by("game_id", "team").agg(
            pl.col("team_pace_plays_pg").first(), pl.col("team_pass_rate").first()
        )
        return team_style.join(ctx, on=["game_id", "team"], how="left").with_columns(
            pl.col("team_mu").fill_null(self.league_mu),
            pl.col("opp_mu").fill_null(self.league_mu),
            pl.col("margin_own").fill_null(0.0),
            pl.col("total").fill_null(2.0 * self.league_mu),
        )

    def fit(self, stage2_df: pl.DataFrame, stage1_preds: pl.DataFrame) -> Stage2FootballPlayerModel:
        """Fit the opportunity ridge models and dispersion estimates on completed player rows."""
        train = stage2_df.filter(pl.col("actual_targets").is_not_null())
        if train.is_empty():
            raise ValueError("No completed player rows available to fit Stage 2.")
        self.feature_columns = [c for c in STAGE2_PLAYER_FEATURE_COLUMNS if c in stage2_df.columns]
        ctx_all = team_context_from_stage1(stage1_preds)
        self.league_mu = float(ctx_all["team_mu"].mean()) if ctx_all.height else LEAGUE_MEAN_POINTS  # type: ignore[arg-type]

        team_actual = train.group_by("game_id", "team").agg(
            pl.col("actual_pass_att").sum().alias("y_pass"),
            pl.col("actual_carries").sum().alias("y_rush"),
        )
        team = self._team_frame(train, stage1_preds).join(team_actual, on=["game_id", "team"])
        team = team.filter((pl.col("y_pass") > 0) & (pl.col("y_rush") > 0))
        if team.height < 10:
            raise ValueError("Not enough team-games with opportunities to fit Stage 2.")
        x_pass, x_rush = self._opp_features(team)
        # Game-script prior: each point of expected margin shifts >= 0.10 attempts to the run
        # (or away from the pass); the ridge learns the residual with non-negative coefficients.
        self._pass_model = Ridge(alpha=self.ridge_alpha, positive=True, solver="lbfgs").fit(
            x_pass, _f64(team, "y_pass") - SCRIPT_FLOOR_PASS * x_pass[:, 0]
        )
        self._rush_model = Ridge(alpha=self.ridge_alpha, positive=True, solver="lbfgs").fit(
            x_rush, _f64(team, "y_rush") - SCRIPT_FLOOR_RUSH * x_rush[:, 0]
        )
        self._is_fitted = True

        # Dispersion (coefficient of variation) from training residuals
        proj = self.predict(train, stage1_preds, with_probabilities=False)
        merged = proj.join(
            train.select("game_id", "team", "player_id", *_ACTUAL_TO_PROJ),
            on=["game_id", "team", "player_id"],
        )
        for key, actual_col, proj_col, pos in (
            ("pass_yds", "actual_pass_yds", "proj_pass_yds", ["QB"]),
            ("rush_yds", "actual_rush_yds", "proj_rush_yds", ["RB"]),
            ("rec_yds", "actual_rec_yds", "proj_rec_yds", ["WR", "TE"]),
        ):
            sub = merged.filter(pl.col("position").is_in(pos) & (pl.col(proj_col) > 1.0))
            if sub.height >= 30:
                mean = float(sub[proj_col].mean())  # type: ignore[arg-type]
                resid = _f64(sub, actual_col) - _f64(sub, proj_col)
                self.cv[key] = float(np.clip(np.std(resid) / max(mean, 1e-9), *CV_BOUNDS))
        return self

    def project_team_opportunity(
        self, rows: pl.DataFrame, stage1_preds: pl.DataFrame
    ) -> pl.DataFrame:
        """Return `(game_id, team, team_mu, team_plays, team_pass_att, team_rush_att)`."""
        if not self._is_fitted or self._pass_model is None or self._rush_model is None:
            raise RuntimeError("Stage2FootballPlayerModel must be fitted first.")
        team = self._team_frame(rows, stage1_preds)
        x_pass, x_rush = self._opp_features(team)
        pass_att = np.clip(
            self._pass_model.predict(x_pass) + SCRIPT_FLOOR_PASS * x_pass[:, 0], 0.0, None
        )
        rush_att = np.clip(
            self._rush_model.predict(x_rush) + SCRIPT_FLOOR_RUSH * x_rush[:, 0], 0.0, None
        )
        return team.select(
            "game_id", "team", "team_mu", "opp_mu", "margin_own", "total"
        ).with_columns(
            pl.Series("team_pass_att", pass_att),
            pl.Series("team_rush_att", rush_att),
            pl.Series("team_plays", pass_att + rush_att),
        )

    # ---------------------------------------------------------------- projection

    def _raw_rates(self, rows: pl.DataFrame, team_ctx: pl.DataFrame) -> pl.DataFrame:
        df = rows.join(
            team_ctx.select("game_id", "team", "team_mu", "team_pass_att", "team_rush_att"),
            on=["game_id", "team"],
            how="left",
        )
        mu_scale = (_f64(df, "team_mu", self.league_mu) / self.league_mu) ** TD_EXPOSURE_POWER
        pos = df["position"].to_list()
        is_qb = np.array([p == "QB" for p in pos])
        pass_idx = 1.0 + DEFENSE_SENSITIVITY * (_f64(df, "opp_def_pass_yds_idx", 1.0) - 1.0)
        rush_idx = 1.0 + DEFENSE_SENSITIVITY * (_f64(df, "opp_def_rush_yds_idx", 1.0) - 1.0)
        rec_idx = 1.0 + DEFENSE_SENSITIVITY * (_f64(df, "opp_def_rec_yds_idx", 1.0) - 1.0)
        rz_prior = np.array(
            [
                POSITION_PRIORS.get(p, POSITION_PRIORS["WR"])["rz_target_share"]
                if p != "QB"
                else 1.0
                for p in pos
            ]
        )
        rz_mult = np.clip(
            np.sqrt(np.maximum(_f64(df, "rz_target_share", 0.0), 1e-3) / rz_prior), 0.6, 1.8
        )
        gl_prior = np.array(
            [POSITION_PRIORS["RB"]["gl_carry_share"] if p == "RB" else 0.02 for p in pos]
        )
        gl_mult = np.clip(
            np.sqrt(np.maximum(_f64(df, "gl_carry_share", 0.0), 1e-3) / gl_prior), 0.6, 1.8
        )
        p_act = np.clip(_f64(df, "p_active", 1.0), 0.0, 1.0)
        pass_share = _f64(df, "pass_att_share", 0.0)
        qb_w = np.where(is_qb, pass_share * p_act, 0.0)
        team_pass = _f64(df, "team_pass_att", 0.0)
        team_rush = _f64(df, "team_rush_att", 0.0)
        qb_sum = (
            df.with_columns(pl.Series("_w", qb_w))
            .group_by("game_id", "team")
            .agg(pl.col("_w").sum().alias("_s"))
        )
        qb_norm = _f64(df.join(qb_sum, on=["game_id", "team"], how="left"), "_s", 0.0)
        carry_rate = np.where(is_qb, _f64(df, "qb_rush_ypc"), _f64(df, "rush_ypc"))
        carry_td = np.where(is_qb, _f64(df, "qb_rush_td_per_carry"), _f64(df, "rush_td_per_carry"))
        return df.with_columns(
            pl.Series("proj_pass_att", np.where(is_qb, team_pass * _safe_div(qb_w, qb_norm), 0.0)),
            pl.Series("proj_carries", team_rush * _f64(df, "carry_share") * p_act),
            pl.Series(
                "proj_targets",
                np.where(is_qb, 0.0, team_pass * _f64(df, "target_share") * p_act),
            ),
            pl.Series("r_comp", np.where(is_qb, _f64(df, "qb_comp_rate"), 0.0)),
            pl.Series("r_ypa", np.where(is_qb, _f64(df, "qb_yds_per_att") * pass_idx, 0.0)),
            pl.Series("r_td_att", np.where(is_qb, _f64(df, "qb_td_per_att") * mu_scale, 0.0)),
            pl.Series("r_int_att", np.where(is_qb, _f64(df, "qb_int_per_att"), 0.0)),
            pl.Series("r_ypc", carry_rate * rush_idx),
            pl.Series("r_td_carry", carry_td * mu_scale * np.where(is_qb, 1.0, gl_mult)),
            pl.Series("r_catch", _f64(df, "catch_rate")),
            pl.Series("r_ypt", _f64(df, "rec_yds_per_target") * rec_idx),
            pl.Series(
                "r_td_tgt", _f64(df, "rec_td_per_target") * mu_scale * np.where(is_qb, 0.0, rz_mult)
            ),
        )

    def predict(
        self,
        rows: pl.DataFrame,
        stage1_preds: pl.DataFrame,
        *,
        with_probabilities: bool = True,
    ) -> pl.DataFrame:
        """Return reconciled player projections for `rows` (Stage 2 Gold) given Stage 1."""
        if not self._is_fitted:
            raise RuntimeError("Stage2FootballPlayerModel must be fitted first.")
        if rows.is_empty():
            return rows.select(
                "game_id", "team", "player_id", "player_name", "position", "p_active"
            ).with_columns(
                *[pl.lit(0.0).alias(c) for c in (*PROJECTION_COLUMNS, *PROBABILITY_COLUMNS)]
            )
        team_ctx = self.project_team_opportunity(rows, stage1_preds)
        raw = self._raw_rates(rows, team_ctx)
        proj = reconcile_football_stage2_projections(raw, team_ctx, gamma=self.gamma)
        keep = [
            "game_id",
            "team",
            "opponent",
            "player_id",
            "player_name",
            "position",
            "p_active",
            "target_share",
            "carry_share",
            *PROJECTION_COLUMNS,
        ]
        out = proj.select([c for c in keep if c in proj.columns]).with_columns(
            pl.col(list(PROJECTION_COLUMNS)).clip(lower_bound=0.0)
        )
        if with_probabilities:
            out = add_threshold_probabilities(out, self.cv)
        return out.sort("game_id", "team", "position", "player_id")

    def evaluate(self, rows: pl.DataFrame, stage1_preds: pl.DataFrame) -> dict[str, float]:
        """MAE of key projections vs. actuals, and vs. a position-mean baseline."""
        done = rows.filter(pl.col("actual_targets").is_not_null())
        if done.is_empty():
            raise ValueError("No completed player rows to evaluate.")
        proj = self.predict(done, stage1_preds, with_probabilities=False).join(
            done.select("game_id", "team", "player_id", *_ACTUAL_TO_PROJ),
            on=["game_id", "team", "player_id"],
        )
        metrics: dict[str, float] = {}
        for pos, actual, projected in (
            ("QB", "actual_pass_att", "proj_pass_att"),
            ("QB", "actual_pass_yds", "proj_pass_yds"),
            ("RB", "actual_carries", "proj_carries"),
            ("RB", "actual_rush_yds", "proj_rush_yds"),
            ("WR", "actual_targets", "proj_targets"),
            ("WR", "actual_rec_yds", "proj_rec_yds"),
            ("TE", "actual_targets", "proj_targets"),
        ):
            sub = proj.filter(pl.col("position") == pos)
            if sub.is_empty():
                continue
            a, p = _f64(sub, actual), _f64(sub, projected)
            metrics[f"mae_{pos.lower()}_{projected}"] = float(np.mean(np.abs(a - p)))
            metrics[f"baseline_mae_{pos.lower()}_{projected}"] = float(
                np.mean(np.abs(a - np.mean(a)))
            )
        return metrics


__all__ = [
    "GAMMA_RECONCILE",
    "PROBABILITY_COLUMNS",
    "PROJECTION_COLUMNS",
    "STAGE2_TARGET_COLUMNS",
    "Stage2FootballPlayerModel",
    "add_threshold_probabilities",
    "reconcile_football_stage2_projections",
    "team_context_from_stage1",
]
