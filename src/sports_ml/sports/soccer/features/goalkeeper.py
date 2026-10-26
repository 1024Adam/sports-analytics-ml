"""Starting goalkeeper resolver and rolling Bayesian-shrunk GSAA/90 features (`goalkeeper.py`).

Computes strictly leak-free rolling 10-start and 19-start shrunk GSAA/90 (`k = 12.0`),
save percentage (`SV%`), clean-sheet rate, starter fatigue (`gk_days_rest`,
`gk_starts_last_14d`), and a `StarterModifierHook` for pre-match Elo adjustments.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Final

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import (
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.soccer.constants import normalize_soccer_club

GK_SHRINKAGE_K: Final[float] = 12.0
DEFAULT_GK_WINDOWS: Final[tuple[int, ...]] = (10, 19)
PRIOR_GK_GSAA_PER_90: Final[float] = 0.0
PRIOR_GK_SAVE_PCT: Final[float] = 0.710
PRIOR_GK_CLEAN_SHEET_RATE: Final[float] = 0.280
MAX_GK_REST_DAYS: Final[int] = 14


def _compute_gk_fatigue_columns(
    sorted_df: pl.DataFrame,
    *,
    max_rest_days: int = MAX_GK_REST_DAYS,
) -> pl.DataFrame:
    """Compute strictly pre-match `gk_days_rest` and `gk_starts_last_14d` per goalkeeper."""
    if sorted_df.is_empty():
        return sorted_df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("gk_days_rest"),
            pl.lit(None, dtype=pl.Int32).alias("gk_starts_last_14d"),
        )

    rows = sorted_df.to_dicts()
    n_rows = len(rows)
    days_rest_out = [max_rest_days] * n_rows
    starts_14d_out = [0] * n_rows

    history: dict[tuple[int, int], list[tuple[date, bool]]] = {}

    for idx, row in enumerate(rows):
        pid = int(row["player_id"])
        season = int(row.get("season") or 0)
        start_utc: datetime = row["start_time_utc"]
        ev_date: date = (
            row["match_date"]
            if "match_date" in row and row["match_date"] is not None
            else start_utc.date()
        )
        is_starter = bool(row.get("is_starter", True))

        key = (pid, season)
        g_hist = history.setdefault(key, [])

        if not g_hist:
            days_rest_out[idx] = max_rest_days
            starts_14d_out[idx] = 0
        else:
            prev_date, _ = g_hist[-1]
            raw_days = max(0, (ev_date - prev_date).days)
            days_rest_out[idx] = min(raw_days, max_rest_days)
            starts_14d_out[idx] = sum(
                1 for (d, started) in g_hist if started and 1 <= (ev_date - d).days <= 14
            )

        g_hist.append((ev_date, is_starter))

    return sorted_df.with_columns(
        pl.Series("gk_days_rest", days_rest_out, dtype=pl.Int32),
        pl.Series("gk_starts_last_14d", starts_14d_out, dtype=pl.Int32),
    )


def build_goalkeeper_rolling_features(
    gk_match_logs: pl.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_GK_WINDOWS,
    shrinkage_k: float = GK_SHRINKAGE_K,
) -> pl.DataFrame:
    """Compute leak-free rolling shrunk GSAA/90, SV%, clean-sheet rate, and fatigue features."""
    working = gk_match_logs
    if "match_id" in working.columns and "event_id" not in working.columns:
        working = working.with_columns(pl.col("match_id").cast(pl.Int64).alias("event_id"))
    if "gsaa" not in working.columns and {"xga_faced", "goals_against"}.issubset(working.columns):
        working = working.with_columns(
            (pl.col("xga_faced").cast(pl.Float64) - pl.col("goals_against").cast(pl.Float64)).alias(
                "gsaa"
            )
        )
    if "shots_on_target_against" not in working.columns and {"saves", "goals_against"}.issubset(
        working.columns
    ):
        working = working.with_columns(
            (pl.col("saves") + pl.col("goals_against"))
            .cast(pl.Int16)
            .alias("shots_on_target_against")
        )
    if "clean_sheet" not in working.columns and {"minutes", "goals_against"}.issubset(
        working.columns
    ):
        working = working.with_columns(
            pl.when((pl.col("minutes") >= 60.0) & (pl.col("goals_against") == 0))
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("clean_sheet")
        )

    sorted_df = sort_events_chronologically(working, extra_tiebreakers=["player_id"])
    with_fatigue = _compute_gk_fatigue_columns(sorted_df)

    exprs: list[pl.Expr] = [
        pregame_sample_count("gsaa", group_by="player_id").alias("gk_prior_matches_n"),
    ]

    for w in windows:
        n_w = pregame_sample_count("gsaa", group_by="player_id", window=w)
        raw_gsaa_90 = pregame_exposure_rate(
            "gsaa", "minutes", window=w, group_by="player_id", scale_factor=90.0
        )
        raw_sv_pct = pregame_exposure_rate(
            "saves", "shots_on_target_against", window=w, group_by="player_id", scale_factor=1.0
        )
        raw_cs_rate = pregame_rolling_mean("clean_sheet", window=w, group_by="player_id")
        exprs.extend(
            [
                bayesian_shrinkage(raw_gsaa_90, n_w, PRIOR_GK_GSAA_PER_90, shrinkage_k).alias(
                    f"gsaa_per_90_shrunk_w{w}"
                ),
                bayesian_shrinkage(raw_sv_pct, n_w, PRIOR_GK_SAVE_PCT, shrinkage_k).alias(
                    f"save_pct_shrunk_w{w}"
                ),
                bayesian_shrinkage(raw_cs_rate, n_w, PRIOR_GK_CLEAN_SHEET_RATE, shrinkage_k).alias(
                    f"clean_sheet_rate_shrunk_w{w}"
                ),
            ]
        )

    result = with_fatigue.with_columns(exprs)
    ref_w = 10 if 10 in windows else windows[0]
    return result.with_columns(
        pl.col(f"gsaa_per_90_shrunk_w{ref_w}").alias("gsaa_per_90_shrunk"),
        pl.col(f"save_pct_shrunk_w{ref_w}").alias("save_pct_shrunk"),
        pl.col(f"clean_sheet_rate_shrunk_w{ref_w}").alias("clean_sheet_rate_shrunk"),
        (pl.col("gk_prior_matches_n") < shrinkage_k).alias("is_low_sample_gk"),
    )


def resolve_team_starting_gk_features(
    gk_features_df: pl.DataFrame,
) -> pl.DataFrame:
    """Select one starting goalkeeper row per `(match_id, team)` for Stage 1 & Stage 2 joins."""
    if gk_features_df.is_empty():
        return gk_features_df

    event_col = "match_id" if "match_id" in gk_features_df.columns else "event_id"
    sorted_starters = gk_features_df.sort(
        [event_col, "team", "is_starter", "minutes", "player_id"],
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


class GoalkeeperStarterEloHook:
    """Implements `StarterModifierHook` mapping starting GK `gsaa_per_90_shrunk` to Elo points."""

    def __init__(
        self,
        gk_features_df: pl.DataFrame,
        *,
        elo_points_per_gsaa90: float = 35.0,
    ) -> None:
        self.elo_points_per_gsaa90 = elo_points_per_gsaa90
        self._lookup: dict[tuple[int, str], float] = {}
        if not gk_features_df.is_empty():
            starters = resolve_team_starting_gk_features(gk_features_df)
            ev_col = "match_id" if "match_id" in starters.columns else "event_id"
            for row in starters.iter_rows(named=True):
                ev_id = int(row[ev_col])
                team = normalize_soccer_club(str(row["team"]))
                gsaa = float(row.get("gsaa_per_90_shrunk") or 0.0)
                self._lookup[(ev_id, team)] = gsaa * self.elo_points_per_gsaa90

    def get_starter_elo_delta(self, event_id: int, entity_id: str) -> float:
        """Return pre-match Elo point adjustment for `entity_id`'s starting goalkeeper."""
        return self._lookup.get((int(event_id), normalize_soccer_club(entity_id)), 0.0)
