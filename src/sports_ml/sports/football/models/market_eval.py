"""Evaluation-only market benchmark for football predictions (`models/market_eval.py`).

Joins model predictions (walk-forward out-of-sample rows or the live ledger) to Silver games and
compares them with the de-vigged closing moneyline, the closing spread (ATS), and the closing
total (over/under). The market columns are read here and nowhere in the feature path: nothing in
this module can influence a prediction, and mutating `market_*` only changes the reports.

`market_spread_home_at_pred` / `market_total_at_pred` (optional prediction-frame columns recorded
in the ledger when the prediction was made) enable closing-line-value summaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from sports_ml.core.evaluation import (
    BenchmarkComparisonReport,
    LineMarketReport,
    compare_against_baselines,
    evaluate_line_markets,
)

GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "home_score",
    "away_score",
    "home_win",
    "is_tie",
    "market_home_decimal_odds",
    "market_away_decimal_odds",
    "market_prob_home",
    "market_spread_home",
    "market_total_points",
)


@dataclass(frozen=True)
class FootballMarketEvaluation:
    """Moneyline benchmark plus ATS / over-under reports for games that have closing lines."""

    n_games: int
    n_with_moneyline: int
    moneyline: BenchmarkComparisonReport | None
    lines: LineMarketReport

    @property
    def has_market(self) -> bool:
        return (
            self.moneyline is not None
            or self.lines.spread is not None
            or self.lines.total is not None
        )


def _col(frame: pl.DataFrame, name: str) -> np.ndarray:
    return np.asarray(frame[name].cast(pl.Float64).fill_null(np.nan).to_numpy(), dtype=np.float64)


def evaluate_football_market(
    predictions: pl.DataFrame, games: pl.DataFrame
) -> FootballMarketEvaluation | None:
    """Benchmark `predictions` against closing odds; `None` when no completed game has a line.

    `predictions` needs `game_id`, `home_win_prob`, `expected_margin`, and `expected_total`.
    """
    present = [c for c in GAME_COLUMNS if c in games.columns]
    completed = games.select(present).filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    pred_cols = [
        c
        for c in (
            "game_id",
            "home_win_prob",
            "expected_margin",
            "expected_total",
            "market_spread_home_at_pred",
            "market_total_at_pred",
        )
        if c in predictions.columns
    ]
    joined = (
        predictions.select(pred_cols)
        .unique(subset=["game_id"], keep="last")
        .join(completed, on="game_id", how="inner")
    )
    if joined.is_empty():
        return None
    for name in GAME_COLUMNS[5:]:
        if name not in joined.columns:
            joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    if "is_tie" not in joined.columns:
        joined = joined.with_columns(pl.lit(0).alias("is_tie"))

    margin = _col(joined, "home_score") - _col(joined, "away_score")
    lines = evaluate_line_markets(
        model_margin=_col(joined, "expected_margin"),
        actual_margin=margin,
        closing_spread_home=_col(joined, "market_spread_home"),
        taken_spread_home=(
            _col(joined, "market_spread_home_at_pred")
            if "market_spread_home_at_pred" in joined.columns
            else None
        ),
        model_total=_col(joined, "expected_total"),
        actual_total=_col(joined, "home_score") + _col(joined, "away_score"),
        closing_total=_col(joined, "market_total_points"),
        taken_total=(
            _col(joined, "market_total_at_pred")
            if "market_total_at_pred" in joined.columns
            else None
        ),
    )

    two_way = joined.filter((pl.col("is_tie").fill_null(0) == 0) & pl.col("home_win").is_not_null())
    moneyline: BenchmarkComparisonReport | None = None
    n_ml = 0
    if not two_way.is_empty():
        y = np.asarray(two_way["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        market = _col(two_way, "market_prob_home")
        n_ml = int(np.isfinite(market).sum())
        if n_ml > 0:
            moneyline = compare_against_baselines(
                _col(two_way, "home_win_prob"),
                y,
                train_home_win_rate=float(np.mean(y)),
                market_probs=market,
                market_decimal_odds=np.column_stack(
                    [
                        _col(two_way, "market_home_decimal_odds"),
                        _col(two_way, "market_away_decimal_odds"),
                    ]
                ),
            )
    if moneyline is None and lines.spread is None and lines.total is None:
        return None
    return FootballMarketEvaluation(
        n_games=joined.height, n_with_moneyline=n_ml, moneyline=moneyline, lines=lines
    )


__all__ = ["FootballMarketEvaluation", "evaluate_football_market"]
