"""Rich renderers for the football CLI (`predict`, `backtest`, `score-predictions`)."""

from __future__ import annotations

import polars as pl
from rich.console import Console
from rich.table import Table

from sports_ml.sports.football.models.predictor import (
    FootballLedgerScore,
    FootballMatchupPrediction,
    FootballQBCard,
)
from sports_ml.sports.football.models.stage1_game import FootballStage1BacktestResult

POSITION_ORDER: tuple[str, ...] = ("QB", "RB", "WR", "TE")

_POSITION_COLUMNS: dict[str, tuple[tuple[str, str, int], ...]] = {
    "QB": (
        ("Att", "proj_pass_att", 1),
        ("Cmp", "proj_completions", 1),
        ("Pass Yds", "proj_pass_yds", 0),
        ("Pass TD", "proj_pass_td", 2),
        ("INT", "proj_int", 2),
        ("Rush Yds", "proj_rush_yds", 0),
    ),
    "RB": (
        ("Car", "proj_carries", 1),
        ("Rush Yds", "proj_rush_yds", 0),
        ("Rush TD", "proj_rush_td", 2),
        ("Tgt", "proj_targets", 1),
        ("Rec", "proj_rec", 1),
        ("Rec Yds", "proj_rec_yds", 0),
    ),
    "WR": (
        ("Tgt", "proj_targets", 1),
        ("Rec", "proj_rec", 1),
        ("Rec Yds", "proj_rec_yds", 0),
        ("Rec TD", "proj_rec_td", 2),
        ("P(75+ Yds)", "prob_75plus_rec_yds", 3),
    ),
    "TE": (
        ("Tgt", "proj_targets", 1),
        ("Rec", "proj_rec", 1),
        ("Rec Yds", "proj_rec_yds", 0),
        ("Rec TD", "proj_rec_td", 2),
        ("P(75+ Yds)", "prob_75plus_rec_yds", 3),
    ),
}


def _qb_line(card: FootballQBCard) -> str:
    name = card.qb_name or (f"QB {card.qb_id}" if card.qb_id is not None else "unknown")
    p = f" ({card.p_starts:.0%})" if card.p_starts is not None else ""
    tags = [t for t, on in (("backup", card.is_backup), ("debut/new", card.is_debut)) if on]
    suffix = f" [{', '.join(tags)}]" if tags else ""
    return f"{card.team}: {name}{p} — {card.status}{suffix}"


def render_football_predictions(
    console: Console, predictions: list[FootballMatchupPrediction]
) -> None:
    """Print the probability table, QB matchup cards and position-grouped projections."""
    if not predictions:
        console.print("[yellow]No football games matched the requested slate.[/yellow]")
        return

    table = Table(title="NFL Matchup Predictions (Stage 1)")
    for col, just in (
        ("Wk", "right"),
        ("Matchup", "left"),
        ("P(Away)", "right"),
        ("P(Home)", "right"),
        ("Projected Score", "right"),
        ("Margin (Home)", "right"),
        ("Total", "right"),
        ("Warnings", "left"),
    ):
        table.add_column(col, justify=just, style="cyan" if col == "Matchup" else None)  # type: ignore[arg-type]
    for p in predictions:
        table.add_row(
            f"{p.season} W{p.week}",
            f"{p.away_team} @ {p.home_team}",
            f"{p.away_win_prob:.1%}",
            f"{p.home_win_prob:.1%}",
            f"{p.away_team} {p.expected_away_score:.1f} - "
            f"{p.home_team} {p.expected_home_score:.1f}",
            f"{p.expected_margin:+.1f}",
            f"{p.expected_total:.1f}",
            ", ".join(p.warnings) or "-",
        )
    console.print(table)

    for p in predictions:
        console.print(f"[bold]QB matchup[/bold] {p.away_team} @ {p.home_team}")
        console.print(f"  {_qb_line(p.away_qb)}")
        console.print(f"  {_qb_line(p.home_qb)}")
        if p.player_projections is None:
            continue
        for team, frame in (
            (p.away_team, p.player_projections.away),
            (p.home_team, p.player_projections.home),
        ):
            _render_team_projections(console, team, frame)


def _render_team_projections(console: Console, team: str, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    for pos in POSITION_ORDER:
        sub = frame.filter(pl.col("position") == pos).sort("player_name")
        if sub.is_empty():
            continue
        spec = _POSITION_COLUMNS[pos]
        table = Table(title=f"{team} {pos} projections")
        table.add_column("Player", style="cyan")
        table.add_column("P(Active)", justify="right")
        for label, _col, _dp in spec:
            table.add_column(label, justify="right")
        table.add_column("P(Anytime TD)", justify="right", style="bold green")
        for r in sub.iter_rows(named=True):
            cells = [
                f"{float(r.get(col) or 0.0):.{dp}f}" if dp else f"{float(r.get(col) or 0.0):.0f}"
                for _label, col, dp in spec
            ]
            table.add_row(
                str(r["player_name"] or r["player_id"]),
                f"{float(r['p_active']):.2f}",
                *cells,
                f"{float(r.get('prob_anytime_td') or 0.0):.1%}",
            )
        console.print(table)


def render_football_backtest(
    console: Console,
    result: FootballStage1BacktestResult | None,
    stage2_metrics: dict[str, float] | None,
) -> None:
    """Print Stage 1 benchmark and Stage 2 MAE tables."""
    if result is not None:
        rep = result.benchmark_report
        table = Table(title="Walk-Forward Backtest (FOOTBALL NFL — Stage 1)")
        table.add_column("Benchmark / Model", style="cyan")
        for col in ("Log Loss", "Brier Score", "ECE", "Accuracy"):
            table.add_column(col, justify="right")
        rows = [
            ("Stage 1 Calibrated Model", rep.model),
            ("QB-Adjusted Elo Baseline", rep.elo_baseline),
            ("Plain Elo Baseline", result.plain_elo_metrics),
            ("Home Win-Rate Baseline", rep.home_win_rate_baseline),
            ("Coin-Flip Baseline", rep.coin_flip_baseline),
        ]
        for label, m in rows:
            if m is not None:
                table.add_row(
                    label,
                    f"{m.log_loss:.4f}",
                    f"{m.brier_score:.4f}",
                    f"{m.ece:.4f}",
                    f"{m.accuracy:.3f}",
                )
        console.print(table)
        sm = result.score_metrics
        console.print(
            f"Expected-score MAE: margin={sm.margin_mae:.2f}, total={sm.total_mae:.2f} "
            f"over {sm.n_games} games."
        )
    if stage2_metrics:
        table = Table(title="Stage 2 Player Projections — holdout MAE vs. position-mean baseline")
        table.add_column("Position / Stat", style="cyan")
        table.add_column("Model MAE", justify="right", style="bold green")
        table.add_column("Baseline MAE", justify="right")
        for key, value in stage2_metrics.items():
            if key.startswith("baseline_"):
                continue
            table.add_row(
                key.removeprefix("mae_"), f"{value:.3f}", f"{stage2_metrics['baseline_' + key]:.3f}"
            )
        console.print(table)


def render_football_score(console: Console, score: FootballLedgerScore) -> None:
    """Print live-ledger accuracy and player projection MAE."""
    rep = score.report
    table = Table(title=f"Live Prediction Ledger Accuracy (FOOTBALL NFL — {score.n_games} games)")
    table.add_column("Metric", style="cyan")
    table.add_column("Live Predictions", justify="right", style="bold green")
    table.add_column("Home Win-Rate Baseline", justify="right")
    table.add_column("Coin-Flip Baseline", justify="right")
    for label, attr in (("Log Loss", "log_loss"), ("Brier Score", "brier_score")):
        table.add_row(
            label,
            f"{getattr(rep.model, attr):.4f}",
            f"{getattr(rep.home_win_rate_baseline, attr):.4f}",
            f"{getattr(rep.coin_flip_baseline, attr):.4f}",
        )
    table.add_row(
        "Accuracy",
        f"{rep.model.accuracy:.3f}",
        f"{rep.home_win_rate_baseline.accuracy:.3f}",
        f"{rep.coin_flip_baseline.accuracy:.3f}",
    )
    console.print(table)
    if score.player_mae:
        ptable = Table(title=f"Player Projection MAE ({score.n_player_rows} matched rows)")
        ptable.add_column("Position / Stat", style="cyan")
        ptable.add_column("MAE", justify="right", style="bold green")
        for key, value in score.player_mae.items():
            ptable.add_row(key, f"{value:.3f}")
        console.print(ptable)
