"""Command-line interface (`sports-ml`) built with Typer and Rich."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="sports-ml",
    help="Leak-free multi-sport analytics and two-stage machine learning CLI.",
    no_args_is_help=True,
)
features_app = typer.Typer(
    name="features",
    help="Build and inspect pre-game Gold feature matrices.",
    no_args_is_help=True,
)
app.add_typer(features_app, name="features")

console = Console()


@app.command("ingest")
def ingest_command(
    sport: Annotated[str, typer.Option("--sport", help="Sport plugin slug (e.g., hockey).")] = (
        "hockey"
    ),
    seasons: Annotated[
        str | None,
        typer.Option("--seasons", help="Comma-separated season IDs (e.g., 20232024,20242025)."),
    ] = None,
    current_season: Annotated[
        bool,
        typer.Option("--current-season", help="Sync active season-to-date schedule and boxscores."),
    ] = False,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="Start date (YYYY-MM-DD) for incremental sync."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="End date (YYYY-MM-DD) for incremental sync."),
    ] = None,
) -> None:
    """Download, cache, and reconcile raw sport feeds into Silver Parquet tables."""
    raise NotImplementedError(
        f"CLI subcommand 'ingest' for sport='{sport}' will be wired in Milestone 6 "
        f"(seasons={seasons}, current_season={current_season}, "
        f"start_date={start_date}, end_date={end_date})."
    )


@features_app.command("build")
def features_build_command(
    sport: Annotated[str, typer.Option("--sport", help="Sport plugin slug (e.g., hockey).")] = (
        "hockey"
    ),
) -> None:
    """Recompute leak-free pre-game Gold feature tables from local Silver Parquet data."""
    raise NotImplementedError(
        f"CLI subcommand 'features build' for sport='{sport}' will be wired in Milestone 6."
    )


@app.command("backtest")
def backtest_command(
    sport: Annotated[str, typer.Option("--sport", help="Sport plugin slug (e.g., hockey).")] = (
        "hockey"
    ),
    stage: Annotated[
        str,
        typer.Option("--stage", help="Pipeline stage to evaluate ('1', '2', or 'all')."),
    ] = "1",
) -> None:
    """Run walk-forward time-series backtesting across historical seasons."""
    raise NotImplementedError(
        f"CLI subcommand 'backtest' for sport='{sport}' (stage='{stage}') "
        "will be wired in Milestone 6."
    )


@app.command("train")
def train_command(
    sport: Annotated[str, typer.Option("--sport", help="Sport plugin slug (e.g., hockey).")] = (
        "hockey"
    ),
    include_stage2: Annotated[
        bool,
        typer.Option(
            "--include-stage2/--no-include-stage2",
            help="Also train Stage 2 player stat projection models.",
        ),
    ] = False,
) -> None:
    """Build/refresh Gold features and fit production model artifacts."""
    raise NotImplementedError(
        f"CLI subcommand 'train' for sport='{sport}' (include_stage2={include_stage2}) "
        "will be wired in Milestone 6."
    )


@app.command("predict")
def predict_command(
    sport: Annotated[str, typer.Option("--sport", help="Sport plugin slug (e.g., hockey).")] = (
        "hockey"
    ),
    game_date: Annotated[
        str | None,
        typer.Option("--date", help="Target prediction date (YYYY-MM-DD)."),
    ] = None,
    matchup: Annotated[
        str | None,
        typer.Option("--matchup", help="Optional matchup filter (e.g., TOR@BOS)."),
    ] = None,
    include_players: Annotated[
        bool,
        typer.Option(
            "--include-players/--no-include-players",
            help="Run Stage 2 player stat projections.",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Skip live network sync and use cached Silver tables."),
    ] = False,
) -> None:
    """Generate Stage 1 game win probabilities and optional Stage 2 player projections."""
    raise NotImplementedError(
        f"CLI subcommand 'predict' for sport='{sport}' (date={game_date}, matchup={matchup}, "
        f"include_players={include_players}, offline={offline}) will be wired in Milestone 6."
    )


@app.command("score-predictions")
def score_predictions_command(
    sport: Annotated[str, typer.Option("--sport", help="Sport plugin slug (e.g., hockey).")] = (
        "hockey"
    ),
) -> None:
    """Score the live append-only prediction ledger against completed Silver games."""
    raise NotImplementedError(
        f"CLI subcommand 'score-predictions' for sport='{sport}' will be wired in Milestone 6."
    )
