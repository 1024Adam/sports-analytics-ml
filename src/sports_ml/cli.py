"""Command-line interface (`sports-ml`) built with Typer and Rich."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from sports_ml.config import RANDOM_SEED, Settings, get_settings
from sports_ml.core.evaluation import evaluate_probabilities
from sports_ml.core.registry import ModelRegistry
from sports_ml.sports.hockey.features.player import build_stage2_player_features
from sports_ml.sports.hockey.ingestion.moneypuck import MoneyPuckClient
from sports_ml.sports.hockey.ingestion.nhl_api import (
    NHLWebAPIClient,
    derive_active_nhl_season,
    parse_boxscore_payload,
    parse_schedule_payload,
)
from sports_ml.sports.hockey.ingestion.reconciler import HockeySilverReconciler
from sports_ml.sports.hockey.models.predictor import (
    HockeyMatchupPredictor,
    MatchupPrediction,
    build_and_persist_gold_features,
    score_prediction_ledger,
)
from sports_ml.sports.hockey.models.stage1_game import (
    Stage1GameOutcomeModel,
    run_stage1_walk_forward_backtest,
)
from sports_ml.sports.hockey.models.stage2_player import Stage2PlayerProjectionModel

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


def _ensure_supported_sport(sport: str) -> str:
    slug = sport.strip().lower()
    if slug != "hockey":
        raise typer.BadParameter(f"Unsupported sport '{sport}'. Currently active plugin: 'hockey'.")
    return slug


def _run_hockey_ingestion(
    settings: Settings,
    *,
    seasons: str | None = None,
    current_season: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    offline: bool = False,
) -> pl.DataFrame:
    """Execute hybrid NHL Web API + MoneyPuck ingestion and reconcile into Silver tables."""
    settings.paths.ensure_dirs(sport="hockey")
    raw_nhl_dir = settings.paths.raw_dir / "nhl_api"
    raw_mp_dir = settings.paths.raw_dir / "moneypuck"
    silver_dir = settings.paths.silver_dir / "hockey"

    reconciler = HockeySilverReconciler(silver_dir=silver_dir)
    nhl_client = NHLWebAPIClient(cache_dir=raw_nhl_dir)
    mp_client = MoneyPuckClient(cache_dir=raw_mp_dir)

    schedule_frames: list[pl.DataFrame] = []
    team_frames: list[pl.DataFrame] = []
    skater_frames: list[pl.DataFrame] = []
    goalie_frames: list[pl.DataFrame] = []

    # 1. Process any pre-cached JSON schedules/boxscores in data/raw/nhl_api/
    schedules_dir = raw_nhl_dir / "schedules"
    boxscores_dir = raw_nhl_dir / "boxscores"
    if schedules_dir.exists():
        for sched_file in sorted(schedules_dir.glob("*.json")):
            payload = json.loads(sched_file.read_text(encoding="utf-8"))
            parsed = parse_schedule_payload(payload)
            if not parsed.is_empty():
                schedule_frames.append(parsed)

    if boxscores_dir.exists():
        for box_file in sorted(boxscores_dir.glob("*.json")):
            payload = json.loads(box_file.read_text(encoding="utf-8"))
            t_df, s_df, g_df = parse_boxscore_payload(payload)
            team_frames.append(t_df)
            if not s_df.is_empty():
                skater_frames.append(s_df)
            if not g_df.is_empty():
                goalie_frames.append(g_df)

    # 2. If online and date range or current_season requested, query live endpoints
    if not offline and (start_date or end_date or current_season or seasons):
        if start_date and end_date:
            s_dt = date.fromisoformat(start_date)
            e_dt = date.fromisoformat(end_date)
            range_games = nhl_client.fetch_date_range_games(s_dt, e_dt)
            if not range_games.is_empty():
                schedule_frames.append(range_games)
                for gid in range_games.filter(pl.col("game_state") == "FINAL")["game_id"].to_list():
                    box = nhl_client.fetch_game_boxscore(int(gid))
                    t_df, s_df, g_df = parse_boxscore_payload(box)
                    team_frames.append(t_df)
                    skater_frames.append(s_df)
                    goalie_frames.append(g_df)
        elif current_season:
            today = datetime.now(tz=UTC).date()
            s_dt = today - timedelta(days=7)
            range_games = nhl_client.fetch_date_range_games(s_dt, today)
            if not range_games.is_empty():
                schedule_frames.append(range_games)

    # 3. Load any cached MoneyPuck CSVs under data/raw/moneypuck/
    mp_teams_list: list[pl.DataFrame] = []
    mp_skaters_list: list[pl.DataFrame] = []
    mp_goalies_list: list[pl.DataFrame] = []
    if raw_mp_dir.exists():
        for t_csv in sorted(raw_mp_dir.glob("teams_*.csv")):
            season_yr = int(t_csv.stem.split("_")[1])
            mp_teams_list.append(
                mp_client.load_team_game_stats(season_yr, csv_bytes=t_csv.read_bytes())
            )
        for s_csv in sorted(raw_mp_dir.glob("skaters_*.csv")):
            season_yr = int(s_csv.stem.split("_")[1])
            mp_skaters_list.append(
                mp_client.load_skater_game_stats(season_yr, csv_bytes=s_csv.read_bytes())
            )
        for g_csv in sorted(raw_mp_dir.glob("goalies_*.csv")):
            season_yr = int(g_csv.stem.split("_")[1])
            mp_goalies_list.append(
                mp_client.load_goalie_game_stats(season_yr, csv_bytes=g_csv.read_bytes())
            )

    mp_teams_df = pl.concat(mp_teams_list, how="vertical") if mp_teams_list else None
    mp_skaters_df = pl.concat(mp_skaters_list, how="vertical") if mp_skaters_list else None
    mp_goalies_df = pl.concat(mp_goalies_list, how="vertical") if mp_goalies_list else None

    if schedule_frames:
        reconciler.reconcile_games(pl.concat(schedule_frames, how="vertical"))
    if team_frames:
        reconciler.reconcile_team_game_logs(
            pl.concat(team_frames, how="vertical"),
            moneypuck_teams_df=mp_teams_df,
        )
    if skater_frames:
        reconciler.reconcile_skater_game_logs(
            pl.concat(skater_frames, how="vertical"),
            moneypuck_skaters_df=mp_skaters_df,
        )
    if goalie_frames:
        reconciler.reconcile_goalie_game_logs(
            pl.concat(goalie_frames, how="vertical"),
            moneypuck_goalies_df=mp_goalies_df,
        )

    if reconciler.games_path.exists():
        return pl.read_parquet(reconciler.games_path)
    return pl.DataFrame()


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
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Reconcile from cached raw files without network calls."),
    ] = False,
) -> None:
    """Download, cache, and reconcile raw sport feeds into Silver Parquet tables."""
    slug = _ensure_supported_sport(sport)
    settings = get_settings()
    active_season = derive_active_nhl_season()
    games_df = _run_hockey_ingestion(
        settings,
        seasons=seasons,
        current_season=current_season,
        start_date=start_date,
        end_date=end_date,
        offline=offline,
    )
    console.print(
        f"[bold green]Ingestion complete[/bold green] for sport='{slug}' "
        f"(active_season={active_season}, silver_games={games_df.height})."
    )


@features_app.command("build")
def features_build_command(
    sport: Annotated[str, typer.Option("--sport", help="Sport plugin slug (e.g., hockey).")] = (
        "hockey"
    ),
) -> None:
    """Recompute leak-free pre-game Gold feature tables from local Silver Parquet data."""
    slug = _ensure_supported_sport(sport)
    settings = get_settings()
    s1_df, s2_df, _, _ = build_and_persist_gold_features(settings)
    console.print(
        f"[bold green]Gold features built[/bold green] for sport='{slug}': "
        f"Stage 1 rows={s1_df.height}, Stage 2 rows={s2_df.height}."
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
    n_splits: Annotated[
        int,
        typer.Option("--splits", help="Number of walk-forward folds."),
    ] = 2,
) -> None:
    """Run walk-forward time-series backtesting across historical seasons."""
    slug = _ensure_supported_sport(sport)
    settings = get_settings()
    s1_df, s2_df, _, _ = build_and_persist_gold_features(settings)
    completed_s1 = s1_df.filter(pl.col("home_win").is_not_null())

    n_unique_ts = completed_s1["start_time_utc"].n_unique()
    eff_splits = min(n_splits, max(1, n_unique_ts // 3))
    eff_min_train = max(2, n_unique_ts - eff_splits)

    bt = run_stage1_walk_forward_backtest(
        completed_s1,
        n_splits=eff_splits,
        min_train_groups=eff_min_train,
    )
    rep = bt.benchmark_report

    table = Table(title=f"Walk-Forward Backtest ({slug.upper()} — Stage {stage})")
    table.add_column("Benchmark / Model", style="cyan")
    table.add_column("Log Loss", justify="right")
    table.add_column("Brier Score", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Accuracy", justify="right")

    table.add_row(
        "Stage 1 Calibrated Model",
        f"{rep.model.log_loss:.4f}",
        f"{rep.model.brier_score:.4f}",
        f"{rep.model.ece:.4f}",
        f"{rep.model.accuracy:.3f}",
    )
    table.add_row(
        "Home Win-Rate Baseline",
        f"{rep.home_win_rate_baseline.log_loss:.4f}",
        f"{rep.home_win_rate_baseline.brier_score:.4f}",
        f"{rep.home_win_rate_baseline.ece:.4f}",
        f"{rep.home_win_rate_baseline.accuracy:.3f}",
    )
    table.add_row(
        "Coin-Flip Baseline (0.50)",
        f"{rep.coin_flip_baseline.log_loss:.4f}",
        f"{rep.coin_flip_baseline.brier_score:.4f}",
        f"{rep.coin_flip_baseline.ece:.4f}",
        f"{rep.coin_flip_baseline.accuracy:.3f}",
    )
    console.print(table)

    if stage in {"2", "all"} and not s2_df.is_empty():
        s2_model = Stage2PlayerProjectionModel(random_seed=RANDOM_SEED)
        s2_model.fit(s2_df)
        s2_preds = s2_model.predict(s2_df)
        err_arr = (s2_preds["proj_points"] - s2_preds["points"].cast(pl.Float64)).abs().to_numpy()
        mae_pts = float(err_arr.mean()) if err_arr.size > 0 else 0.0
        console.print(f"[bold cyan]Stage 2 Skater Points MAE[/bold cyan]: {mae_pts:.4f}")


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
    version: Annotated[
        str,
        typer.Option("--version", help="Model artifact version tag."),
    ] = "v1",
) -> None:
    """Build/refresh Gold features and fit production model artifacts."""
    slug = _ensure_supported_sport(sport)
    settings = get_settings()

    # Auto-build/update Gold feature tables from Silver data
    s1_df, s2_df, team_feat_df, goalie_feat_df = build_and_persist_gold_features(settings)
    completed_s1 = s1_df.filter(pl.col("home_win").is_not_null())
    if completed_s1.is_empty():
        raise typer.BadParameter(
            "No completed FINAL games found in Silver/Gold tables to train on."
        )

    s1_model = Stage1GameOutcomeModel(random_seed=RANDOM_SEED)
    s1_model.fit(completed_s1)
    s1_preds = s1_model.predict(completed_s1)

    metrics = evaluate_probabilities(
        s1_preds["home_win_prob"].to_numpy(),
        completed_s1["home_win"].cast(pl.Int64).to_numpy(),
    )

    s2_model: Stage2PlayerProjectionModel | None = None
    s2_train_df: pl.DataFrame | None = None
    s2_cols: list[str] | None = None

    if include_stage2 and not s2_df.is_empty():
        silver_skaters = pl.read_parquet(
            settings.paths.silver_dir / "hockey" / "skater_game_logs.parquet"
        )
        s2_train_df = build_stage2_player_features(
            silver_skaters,
            team_feat_df,
            goalie_feat_df,
            stage1_predictions=s1_preds,
        )
        s2_model = Stage2PlayerProjectionModel(random_seed=RANDOM_SEED)
        s2_model.fit(s2_train_df)
        s2_cols = s2_model.feature_columns

    registry = ModelRegistry(artifacts_dir=settings.paths.artifacts_dir)
    bundle_dir = registry.save_bundle(
        sport=slug,
        model_version=version,
        stage1_model=s1_model,
        stage1_train_df=completed_s1,
        stage1_feature_columns=s1_model.feature_columns,
        validation_metrics={
            "log_loss": metrics.log_loss,
            "brier_score": metrics.brier_score,
            "ece": metrics.ece,
            "accuracy": metrics.accuracy,
        },
        stage2_model=s2_model,
        stage2_train_df=s2_train_df,
        stage2_feature_columns=s2_cols,
        update_latest=True,
    )
    console.print(
        f"[bold green]Trained & saved model bundle[/bold green] at '{bundle_dir}' "
        f"(include_stage2={include_stage2}, log_loss={metrics.log_loss:.4f})."
    )


def _render_predictions(predictions: list[MatchupPrediction]) -> None:
    """Render Stage 1 and optional Stage 2 predictions as Rich terminal tables."""
    game_table = Table(title="Stage 1 — Calibrated Game Win Probabilities")
    game_table.add_column("Date", style="cyan")
    game_table.add_column("Matchup", style="bold")
    game_table.add_column("P(Home Win)", justify="right", style="green")
    game_table.add_column("P(Away Win)", justify="right", style="yellow")
    game_table.add_column("E[Home Goals]", justify="right")
    game_table.add_column("E[Away Goals]", justify="right")
    game_table.add_column("Goalie Status")

    for pred in predictions:
        game_table.add_row(
            pred.game_date.isoformat(),
            f"{pred.away_team} @ {pred.home_team}",
            f"{pred.home_win_prob:.1%}",
            f"{pred.away_win_prob:.1%}",
            f"{pred.expected_home_goals:.2f}",
            f"{pred.expected_away_goals:.2f}",
            f"H:{pred.home_goalie_status} / A:{pred.away_goalie_status}",
        )
    console.print(game_table)

    for pred in predictions:
        for warning in pred.warnings:
            console.print(
                f"[yellow]Warning ({pred.away_team}@{pred.home_team}):[/yellow] {warning}"
            )

        if pred.player_projections is not None:
            proj = pred.player_projections
            skater_table = Table(
                title=f"Stage 2 — Skater Stat Projections ({pred.away_team} @ {pred.home_team})"
            )
            skater_table.add_column("Team", style="cyan")
            skater_table.add_column("Player")
            skater_table.add_column("Pos")
            skater_table.add_column("E[TOI]", justify="right")
            skater_table.add_column("E[G]", justify="right")
            skater_table.add_column("E[A]", justify="right")
            skater_table.add_column("E[PTS]", justify="right", style="bold green")
            skater_table.add_column("E[SOG]", justify="right")
            skater_table.add_column("P(1+ PT)", justify="right")

            combined_skaters = pl.concat(
                [proj.home_skaters.head(6), proj.away_skaters.head(6)],
                how="vertical",
            )
            for s_row in combined_skaters.iter_rows(named=True):
                skater_table.add_row(
                    str(s_row["team"]),
                    str(s_row["player_name"]),
                    str(s_row["position_group"]),
                    f"{float(s_row['proj_toi_min']):.1f}",
                    f"{float(s_row['proj_goals']):.2f}",
                    f"{float(s_row['proj_assists']):.2f}",
                    f"{float(s_row['proj_points']):.2f}",
                    f"{float(s_row['proj_sog']):.2f}",
                    f"{float(s_row['prob_1plus_point']):.1%}",
                )
            console.print(skater_table)


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
    slug = _ensure_supported_sport(sport)
    settings = get_settings()

    if not offline:
        _run_hockey_ingestion(settings, current_season=True, offline=False)

    target_date = (
        date.fromisoformat(game_date) if game_date is not None else datetime.now(tz=UTC).date()
    )

    artifact_dir = settings.paths.artifacts_dir / slug / "latest"
    predictor = HockeyMatchupPredictor.from_artifacts(artifact_dir, settings=settings)
    predictions = predictor.predict_date(
        target_date,
        include_players=include_players,
        matchup_filter=matchup,
        persist_ledger=True,
    )
    _render_predictions(predictions)


@app.command("score-predictions")
def score_predictions_command(
    sport: Annotated[str, typer.Option("--sport", help="Sport plugin slug (e.g., hockey).")] = (
        "hockey"
    ),
) -> None:
    """Score the live append-only prediction ledger against completed Silver games."""
    slug = _ensure_supported_sport(sport)
    settings = get_settings()
    report = score_prediction_ledger(settings)

    table = Table(title=f"Live Prediction Ledger Accuracy ({slug.upper()})")
    table.add_column("Metric", style="cyan")
    table.add_column("Live Predictions", justify="right", style="bold green")
    table.add_column("Home Win-Rate Baseline", justify="right")
    table.add_column("Coin-Flip Baseline", justify="right")

    table.add_row(
        "Log Loss",
        f"{report.model.log_loss:.4f}",
        f"{report.home_win_rate_baseline.log_loss:.4f}",
        f"{report.coin_flip_baseline.log_loss:.4f}",
    )
    table.add_row(
        "Brier Score",
        f"{report.model.brier_score:.4f}",
        f"{report.home_win_rate_baseline.brier_score:.4f}",
        f"{report.coin_flip_baseline.brier_score:.4f}",
    )
    table.add_row(
        "ECE",
        f"{report.model.ece:.4f}",
        f"{report.home_win_rate_baseline.ece:.4f}",
        f"{report.coin_flip_baseline.ece:.4f}",
    )
    table.add_row(
        "Accuracy",
        f"{report.model.accuracy:.3f}",
        f"{report.home_win_rate_baseline.accuracy:.3f}",
        f"{report.coin_flip_baseline.accuracy:.3f}",
    )
    console.print(table)
