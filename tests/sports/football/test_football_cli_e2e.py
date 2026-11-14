"""Offline football CLI end-to-end: features build -> train -> backtest -> predict -> score."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from rich.console import Console
from typer.testing import CliRunner

from sports_ml import cli
from sports_ml.config import get_settings
from sports_ml.sports.football.ingestion import (
    FootballSilverReconciler,
    NflverseClient,
    NflverseIngestor,
)
from tests.sports.football.synthetic import build_synthetic_silver, write_synthetic_silver
from tests.sports.football.test_football_ingestion import _routes, _transport

runner = CliRunner()


def _invoke(*args: str) -> str:
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == 0, f"{args}: {result.output}\n{result.exception!r}"
    return result.output


@pytest.fixture
def football_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cli, "console", Console(width=300))
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))
    write_synthetic_silver(tmp_path, build_synthetic_silver(future_weeks=1))
    return tmp_path


def test_football_cli_pipeline_end_to_end(football_root: Path) -> None:
    out = _invoke("features", "build", "--sport", "football", "--league", "nfl")
    assert "Stage 1 rows=96" in out and "Stage 2 rows=1254" in out

    out = _invoke("train", "--sport", "football", "--include-stage2", "--version", "e2e")
    assert "Training complete" in out and "Stage 2 rows=" in out
    settings = get_settings(root_dir=football_root)
    assert (settings.paths.artifacts_dir / "football" / "latest" / "metadata.json").exists()

    out = _invoke("backtest", "--sport", "football", "--stage", "all", "--splits", "1")
    assert "Stage 1" in out and "holdout MAE" in out

    # upcoming slate (Stage 1)
    out = _invoke("predict", "--sport", "football", "--offline")
    assert "P(Away)" in out and "P(Home)" in out and "QB matchup" in out

    # explicit week + matchup with Stage 2 tables
    out = _invoke(
        "predict", "--sport", "football", "--offline", "--include-players",
        "--season", "2023", "--week", "9", "--matchup", "KC@BUF",
    )  # fmt: skip
    assert "BUF QB projections" in out and "P(Anytime TD)" in out

    out = _invoke("predict", "--sport", "football", "--offline", "--team", "KC", "--games", "1")
    assert "KC @ BUF" in out.replace("  ", " ") or "BUF" in out

    # completed week -> ledger rows that can be scored
    _invoke(
        "predict", "--sport", "football", "--offline", "--include-players",
        "--season", "2023", "--week", "3",
    )  # fmt: skip
    ledger = pl.read_parquet(settings.paths.predictions_dir / "football_predictions.parquet")
    assert ledger.height > 0
    out = _invoke("score-predictions", "--sport", "football")
    assert "Live Prediction Ledger Accuracy" in out and "Player Projection MAE" in out


def test_football_cli_validation_errors(football_root: Path) -> None:
    _invoke("features", "build", "--sport", "football")
    _invoke("train", "--sport", "football", "--include-stage2")
    both = runner.invoke(
        cli.app,
        ["predict", "--sport", "football", "--offline", "--team", "KC", "--matchup", "KC@BUF"],
    )
    assert both.exit_code != 0
    games_only = runner.invoke(cli.app, ["predict", "--sport", "football", "--games", "2"])
    assert games_only.exit_code != 0
    bad_stage = runner.invoke(cli.app, ["backtest", "--sport", "football", "--stage", "9"])
    assert bad_stage.exit_code != 0
    missing = runner.invoke(
        cli.app,
        ["predict", "--sport", "football", "--offline", "--season", "2023", "--week", "1",
         "--matchup", "KC@KC"],
    )  # fmt: skip
    assert missing.exit_code != 0


def test_football_cli_offline_ingest_uses_cache_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "console", Console(width=300))
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))
    settings = get_settings(root_dir=tmp_path)
    settings.paths.ensure_dirs(sport="football")
    raw = settings.paths.raw_dir / "football" / "nflverse"
    NflverseIngestor(
        NflverseClient(raw, transport=_transport(_routes())),
        FootballSilverReconciler(tmp_path / "scratch_silver"),
        active_season=2026,
    ).ingest(from_year=2023, to_year=2023)

    out = _invoke("ingest", "--sport", "football", "--from-year", "2023", "--offline")
    assert "Ingestion complete" in out
    games = pl.read_parquet(settings.paths.silver_dir / "football" / "games.parquet")
    assert games.height > 0
    players = pl.read_parquet(settings.paths.silver_dir / "football" / "player_game_logs.parquet")
    assert {"rz_targets", "rz_carries", "gl_carries", "completions"} <= set(players.columns)
