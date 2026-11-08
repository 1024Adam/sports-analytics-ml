"""Offline end-to-end tests for `starred` and `predict --starred` across all four sports.

Hockey uses a stubbed predictor (no hockey Silver fixtures exist); baseball, soccer and football
run the real offline pipeline (ingest -> features -> train) on tiny fixtures.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from rich.console import Console
from typer.testing import CliRunner

from sports_ml import cli
from sports_ml.cli import app
from sports_ml.sports.hockey.models.predictor import HockeyMatchupPredictor, MatchupPrediction
from tests.sports.baseball.test_baseball_cli_e2e import _seed_offline_baseball_raw_cache
from tests.sports.football.synthetic import build_synthetic_silver, write_synthetic_silver
from tests.sports.soccer.test_soccer_cli_e2e import _write_offline_soccer_fixtures

runner = CliRunner()

HOCKEY_FUT = datetime(2024, 1, 15, 0, 30, tzinfo=UTC)
HOCKEY_GAMES = [
    # (game_id, kickoff, home, away, state)
    (2023020900, HOCKEY_FUT, "COL", "VGK", "FUT"),
    (2023020100, datetime(2023, 10, 20, 0, 0, tzinfo=UTC), "BOS", "TOR", "FINAL"),
]
HOCKEY_PROBS = {"COL": (0.6123456789, 0.3876543211), "BOS": (0.5512345678, 0.4487654322)}


class _StubHockeyPredictor:
    """Deterministic stand-in for `HockeyMatchupPredictor` (records every call)."""

    calls: list[tuple[str, int, bool]] = []

    def predict_team_next_games(
        self,
        team: str,
        n_games: int = 1,
        *,
        from_date: date | None = None,
        include_players: bool = False,
        offline: bool = False,
        persist_ledger: bool = True,
    ) -> list[MatchupPrediction]:
        type(self).calls.append((team, n_games, offline))
        gid, kickoff, home, away, _state = next(g for g in HOCKEY_GAMES if team in (g[2], g[3]))
        p_home, p_away = HOCKEY_PROBS[team]
        return [
            MatchupPrediction(
                game_id=gid,
                game_date=kickoff.date(),
                home_team=home,
                away_team=away,
                home_win_prob=p_home,
                away_win_prob=p_away,
                expected_home_goals=3.1,
                expected_away_goals=2.7,
            )
        ][:n_games]


def _write_hockey_fixture(root: Path) -> None:
    silver = root / "data" / "silver" / "hockey"
    silver.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "game_id": [g[0] for g in HOCKEY_GAMES],
            "game_date": [g[1].date() for g in HOCKEY_GAMES],
            "start_time_utc": [g[1] for g in HOCKEY_GAMES],
            "home_team": [g[2] for g in HOCKEY_GAMES],
            "away_team": [g[3] for g in HOCKEY_GAMES],
            "game_state": [g[4] for g in HOCKEY_GAMES],
        }
    ).write_parquet(silver / "games.parquet")
    bundle = root / "artifacts" / "hockey" / "latest"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "metadata.json").write_text("{}", encoding="utf-8")


def _invoke(*args: str, ok: bool = True) -> str:
    result = runner.invoke(app, list(args))
    if ok:
        assert result.exit_code == 0, f"{args}: {result.output}{result.exception!r}"
    return result.output


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("starred_world")
    mp = pytest.MonkeyPatch()
    mp.setenv("SPORTS_ML_ROOT_DIR", str(root))
    mp.delenv("SPORTS_ML_STARRED_TEAMS_FILE", raising=False)
    mp.delenv("THE_ODDS_API_KEY", raising=False)
    mp.setattr(cli, "console", Console(width=300))
    _StubHockeyPredictor.calls = []
    mp.setattr(
        HockeyMatchupPredictor,
        "from_artifacts",
        classmethod(lambda cls, artifact_dir, *, settings=None: _StubHockeyPredictor()),
    )
    _write_hockey_fixture(root)

    _seed_offline_baseball_raw_cache(root / "data" / "raw")
    _invoke("ingest", "--sport", "baseball", "--league", "all", "--offline")
    _invoke("features", "build", "--sport", "baseball", "--league", "all")
    _invoke("train", "--sport", "baseball", "--league", "all")

    _write_offline_soccer_fixtures(root / "data" / "raw")
    _invoke("ingest", "--sport", "soccer", "--offline")
    _invoke("features", "build", "--sport", "soccer")
    _invoke("train", "--sport", "soccer")

    write_synthetic_silver(root, build_synthetic_silver(future_weeks=1))
    _invoke("features", "build", "--sport", "football")
    _invoke("train", "--sport", "football")
    try:
        yield root
    finally:
        mp.undo()


@pytest.fixture
def starred_file(world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "starred_teams.toml"
    monkeypatch.setenv("SPORTS_ML_STARRED_TEAMS_FILE", str(path))
    return path


def _star(*pairs: tuple[str, str]) -> None:
    for sport, team in pairs:
        _invoke("starred", "add", "--sport", sport, "--team", team)


def _ledger(world: Path, sport: str) -> pl.DataFrame:
    return pl.read_parquet(world / "data" / "predictions" / f"{sport}_predictions.parquet")


def _float_values(row: dict[str, Any]) -> dict[str, float]:
    return {k: v for k, v in row.items() if isinstance(v, float)}


# --------------------------------------------------------------------------- management CLI


def test_starred_add_list_remove_cli(starred_file: Path) -> None:
    assert "No starred teams yet" in _invoke("starred", "list")
    assert "Starred: hockey / nhl / COL" in _invoke(
        "starred", "add", "--sport", "hockey", "--team", "COL"
    )
    assert "Starred: baseball / mlb / ATH" in _invoke(
        "starred", "add", "--sport", "baseball", "--team", "OAK"
    )
    out = _invoke("starred", "add", "--sport", "baseball", "--team", "ATH")
    assert "Already starred" in out
    _invoke("starred", "add", "--sport", "baseball", "--team", "HAN")
    listing = _invoke("starred", "list")
    assert "Colorado Avalanche" in listing and "npb" in listing and "Athletics" in listing
    assert starred_file.exists()

    bad_sport = _invoke("starred", "add", "--sport", "cricket", "--team", "MI", ok=False)
    assert "Unknown sport" in bad_sport
    assert "Unknown NHL team" in _invoke(
        "starred", "add", "--sport", "hockey", "--team", "ZZZ", ok=False
    )
    assert "Not starred" in _invoke(
        "starred", "remove", "--sport", "hockey", "--team", "BOS", ok=False
    )
    assert "Removed: hockey / nhl / COL" in _invoke(
        "starred", "remove", "--sport", "hockey", "--team", "col"
    )
    assert "colorado" not in _invoke("starred", "list").lower()


def test_empty_list_prints_hint_and_exits_zero(starred_file: Path) -> None:
    out = _invoke("predict", "--starred", "--offline")
    assert "sports-ml starred add --sport" in out
    assert not starred_file.exists()


# --------------------------------------------------------------------- predict --starred


def test_starred_predictions_are_chronological_and_identical_to_single_team_runs(
    world: Path, starred_file: Path
) -> None:
    _star(("baseball", "LAD"), ("hockey", "COL"), ("soccer", "ARS"), ("football", "KC"))
    _StubHockeyPredictor.calls.clear()
    out = _invoke("predict", "--starred", "--offline")

    # one chronological overview: football 2023 < hockey 2024-01 < soccer 2024-10 < baseball 2025
    order = [out.index(f"{s} / {lg}") for s, lg in (
        ("football", "nfl"), ("hockey", "nhl"), ("soccer", "epl"), ("baseball", "mlb"),
    )]  # fmt: skip
    assert order == sorted(order)
    assert "Starred Teams — Upcoming Games" in out and "SKIPPED" not in out
    assert "KC @ BUF" in out
    # each sport keeps its own renderer (2-way vs 3-way tables stay separate)
    assert "Stage 1 — Calibrated Game Win Probabilities" in out
    assert "Stage 1 — Calibrated EPL 3-Way (1X2)" in out and "P(Draw)" in out
    for header in ("HOCKEY", "SOCCER", "BASEBALL", "FOOTBALL"):
        assert header in out

    starred_rows = {
        s: _ledger(world, s).tail(1).row(0, named=True) for s in ("baseball", "soccer", "football")
    }
    assert _StubHockeyPredictor.calls == [("COL", 1, True)]

    single = {
        "baseball": _invoke(
            "predict", "--sport", "baseball", "--team", "LAD", "--games", "1", "--offline"
        ),
        "soccer": _invoke(
            "predict", "--sport", "soccer", "--team", "ARS", "--games", "1", "--offline"
        ),
        "football": _invoke(
            "predict", "--sport", "football", "--team", "KC", "--games", "1", "--offline"
        ),
    }
    hockey_single = _invoke(
        "predict", "--sport", "hockey", "--team", "COL", "--games", "1", "--offline"
    )
    assert "61.2%" in hockey_single and "61.2%" in out  # stub probabilities render identically
    for sport, text in single.items():
        assert text.strip()
        again = _ledger(world, sport).tail(1).row(0, named=True)
        a, b = _float_values(starred_rows[sport]), _float_values(again)
        assert a and a.keys() == b.keys()
        assert a == b, f"{sport} probabilities differ between starred and single-team runs"


def test_partial_failure_skips_without_aborting_and_all_failed_exits_nonzero(
    world: Path, starred_file: Path
) -> None:
    # BOS: stub team whose only game is already final -> NO_UPCOMING_GAME
    _star(("hockey", "COL"), ("hockey", "BOS"), ("football", "KC"), ("soccer", "ARS"))
    artifacts = world / "artifacts" / "football" / "latest"
    parked = artifacts.with_name("latest_parked")
    shutil.move(artifacts, parked)
    try:
        result = runner.invoke(app, ["predict", "--starred", "--offline"])
        assert result.exit_code == 0, result.output  # other teams still rendered
        out = result.output
        assert "SKIPPED" in out and "hockey/nhl BOS" in out and "NO_UPCOMING_GAME" in out
        assert "football/nfl KC" in out and "NO_TRAINED_ARTIFACTS" in out
        assert "COL" in out and "ARS" in out
        assert "SOURCE_UNAVAILABLE" not in out

        for sport, team in (("hockey", "COL"), ("soccer", "ARS")):
            _invoke("starred", "remove", "--sport", sport, "--team", team)
        failed = runner.invoke(app, ["predict", "--starred", "--offline"])
        assert failed.exit_code != 0
        assert failed.output.count("SKIPPED") == 2
    finally:
        shutil.move(parked, artifacts)


def test_source_failures_are_reported_per_team(world: Path, starred_file: Path) -> None:
    _star(("baseball", "LAD"), ("baseball", "OAK"))  # ATH has no games in the fixture schedule
    out = _invoke("predict", "--starred", "--offline")
    assert "LAD" in out and "SKIPPED" in out and "baseball/mlb ATH" in out
    assert "NO_UPCOMING_GAME" in out or "SOURCE_UNAVAILABLE" in out


def test_each_sport_syncs_once_regardless_of_team_count(
    world: Path, starred_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, int] = {"hockey": 0, "football": 0, "soccer": 0, "baseball": 0}

    def counter(sport: str) -> Any:
        def run(*args: Any, **kwargs: Any) -> None:
            calls[sport] += 1

        return run

    monkeypatch.setattr(cli, "_run_hockey_ingestion", counter("hockey"))
    monkeypatch.setattr(cli, "_run_football_ingestion", counter("football"))
    monkeypatch.setattr(cli, "_run_soccer_ingestion", counter("soccer"))
    monkeypatch.setattr(cli, "_run_baseball_ingestion", counter("baseball"))
    _star(("hockey", "COL"), ("hockey", "BOS"), ("football", "KC"), ("football", "BUF"))
    out = _invoke("predict", "--starred")
    assert calls == {"hockey": 1, "football": 1, "soccer": 0, "baseball": 0}  # once per sport
    assert "COL" in out and "BUF" in out

    calls.update(hockey=0, football=0)
    _invoke("predict", "--starred", "--offline")
    assert calls == {
        "hockey": 0,
        "football": 0,
        "soccer": 0,
        "baseball": 0,
    }  # --offline never syncs


def test_games_option_and_alias_and_exclusive_options(world: Path, starred_file: Path) -> None:
    _star(("baseball", "LAD"), ("hockey", "COL"))
    flag = _invoke("predict", "--starred", "--offline")
    alias = _invoke("predict", "starred", "--offline")
    assert flag == alias

    two = _invoke("predict", "--starred", "--games", "2", "--offline")
    assert two.count("baseball / mlb") == 2  # LAD has two scheduled games (days 13 and 14)
    assert flag.count("baseball / mlb") == 1

    for extra in (
        ["--sport", "hockey"],
        ["--team", "COL"],
        ["--matchup", "COL@BOS"],
        ["--date", "2024-01-15"],
    ):
        bad = runner.invoke(app, ["predict", "--starred", *extra])
        assert bad.exit_code != 0
        assert "cannot be combined" in bad.output and extra[0] in bad.output
        alias_bad = runner.invoke(app, ["predict", "starred", *extra])
        assert alias_bad.exit_code != 0 and "cannot be combined" in alias_bad.output
    assert runner.invoke(app, ["predict", "stared"]).exit_code != 0
    # without --starred nothing changes: `--sport` still defaults to hockey
    assert "61.2%" in _invoke("predict", "--team", "COL", "--offline")


def test_missing_artifacts_skip_before_any_download_and_progress_is_streamed(
    world: Path, starred_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    syncs: list[str] = []
    monkeypatch.setattr(cli, "_run_football_ingestion", lambda *a, **k: syncs.append("football"))
    monkeypatch.setattr(cli, "_run_hockey_ingestion", lambda *a, **k: syncs.append("hockey"))
    _star(("football", "KC"), ("hockey", "COL"))
    artifacts = world / "artifacts" / "football" / "latest"
    parked = artifacts.with_name("latest_parked")
    shutil.move(artifacts, parked)
    try:
        out = _invoke("predict", "--starred")
    finally:
        shutil.move(parked, artifacts)
    assert syncs == ["hockey"]  # football has no model, so nothing is downloaded for it
    assert "football: skipped (NO_TRAINED_ARTIFACTS)" in out
    assert "hockey: syncing live data..." in out and "hockey COL: predicted in" in out
    assert out.index("hockey COL: predicted in") < out.index("SKIPPED")  # streamed, not deferred
