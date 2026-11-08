"""CLI glue for starred teams: per-sport adapters over the existing predictors, and rendering.

Each adapter calls the exact plugin entry point behind `predict --sport X --team Y --games N`
(`predict_team_next_games` with the same arguments and ledger append), so starred predictions are
identical to the single-team command.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console
from rich.table import Table

from sports_ml.config import Settings
from sports_ml.core.starred import (
    ArtifactsMissingError,
    GameSlot,
    StarredRunResult,
    StarredSportAdapter,
    StarredTeam,
    StarredTeamsConfig,
    run_starred_predictions,
)
from sports_ml.sports.baseball.models.predictor import BaseballMatchupPredictor
from sports_ml.sports.football.models.predictor import FootballMatchupPredictor
from sports_ml.sports.hockey.models.predictor import HockeyMatchupPredictor
from sports_ml.sports.soccer.models.predictor import SoccerMatchupPredictor

SyncHooks = Mapping[str, Callable[[], None]]
Renderers = Mapping[str, Callable[[list[Any]], None]]

EMPTY_HINT = (
    "No starred teams yet. Add one with `sports-ml starred add --sport hockey --team COL` "
    "(sports: hockey, soccer, baseball, football)."
)


def _artifact_dir(settings: Settings, sport: str) -> Path:
    path = settings.paths.artifacts_dir / sport / "latest"
    if not (path / "metadata.json").exists():
        raise ArtifactsMissingError(
            f"No trained {sport} bundle at '{path}'. Run `sports-ml train --sport {sport}` first."
        )
    return path


class _SilverLocator:
    """Finds a prediction's kickoff and state in a Silver games/matches table (read lazily)."""

    def __init__(
        self,
        path: Path,
        *,
        id_attr: str,
        id_col: str,
        date_attr: str,
        date_col: str,
        event_seq: bool = False,
    ) -> None:
        self.path = path
        self.id_attr, self.id_col = id_attr, id_col
        self.date_attr, self.date_col = date_attr, date_col
        self.event_seq = event_seq
        self._by_id: dict[Any, tuple[datetime, bool]] | None = None
        self._by_key: dict[tuple[Any, ...], tuple[datetime, bool]] = {}

    def _load(self) -> None:
        self._by_id = {}
        if not self.path.exists():
            return
        df = pl.read_parquet(self.path)
        cols = [
            self.id_col,
            self.date_col,
            "home_team",
            "away_team",
            "start_time_utc",
            "game_state",
        ]
        if self.event_seq and "event_seq" in df.columns:
            cols.append("event_seq")
        for r in df.select([c for c in cols if c in df.columns]).iter_rows(named=True):
            slot = (r["start_time_utc"], str(r["game_state"]).upper() == "FINAL")
            self._by_id[r[self.id_col]] = slot
            key: tuple[Any, ...] = (r[self.date_col], r["home_team"], r["away_team"])
            if self.event_seq:
                key += (r.get("event_seq") or 1,)
            self._by_key[key] = slot

    def __call__(self, pred: Any) -> GameSlot:
        if self._by_id is None:
            self._load()
        assert self._by_id is not None
        found = None
        pid = getattr(pred, self.id_attr, None)
        if pid is not None:
            found = self._by_id.get(pid)
        game_date: date = getattr(pred, self.date_attr)
        if found is None:
            key: tuple[Any, ...] = (game_date, pred.home_team, pred.away_team)
            if self.event_seq:
                key += (getattr(pred, "event_seq", 1) or 1,)
            found = self._by_key.get(key)
        if found is None:
            return GameSlot(datetime.combine(game_date, time(), tzinfo=UTC), False)
        return GameSlot(found[0], found[1])


def build_starred_adapters(
    settings: Settings,
    *,
    include_players: bool,
    offline: bool,
    sync: SyncHooks,
) -> dict[str, StarredSportAdapter]:
    """Adapters for every supported sport. `sync[sport]` runs the live refresh (skipped offline)."""

    def hook(sport: str) -> Callable[[], None]:
        def run() -> None:
            if not offline:
                sync[sport]()

        return run

    silver = settings.paths.silver_dir

    def hockey_load() -> HockeyMatchupPredictor:
        return HockeyMatchupPredictor.from_artifacts(
            _artifact_dir(settings, "hockey"), settings=settings
        )

    def hockey_predict(p: HockeyMatchupPredictor, e: StarredTeam, n: int) -> list[Any]:
        return list(
            p.predict_team_next_games(
                e.team,
                n_games=n,
                from_date=None,
                include_players=include_players,
                offline=offline,
                persist_ledger=True,
            )
        )

    def soccer_load() -> SoccerMatchupPredictor:
        return SoccerMatchupPredictor.from_artifacts(
            _artifact_dir(settings, "soccer"), settings=settings
        )

    def soccer_predict(p: SoccerMatchupPredictor, e: StarredTeam, n: int) -> list[Any]:
        return list(
            p.predict_team_next_games(
                e.team,
                n_games=n,
                from_date=None,
                include_players=include_players,
                offline=offline,
                persist_ledger=True,
            )
        )

    def baseball_load() -> BaseballMatchupPredictor:
        return BaseballMatchupPredictor.from_artifacts(
            _artifact_dir(settings, "baseball"), settings=settings
        )

    def baseball_predict(p: BaseballMatchupPredictor, e: StarredTeam, n: int) -> list[Any]:
        return list(
            p.predict_team_next_games(
                e.team,
                n_games=n,
                from_date=None,
                include_players=include_players,
                offline=offline,
                persist_ledger=True,
            )
        )

    def football_load() -> FootballMatchupPredictor:
        return FootballMatchupPredictor.from_artifacts(
            _artifact_dir(settings, "football"), settings=settings
        )

    def football_predict(p: FootballMatchupPredictor, e: StarredTeam, n: int) -> list[Any]:
        return list(
            p.predict_team_next_games(
                e.team,
                n_games=n,
                from_date=None,
                include_players=include_players,
                persist_ledger=True,
            )
        )

    def football_locate(pred: Any) -> GameSlot:
        return GameSlot(pred.start_time_utc, str(pred.game_state).upper() == "FINAL")

    return {
        "hockey": StarredSportAdapter(
            "hockey",
            hook("hockey"),
            hockey_load,
            hockey_predict,
            _SilverLocator(
                silver / "hockey" / "games.parquet",
                id_attr="game_id",
                id_col="game_id",
                date_attr="game_date",
                date_col="game_date",
            ),
        ),
        "soccer": StarredSportAdapter(
            "soccer",
            hook("soccer"),
            soccer_load,
            soccer_predict,
            _SilverLocator(
                silver / "soccer" / "matches.parquet",
                id_attr="match_id",
                id_col="match_id",
                date_attr="match_date",
                date_col="match_date",
            ),
        ),
        "baseball": StarredSportAdapter(
            "baseball",
            hook("baseball"),
            baseball_load,
            baseball_predict,
            _SilverLocator(
                silver / "baseball" / "games.parquet",
                id_attr="game_id",
                id_col="game_id",
                date_attr="game_date",
                date_col="game_date",
                event_seq=True,
            ),
        ),
        "football": StarredSportAdapter(
            "football", hook("football"), football_load, football_predict, football_locate
        ),
    }


def render_starred_overview(console: Console, result: StarredRunResult) -> None:
    """One chronological table of upcoming starred games (kickoff shown in local time)."""
    for warning in result.warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")
    if result.predictions:
        table = Table(title="Starred Teams — Upcoming Games (chronological, local time)")
        for col in ("Kickoff", "Sport / League", "Team", "Matchup"):
            table.add_column(col, no_wrap=True)
        for p in result.predictions:
            local = p.kickoff_utc.astimezone()
            label = p.entry.label or p.entry.team
            table.add_row(
                local.strftime("%Y-%m-%d %H:%M %Z"),
                f"{p.entry.sport} / {p.entry.league}",
                label,
                f"{p.prediction.away_team} @ {p.prediction.home_team}",
            )
        console.print(table)
    for skip in result.skips:
        detail = f" — {skip.detail}" if skip.detail else ""
        console.print(
            f"[red]SKIPPED[/red] {skip.entry.sport}/{skip.entry.league} {skip.entry.team}: "
            f"[bold]{skip.reason.value}[/bold]{detail}"
        )


def run_starred_command(
    console: Console,
    settings: Settings,
    config: StarredTeamsConfig,
    *,
    n_games: int,
    include_players: bool,
    offline: bool,
    sync: SyncHooks,
    renderers: Renderers,
) -> int:
    """Run `predict --starred`; returns the process exit code (non-zero only if all failed)."""
    if not config.teams:
        console.print(EMPTY_HINT)
        return 0
    adapters = build_starred_adapters(
        settings, include_players=include_players, offline=offline, sync=sync
    )
    result = run_starred_predictions(config.teams, adapters, n_games=n_games)
    render_starred_overview(console, result)
    for sport, group in result.by_sport().items():
        console.print(f"\n[bold]{sport.upper()}[/bold]")
        renderers[sport]([p.prediction for p in group])
    return 0 if result.succeeded else 1


__all__ = [
    "EMPTY_HINT",
    "build_starred_adapters",
    "render_starred_overview",
    "run_starred_command",
]
