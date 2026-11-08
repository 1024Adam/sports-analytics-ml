"""Starred teams: a personal cross-sport watch list and its prediction orchestrator.

Entries are keyed by `(sport, league, team)` so tricodes that collide across sports (`PHI`,
`BOS`, `LAC`, ...) coexist. This module is sport-agnostic: team validation and normalization are
supplied by per-sport `TeamResolver` implementations registered through `register_team_resolver`
(see `sports_ml/sports/starred.py`), and prediction is delegated to per-sport
`StarredSportAdapter` callables that wrap each plugin's existing `predict_team_next_games`. It adds
no modeling and changes no prediction output.

The list lives in `config/starred_teams.toml` (git-ignored). The path honors
`SPORTS_ML_STARRED_TEAMS_FILE`, then `SPORTS_ML_ROOT_DIR`. A missing file is an empty list, and
saves are atomic (write to a sibling temp file, then `os.replace`).
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from sports_ml.config import get_settings

STARRED_FILE_ENV = "SPORTS_ML_STARRED_TEAMS_FILE"
DEFAULT_RELATIVE_PATH = Path("config") / "starred_teams.toml"


class StarredConfigError(ValueError):
    """Invalid sport/team/league, or a malformed starred-teams file."""


class ResolvedTeam(BaseModel):
    """A team after sport-specific validation/normalization."""

    model_config = ConfigDict(frozen=True)

    sport: str
    league: str
    team: str
    full_name: str


class TeamResolver(Protocol):
    """Per-sport team validation. Implementations raise `StarredConfigError` for unknown teams."""

    def resolve(self, team: str, league: str | None) -> ResolvedTeam: ...


_RESOLVERS: dict[str, TeamResolver] = {}


def register_team_resolver(sport: str, resolver: TeamResolver) -> None:
    """Register (or replace) the `TeamResolver` for `sport`."""
    _RESOLVERS[sport.strip().lower()] = resolver


def registered_sports() -> tuple[str, ...]:
    """Sports that currently have a registered resolver."""
    return tuple(sorted(_RESOLVERS))


def resolve_team(sport: str, team: str, league: str | None = None) -> ResolvedTeam:
    """Resolve `(sport, team[, league])` through the registered resolver."""
    key = sport.strip().lower()
    resolver = _RESOLVERS.get(key)
    if resolver is None:
        supported = ", ".join(registered_sports()) or "none"
        raise StarredConfigError(f"Unknown sport '{sport}'. Supported sports: {supported}.")
    if not team.strip():
        raise StarredConfigError("Team code cannot be empty.")
    return resolver.resolve(team, league.strip().lower() if league else None)


class StarredTeam(BaseModel):
    """One starred team, keyed by `(sport, league, team)`."""

    model_config = ConfigDict(frozen=True)

    sport: str
    league: str
    team: str
    label: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.sport, self.league, self.team)


def starred_teams_path(root_dir: Path | None = None) -> Path:
    """Resolve the starred-teams file: env override, else `<root>/config/starred_teams.toml`."""
    override = os.environ.get(STARRED_FILE_ENV)
    if override:
        return Path(override).expanduser().resolve()
    root = root_dir if root_dir is not None else get_settings().paths.root_dir
    return root / DEFAULT_RELATIVE_PATH


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class StarredTeamsConfig(BaseModel):
    """Ordered, de-duplicated starred teams with load/save/add/remove/list."""

    model_config = ConfigDict(frozen=True)

    teams: tuple[StarredTeam, ...] = ()

    def list(self) -> tuple[StarredTeam, ...]:
        return self.teams

    def add(
        self,
        sport: str,
        team: str,
        league: str | None = None,
        label: str | None = None,
    ) -> tuple[StarredTeamsConfig, StarredTeam, bool]:
        """Return `(new_config, entry, added)`; re-adding an existing key is a no-op."""
        resolved = resolve_team(sport, team, league)
        entry = StarredTeam(
            sport=resolved.sport, league=resolved.league, team=resolved.team, label=label
        )
        existing = next((t for t in self.teams if t.key == entry.key), None)
        if existing is not None:
            return self, existing, False
        return StarredTeamsConfig(teams=(*self.teams, entry)), entry, True

    def remove(
        self, sport: str, team: str, league: str | None = None
    ) -> tuple[StarredTeamsConfig, tuple[StarredTeam, ...]]:
        """Return `(new_config, removed_entries)`; `league=None` matches any league."""
        resolved = resolve_team(sport, team, league)
        match_league = league is not None
        removed = tuple(
            t
            for t in self.teams
            if t.sport == resolved.sport
            and t.team == resolved.team
            and (not match_league or t.league == resolved.league)
        )
        keep = tuple(t for t in self.teams if t not in removed)
        return StarredTeamsConfig(teams=keep), removed

    @classmethod
    def load(cls, path: Path) -> StarredTeamsConfig:
        """Load and validate `path`; a missing file is an empty list."""
        if not path.exists():
            return cls()
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise StarredConfigError(f"Invalid TOML in '{path}': {exc}") from exc
        rows = raw.get("starred", [])
        if not isinstance(rows, list):
            raise StarredConfigError(f"'starred' in '{path}' must be an array of tables.")
        config = cls()
        for i, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or "sport" not in row or "team" not in row:
                raise StarredConfigError(
                    f"Entry #{i} in '{path}' needs at least `sport` and `team`."
                )
            label = row.get("label")
            config, _, _ = config.add(
                str(row["sport"]),
                str(row["team"]),
                str(row["league"]) if row.get("league") else None,
                str(label) if label else None,
            )
        return config

    def to_toml(self) -> str:
        blocks = []
        for t in self.teams:
            lines = [
                "[[starred]]",
                f"sport = {_toml_string(t.sport)}",
                f"league = {_toml_string(t.league)}",
                f"team = {_toml_string(t.team)}",
            ]
            if t.label:
                lines.append(f"label = {_toml_string(t.label)}")
            blocks.append("\n".join(lines))
        header = "# Managed by `sports-ml starred`; safe to edit by hand.\n"
        return header + ("\n\n".join(blocks) + "\n" if blocks else "")

    def save(self, path: Path) -> None:
        """Atomically write the list to `path` (the old file survives any failure)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self.to_toml())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


# ----------------------------------------------------------------------------- orchestration


class StarredSkipReason(StrEnum):
    """Why a starred team produced no prediction."""

    NO_UPCOMING_GAME = "NO_UPCOMING_GAME"
    NO_TRAINED_ARTIFACTS = "NO_TRAINED_ARTIFACTS"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"


class ArtifactsMissingError(RuntimeError):
    """Raised by an adapter's `load_predictor` when no trained bundle exists."""


@dataclass(frozen=True)
class GameSlot:
    """Where a prediction sits in time: UTC kickoff and whether the game is already final."""

    kickoff_utc: datetime
    is_final: bool = False


@dataclass(frozen=True)
class StarredSportAdapter:
    """Per-sport hooks over an existing plugin predictor (no new modeling).

    `sync` runs at most once per sport per run, `load_predictor` once per sport, and `predict`
    once per starred team (it must call the plugin's `predict_team_next_games`).
    """

    sport: str
    sync: Callable[[], None]
    load_predictor: Callable[[], Any]
    predict: Callable[[Any, StarredTeam, int], Sequence[Any]]
    locate: Callable[[Any], GameSlot]


@dataclass(frozen=True)
class StarredPrediction:
    entry: StarredTeam
    prediction: Any
    kickoff_utc: datetime


@dataclass(frozen=True)
class StarredSkip:
    entry: StarredTeam
    reason: StarredSkipReason
    detail: str = ""


@dataclass
class StarredRunResult:
    predictions: list[StarredPrediction] = field(default_factory=list)
    skips: list[StarredSkip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return bool(self.predictions)

    def by_sport(self) -> dict[str, list[StarredPrediction]]:
        """Predictions per sport (chronological), sports ordered by their first kickoff."""
        groups: dict[str, list[StarredPrediction]] = {}
        for p in self.predictions:
            groups.setdefault(p.entry.sport, []).append(p)
        return dict(sorted(groups.items(), key=lambda kv: kv[1][0].kickoff_utc))


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def run_starred_predictions(
    entries: Iterable[StarredTeam],
    adapters: Mapping[str, StarredSportAdapter],
    *,
    n_games: int = 1,
) -> StarredRunResult:
    """Predict each starred team's next `n_games`, never aborting on a per-team failure.

    Entries are grouped by sport: each sport is synced once and gets one predictor. A sync error
    is recorded as a warning and the cached Silver tables are used. Per-team failures become
    `StarredSkip` rows; finished games are not "upcoming" and are dropped. Predictions are
    returned sorted by kickoff time.
    """
    result = StarredRunResult()
    by_sport: dict[str, list[StarredTeam]] = {}
    for entry in entries:
        by_sport.setdefault(entry.sport, []).append(entry)

    for sport, teams in by_sport.items():
        adapter = adapters.get(sport)
        if adapter is None:
            result.skips.extend(
                StarredSkip(t, StarredSkipReason.SOURCE_UNAVAILABLE, f"no adapter for '{sport}'")
                for t in teams
            )
            continue
        try:
            adapter.sync()
        except Exception as exc:
            result.warnings.append(f"{sport}: live sync failed ({exc}); using cached data.")
        try:
            predictor = adapter.load_predictor()
        except (ArtifactsMissingError, FileNotFoundError) as exc:
            result.skips.extend(
                StarredSkip(t, StarredSkipReason.NO_TRAINED_ARTIFACTS, str(exc)) for t in teams
            )
            continue
        except Exception as exc:
            result.skips.extend(
                StarredSkip(t, StarredSkipReason.SOURCE_UNAVAILABLE, str(exc)) for t in teams
            )
            continue

        for entry in teams:
            try:
                predictions = list(adapter.predict(predictor, entry, n_games))
            except Exception as exc:
                result.skips.append(
                    StarredSkip(entry, StarredSkipReason.SOURCE_UNAVAILABLE, str(exc))
                )
                continue
            upcoming = []
            for pred in predictions:
                slot = adapter.locate(pred)
                if not slot.is_final:
                    upcoming.append(StarredPrediction(entry, pred, _as_utc(slot.kickoff_utc)))
            if not upcoming:
                result.skips.append(
                    StarredSkip(entry, StarredSkipReason.NO_UPCOMING_GAME, "no scheduled game")
                )
                continue
            result.predictions.extend(upcoming)

    result.predictions.sort(key=lambda p: (p.kickoff_utc, p.entry.sport, p.entry.team))
    return result


__all__ = [
    "ArtifactsMissingError",
    "GameSlot",
    "ResolvedTeam",
    "StarredConfigError",
    "StarredPrediction",
    "StarredRunResult",
    "StarredSkip",
    "StarredSkipReason",
    "StarredSportAdapter",
    "StarredTeam",
    "StarredTeamsConfig",
    "TeamResolver",
    "register_team_resolver",
    "registered_sports",
    "resolve_team",
    "starred_teams_path",
]
