"""Tests for the starred-teams store and orchestrator (`core/starred.py`)."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from sports_ml.core.starred import (
    ArtifactsMissingError,
    GameSlot,
    StarredConfigError,
    StarredSkipReason,
    StarredSportAdapter,
    StarredTeam,
    StarredTeamsConfig,
    registered_sports,
    resolve_team,
    run_starred_predictions,
    starred_teams_path,
)
from sports_ml.sports.starred import register_default_team_resolvers

T0 = datetime(2026, 1, 10, 20, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _resolvers() -> None:
    register_default_team_resolvers()


def test_default_resolvers_cover_every_sport() -> None:
    assert set(registered_sports()) >= {"hockey", "soccer", "baseball", "football"}


def test_add_remove_list_round_trip_through_toml(tmp_path: Path) -> None:
    path = tmp_path / "config" / "starred_teams.toml"
    assert StarredTeamsConfig.load(path).list() == ()

    cfg, col, added = StarredTeamsConfig().add("hockey", "COL", label='The "Avs" \\ crew')
    assert added and (col.sport, col.league, col.team) == ("hockey", "nhl", "COL")
    cfg, lad, _ = cfg.add("baseball", "LAD")
    assert lad.league == "mlb"  # league auto-resolved and stored
    cfg, han, _ = cfg.add("baseball", "HAN")
    assert han.league == "npb"
    cfg, _, _ = cfg.add("soccer", "Arsenal")
    cfg, kc, _ = cfg.add("football", "KC")
    cfg.save(path)

    loaded = StarredTeamsConfig.load(path)
    assert loaded == cfg
    assert [t.key for t in loaded.list()] == [
        ("hockey", "nhl", "COL"),
        ("baseball", "mlb", "LAD"),
        ("baseball", "npb", "HAN"),
        ("soccer", "epl", "ARS"),
        ("football", "nfl", "KC"),
    ]
    assert loaded.list()[0].label == 'The "Avs" \\ crew'

    after, removed = loaded.remove("baseball", "LAD")
    assert [t.key for t in removed] == [("baseball", "mlb", "LAD")]
    assert len(after.list()) == 4
    after.save(path)
    assert len(StarredTeamsConfig.load(path).list()) == 4
    _, nothing = after.remove("hockey", "BOS")
    assert nothing == ()


def test_deduplication_and_cross_sport_collisions() -> None:
    cfg, _, first = StarredTeamsConfig().add("hockey", "PHI")
    cfg, again, second = cfg.add("hockey", "phi", league="nhl")
    assert first and not second and len(cfg.list()) == 1 and again.team == "PHI"
    cfg, baseball_phi, added = cfg.add("baseball", "PHI")  # Phillies share the tricode
    assert added and baseball_phi.sport == "baseball"
    assert {t.key for t in cfg.list()} == {("hockey", "nhl", "PHI"), ("baseball", "mlb", "PHI")}
    cfg, removed = cfg.remove("hockey", "PHI")
    assert [t.key for t in removed] == [("hockey", "nhl", "PHI")]
    assert [t.key for t in cfg.list()] == [("baseball", "mlb", "PHI")]


def test_alias_normalization_and_league_filtered_removal() -> None:
    assert resolve_team("baseball", "OAK").team == "ATH"
    assert resolve_team("baseball", "WSH").team == resolve_team("baseball", "WAS").team
    assert resolve_team("hockey", "WSH").full_name == "Washington Capitals"
    assert resolve_team("football", "OAK").team == "LV"
    cfg, entry, _ = StarredTeamsConfig().add("baseball", "OAK")
    assert entry.team == "ATH"
    _, removed = cfg.remove("baseball", "oakland athletics")
    assert [t.team for t in removed] == ["ATH"]


def test_rejects_unknown_sports_teams_and_league_mismatches() -> None:
    with pytest.raises(StarredConfigError, match="Unknown sport 'cricket'"):
        resolve_team("cricket", "MI")
    with pytest.raises(StarredConfigError, match="Unknown NHL team"):
        resolve_team("hockey", "ZZZ")
    with pytest.raises(StarredConfigError, match="Unknown MLB/NPB club"):
        resolve_team("baseball", "ZZZ")
    with pytest.raises(StarredConfigError, match="plays in NPB"):
        resolve_team("baseball", "HAN", "mlb")
    with pytest.raises(StarredConfigError, match="Unsupported hockey league"):
        resolve_team("hockey", "COL", "khl")
    with pytest.raises(StarredConfigError, match="Unknown EPL club"):
        resolve_team("soccer", "Barcelona")
    with pytest.raises(StarredConfigError, match="cannot be empty"):
        resolve_team("hockey", "  ")


def test_load_validates_schema_and_renormalizes_hand_edited_entries(tmp_path: Path) -> None:
    path = tmp_path / "s.toml"
    path.write_text(
        '[[starred]]\nsport = "baseball"\nteam = "OAK"\n\n'
        '[[starred]]\nsport = "baseball"\nteam = "ATH"\nleague = "mlb"\n',
        encoding="utf-8",
    )
    loaded = StarredTeamsConfig.load(path)
    assert [t.key for t in loaded.list()] == [
        ("baseball", "mlb", "ATH")
    ]  # alias + duplicate merged

    path.write_text('[[starred]]\nsport = "curling"\nteam = "X"\n', encoding="utf-8")
    with pytest.raises(StarredConfigError, match="Unknown sport 'curling'"):
        StarredTeamsConfig.load(path)
    path.write_text('[[starred]]\nteam = "COL"\n', encoding="utf-8")
    with pytest.raises(StarredConfigError, match="Entry #1"):
        StarredTeamsConfig.load(path)
    path.write_text("this is = not [valid", encoding="utf-8")
    with pytest.raises(StarredConfigError, match="Invalid TOML"):
        StarredTeamsConfig.load(path)
    path.write_text("starred = 3\n", encoding="utf-8")
    with pytest.raises(StarredConfigError, match="array of tables"):
        StarredTeamsConfig.load(path)


def test_save_is_atomic_and_never_leaves_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "starred.toml"
    original, _, _ = StarredTeamsConfig().add("hockey", "COL")
    original.save(path)
    before = path.read_text(encoding="utf-8")

    bigger, _, _ = original.add("soccer", "ARS")

    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        bigger.save(path)
    monkeypatch.undo()
    assert path.read_text(encoding="utf-8") == before  # old file untouched
    assert [p.name for p in tmp_path.iterdir()] == ["starred.toml"]  # temp file cleaned up

    fresh = tmp_path / "nested" / "dir" / "new.toml"
    bigger.save(fresh)
    assert StarredTeamsConfig.load(fresh) == bigger


def test_path_resolution_honors_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SPORTS_ML_STARRED_TEAMS_FILE", raising=False)
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path / "root"))
    assert starred_teams_path() == (tmp_path / "root").resolve() / "config" / "starred_teams.toml"
    explicit = tmp_path / "elsewhere" / "mine.toml"
    monkeypatch.setenv("SPORTS_ML_STARRED_TEAMS_FILE", str(explicit))
    assert starred_teams_path() == explicit.resolve()  # the file override wins over the root
    monkeypatch.delenv("SPORTS_ML_STARRED_TEAMS_FILE")
    monkeypatch.delenv("SPORTS_ML_ROOT_DIR")
    assert starred_teams_path(tmp_path) == tmp_path / "config" / "starred_teams.toml"


def test_repo_ships_an_example_file_and_ignores_the_personal_one() -> None:
    root = Path(__file__).resolve().parents[2]
    example = root / "config" / "starred_teams.example.toml"
    loaded = StarredTeamsConfig.load(example)
    assert {t.sport for t in loaded.list()} == {"hockey", "soccer", "baseball", "football"}
    assert "config/starred_teams.toml" in (root / ".gitignore").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- orchestrator


class _Pred:
    def __init__(self, tag: str, kickoff: datetime, final: bool = False) -> None:
        self.tag, self.kickoff, self.final = tag, kickoff, final
        self.home_team, self.away_team = "H", "A"


def _adapter(
    sport: str,
    games: dict[str, list[_Pred]],
    log: list[str],
    *,
    load_error: Exception | None = None,
    sync_error: Exception | None = None,
) -> StarredSportAdapter:
    def sync() -> None:
        log.append(f"sync:{sport}")
        if sync_error is not None:
            raise sync_error

    def load() -> object:
        log.append(f"load:{sport}")
        if load_error is not None:
            raise load_error
        return object()

    def predict(_p: Any, entry: StarredTeam, n: int) -> list[_Pred]:
        log.append(f"predict:{sport}:{entry.team}")
        value: Any = games[entry.team]
        if isinstance(value, Exception):
            raise value
        return list(value)[:n]

    return StarredSportAdapter(sport, sync, load, predict, lambda p: GameSlot(p.kickoff, p.final))


def _entries(*specs: tuple[str, str]) -> list[StarredTeam]:
    return [
        StarredTeam(sport=s, league={"hockey": "nhl", "soccer": "epl"}[s], team=t) for s, t in specs
    ]


def test_orchestrator_syncs_once_per_sport_and_sorts_chronologically() -> None:
    log: list[str] = []
    hockey = _adapter(
        "hockey",
        {
            "COL": [_Pred("col", T0 + timedelta(hours=5))],
            "BOS": [_Pred("bos", T0 + timedelta(hours=1))],
        },
        log,
    )
    soccer = _adapter("soccer", {"ARS": [_Pred("ars", T0 + timedelta(hours=3))]}, log)
    result = run_starred_predictions(
        _entries(("hockey", "COL"), ("soccer", "ARS"), ("hockey", "BOS")),
        {"hockey": hockey, "soccer": soccer},
    )
    assert [p.prediction.tag for p in result.predictions] == ["bos", "ars", "col"]
    assert log.count("sync:hockey") == 1 and log.count("sync:soccer") == 1
    assert log.count("load:hockey") == 1 and log.count("load:soccer") == 1
    assert [t for t in log if t.startswith("predict:hockey")] == [
        "predict:hockey:COL",
        "predict:hockey:BOS",
    ]
    assert list(result.by_sport()) == ["hockey", "soccer"]  # ordered by first kickoff
    assert result.succeeded and not result.skips


def test_orchestrator_partial_failures_become_skips() -> None:
    log: list[str] = []
    hockey = _adapter(
        "hockey",
        {
            "COL": [_Pred("col", T0)],
            "BOS": [_Pred("done", T0 - timedelta(days=2), final=True)],  # only finished games
            "TOR": [],
            "MTL": ValueError("No Silver games for team"),  # type: ignore[dict-item]
        },
        log,
    )
    soccer = _adapter("soccer", {"ARS": []}, log, load_error=ArtifactsMissingError("no bundle"))
    result = run_starred_predictions(
        _entries(
            ("hockey", "COL"),
            ("hockey", "BOS"),
            ("hockey", "TOR"),
            ("hockey", "MTL"),
            ("soccer", "ARS"),
        ),
        {"hockey": hockey, "soccer": soccer},
    )
    assert [p.entry.team for p in result.predictions] == ["COL"]
    reasons = {s.entry.team: s.reason for s in result.skips}
    assert reasons == {
        "BOS": StarredSkipReason.NO_UPCOMING_GAME,
        "TOR": StarredSkipReason.NO_UPCOMING_GAME,
        "MTL": StarredSkipReason.SOURCE_UNAVAILABLE,
        "ARS": StarredSkipReason.NO_TRAINED_ARTIFACTS,
    }
    assert "predict:soccer:ARS" not in log  # no predictor, nothing attempted


def test_orchestrator_missing_adapter_file_not_found_and_sync_failures() -> None:
    log: list[str] = []
    cached = _adapter(
        "hockey", {"COL": [_Pred("col", T0)]}, log, sync_error=ConnectionError("offline")
    )
    result = run_starred_predictions(
        _entries(("hockey", "COL"), ("soccer", "ARS")), {"hockey": cached}
    )
    assert result.succeeded  # a failed live sync falls back to cached tables
    assert any("hockey: live sync failed (offline)" in w for w in result.warnings)
    assert [(s.entry.team, s.reason) for s in result.skips] == [
        ("ARS", StarredSkipReason.SOURCE_UNAVAILABLE)
    ]

    missing = _adapter("soccer", {}, [], load_error=FileNotFoundError("Model bundle not found"))
    only = run_starred_predictions(_entries(("soccer", "ARS")), {"soccer": missing})
    assert not only.succeeded
    assert only.skips[0].reason is StarredSkipReason.NO_TRAINED_ARTIFACTS
    assert "Model bundle not found" in only.skips[0].detail

    broken = _adapter("soccer", {}, [], load_error=RuntimeError("corrupt pickle"))
    assert (
        run_starred_predictions(_entries(("soccer", "ARS")), {"soccer": broken}).skips[0].reason
        is StarredSkipReason.SOURCE_UNAVAILABLE
    )


def test_orchestrator_normalizes_naive_kickoffs_to_utc() -> None:
    adapter = _adapter("hockey", {"COL": [_Pred("col", datetime(2026, 1, 1, 12, 0))]}, [])
    result = run_starred_predictions(_entries(("hockey", "COL")), {"hockey": adapter})
    assert result.predictions[0].kickoff_utc == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_preflight_skips_before_sync_and_events_report_progress() -> None:
    log: list[str] = []
    events: list[str] = []

    def no_model() -> None:
        raise ArtifactsMissingError("no bundle")

    guarded = _adapter("soccer", {"ARS": [_Pred("ars", T0)]}, log)
    guarded = StarredSportAdapter(
        guarded.sport,
        guarded.sync,
        guarded.load_predictor,
        guarded.predict,
        guarded.locate,
        no_model,
    )
    hockey = _adapter("hockey", {"COL": [_Pred("col", T0)]}, log)
    result = run_starred_predictions(
        _entries(("soccer", "ARS"), ("hockey", "COL")),
        {"soccer": guarded, "hockey": hockey},
        on_event=events.append,
    )
    assert "sync:soccer" not in log and "load:soccer" not in log  # nothing touched
    assert result.skips[0].reason is StarredSkipReason.NO_TRAINED_ARTIFACTS
    assert events[0] == "soccer: skipped (NO_TRAINED_ARTIFACTS)"
    assert any(e.startswith("hockey: sync done") for e in events)
    assert any(e.startswith("hockey COL: predicted in") for e in events)
