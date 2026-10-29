"""End-to-end offline CLI integration test for `--sport soccer` (`test_soccer_cli_e2e.py`)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sports_ml.cli import app

runner = CliRunner()


def _write_offline_soccer_fixtures(raw_dir: Path) -> None:
    """Populate `data/raw/` with deterministic offline FPL, Understat, and Football-Data files."""
    fpl_dir = raw_dir / "fpl_api"
    summaries_dir = fpl_dir / "element_summaries"
    us_dir = raw_dir / "understat"
    fd_dir = raw_dir / "football_data"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    us_dir.mkdir(parents=True, exist_ok=True)
    fd_dir.mkdir(parents=True, exist_ok=True)

    bootstrap: dict[str, list[dict[str, Any]]] = {
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "name": "Man City", "short_name": "MCI"},
            {"id": 3, "name": "Liverpool", "short_name": "LIV"},
            {"id": 4, "name": "Chelsea", "short_name": "CHE"},
        ],
        "elements": [
            {
                "id": 101,
                "first_name": "David",
                "second_name": "Raya",
                "web_name": "Raya",
                "team": 1,
                "element_type": 1,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "minutes": 1080,
                "starts": 12,
            },
            {
                "id": 102,
                "first_name": "Bukayo",
                "second_name": "Saka",
                "web_name": "Saka",
                "team": 1,
                "element_type": 3,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "penalties_order": 1,
                "corners_and_indirect_freekicks_order": 1,
                "minutes": 1050,
                "starts": 12,
            },
            {
                "id": 103,
                "first_name": "Kai",
                "second_name": "Havertz",
                "web_name": "Havertz",
                "team": 1,
                "element_type": 4,
                "status": "d",
                "chance_of_playing_next_round": 75,
                "minutes": 1020,
                "starts": 12,
            },
            {
                "id": 104,
                "first_name": "William",
                "second_name": "Saliba",
                "web_name": "Saliba",
                "team": 1,
                "element_type": 2,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "minutes": 1080,
                "starts": 12,
            },
            {
                "id": 201,
                "first_name": "Ederson",
                "second_name": "Moraes",
                "web_name": "Ederson",
                "team": 2,
                "element_type": 1,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "minutes": 1080,
                "starts": 12,
            },
            {
                "id": 202,
                "first_name": "Erling",
                "second_name": "Haaland",
                "web_name": "Haaland",
                "team": 2,
                "element_type": 4,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "penalties_order": 1,
                "minutes": 1080,
                "starts": 12,
            },
            {
                "id": 203,
                "first_name": "Kevin",
                "second_name": "De Bruyne",
                "web_name": "De Bruyne",
                "team": 2,
                "element_type": 3,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "corners_and_indirect_freekicks_order": 1,
                "minutes": 950,
                "starts": 11,
            },
            {
                "id": 204,
                "first_name": "Ruben",
                "second_name": "Dias",
                "web_name": "Dias",
                "team": 2,
                "element_type": 2,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "minutes": 1020,
                "starts": 11,
            },
        ],
    }
    (fpl_dir / "bootstrap_static.json").write_text(json.dumps(bootstrap), encoding="utf-8")

    # Build 12 completed matches across 12 distinct dates + 1 upcoming match
    base_dt = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
    pairings = [
        (1, 2, 2, 1),
        (3, 4, 1, 1),
        (2, 3, 3, 1),
        (4, 1, 0, 2),
        (1, 3, 2, 2),
        (2, 4, 2, 0),
        (2, 1, 1, 1),
        (4, 3, 1, 2),
        (3, 2, 2, 1),
        (1, 4, 3, 0),
        (3, 1, 1, 2),
        (4, 2, 0, 3),
    ]
    fixtures = []
    for idx, (th, ta, sh, sa) in enumerate(pairings, start=1):
        dt = base_dt + timedelta(days=idx * 3)
        fixtures.append(
            {
                "id": idx,
                "event": idx,
                "kickoff_time": dt.isoformat().replace("+00:00", "Z"),
                "team_h": th,
                "team_a": ta,
                "team_h_score": sh,
                "team_a_score": sa,
                "finished": True,
            }
        )
    # Add two upcoming fixtures on 2024-10-05 and 2024-10-19
    fixtures.append(
        {
            "id": 13,
            "event": 13,
            "kickoff_time": "2024-10-05T14:00:00Z",
            "team_h": 1,
            "team_a": 2,
            "team_h_score": None,
            "team_a_score": None,
            "finished": False,
        }
    )
    fixtures.append(
        {
            "id": 14,
            "event": 14,
            "kickoff_time": "2024-10-19T14:00:00Z",
            "team_h": 3,
            "team_a": 1,
            "team_h_score": None,
            "team_a_score": None,
            "finished": False,
        }
    )
    (fpl_dir / "fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")

    # Write element summaries for ARS and MCI players across their completed matches
    for el in bootstrap["elements"]:
        pid = int(el["id"])
        t_id = int(el["team"])
        el_type = int(el["element_type"])
        hist = []
        for fx in fixtures:
            if not fx["finished"]:
                continue
            if fx["team_h"] != t_id and fx["team_a"] != t_id:
                continue
            was_h = fx["team_h"] == t_id
            opp_id = fx["team_a"] if was_h else fx["team_h"]
            if el_type == 1:
                hist.append(
                    {
                        "fixture": fx["id"],
                        "opponent_team": opp_id,
                        "was_home": was_h,
                        "kickoff_time": fx["kickoff_time"],
                        "minutes": 90,
                        "starts": 1,
                        "saves": 3,
                        "goals_conceded": fx["team_a_score"] if was_h else fx["team_h_score"],
                        "clean_sheets": 1
                        if (fx["team_a_score"] if was_h else fx["team_h_score"]) == 0
                        else 0,
                        "expected_goals_conceded": "1.20",
                    }
                )
            else:
                hist.append(
                    {
                        "fixture": fx["id"],
                        "opponent_team": opp_id,
                        "was_home": was_h,
                        "kickoff_time": fx["kickoff_time"],
                        "minutes": 88,
                        "starts": 1,
                        "goals_scored": 1 if el_type in (3, 4) else 0,
                        "assists": 1 if el_type == 3 else 0,
                        "expected_goals": "0.45" if el_type in (3, 4) else "0.08",
                        "expected_assists": "0.28" if el_type == 3 else "0.08",
                        "expected_goal_involvements": "0.73" if el_type in (3, 4) else "0.16",
                        "yellow_cards": 0,
                        "red_cards": 0,
                    }
                )
        (summaries_dir / f"{pid}.json").write_text(json.dumps({"history": hist}), encoding="utf-8")


def test_soccer_cli_end_to_end_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ingest -> features -> train -> backtest -> predict -> score-predictions for soccer."""
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))
    _write_offline_soccer_fixtures(tmp_path / "data" / "raw")

    # 1. Ingest offline
    r_ingest = runner.invoke(app, ["ingest", "--sport", "soccer", "--offline"])
    assert r_ingest.exit_code == 0, r_ingest.output
    assert "Ingestion complete" in r_ingest.output

    # 2. Features build
    r_feat = runner.invoke(app, ["features", "build", "--sport", "soccer"])
    assert r_feat.exit_code == 0, r_feat.output
    assert "Gold features built" in r_feat.output

    # 3. Train Stage 1 + Stage 2
    r_train = runner.invoke(app, ["train", "--sport", "soccer", "--include-stage2"])
    assert r_train.exit_code == 0, r_train.output
    assert "Trained & saved soccer model bundle" in r_train.output

    # 4. Backtest Stage 1 + Stage 2
    r_bt = runner.invoke(app, ["backtest", "--sport", "soccer", "--stage", "all", "--splits", "2"])
    assert r_bt.exit_code == 0, r_bt.output
    assert "Walk-Forward Backtest (SOCCER 1X2" in r_bt.output

    # 5. Predict Stage 1 only (on a completed match date so score-predictions can evaluate it)
    r_pred_s1 = runner.invoke(
        app,
        [
            "predict",
            "--sport",
            "soccer",
            "--date",
            "2024-08-20",
            "--matchup",
            "MCI@ARS",
            "--offline",
        ],
    )
    assert r_pred_s1.exit_code == 0, r_pred_s1.output
    assert "Stage 1 — Calibrated EPL 3-Way (1X2)" in r_pred_s1.output
    assert "Date" in r_pred_s1.output
    assert "Matchup" in r_pred_s1.output
    assert "P(Away)" in r_pred_s1.output
    assert "P(Draw)" in r_pred_s1.output
    assert "P(Home)" in r_pred_s1.output
    assert "Mkt 1X2 (Edge)" not in r_pred_s1.output
    assert "E[G] A/H" not in r_pred_s1.output
    assert "Lineup Status" not in r_pred_s1.output

    # 6. Predict Stage 1 + Stage 2 (--include-players)
    r_pred_s2 = runner.invoke(
        app,
        [
            "predict",
            "--sport",
            "soccer",
            "--date",
            "2024-10-05",
            "--matchup",
            "MCI@ARS",
            "--include-players",
            "--offline",
        ],
    )
    assert r_pred_s2.exit_code == 0, r_pred_s2.output
    assert "Stage 2 — EPL Player Projections" in r_pred_s2.output
    assert "Goalkeepers" in r_pred_s2.output

    # 7. Predict next game and next N games for a team (--team ARS, --games 2)
    r_team_1 = runner.invoke(
        app,
        ["predict", "--sport", "soccer", "--team", "ARS", "--offline"],
    )
    assert r_team_1.exit_code == 0, r_team_1.output
    assert "2024-10-05" in r_team_1.output
    assert "MCI @ ARS" in r_team_1.output
    assert "2024-10-19" not in r_team_1.output

    r_team_2 = runner.invoke(
        app,
        ["predict", "--sport", "soccer", "--team", "ARS", "--games", "2", "--offline"],
    )
    assert r_team_2.exit_code == 0, r_team_2.output
    assert "2024-10-05" in r_team_2.output
    assert "2024-10-19" in r_team_2.output
    assert "ARS @ LIV" in r_team_2.output

    # 8. Score predictions ledger against completed Silver matches
    r_score = runner.invoke(app, ["score-predictions", "--sport", "soccer"])
    assert r_score.exit_code == 0, r_score.output
    assert "Live Prediction Ledger Accuracy (SOCCER 1X2)" in r_score.output
    assert "3-Class Log Loss" in r_score.output
