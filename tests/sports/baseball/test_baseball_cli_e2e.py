"""End-to-end offline CLI integration tests for `--sport baseball` (`MLB` & `NPB`)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from tests.sports.baseball.test_baseball_ingestion import _sample_mlb_boxscore_payload
from typer.testing import CliRunner

from sports_ml.cli import app
from sports_ml.core.cache import write_cached_json

runner = CliRunner()


def _seed_offline_baseball_raw_cache(raw_dir: Path) -> None:
    """Seed deterministic MLB and NPB raw `.json.gz` schedules & boxscores for E2E CLI tests."""
    mlb_sched_dir = raw_dir / "baseball" / "mlb_api" / "schedules"
    mlb_box_dir = raw_dir / "baseball" / "mlb_api" / "boxscores"
    npb_sched_dir = raw_dir / "baseball" / "npb" / "schedules"
    npb_box_dir = raw_dir / "baseball" / "npb" / "boxscores"

    mlb_dates: list[dict[str, Any]] = []
    npb_games: list[dict[str, Any]] = []
    start_d = date(2025, 5, 1)

    for i in range(14):
        d = start_d + timedelta(days=i)
        d_str = d.isoformat()
        gid_mlb = 780000 + i
        is_final = i < 12

        mlb_dates.append(
            {
                "date": d_str,
                "games": [
                    {
                        "gamePk": gid_mlb,
                        "gameNumber": 1,
                        "gameType": "R",
                        "season": "2025",
                        "officialDate": d_str,
                        "gameDate": f"{d_str}T20:10:00Z",
                        "scheduledInnings": 9,
                        "status": {"abstractGameState": "Final" if is_final else "Preview"},
                        "linescore": {"currentInning": 9},
                        "venue": {"name": "Dodger Stadium"},
                        "teams": {
                            "home": {
                                "team": {"abbreviation": "LAD"},
                                "score": 5 if (i % 2 == 0) else 2,
                                "probablePitcher": {"id": 801, "pitchHand": {"code": "R"}},
                            },
                            "away": {
                                "team": {"abbreviation": "SDP"},
                                "score": 2 if (i % 2 == 0) else 5,
                                "probablePitcher": {"id": 802, "pitchHand": {"code": "L"}},
                            },
                        },
                    }
                ],
            }
        )
        if is_final:
            write_cached_json(mlb_box_dir / f"{gid_mlb}.json.gz", _sample_mlb_boxscore_payload())

        gid_npb = 9_000_100_000 + i
        h_runs = 2 if i % 5 == 0 else (4 if i % 2 == 0 else 1)
        a_runs = 2 if i % 5 == 0 else (1 if i % 2 == 0 else 4)
        npb_games.append(
            {
                "game_id": gid_npb,
                "game_date": d_str,
                "start_time_utc": f"{d_str}T09:00:00Z",
                "home_team": "HAN",
                "away_team": "YOM",
                "venue": "NPB_KOSHIEN",
                "game_state": "FINAL" if is_final else "FUT",
                "innings_played": 12 if h_runs == a_runs else 9,
                "home_runs": h_runs if is_final else None,
                "away_runs": a_runs if is_final else None,
                "home_sp_id": 9001,
                "away_sp_id": 9002,
                "home_sp_hand": "R",
                "away_sp_hand": "R",
            }
        )
        if is_final:
            write_cached_json(
                npb_box_dir / f"{gid_npb}.json.gz",
                {
                    "home_team": "HAN",
                    "away_team": "YOM",
                    "home": {
                        "runs": h_runs,
                        "pitchers": [
                            {
                                "player_id": 9001,
                                "player_name": "Hiroto Saiki",
                                "pitch_hand": "R",
                                "ip": "7.0",
                                "batters_faced": 26,
                                "pitches_thrown": 100,
                                "hits_allowed": 5,
                                "runs_allowed": a_runs,
                                "er_allowed": a_runs,
                                "bb_allowed": 2,
                                "k_recorded": 8,
                            }
                        ],
                        "batters": [
                            {
                                "player_id": 9100 + s,
                                "player_name": f"Tigers {s}",
                                "bat_side": "L" if s <= 5 else "R",
                                "batting_order_slot": s,
                                "position_group": "OF",
                                "pa": 4,
                                "ab": 3,
                                "hits": 1,
                                "hr": 1 if s == 4 else 0,
                                "runs": 1 if s <= h_runs else 0,
                                "rbi": 1 if s <= h_runs else 0,
                                "bb": 1,
                                "k": 1,
                            }
                            for s in range(1, 10)
                        ],
                    },
                    "away": {
                        "runs": a_runs,
                        "pitchers": [
                            {
                                "player_id": 9002,
                                "player_name": "Shosei Togo",
                                "pitch_hand": "R",
                                "ip": "7.0",
                                "batters_faced": 26,
                                "pitches_thrown": 102,
                                "hits_allowed": 6,
                                "runs_allowed": h_runs,
                                "er_allowed": h_runs,
                                "bb_allowed": 2,
                                "k_recorded": 7,
                            }
                        ],
                        "batters": [
                            {
                                "player_id": 9200 + s,
                                "player_name": f"Giants {s}",
                                "bat_side": "R",
                                "batting_order_slot": s,
                                "position_group": "INF",
                                "pa": 4,
                                "ab": 3,
                                "hits": 1,
                                "hr": 0,
                                "runs": 1 if s <= a_runs else 0,
                                "rbi": 1 if s <= a_runs else 0,
                                "bb": 1,
                                "k": 1,
                            }
                            for s in range(1, 10)
                        ],
                    },
                },
            )

    write_cached_json(
        mlb_sched_dir / "schedule_2025-05-01_2025-05-14.json.gz", {"dates": mlb_dates}
    )
    write_cached_json(npb_sched_dir / "npb_schedule_2025.json.gz", {"games": npb_games})


def test_baseball_cli_end_to_end_mlb_and_npb(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Run offline E2E CLI workflow: ingest -> features -> train -> backtest -> predict -> score."""
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))
    _seed_offline_baseball_raw_cache(tmp_path / "data" / "raw")

    # 1. Ingest MLB + NPB offline
    res_ing = runner.invoke(app, ["ingest", "--sport", "baseball", "--league", "all", "--offline"])
    assert res_ing.exit_code == 0, res_ing.output
    assert "Ingestion complete" in res_ing.output

    # 2. Build Gold features
    res_feat = runner.invoke(app, ["features", "build", "--sport", "baseball", "--league", "all"])
    assert res_feat.exit_code == 0, res_feat.output
    assert "Gold features built" in res_feat.output

    # 3. Train Stage 1 + Stage 2
    res_train = runner.invoke(
        app, ["train", "--sport", "baseball", "--league", "all", "--include-stage2"]
    )
    assert res_train.exit_code == 0, res_train.output
    assert "Trained & saved baseball model bundle" in res_train.output

    # 4. Walk-forward Backtest (Stage 1 + Stage 2)
    res_bt = runner.invoke(
        app,
        ["backtest", "--sport", "baseball", "--league", "all", "--stage", "all", "--splits", "2"],
    )
    assert res_bt.exit_code == 0, res_bt.output
    assert "Walk-Forward Backtest (BASEBALL MLB 2-Way" in res_bt.output
    assert "Walk-Forward Backtest (BASEBALL NPB 3-Way" in res_bt.output

    # 5. Predict MLB matchup with Stage 2 players on a completed date (for ledger scoring)
    res_pred_mlb = runner.invoke(
        app,
        [
            "predict",
            "--sport",
            "baseball",
            "--league",
            "mlb",
            "--date",
            "2025-05-05",
            "--matchup",
            "SDP@LAD",
            "--include-players",
            "--offline",
        ],
    )
    assert res_pred_mlb.exit_code == 0, res_pred_mlb.output
    assert "SDP @ LAD" in res_pred_mlb.output
    assert "Starting Pitchers (SP)" in res_pred_mlb.output

    # 6. Predict NPB matchup with Stage 2 players and --team next games
    res_pred_npb = runner.invoke(
        app,
        [
            "predict",
            "--sport",
            "baseball",
            "--league",
            "npb",
            "--date",
            "2025-05-05",
            "--matchup",
            "YOM@HAN",
            "--include-players",
            "--offline",
        ],
    )
    assert res_pred_npb.exit_code == 0, res_pred_npb.output
    assert "YOM @ HAN" in res_pred_npb.output

    res_pred_team = runner.invoke(
        app,
        [
            "predict",
            "--sport",
            "baseball",
            "--team",
            "HAN",
            "--games",
            "2",
            "--offline",
        ],
    )
    assert res_pred_team.exit_code == 0, res_pred_team.output

    # 7. Score the live baseball prediction ledger against completed Silver games
    res_score = runner.invoke(app, ["score-predictions", "--sport", "baseball"])
    assert res_score.exit_code == 0, res_score.output
    assert "Live Prediction Ledger Accuracy (BASEBALL" in res_score.output
    assert datetime(2025, 5, 5, tzinfo=UTC).year == 2025
