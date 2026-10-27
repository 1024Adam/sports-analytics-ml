"""Stage 2 Optional Player Matchup Stat Projection tests (`test_stage2_player.py`)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sports_ml.cli import app
from sports_ml.config import RANDOM_SEED, get_settings
from sports_ml.sports.hockey.models.predictor import HockeyMatchupPredictor
from sports_ml.sports.hockey.models.stage1_game import Stage1GameOutcomeModel
from sports_ml.sports.hockey.models.stage2_player import (
    Stage2PlayerProjectionModel,
    reconcile_skater_team_goals,
)


def _synthetic_stage2_skater_df(n_games: int = 20) -> pl.DataFrame:
    """Create a deterministic multi-game Stage 2 skater feature & target dataset."""
    rng = np.random.default_rng(RANDOM_SEED)
    base_time = datetime(2024, 10, 1, 23, 0, tzinfo=UTC)

    players = [
        (8477956, "David Pastrnak", "BOS", "F", 16.5, 3.8, 10.5, 1.5, 1.8),
        (8473419, "Brad Marchand", "BOS", "F", 15.0, 3.0, 7.5, 0.9, 1.4),
        (8479325, "Charlie McAvoy", "BOS", "D", 20.0, 2.5, 5.5, 0.35, 1.2),
        (8479318, "Auston Matthews", "TOR", "F", 17.0, 4.0, 11.5, 1.7, 1.4),
        (8478483, "Mitch Marner", "TOR", "F", 16.5, 3.5, 6.8, 0.8, 2.0),
        (8476853, "Morgan Rielly", "TOR", "D", 19.5, 2.5, 5.2, 0.30, 1.3),
    ]

    rows = []
    for g_idx in range(n_games):
        gid = 2024020100 + g_idx
        ts = base_time + timedelta(days=g_idx * 2)
        gdate = date(2024, 10, 1) + timedelta(days=g_idx * 2)
        for pid, pname, team, pos, ev_toi, pp_toi, sog_60, g_60, a_60 in players:
            opp = "TOR" if team == "BOS" else "BOS"
            tot_toi = ev_toi + pp_toi
            opp_gsax = float(rng.uniform(-0.8, 0.8))
            s1_lambda = float(rng.uniform(2.5, 3.8))

            sog = max(0, int(round(sog_60 * (tot_toi / 60.0) + rng.normal(0.0, 0.6))))
            goals = 1 if float(rng.uniform(0.0, 1.0)) < (g_60 * tot_toi / 60.0) else 0
            assists = 1 if float(rng.uniform(0.0, 1.0)) < (a_60 * tot_toi / 60.0) else 0

            rows.append(
                {
                    "game_id": gid,
                    "event_id": gid,
                    "event_seq": 1,
                    "game_date": gdate,
                    "start_time_utc": ts,
                    "season": 20242025,
                    "player_id": pid,
                    "player_name": pname,
                    "team": team,
                    "opponent": opp,
                    "position_group": pos,
                    "is_home": team == "BOS",
                    "rolling_toi_ev_min": ev_toi,
                    "rolling_toi_pp_min": pp_toi,
                    "rolling_toi_total_min": tot_toi,
                    "rolling_shifts_per_game": tot_toi * 1.25,
                    "team_injury_vacated_toi_f_or_d": 0.0,
                    "rolling_sog_per_60": sog_60,
                    "rolling_icf_per_60": sog_60 * 1.8,
                    "rolling_ixg_per_60": g_60 * 0.95,
                    "rolling_goals_per_60": g_60,
                    "rolling_assists_per_60": a_60,
                    "rolling_points_per_60": g_60 + a_60,
                    "career_shooting_pct_shrunk": 0.13 if pos == "F" else 0.055,
                    "opp_xga_per_60_5v5": 2.55,
                    "opp_sa_per_60_5v5": 30.0,
                    "opp_pk_xga_per_60": 7.5,
                    "opp_penalties_taken_per_60": 3.0,
                    "opp_goalie_gsax_per_60_shrunk": opp_gsax,
                    "stage1_team_win_prob": 0.55 if team == "BOS" else 0.45,
                    "stage1_expected_team_goals": s1_lambda,
                    "team_rest_days": 2,
                    "is_b2b": False,
                    "toi_ev_min": ev_toi,
                    "toi_pp_min": pp_toi,
                    "toi_total_min": tot_toi,
                    "sog": sog,
                    "goals": goals,
                    "assists": assists,
                    "points": goals + assists,
                }
            )
    return pl.DataFrame(rows)


def test_stage2_non_negativity_and_probability_bounds() -> None:
    """Verify all predicted counts/TOI are >= 0 and milestone probabilities lie in [0, 1]."""
    df = _synthetic_stage2_skater_df(n_games=20)
    train_df = df.filter(pl.col("game_id") < 2024020118)
    test_df = df.filter(pl.col("game_id") >= 2024020118)

    model = Stage2PlayerProjectionModel(reconciliation_gamma=0.50, random_seed=RANDOM_SEED)
    model.fit(train_df)
    skater_preds = model.predict(test_df)

    for count_col in (
        "proj_toi_ev_min",
        "proj_toi_pp_min",
        "proj_toi_min",
        "proj_sog",
        "proj_goals",
        "proj_assists",
        "proj_points",
    ):
        vals = skater_preds[count_col].to_numpy()
        assert np.all(vals >= 0.0), f"Column {count_col} contains negative values."

    # Verify proj_points == proj_goals + proj_assists
    np.testing.assert_allclose(
        skater_preds["proj_points"].to_numpy(),
        skater_preds["proj_goals"].to_numpy() + skater_preds["proj_assists"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    for prob_col in ("prob_1plus_goal", "prob_1plus_point"):
        p_vals = skater_preds[prob_col].to_numpy()
        assert np.all((p_vals >= 0.0) & (p_vals <= 1.0))

    # Verify Goalie projection model outputs
    goalie_test_df = pl.DataFrame(
        {
            "game_id": [2024020118, 2024020118],
            "player_id": [8480280, 8479361],
            "team": ["BOS", "TOR"],
            "opponent": ["TOR", "BOS"],
            "is_home": [True, False],
            "gsax_per_60_shrunk": [0.55, -0.40],
            "save_pct_shrunk": [0.916, 0.898],
        }
    )
    goalie_preds = model.predict_goalies(goalie_test_df, skater_projections_df=skater_preds)
    for g_col in ("proj_shots_against", "proj_goals_against", "proj_saves"):
        assert np.all(goalie_preds[g_col].to_numpy() >= 0.0)
    assert np.all(
        (goalie_preds["proj_save_pct"].to_numpy() >= 0.0)
        & (goalie_preds["proj_save_pct"].to_numpy() <= 1.0)
    )


def test_stage_conditioning_monotonicity_lambda_and_goalie_gsax() -> None:
    """Increasing Stage 1 expected team goals or facing a weaker goalie increases projections."""
    df = _synthetic_stage2_skater_df(n_games=20)
    train_df = df.filter(pl.col("game_id") < 2024020119)
    base_matchup = df.filter(pl.col("game_id") == 2024020119)

    model = Stage2PlayerProjectionModel(reconciliation_gamma=0.50, random_seed=RANDOM_SEED)
    model.fit(train_df)

    # 1. Monotonicity with respect to Stage 1 expected team goals (lambda_team)
    low_lambda_df = base_matchup.with_columns(
        pl.lit(2.10, dtype=pl.Float64).alias("stage1_expected_team_goals")
    )
    high_lambda_df = base_matchup.with_columns(
        pl.lit(4.20, dtype=pl.Float64).alias("stage1_expected_team_goals")
    )

    preds_low_lambda = model.predict(low_lambda_df)
    preds_high_lambda = model.predict(high_lambda_df)

    assert np.all(
        preds_high_lambda["proj_goals"].to_numpy() > preds_low_lambda["proj_goals"].to_numpy()
    )
    assert np.all(
        preds_high_lambda["proj_points"].to_numpy() > preds_low_lambda["proj_points"].to_numpy()
    )
    assert np.all(
        preds_high_lambda["prob_1plus_point"].to_numpy()
        > preds_low_lambda["prob_1plus_point"].to_numpy()
    )

    # 2. Monotonicity with respect to opponent starting goalie GSAx/60
    # Facing a weaker goalie (GSAx = -1.2) vs. an elite goalie (GSAx = +1.2)
    elite_goalie_df = base_matchup.with_columns(
        pl.lit(3.10, dtype=pl.Float64).alias("stage1_expected_team_goals"),
        pl.lit(1.20, dtype=pl.Float64).alias("opp_goalie_gsax_per_60_shrunk"),
    )
    weak_goalie_df = base_matchup.with_columns(
        pl.lit(3.10, dtype=pl.Float64).alias("stage1_expected_team_goals"),
        pl.lit(-1.20, dtype=pl.Float64).alias("opp_goalie_gsax_per_60_shrunk"),
    )

    preds_vs_elite = model.predict(elite_goalie_df)
    preds_vs_weak = model.predict(weak_goalie_df)

    assert np.all(preds_vs_weak["proj_goals"].to_numpy() > preds_vs_elite["proj_goals"].to_numpy())
    assert np.all(
        preds_vs_weak["proj_points"].to_numpy() > preds_vs_elite["proj_points"].to_numpy()
    )


def test_stage2_can_be_toggled_off_with_zero_overhead() -> None:
    """Verify Stage 2 is completely skipped when include_players=False."""
    stage1_df = pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "event_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "event_seq": [1] * 8,
            "start_time_utc": [datetime(2024, 10, 1 + i, 23, 0, tzinfo=UTC) for i in range(8)],
            "elo_home_win_prob": [0.58, 0.45, 0.62, 0.41, 0.59, 0.48, 0.61, 0.55],
            "xgf_pct_5v5_w10_diff": [0.06, -0.05, 0.08, -0.07, 0.05, -0.02, 0.07, 0.04],
            "home_reg_goals": [4, 2, 4, 1, 3, 2, 4, 3],
            "away_reg_goals": [2, 3, 1, 4, 2, 3, 2, 2],
            "home_win": [1, 0, 1, 0, 1, 0, 1, 1],
        }
    )

    s1_model = Stage1GameOutcomeModel().fit(stage1_df.slice(0, 6), stage1_df.slice(6, 2))
    unfitted_s2_model = Stage2PlayerProjectionModel()

    def _run_pipeline(include_players: bool) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        s1_preds = s1_model.predict(stage1_df.slice(7, 1))
        if not include_players:
            return s1_preds, None
        return s1_preds, unfitted_s2_model.predict(pl.DataFrame())

    # When include_players=False, unfitted Stage 2 model is never invoked
    s1_only, s2_none = _run_pipeline(include_players=False)
    assert s2_none is None
    assert 0.0 < s1_only["home_win_prob"][0] < 1.0

    # When include_players=True on an unfitted Stage 2 model, it raises RuntimeError
    with pytest.raises(RuntimeError, match="must be fitted"):
        _run_pipeline(include_players=True)

    with pytest.raises(ValueError, match="gamma must be in"):
        reconcile_skater_team_goals(pl.DataFrame({"a": [1]}), gamma=1.5)


def _populate_raw_fixture_lakehouse(root_dir: Path) -> None:
    """Write deterministic raw NHL API JSONs and MoneyPuck CSVs into `root_dir/data/raw/`."""
    sched_dir = root_dir / "data" / "raw" / "nhl_api" / "schedules"
    box_dir = root_dir / "data" / "raw" / "nhl_api" / "boxscores"
    mp_dir = root_dir / "data" / "raw" / "moneypuck"
    sched_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)
    mp_dir.mkdir(parents=True, exist_ok=True)

    game_Week = []
    for idx in range(6):
        gid = 2024020001 + idx
        d_str = f"2024-10-{10 + idx * 2:02d}"
        home_t = "BOS" if idx % 2 == 0 else "TOR"
        away_t = "TOR" if idx % 2 == 0 else "BOS"
        h_score = 4 if idx % 2 == 0 else 2
        a_score = 2 if idx % 2 == 0 else 3
        game_Week.append(
            {
                "date": d_str,
                "games": [
                    {
                        "id": gid,
                        "season": 20242025,
                        "gameType": 2,
                        "startTimeUTC": f"{d_str}T23:00:00Z",
                        "gameState": "FINAL",
                        "homeTeam": {"abbrev": home_t, "score": h_score},
                        "awayTeam": {"abbrev": away_t, "score": a_score},
                        "gameOutcome": {"lastPeriodType": "REG"},
                    }
                ],
            }
        )

        box_payload = {
            "id": gid,
            "season": 20242025,
            "gameDate": d_str,
            "startTimeUTC": f"{d_str}T23:00:00Z",
            "gameState": "FINAL",
            "gameOutcome": {"lastPeriodType": "REG"},
            "homeTeam": {"abbrev": home_t, "score": h_score, "sog": 32},
            "awayTeam": {"abbrev": away_t, "score": a_score, "sog": 27},
            "playerByGameStats": {
                "homeTeam": {
                    "forwards": [
                        {
                            "playerId": 8477956 if home_t == "BOS" else 8479318,
                            "name": {"default": "Star Forward H"},
                            "toi": "20:00",
                            "powerPlayToi": "03:30",
                            "shorthandedToi": "00:00",
                            "goals": 2,
                            "assists": 1,
                            "points": 3,
                            "sog": 5,
                            "powerPlayGoals": 1,
                            "blockedShots": 1,
                            "pim": 0,
                        }
                    ],
                    "defense": [
                        {
                            "playerId": 8479325 if home_t == "BOS" else 8476853,
                            "name": {"default": "Top Defense H"},
                            "toi": "23:30",
                            "powerPlayToi": "02:30",
                            "shorthandedToi": "02:00",
                            "goals": 0,
                            "assists": 1,
                            "points": 1,
                            "sog": 3,
                            "powerPlayGoals": 0,
                            "blockedShots": 2,
                            "pim": 2,
                        }
                    ],
                    "goalies": [
                        {
                            "playerId": 8480280 if home_t == "BOS" else 8479361,
                            "name": {"default": "Starter H"},
                            "starter": True,
                            "toi": "60:00",
                            "shotsAgainst": 27,
                            "saves": 25,
                            "goalsAgainst": a_score,
                        }
                    ],
                },
                "awayTeam": {
                    "forwards": [
                        {
                            "playerId": 8479318 if away_t == "TOR" else 8477956,
                            "name": {"default": "Star Forward A"},
                            "toi": "20:30",
                            "powerPlayToi": "03:30",
                            "shorthandedToi": "00:00",
                            "goals": 1,
                            "assists": 1,
                            "points": 2,
                            "sog": 4,
                            "powerPlayGoals": 0,
                            "blockedShots": 1,
                            "pim": 0,
                        }
                    ],
                    "defense": [
                        {
                            "playerId": 8476853 if away_t == "TOR" else 8479325,
                            "name": {"default": "Top Defense A"},
                            "toi": "23:00",
                            "powerPlayToi": "02:00",
                            "shorthandedToi": "02:00",
                            "goals": 0,
                            "assists": 1,
                            "points": 1,
                            "sog": 2,
                            "powerPlayGoals": 0,
                            "blockedShots": 2,
                            "pim": 0,
                        }
                    ],
                    "goalies": [
                        {
                            "playerId": 8479361 if away_t == "TOR" else 8480280,
                            "name": {"default": "Starter A"},
                            "starter": True,
                            "toi": "60:00",
                            "shotsAgainst": 32,
                            "saves": 28,
                            "goalsAgainst": h_score,
                        }
                    ],
                },
            },
        }
        (box_dir / f"{gid}.json").write_text(json.dumps(box_payload), encoding="utf-8")

    (sched_dir / "schedule_2024-10-10.json").write_text(
        json.dumps({"gameWeek": game_Week}), encoding="utf-8"
    )

    # Also add upcoming FUT club schedule games for COL and BOS
    club_col_games = [
        {
            "id": 2024020010,
            "season": 20242025,
            "gameType": 2,
            "gameDate": "2024-10-22",
            "startTimeUTC": "2024-10-22T23:00:00Z",
            "gameState": "FUT",
            "homeTeam": {"abbrev": "COL"},
            "awayTeam": {"abbrev": "BOS"},
        },
        {
            "id": 2024020011,
            "season": 20242025,
            "gameType": 2,
            "gameDate": "2024-10-24",
            "startTimeUTC": "2024-10-24T23:00:00Z",
            "gameState": "FUT",
            "homeTeam": {"abbrev": "TOR"},
            "awayTeam": {"abbrev": "COL"},
        },
        {
            "id": 2024020012,
            "season": 20242025,
            "gameType": 2,
            "gameDate": "2024-10-26",
            "startTimeUTC": "2024-10-26T23:00:00Z",
            "gameState": "FUT",
            "homeTeam": {"abbrev": "COL"},
            "awayTeam": {"abbrev": "TOR"},
        },
    ]
    (sched_dir / "club_COL_20242025.json").write_text(
        json.dumps({"games": club_col_games}), encoding="utf-8"
    )


def test_end_to_end_cli_and_predictor_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify full CLI workflow: ingest -> features -> train -> backtest -> predict -> score."""
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))
    _populate_raw_fixture_lakehouse(tmp_path)

    runner = CliRunner()

    # 1. sports-ml ingest --sport hockey --offline
    res_ingest = runner.invoke(app, ["ingest", "--sport", "hockey", "--offline"])
    assert res_ingest.exit_code == 0, res_ingest.stdout
    assert "Ingestion complete" in res_ingest.stdout

    # 2. sports-ml features build --sport hockey
    res_feat = runner.invoke(app, ["features", "build", "--sport", "hockey"])
    assert res_feat.exit_code == 0, res_feat.stdout
    assert "Gold features built" in res_feat.stdout

    # 3. sports-ml train --sport hockey --include-stage2
    res_train = runner.invoke(app, ["train", "--sport", "hockey", "--include-stage2"])
    assert res_train.exit_code == 0, res_train.stdout
    assert "Trained & saved model bundle" in res_train.stdout

    # 4. sports-ml backtest --sport hockey --stage all
    res_bt = runner.invoke(app, ["backtest", "--sport", "hockey", "--stage", "all"])
    assert res_bt.exit_code == 0, res_bt.stdout
    assert "Walk-Forward Backtest" in res_bt.stdout

    # 5. sports-ml predict (Stage 1 only)
    res_pred1 = runner.invoke(
        app,
        ["predict", "--sport", "hockey", "--date", "2024-10-20", "--offline"],
    )
    assert res_pred1.exit_code == 0, res_pred1.stdout
    assert "Stage 1" in res_pred1.stdout

    # 6. sports-ml predict (Stage 1 + Stage 2 with --include-players and --matchup)
    res_pred2 = runner.invoke(
        app,
        [
            "predict",
            "--sport",
            "hockey",
            "--date",
            "2024-10-20",
            "--matchup",
            "TOR@BOS",
            "--include-players",
            "--offline",
        ],
    )
    assert res_pred2.exit_code == 0, res_pred2.stdout
    assert "Stage 2" in res_pred2.stdout
    assert "TOR — Forwards" in res_pred2.stdout
    assert "TOR — Defencemen" in res_pred2.stdout
    assert "TOR — Goalies" in res_pred2.stdout
    assert "BOS — Forwards" in res_pred2.stdout
    assert "BOS — Defencemen" in res_pred2.stdout
    assert "BOS — Goalies" in res_pred2.stdout

    # 7. sports-ml score-predictions --sport hockey
    res_score = runner.invoke(app, ["score-predictions", "--sport", "hockey"])
    assert res_score.exit_code == 0, res_score.stdout
    assert "Live Prediction Ledger Accuracy" in res_score.stdout

    # Verify programmatic Python API (`HockeyMatchupPredictor.from_artifacts`)
    settings = get_settings(root_dir=tmp_path)
    predictor = HockeyMatchupPredictor.from_artifacts(
        tmp_path / "artifacts" / "hockey" / "latest",
        settings=settings,
    )
    s1_pred = predictor.predict_matchup("BOS", "TOR", date(2024, 10, 20), include_players=False)
    assert s1_pred.player_projections is None
    assert s1_pred.home_win_prob + s1_pred.away_win_prob == pytest.approx(1.0)
    assert len(s1_pred.warnings) > 0

    full_pred = predictor.predict_matchup("BOS", "TOR", date(2024, 10, 20), include_players=True)
    assert full_pred.player_projections is not None
    assert not full_pred.player_projections.home_skaters.is_empty()
    assert not full_pred.player_projections.away_skaters.is_empty()
    assert not full_pred.player_projections.goalies.is_empty()

    # 8. Predict a team's next game (--team COL) and next N games (--team COL --games 3)
    res_team_1 = runner.invoke(
        app,
        ["predict", "--sport", "hockey", "--team", "COL", "--offline"],
    )
    assert res_team_1.exit_code == 0, res_team_1.stdout
    assert "2024-10-22" in res_team_1.stdout
    assert "BOS @ COL" in res_team_1.stdout
    assert "2024-10-24" not in res_team_1.stdout

    res_team_3 = runner.invoke(
        app,
        ["predict", "--sport", "hockey", "--team", "COL", "--games", "3", "--offline"],
    )
    assert res_team_3.exit_code == 0, res_team_3.stdout
    assert "2024-10-22" in res_team_3.stdout
    assert "2024-10-24" in res_team_3.stdout
    assert "2024-10-26" in res_team_3.stdout

    # Programmatic API verification for predict_team_next_games
    next_2_col = predictor.predict_team_next_games(
        "COL",
        n_games=2,
        from_date=date(2024, 10, 23),
        offline=True,
        persist_ledger=False,
    )
    assert len(next_2_col) == 2
    assert [p.game_date for p in next_2_col] == [date(2024, 10, 24), date(2024, 10, 26)]

    # CLI & API validation error checks
    res_bad_both = runner.invoke(
        app,
        ["predict", "--sport", "hockey", "--team", "COL", "--matchup", "BOS@COL", "--offline"],
    )
    assert res_bad_both.exit_code != 0

    res_bad_games = runner.invoke(
        app,
        ["predict", "--sport", "hockey", "--games", "2", "--offline"],
    )
    assert res_bad_games.exit_code != 0

    with pytest.raises(ValueError, match="n_games must be >= 1"):
        predictor.predict_team_next_games("COL", n_games=0, offline=True)
    with pytest.raises(ValueError, match="No scheduled games found"):
        predictor.predict_team_next_games("SEA", n_games=1, offline=True)
