"""Strict anti-contamination and odds-firewall tests for EPL Soccer (`test_odds_firewall.py`)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sports_ml.config import RANDOM_SEED
from sports_ml.core.market import is_forbidden_market_column
from sports_ml.core.ratings import compute_elo_1x2_probabilities
from sports_ml.sports.soccer.features.pipeline import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    assert_zero_odds_contamination,
    strip_market_columns_from_frame,
)
from sports_ml.sports.soccer.ingestion.football_data import parse_football_data_csv
from sports_ml.sports.soccer.ingestion.fpl_api import parse_fixtures_payload
from sports_ml.sports.soccer.ingestion.odds import parse_espn_epl_1x2_odds_items
from sports_ml.sports.soccer.ingestion.reconciler import SoccerSilverReconciler
from sports_ml.sports.soccer.models.stage1_match import (
    Stage1SoccerMatchModel,
    run_soccer_stage1_walk_forward_backtest,
)
from sports_ml.sports.soccer.models.stage2_player import Stage2SoccerPlayerModel


def _make_synthetic_soccer_stage1_with_odds(n_matches: int = 90) -> pl.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    base_time = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
    rows = []
    for i in range(n_matches):
        ts = base_time + timedelta(days=i // 2, hours=(i % 2) * 2)
        strength = float(rng.uniform(-1.5, 1.5))
        elo_diff = 48.0 + 130.0 * strength
        p_h_elo, p_d_elo, p_a_elo = compute_elo_1x2_probabilities(elo_diff, draw_param=0.26)

        outcome_1x2 = int(rng.choice(3, p=[p_h_elo, p_d_elo, p_a_elo]))
        h_goals = 2 if outcome_1x2 == 0 else (1 if outcome_1x2 == 1 else 0)
        a_goals = 0 if outcome_1x2 == 0 else (1 if outcome_1x2 == 1 else 2)

        h_odds = round(1.0 / (p_h_elo * 1.03), 3)
        d_odds = round(1.0 / (p_d_elo * 1.03), 3)
        a_odds = round(1.0 / (p_a_elo * 1.03), 3)

        rows.append(
            {
                "sport": "soccer",
                "season": 20242025,
                "matchweek": (i // 10) + 1,
                "match_id": 202400001 + i,
                "event_id": 202400001 + i,
                "event_seq": 1,
                "match_date": ts.date(),
                "start_time_utc": ts,
                "home_team": "ARS" if i % 2 == 0 else "LIV",
                "away_team": "MCI" if i % 2 == 0 else "CHE",
                "elo_home_pre": 1500.0 + 65.0 * strength,
                "elo_away_pre": 1500.0 - 65.0 * strength,
                "elo_diff_pre": elo_diff,
                "elo_home_win_prob": p_h_elo,
                "elo_draw_prob": p_d_elo,
                "elo_away_win_prob": p_a_elo,
                "npxgd_per_90_w10_home": 0.50 * strength,
                "npxgd_per_90_w10_away": -0.50 * strength,
                "npxgd_per_90_w10_diff": 1.00 * strength,
                "npxgf_pct_w10_diff": float(np.clip(0.12 * strength, -0.35, 0.35)),
                "pressing_intensity_diff": 1.1 * strength,
                "deep_completions_per_90_w10_diff": 1.8 * strength,
                "rest_advantage": 1 if strength > 0 else 0,
                "gk_gsaa_per_90_diff": 0.30 * strength,
                "fwd_npxg_per_90_diff": 0.20 * strength,
                "mid_xg_chain_per_90_diff": 0.18 * strength,
                "home_goals": h_goals,
                "away_goals": a_goals,
                "outcome_1x2": outcome_1x2,
                "home_win": 1 if outcome_1x2 == 0 else 0,
                "is_draw": 1 if outcome_1x2 == 1 else 0,
                "away_win": 1 if outcome_1x2 == 2 else 0,
                "market_home_decimal_odds": h_odds,
                "market_draw_decimal_odds": d_odds,
                "market_away_decimal_odds": a_odds,
                "market_prob_home": p_h_elo,
                "market_prob_draw": p_d_elo,
                "market_prob_away": p_a_elo,
                "market_overround": 0.03,
            }
        )
    return pl.DataFrame(rows)


def test_soccer_feature_columns_never_contain_market_or_odds() -> None:
    """Verify EPL Stage 1 & Stage 2 feature lists contain zero market/odds columns."""
    assert len(STAGE1_FEATURE_COLUMNS) > 0
    assert len(STAGE2_FEATURE_COLUMNS) > 0
    for col in (*STAGE1_FEATURE_COLUMNS, *STAGE2_FEATURE_COLUMNS):
        assert not is_forbidden_market_column(col), f"Forbidden column leaked: {col}"

    with pytest.raises(ValueError, match="Market/odds contamination detected"):
        assert_zero_odds_contamination(["elo_diff_pre", "market_draw_decimal_odds"])

    df_with_odds = _make_synthetic_soccer_stage1_with_odds(20).with_columns(
        pl.lit(3.40).alias("b365_draw_odds")
    )
    resolved_s1 = Stage1SoccerMatchModel.resolve_feature_columns(df_with_odds)
    assert all(not is_forbidden_market_column(c) for c in resolved_s1)

    stripped = strip_market_columns_from_frame(df_with_odds)
    assert all(not is_forbidden_market_column(c) for c in stripped.columns)

    minimal_s2 = pl.DataFrame(
        {
            "player_id": [1, 2],
            "custom_npxg": [0.45, 0.25],
            "market_prob_home": [0.60, 0.40],
            "anytime_scorer_odds": [2.10, 3.50],
        }
    )
    resolved_s2 = Stage2SoccerPlayerModel.resolve_feature_columns(minimal_s2)
    assert resolved_s2 == ["custom_npxg"]


def test_soccer_mutating_market_cols_changes_zero_predictions_while_updating_benchmark() -> None:
    """Verify mutating market_* columns leaves EPL 1X2 predictions unchanged."""
    base_df = _make_synthetic_soccer_stage1_with_odds(n_matches=90)

    mutated_df = base_df.with_columns(
        pl.lit(6.50, dtype=pl.Float64).alias("market_home_decimal_odds"),
        pl.lit(4.20, dtype=pl.Float64).alias("market_draw_decimal_odds"),
        pl.lit(1.45, dtype=pl.Float64).alias("market_away_decimal_odds"),
        pl.lit(0.14, dtype=pl.Float64).alias("market_prob_home"),
        pl.lit(0.22, dtype=pl.Float64).alias("market_prob_draw"),
        pl.lit(0.64, dtype=pl.Float64).alias("market_prob_away"),
        pl.lit(0.05, dtype=pl.Float64).alias("market_overround"),
    )

    bt_base = run_soccer_stage1_walk_forward_backtest(
        base_df,
        n_splits=2,
        min_train_groups=15,
        random_seed=RANDOM_SEED,
    )
    bt_mutated = run_soccer_stage1_walk_forward_backtest(
        mutated_df,
        n_splits=2,
        min_train_groups=15,
        random_seed=RANDOM_SEED,
    )

    for prob_col in ("home_win_prob", "draw_prob", "away_win_prob"):
        p_base = bt_base.oof_predictions[prob_col].to_numpy()
        p_mut = bt_mutated.oof_predictions[prob_col].to_numpy()
        assert float(np.max(np.abs(p_base - p_mut))) == 0.0
        np.testing.assert_array_equal(p_base, p_mut)

    assert bt_base.benchmark_report.market_closing_baseline is not None
    assert bt_mutated.benchmark_report.market_closing_baseline is not None
    assert (
        bt_base.benchmark_report.market_closing_baseline.rps
        != bt_mutated.benchmark_report.market_closing_baseline.rps
    )
    assert bt_base.benchmark_report.market_edge_report is not None
    assert bt_mutated.benchmark_report.market_edge_report is not None
    assert (
        bt_base.benchmark_report.market_edge_report.market_brier_score
        != bt_mutated.benchmark_report.market_edge_report.market_brier_score
    )


def test_football_data_csv_closing_odds_waterfall_and_reconciler(tmp_path: Path) -> None:
    """Verify Football-Data E0.csv extracts Pinnacle/Bet365 odds and preserves across upsert."""
    csv_text = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR,"
        "PSCH,PSCD,PSCA,B365H,B365D,B365A\n"
        "E0,17/08/2024,15:00,Arsenal,Wolves,2,0,H,18,9,6,3,8,2,10,12,1,2,0,0,"
        "1.22,6.80,13.50,1.20,6.50,13.00\n"
        "E0,18/08/2024,16:30,Chelsea,Man City,0,2,A,10,11,3,5,4,3,9,8,2,1,0,0,"
        ",,,4.20,3.80,1.80\n"
    )
    matches_fd, _ = parse_football_data_csv(csv_text, season=20242025)
    assert matches_fd.height == 2

    # Match 0 used Pinnacle closing (PSCH/PSCD/PSCA)
    assert matches_fd["market_home_decimal_odds"][0] == pytest.approx(1.22)
    assert (
        matches_fd["market_prob_home"][0]
        + matches_fd["market_prob_draw"][0]
        + matches_fd["market_prob_away"][0]
    ) == pytest.approx(1.0, abs=1e-9)

    # Match 1 fell back cleanly to Bet365 opening (B365H/B365D/B365A) when PSCH was blank
    assert matches_fd["market_home_decimal_odds"][1] == pytest.approx(4.20)
    assert matches_fd["market_away_decimal_odds"][1] == pytest.approx(1.80)

    # Verify SoccerSilverReconciler preserves Football-Data odds when FPL fixtures are upserted
    fpl_fx = parse_fixtures_payload(
        [
            {
                "id": 1,
                "event": 1,
                "kickoff_time": "2024-08-17T15:00:00Z",
                "team_h": 1,
                "team_a": 2,
                "team_h_score": 2,
                "team_a_score": 0,
                "finished": True,
            }
        ],
        teams_map={1: "ARS", 2: "WOL"},
        season=20242025,
    )
    reconciler = SoccerSilverReconciler(silver_dir=tmp_path / "silver" / "soccer")
    reconciler.reconcile_matches(matches_fd)
    merged = reconciler.reconcile_matches(fpl_fx)

    ars_row = merged.filter(pl.col("home_team") == "ARS").row(0, named=True)
    assert ars_row["market_home_decimal_odds"] == pytest.approx(1.22)
    assert ars_row["market_prob_home"] > 0.75

    # Verify ESPN EPL 1X2 odds parser
    espn_epl_items = [
        {
            "provider": {"name": "DraftKings"},
            "homeTeamOdds": {"moneyLine": -125},
            "drawOdds": {"moneyLine": 295.0},
            "awayTeamOdds": {"moneyLine": 310},
        },
        {
            "provider": {"name": "ESPN Bet - Live Odds"},
            "homeTeamOdds": {"moneyLine": -1000},
            "drawOdds": {"moneyLine": 800.0},
            "awayTeamOdds": {"moneyLine": 1500},
        },
    ]
    triplet = parse_espn_epl_1x2_odds_items(espn_epl_items)
    assert triplet is not None
    assert triplet[0] == pytest.approx(1.80)
    assert triplet[1] == pytest.approx(3.95)
    assert triplet[2] == pytest.approx(4.10)
