"""Strict anti-contamination and odds-firewall tests for NHL Hockey (`test_odds_firewall.py`)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sports_ml.config import RANDOM_SEED
from sports_ml.core.market import is_forbidden_market_column
from sports_ml.sports.hockey.features.pipeline import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    assert_zero_odds_contamination,
    strip_market_columns_from_frame,
)
from sports_ml.sports.hockey.ingestion.nhl_api import parse_schedule_payload
from sports_ml.sports.hockey.ingestion.odds import (
    NHLOddsIngestor,
    parse_nhl_historical_odds_csv,
    parse_the_odds_api_payload,
)
from sports_ml.sports.hockey.ingestion.reconciler import HockeySilverReconciler
from sports_ml.sports.hockey.models.stage1_game import (
    Stage1GameOutcomeModel,
    run_stage1_walk_forward_backtest,
)
from sports_ml.sports.hockey.models.stage2_player import Stage2PlayerProjectionModel


def _make_synthetic_hockey_stage1_with_odds(n_games: int = 80) -> pl.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    base_time = datetime(2024, 10, 10, 23, 0, tzinfo=UTC)
    rows = []
    for i in range(n_games):
        ts = base_time + timedelta(days=i // 2, hours=i % 2)
        strength = float(rng.uniform(-1.4, 1.4))
        elo_diff = 35.0 + 115.0 * strength
        elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        home_win = 1 if float(rng.uniform(0.0, 1.0)) < elo_prob else 0

        mkt_p_home = float(np.clip(elo_prob + rng.normal(0.0, 0.03), 0.15, 0.85))
        mkt_p_away = 1.0 - mkt_p_home
        h_odds = round(1.0 / (mkt_p_home * 1.04), 3)
        a_odds = round(1.0 / (mkt_p_away * 1.04), 3)

        rows.append(
            {
                "sport": "hockey",
                "season": 20242025,
                "game_id": 2024020001 + i,
                "event_id": 2024020001 + i,
                "event_seq": 1,
                "game_date": ts.date(),
                "start_time_utc": ts,
                "home_team": "BOS" if i % 2 == 0 else "COL",
                "away_team": "TOR" if i % 2 == 0 else "VGK",
                "elo_home_pre": 1500.0 + 55.0 * strength,
                "elo_away_pre": 1500.0 - 55.0 * strength,
                "elo_diff_pre": elo_diff,
                "elo_home_win_prob": elo_prob,
                "xgf_pct_5v5_w10_home": float(np.clip(0.50 + 0.06 * strength, 0.35, 0.65)),
                "xgf_pct_5v5_w10_away": float(np.clip(0.50 - 0.06 * strength, 0.35, 0.65)),
                "xgf_pct_5v5_w10_diff": 0.12 * strength,
                "goalie_gsax_per_60_diff": 0.45 * strength,
                "forward_ixg_per_60_diff": 0.20 * strength,
                "st_net_edge_home_vs_away": 0.08 * strength,
                "rest_advantage": 1 if strength > 0 else 0,
                "home_reg_goals": 3 if home_win == 1 else 2,
                "away_reg_goals": 2 if home_win == 1 else 3,
                "home_goals": 3 if home_win == 1 else 2,
                "away_goals": 2 if home_win == 1 else 3,
                "home_win": home_win,
                "market_home_decimal_odds": h_odds,
                "market_away_decimal_odds": a_odds,
                "market_prob_home": mkt_p_home,
                "market_prob_away": mkt_p_away,
                "market_overround": 0.04,
            }
        )
    return pl.DataFrame(rows)


def test_hockey_feature_columns_never_contain_market_or_odds() -> None:
    """Verify STAGE1_FEATURE_COLUMNS and STAGE2_FEATURE_COLUMNS contain zero market/odds columns."""
    assert len(STAGE1_FEATURE_COLUMNS) > 0
    assert len(STAGE2_FEATURE_COLUMNS) > 0
    for col in (*STAGE1_FEATURE_COLUMNS, *STAGE2_FEATURE_COLUMNS):
        assert not is_forbidden_market_column(col), f"Forbidden column leaked: {col}"

    with pytest.raises(ValueError, match="Market/odds contamination detected"):
        assert_zero_odds_contamination(["elo_diff_pre", "market_prob_home"])

    df_with_odds = _make_synthetic_hockey_stage1_with_odds(20).with_columns(
        pl.lit(1.91).alias("closing_odds_home")
    )
    resolved_s1 = Stage1GameOutcomeModel.resolve_feature_columns(df_with_odds)
    assert all(not is_forbidden_market_column(c) for c in resolved_s1)

    stripped = strip_market_columns_from_frame(df_with_odds)
    assert all(not is_forbidden_market_column(c) for c in stripped.columns)

    # Also test fallback branch when only generic numeric + market columns are present
    minimal_s2 = pl.DataFrame(
        {
            "player_id": [1, 2],
            "custom_rate": [1.2, 0.8],
            "market_prob_home": [0.55, 0.45],
            "player_goal_odds": [2.40, 3.10],
        }
    )
    resolved_s2 = Stage2PlayerProjectionModel.resolve_feature_columns(minimal_s2)
    assert resolved_s2 == ["custom_rate"]


def test_mutating_market_cols_changes_zero_predictions_while_updating_benchmark() -> None:
    """Verify mutating market_* columns leaves predictions unchanged while updating benchmark."""
    base_df = _make_synthetic_hockey_stage1_with_odds(n_games=84)

    # Mutate all market_* columns drastically (flipping heavy favorites and underdogs)
    mutated_df = base_df.with_columns(
        pl.lit(4.50, dtype=pl.Float64).alias("market_home_decimal_odds"),
        pl.lit(1.22, dtype=pl.Float64).alias("market_away_decimal_odds"),
        pl.lit(0.20, dtype=pl.Float64).alias("market_prob_home"),
        pl.lit(0.80, dtype=pl.Float64).alias("market_prob_away"),
        pl.lit(0.042, dtype=pl.Float64).alias("market_overround"),
    )

    bt_base = run_stage1_walk_forward_backtest(
        base_df,
        n_splits=2,
        min_train_groups=15,
        random_seed=RANDOM_SEED,
    )
    bt_mutated = run_stage1_walk_forward_backtest(
        mutated_df,
        n_splits=2,
        min_train_groups=15,
        random_seed=RANDOM_SEED,
    )

    # 1. Model's predicted probabilities and expected goals are bit-for-bit identical (0.0 diff)
    p_base = bt_base.oof_predictions["home_win_prob"].to_numpy()
    p_mut = bt_mutated.oof_predictions["home_win_prob"].to_numpy()
    assert float(np.max(np.abs(p_base - p_mut))) == 0.0
    np.testing.assert_array_equal(p_base, p_mut)

    # 2. Meanwhile, the evaluation-only market benchmark report updates to reflect the mutated odds
    assert bt_base.benchmark_report.market_closing_baseline is not None
    assert bt_mutated.benchmark_report.market_closing_baseline is not None
    assert (
        bt_base.benchmark_report.market_closing_baseline.brier_score
        != bt_mutated.benchmark_report.market_closing_baseline.brier_score
    )
    assert bt_base.benchmark_report.market_edge_report is not None
    assert bt_mutated.benchmark_report.market_edge_report is not None
    assert (
        bt_base.benchmark_report.market_edge_report.market_log_loss
        != bt_mutated.benchmark_report.market_edge_report.market_log_loss
    )


def test_nhl_odds_ingestor_and_silver_reconciler_persistence(tmp_path: Path) -> None:
    """Verify NHLOddsIngestor parses CSVs and HockeySilverReconciler persists market_* cols."""
    raw_odds_dir = tmp_path / "raw" / "odds"
    silver_dir = tmp_path / "silver" / "hockey"
    raw_odds_dir.mkdir(parents=True, exist_ok=True)

    csv_content = (
        "game_id,game_date,home_team,away_team,home_moneyline,away_moneyline\n"
        "2024020001,2024-10-10,BOS,TOR,-150,+130\n"
        ",2024-10-11,Colorado Avalanche,Vegas Golden Knights,-125,+105\n"
    )
    (raw_odds_dir / "nhl_historical_odds.csv").write_text(csv_content, encoding="utf-8")

    ingestor = NHLOddsIngestor(cache_dir=raw_odds_dir, api_key=None)
    odds_df = ingestor.load_all_odds(offline=True)
    assert odds_df.height == 2
    assert odds_df["market_prob_home"][0] + odds_df["market_prob_away"][0] == pytest.approx(
        1.0, abs=1e-9
    )
    assert odds_df["market_prob_home"][0] > 0.50

    # Parse a schedule payload and reconcile with odds_df
    sched_payload = {
        "gameWeek": [
            {
                "date": "2024-10-10",
                "games": [
                    {
                        "id": 2024020001,
                        "season": 20242025,
                        "gameType": 2,
                        "startTimeUTC": "2024-10-10T23:00:00Z",
                        "gameState": "FINAL",
                        "homeTeam": {"abbrev": "BOS", "score": 4},
                        "awayTeam": {"abbrev": "TOR", "score": 2},
                    }
                ],
            },
            {
                "date": "2024-10-11",
                "games": [
                    {
                        "id": 2024020002,
                        "season": 20242025,
                        "gameType": 2,
                        "startTimeUTC": "2024-10-11T23:30:00Z",
                        "gameState": "FINAL",
                        "homeTeam": {"abbrev": "COL", "score": 3},
                        "awayTeam": {"abbrev": "VGK", "score": 1},
                    }
                ],
            },
        ]
    }
    games_df = parse_schedule_payload(sched_payload)
    reconciler = HockeySilverReconciler(silver_dir=silver_dir)
    merged = reconciler.reconcile_games(games_df, odds_df=odds_df)

    assert merged.height == 2
    assert merged["market_prob_home"].null_count() == 0
    assert merged["market_home_decimal_odds"][0] == pytest.approx(1.0 + 100.0 / 150.0, abs=1e-6)

    # Also verify parse_the_odds_api_payload helper
    api_events = [
        {
            "commence_time": "2024-10-12T23:00:00Z",
            "home_team": "Boston Bruins",
            "away_team": "Montreal Canadiens",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Bruins", "price": 1.55},
                                {"name": "Montreal Canadiens", "price": 2.60},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    live_parsed = parse_the_odds_api_payload(api_events)
    assert live_parsed.height == 1
    assert live_parsed["home_team"][0] == "BOS"
    assert live_parsed["away_team"][0] == "MTL"
    assert live_parsed["market_prob_home"][0] + live_parsed["market_prob_away"][0] == pytest.approx(
        1.0, abs=1e-9
    )
    assert parse_nhl_historical_odds_csv("").is_empty()
    assert date(2024, 10, 12) == live_parsed["game_date"][0]
