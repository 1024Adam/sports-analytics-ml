"""Strict anti-contamination and closing-odds firewall tests for MLB & NPB Baseball."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from tests.sports.baseball.test_stage1_model import (
    _generate_synthetic_directional_baseball_features,
)

from sports_ml.core.cache import write_gzip_bytes
from sports_ml.core.market import is_forbidden_market_column
from sports_ml.sports.baseball.features import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_BATTER_FEATURE_COLUMNS,
    STAGE2_PITCHER_FEATURE_COLUMNS,
    assert_zero_odds_contamination,
    strip_market_columns_from_frame,
)
from sports_ml.sports.baseball.ingestion import (
    BaseballOddsIngestor,
    parse_baseball_historical_odds_csv,
)
from sports_ml.sports.baseball.models import (
    Stage1BaseballGameModel,
    Stage2BaseballPlayerModel,
    run_baseball_stage1_walk_forward_backtest,
)


def test_baseball_odds_ingestor_2way_mlb_and_3way_npb_power_devigging(tmp_path: Path) -> None:
    """Verify MLB American moneyline and NPB 3-way decimal odds parse and power-de-vig to 1.0."""
    csv_bytes = (
        b"game_id,event_seq,league,game_date,home_team,away_team,home_moneyline,draw_odds,away_moneyline\n"
        b"777001,1,MLB,2025-05-10,LAD,SDP,-160,,+140\n"
        b"9000001,1,NPB,2025-06-01,HAN,YOM,1.80,8.50,2.25\n"
    )
    odds_dir = tmp_path / "odds"
    write_gzip_bytes(odds_dir / "baseball_historical_odds.csv.gz", csv_bytes)

    ingestor = BaseballOddsIngestor(odds_dir)
    df = ingestor.load_cached_odds()
    assert df.height == 2

    mlb_r = df.filter(pl.col("league") == "MLB").row(0, named=True)
    assert mlb_r["market_prob_draw"] is None
    assert pytest.approx(mlb_r["market_prob_home"] + mlb_r["market_prob_away"], abs=1e-9) == 1.0
    assert mlb_r["market_prob_home"] > 0.55

    npb_r = df.filter(pl.col("league") == "NPB").row(0, named=True)
    assert npb_r["market_prob_draw"] is not None
    assert 0.05 < npb_r["market_prob_draw"] < 0.15
    assert (
        pytest.approx(
            npb_r["market_prob_home"] + npb_r["market_prob_draw"] + npb_r["market_prob_away"],
            abs=1e-9,
        )
        == 1.0
    )

    empty_df = parse_baseball_historical_odds_csv(b"game_id,home_team,away_team\n")
    assert empty_df.is_empty()


def test_baseball_feature_lists_never_contain_market_or_odds_columns() -> None:
    """Verify STAGE1/STAGE2 feature lists and resolvers reject all market_* and *odds* columns."""
    for col in (
        *STAGE1_FEATURE_COLUMNS,
        *STAGE2_PITCHER_FEATURE_COLUMNS,
        *STAGE2_BATTER_FEATURE_COLUMNS,
    ):
        assert not is_forbidden_market_column(col), f"Forbidden market/odds column leaked: {col}"

    with pytest.raises(ValueError, match="Market/odds contamination detected"):
        assert_zero_odds_contamination(["diff_elo_pre", "market_prob_home"])

    df = _generate_synthetic_directional_baseball_features(n_mlb=30, n_npb=20).with_columns(
        pl.lit(1.75).alias("market_home_decimal_odds"),
        pl.lit(2.15).alias("closing_moneyline_odds"),
    )
    resolved_s1 = Stage1BaseballGameModel.resolve_feature_columns(df)
    assert all(not is_forbidden_market_column(c) for c in resolved_s1)

    stripped = strip_market_columns_from_frame(df)
    assert all(not is_forbidden_market_column(c) for c in stripped.columns)

    minimal_p = pl.DataFrame(
        {"player_id": [1], "custom_metric": [1.5], "market_prob_home": [0.55], "sp_k_odds": [1.90]}
    )
    assert Stage2BaseballPlayerModel.resolve_pitcher_feature_columns(minimal_p) == ["custom_metric"]
    assert Stage2BaseballPlayerModel.resolve_batter_feature_columns(minimal_p) == ["custom_metric"]


def test_mutating_market_columns_changes_zero_predictions_while_updating_benchmark() -> None:
    """Prove mutating market_* columns in Silver/Gold changes 0.0 of predicted probabilities."""
    base_df = _generate_synthetic_directional_baseball_features(n_mlb=60, n_npb=50)

    # Attach realistic market_* columns for both MLB (2-way) and NPB (3-way)
    with_odds_a = base_df.with_columns(
        pl.lit(1.85).alias("market_home_decimal_odds"),
        pl.when(pl.col("league") == "NPB")
        .then(pl.lit(8.00))
        .otherwise(None)
        .alias("market_draw_decimal_odds"),
        pl.lit(2.05).alias("market_away_decimal_odds"),
        pl.when(pl.col("league") == "NPB")
        .then(pl.lit(0.50))
        .otherwise(pl.lit(0.53))
        .alias("market_prob_home"),
        pl.when(pl.col("league") == "NPB")
        .then(pl.lit(0.10))
        .otherwise(None)
        .alias("market_prob_draw"),
        pl.when(pl.col("league") == "NPB")
        .then(pl.lit(0.40))
        .otherwise(pl.lit(0.47))
        .alias("market_prob_away"),
    )

    # Radically mutate all market_* columns in copy B
    with_odds_b = base_df.with_columns(
        pl.lit(4.50).alias("market_home_decimal_odds"),
        pl.when(pl.col("league") == "NPB")
        .then(pl.lit(3.20))
        .otherwise(None)
        .alias("market_draw_decimal_odds"),
        pl.lit(1.25).alias("market_away_decimal_odds"),
        pl.when(pl.col("league") == "NPB")
        .then(pl.lit(0.15))
        .otherwise(pl.lit(0.20))
        .alias("market_prob_home"),
        pl.when(pl.col("league") == "NPB")
        .then(pl.lit(0.25))
        .otherwise(None)
        .alias("market_prob_draw"),
        pl.when(pl.col("league") == "NPB")
        .then(pl.lit(0.60))
        .otherwise(pl.lit(0.80))
        .alias("market_prob_away"),
    )

    bt_a = run_baseball_stage1_walk_forward_backtest(with_odds_a, n_splits=2, min_train_groups=15)
    bt_b = run_baseball_stage1_walk_forward_backtest(with_odds_b, n_splits=2, min_train_groups=15)

    # Model predictions must be bit-for-bit identical (0.0 diff)!
    for col in (
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "home_dnb_prob",
        "expected_home_runs",
        "expected_away_runs",
    ):
        diff = np.max(
            np.abs(bt_a.oof_predictions[col].to_numpy() - bt_b.oof_predictions[col].to_numpy())
        )
        assert diff == 0.0, f"Market contamination detected in {col}: max diff={diff}"

    # Meanwhile, the evaluation-only market closing baseline and MarketEdgeReport must update!
    assert bt_a.mlb_benchmark_report is not None and bt_b.mlb_benchmark_report is not None
    assert bt_a.mlb_benchmark_report.market_closing_baseline is not None
    assert bt_b.mlb_benchmark_report.market_closing_baseline is not None
    assert (
        bt_a.mlb_benchmark_report.market_closing_baseline.brier_score
        != bt_b.mlb_benchmark_report.market_closing_baseline.brier_score
    )
    assert bt_a.npb_benchmark_report is not None and bt_b.npb_benchmark_report is not None
    assert bt_a.npb_benchmark_report.market_closing_baseline is not None
    assert bt_b.npb_benchmark_report.market_closing_baseline is not None
    assert (
        bt_a.npb_benchmark_report.market_closing_baseline.rps
        != bt_b.npb_benchmark_report.market_closing_baseline.rps
    )
