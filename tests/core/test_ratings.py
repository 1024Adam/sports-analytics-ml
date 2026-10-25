"""Dynamic Elo and multi-entrant rating engine tests (`tests/core/test_ratings.py`)."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.core.ratings import (
    EloConfig,
    MultiEntrantSkillEngine,
    PairwiseEloEngine,
)


def _sample_games_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [20232024, 20232024, 20232024, 20242025],
            "event_id": [2023020001, 2023020002, 2023020003, 2024020001],
            "event_seq": [1, 1, 1, 1],
            "start_time_utc": [
                datetime(2023, 10, 10, 23, 0, tzinfo=UTC),
                datetime(2023, 10, 12, 23, 0, tzinfo=UTC),
                datetime(2023, 10, 14, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 10, 23, 0, tzinfo=UTC),
            ],
            "home_team": ["BOS", "TOR", "BOS", "BOS"],
            "away_team": ["TOR", "BOS", "TOR", "TOR"],
            "home_goals": [4, 2, 3, 2],
            "away_goals": [1, 3, 2, 1],
            "finish_period_type": ["REG", "OT", "SO", "REG"],
            "game_state": ["FINAL", "FINAL", "FINAL", "FINAL"],
        }
    )


def test_elo_zero_sum_hfa_and_offseason_regression() -> None:
    """Verify Elo zero-sum conservation, HFA shift, shootout dampening & offseason regression."""
    engine = PairwiseEloEngine(
        EloConfig(home_field_advantage=35.0, k_factor=6.0, offseason_retention=0.67)
    )
    rated = engine.compute_ratings(_sample_games_df())

    # 1. Pre-game Game 1: both teams start at 1500.0, HFA = 35.0 -> P(Home) > 0.50
    assert rated["elo_home_pre"][0] == pytest.approx(1500.0)
    assert rated["elo_away_pre"][0] == pytest.approx(1500.0)
    assert rated["elo_diff_pre"][0] == pytest.approx(35.0)
    assert rated["elo_home_win_prob"][0] > 0.50
    assert rated["elo_home_win_prob"][0] + rated["elo_away_win_prob"][0] == pytest.approx(1.0)

    # 2. Strict zero-sum conservation across every game: delta_home + delta_away == 0
    home_deltas = rated["elo_home_post"] - rated["elo_home_pre"]
    away_deltas = rated["elo_away_post"] - rated["elo_away_pre"]
    for h_d, a_d in zip(home_deltas.to_list(), away_deltas.to_list(), strict=True):
        assert h_d + a_d == pytest.approx(0.0, abs=1e-12)
        assert rated["elo_home_post"][0] + rated["elo_away_post"][0] == pytest.approx(3000.0)

    # 3. Off-season regression between 20232024 (row 2) and 20242025 (row 3):
    bos_end_2324 = rated["elo_home_post"][2]
    bos_start_2425 = rated["elo_home_pre"][3]
    expected_regressed = 0.67 * bos_end_2324 + 0.33 * 1500.0
    assert bos_start_2425 == pytest.approx(expected_regressed)
    assert abs(bos_start_2425 - 1500.0) < abs(bos_end_2324 - 1500.0)


def test_elo_shootout_dampening_and_mov() -> None:
    """Verify shootout win transfers smaller Elo delta than regulation win."""
    engine = PairwiseEloEngine(EloConfig(use_mov_multiplier=True))
    reg_game = pl.DataFrame(
        {
            "season": [20242025],
            "event_id": [1],
            "event_seq": [1],
            "start_time_utc": [datetime(2024, 10, 10, 23, 0, tzinfo=UTC)],
            "home_team": ["BOS"],
            "away_team": ["TOR"],
            "home_goals": [3],
            "away_goals": [2],
            "finish_period_type": ["REG"],
            "game_state": ["FINAL"],
        }
    )
    so_game = reg_game.with_columns(pl.lit("SO").alias("finish_period_type"))
    blowout_game = reg_game.with_columns(pl.lit(6).alias("home_goals"))

    reg_res = engine.compute_ratings(reg_game)
    so_res = engine.compute_ratings(so_game)
    blowout_res = engine.compute_ratings(blowout_game)

    reg_delta = reg_res["elo_home_post"][0] - reg_res["elo_home_pre"][0]
    so_delta = so_res["elo_home_post"][0] - so_res["elo_home_pre"][0]
    blowout_delta = blowout_res["elo_home_post"][0] - blowout_res["elo_home_pre"][0]

    assert 0.0 < so_delta < reg_delta < blowout_delta


def test_elo_future_mutation_invariance() -> None:
    """Mutating scores at index >= t must never change pre-game Elo at index <= t."""
    engine = PairwiseEloEngine()
    games = _sample_games_df()
    base = engine.compute_ratings(games)

    cutoff = datetime(2023, 10, 14, 23, 0, tzinfo=UTC)
    corrupted = games.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff)
        .then(pl.lit(0))
        .otherwise(pl.col("home_goals"))
        .alias("home_goals"),
        pl.when(pl.col("start_time_utc") >= cutoff)
        .then(pl.lit(15))
        .otherwise(pl.col("away_goals"))
        .alias("away_goals"),
    )
    mutated = engine.compute_ratings(corrupted)

    pre_cols = [
        "event_id",
        "elo_home_pre",
        "elo_away_pre",
        "elo_diff_pre",
        "elo_home_win_prob",
        "elo_away_win_prob",
    ]
    assert_frame_equal(
        base.filter(pl.col("start_time_utc") <= cutoff).select(pre_cols),
        mutated.filter(pl.col("start_time_utc") <= cutoff).select(pre_cols),
    )


def test_multi_entrant_field_skill_engine() -> None:
    """Verify multi-entrant tournament field win probabilities sum to 1.0 and are leak-free."""
    field_df = pl.DataFrame(
        {
            "season": [2025] * 6,
            "event_id": [501, 501, 501, 501, 501, 501],
            "event_seq": [1, 1, 1, 2, 2, 2],
            "start_time_utc": [
                datetime(2025, 4, 10, 12, 0, tzinfo=UTC),
                datetime(2025, 4, 10, 12, 0, tzinfo=UTC),
                datetime(2025, 4, 10, 12, 0, tzinfo=UTC),
                datetime(2025, 4, 11, 12, 0, tzinfo=UTC),
                datetime(2025, 4, 11, 12, 0, tzinfo=UTC),
                datetime(2025, 4, 11, 12, 0, tzinfo=UTC),
            ],
            "entity_id": ["G1", "G2", "G3", "G1", "G2", "G3"],
            "strokes_gained": [3.5, 0.0, -3.5, 1.0, 0.5, -1.5],
        }
    )

    engine = MultiEntrantSkillEngine()
    rated = engine.compute_ratings(field_df)

    # Probabilities in each round sum to 1.0
    r1 = rated.filter(pl.col("event_seq") == 1)
    r2 = rated.filter(pl.col("event_seq") == 2)
    assert r1["field_win_prob"].sum() == pytest.approx(1.0)
    assert r2["field_win_prob"].sum() == pytest.approx(1.0)

    # G1 had +3.5 SG in Round 1, so entering Round 2 G1 has highest pre-round skill & win prob
    g1_r2 = r2.filter(pl.col("entity_id") == "G1")
    g3_r2 = r2.filter(pl.col("entity_id") == "G3")
    assert g1_r2["skill_rating_pre"][0] > g3_r2["skill_rating_pre"][0]
    assert g1_r2["field_win_prob"][0] > g3_r2["field_win_prob"][0]
