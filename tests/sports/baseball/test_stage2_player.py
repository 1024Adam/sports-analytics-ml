"""Offline unit tests for Stage 2 Starting Pitcher & Lineup Batter projections."""

from __future__ import annotations

import polars as pl
import pytest
from tests.sports.baseball.test_baseball_features import (
    _build_synthetic_baseball_silver_tables,
)

from sports_ml.sports.baseball.features import (
    build_baseball_stage1_game_features,
    build_baseball_stage2_batter_features,
    build_baseball_stage2_pitcher_features,
)
from sports_ml.sports.baseball.models import (
    Stage2BaseballPlayerModel,
)


def test_stage2_pitcher_and_batter_bounds_and_slot_monotonicity() -> None:
    """Verify non-negativity, probability bounds, and Batting Order Slot 1 > Slot 9 PA."""
    games_df, team_df, pitcher_df, batter_df = _build_synthetic_baseball_silver_tables()
    s1_df = build_baseball_stage1_game_features(games_df, team_df, pitcher_df, batter_df)
    s2_bat = build_baseball_stage2_batter_features(games_df, batter_df, s1_df)
    s2_pit = build_baseball_stage2_pitcher_features(games_df, pitcher_df, s1_df)

    model = Stage2BaseballPlayerModel(gbdt_blend_weight=0.35)
    model.fit(s2_bat, s2_pit)

    bat_preds = model.predict_batters(s2_bat)
    pit_preds = model.predict_pitchers(s2_pit)

    # 1. Non-negativity and bounds on Batter projections
    for r in bat_preds.iter_rows(named=True):
        assert 0.2 <= float(r["proj_pa"]) <= 6.5
        assert float(r["proj_hits"]) >= 0.0
        assert float(r["proj_total_bases"]) >= float(r["proj_hits"])
        assert float(r["proj_hr"]) >= 0.0
        assert float(r["proj_runs"]) >= 0.0
        assert float(r["proj_rbi"]) >= 0.0
        assert float(r["proj_bb"]) >= 0.0
        assert float(r["proj_k"]) >= 0.0
        assert 0.0 <= float(r["prob_1plus_hit"]) <= 1.0
        assert 0.0 <= float(r["prob_2plus_tb"]) <= 1.0
        assert 0.0 <= float(r["prob_1plus_hr"]) <= 1.0

    # 2. Non-negativity and bounds on Starting Pitcher projections
    for r in pit_preds.iter_rows(named=True):
        assert 0.0 <= float(r["proj_outs_pitched"]) <= 27.0
        assert pytest.approx(float(r["proj_ip"]), abs=1e-6) == float(r["proj_outs_pitched"]) / 3.0
        assert float(r["proj_batters_faced"]) >= float(r["proj_outs_pitched"])
        assert float(r["proj_k"]) >= 0.0
        assert float(r["proj_bb"]) >= 0.0
        assert float(r["proj_hits_allowed"]) >= 0.0
        assert float(r["proj_er"]) >= 0.0
        assert 0.0 <= float(r["prob_5plus_k"]) <= 1.0
        assert 0.0 <= float(r["prob_6plus_k"]) <= float(r["prob_5plus_k"]) + 1e-9
        assert 0.0 <= float(r["prob_7plus_k"]) <= float(r["prob_6plus_k"]) + 1e-9
        assert 0.0 <= float(r["prob_quality_start"]) <= 1.0

    # 3. Batting Order Slot Monotonicity: identical batter in Slot 1 vs Slot 9
    base_b = s2_bat.head(1)
    slot1_b = base_b.with_columns(
        pl.lit(1, dtype=pl.Int8).alias("batting_order_slot"),
        pl.lit(1.0, dtype=pl.Float64).alias("batting_slot_num"),
    )
    slot9_b = base_b.with_columns(
        pl.lit(9, dtype=pl.Int8).alias("batting_order_slot"),
        pl.lit(9.0, dtype=pl.Float64).alias("batting_slot_num"),
    )
    pred_s1 = model.predict_batters(slot1_b).row(0, named=True)
    pred_s9 = model.predict_batters(slot9_b).row(0, named=True)
    assert float(pred_s1["proj_pa"]) > float(pred_s9["proj_pa"])


def test_stage_conditioning_park_factor_and_strikeout_monotonicity() -> None:
    """Verify Stage 1 expected runs, park factor, and SP K% shift Stage 2 monotonically."""
    games_df, team_df, pitcher_df, batter_df = _build_synthetic_baseball_silver_tables()
    s1_df = build_baseball_stage1_game_features(games_df, team_df, pitcher_df, batter_df)
    s2_bat = build_baseball_stage2_batter_features(games_df, batter_df, s1_df)
    s2_pit = build_baseball_stage2_pitcher_features(games_df, pitcher_df, s1_df)

    model = Stage2BaseballPlayerModel(gbdt_blend_weight=0.20)
    model.fit(s2_bat, s2_pit)

    game1_bat = s2_bat.filter((pl.col("game_id") == 1001) & (pl.col("team") == "LAD"))

    # 1. Stage 1 Team Expected Runs Monotonicity (3.0 runs vs 6.5 runs)
    low_env = game1_bat.with_columns(
        pl.lit(3.0, dtype=pl.Float64).alias("stage1_team_expected_runs"),
        pl.lit(0.86, dtype=pl.Float64).alias("venue_run_factor"),
    )
    high_env = game1_bat.with_columns(
        pl.lit(6.5, dtype=pl.Float64).alias("stage1_team_expected_runs"),
        pl.lit(1.25, dtype=pl.Float64).alias("venue_run_factor"),
    )
    preds_low = model.predict_batters(low_env)
    preds_high = model.predict_batters(high_env)

    assert float(preds_high["proj_runs"].sum()) > float(preds_low["proj_runs"].sum())
    assert float(preds_high["proj_rbi"].sum()) > float(preds_low["proj_rbi"].sum())
    assert float(preds_high["proj_total_bases"].sum()) > float(preds_low["proj_total_bases"].sum())

    # 2. Facing a high-K% ace increases opposing batter proj_k and starter prob_6plus_k
    bat_vs_contact_sp = game1_bat.with_columns(
        pl.lit(0.150, dtype=pl.Float64).alias("opp_sp_k_pct_shrunk")
    )
    bat_vs_ace_sp = game1_bat.with_columns(
        pl.lit(0.320, dtype=pl.Float64).alias("opp_sp_k_pct_shrunk")
    )
    assert float(model.predict_batters(bat_vs_ace_sp)["proj_k"].sum()) > float(
        model.predict_batters(bat_vs_contact_sp)["proj_k"].sum()
    )

    sp1 = s2_pit.head(1)
    sp_low_k = sp1.with_columns(pl.lit(0.150, dtype=pl.Float64).alias("sp_k_pct_shrunk"))
    sp_high_k = sp1.with_columns(pl.lit(0.320, dtype=pl.Float64).alias("sp_k_pct_shrunk"))
    p_low = model.predict_pitchers(sp_low_k).row(0, named=True)
    p_high = model.predict_pitchers(sp_high_k).row(0, named=True)
    assert float(p_high["proj_k"]) > float(p_low["proj_k"])
    assert float(p_high["prob_6plus_k"]) > float(p_low["prob_6plus_k"])
