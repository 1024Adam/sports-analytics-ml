"""Stage 2 EPL outfield & goalkeeper projection model tests (`test_stage2_player.py`)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from sports_ml.config import RANDOM_SEED
from sports_ml.sports.soccer.models.stage2_player import (
    Stage2SoccerPlayerModel,
    reconcile_outfield_team_goals,
)


def _synthetic_soccer_stage2_df(n_matches: int = 20) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic outfield and goalkeeper feature frames for Stage 2 testing."""
    rng = np.random.default_rng(RANDOM_SEED)
    base_time = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)

    outfield_rows = []
    gk_rows = []
    for m_idx in range(n_matches):
        match_id = 2024001 + m_idx
        ts = base_time + timedelta(days=m_idx * 3)
        m_date = ts.date()

        for side_idx, (team, opp) in enumerate([("ARS", "MCI"), ("MCI", "ARS")]):
            is_home = side_idx == 0
            base_pid = 1000 if team == "ARS" else 2000

            gk_rows.append(
                {
                    "match_id": match_id,
                    "event_id": match_id,
                    "match_date": m_date,
                    "start_time_utc": ts,
                    "season": 20242025,
                    "player_id": base_pid + 1,
                    "player_name": f"{team} GK",
                    "team": team,
                    "opponent": opp,
                    "is_home": is_home,
                    "is_starter": True,
                    "minutes": 90.0,
                    "gsaa_per_90_shrunk": 0.15 if team == "ARS" else -0.05,
                    "save_pct_shrunk": 0.74 if team == "ARS" else 0.69,
                }
            )

            for p_idx, pos in enumerate(["FWD", "FWD", "MID", "MID", "DEF", "DEF"], start=2):
                is_fwd = pos == "FWD"
                outfield_rows.append(
                    {
                        "match_id": match_id,
                        "event_id": match_id,
                        "event_seq": 1,
                        "match_date": m_date,
                        "start_time_utc": ts,
                        "season": 20242025,
                        "player_id": base_pid + p_idx,
                        "player_name": f"{team}_{pos}_{p_idx}",
                        "team": team,
                        "opponent": opp,
                        "is_home": is_home,
                        "position_group": pos,
                        "is_starter": True,
                        "minutes": float(rng.uniform(70.0, 90.0)),
                        "goals": int(rng.poisson(0.45 if is_fwd else 0.12)),
                        "assists": int(rng.poisson(0.25 if pos != "DEF" else 0.08)),
                        "shots": int(rng.poisson(2.8 if is_fwd else 1.1)),
                        "sot": int(rng.poisson(1.2 if is_fwd else 0.4)),
                        "rolling_minutes": 82.0 if is_fwd else 85.0,
                        "rolling_start_rate": 0.90,
                        "availability_weight": 1.0,
                        "team_injury_vacated_minutes_pos": 0.0,
                        "rolling_npxg_per_90": 0.48 if is_fwd else 0.14,
                        "rolling_xg_per_90": 0.55 if is_fwd else 0.14,
                        "rolling_xa_per_90": 0.22 if pos != "DEF" else 0.08,
                        "rolling_xg_chain_per_90": 0.72 if is_fwd else 0.45,
                        "rolling_xg_buildup_per_90": 0.25 if is_fwd else 0.35,
                        "rolling_shots_per_90": 3.10 if is_fwd else 1.20,
                        "rolling_sot_per_90": 1.35 if is_fwd else 0.42,
                        "rolling_key_passes_per_90": 1.40 if pos != "DEF" else 0.60,
                        "rolling_goals_per_90": 0.50 if is_fwd else 0.12,
                        "rolling_assists_per_90": 0.22 if pos != "DEF" else 0.08,
                        "career_shot_conversion_shrunk": 0.16 if is_fwd else 0.09,
                        "is_penalty_taker": p_idx == 2,
                        "is_set_piece_taker": p_idx == 4,
                        "opp_xga_per_90": 1.45,
                        "opp_sot_against_per_90": 4.5,
                        "opp_ppda": 11.5,
                        "opp_gk_gsaa_per_90_shrunk": 0.0,
                        "stage1_team_win_prob": 0.45 if is_home else 0.30,
                        "stage1_draw_prob": 0.25,
                        "stage1_expected_team_goals": 1.65 if is_home else 1.25,
                        "team_rest_days": 6,
                        "is_midweek_turnaround": False,
                    }
                )

    return pl.DataFrame(outfield_rows), pl.DataFrame(gk_rows)


def test_stage2_soccer_bounds_and_stage_conditioning_monotonicity() -> None:
    """Verify non-negativity, minutes in [0, 90], probability bounds, and Stage 1 monotonicity."""
    outfield_df, gk_df = _synthetic_soccer_stage2_df(n_matches=18)
    model = Stage2SoccerPlayerModel(reconciliation_gamma=0.50, random_seed=RANDOM_SEED)
    model.fit(outfield_df)

    test_match = outfield_df.filter(pl.col("match_id") == 2024018)
    preds_low = model.predict(
        test_match.with_columns(
            pl.lit(1.05, dtype=pl.Float64).alias("stage1_expected_team_goals"),
            pl.lit(0.35, dtype=pl.Float64).alias("opp_gk_gsaa_per_90_shrunk"),
        )
    )
    preds_high = model.predict(
        test_match.with_columns(
            pl.lit(2.40, dtype=pl.Float64).alias("stage1_expected_team_goals"),
            pl.lit(-0.35, dtype=pl.Float64).alias("opp_gk_gsaa_per_90_shrunk"),
        )
    )

    for df in (preds_low, preds_high):
        min_arr = df["proj_minutes"].to_numpy()
        assert np.all((min_arr >= 0.0) & (min_arr <= 90.0))
        assert np.all(df["proj_goals"].to_numpy() >= 0.0)
        assert np.all(df["proj_assists"].to_numpy() >= 0.0)
        assert np.all(df["proj_goal_involvements"].to_numpy() >= 0.0)
        assert np.all(df["proj_shots"].to_numpy() >= 0.0)
        assert np.all(df["proj_sot"].to_numpy() >= 0.0)
        for p_col in ("prob_1plus_goal", "prob_1plus_assist", "prob_1plus_gi"):
            arr = df[p_col].to_numpy()
            assert np.all((arr >= 0.0) & (arr <= 1.0))

    # Stage-conditioning monotonicity: higher Stage 1 team goals + weaker opp GK increases goals
    assert float(preds_high["proj_goals"].sum()) > float(preds_low["proj_goals"].sum())
    assert float(preds_high["proj_goal_involvements"].sum()) > float(
        preds_low["proj_goal_involvements"].sum()
    )

    # Goalkeeper clean-sheet monotonicity: higher opponent expected goals lowers P(Clean Sheet)
    test_gk = gk_df.filter(pl.col("match_id") == 2024018)
    s1_low_opp = pl.DataFrame(
        {
            "match_id": [2024018],
            "expected_home_goals": [0.85],
            "expected_away_goals": [0.85],
        }
    )
    s1_high_opp = pl.DataFrame(
        {
            "match_id": [2024018],
            "expected_home_goals": [2.45],
            "expected_away_goals": [2.45],
        }
    )
    gk_low_opp = model.predict_goalkeepers(
        test_gk, outfield_projections_df=preds_low, stage1_predictions_df=s1_low_opp
    )
    gk_high_opp = model.predict_goalkeepers(
        test_gk, outfield_projections_df=preds_high, stage1_predictions_df=s1_high_opp
    )

    for g_df in (gk_low_opp, gk_high_opp):
        assert np.all(g_df["proj_saves"].to_numpy() >= 0.0)
        assert np.all(g_df["proj_goals_against"].to_numpy() >= 0.0)
        sv_arr = g_df["proj_save_pct"].to_numpy()
        assert np.all((sv_arr >= 0.0) & (sv_arr <= 1.0))
        assert np.all(
            (g_df["prob_clean_sheet"].to_numpy() >= 0.0)
            & (g_df["prob_clean_sheet"].to_numpy() <= 1.0)
        )

    assert np.all(
        gk_low_opp["prob_clean_sheet"].to_numpy() > gk_high_opp["prob_clean_sheet"].to_numpy()
    )

    # Verify reconcile_outfield_team_goals gamma validation
    with pytest.raises(ValueError, match="gamma"):
        reconcile_outfield_team_goals(preds_low, gamma=1.5)
    _ = date(2024, 8, 17)
