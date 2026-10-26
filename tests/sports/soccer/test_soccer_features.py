"""Strict anti-leakage, cold-start/promoted-club, and lineup/injury tests for EPL features."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.sports.soccer.features import (
    GoalkeeperStarterEloHook,
    build_goalkeeper_rolling_features,
    build_soccer_roster_strength_features,
    build_soccer_stage1_match_features,
    build_soccer_stage2_player_features,
    build_soccer_team_rolling_features,
)


def _synthetic_epl_dataset() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create a multi-season + double-gameweek EPL fixture dataset for feature testing."""
    matches_df = pl.DataFrame(
        {
            "sport": ["soccer"] * 5,
            "season": [20232024, 20232024, 20242025, 20242025, 20242025],
            "matchweek": [1, 2, 1, 2, 2],
            "match_id": [2023001, 2023002, 2024001, 2024002, 2024003],
            "event_seq": [1, 1, 1, 1, 2],  # Double-gameweek sequence on 2024-08-24
            "match_date": [
                date(2023, 8, 12),
                date(2023, 8, 19),
                date(2024, 8, 17),
                date(2024, 8, 24),
                date(2024, 8, 24),
            ],
            "start_time_utc": [
                datetime(2023, 8, 12, 14, 0, tzinfo=UTC),
                datetime(2023, 8, 19, 14, 0, tzinfo=UTC),
                datetime(2024, 8, 17, 14, 0, tzinfo=UTC),
                datetime(2024, 8, 24, 11, 30, tzinfo=UTC),
                datetime(2024, 8, 24, 16, 30, tzinfo=UTC),
            ],
            "home_team": ["ARS", "MCI", "ARS", "IPS", "ARS"],
            "away_team": ["MCI", "ARS", "IPS", "ARS", "MCI"],
            "venue_timezone": ["Europe/London"] * 5,
            "game_state": ["FINAL"] * 5,
            "home_goals": [2, 1, 3, 0, 2],
            "away_goals": [1, 1, 0, 2, 1],
            "outcome_1x2": [0, 1, 0, 2, 0],
            "home_win": [1, 0, 1, 0, 1],
            "is_draw": [0, 1, 0, 0, 0],
            "away_win": [0, 0, 0, 1, 0],
        }
    )

    team_rows = []
    for row in matches_df.iter_rows(named=True):
        mid = int(row["match_id"])
        mdate = row["match_date"]
        s_utc = row["start_time_utc"]
        s_val = int(row["season"])
        ev_seq = int(row["event_seq"])
        ht = str(row["home_team"])
        at = str(row["away_team"])
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])

        team_rows.append(
            {
                "match_id": mid,
                "event_seq": ev_seq,
                "match_date": mdate,
                "start_time_utc": s_utc,
                "season": s_val,
                "team": ht,
                "opponent": at,
                "is_home": True,
                "goals_for": hg,
                "goals_against": ag,
                "shots_for": 15,
                "shots_against": 10,
                "sot_for": 6,
                "sot_against": 4,
                "xgf": 1.85,
                "xga": 1.05,
                "npxgf": 1.70,
                "npxga": 1.05,
                "ppda": 9.5,
                "opp_ppda": 13.0,
                "deep_completions": 8,
                "opp_deep_completions": 4,
                "corners_for": 7,
                "corners_against": 3,
                "fouls_for": 10,
                "fouls_against": 12,
                "yellow_cards": 1,
                "red_cards": 0,
                "has_understat_xg": True,
            }
        )
        team_rows.append(
            {
                "match_id": mid,
                "event_seq": ev_seq,
                "match_date": mdate,
                "start_time_utc": s_utc,
                "season": s_val,
                "team": at,
                "opponent": ht,
                "is_home": False,
                "goals_for": ag,
                "goals_against": hg,
                "shots_for": 10,
                "shots_against": 15,
                "sot_for": 4,
                "sot_against": 6,
                "xgf": 1.05,
                "xga": 1.85,
                "npxgf": 1.05,
                "npxga": 1.70,
                "ppda": 13.0,
                "opp_ppda": 9.5,
                "deep_completions": 4,
                "opp_deep_completions": 8,
                "corners_for": 3,
                "corners_against": 7,
                "fouls_for": 12,
                "fouls_against": 10,
                "yellow_cards": 2,
                "red_cards": 0,
                "has_understat_xg": True,
            }
        )
    team_logs_df = pl.DataFrame(team_rows)

    outfield_rows = []
    gk_rows = []
    for row in matches_df.iter_rows(named=True):
        mid = int(row["match_id"])
        mdate = row["match_date"]
        s_utc = row["start_time_utc"]
        s_val = int(row["season"])
        ev_seq = int(row["event_seq"])
        ht = str(row["home_team"])
        at = str(row["away_team"])

        for side_idx, (team_code, opp_code) in enumerate([(ht, at), (at, ht)]):
            is_h = side_idx == 0
            base_pid = {"ARS": 1000, "MCI": 2000, "IPS": 3000}[team_code]
            gk_rows.append(
                {
                    "match_id": mid,
                    "event_seq": ev_seq,
                    "match_date": mdate,
                    "start_time_utc": s_utc,
                    "season": s_val,
                    "player_id": base_pid + 1,
                    "player_name": f"{team_code} GK",
                    "team": team_code,
                    "opponent": opp_code,
                    "is_home": is_h,
                    "is_starter": True,
                    "minutes": 90.0,
                    "shots_on_target_against": 4,
                    "saves": 3,
                    "goals_against": 1,
                    "clean_sheet": 0,
                    "xga_faced": 1.25,
                    "gsaa": 0.25,
                }
            )
            for pos_idx, pos_grp in enumerate(["FWD", "MID", "DEF"], start=2):
                outfield_rows.append(
                    {
                        "match_id": mid,
                        "event_seq": ev_seq,
                        "match_date": mdate,
                        "start_time_utc": s_utc,
                        "season": s_val,
                        "player_id": base_pid + pos_idx,
                        "player_name": f"{team_code} {pos_grp}",
                        "team": team_code,
                        "opponent": opp_code,
                        "is_home": is_h,
                        "position_group": pos_grp,
                        "is_starter": True,
                        "minutes": 90.0,
                        "goals": 1 if pos_grp == "FWD" else 0,
                        "assists": 1 if pos_grp == "MID" else 0,
                        "shots": 3 if pos_grp == "FWD" else 1,
                        "sot": 2 if pos_grp == "FWD" else 0,
                        "key_passes": 2 if pos_grp == "MID" else 1,
                        "xg": 0.55 if pos_grp == "FWD" else 0.15,
                        "npxg": 0.45 if pos_grp == "FWD" else 0.15,
                        "xa": 0.30 if pos_grp == "MID" else 0.10,
                        "xg_chain": 0.75,
                        "xg_buildup": 0.35,
                        "yellow_cards": 0,
                        "red_cards": 0,
                    }
                )

    outfield_logs_df = pl.DataFrame(outfield_rows)
    gk_logs_df = pl.DataFrame(gk_rows)
    return matches_df, team_logs_df, outfield_logs_df, gk_logs_df


def test_soccer_features_future_mutation_invariance() -> None:
    """Mutating match goals, xG, cards, or player stats at >= t must not alter features at < t."""
    matches_df, team_logs_df, outfield_logs_df, gk_logs_df = _synthetic_epl_dataset()

    base_team = build_soccer_team_rolling_features(team_logs_df)
    base_gk = build_goalkeeper_rolling_features(gk_logs_df)
    base_roster = build_soccer_roster_strength_features(outfield_logs_df)
    base_stage1 = build_soccer_stage1_match_features(
        matches_df, team_logs_df, outfield_logs_df, gk_logs_df
    )
    base_stage2 = build_soccer_stage2_player_features(outfield_logs_df, base_team, base_gk)

    # Corrupt all matches at or after double-gameweek event_seq == 2 on 2024-08-24 16:30 UTC
    cutoff_utc = datetime(2024, 8, 24, 16, 30, tzinfo=UTC)
    corrupted_matches = matches_df.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(9, dtype=pl.Int16))
        .otherwise(pl.col("home_goals"))
        .alias("home_goals"),
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(8, dtype=pl.Int16))
        .otherwise(pl.col("away_goals"))
        .alias("away_goals"),
    )
    corrupted_team_logs = team_logs_df.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(9, dtype=pl.Int16))
        .otherwise(pl.col("goals_for"))
        .alias("goals_for"),
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(7.5, dtype=pl.Float64))
        .otherwise(pl.col("npxgf"))
        .alias("npxgf"),
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(3, dtype=pl.Int16))
        .otherwise(pl.col("red_cards"))
        .alias("red_cards"),
    )
    corrupted_outfield = outfield_logs_df.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(5, dtype=pl.Int16))
        .otherwise(pl.col("goals"))
        .alias("goals"),
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(4.2, dtype=pl.Float64))
        .otherwise(pl.col("npxg"))
        .alias("npxg"),
    )
    corrupted_gk = gk_logs_df.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(8, dtype=pl.Int16))
        .otherwise(pl.col("goals_against"))
        .alias("goals_against"),
        pl.when(pl.col("start_time_utc") >= cutoff_utc)
        .then(pl.lit(-5.0, dtype=pl.Float64))
        .otherwise(pl.col("gsaa"))
        .alias("gsaa"),
    )

    mut_team = build_soccer_team_rolling_features(corrupted_team_logs)
    mut_gk = build_goalkeeper_rolling_features(corrupted_gk)
    mut_roster = build_soccer_roster_strength_features(corrupted_outfield)
    mut_stage1 = build_soccer_stage1_match_features(
        corrupted_matches, corrupted_team_logs, corrupted_outfield, corrupted_gk
    )
    mut_stage2 = build_soccer_stage2_player_features(corrupted_outfield, mut_team, mut_gk)

    # All pre-match features strictly before cutoff_utc (including double-GW match 1 at 11:30 UTC)
    # must be bit-for-bit identical!
    assert_frame_equal(
        base_team.filter(pl.col("start_time_utc") < cutoff_utc),
        mut_team.filter(pl.col("start_time_utc") < cutoff_utc),
    )
    assert_frame_equal(
        base_gk.filter(pl.col("start_time_utc") < cutoff_utc),
        mut_gk.filter(pl.col("start_time_utc") < cutoff_utc),
    )
    assert_frame_equal(
        base_roster.filter(pl.col("match_id") != 2024003),
        mut_roster.filter(pl.col("match_id") != 2024003),
    )
    pre_s1_cols = [
        c
        for c in base_stage1.columns
        if c not in {"home_goals", "away_goals", "elo_home_post", "elo_away_post"}
    ]
    assert_frame_equal(
        base_stage1.filter(pl.col("start_time_utc") <= cutoff_utc).select(pre_s1_cols),
        mut_stage1.filter(pl.col("start_time_utc") <= cutoff_utc).select(pre_s1_cols),
    )
    pre_s2_cols = [
        c for c in base_stage2.columns if c not in {"goals", "npxg", "assists", "shots", "sot"}
    ]
    assert_frame_equal(
        base_stage2.filter(pl.col("start_time_utc") <= cutoff_utc).select(pre_s2_cols),
        mut_stage2.filter(pl.col("start_time_utc") <= cutoff_utc).select(pre_s2_cols),
    )


def test_cold_start_and_promoted_club_priors() -> None:
    """Verify Matchweek 1, promoted clubs, and debuting players produce finite shrunk features."""
    matches_df, team_logs_df, outfield_logs_df, gk_logs_df = _synthetic_epl_dataset()
    stage1_df = build_soccer_stage1_match_features(
        matches_df, team_logs_df, outfield_logs_df, gk_logs_df
    )
    team_feat = build_soccer_team_rolling_features(team_logs_df)
    gk_feat = build_goalkeeper_rolling_features(gk_logs_df)
    stage2_df = build_soccer_stage2_player_features(outfield_logs_df, team_feat, gk_feat)

    # Zero nulls across all numeric feature columns on Matchweek 1 and promoted-club debuts
    for col, dtype in stage1_df.schema.items():
        if dtype.is_numeric():
            assert stage1_df[col].null_count() == 0, f"Nulls found in Stage 1 column {col}"

    for col, dtype in stage2_df.schema.items():
        if dtype.is_numeric():
            assert stage2_df[col].null_count() == 0, f"Nulls found in Stage 2 column {col}"

    # Verify IPS is flagged as a promoted club in 20242025 (since it did not play in 20232024)
    m_2024001 = stage1_df.filter(pl.col("match_id") == 2024001).to_dicts()[0]
    assert m_2024001["is_promoted_team_home"] is False  # ARS
    assert m_2024001["is_promoted_team_away"] is True  # IPS
    assert m_2024001["elo_away_pre"] == pytest.approx(1390.0)
    assert m_2024001["npxgf_per_90_w10_away"] < m_2024001["npxgf_per_90_w10_home"]

    # Verify GoalkeeperStarterEloHook returns finite Elo adjustment
    hook = GoalkeeperStarterEloHook(gk_feat)
    assert isinstance(hook.get_starter_elo_delta(2024001, "ARS"), float)


def test_injury_and_suspension_redistribution_in_roster_and_stage2() -> None:
    """Verify marking a striker/penalty taker unavailable updates missing minutes & set pieces."""
    _, team_logs_df, outfield_logs_df, gk_logs_df = _synthetic_epl_dataset()
    team_feat = build_soccer_team_rolling_features(team_logs_df)
    gk_feat = build_goalkeeper_rolling_features(gk_logs_df)

    squad_meta = pl.DataFrame(
        {
            "player_id": [1002, 1003, 1004],
            "team": ["ARS", "ARS", "ARS"],
            "chance_of_playing_next_round": [0, 100, 100],  # 1002 (ARS FWD, pen order 1) suspended
            "penalties_order": [1, 2, 99],
            "direct_freekicks_order": [2, 1, 99],
            "corners_order": [99, 1, 99],
        }
    )

    excluded_map = {(2024003, "ARS"): [1002]}
    roster_normal = build_soccer_roster_strength_features(outfield_logs_df)
    roster_missing = build_soccer_roster_strength_features(
        outfield_logs_df, excluded_ids_by_match_team=excluded_map
    )

    norm_row = roster_normal.filter(
        (pl.col("match_id") == 2024003) & (pl.col("team") == "ARS")
    ).to_dicts()[0]
    miss_row = roster_missing.filter(
        (pl.col("match_id") == 2024003) & (pl.col("team") == "ARS")
    ).to_dicts()[0]

    assert norm_row["missing_regular_minutes"] == pytest.approx(0.0)
    assert miss_row["missing_regular_minutes"] > 60.0
    assert miss_row["missing_regular_minutes_fwd"] > 60.0

    # Stage 2 set-piece redistribution: with player 1002 out, player 1003 becomes penalty taker
    s2_df = build_soccer_stage2_player_features(
        outfield_logs_df,
        team_feat,
        gk_feat,
        squad_metadata_df=squad_meta,
        excluded_ids_by_match_team=excluded_map,
    )
    m3_ars = s2_df.filter((pl.col("match_id") == 2024003) & (pl.col("team") == "ARS"))
    p1002 = m3_ars.filter(pl.col("player_id") == 1002).to_dicts()[0]
    p1003 = m3_ars.filter(pl.col("player_id") == 1003).to_dicts()[0]

    assert p1002["availability_weight"] == pytest.approx(0.0)
    assert p1002["is_penalty_taker"] is False
    assert p1002["team_injury_vacated_minutes_pos"] > 60.0
    assert p1003["is_penalty_taker"] is True
    assert p1003["is_set_piece_taker"] is True
