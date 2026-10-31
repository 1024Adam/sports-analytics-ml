"""Offline unit and anti-leakage tests for Baseball Stage 1 & Stage 2 feature engineering."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import polars as pl
import pytest

from sports_ml.sports.baseball.features import (
    STAGE1_FEATURE_COLUMNS,
    STAGE2_BATTER_FEATURE_COLUMNS,
    STAGE2_PITCHER_FEATURE_COLUMNS,
    assert_zero_odds_contamination,
    build_baseball_stage1_game_features,
    build_baseball_stage2_batter_features,
    build_baseball_stage2_pitcher_features,
    build_baseball_team_rolling_features,
    resolve_batter_prior,
    resolve_pitcher_prior,
)


def _build_synthetic_baseball_silver_tables() -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame
]:
    """Build deterministic multi-game MLB (with doubleheader) + NPB Silver tables."""
    games_rows: list[dict[str, Any]] = [
        # Game 1: Opening Day MLB (LAD vs SDP)
        {
            "sport": "baseball",
            "league": "MLB",
            "sub_league": "NL",
            "season": 2025,
            "game_id": 1001,
            "event_seq": 1,
            "game_date": date(2025, 5, 1),
            "start_time_utc": datetime(2025, 5, 1, 20, 0, tzinfo=UTC),
            "home_team": "LAD",
            "away_team": "SDP",
            "venue_id": "MLB_LAD",
            "is_dome_or_retractable": False,
            "is_neutral_or_regional_venue": False,
            "uses_dh": True,
            "game_state": "FINAL",
            "home_sp_id": 501,
            "away_sp_id": 601,
            "home_sp_hand": "R",
            "away_sp_hand": "R",
            "home_runs": 6,
            "away_runs": 2,
            "outcome_1x2": 0,
            "home_win": 1,
            "is_tie": 0,
        },
        # Game 2: Next Day MLB (LAD vs SDP)
        {
            "sport": "baseball",
            "league": "MLB",
            "sub_league": "NL",
            "season": 2025,
            "game_id": 1002,
            "event_seq": 1,
            "game_date": date(2025, 5, 2),
            "start_time_utc": datetime(2025, 5, 2, 20, 0, tzinfo=UTC),
            "home_team": "LAD",
            "away_team": "SDP",
            "venue_id": "MLB_LAD",
            "is_dome_or_retractable": False,
            "is_neutral_or_regional_venue": False,
            "uses_dh": True,
            "game_state": "FINAL",
            "home_sp_id": 502,
            "away_sp_id": 602,
            "home_sp_hand": "L",
            "away_sp_hand": "R",
            "home_runs": 4,
            "away_runs": 3,
            "outcome_1x2": 0,
            "home_win": 1,
            "is_tie": 0,
        },
        # Game 3: Doubleheader Game 1 on May 3 (event_seq = 1)
        {
            "sport": "baseball",
            "league": "MLB",
            "sub_league": "NL",
            "season": 2025,
            "game_id": 1003,
            "event_seq": 1,
            "game_date": date(2025, 5, 3),
            "start_time_utc": datetime(2025, 5, 3, 17, 0, tzinfo=UTC),
            "home_team": "LAD",
            "away_team": "SDP",
            "venue_id": "MLB_LAD",
            "is_dome_or_retractable": False,
            "is_neutral_or_regional_venue": False,
            "uses_dh": True,
            "game_state": "FINAL",
            "home_sp_id": 501,
            "away_sp_id": 601,
            "home_sp_hand": "R",
            "away_sp_hand": "R",
            "home_runs": 5,
            "away_runs": 1,
            "outcome_1x2": 0,
            "home_win": 1,
            "is_tie": 0,
        },
        # Game 4: Doubleheader Game 2 on May 3 (event_seq = 2)
        {
            "sport": "baseball",
            "league": "MLB",
            "sub_league": "NL",
            "season": 2025,
            "game_id": 1003,
            "event_seq": 2,
            "game_date": date(2025, 5, 3),
            "start_time_utc": datetime(2025, 5, 3, 22, 30, tzinfo=UTC),
            "home_team": "LAD",
            "away_team": "SDP",
            "venue_id": "MLB_LAD",
            "is_dome_or_retractable": False,
            "is_neutral_or_regional_venue": False,
            "uses_dh": True,
            "game_state": "FINAL",
            "home_sp_id": 502,
            "away_sp_id": 602,
            "home_sp_hand": "L",
            "away_sp_hand": "L",
            "home_runs": 2,
            "away_runs": 7,
            "outcome_1x2": 2,
            "home_win": 0,
            "is_tie": 0,
        },
        # Game 5: NPB August Koshien road-trip displacement game (HAN vs YOM at Kyocera Dome)
        {
            "sport": "baseball",
            "league": "NPB",
            "sub_league": "CL",
            "season": 2025,
            "game_id": 9000002001,
            "event_seq": 1,
            "game_date": date(2025, 8, 12),
            "start_time_utc": datetime(2025, 8, 12, 9, 0, tzinfo=UTC),
            "home_team": "HAN",
            "away_team": "YOM",
            "venue_id": "NPB_KYOCERA_DOME",
            "is_dome_or_retractable": True,
            "is_neutral_or_regional_venue": True,
            "uses_dh": False,
            "game_state": "FINAL",
            "home_sp_id": 9001,
            "away_sp_id": 9002,
            "home_sp_hand": "R",
            "away_sp_hand": "R",
            "home_runs": 2,
            "away_runs": 2,
            "outcome_1x2": 1,
            "home_win": 0,
            "is_tie": 1,
        },
    ]

    team_rows: list[dict[str, Any]] = []
    pitcher_rows: list[dict[str, Any]] = []
    batter_rows: list[dict[str, Any]] = []

    for g in games_rows:
        gid = g["game_id"]
        seq = g["event_seq"]
        gdate = g["game_date"]
        gutc = g["start_time_utc"]
        lg = g["league"]
        for is_home in (True, False):
            t_code = g["home_team"] if is_home else g["away_team"]
            o_code = g["away_team"] if is_home else g["home_team"]
            r_for = g["home_runs"] if is_home else g["away_runs"]
            r_ag = g["away_runs"] if is_home else g["home_runs"]
            sp_id = g["home_sp_id"] if is_home else g["away_sp_id"]
            sp_hand = g["home_sp_hand"] if is_home else g["away_sp_hand"]
            rp_id = sp_id + 50

            team_rows.append(
                {
                    "game_id": gid,
                    "event_seq": seq,
                    "game_date": gdate,
                    "start_time_utc": gutc,
                    "season": 2025,
                    "league": lg,
                    "team": t_code,
                    "opponent": o_code,
                    "is_home": is_home,
                    "runs_for": r_for,
                    "runs_against": r_ag,
                    "hits_for": r_for + 4,
                    "hits_against": r_ag + 4,
                    "doubles_for": 2,
                    "triples_for": 0,
                    "hr_for": 1,
                    "hr_against": 1,
                    "bb_for": 3,
                    "bb_against": 3,
                    "k_for": 8,
                    "k_against": 8,
                    "hbp_for": 0,
                    "sf_for": 0,
                    "pa_for": 38,
                    "outs_pitched": 27,
                    "lob_for": 6,
                    "errors_committed": 0,
                    "bullpen_outs_pitched": 9,
                    "bullpen_pitches": 42,
                    "bullpen_er": 1,
                    "team_woba": 0.330 if r_for >= r_ag else 0.290,
                    "opp_woba": 0.290 if r_for >= r_ag else 0.330,
                    "team_xwoba": 0.335,
                    "opp_xwoba": 0.295,
                    "has_statcast_metrics": lg == "MLB",
                }
            )

            # Starter + Reliever
            pitcher_rows.append(
                {
                    "game_id": gid,
                    "event_seq": seq,
                    "game_date": gdate,
                    "start_time_utc": gutc,
                    "season": 2025,
                    "league": lg,
                    "player_id": sp_id,
                    "player_name": f"SP {sp_id}",
                    "team": t_code,
                    "opponent": o_code,
                    "is_home": is_home,
                    "pitch_hand": sp_hand,
                    "is_starter": True,
                    "outs_pitched": 18,
                    "innings_pitched": 6.0,
                    "batters_faced": 24,
                    "pitches_thrown": 95,
                    "strikes_thrown": 64,
                    "hits_allowed": 5,
                    "runs_allowed": r_ag,
                    "er_allowed": max(0, r_ag - 1),
                    "hr_allowed": 1,
                    "bb_allowed": 2,
                    "ibb_allowed": 0,
                    "hbp_allowed": 0,
                    "k_recorded": 7,
                    "gb_count": 8,
                    "fb_count": 5,
                    "ld_count": 3,
                    "quality_start": 1,
                    "xwoba_against": 0.285,
                    "barrel_pct_against": 0.06,
                    "swstr_pct": 0.135,
                    "avg_fastball_velo": 95.5,
                }
            )
            pitcher_rows.append(
                {
                    "game_id": gid,
                    "event_seq": seq,
                    "game_date": gdate,
                    "start_time_utc": gutc,
                    "season": 2025,
                    "league": lg,
                    "player_id": rp_id,
                    "player_name": f"Closer {rp_id}",
                    "team": t_code,
                    "opponent": o_code,
                    "is_home": is_home,
                    "pitch_hand": "R",
                    "is_starter": False,
                    "outs_pitched": 9,
                    "innings_pitched": 3.0,
                    "batters_faced": 11,
                    "pitches_thrown": 32,
                    "strikes_thrown": 22,
                    "hits_allowed": 2,
                    "runs_allowed": 1,
                    "er_allowed": 1,
                    "hr_allowed": 0,
                    "bb_allowed": 1,
                    "ibb_allowed": 0,
                    "hbp_allowed": 0,
                    "k_recorded": 4,
                    "gb_count": 3,
                    "fb_count": 2,
                    "ld_count": 1,
                    "quality_start": 0,
                    "xwoba_against": 0.270,
                    "barrel_pct_against": 0.04,
                    "swstr_pct": 0.150,
                    "avg_fastball_velo": 97.2,
                }
            )

            # 9 Starting Batters
            base_bid = (
                (1000 if t_code == "LAD" else 2000) if lg == "MLB" else (3000 if is_home else 4000)
            )
            for slot in range(1, 10):
                bid = base_bid + slot
                batter_rows.append(
                    {
                        "game_id": gid,
                        "event_seq": seq,
                        "game_date": gdate,
                        "start_time_utc": gutc,
                        "season": 2025,
                        "league": lg,
                        "player_id": bid,
                        "player_name": f"{t_code} Batter {slot}",
                        "team": t_code,
                        "opponent": o_code,
                        "is_home": is_home,
                        "bat_side": "L" if slot <= 5 else "R",
                        "batting_order_slot": slot,
                        "position_group": "OF",
                        "is_starter": True,
                        "pa": 4,
                        "ab": 3,
                        "hits": 1,
                        "singles": 1 if slot != 4 else 0,
                        "doubles": 0,
                        "triples": 0,
                        "hr": 1 if slot == 4 else 0,
                        "total_bases": 4 if slot == 4 else 1,
                        "runs": 1 if slot <= r_for else 0,
                        "rbi": 1 if slot <= r_for else 0,
                        "bb": 1,
                        "ibb": 0,
                        "hbp": 0,
                        "k": 1,
                        "sf": 0,
                        "sb": 0,
                        "cs": 0,
                        "woba_num": 2.791 if slot == 4 else 1.578,
                        "xwoba": 0.365,
                        "hard_hit_rate": 0.45,
                        "barrel_rate": 0.11,
                    }
                )

    return (
        pl.DataFrame(games_rows),
        pl.DataFrame(team_rows),
        pl.DataFrame(pitcher_rows),
        pl.DataFrame(batter_rows),
    )


def test_strict_doubleheader_and_temporal_anti_leakage_invariance() -> None:
    """Prove mutating Game 2 of a same-day doubleheader (or any event >= t) alters 0.0 of Game 1."""
    games_df, team_df, pitcher_df, batter_df = _build_synthetic_baseball_silver_tables()

    assert_zero_odds_contamination(STAGE1_FEATURE_COLUMNS)
    assert_zero_odds_contamination(STAGE2_PITCHER_FEATURE_COLUMNS)
    assert_zero_odds_contamination(STAGE2_BATTER_FEATURE_COLUMNS)

    base_s1 = build_baseball_stage1_game_features(games_df, team_df, pitcher_df, batter_df)
    base_s2_p = build_baseball_stage2_pitcher_features(games_df, pitcher_df, base_s1)
    base_s2_b = build_baseball_stage2_batter_features(games_df, batter_df, base_s1)

    # Verify zero nulls/NaNs across all Stage 1 and Stage 2 feature columns (including Opening Day!)
    for col in STAGE1_FEATURE_COLUMNS:
        assert col in base_s1.columns, f"Missing Stage 1 col {col}"
        assert base_s1[col].null_count() == 0, f"Nulls in Stage 1 {col}"
        assert base_s1[col].is_nan().sum() == 0, f"NaNs in Stage 1 {col}"

    for col in STAGE2_PITCHER_FEATURE_COLUMNS:
        assert col in base_s2_p.columns, f"Missing Stage 2 Pitcher col {col}"
        assert base_s2_p[col].null_count() == 0

    for col in STAGE2_BATTER_FEATURE_COLUMNS:
        assert col in base_s2_b.columns, f"Missing Stage 2 Batter col {col}"
        assert base_s2_b[col].null_count() == 0

    # Now mutate Doubleheader Game 2 (game_id=1003, event_seq=2) and all subsequent games:
    is_g2_or_later = (pl.col("game_id") > 1003) | (
        (pl.col("game_id") == 1003) & (pl.col("event_seq") == 2)
    )
    mut_games = games_df.with_columns(
        pl.when(is_g2_or_later).then(pl.lit(28)).otherwise(pl.col("home_runs")).alias("home_runs"),
        pl.when(is_g2_or_later).then(pl.lit(0)).otherwise(pl.col("away_runs")).alias("away_runs"),
        pl.when(is_g2_or_later).then(pl.lit(1)).otherwise(pl.col("home_win")).alias("home_win"),
    )
    mut_teams = team_df.with_columns(
        pl.when(is_g2_or_later).then(pl.lit(28)).otherwise(pl.col("runs_for")).alias("runs_for"),
        pl.when(is_g2_or_later)
        .then(pl.lit(140))
        .otherwise(pl.col("bullpen_pitches"))
        .alias("bullpen_pitches"),
        pl.when(is_g2_or_later)
        .then(pl.lit(0.650))
        .otherwise(pl.col("team_woba"))
        .alias("team_woba"),
    )
    mut_pitchers = pitcher_df.with_columns(
        pl.when(is_g2_or_later)
        .then(pl.lit(18))
        .otherwise(pl.col("k_recorded"))
        .alias("k_recorded"),
        pl.when(is_g2_or_later)
        .then(pl.lit(12))
        .otherwise(pl.col("er_allowed"))
        .alias("er_allowed"),
        pl.when(is_g2_or_later)
        .then(pl.lit(65))
        .otherwise(pl.col("pitches_thrown"))
        .alias("pitches_thrown"),
    )
    mut_batters = batter_df.with_columns(
        pl.when(is_g2_or_later).then(pl.lit(4)).otherwise(pl.col("hr")).alias("hr"),
        pl.when(is_g2_or_later).then(pl.lit(8.4)).otherwise(pl.col("woba_num")).alias("woba_num"),
    )

    mut_s1 = build_baseball_stage1_game_features(mut_games, mut_teams, mut_pitchers, mut_batters)
    mut_s2_p = build_baseball_stage2_pitcher_features(mut_games, mut_pitchers, mut_s1)
    mut_s2_b = build_baseball_stage2_batter_features(mut_games, mut_batters, mut_s1)

    # All games strictly before Doubleheader Game 2 (including Doubleheader Game 1)
    # MUST be bit-for-bit identical!
    before_g2 = ~is_g2_or_later
    base_s1_pre = base_s1.filter(before_g2)
    mut_s1_pre = mut_s1.filter(before_g2)
    assert base_s1_pre.height == 3
    for col in STAGE1_FEATURE_COLUMNS:
        assert base_s1_pre[col].to_list() == mut_s1_pre[col].to_list(), f"Leakage in Stage 1 {col}"

    base_s2_p_pre = base_s2_p.filter(before_g2)
    mut_s2_p_pre = mut_s2_p.filter(before_g2)
    for col in STAGE2_PITCHER_FEATURE_COLUMNS:
        assert base_s2_p_pre[col].to_list() == mut_s2_p_pre[col].to_list(), (
            f"Leakage in Stage 2 Pitcher {col}"
        )

    base_s2_b_pre = base_s2_b.filter(before_g2)
    mut_s2_b_pre = mut_s2_b.filter(before_g2)
    for col in STAGE2_BATTER_FEATURE_COLUMNS:
        assert base_s2_b_pre[col].to_list() == mut_s2_b_pre[col].to_list(), (
            f"Leakage in Stage 2 Batter {col}"
        )


def test_platoon_handedness_sensitivity_and_missing_regulars() -> None:
    """Verify switching opposing SP from RHP to LHP shifts lineup_platoon_woba & same_hand_share."""
    games_df, team_df, pitcher_df, batter_df = _build_synthetic_baseball_silver_tables()

    # In Game 3 (game_id=1003, seq=1), LAD has 5 LHB + 4 RHB starters and faces away_sp_hand='R'
    s1_vs_rhp = build_baseball_stage1_game_features(games_df, team_df, pitcher_df, batter_df)

    # Flip away_sp_hand in Game 3 from 'R' to 'L'
    games_vs_lhp = games_df.with_columns(
        pl.when((pl.col("game_id") == 1003) & (pl.col("event_seq") == 1))
        .then(pl.lit("L"))
        .otherwise(pl.col("away_sp_hand"))
        .alias("away_sp_hand")
    )
    s1_vs_lhp = build_baseball_stage1_game_features(games_vs_lhp, team_df, pitcher_df, batter_df)

    g3_rhp = s1_vs_rhp.filter((pl.col("game_id") == 1003) & (pl.col("event_seq") == 1)).row(
        0, named=True
    )
    g3_lhp = s1_vs_lhp.filter((pl.col("game_id") == 1003) & (pl.col("event_seq") == 1)).row(
        0, named=True
    )

    # Because slots 1..5 are LHB, facing RHP yields lower same_hand_share and higher platoon wOBA
    assert g3_rhp["home_lineup_same_hand_share"] < g3_lhp["home_lineup_same_hand_share"]
    assert g3_rhp["home_lineup_platoon_woba"] > g3_lhp["home_lineup_platoon_woba"]
    assert g3_rhp["home_missing_regular_pa_woba"] == 0.0

    # Now replace top-2 regular batters (1001, 1002) with bench rookies (1091, 1092) in Game 3
    s1_with_absent_stars = build_baseball_stage1_game_features(
        games_df,
        team_df,
        pitcher_df,
        batter_df,
        custom_lineups_by_game_team={
            (1003, 1, "LAD"): [1091, 1092, 1003, 1004, 1005, 1006, 1007, 1008, 1009]
        },
    )
    g3_absent = s1_with_absent_stars.filter(
        (pl.col("game_id") == 1003) & (pl.col("event_seq") == 1)
    ).row(0, named=True)
    assert g3_absent["home_missing_regular_pa_woba"] > 0.050
    assert g3_absent["home_lineup_platoon_woba"] < g3_rhp["home_lineup_platoon_woba"]


def test_bullpen_3day_fatigue_and_rest_reset() -> None:
    """Verify back-to-back reliever usage increments unavailable count and resets after rest."""
    games_df, team_df, pitcher_df, batter_df = _build_synthetic_baseball_silver_tables()

    # Add a Game 6 on May 6 (after 2 full off-days: May 4 and May 5)
    g6 = games_df.filter((pl.col("game_id") == 1003) & (pl.col("event_seq") == 1)).with_columns(
        pl.lit(1006).cast(pl.Int64).alias("game_id"),
        pl.lit(date(2025, 5, 6)).alias("game_date"),
        pl.lit(datetime(2025, 5, 6, 20, 0, tzinfo=UTC)).alias("start_time_utc"),
    )
    s1 = build_baseball_stage1_game_features(
        pl.concat([games_df, g6]), team_df, pitcher_df, batter_df
    )

    # On May 3 Game 1 (after closer 551 pitched on May 1 & May 2 + threw 32 pitches on May 2):
    g3_g1 = s1.filter((pl.col("game_id") == 1003) & (pl.col("event_seq") == 1)).row(0, named=True)
    assert g3_g1["home_high_leverage_arms_unavailable_count"] >= 1.0
    assert g3_g1["home_bp_pitches_last_3d"] == 84.0  # 42 on May 1 + 42 on May 2

    # On May 3 Game 2 (doubleheader nightcap): includes May 3 Game 1 bullpen pitches (42 * 3 = 126)!
    g3_g2 = s1.filter((pl.col("game_id") == 1003) & (pl.col("event_seq") == 2)).row(0, named=True)
    assert g3_g2["home_bp_pitches_last_3d"] == 126.0
    assert g3_g2["is_doubleheader_g2"] == 1.0

    # On May 6 (after May 4 and May 5 off-days): closer rested -> unavailable count resets to 0!
    g6_row = s1.filter(pl.col("game_id") == 1006).row(0, named=True)
    assert g6_row["home_high_leverage_arms_unavailable_count"] == 0.0
    assert g6_row["home_bp_pitches_last_1d"] == 0.0


def test_cross_league_priors_and_npb_koshien_context_in_pipeline() -> None:
    """Verify NPB -> MLB cross-league priors and August Koshien road-trip flag in Stage 1."""
    games_df, team_df, pitcher_df, batter_df = _build_synthetic_baseball_silver_tables()

    # Suppose home_sp_id=501 in Game 1 (Opening Day) is a posted NPB ace
    npb_ace_prior = resolve_pitcher_prior(
        target_league="MLB",
        prev_league="NPB",
        prev_fip=1.80,
        prev_xfip=1.95,
        prev_k_pct=0.300,
        prev_bb_pct=0.050,
        prev_bf=700,
    )
    npb_slugger_prior = resolve_batter_prior(
        target_league="MLB",
        prev_league="NPB",
        prev_woba=0.415,
        prev_iso=0.275,
        prev_k_pct=0.175,
        prev_bb_pct=0.135,
        prev_pa=600,
    )

    s1_default = build_baseball_stage1_game_features(games_df, team_df, pitcher_df, batter_df)
    s1_translated = build_baseball_stage1_game_features(
        games_df,
        team_df,
        pitcher_df,
        batter_df,
        cross_league_pitcher_priors={501: npb_ace_prior},
        cross_league_batter_priors={1001: npb_slugger_prior},
    )

    g1_def = s1_default.filter(pl.col("game_id") == 1001).row(0, named=True)
    g1_tr = s1_translated.filter(pl.col("game_id") == 1001).row(0, named=True)

    # Opening Day SP 501 with NPB ace translation has lower (better) xFIP and higher Elo win prob
    assert g1_tr["home_sp_xfip_shrunk"] < g1_def["home_sp_xfip_shrunk"] - 0.80
    assert g1_tr["home_sp_is_cross_league_prior"] == 1.0
    assert g1_tr["elo_home_win_prob"] > g1_def["elo_home_win_prob"]

    # Check NPB Game 5 (August Hanshin Tigers displaced home game at Kyocera Dome)
    npb_row = s1_translated.filter(pl.col("game_id") == 9000002001).row(0, named=True)
    assert npb_row["is_koshien_summer_road_trip_flag"] == 1.0
    assert npb_row["is_neutral_or_regional_flag"] == 1.0
    assert npb_row["is_dome_flag"] == 1.0
    assert 0.01 < npb_row["elo_draw_prob"] < 0.10

    # Also test build_baseball_team_rolling_features standalone
    rolled_teams = build_baseball_team_rolling_features(team_df)
    assert "team_pythag_luck_w30" in rolled_teams.columns
    assert pytest.approx(rolled_teams["team_pythag_luck_w30"][0], abs=1e-9) == 0.0
