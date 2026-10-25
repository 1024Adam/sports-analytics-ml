"""Hockey pre-game team, goalie, roster, and Stage 2 player feature tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.sports.hockey.features import (
    build_goalie_rolling_features,
    build_roster_strength_features,
    build_stage1_game_features,
    build_stage2_player_features,
    build_team_rolling_features,
)


def _sample_hockey_datasets() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Generate deterministic multi-game NHL games, team logs, skater logs, and goalie logs."""
    games_df = pl.DataFrame(
        {
            "sport": ["hockey"] * 3,
            "season": [20242025] * 3,
            "game_type": [2, 2, 2],
            "game_id": [2024020001, 2024020002, 2024020003],
            "game_date": [date(2024, 10, 10), date(2024, 10, 12), date(2024, 10, 15)],
            "start_time_utc": [
                datetime(2024, 10, 10, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 12, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 23, 0, tzinfo=UTC),
            ],
            "home_team": ["BOS", "TOR", "BOS"],
            "away_team": ["TOR", "BOS", "TOR"],
            "game_state": ["FINAL", "FINAL", "FINAL"],
            "finish_period_type": ["REG", "REG", "REG"],
            "home_goals": [4, 2, 5],
            "away_goals": [2, 3, 1],
            "home_reg_goals": [4, 2, 5],
            "away_reg_goals": [2, 3, 1],
            "home_win": [1, 0, 1],
        }
    )

    team_logs = pl.DataFrame(
        {
            "game_id": [2024020001, 2024020001, 2024020002, 2024020002, 2024020003, 2024020003],
            "game_date": [
                date(2024, 10, 10),
                date(2024, 10, 10),
                date(2024, 10, 12),
                date(2024, 10, 12),
                date(2024, 10, 15),
                date(2024, 10, 15),
            ],
            "start_time_utc": [
                datetime(2024, 10, 10, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 10, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 12, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 12, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 23, 0, tzinfo=UTC),
            ],
            "season": [20242025] * 6,
            "team": ["BOS", "TOR", "TOR", "BOS", "BOS", "TOR"],
            "opponent": ["TOR", "BOS", "BOS", "TOR", "TOR", "BOS"],
            "is_home": [True, False, True, False, True, False],
            "goals_for": [4, 2, 2, 3, 5, 1],
            "goals_against": [2, 4, 3, 2, 1, 5],
            "sog_for": [34, 26, 28, 32, 36, 25],
            "sog_against": [26, 34, 32, 28, 25, 36],
            "xgf_5v5": [2.9, 1.8, 2.0, 2.6, 3.3, 1.5],
            "xga_5v5": [1.8, 2.9, 2.6, 2.0, 1.5, 3.3],
            "cf_5v5": [58, 42, 46, 54, 62, 40],
            "ca_5v5": [42, 58, 54, 46, 40, 62],
            "ff_5v5": [44, 31, 35, 41, 47, 30],
            "fa_5v5": [31, 44, 41, 35, 30, 47],
            "hd_xgf_5v5": [1.3, 0.7, 0.8, 1.1, 1.5, 0.6],
            "hd_xga_5v5": [0.7, 1.3, 1.1, 0.8, 0.6, 1.5],
            "toi_5v5_minutes": [49.0, 49.0, 50.0, 50.0, 48.0, 48.0],
            "pp_goals": [1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            "pp_opportunities": [3.0, 3.0, 2.0, 3.0, 4.0, 3.0],
            "pp_xgf": [0.8, 0.4, 0.3, 0.7, 1.1, 0.4],
            "pp_toi_min": [5.0, 5.0, 4.0, 5.0, 6.0, 5.0],
            "pk_goals_against": [0.0, 1.0, 1.0, 0.0, 0.0, 2.0],
            "pk_times_shorthanded": [3.0, 3.0, 3.0, 2.0, 3.0, 4.0],
            "pk_xga": [0.4, 0.8, 0.7, 0.3, 0.4, 1.1],
            "pk_toi_min": [5.0, 5.0, 5.0, 4.0, 5.0, 6.0],
        }
    )

    skater_rows = []
    # 2 forwards + 1 defenseman per team per game
    players_spec = [
        (8477956, "David Pastrnak", "BOS", "F", 21.0, 17.0, 4.0, 2, 1, 6, 0.85, 2.4),
        (8473419, "Brad Marchand", "BOS", "F", 18.0, 15.0, 3.0, 1, 1, 3, 0.40, 1.3),
        (8479325, "Charlie McAvoy", "BOS", "D", 24.0, 20.0, 2.0, 0, 2, 3, 0.25, 1.6),
        (8479318, "Auston Matthews", "TOR", "F", 21.0, 17.0, 4.0, 1, 0, 5, 0.70, 1.5),
        (8478483, "Mitch Marner", "TOR", "F", 20.0, 16.0, 3.0, 0, 1, 2, 0.30, 1.0),
        (8476853, "Morgan Rielly", "TOR", "D", 23.0, 19.0, 2.5, 0, 1, 2, 0.20, 0.9),
    ]
    for g_row in games_df.iter_rows(named=True):
        gid = g_row["game_id"]
        gdate = g_row["game_date"]
        gtime = g_row["start_time_utc"]
        home_t = g_row["home_team"]
        away_t = g_row["away_team"]
        for pid, pname, team, pos, toi_tot, toi_ev, toi_pp, g, a, sog, ixg, gs in players_spec:
            opp = away_t if team == home_t else home_t
            skater_rows.append(
                {
                    "game_id": gid,
                    "game_date": gdate,
                    "start_time_utc": gtime,
                    "season": 20242025,
                    "player_id": pid,
                    "player_name": pname,
                    "team": team,
                    "opponent": opp,
                    "is_home": team == home_t,
                    "position_group": pos,
                    "toi_total_min": toi_tot,
                    "toi_ev_min": toi_ev,
                    "toi_pp_min": toi_pp,
                    "toi_pk_min": max(0.0, toi_tot - toi_ev - toi_pp),
                    "goals": g,
                    "assists": a,
                    "points": g + a,
                    "sog": sog,
                    "ixg_all": ixg,
                    "ixg_5v5": ixg * 0.7,
                    "icf_all": float(sog + 2),
                    "on_ice_xgf_5v5": 1.1 if team == "BOS" else 0.7,
                    "on_ice_xga_5v5": 0.7 if team == "BOS" else 1.1,
                    "game_score": gs,
                }
            )
    skater_logs = pl.DataFrame(skater_rows)

    goalie_logs = pl.DataFrame(
        {
            "game_id": [2024020001, 2024020001, 2024020002, 2024020002, 2024020003, 2024020003],
            "game_date": [
                date(2024, 10, 10),
                date(2024, 10, 10),
                date(2024, 10, 12),
                date(2024, 10, 12),
                date(2024, 10, 15),
                date(2024, 10, 15),
            ],
            "start_time_utc": [
                datetime(2024, 10, 10, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 10, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 12, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 12, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 23, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 23, 0, tzinfo=UTC),
            ],
            "season": [20242025] * 6,
            "player_id": [8480280, 8479361, 8479361, 8480280, 8480280, 8479361],
            "team": ["BOS", "TOR", "TOR", "BOS", "BOS", "TOR"],
            "opponent": ["TOR", "BOS", "BOS", "TOR", "TOR", "BOS"],
            "is_home": [True, False, True, False, True, False],
            "is_starter": [True] * 6,
            "toi_min": [60.0] * 6,
            "shots_against": [26, 34, 32, 28, 25, 36],
            "saves": [24, 30, 29, 26, 24, 31],
            "goals_against": [2, 4, 3, 2, 1, 5],
            "xga_all": [2.6, 3.2, 2.9, 2.5, 2.2, 3.6],
            "gsax": [0.6, -0.8, -0.1, 0.5, 1.2, -1.4],
        }
    )
    return games_df, team_logs, skater_logs, goalie_logs


def test_strict_anti_leakage_across_all_hockey_features() -> None:
    """Mutating tonight's game goals, xG, or player points produces zero change in features."""
    games_df, team_logs, skater_logs, goalie_logs = _sample_hockey_datasets()
    cutoff_time = datetime(2024, 10, 15, 23, 0, tzinfo=UTC)  # Game 3 (2024020003)

    # Baseline features across all 4 feature modules
    base_team = build_team_rolling_features(team_logs)
    base_goalie = build_goalie_rolling_features(goalie_logs)
    base_roster = build_roster_strength_features(skater_logs)
    base_stage1 = build_stage1_game_features(games_df, team_logs, skater_logs, goalie_logs)
    base_stage2 = build_stage2_player_features(skater_logs, base_team, base_goalie)

    # Corrupt all post-game stats for Game 3 (start_time_utc >= cutoff_time)
    mut_games = games_df.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_time)
        .then(pl.lit(99))
        .otherwise(pl.col("home_goals"))
        .alias("home_goals"),
        pl.when(pl.col("start_time_utc") >= cutoff_time)
        .then(pl.lit(0))
        .otherwise(pl.col("home_win"))
        .alias("home_win"),
    )
    mut_teams = team_logs.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_time)
        .then(pl.lit(999.0))
        .otherwise(pl.col(c))
        .alias(c)
        for c in ["goals_for", "goals_against", "xgf_5v5", "xga_5v5", "pp_xgf", "pk_xga"]
    )
    mut_skaters = skater_logs.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_time)
        .then(pl.lit(999.0))
        .otherwise(pl.col(c))
        .alias(c)
        for c in ["goals", "assists", "points", "sog", "ixg_all", "on_ice_xgf_5v5", "game_score"]
    )
    mut_goalies = goalie_logs.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_time)
        .then(pl.lit(-99.0))
        .otherwise(pl.col(c))
        .alias(c)
        for c in ["goals_against", "xga_all", "gsax"]
    )

    mut_team_feat = build_team_rolling_features(mut_teams)
    mut_goalie_feat = build_goalie_rolling_features(mut_goalies)
    mut_roster_feat = build_roster_strength_features(mut_skaters)
    mut_stage1_feat = build_stage1_game_features(mut_games, mut_teams, mut_skaters, mut_goalies)
    mut_stage2_feat = build_stage2_player_features(mut_skaters, mut_team_feat, mut_goalie_feat)

    # Compare strictly pre-game feature columns for all games <= cutoff_time (including Game 3!)
    team_pre_cols = [
        "game_id",
        "team",
        "xgf_pct_5v5_w5",
        "xgf_pct_5v5_w10",
        "xgf_pct_5v5_w25",
        "xgf_pct_5v5_ewma",
        "xgf_per_60_5v5_w10",
        "xga_per_60_5v5_w10",
        "cf_pct_5v5_w10",
        "ff_pct_5v5_w10",
        "hd_xgf_pct_5v5_w10",
        "pdo_5v5_w10",
        "pp_xgf_per_60",
        "pk_xga_per_60",
    ]
    assert_frame_equal(
        base_team.filter(pl.col("start_time_utc") <= cutoff_time).select(team_pre_cols),
        mut_team_feat.filter(pl.col("start_time_utc") <= cutoff_time).select(team_pre_cols),
    )

    goalie_pre_cols = [
        "game_id",
        "player_id",
        "gsax_per_60_shrunk_w10",
        "gsax_per_60_shrunk_w25",
        "save_pct_shrunk",
        "goalie_days_rest",
        "goalie_starts_last_7d",
    ]
    assert_frame_equal(
        base_goalie.filter(pl.col("start_time_utc") <= cutoff_time).select(goalie_pre_cols),
        mut_goalie_feat.filter(pl.col("start_time_utc") <= cutoff_time).select(goalie_pre_cols),
    )

    assert_frame_equal(
        base_roster.filter(pl.col("start_time_utc") <= cutoff_time),
        mut_roster_feat.filter(pl.col("start_time_utc") <= cutoff_time),
    )

    stage1_pre_cols = [
        "game_id",
        "elo_home_pre",
        "elo_away_pre",
        "elo_home_win_prob",
        "st_net_edge_home_vs_away",
        "xgf_pct_5v5_w10_diff",
        "goalie_gsax_per_60_diff",
        "forward_ixg_per_60_diff",
        "defense_xgd_per_60_diff",
    ]
    assert_frame_equal(
        base_stage1.filter(pl.col("start_time_utc") <= cutoff_time).select(stage1_pre_cols),
        mut_stage1_feat.filter(pl.col("start_time_utc") <= cutoff_time).select(stage1_pre_cols),
    )

    stage2_pre_cols = [
        "game_id",
        "player_id",
        "rolling_toi_ev_min",
        "rolling_toi_pp_min",
        "team_injury_vacated_toi_f_or_d",
        "rolling_sog_per_60",
        "rolling_ixg_per_60",
        "rolling_goals_per_60",
        "rolling_points_per_60",
        "career_shooting_pct_shrunk",
        "opp_xga_per_60_5v5",
        "opp_goalie_gsax_per_60_shrunk",
    ]
    assert_frame_equal(
        base_stage2.filter(pl.col("start_time_utc") <= cutoff_time).select(stage2_pre_cols),
        mut_stage2_feat.filter(pl.col("start_time_utc") <= cutoff_time).select(stage2_pre_cols),
    )


def test_cold_start_game1_and_rookie_debut_no_nulls() -> None:
    """Verify Game 1 of a season and a debuting rookie/goalie produce finite shrunk features."""
    games_df, team_logs, skater_logs, goalie_logs = _sample_hockey_datasets()

    # Add a debuting rookie skater and debuting goalie in Game 3
    rookie_skater = skater_logs.filter(
        (pl.col("game_id") == 2024020003) & (pl.col("player_id") == 8477956)
    ).with_columns(
        pl.lit(9999999, dtype=pl.Int64).alias("player_id"),
        pl.lit("Rookie Debut").alias("player_name"),
    )
    skater_with_rookie = pl.concat([skater_logs, rookie_skater])

    team_feat = build_team_rolling_features(team_logs)
    goalie_feat = build_goalie_rolling_features(goalie_logs)
    stage2_feat = build_stage2_player_features(skater_with_rookie, team_feat, goalie_feat)

    # Game 1 (2024020001) has n=0 prior games -> must have zero nulls and equal priors
    g1_team = team_feat.filter(pl.col("game_id") == 2024020001)
    assert g1_team["xgf_pct_5v5_w10"].null_count() == 0
    assert g1_team["xgf_pct_5v5_w10"][0] == pytest.approx(0.50)
    assert g1_team["pdo_5v5_w10"][0] == pytest.approx(1.000)

    g1_goalie = goalie_feat.filter(pl.col("game_id") == 2024020001)
    assert g1_goalie["gsax_per_60_shrunk"].null_count() == 0
    assert g1_goalie["gsax_per_60_shrunk"][0] == pytest.approx(0.0)
    assert g1_goalie["save_pct_shrunk"][0] == pytest.approx(0.905)
    assert g1_goalie["is_low_sample_goalie"][0] is True

    # Debuting rookie in Game 3 must have finite non-null shrunk features
    rookie_row = stage2_feat.filter(pl.col("player_id") == 9999999).row(0, named=True)
    assert rookie_row["player_prior_games_n"] == 0
    assert rookie_row["is_low_sample_skater"] is True
    assert rookie_row["rolling_toi_ev_min"] == pytest.approx(14.0)
    assert rookie_row["career_shooting_pct_shrunk"] == pytest.approx(0.105)


def test_roster_weighting_normalization_and_missing_regular_toi() -> None:
    """Verify TOI weights normalize to 1.0 and reflect lineup changes when a star is out."""
    _, team_logs, skater_logs, goalie_logs = _sample_hockey_datasets()

    full_roster_feat = build_roster_strength_features(skater_logs)
    for w_val in full_roster_feat["forward_toi_weight_sum"].to_list():
        assert w_val == pytest.approx(1.0)
    for w_val in full_roster_feat["defense_toi_weight_sum"].to_list():
        assert w_val == pytest.approx(1.0)

    # Remove top-line star David Pastrnak (8477956) from BOS active lineup in Game 3 (2024020003)
    without_star_logs = skater_logs.filter(
        ~((pl.col("game_id") == 2024020003) & (pl.col("player_id") == 8477956))
    )
    missing_star_feat = build_roster_strength_features(without_star_logs)

    bos_g3_full = full_roster_feat.filter(
        (pl.col("game_id") == 2024020003) & (pl.col("team") == "BOS")
    ).row(0, named=True)
    bos_g3_no_star = missing_star_feat.filter(
        (pl.col("game_id") == 2024020003) & (pl.col("team") == "BOS")
    ).row(0, named=True)

    # Forward weights still normalize cleanly to 1.0
    assert bos_g3_no_star["forward_toi_weight_sum"] == pytest.approx(1.0)
    # Missing regular TOI increases by Pastrnak's ~16-21 min projected TOI
    assert bos_g3_full["missing_regular_toi"] == pytest.approx(0.0)
    assert bos_g3_no_star["missing_regular_toi"] > 15.0
    assert bos_g3_no_star["missing_regular_toi_f"] > 15.0
    # Forward lineup weighted ixG/60 drops when top scorer Pastrnak is absent
    assert (
        bos_g3_no_star["forward_weighted_ixg_per_60"] < bos_g3_full["forward_weighted_ixg_per_60"]
    )

    # Verify vacated forward TOI propagates to remaining BOS forward Brad Marchand in Stage 2
    team_feat = build_team_rolling_features(team_logs)
    goalie_feat = build_goalie_rolling_features(goalie_logs)
    s2_no_star = build_stage2_player_features(without_star_logs, team_feat, goalie_feat)
    marchand_g3 = s2_no_star.filter(
        (pl.col("game_id") == 2024020003) & (pl.col("player_id") == 8473419)
    ).row(0, named=True)
    assert marchand_g3["team_injury_vacated_toi_f_or_d"] > 15.0
