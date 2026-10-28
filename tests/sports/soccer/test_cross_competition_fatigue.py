"""Cross-competition fatigue, tactical isolation, and temporal firewall tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sports_ml.sports.soccer.features.roster import build_soccer_roster_strength_features
from sports_ml.sports.soccer.features.team import (
    build_soccer_stage1_match_features,
    build_soccer_team_rolling_features,
)
from sports_ml.sports.soccer.ingestion.non_league_fixtures import (
    NonLeagueFixturesIngestor,
    parse_non_league_fixtures_csv,
    parse_non_league_fixtures_records,
)
from sports_ml.sports.soccer.ingestion.reconciler import SoccerSilverReconciler


def _make_two_saturday_epl_fixtures() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create two Saturday EPL fixtures (Oct 19 and Oct 26, 2024) for ARS vs CHE / LIV."""
    sat1_utc = datetime(2024, 10, 19, 14, 0, tzinfo=UTC)
    sat2_utc = datetime(2024, 10, 26, 14, 0, tzinfo=UTC)

    matches_df = pl.DataFrame(
        [
            {
                "sport": "soccer",
                "season": 20242025,
                "matchweek": 8,
                "match_id": 202400081,
                "event_seq": 1,
                "match_date": date(2024, 10, 19),
                "start_time_utc": sat1_utc,
                "home_team": "ARS",
                "away_team": "CHE",
                "venue_timezone": "Europe/London",
                "game_state": "FINAL",
                "home_goals": 2,
                "away_goals": 1,
                "outcome_1x2": 0,
                "home_win": 1,
                "is_draw": 0,
                "away_win": 0,
            },
            {
                "sport": "soccer",
                "season": 20242025,
                "matchweek": 9,
                "match_id": 202400091,
                "event_seq": 1,
                "match_date": date(2024, 10, 26),
                "start_time_utc": sat2_utc,
                "home_team": "ARS",
                "away_team": "LIV",
                "venue_timezone": "Europe/London",
                "game_state": "FINAL",
                "home_goals": 1,
                "away_goals": 1,
                "outcome_1x2": 1,
                "home_win": 0,
                "is_draw": 1,
                "away_win": 0,
            },
        ]
    )

    team_logs_df = pl.DataFrame(
        [
            {
                "match_id": 202400081,
                "match_date": date(2024, 10, 19),
                "start_time_utc": sat1_utc,
                "season": 20242025,
                "team": "ARS",
                "opponent": "CHE",
                "is_home": True,
                "goals_for": 2,
                "goals_against": 1,
                "xgf": 1.85,
                "xga": 0.95,
                "npxgf": 1.85,
                "npxga": 0.95,
                "ppda": 9.2,
                "opp_ppda": 13.4,
            },
            {
                "match_id": 202400081,
                "match_date": date(2024, 10, 19),
                "start_time_utc": sat1_utc,
                "season": 20242025,
                "team": "CHE",
                "opponent": "ARS",
                "is_home": False,
                "goals_for": 1,
                "goals_against": 2,
                "xgf": 0.95,
                "xga": 1.85,
                "npxgf": 0.95,
                "npxga": 1.85,
                "ppda": 13.4,
                "opp_ppda": 9.2,
            },
            {
                "match_id": 202400091,
                "match_date": date(2024, 10, 26),
                "start_time_utc": sat2_utc,
                "season": 20242025,
                "team": "ARS",
                "opponent": "LIV",
                "is_home": True,
                "goals_for": 1,
                "goals_against": 1,
                "xgf": 1.40,
                "xga": 1.25,
                "npxgf": 1.40,
                "npxga": 1.25,
                "ppda": 10.5,
                "opp_ppda": 10.8,
            },
            {
                "match_id": 202400091,
                "match_date": date(2024, 10, 26),
                "start_time_utc": sat2_utc,
                "season": 20242025,
                "team": "LIV",
                "opponent": "ARS",
                "is_home": False,
                "goals_for": 1,
                "goals_against": 1,
                "xgf": 1.25,
                "xga": 1.40,
                "npxgf": 1.25,
                "npxga": 1.40,
                "ppda": 10.8,
                "opp_ppda": 10.5,
            },
        ]
    )

    outfield_rows = []
    for m_id, m_dt, s_utc, opp in (
        (202400081, date(2024, 10, 19), sat1_utc, "CHE"),
        (202400091, date(2024, 10, 26), sat2_utc, "LIV"),
    ):
        for p_idx in range(1, 11):
            outfield_rows.append(
                {
                    "match_id": m_id,
                    "match_date": m_dt,
                    "start_time_utc": s_utc,
                    "season": 20242025,
                    "player_id": 1000 + p_idx,
                    "player_name": f"ARS Starter {p_idx}",
                    "team": "ARS",
                    "opponent": opp,
                    "is_home": True,
                    "position_group": "FWD" if p_idx <= 3 else ("MID" if p_idx <= 6 else "DEF"),
                    "is_starter": True,
                    "minutes": 90.0,
                    "npxg": 0.25 if p_idx <= 3 else 0.08,
                    "xa": 0.15,
                    "xg_chain": 0.45,
                    "xg_buildup": 0.25,
                }
            )
    outfield_df = pl.DataFrame(outfield_rows)
    return matches_df, team_logs_df, outfield_df


def test_wednesday_ucl_away_match_updates_rest_congestion_travel_and_xi_minutes() -> None:
    """Verify inserting a Wednesday UCL away match reduces Saturday EPL rest from 7.0 to 3.0."""
    matches_df, team_logs_df, outfield_df = _make_two_saturday_epl_fixtures()

    # 1. Baseline without midweek UEFA match: ARS has 7.0 days rest before Oct 26 EPL match
    s1_epl_only = build_soccer_stage1_match_features(
        matches_df,
        team_logs_df,
        outfield_df,
        all_comp_schedule_df=None,
    )
    row_sat2_base = s1_epl_only.filter(pl.col("match_id") == 202400091).row(0, named=True)
    assert row_sat2_base["true_all_comp_rest_days_home"] == pytest.approx(7.0)
    assert row_sat2_base["had_midweek_european_match_home"] == 0
    assert row_sat2_base["had_midweek_away_european_match_home"] == 0
    assert row_sat2_base["european_travel_km_last_7d_home"] == pytest.approx(0.0)
    assert row_sat2_base["all_comp_matches_last_14d_home"] == 2
    assert row_sat2_base["active_xi_14d_all_comp_minutes_home"] == pytest.approx(900.0)

    # 2. Insert a Wednesday UEFA Champions League away match at Bayern Munich (Oct 23, 2024)
    wed_ucl_df = parse_non_league_fixtures_records(
        [
            {
                "season": 20242025,
                "match_date": "2024-10-23",
                "start_time_utc": "2024-10-23T19:00:00Z",
                "team": "ARS",
                "opponent": "Bayern Munich",
                "competition_type": "UCL",
                "is_home": False,
                "extra_time_played": True,
                "squad_minutes_load": 1320.0,
            }
        ]
    )

    s1_with_ucl = build_soccer_stage1_match_features(
        matches_df,
        team_logs_df,
        outfield_df,
        all_comp_schedule_df=wed_ucl_df,
    )
    row_sat2_ucl = s1_with_ucl.filter(pl.col("match_id") == 202400091).row(0, named=True)

    assert row_sat2_ucl["true_all_comp_rest_days_home"] == pytest.approx(3.0)
    assert row_sat2_ucl["rest_days_home"] == 3
    assert row_sat2_ucl["had_midweek_european_match_home"] == 1
    assert row_sat2_ucl["had_midweek_away_european_match_home"] == 1
    assert row_sat2_ucl["european_travel_km_last_7d_home"] > 1500.0
    assert (
        row_sat2_ucl["all_comp_matches_last_14d_home"]
        == row_sat2_base["all_comp_matches_last_14d_home"] + 1
    )
    assert row_sat2_ucl["cup_extra_time_last_7d_home"] == 1
    assert (
        row_sat2_ucl["active_xi_14d_all_comp_minutes_home"]
        > row_sat2_base["active_xi_14d_all_comp_minutes_home"]
    )


def test_tactical_isolation_between_cup_european_fixtures_and_epl_metrics() -> None:
    """Verify mutating goals/xG in UCL/Cup fixtures leaves EPL Elo and npxG/PPDA unchanged."""
    matches_df, team_logs_df, outfield_df = _make_two_saturday_epl_fixtures()

    # Create non-EPL rows in team_logs and matches with modest stats vs extreme 9-0 / 6.5 xG stats
    wed_utc = datetime(2024, 10, 23, 19, 0, tzinfo=UTC)
    cup_log_modest = pl.DataFrame(
        [
            {
                "match_id": 999001,
                "match_date": date(2024, 10, 23),
                "start_time_utc": wed_utc,
                "season": 20242025,
                "team": "ARS",
                "opponent": "BAY",
                "competition_type": "UCL",
                "is_home": False,
                "goals_for": 1,
                "goals_against": 1,
                "xgf": 1.10,
                "xga": 1.15,
                "npxgf": 1.10,
                "npxga": 1.15,
                "ppda": 11.0,
                "opp_ppda": 11.0,
            }
        ]
    )
    cup_log_extreme = cup_log_modest.with_columns(
        pl.lit(9, dtype=pl.Int64).alias("goals_for"),
        pl.lit(0, dtype=pl.Int64).alias("goals_against"),
        pl.lit(6.80, dtype=pl.Float64).alias("xgf"),
        pl.lit(0.05, dtype=pl.Float64).alias("xga"),
        pl.lit(6.80, dtype=pl.Float64).alias("npxgf"),
        pl.lit(0.05, dtype=pl.Float64).alias("npxga"),
        pl.lit(3.5, dtype=pl.Float64).alias("ppda"),
        pl.lit(28.0, dtype=pl.Float64).alias("opp_ppda"),
    )

    logs_a = pl.concat(
        [team_logs_df.with_columns(pl.lit("EPL").alias("competition_type")), cup_log_modest],
        how="diagonal_relaxed",
    )
    logs_b = pl.concat(
        [team_logs_df.with_columns(pl.lit("EPL").alias("competition_type")), cup_log_extreme],
        how="diagonal_relaxed",
    )

    s1_a = build_soccer_stage1_match_features(matches_df, logs_a, outfield_df)
    s1_b = build_soccer_stage1_match_features(matches_df, logs_b, outfield_df)

    tactical_cols = [
        "elo_home_pre",
        "elo_away_pre",
        "elo_diff_pre",
        "elo_home_win_prob",
        "elo_draw_prob",
        "elo_away_win_prob",
        "npxgf_per_90_w10_home",
        "npxga_per_90_w10_home",
        "npxgd_per_90_w10_home",
        "npxgf_pct_w10_home",
        "ppda_w10_home",
        "opp_ppda_w10_home",
        "finishing_above_xg_w10_home",
    ]
    for col in tactical_cols:
        arr_a = s1_a[col].to_numpy()
        arr_b = s1_b[col].to_numpy()
        np.testing.assert_array_equal(arr_a, arr_b)


def test_cross_competition_temporal_firewall_no_future_leakage() -> None:
    """Verify UEFA/Cup fixtures at >= T_match never leak into pre-match features at < T_match."""
    matches_df, team_logs_df, outfield_df = _make_two_saturday_epl_fixtures()

    # Schedule a Wednesday UCL fixture on Oct 23 (strictly AFTER Oct 19 match 202400081)
    # and a future FA Cup fixture on Oct 26 at 18:00 UTC (at/after Oct 26 14:00 EPL match 202400091)
    future_nl_df = parse_non_league_fixtures_records(
        [
            {
                "season": 20242025,
                "match_date": "2024-10-23",
                "start_time_utc": "2024-10-23T19:00:00Z",
                "team": "ARS",
                "opponent": "Real Madrid",
                "competition_type": "UCL",
                "is_home": False,
                "extra_time_played": True,
            },
            {
                "season": 20242025,
                "match_date": "2024-10-26",
                "start_time_utc": "2024-10-26T18:00:00Z",
                "team": "ARS",
                "opponent": "CHE",
                "competition_type": "EFL_CUP",
                "is_home": True,
                "extra_time_played": True,
            },
        ]
    )

    s1_base = build_soccer_stage1_match_features(matches_df, team_logs_df, outfield_df)
    s1_with_future = build_soccer_stage1_match_features(
        matches_df,
        team_logs_df,
        outfield_df,
        all_comp_schedule_df=future_nl_df,
    )

    # Match 1 (Oct 19, 2024) occurs strictly before both non-league fixtures
    m1_base = s1_base.filter(pl.col("match_id") == 202400081)
    m1_fut = s1_with_future.filter(pl.col("match_id") == 202400081)
    for col in m1_base.columns:
        assert m1_base[col].to_list() == m1_fut[col].to_list(), f"Future leak in {col}"

    # Verify Oct 26 18:00 EFL_CUP fixture (>= Oct 26 14:00 EPL match) did NOT leak into Oct 26 14:00
    only_wed_df = future_nl_df.slice(0, 1)
    s1_only_wed = build_soccer_stage1_match_features(
        matches_df,
        team_logs_df,
        outfield_df,
        all_comp_schedule_df=only_wed_df,
    )
    m2_only_wed = s1_only_wed.filter(pl.col("match_id") == 202400091)
    m2_with_later_same_day = s1_with_future.filter(pl.col("match_id") == 202400091)
    for col in m2_only_wed.columns:
        assert m2_only_wed[col].to_list() == m2_with_later_same_day[col].to_list(), (
            f"Same-day future fixture leaked into {col}"
        )


def test_non_league_ingestor_and_silver_reconciler_roundtrip(tmp_path: Path) -> None:
    """Verify NonLeagueFixturesIngestor and SoccerSilverReconciler persist all-comp schedule."""
    matches_df, _, outfield_df = _make_two_saturday_epl_fixtures()
    cache_dir = tmp_path / "raw" / "non_league_fixtures"
    silver_dir = tmp_path / "silver" / "soccer"
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_text = (
        "season,match_date,time,team,opponent,competition_type,is_home,extra_time_played,squad_minutes_load\n"
        "20242025,2024-10-23,19:45,Arsenal,PSG,UCL,false,false,990\n"
        "20242025,2024-10-30,19:45,Liverpool,Brighton,EFL_CUP,false,true,1320\n"
    )
    (cache_dir / "uefa_and_cups.csv").write_text(csv_text, encoding="utf-8")

    ingestor = NonLeagueFixturesIngestor(cache_dir=cache_dir)
    nl_df = ingestor.load_cached_non_league_fixtures()
    assert nl_df.height == 2

    reconciler = SoccerSilverReconciler(silver_dir=silver_dir)
    merged = reconciler.reconcile_all_competitions_schedule(nl_df, epl_matches_df=matches_df)
    assert reconciler.all_competitions_schedule_path.exists()
    # 2 EPL matches * 2 teams + 2 non-league team rows = 6 rows
    assert merged.height == 6
    assert set(merged["competition_type"].unique().to_list()) == {"EPL", "UCL", "EFL_CUP"}

    # Verify team rolling and roster strength builders work directly with merged schedule
    _, team_logs_df, _ = _make_two_saturday_epl_fixtures()
    t_feat = build_soccer_team_rolling_features(team_logs_df, all_comp_schedule_df=merged)
    assert "true_all_comp_rest_days" in t_feat.columns
    r_feat = build_soccer_roster_strength_features(outfield_df, all_comp_schedule_df=merged)
    assert "active_xi_14d_all_comp_minutes" in r_feat.columns
    assert parse_non_league_fixtures_csv("").is_empty()
