"""Offline deterministic tests for EPL Soccer ingestion, lineups, and Silver reconciler."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from sports_ml.sports.soccer.constants import (
    SILVER_GK_MATCH_LOGS_SCHEMA,
    SILVER_MATCHES_SCHEMA,
    SILVER_OUTFIELD_MATCH_LOGS_SCHEMA,
    SILVER_TEAM_MATCH_LOGS_SCHEMA,
)
from sports_ml.sports.soccer.ingestion import (
    FootballDataClient,
    FPLAPIClient,
    SoccerPreMatchLineupResolver,
    SoccerSilverReconciler,
    UnderstatClient,
    check_soccer_data_sources,
    derive_active_epl_season,
    parse_bootstrap_squad_df,
    parse_bootstrap_teams_map,
    parse_element_summary_payload,
    parse_fixtures_payload,
    parse_football_data_csv,
    parse_understat_player_matches_payload,
    parse_understat_team_matches_payload,
    season_to_football_data_code,
)


def _sample_fpl_bootstrap_payload() -> dict[str, Any]:
    return {
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "name": "Man City", "short_name": "MCI"},
        ],
        "elements": [
            {
                "id": 101,
                "first_name": "David",
                "second_name": "Raya",
                "web_name": "Raya",
                "team": 1,
                "element_type": 1,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "minutes": 900,
                "starts": 10,
            },
            {
                "id": 102,
                "first_name": "Bukayo",
                "second_name": "Saka",
                "web_name": "Saka",
                "team": 1,
                "element_type": 3,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "penalties_order": 1,
                "corners_and_indirect_freekicks_order": 1,
                "minutes": 880,
                "starts": 10,
            },
            {
                "id": 103,
                "first_name": "Martin",
                "second_name": "Odegaard",
                "web_name": "Odegaard",
                "team": 1,
                "element_type": 3,
                "status": "i",
                "chance_of_playing_next_round": 0,
                "minutes": 450,
                "starts": 5,
            },
            {
                "id": 104,
                "first_name": "Declan",
                "second_name": "Rice",
                "web_name": "Rice",
                "team": 1,
                "element_type": 3,
                "status": "d",
                "chance_of_playing_next_round": 75,
                "corners_and_indirect_freekicks_order": 2,
                "minutes": 850,
                "starts": 10,
            },
            {
                "id": 201,
                "first_name": "Ederson",
                "second_name": "Moraes",
                "web_name": "Ederson",
                "team": 2,
                "element_type": 1,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "minutes": 900,
                "starts": 10,
            },
            {
                "id": 202,
                "first_name": "Erling",
                "second_name": "Haaland",
                "web_name": "Haaland",
                "team": 2,
                "element_type": 4,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "penalties_order": 1,
                "minutes": 900,
                "starts": 10,
            },
        ],
    }


def _sample_fpl_fixtures_payload(*, match2_finished: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "event": 1,
            "kickoff_time": "2024-08-17T14:00:00Z",
            "team_h": 1,
            "team_a": 2,
            "team_h_score": 2,
            "team_a_score": 1,
            "finished": True,
        },
        {
            "id": 2,
            "event": 2,
            "kickoff_time": "2024-08-24T16:30:00Z",
            "team_h": 2,
            "team_a": 1,
            "team_h_score": 2 if match2_finished else None,
            "team_a_score": 2 if match2_finished else None,
            "finished": match2_finished,
        },
    ]


def _sample_football_data_csv() -> str:
    return (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR\n"
        "E0,17/08/2024,15:00,Arsenal,Man City,2,1,H,1,0,H,15,11,6,4,10,12,7,4,2,3,0,0\n"
        "E0,24/08/2024,17:30,Man City,Arsenal,2,2,D,1,1,D,18,9,7,3,9,14,8,2,1,4,0,1\n"
    )


def _sample_understat_teams_payload() -> dict[str, Any]:
    return {
        "88": {
            "id": "88",
            "title": "Arsenal",
            "history": [
                {
                    "h_a": "h",
                    "xG": 1.85,
                    "xGA": 1.10,
                    "npxG": 1.85,
                    "npxGA": 1.10,
                    "ppda": {"att": 210, "def": 24},
                    "ppda_allowed": {"att": 260, "def": 20},
                    "deep": 9,
                    "deep_allowed": 5,
                    "date": "2024-08-17 14:00:00",
                }
            ],
        },
        "89": {
            "id": "89",
            "title": "Manchester City",
            "history": [
                {
                    "h_a": "a",
                    "xG": 1.10,
                    "xGA": 1.85,
                    "npxG": 1.10,
                    "npxGA": 1.85,
                    "ppda": {"att": 260, "def": 20},
                    "ppda_allowed": {"att": 210, "def": 24},
                    "deep": 5,
                    "deep_allowed": 9,
                    "date": "2024-08-17 14:00:00",
                }
            ],
        },
    }


def test_fpl_understat_football_data_parsing_and_silver_reconciliation(tmp_path: Path) -> None:
    """Verify FPL, Understat, and Football-Data parsing, FUT->FINAL upsert, and Understat lag."""
    season = 20242025
    assert derive_active_epl_season(date(2024, 9, 1)) == 20242025
    assert derive_active_epl_season(date(2025, 4, 15)) == 20242025
    assert season_to_football_data_code(20242025) == "2425"

    bootstrap = _sample_fpl_bootstrap_payload()
    teams_map = parse_bootstrap_teams_map(bootstrap)
    squad_df = parse_bootstrap_squad_df(bootstrap)
    assert teams_map == {1: "ARS", 2: "MCI"}
    assert squad_df.height == 6

    # Initial ingest: Match 1 is FINAL (2-1), Match 2 is FUT
    fixtures_initial = parse_fixtures_payload(
        _sample_fpl_fixtures_payload(match2_finished=False),
        teams_map,
        season=season,
    )
    assert dict(fixtures_initial.schema) == SILVER_MATCHES_SCHEMA
    assert fixtures_initial.filter(pl.col("game_state") == "FINAL").height == 1
    assert fixtures_initial.filter(pl.col("game_state") == "FUT").height == 1

    reconciler = SoccerSilverReconciler(silver_dir=tmp_path / "silver" / "soccer")
    reconciled_m1 = reconciler.reconcile_matches(fixtures_initial)
    assert reconciled_m1.height == 2

    # Incremental upsert: Match 2 transitions from FUT to FINAL (2-2 draw -> outcome_1x2 == 1)
    fixtures_updated = parse_fixtures_payload(
        _sample_fpl_fixtures_payload(match2_finished=True),
        teams_map,
        season=season,
    )
    reconciled_m2 = reconciler.reconcile_matches(fixtures_updated)
    assert reconciled_m2.height == 2
    assert reconciled_m2.select("match_id").n_unique() == 2
    m2_row = reconciled_m2.filter(pl.col("matchweek") == 2).to_dicts()[0]
    assert m2_row["game_state"] == "FINAL"
    assert m2_row["outcome_1x2"] == 1
    assert m2_row["is_draw"] == 1

    # Reconcile team match logs: Match 1 has Understat xG; Match 2 lags in Understat (fallback)
    _, fd_box_df = parse_football_data_csv(_sample_football_data_csv(), season=season)
    us_teams_df = parse_understat_team_matches_payload(
        _sample_understat_teams_payload(), season=season
    )
    team_logs = reconciler.reconcile_team_match_logs(
        reconciled_m2,
        football_data_box_df=fd_box_df,
        understat_teams_df=us_teams_df,
    )
    assert dict(team_logs.schema) == SILVER_TEAM_MATCH_LOGS_SCHEMA
    assert team_logs.height == 4
    assert team_logs.select(["match_id", "team"]).n_unique() == 4

    m1_ars = team_logs.filter(
        (pl.col("match_date") == date(2024, 8, 17)) & (pl.col("team") == "ARS")
    ).to_dicts()[0]
    assert m1_ars["has_understat_xg"] is True
    assert m1_ars["npxgf"] == pytest.approx(1.85)
    assert m1_ars["shots_for"] == 15

    # Match 2 (2024-08-24) was absent from Understat fixture -> graceful fallback
    m2_mci = team_logs.filter(
        (pl.col("match_date") == date(2024, 8, 24)) & (pl.col("team") == "MCI")
    ).to_dicts()[0]
    assert m2_mci["has_understat_xg"] is False
    assert m2_mci["xgf"] > 0.0
    assert m2_mci["shots_for"] == 18

    # Player logs & Understat enrichment
    saka_summary = {
        "history": [
            {
                "fixture": 1,
                "opponent_team": 2,
                "was_home": True,
                "kickoff_time": "2024-08-17T14:00:00Z",
                "minutes": 90,
                "starts": 1,
                "goals_scored": 1,
                "assists": 1,
                "expected_goals": "0.52",
                "expected_assists": "0.38",
                "expected_goal_involvements": "0.90",
                "yellow_cards": 0,
                "red_cards": 0,
            }
        ]
    }
    raya_summary = {
        "history": [
            {
                "fixture": 1,
                "opponent_team": 2,
                "was_home": True,
                "kickoff_time": "2024-08-17T14:00:00Z",
                "minutes": 90,
                "starts": 1,
                "saves": 3,
                "goals_conceded": 1,
                "clean_sheets": 0,
                "expected_goals_conceded": "1.10",
            }
        ]
    }
    saka_out, _ = parse_element_summary_payload(
        102, "Bukayo Saka", "ARS", "MID", saka_summary, teams_map, season=season
    )
    _, raya_gk = parse_element_summary_payload(
        101, "David Raya", "ARS", "GKP", raya_summary, teams_map, season=season
    )

    us_players_df = parse_understat_player_matches_payload(
        [
            {
                "date": "2024-08-17",
                "player_name": "Bukayo Saka",
                "team": "Arsenal",
                "shots": 4,
                "sot": 2,
                "key_passes": 3,
                "xG": 0.58,
                "npxG": 0.58,
                "xA": 0.41,
                "xGChain": 1.15,
                "xGBuildup": 0.35,
            }
        ],
        season=season,
    )
    outfield_logs = reconciler.reconcile_outfield_match_logs(saka_out, us_players_df)
    gk_logs = reconciler.reconcile_gk_match_logs(raya_gk, team_logs_df=team_logs)

    assert dict(outfield_logs.schema) == SILVER_OUTFIELD_MATCH_LOGS_SCHEMA
    assert dict(gk_logs.schema) == SILVER_GK_MATCH_LOGS_SCHEMA
    assert outfield_logs.select(["match_id", "player_id"]).n_unique() == outfield_logs.height
    assert gk_logs.select(["match_id", "player_id"]).n_unique() == gk_logs.height
    assert outfield_logs["npxg"][0] == pytest.approx(0.58)
    assert outfield_logs["xg_chain"][0] == pytest.approx(1.15)
    assert gk_logs["gsaa"][0] == pytest.approx(1.10 - 1.0)


def test_two_tier_soccer_lineup_resolver() -> None:
    """Verify Tier 1 official starting XI, Tier 2 FPL injury/suspension resolution, and fallback."""
    bootstrap = _sample_fpl_bootstrap_payload()
    squad_df = parse_bootstrap_squad_df(bootstrap)
    resolver = SoccerPreMatchLineupResolver()

    # Tier 1: Confirmed official starting XI (~60m pre-kickoff)
    official_payload = {
        "ARS": {
            "starting_gk_id": 101,
            "starting_outfield_ids": [102, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            "bench_outfield_ids": [113, 114],
        }
    }
    tier1 = resolver.resolve_team_lineup(
        "Arsenal",
        official_lineup_payload=official_payload,
        squad_df=squad_df,
    )
    assert tier1.lineup_status == "CONFIRMED_OFFICIAL_XI"
    assert tier1.starting_gk_id == 101
    assert 102 in tier1.starting_outfield_ids
    assert 103 in tier1.excluded_unavailable_ids

    # Tier 2: Daytime FPL availability (Odegaard id=103 injured excluded; Rice id=104 doubtful)
    tier2 = resolver.resolve_team_lineup("Arsenal", squad_df=squad_df)
    assert tier2.lineup_status == "FPL_PROJECTED_XI"
    assert tier2.starting_gk_id == 101
    assert 103 in tier2.excluded_unavailable_ids
    assert 103 not in tier2.active_outfield_ids
    assert 104 in tier2.doubtful_player_ids
    assert 102 in tier2.starting_outfield_ids

    # Fallback: Heuristic when no live squad/official feed is provided
    fallback = resolver.resolve_team_lineup("Arsenal")
    assert fallback.lineup_status == "HEURISTIC_XI"


def test_cached_clients_and_health_checks_offline(tmp_path: Path) -> None:
    """Verify FPL, Understat, FootballData clients and health checks via MockTransport."""
    bootstrap = _sample_fpl_bootstrap_payload()
    fixtures = _sample_fpl_fixtures_payload(match2_finished=True)
    us_payload = {"teams": _sample_understat_teams_payload()}
    fd_csv = _sample_football_data_csv()

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "bootstrap-static" in url:
            return httpx.Response(200, json=bootstrap)
        if "fixtures" in url:
            return httpx.Response(200, json=fixtures)
        if "element-summary/102" in url:
            return httpx.Response(200, json={"history": []})
        if "getLeagueData/EPL" in url:
            return httpx.Response(200, json=us_payload)
        if "E0.csv" in url:
            return httpx.Response(200, text=fd_csv)
        return httpx.Response(404)

    transport = httpx.MockTransport(_handler)
    results = check_soccer_data_sources(
        tmp_path / "raw",
        season=20242025,
        transport=transport,
    )
    assert len(results) == 3
    assert all(r.healthy for r in results)

    # Verify disk cache hits work offline without further network calls
    fpl_client = FPLAPIClient(cache_dir=tmp_path / "raw" / "fpl_api")
    assert "teams" in fpl_client.fetch_bootstrap_static(ttl_seconds=3600.0)
    assert len(fpl_client.fetch_fixtures(ttl_seconds=3600.0)) == 2

    us_client = UnderstatClient(cache_dir=tmp_path / "raw" / "understat")
    assert us_client.fetch_season_team_matches(20242025, ttl_seconds=3600.0).height == 2

    fd_client = FootballDataClient(cache_dir=tmp_path / "raw" / "football_data")
    m_df, b_df = fd_client.fetch_season_csv(20242025, is_active_season=False)
    assert m_df.height == 2
    assert b_df.height == 4
    _ = json.dumps(bootstrap)
