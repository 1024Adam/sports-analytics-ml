"""NHL ingestion, tricode normalization, 2-Tier lineups, and Silver reconciliation tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from sports_ml.core.contracts import SportAdapter
from sports_ml.sports.hockey import (
    NHL_TEAMS_32,
    SILVER_GAMES_SCHEMA,
    SILVER_GOALIE_GAME_LOGS_SCHEMA,
    SILVER_SKATER_GAME_LOGS_SCHEMA,
    SILVER_TEAM_GAME_LOGS_SCHEMA,
    HockeyAdapter,
    get_arena_metadata,
    normalize_tricode,
)
from sports_ml.sports.hockey.ingestion import (
    HockeySilverReconciler,
    MoneyPuckClient,
    NHLWebAPIClient,
    PreGameLineupResolver,
    check_hockey_data_sources,
    derive_active_nhl_season,
    parse_boxscore_payload,
    parse_roster_payload,
    parse_schedule_payload,
    parse_toi_minutes,
)


def _sample_schedule_payload(game2_state: str = "FUT") -> dict[str, Any]:
    game2: dict[str, Any] = {
        "id": 2024020002,
        "season": 20242025,
        "gameType": 2,
        "startTimeUTC": "2024-10-12T23:00:00Z",
        "gameState": game2_state,
        "homeTeam": {"abbrev": "S.J", "score": 3 if game2_state == "FINAL" else None},
        "awayTeam": {"abbrev": "ARI", "score": 2 if game2_state == "FINAL" else None},
        "gameOutcome": {"lastPeriodType": "SO"} if game2_state == "FINAL" else {},
    }
    return {
        "gameWeek": [
            {
                "date": "2024-10-10",
                "games": [
                    {
                        "id": 2024010001,
                        "season": 20242025,
                        "gameType": 1,  # Preseason -> must be excluded by default
                        "startTimeUTC": "2024-10-05T23:00:00Z",
                        "gameState": "FINAL",
                        "homeTeam": {"abbrev": "BOS", "score": 3},
                        "awayTeam": {"abbrev": "TOR", "score": 1},
                    },
                    {
                        "id": 2024020001,
                        "season": 20242025,
                        "gameType": 2,
                        "startTimeUTC": "2024-10-10T23:00:00Z",
                        "gameState": "FINAL",
                        "homeTeam": {"abbrev": "BOS", "score": 4},
                        "awayTeam": {"abbrev": "TOR", "score": 2},
                        "gameOutcome": {"lastPeriodType": "REG"},
                    },
                ],
            },
            {
                "date": "2024-10-12",
                "games": [game2],
            },
        ]
    }


def _sample_boxscore_payload(game_id: int = 2024020001) -> dict[str, Any]:
    return {
        "id": game_id,
        "season": 20242025,
        "gameDate": "2024-10-10",
        "startTimeUTC": "2024-10-10T23:00:00Z",
        "gameState": "FINAL",
        "gameOutcome": {"lastPeriodType": "REG"},
        "homeTeam": {"abbrev": "BOS", "score": 4, "sog": 32},
        "awayTeam": {"abbrev": "TOR", "score": 2, "sog": 28},
        "playerByGameStats": {
            "homeTeam": {
                "forwards": [
                    {
                        "playerId": 8477956,
                        "name": {"default": "David Pastrnak"},
                        "toi": "19:30",
                        "powerPlayToi": "03:30",
                        "shorthandedToi": "00:00",
                        "goals": 2,
                        "assists": 1,
                        "points": 3,
                        "sog": 6,
                        "powerPlayGoals": 1,
                        "blockedShots": 1,
                        "pim": 2,
                    }
                ],
                "defense": [
                    {
                        "playerId": 8479325,
                        "name": {"default": "Charlie McAvoy"},
                        "toi": "24:15",
                        "powerPlayToi": "02:15",
                        "shorthandedToi": "03:00",
                        "goals": 0,
                        "assists": 2,
                        "points": 2,
                        "sog": 3,
                        "powerPlayGoals": 0,
                        "blockedShots": 3,
                        "pim": 0,
                    }
                ],
                "goalies": [
                    {
                        "playerId": 8480280,
                        "name": {"default": "Jeremy Swayman"},
                        "starter": True,
                        "toi": "60:00",
                        "shotsAgainst": 28,
                        "saves": 26,
                        "goalsAgainst": 2,
                    },
                    {
                        "playerId": 8476914,
                        "name": {"default": "Joonas Korpisalo"},
                        "starter": False,
                        "toi": "00:00",
                        "shotsAgainst": 0,
                        "saves": 0,
                        "goalsAgainst": 0,
                    },
                ],
            },
            "awayTeam": {
                "forwards": [
                    {
                        "playerId": 8479318,
                        "name": {"default": "Auston Matthews"},
                        "toi": "21:00",
                        "powerPlayToi": "04:00",
                        "shorthandedToi": "00:30",
                        "goals": 1,
                        "assists": 0,
                        "points": 1,
                        "sog": 5,
                        "powerPlayGoals": 0,
                        "blockedShots": 1,
                        "pim": 0,
                    }
                ],
                "defense": [
                    {
                        "playerId": 8476853,
                        "name": {"default": "Morgan Rielly"},
                        "toi": "23:00",
                        "powerPlayToi": "03:00",
                        "shorthandedToi": "01:30",
                        "goals": 0,
                        "assists": 1,
                        "points": 1,
                        "sog": 2,
                        "powerPlayGoals": 0,
                        "blockedShots": 2,
                        "pim": 0,
                    }
                ],
                "goalies": [
                    {
                        "playerId": 8479361,
                        "name": {"default": "Joseph Woll"},
                        "starter": True,
                        "toi": "58:45",
                        "shotsAgainst": 32,
                        "saves": 28,
                        "goalsAgainst": 4,
                    }
                ],
            },
        },
    }


def _sample_roster_payload() -> dict[str, Any]:
    return {
        "forwards": [
            {
                "id": 8477956,
                "firstName": {"default": "David"},
                "lastName": {"default": "Pastrňák"},
                "positionCode": "R",
            },
            {
                "id": 8473419,
                "firstName": {"default": "Brad"},
                "lastName": {"default": "Marchand"},
                "positionCode": "L",
            },
        ],
        "defensemen": [
            {
                "id": 8479325,
                "firstName": {"default": "Charlie"},
                "lastName": {"default": "McAvoy"},
                "positionCode": "D",
            }
        ],
        "goalies": [
            {
                "id": 8480280,
                "firstName": {"default": "Jeremy"},
                "lastName": {"default": "Swayman"},
                "positionCode": "G",
            },
            {
                "id": 8476914,
                "firstName": {"default": "Joonas"},
                "lastName": {"default": "Korpisalo"},
                "positionCode": "G",
            },
        ],
    }


def test_hockey_32_team_registry_and_tricode_normalization() -> None:
    """Verify 32-team registry, arena coordinates, and legacy tricode normalization."""
    adapter = HockeyAdapter()
    assert isinstance(adapter, SportAdapter)
    assert adapter.sport_slug == "hockey"
    assert adapter.topology == "head_to_head"
    assert len(NHL_TEAMS_32) == 32

    assert normalize_tricode("S.J") == "SJS"
    assert normalize_tricode("L.A") == "LAK"
    assert normalize_tricode("N.J") == "NJD"
    assert normalize_tricode("T.B") == "TBL"
    assert normalize_tricode("ARI") == "UTA"
    assert normalize_tricode("PHX") == "UTA"
    assert normalize_tricode("ATL") == "WPG"
    assert normalize_tricode("bos") == "BOS"

    uta_arena = get_arena_metadata("ARI")
    assert uta_arena is not None
    assert uta_arena.team == "UTA"
    assert uta_arena.timezone == "America/Denver"


def test_toi_parsing_and_season_derivation() -> None:
    """Verify MM:SS TOI parsing and active NHL season derivation."""
    assert parse_toi_minutes("19:45") == pytest.approx(19.75)
    assert parse_toi_minutes("00:30") == pytest.approx(0.5)
    assert parse_toi_minutes(None) == pytest.approx(0.0)
    assert parse_toi_minutes("") == pytest.approx(0.0)
    assert parse_toi_minutes(15.5) == pytest.approx(15.5)

    assert derive_active_nhl_season(date(2026, 10, 24)) == 20262027
    assert derive_active_nhl_season(date(2026, 4, 15)) == 20252026


def test_schedule_and_boxscore_parsing_and_shootout_reg_goals() -> None:
    """Verify preseason exclusion, shootout reg_goals subtraction, and schema adherence."""
    # First parse with game 2 as FUT, then as FINAL (SO 3-2)
    sched_fut = parse_schedule_payload(_sample_schedule_payload(game2_state="FUT"))
    assert sched_fut.height == 2  # Preseason game 2024010001 excluded
    assert dict(sched_fut.schema) == SILVER_GAMES_SCHEMA

    row_fut = sched_fut.filter(pl.col("game_id") == 2024020002).row(0, named=True)
    assert row_fut["home_team"] == "SJS"
    assert row_fut["away_team"] == "UTA"
    assert row_fut["game_state"] == "FUT"
    assert row_fut["home_goals"] is None
    assert row_fut["home_win"] is None

    sched_final = parse_schedule_payload(_sample_schedule_payload(game2_state="FINAL"))
    row_final = sched_final.filter(pl.col("game_id") == 2024020002).row(0, named=True)
    assert row_final["game_state"] == "FINAL"
    assert row_final["finish_period_type"] == "SO"
    assert row_final["home_goals"] == 3
    assert row_final["away_goals"] == 2
    # Shootout winner goal subtracted for reg_goals!
    assert row_final["home_reg_goals"] == 2
    assert row_final["away_reg_goals"] == 2
    assert row_final["home_win"] == 1

    team_df, skater_df, goalie_df = parse_boxscore_payload(_sample_boxscore_payload())
    assert dict(team_df.schema) == SILVER_TEAM_GAME_LOGS_SCHEMA
    assert dict(skater_df.schema) == SILVER_SKATER_GAME_LOGS_SCHEMA
    assert dict(goalie_df.schema) == SILVER_GOALIE_GAME_LOGS_SCHEMA
    assert team_df.height == 2
    assert skater_df.height == 4
    assert goalie_df.height == 2  # Backup with 00:00 TOI and starter=False excluded


def test_nhl_api_client_caching_and_deprecated_url_guard(tmp_path: Path) -> None:
    """Verify permanent caching for FINAL games and rejection of deprecated NHL API URL."""
    with pytest.raises(ValueError, match="Deprecated NHL API"):
        NHLWebAPIClient(tmp_path, base_url="https://statsapi.web.nhl.com/api/v1")

    call_counts: dict[str, int] = {"boxscore": 0, "schedule": 0}

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "boxscore" in path:
            call_counts["boxscore"] += 1
            return httpx.Response(200, json=_sample_boxscore_payload(2024020001))
        if "schedule" in path:
            call_counts["schedule"] += 1
            state = "FUT" if call_counts["schedule"] == 1 else "FINAL"
            return httpx.Response(200, json=_sample_schedule_payload(game2_state=state))
        if "roster" in path:
            return httpx.Response(200, json=_sample_roster_payload())
        return httpx.Response(404)

    client = NHLWebAPIClient(
        cache_dir=tmp_path / "nhl_api",
        transport=httpx.MockTransport(_mock_handler),
    )

    # Boxscore is FINAL -> fetched once, cached permanently on second call
    b1 = client.fetch_game_boxscore(2024020001)
    b2 = client.fetch_game_boxscore(2024020001)
    assert b1["id"] == b2["id"] == 2024020001
    assert call_counts["boxscore"] == 1

    # Schedule week has a FUT game on call 1 -> not cached permanently; re-queried on call 2
    s1 = client.fetch_schedule_by_date("2024-10-10")
    s2 = client.fetch_schedule_by_date("2024-10-10")
    s3 = client.fetch_schedule_by_date("2024-10-10")
    assert s1["gameWeek"][1]["games"][0]["gameState"] == "FUT"
    assert s2["gameWeek"][1]["games"][0]["gameState"] == "FINAL"
    assert s3["gameWeek"][1]["games"][0]["gameState"] == "FINAL"
    assert call_counts["schedule"] == 2  # Call 3 served from permanent cache once all games FINAL


def test_moneypuck_csv_parsing_and_silver_reconciliation(tmp_path: Path) -> None:
    """Verify MoneyPuck parsing, incremental FUT->FINAL upsert, fallback, and Silver uniqueness."""
    mp_teams_csv = (
        b"gameId,team,situation,xGoalsFor,xGoalsAgainst,shotAttemptsFor,shotAttemptsAgainst,"
        b"unblockedShotAttemptsFor,unblockedShotAttemptsAgainst,highDangerxGoalsFor,"
        b"highDangerxGoalsAgainst,iceTime,goalsFor,goalsAgainst\n"
        b"2024020001,BOS,5on5,2.75,1.85,55,44,41,32,1.20,0.75,2940,3,1\n"
        b"2024020001,BOS,5on4,0.85,0.05,12,1,9,1,0.45,0.0,360,1,0\n"
        b"2024020001,BOS,4on5,0.05,0.60,2,10,1,8,0.0,0.30,240,0,1\n"
        b"2024020001,TOR,5on5,1.85,2.75,44,55,32,41,0.75,1.20,2940,1,3\n"
        b"2024020001,TOR,5on4,0.60,0.05,10,2,8,1,0.30,0.0,240,1,0\n"
        b"2024020001,TOR,4on5,0.05,0.85,1,12,1,9,0.0,0.45,360,0,1\n"
    )
    mp_skaters_csv = (
        b"gameId,playerId,playerTeam,situation,I_F_xGoals,I_F_shotAttempts,"
        b"OnIce_F_xGoals,OnIce_A_xGoals,gameScore\n"
        b"2024020001,8477956,BOS,all,0.92,9,1.80,0.50,2.85\n"
        b"2024020001,8477956,BOS,5on5,0.65,6,1.25,0.45,2.85\n"
    )
    mp_goalies_csv = (
        b"gameId,playerId,playerTeam,situation,xGoals,goals\n"
        b"2024020001,8480280,BOS,all,2.65,2\n"
        b"2024020001,8479361,TOR,all,3.55,4\n"
    )

    mp_client = MoneyPuckClient(cache_dir=tmp_path / "moneypuck")
    mp_teams_df = mp_client.load_team_game_stats(2024, csv_bytes=mp_teams_csv)
    mp_skaters_df = mp_client.load_skater_game_stats(2024, csv_bytes=mp_skaters_csv)
    mp_goalies_df = mp_client.load_goalie_game_stats(2024, csv_bytes=mp_goalies_csv)

    # Verify iceTime (2940 sec) divided by 60 -> 49.0 minutes
    bos_mp = mp_teams_df.filter(pl.col("team") == "BOS").row(0, named=True)
    assert bos_mp["toi_5v5_minutes"] == pytest.approx(49.0)
    assert bos_mp["xgf_5v5"] == pytest.approx(2.75)

    reconciler = HockeySilverReconciler(silver_dir=tmp_path / "silver" / "hockey")

    # 1. Reconcile games with game 2024020002 initially FUT, then upserted as FINAL
    games_v1 = reconciler.reconcile_games(
        parse_schedule_payload(_sample_schedule_payload(game2_state="FUT"))
    )
    assert games_v1.filter(pl.col("game_id") == 2024020002)["game_state"][0] == "FUT"

    games_v2 = reconciler.reconcile_games(
        parse_schedule_payload(_sample_schedule_payload(game2_state="FINAL"))
    )
    assert games_v2.height == 2
    assert games_v2["game_id"].n_unique() == 2
    assert games_v2.filter(pl.col("game_id") == 2024020002)["game_state"][0] == "FINAL"

    # 2. Reconcile Game 1 (with MoneyPuck) and Game 2 (MoneyPuck lags -> graceful fallback)
    t1, s1, g1 = parse_boxscore_payload(_sample_boxscore_payload(2024020001))
    t2, s2, g2 = parse_boxscore_payload(_sample_boxscore_payload(2024020002))

    team_logs = reconciler.reconcile_team_game_logs(
        pl.concat([t1, t2]), moneypuck_teams_df=mp_teams_df
    )
    skater_logs = reconciler.reconcile_skater_game_logs(
        pl.concat([s1, s2]), moneypuck_skaters_df=mp_skaters_df
    )
    goalie_logs = reconciler.reconcile_goalie_game_logs(
        pl.concat([g1, g2]), moneypuck_goalies_df=mp_goalies_df
    )

    # Verify uniqueness constraints on Silver tables
    assert team_logs.select(["game_id", "team"]).n_unique() == team_logs.height
    assert skater_logs.select(["game_id", "player_id"]).n_unique() == skater_logs.height
    assert goalie_logs.select(["game_id", "player_id"]).n_unique() == goalie_logs.height

    # Game 2024020001 has MoneyPuck xG; Game 2024020002 falls back (has_moneypuck_xg=False)
    g1_bos = team_logs.filter((pl.col("game_id") == 2024020001) & (pl.col("team") == "BOS")).row(
        0, named=True
    )
    g2_bos = team_logs.filter((pl.col("game_id") == 2024020002) & (pl.col("team") == "BOS")).row(
        0, named=True
    )
    assert g1_bos["has_moneypuck_xg"] is True
    assert g1_bos["xgf_5v5"] == pytest.approx(2.75)
    assert g2_bos["has_moneypuck_xg"] is False
    assert g2_bos["xgf_5v5"] > 0.0

    swayman_g1 = goalie_logs.filter(
        (pl.col("game_id") == 2024020001) & (pl.col("player_id") == 8480280)
    ).row(0, named=True)
    assert swayman_g1["gsax"] == pytest.approx(0.65)


def test_two_tier_pregame_lineup_goalie_and_injury_resolver() -> None:
    """Verify Tier 1 warmup payload, Tier 2 morning-skate + injury matching, and heuristic."""
    resolver = PreGameLineupResolver()
    roster_df = parse_roster_payload("BOS", _sample_roster_payload())

    # Tier 1: Official NHL API boxscore with dressed lineup & starter
    tier1_res = resolver.resolve_team_lineup(
        "BOS",
        roster_df,
        boxscore_payload=_sample_boxscore_payload(),
    )
    assert tier1_res.goalie_status == "CONFIRMED_NHL_OFFICIAL"
    assert tier1_res.starting_goalie_id == 8480280
    assert 8477956 in tier1_res.active_skater_ids
    assert 8473419 in tier1_res.excluded_injured_ids  # Marchand on roster but not dressed

    # Tier 2: Daytime morning-skate payload with diacritic stripping & injury exclusion
    morning_skate = {
        "starting_goalie_name": "Jeremy Swayman",
        "goalie_status": "Confirmed",
        "injured_players": ["Brad Marchand"],
        "projected_skaters": ["David Pastrnak", "Brad Marchand", "Charlie McAvoy"],
    }
    tier2_res = resolver.resolve_team_lineup(
        "BOS",
        roster_df,
        morning_skate_payload=morning_skate,
    )
    assert tier2_res.goalie_status == "CONFIRMED_MORNING_SKATE"
    assert tier2_res.starting_goalie_id == 8480280
    assert tier2_res.excluded_injured_ids == [8473419]
    assert tier2_res.active_skater_ids == [8477956, 8479325]

    # Fallback Heuristic: B2B switches from last night's primary starter to backup
    _, _, g_logs = parse_boxscore_payload(_sample_boxscore_payload())
    fallback_res = resolver.resolve_team_lineup(
        "BOS",
        roster_df,
        recent_goalie_logs=g_logs,
        is_b2b=True,
    )
    assert fallback_res.goalie_status == "PROJECTED_HEURISTIC"
    assert fallback_res.starting_goalie_id == 8480280  # Only 1 starter in history; stays valid
    assert len(fallback_res.active_skater_ids) == 3

    # Verify json serializability of ResolvedTeamLineup
    assert json.loads(tier2_res.model_dump_json())["team"] == "BOS"


def test_expand_target_seasons_and_cli_from_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--from-year` and season range syntax expand to all seasons through active."""
    from typer.testing import CliRunner

    from sports_ml.cli import app, expand_target_seasons

    # 1. Direct helper unit tests
    assert expand_target_seasons(active_season=20262027, from_year=2023) == [
        20232024,
        20242025,
        20252026,
        20262027,
    ]
    assert expand_target_seasons(active_season=20262027, from_year=20242025) == [
        20242025,
        20252026,
        20262027,
    ]
    assert expand_target_seasons(active_season=20262027, seasons="2023-2025") == [
        20232024,
        20242025,
        20252026,
    ]
    assert expand_target_seasons(active_season=20262027, seasons="2024+") == [
        20242025,
        20252026,
        20262027,
    ]
    assert expand_target_seasons(active_season=20262027, seasons="20232024-20252026") == [
        20232024,
        20242025,
        20252026,
    ]

    with pytest.raises(ValueError, match="cannot be after active season"):
        expand_target_seasons(active_season=20252026, from_year=2028)
    with pytest.raises(ValueError, match="expected consecutive years"):
        expand_target_seasons(active_season=20252026, seasons="20232025")
    with pytest.raises(ValueError, match="end year"):
        expand_target_seasons(active_season=20252026, seasons="2025-2023")

    # 2. CLI `--from-year` offline invocation for hockey and soccer
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))
    runner = CliRunner()

    res_h = runner.invoke(
        app,
        ["ingest", "--sport", "hockey", "--from-year", "2023", "--offline"],
    )
    assert res_h.exit_code == 0, res_h.stdout
    assert "Ingestion complete" in res_h.stdout
    assert "seasons=20232024,20242025" in res_h.stdout

    res_s = runner.invoke(
        app,
        ["ingest", "--sport", "soccer", "--from-year", "2023", "--offline"],
    )
    assert res_s.exit_code == 0, res_s.stdout
    assert "Ingestion complete" in res_s.stdout
    assert "seasons=20232024,20242025" in res_s.stdout

    res_bad = runner.invoke(
        app,
        ["ingest", "--sport", "hockey", "--from-year", "2099", "--offline"],
    )
    assert res_bad.exit_code != 0


def test_data_source_health_probes_success_and_failure(tmp_path: Path) -> None:
    """Verify upstream health probes validate NHL API, MoneyPuck, and DailyFaceoff contracts."""
    mp_teams_csv = (
        b"gameId,team,situation,xGoalsFor,xGoalsAgainst,shotAttemptsFor,shotAttemptsAgainst,"
        b"unblockedShotAttemptsFor,unblockedShotAttemptsAgainst,highDangerxGoalsFor,"
        b"highDangerxGoalsAgainst,iceTime,goalsFor,goalsAgainst\n"
        b"2024020001,BOS,5on5,2.75,1.85,55,44,41,32,1.20,0.75,2940,3,1\n"
        b"2024020001,BOS,5on4,0.85,0.05,12,1,9,1,0.45,0.0,360,1,0\n"
        b"2024020001,BOS,4on5,0.05,0.60,2,10,1,8,0.0,0.30,240,0,1\n"
    )
    df_html = (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"data":[{"homeTeamName":"Boston Bruins",'
        '"homeGoalieName":"Jeremy Swayman","homeNewsStrengthName":"Confirmed"}]}}}'
        "</script></body></html>"
    )

    def _healthy_nhl(req: httpx.Request) -> httpx.Response:
        if "schedule" in req.url.path:
            return httpx.Response(200, json=_sample_schedule_payload())
        if "roster" in req.url.path:
            return httpx.Response(200, json=_sample_roster_payload())
        return httpx.Response(404)

    def _healthy_mp(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=mp_teams_csv)

    def _healthy_df(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=df_html)

    results = check_hockey_data_sources(
        tmp_path,
        target_date=date(2024, 10, 10),
        team_code="BOS",
        nhl_transport=httpx.MockTransport(_healthy_nhl),
        moneypuck_transport=httpx.MockTransport(_healthy_mp),
        dailyfaceoff_transport=httpx.MockTransport(_healthy_df),
    )
    assert len(results) == 4
    assert all(r.healthy for r in results)

    # Simulate broken DailyFaceoff HTML (missing __NEXT_DATA__)
    def _broken_df(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>Redesign without Next data</body></html>")

    broken_results = check_hockey_data_sources(
        tmp_path,
        target_date=date(2024, 10, 10),
        team_code="BOS",
        nhl_transport=httpx.MockTransport(_healthy_nhl),
        moneypuck_transport=httpx.MockTransport(_healthy_mp),
        dailyfaceoff_transport=httpx.MockTransport(_broken_df),
    )
    assert broken_results[3].healthy is False
    assert "__NEXT_DATA__" in broken_results[3].details


def test_moneypuck_shared_teams_all_cache_and_gzip_storage(tmp_path: Path) -> None:
    """Verify `MoneyPuckClient` downloads `all_teams.csv` once into `teams_all.csv.gz`."""
    cumulative_teams_csv = (
        b"gameId,season,team,situation,xGoalsFor,xGoalsAgainst,shotAttemptsFor,shotAttemptsAgainst,"
        b"unblockedShotAttemptsFor,unblockedShotAttemptsAgainst,highDangerxGoalsFor,"
        b"highDangerxGoalsAgainst,iceTime,goalsFor,goalsAgainst\n"
        b"2023020001,2023,BOS,5on5,2.50,1.90,50,45,38,33,1.10,0.80,2940,3,2\n"
        b"2023020001,2023,BOS,5on4,0.60,0.05,10,1,8,1,0.30,0.0,300,1,0\n"
        b"2023020001,2023,BOS,4on5,0.05,0.50,1,9,1,7,0.0,0.25,240,0,0\n"
        b"2024020001,2024,BOS,5on5,2.75,1.85,55,44,41,32,1.20,0.75,2940,3,1\n"
        b"2024020001,2024,BOS,5on4,0.85,0.05,12,1,9,1,0.45,0.0,360,1,0\n"
        b"2024020001,2024,BOS,4on5,0.05,0.60,2,10,1,8,0.0,0.30,240,0,1\n"
    )
    skaters_2024_csv = (
        b"gameId,season,playerId,playerTeam,situation,I_F_xGoals,I_F_shotAttempts,"
        b"OnIce_F_xGoals,OnIce_A_xGoals,gameScore\n"
        b"2024020001,2024,8477956,BOS,all,0.92,9,1.80,0.50,2.85\n"
        b"2024020001,2024,8477956,BOS,5on5,0.65,6,1.25,0.45,2.85\n"
    )
    goalies_2024_csv = (
        b"gameId,season,playerId,playerTeam,situation,xGoals,goals\n"
        b"2024020001,2024,8480280,BOS,all,2.65,2\n"
    )

    counts = {"teams": 0, "skaters": 0, "goalies": 0}

    def _mp_handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "all_teams.csv" in url:
            counts["teams"] += 1
            return httpx.Response(
                200,
                content=cumulative_teams_csv,
                headers={"ETag": '"mp-teams-v1"'},
            )
        if "skaters" in url:
            counts["skaters"] += 1
            return httpx.Response(200, content=skaters_2024_csv)
        if "goalies" in url:
            counts["goalies"] += 1
            return httpx.Response(200, content=goalies_2024_csv)
        return httpx.Response(404)

    mp_dir = tmp_path / "raw" / "moneypuck"
    mp_client = MoneyPuckClient(
        cache_dir=mp_dir,
        transport=httpx.MockTransport(_mp_handler),
    )

    # Loading 2023 and 2024 teams downloads `all_teams.csv` ONLY ONCE into `teams_all.csv.gz`
    t_2023 = mp_client.load_team_game_stats(2023, is_active_season=False)
    t_2024 = mp_client.load_team_game_stats(2024, is_active_season=False)
    s_2024 = mp_client.load_skater_game_stats(2024, is_active_season=False)
    g_2024 = mp_client.load_goalie_game_stats(2024, is_active_season=False)

    assert counts["teams"] == 1
    assert counts["skaters"] == 1
    assert counts["goalies"] == 1
    assert (mp_dir / "teams_all.csv.gz").exists()
    assert (mp_dir / "teams_all.meta.json").exists()
    assert (mp_dir / "skaters_2024.csv.gz").exists()
    assert (mp_dir / "goalies_2024.csv.gz").exists()
    assert not (mp_dir / "teams_2023.csv").exists()
    assert not (mp_dir / "teams_2024.csv").exists()

    assert t_2023.height == 1
    assert t_2023["game_id"][0] == 2023020001
    assert t_2024.height == 1
    assert t_2024["game_id"][0] == 2024020001
    assert s_2024.height == 1
    assert g_2024.height == 1

    # Verify NHLWebAPIClient writes `.json.gz` files on ingest
    nhl_dir = tmp_path / "raw" / "nhl_api"
    nhl_client = NHLWebAPIClient(
        cache_dir=nhl_dir,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json=_sample_boxscore_payload(2024020001)
                if "boxscore" in req.url.path
                else _sample_schedule_payload(game2_state="FINAL"),
            )
        ),
    )
    _ = nhl_client.fetch_schedule_by_date("2024-10-10")
    _ = nhl_client.fetch_game_boxscore(2024020001)
    assert (nhl_dir / "schedules" / "schedule_2024-10-10.json.gz").exists()
    assert (nhl_dir / "boxscores" / "2024020001.json.gz").exists()
