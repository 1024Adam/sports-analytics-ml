"""Offline unit tests for MLB & NPB ingestion, 2-Tier lineups, and Silver reconciliation."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from sports_ml.sports.baseball.constants import (
    SILVER_BASEBALL_BATTER_LOGS_SCHEMA,
    SILVER_BASEBALL_GAMES_SCHEMA,
    SILVER_BASEBALL_PITCHER_LOGS_SCHEMA,
    SILVER_BASEBALL_TEAM_LOGS_SCHEMA,
)
from sports_ml.sports.baseball.ingestion import (
    NPB_GAME_ID_OFFSET,
    BaseballPreGameLineupResolver,
    BaseballSilverReconciler,
    MLBStatsAPIClient,
    NPBClient,
    StatcastClient,
    check_baseball_data_sources,
    namespace_npb_game_id,
    parse_mlb_boxscore_payload,
    parse_mlb_roster_payload,
    parse_mlb_schedule_payload,
    parse_npb_boxscore_payload,
    parse_npb_roster_payload,
    parse_npb_schedule_payload,
    resolve_npb_sub_league_and_dh,
    resolve_npb_venue,
)


def _sample_mlb_schedule_payload() -> dict[str, Any]:
    return {
        "dates": [
            {
                "date": "2025-05-10",
                "games": [
                    {
                        "gamePk": 777001,
                        "gameNumber": 1,
                        "gameType": "R",
                        "season": "2025",
                        "officialDate": "2025-05-10",
                        "gameDate": "2025-05-10T17:10:00Z",
                        "scheduledInnings": 9,
                        "status": {"abstractGameState": "Final"},
                        "linescore": {"currentInning": 9},
                        "venue": {"name": "Dodger Stadium"},
                        "teams": {
                            "home": {
                                "team": {"abbreviation": "LAD", "name": "Los Angeles Dodgers"},
                                "score": 5,
                                "probablePitcher": {"id": 801, "pitchHand": {"code": "R"}},
                            },
                            "away": {
                                "team": {"abbreviation": "SD", "name": "San Diego Padres"},
                                "score": 2,
                                "probablePitcher": {"id": 802, "pitchHand": {"code": "L"}},
                            },
                        },
                    },
                    {
                        "gamePk": 777001,
                        "gameNumber": 2,
                        "gameType": "R",
                        "season": "2025",
                        "officialDate": "2025-05-10",
                        "gameDate": "2025-05-10T23:10:00Z",
                        "scheduledInnings": 9,
                        "status": {"abstractGameState": "Preview"},
                        "venue": {"name": "Dodger Stadium"},
                        "teams": {
                            "home": {
                                "team": {"abbreviation": "LAD"},
                                "probablePitcher": {"id": 803, "pitchHand": {"code": "L"}},
                            },
                            "away": {
                                "team": {"abbreviation": "SD"},
                                "probablePitcher": {"id": 804, "pitchHand": {"code": "R"}},
                            },
                        },
                    },
                ],
            }
        ]
    }


def _sample_mlb_boxscore_payload() -> dict[str, Any]:
    def _make_side(
        abbrev: str,
        runs: int,
        sp_id: int,
        rp_id: int,
        batter_base_id: int,
    ) -> dict[str, Any]:
        players: dict[str, Any] = {
            f"ID{sp_id}": {
                "person": {"id": sp_id, "fullName": f"Starter {sp_id}"},
                "pitchHand": {"code": "R"},
                "stats": {
                    "pitching": {
                        "inningsPitched": "6.1",
                        "battersFaced": 24,
                        "numberOfPitches": 96,
                        "strikes": 64,
                        "hits": 4,
                        "runs": 2,
                        "earnedRuns": 2,
                        "homeRuns": 1,
                        "baseOnBalls": 1,
                        "intentionalWalks": 0,
                        "hitByPitch": 0,
                        "strikeOuts": 8,
                        "groundOuts": 6,
                        "airOuts": 5,
                        "lineOuts": 0,
                    }
                },
            },
            f"ID{rp_id}": {
                "person": {"id": rp_id, "fullName": f"Reliever {rp_id}"},
                "pitchHand": {"code": "L"},
                "stats": {
                    "pitching": {
                        "inningsPitched": "2.2",
                        "battersFaced": 9,
                        "numberOfPitches": 35,
                        "strikes": 24,
                        "hits": 1,
                        "runs": 0,
                        "earnedRuns": 0,
                        "homeRuns": 0,
                        "baseOnBalls": 1,
                        "intentionalWalks": 0,
                        "hitByPitch": 0,
                        "strikeOuts": 3,
                        "groundOuts": 3,
                        "airOuts": 2,
                        "lineOuts": 0,
                    }
                },
            },
        }
        batter_ids: list[int] = []
        for slot in range(1, 10):
            bid = batter_base_id + slot
            batter_ids.append(bid)
            players[f"ID{bid}"] = {
                "person": {"id": bid, "fullName": f"Batter {bid}"},
                "batSide": {"code": "L" if slot % 2 == 0 else "R"},
                "position": {"abbreviation": "C" if slot == 9 else "OF"},
                "battingOrder": f"{slot}00",
                "stats": {
                    "batting": {
                        "plateAppearances": 4,
                        "atBats": 3,
                        "hits": 1 if slot <= 5 else 0,
                        "doubles": 1 if slot == 2 else 0,
                        "triples": 0,
                        "homeRuns": 1 if slot == 4 else 0,
                        "runs": 1 if slot <= runs else 0,
                        "rbi": 1 if slot <= runs else 0,
                        "baseOnBalls": 1,
                        "intentionalWalks": 0,
                        "hitByPitch": 0,
                        "strikeOuts": 1,
                        "sacFlies": 0,
                        "stolenBases": 1 if slot == 1 else 0,
                        "caughtStealing": 0,
                    }
                },
            }
        # Add 1 pinch hitter (slot 0, battingOrder '901')
        ph_id = batter_base_id + 99
        batter_ids.append(ph_id)
        players[f"ID{ph_id}"] = {
            "person": {"id": ph_id, "fullName": f"Pinch {ph_id}"},
            "batSide": {"code": "S"},
            "position": {"abbreviation": "PH"},
            "battingOrder": "901",
            "stats": {
                "batting": {
                    "plateAppearances": 1,
                    "atBats": 1,
                    "hits": 0,
                    "doubles": 0,
                    "triples": 0,
                    "homeRuns": 0,
                    "runs": 0,
                    "rbi": 0,
                    "baseOnBalls": 0,
                    "intentionalWalks": 0,
                    "hitByPitch": 0,
                    "strikeOuts": 1,
                    "sacFlies": 0,
                    "stolenBases": 0,
                    "caughtStealing": 0,
                }
            },
        }
        return {
            "team": {"abbreviation": abbrev},
            "pitchers": [sp_id, rp_id],
            "batters": batter_ids,
            "players": players,
            "teamStats": {
                "batting": {
                    "runs": runs,
                    "hits": 5,
                    "doubles": 1,
                    "triples": 0,
                    "homeRuns": 1,
                    "baseOnBalls": 9,
                    "intentionalWalks": 0,
                    "hitByPitch": 0,
                    "strikeOuts": 10,
                    "sacFlies": 0,
                    "atBats": 28,
                    "plateAppearances": 37,
                    "leftOnBase": 6,
                },
                "pitching": {"inningsPitched": "9.0", "outs": 27},
                "fielding": {"errors": 0},
            },
        }

    return {
        "teams": {
            "home": _make_side("LAD", 5, 801, 811, 1000),
            "away": _make_side("SD", 2, 802, 812, 2000),
        }
    }


def test_mlb_schedule_and_boxscore_parsing_with_doubleheader_and_ip_outs() -> None:
    """Verify MLB schedule doubleheaders (event_seq=1,2) and boxscore .1/.2 IP outs parsing."""
    sched_df = parse_mlb_schedule_payload(_sample_mlb_schedule_payload(), season=2025)
    assert dict(sched_df.schema) == SILVER_BASEBALL_GAMES_SCHEMA
    assert sched_df.height == 2
    assert sched_df["event_seq"].to_list() == [1, 2]
    assert sched_df["away_team"].to_list() == ["SDP", "SDP"]
    assert sched_df["game_state"].to_list() == ["FINAL", "FUT"]
    assert sched_df["home_win"].to_list() == [1, None]

    team_df, pitcher_df, batter_df = parse_mlb_boxscore_payload(
        _sample_mlb_boxscore_payload(),
        game_id=777001,
        event_seq=1,
        game_date=date(2025, 5, 10),
        start_time_utc=datetime(2025, 5, 10, 17, 10, tzinfo=UTC),
        season=2025,
    )
    assert dict(team_df.schema) == SILVER_BASEBALL_TEAM_LOGS_SCHEMA
    assert dict(pitcher_df.schema) == SILVER_BASEBALL_PITCHER_LOGS_SCHEMA
    assert dict(batter_df.schema) == SILVER_BASEBALL_BATTER_LOGS_SCHEMA

    # Starter pitched 6.1 IP -> 19 outs (Quality Start = 1); reliever pitched 2.2 IP -> 8 outs
    lad_sp = pitcher_df.filter(pl.col("player_id") == 801).row(0, named=True)
    lad_rp = pitcher_df.filter(pl.col("player_id") == 811).row(0, named=True)
    assert lad_sp["outs_pitched"] == 19
    assert pytest.approx(lad_sp["innings_pitched"], abs=1e-6) == 19.0 / 3.0
    assert lad_sp["quality_start"] == 1
    assert lad_rp["outs_pitched"] == 8
    assert lad_rp["is_starter"] is False

    # Team bullpen outs = 8, pitches = 35
    lad_team = team_df.filter(pl.col("team") == "LAD").row(0, named=True)
    assert lad_team["bullpen_outs_pitched"] == 8
    assert lad_team["bullpen_pitches"] == 35
    assert lad_team["bullpen_er"] == 0
    assert lad_team["team_woba"] > 0.250

    # Pinch hitter has batting_order_slot == 0 and is_starter == False
    ph_row = batter_df.filter(pl.col("player_id") == 1099).row(0, named=True)
    assert ph_row["batting_order_slot"] == 0
    assert ph_row["is_starter"] is False


def test_npb_12_inning_tie_provincial_venue_and_koshien_road_trip_parsing() -> None:
    """Verify NPB 12-inning ties, provincial neutral stadiums, and Koshien displacement."""
    # 1. Hanshin Tigers playing "home" at Kyocera Dome Osaka during August Koshien tournament
    vid_koshien_disp, _, _, is_dome, is_neutral = resolve_npb_venue("京セラドーム大阪", "阪神")
    assert vid_koshien_disp == "NPB_KYOCERA_DOME"
    assert is_dome is True
    assert is_neutral is True  # Displaced from primary Koshien!

    # 2. Yakult Swallows playing at Botchan Stadium Matsuyama (provincial neutral venue)
    vid_matsu, _, _, is_dome_m, is_neutral_m = resolve_npb_venue("松山坊っちゃんスタジアム", "YAK")
    assert vid_matsu == "NPB_MATSUYAMA_BOTCHAN"
    assert is_dome_m is False
    assert is_neutral_m is True

    # 3. Interleague (Koryusen) and DH rules
    sub_cl, dh_cl = resolve_npb_sub_league_and_dh("YOM", "HAN")
    assert sub_cl == "CL" and dh_cl is False
    sub_pl, dh_pl = resolve_npb_sub_league_and_dh("SOF", "ORX")
    assert sub_pl == "PL" and dh_pl is True
    sub_kor, dh_kor = resolve_npb_sub_league_and_dh("SOF", "YOM")
    assert sub_kor == "KORYUSEN" and dh_kor is True

    # 4. Schedule with a 12-inning 2-2 tie
    npb_sched = parse_npb_schedule_payload(
        {
            "games": [
                {
                    "game_id": 2025060101,
                    "game_date": "2025-06-01",
                    "start_time_utc": "2025-06-01T09:00:00Z",
                    "home_team": "阪神タイガース",
                    "away_team": "読売ジャイアンツ",
                    "venue": "甲子園",
                    "game_state": "FINAL",
                    "innings_played": 12,
                    "home_runs": 2,
                    "away_runs": 2,
                    "home_sp_id": 9001,
                    "away_sp_id": 9002,
                    "home_sp_hand": "R",
                    "away_sp_hand": "L",
                }
            ]
        },
        season=2025,
    )
    assert npb_sched.height == 1
    row = npb_sched.row(0, named=True)
    assert row["game_id"] == namespace_npb_game_id(2025060101)
    assert row["game_id"] >= NPB_GAME_ID_OFFSET
    assert row["league"] == "NPB"
    assert row["home_team"] == "HAN"
    assert row["away_team"] == "YOM"
    assert row["innings_played"] == 12
    assert row["outcome_1x2"] == 1
    assert row["is_tie"] == 1
    assert row["home_win"] == 0

    # 5. NPB boxscore parsing
    npb_box = {
        "home_team": "HAN",
        "away_team": "YOM",
        "home": {
            "runs": 2,
            "pitchers": [
                {
                    "player_id": 9001,
                    "player_name": "Hiroto Saiki",
                    "pitch_hand": "R",
                    "ip": "8.0",
                    "batters_faced": 29,
                    "pitches_thrown": 112,
                    "hits_allowed": 5,
                    "runs_allowed": 2,
                    "er_allowed": 2,
                    "bb_allowed": 1,
                    "k_recorded": 9,
                },
                {
                    "player_id": 9011,
                    "player_name": "Suguru Iwazaki",
                    "pitch_hand": "L",
                    "ip": "4.0",
                    "batters_faced": 13,
                    "pitches_thrown": 52,
                    "hits_allowed": 1,
                    "runs_allowed": 0,
                    "er_allowed": 0,
                    "bb_allowed": 1,
                    "k_recorded": 4,
                },
            ],
            "batters": [
                {
                    "player_id": 9100 + s,
                    "player_name": f"Tigers Batter {s}",
                    "bat_side": "L" if s % 2 == 0 else "R",
                    "batting_order_slot": s,
                    "position_group": "OF",
                    "pa": 5,
                    "ab": 4,
                    "hits": 1,
                    "doubles": 0,
                    "triples": 0,
                    "hr": 0,
                    "runs": 1 if s <= 2 else 0,
                    "rbi": 1 if s <= 2 else 0,
                    "bb": 1,
                    "k": 1,
                }
                for s in range(1, 10)
            ],
        },
        "away": {
            "runs": 2,
            "pitchers": [
                {
                    "player_id": 9002,
                    "player_name": "Shosei Togo",
                    "pitch_hand": "R",
                    "ip": "12.0",
                    "batters_faced": 45,
                    "pitches_thrown": 145,
                    "hits_allowed": 9,
                    "runs_allowed": 2,
                    "er_allowed": 2,
                    "bb_allowed": 3,
                    "k_recorded": 11,
                }
            ],
            "batters": [
                {
                    "player_id": 9200 + s,
                    "player_name": f"Giants Batter {s}",
                    "bat_side": "R",
                    "batting_order_slot": s,
                    "position_group": "INF",
                    "pa": 5,
                    "ab": 4,
                    "hits": 1 if s <= 6 else 0,
                    "doubles": 1 if s == 4 else 0,
                    "triples": 0,
                    "hr": 0,
                    "runs": 1 if s <= 2 else 0,
                    "rbi": 1 if s <= 2 else 0,
                    "bb": 1,
                    "k": 1,
                }
                for s in range(1, 10)
            ],
        },
    }
    npb_t, npb_p, npb_b = parse_npb_boxscore_payload(
        npb_box,
        game_id=2025060101,
        event_seq=1,
        game_date=date(2025, 6, 1),
        start_time_utc=datetime(2025, 6, 1, 9, 0, tzinfo=UTC),
        season=2025,
    )
    assert npb_t.height == 2
    assert npb_p.height == 3
    assert npb_b.height == 18
    han_row = npb_t.filter(pl.col("team") == "HAN").row(0, named=True)
    assert han_row["outs_pitched"] == 36  # 12 innings = 36 outs!
    assert han_row["bullpen_outs_pitched"] == 12


def test_two_tier_lineup_resolver_platoon_handedness_and_il_exclusions() -> None:
    """Verify Tier 1 confirmed lineup and Tier 2 platoon-projected lineups vs RHP vs LHP."""
    roster_df = pl.DataFrame(
        {
            "player_id": [
                801,
                802,
                *[100 + i for i in range(1, 15)],
                199,  # Injured star on IL10
            ],
            "player_name": [
                "SP1",
                "SP2",
                *[f"Batter {i}" for i in range(1, 15)],
                "Injured Star",
            ],
            "team": ["LAD"] * 17,
            "league": ["MLB"] * 17,
            "position_group": ["P", "P", *(["OF"] * 15)],
            "is_pitcher": [True, True, *([False] * 15)],
            "bat_side": [
                "R",
                "L",
                # Batters 1..7 are LHB, Batters 8..14 are RHB
                *(["L"] * 7),
                *(["R"] * 7),
                "L",
            ],
            "pitch_hand": ["R"] * 17,
            "status_code": ["A"] * 16 + ["IL10"],
            "is_available": [True] * 16 + [False],
        }
    )

    # 1. Tier 1: Confirmed official 1-9 lineup & SP
    confirmed = BaseballPreGameLineupResolver.resolve_team_lineup(
        "LAD",
        opposing_sp_hand="R",
        official_lineup_payload={
            "LAD": {
                "starting_pitcher_id": 801,
                "starting_pitcher_hand": "R",
                "is_confirmed_sp": True,
                "starting_batter_ids": [101, 102, 103, 104, 105, 106, 107, 108, 109],
                "bench_batter_ids": [110, 111],
            }
        },
        roster_df=roster_df,
    )
    assert confirmed.starter_status == "CONFIRMED_SP"
    assert confirmed.lineup_status == "CONFIRMED_LINEUP_1_9"
    assert confirmed.starting_batter_ids == [101, 102, 103, 104, 105, 106, 107, 108, 109]
    assert 199 in confirmed.excluded_unavailable_ids

    # 2. Tier 2: Platoon-projected lineup from historical batter logs vs RHP vs LHP
    games_hist = pl.DataFrame(
        {
            "game_id": [1, 2],
            "event_seq": [1, 1],
            "home_sp_hand": ["R", "L"],
            "away_sp_hand": ["R", "L"],
        }
    )
    # In game 1 (vs RHP), LHB batters 101..107 + RHB 108..109 started;
    # In game 2 (vs LHP), RHB batters 108..114 + LHB 101..102 started
    b_rows: list[dict[str, Any]] = []
    for idx, bid in enumerate([101, 102, 103, 104, 105, 106, 107, 108, 109], start=1):
        b_rows.append(
            {
                "game_id": 1,
                "event_seq": 1,
                "start_time_utc": datetime(2025, 5, 1, 20, 0, tzinfo=UTC),
                "team": "LAD",
                "is_home": True,
                "player_id": bid,
                "bat_side": "L" if bid <= 107 else "R",
                "batting_order_slot": idx,
                "is_starter": True,
                "pa": 4,
            }
        )
    for idx, bid in enumerate([108, 109, 110, 111, 112, 113, 114, 101, 102], start=1):
        b_rows.append(
            {
                "game_id": 2,
                "event_seq": 1,
                "start_time_utc": datetime(2025, 5, 2, 20, 0, tzinfo=UTC),
                "team": "LAD",
                "is_home": True,
                "player_id": bid,
                "bat_side": "L" if bid <= 107 else "R",
                "batting_order_slot": idx,
                "is_starter": True,
                "pa": 4,
            }
        )
    batter_logs_hist = pl.DataFrame(b_rows)

    vs_rhp = BaseballPreGameLineupResolver.resolve_team_lineup(
        "LAD",
        opposing_sp_hand="R",
        probable_sp_id=802,
        probable_sp_hand="L",
        roster_df=roster_df,
        batter_logs_df=batter_logs_hist,
        games_df=games_hist,
    )
    vs_lhp = BaseballPreGameLineupResolver.resolve_team_lineup(
        "LAD",
        opposing_sp_hand="L",
        probable_sp_id=802,
        probable_sp_hand="L",
        roster_df=roster_df,
        batter_logs_df=batter_logs_hist,
        games_df=games_hist,
    )

    assert vs_rhp.lineup_status == "PLATOON_PROJECTED_LINEUP"
    assert vs_lhp.lineup_status == "PLATOON_PROJECTED_LINEUP"
    assert vs_rhp.starter_status == "PROBABLE_SP"
    # Platoon lineups against RHP vs LHP must differ!
    assert set(vs_rhp.starting_batter_ids) != set(vs_lhp.starting_batter_ids)
    assert 199 not in vs_rhp.active_batter_ids


def test_silver_reconciler_statcast_join_fallback_and_health_probes(tmp_path: Path) -> None:
    """Verify .gz caching, Statcast join + fallback, market_* preservation, and health checks."""
    raw_dir = tmp_path / "raw"
    silver_dir = tmp_path / "silver" / "baseball"

    statcast_teams_csv = b"game_id,event_seq,team,team_xwoba,opp_xwoba\n777001,1,LAD,0.362,0.278\n"
    statcast_pitchers_csv = (
        b"game_id,event_seq,player_id,xwoba_against,barrel_pct_against,swstr_pct,avg_fastball_velo\n"
        b"777001,1,801,0.265,0.055,0.148,96.4\n"
    )
    statcast_batters_csv = (
        b"game_id,event_seq,player_id,xwoba,hard_hit_rate,barrel_rate\n"
        b"777001,1,1001,0.415,0.520,0.160\n"
    )

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "statsapi.mlb.com" in url and "/schedule" in url:
            return httpx.Response(200, json=_sample_mlb_schedule_payload())
        if "statsapi.mlb.com" in url and "/boxscore" in url:
            return httpx.Response(200, json=_sample_mlb_boxscore_payload())
        if "statsapi.mlb.com" in url and "/roster" in url:
            return httpx.Response(
                200,
                json={
                    "roster": [
                        {
                            "person": {"id": 801, "fullName": "Starter 801"},
                            "position": {"abbreviation": "P"},
                            "status": {"code": "A"},
                        }
                    ]
                },
            )
        if "baseballsavant.mlb.com" in url:
            if "type=teams" in url:
                return httpx.Response(200, content=statcast_teams_csv)
            if "type=pitchers" in url:
                return httpx.Response(200, content=statcast_pitchers_csv)
            return httpx.Response(200, content=statcast_batters_csv)
        if "npb.jp" in url:
            return httpx.Response(
                200,
                json={
                    "games": [
                        {
                            "game_id": 2025060101,
                            "game_date": "2025-06-01",
                            "home_team": "HAN",
                            "away_team": "YOM",
                            "game_state": "FINAL",
                            "innings_played": 12,
                            "home_runs": 2,
                            "away_runs": 2,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(_mock_handler)

    # 1. Run health checks across MLB, Statcast, and NPB
    health_results = check_baseball_data_sources(
        raw_dir, league="all", season=2025, transport=transport
    )
    assert len(health_results) == 3
    assert all(r.healthy for r in health_results)

    # Verify .json.gz and .csv.gz files were written to raw_dir
    gz_files = list(raw_dir.rglob("*.gz"))
    assert len(gz_files) >= 3

    # 2. Fetch via clients and reconcile into Silver Parquet
    mlb_client = MLBStatsAPIClient(raw_dir / "baseball" / "mlb_api", transport=transport)
    sc_client = StatcastClient(raw_dir / "baseball" / "statcast", transport=transport)
    npb_client = NPBClient(raw_dir / "baseball" / "npb", transport=transport)

    mlb_sched = parse_mlb_schedule_payload(
        mlb_client.fetch_schedule(date(2025, 5, 10), date(2025, 5, 10)), season=2025
    )
    mlb_box = mlb_client.fetch_boxscore(777001, is_final=True)
    t_df, p_df, b_df = parse_mlb_boxscore_payload(
        mlb_box,
        game_id=777001,
        event_seq=1,
        game_date=date(2025, 5, 10),
        start_time_utc=datetime(2025, 5, 10, 17, 10, tzinfo=UTC),
        season=2025,
    )
    roster_parsed = parse_mlb_roster_payload(mlb_client.fetch_team_roster("LAD"), team="LAD")
    assert roster_parsed.height == 1

    npb_sched = parse_npb_schedule_payload(npb_client.fetch_season_schedule(2025), season=2025)
    npb_rosters = parse_npb_roster_payload(
        [{"player_id": 9001, "player_name": "Hiroto Saiki", "team": "HAN", "position_group": "P"}]
    )
    assert npb_rosters.height == 1

    sc_teams = sc_client.fetch_season_team_metrics(2025)
    sc_pitchers = sc_client.fetch_season_pitcher_metrics(2025)
    sc_batters = sc_client.fetch_season_batter_metrics(2025)

    reconciler = BaseballSilverReconciler(silver_dir)
    # Seed market_* odds on game 777001 seq 1 and verify incremental upsert preserves them
    mlb_sched_with_odds = mlb_sched.with_columns(
        pl.when(pl.col("event_seq") == 1)
        .then(pl.lit(0.58))
        .otherwise(None)
        .alias("market_prob_home"),
        pl.when(pl.col("event_seq") == 1)
        .then(pl.lit(0.42))
        .otherwise(None)
        .alias("market_prob_away"),
    )
    reconciler.reconcile_games(mlb_sched_with_odds)
    # Re-upsert without market_* columns + NPB schedule
    combined_games = reconciler.reconcile_games(pl.concat([mlb_sched, npb_sched]))
    assert combined_games.height == 3
    g1_row = combined_games.filter((pl.col("game_id") == 777001) & (pl.col("event_seq") == 1)).row(
        0, named=True
    )
    assert pytest.approx(g1_row["market_prob_home"], abs=1e-6) == 0.58

    rec_teams = reconciler.reconcile_team_logs(t_df, statcast_team_df=sc_teams)
    rec_pitchers = reconciler.reconcile_pitcher_logs(p_df, statcast_pitcher_df=sc_pitchers)
    rec_batters = reconciler.reconcile_batter_logs(b_df, statcast_batter_df=sc_batters)

    # LAD has Statcast xwOBA joined (has_statcast_metrics=True); SDP did not -> fallback False
    lad_t = rec_teams.filter(pl.col("team") == "LAD").row(0, named=True)
    sdp_t = rec_teams.filter(pl.col("team") == "SDP").row(0, named=True)
    assert lad_t["has_statcast_metrics"] is True
    assert pytest.approx(lad_t["team_xwoba"], abs=1e-6) == 0.362
    assert sdp_t["has_statcast_metrics"] is False

    sp_801 = rec_pitchers.filter(pl.col("player_id") == 801).row(0, named=True)
    assert pytest.approx(sp_801["avg_fastball_velo"], abs=1e-6) == 96.4

    bat_1001 = rec_batters.filter(pl.col("player_id") == 1001).row(0, named=True)
    assert pytest.approx(bat_1001["xwoba"], abs=1e-6) == 0.415
