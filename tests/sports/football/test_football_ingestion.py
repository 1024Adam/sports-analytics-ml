"""Offline tests for nflverse ingestion, Silver reconciliation, and the 2-tier QB resolver."""

from __future__ import annotations

import gzip
import io
import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.core.market import is_forbidden_market_column
from sports_ml.sports.football import FootballAdapter
from sports_ml.sports.football.constants import (
    SILVER_FOOTBALL_AVAILABILITY_SCHEMA,
    SILVER_FOOTBALL_GAMES_SCHEMA,
    canonical_game_id_from_nflverse,
    make_football_game_id,
    parse_nflverse_game_id,
)
from sports_ml.sports.football.ingestion import (
    NFLVERSE_REQUIRED_COLUMNS,
    FootballSilverReconciler,
    FootballStartingQBResolver,
    NflverseAssetUnavailableError,
    NflverseClient,
    NflverseIngestor,
    assert_games_quarantined,
    build_availability_snapshots,
    build_games_frame,
    check_football_data_sources,
    derive_active_nfl_season,
    derive_starting_qbs_from_pbp,
)

QB_KC = "00-0033873"
QB_KC_BACKUP = "00-0099001"
QB_DET = "00-0033106"
QB_WAS = "00-0037077"
QB_WAS_BACKUP = "00-0033999"


def _int(gsis: str) -> int:
    return int(gsis.split("-")[1])


def _game(
    game_id: str,
    game_type: str,
    gameday: str,
    gametime: str | None,
    home_score: int | None,
    away_score: int | None,
    *,
    location: str = "Home",
    stadium_id: str = "KAN00",
    overtime: int | None = 0,
    home_ml: float | None = None,
    away_ml: float | None = None,
    spread: float | None = None,
    total: float | None = None,
    home_qb: str | None = None,
    away_qb: str | None = None,
) -> dict[str, object]:
    season, week, away, home = game_id.split("_")
    return {
        "game_id": game_id,
        "season": int(season),
        "game_type": game_type,
        "week": int(week),
        "gameday": gameday,
        "weekday": "Sunday",
        "gametime": gametime,
        "away_team": away,
        "away_score": away_score,
        "home_team": home,
        "home_score": home_score,
        "location": location,
        "result": None,
        "total": None,
        "overtime": overtime,
        "old_game_id": None,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "spread_line": spread,
        "total_line": total,
        "roof": "outdoors",
        "surface": "grass",
        "temp": 71.0,
        "wind": 6.0,
        "away_qb_id": away_qb,
        "home_qb_id": home_qb,
        "stadium_id": stadium_id,
        "stadium": "fixture",
    }


def _schedule() -> pl.DataFrame:
    rows = [
        _game(
            "2023_01_DET_KC",
            "REG",
            "2023-09-07",
            "20:20",
            20,
            21,
            home_ml=-190,
            away_ml=160,
            spread=4.0,
            total=53.0,
            home_qb="00-0000001",
            away_qb=QB_DET,
        ),
        _game("2023_01_ARI_WAS", "REG", "2023-09-10", "13:00", 20, 16, stadium_id="WAS00"),
        _game(
            "2023_05_TEN_CLE", "REG", "2023-10-08", "13:00", 20, 20, stadium_id="CLE00", overtime=1
        ),
        _game(
            "2023_05_JAX_ATL",
            "REG",
            "2023-10-08",
            "09:30",
            23,
            7,
            location="Neutral",
            stadium_id="LON00",
        ),
        _game("2023_19_MIA_KC", "WC", "2024-01-13", "16:30", 26, 7),
        _game("2023_20_GB_SF", "DIV", "2024-01-20", "15:00", 24, 21, stadium_id="SFO01"),
        _game("2023_21_DET_SF", "CON", "2024-01-28", "15:00", 34, 31, stadium_id="SFO01"),
        _game(
            "2023_22_SF_KC",
            "SB",
            "2024-02-11",
            "18:30",
            25,
            22,
            location="Neutral",
            stadium_id="VEG00",
            overtime=1,
        ),
        _game("2023_00_BUF_NYJ", "PRE", "2023-08-12", "19:00", 10, 3),
        _game("2019_01_DEN_OAK", "REG", "2019-09-09", "22:00", 24, 16, stadium_id="OAK00"),
        _game("2015_02_WAS_STL", "REG", "2015-09-20", None, 10, 24, stadium_id="STL00"),
        _game(
            "2026_09_BUF_MIA",
            "REG",
            "2026-11-08",
            "13:00",
            None,
            None,
            stadium_id="MIA00",
            overtime=None,
        ),
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def _play(
    game_id: str,
    team: str,
    play_id: int,
    play_type: str,
    *,
    epa: float | None = None,
    success: int | None = None,
    yards: float | None = None,
    pass_: int = 0,
    rush: int = 0,
    dropback: int = 0,
    qb: str | None = None,
    sack: int = 0,
    interception: int = 0,
    penalty: int = 0,
    penalty_team: str | None = None,
    penalty_yards: float | None = None,
    drive: int | None = 1,
    top: str | None = "2:30",
    cpoe: float | None = None,
    air_yards: float | None = None,
    pass_attempt: int = 0,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "play_id": play_id,
        "posteam": team,
        "defteam": None,
        "play_type": play_type,
        "epa": epa,
        "success": success,
        "yards_gained": yards,
        "pass": pass_,
        "rush": rush,
        "qb_dropback": dropback,
        "id": qb,
        "sack": sack,
        "interception": interception,
        "fumble_lost": 0,
        "penalty": penalty,
        "penalty_team": penalty_team,
        "penalty_yards": penalty_yards,
        "drive": drive,
        "drive_time_of_possession": top,
        "cpoe": cpoe,
        "air_yards": air_yards,
        "qb_scramble": 0,
        "pass_attempt": pass_attempt,
    }


def _pbp() -> pl.DataFrame:
    g = "2023_01_DET_KC"
    rows = [
        _play(
            g,
            "KC",
            1,
            "pass",
            epa=1.0,
            success=1,
            yards=10,
            pass_=1,
            dropback=1,
            qb=QB_KC,
            cpoe=5.0,
            air_yards=8,
            pass_attempt=1,
        ),
        _play(g, "KC", 2, "run", epa=-0.2, success=0, yards=2, rush=1),
        _play(
            g, "KC", 3, "pass", epa=-2.0, success=0, yards=-7, pass_=1, dropback=1, qb=QB_KC, sack=1
        ),
        _play(
            g,
            "KC",
            4,
            "pass",
            epa=0.5,
            success=1,
            yards=6,
            pass_=1,
            dropback=1,
            qb=QB_KC_BACKUP,
            drive=2,
            top="1:30",
            pass_attempt=1,
        ),
        _play(g, "KC", 5, "punt", drive=2, top="1:30"),
        _play(
            g,
            "KC",
            6,
            "no_play",
            penalty=1,
            penalty_team="KC",
            penalty_yards=10,
            drive=2,
            top="1:30",
        ),
        _play(
            g,
            "DET",
            7,
            "pass",
            epa=0.3,
            success=1,
            yards=12,
            pass_=1,
            dropback=1,
            qb=QB_DET,
            drive=3,
            top="3:00",
            pass_attempt=1,
        ),
        _play(g, "DET", 8, "run", epa=-0.1, success=0, yards=1, rush=1, drive=3, top="3:00"),
        _play(
            g,
            "DET",
            9,
            "pass",
            epa=-4.0,
            success=0,
            yards=0,
            pass_=1,
            dropback=1,
            qb=QB_DET,
            interception=1,
            drive=3,
            top="3:00",
            pass_attempt=1,
        ),
    ]
    g2 = "2023_01_ARI_WAS"
    rows += [
        _play(
            g2,
            "WAS",
            1,
            "pass",
            epa=0.2,
            success=1,
            yards=5,
            pass_=1,
            dropback=1,
            qb=QB_WAS,
            pass_attempt=1,
        ),
        _play(g2, "ARI", 2, "run", epa=0.1, success=1, yards=4, rush=1, drive=2, top="1:00"),
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def _player_stats() -> pl.DataFrame:
    base = {
        "player_display_name": "x",
        "position": "QB",
        "attempts": 30,
        "passing_yards": 250.0,
        "passing_tds": 2,
        "interceptions": 1,
        "carries": 2,
        "rushing_yards": 5.0,
        "rushing_tds": 0,
        "receptions": 0,
        "targets": 0,
        "receiving_yards": 0.0,
        "receiving_tds": 0,
    }
    rows = [
        {**base, "player_id": QB_WAS, "recent_team": "WAS", "season": 2023, "week": 1},
        {**base, "player_id": "00-0030001", "recent_team": "OAK", "season": 2019, "week": 1},
    ]
    return pl.DataFrame(rows)


def _snap_counts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2023_01_ARI_WAS"],
            "season": [2023],
            "week": [1],
            "team": ["WAS"],
            "pfr_player_id": ["PFR_WAS_QB"],
            "offense_snaps": [60.0],
            "offense_pct": [0.97],
        }
    )


def _players() -> pl.DataFrame:
    return pl.DataFrame(
        {"gsis_id": [QB_WAS], "pfr_id": ["PFR_WAS_QB"], "display_name": ["x"], "position": ["QB"]}
    )


def _was_availability_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    injuries = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team": ["WAS", "WAS"],
            "gsis_id": [QB_WAS, QB_WAS],
            "position": ["QB", "QB"],
            "report_status": [None, "Questionable"],
            "practice_status": ["Limited Participation in Practice", None],
            "date_modified": [
                datetime(2023, 9, 7, 20, 0, tzinfo=UTC),
                datetime(2023, 9, 8, 19, 0, tzinfo=UTC),
            ],
        },
        schema_overrides={"report_status": pl.Utf8, "practice_status": pl.Utf8},
    )
    depth = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "club_code": ["WAS", "WAS"],
            "gsis_id": [QB_WAS, QB_WAS_BACKUP],
            "depth_team": ["1", "2"],
            "position": ["QB", "QB"],
            "formation": ["Offense", "Offense"],
        }
    )
    weekly = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "team": ["WAS"],
            "gsis_id": [QB_WAS],
            "position": ["QB"],
            "status": ["INA"],
        }
    )
    return injuries, depth, weekly


@pytest.fixture
def games_silver() -> pl.DataFrame:
    return build_games_frame(_schedule())


# ----------------------------------------------------------------------------------------
# Parsing & reconciliation
# ----------------------------------------------------------------------------------------


def test_game_id_is_deterministic_and_alias_stable() -> None:
    assert parse_nflverse_game_id("2023_01_DET_KC") == (2023, 1, "DET", "KC")
    assert canonical_game_id_from_nflverse("2019_01_DEN_OAK") == make_football_game_id(
        2019, 1, "DEN", "LV"
    )
    with pytest.raises(ValueError, match="Invalid nflverse game id"):
        parse_nflverse_game_id("bad")


def test_schedule_parsing_ties_overtime_playoffs_neutral_and_aliases(
    games_silver: pl.DataFrame,
) -> None:
    games = games_silver
    assert dict(games.schema) == SILVER_FOOTBALL_GAMES_SCHEMA
    assert games["game_id"].n_unique() == games.height
    assert games.filter(pl.col("season_type") == "PRE").is_empty()
    assert set(games["season_type"].unique()) == {"REG", "WC", "DIV", "CON", "SB"}
    FootballAdapter().validate_event_schema(games)

    kc = games.filter(pl.col("nflverse_game_id") == "2023_01_DET_KC").row(0, named=True)
    assert kc["start_time_utc"] == datetime(2023, 9, 8, 0, 20, tzinfo=UTC)
    assert kc["home_win"] == 0
    assert kc["is_tie"] == 0

    tie = games.filter(pl.col("nflverse_game_id") == "2023_05_TEN_CLE").row(0, named=True)
    assert tie["is_tie"] == 1
    assert tie["home_win"] is None
    assert tie["overtime"] is True

    london = games.filter(pl.col("nflverse_game_id") == "2023_05_JAX_ATL").row(0, named=True)
    assert london["is_neutral_site"]
    assert london["is_international"]
    assert london["venue_id"] == "LON_WEMBLEY"
    assert london["start_time_utc"] == datetime(2023, 10, 8, 13, 30, tzinfo=UTC)

    sb = games.filter(pl.col("season_type") == "SB").row(0, named=True)
    assert sb["is_neutral_site"]
    assert not sb["is_international"]
    assert sb["venue_id"] == "LV_ALLEGIANT"

    oak = games.filter(pl.col("season") == 2019).row(0, named=True)
    assert oak["home_team"] == "LV"
    assert oak["venue_id"] == "OAK_COLISEUM"
    stl = games.filter(pl.col("season") == 2015).row(0, named=True)
    assert stl["home_team"] == "LA"
    assert stl["start_time_utc"] == datetime(2015, 9, 20, 17, 0, tzinfo=UTC)  # default 1 PM ET

    fut = games.filter(pl.col("game_state") == "FUT")
    assert fut.height == 1
    assert fut["home_win"][0] is None
    assert fut["is_tie"][0] is None


def test_quarantine_routes_odds_to_market_and_weather_to_obs(games_silver: pl.DataFrame) -> None:
    assert_games_quarantined(games_silver)
    banned = {"spread_line", "total_line", "home_moneyline", "away_moneyline", "temp", "wind"}
    assert banned.isdisjoint(games_silver.columns)
    assert not {"roof", "surface"}.intersection(games_silver.columns)

    for col in games_silver.columns:
        if "odds" in col or "spread" in col or col.startswith("market_"):
            assert col.startswith("market_"), col
        if "weather" in col:
            assert col.startswith("obs_weather_"), col
    market_cols = [c for c in games_silver.columns if c.startswith("market_")]
    assert market_cols
    assert all(is_forbidden_market_column(c) for c in market_cols)

    kc = games_silver.filter(pl.col("nflverse_game_id") == "2023_01_DET_KC").row(0, named=True)
    assert kc["market_home_decimal_odds"] == pytest.approx(1.0 + 100.0 / 190.0)
    assert kc["market_prob_home"] + kc["market_prob_away"] == pytest.approx(1.0)
    assert kc["market_prob_home"] > 0.5
    assert kc["market_spread_home"] == pytest.approx(-4.0)
    assert kc["market_total_points"] == pytest.approx(53.0)
    assert kc["obs_weather_temp"] == pytest.approx(71.0)
    assert kc["obs_weather_roof"] == "outdoors"

    with pytest.raises(ValueError, match="Quarantined"):
        assert_games_quarantined(games_silver.with_columns(pl.lit(1.0).alias("spread_line")))


def test_team_qb_logs_from_pbp_and_starter_precedence(tmp_path: Path) -> None:
    rec = FootballSilverReconciler(tmp_path / "silver")
    schedule = _schedule()
    pbp = _pbp()

    starters = derive_starting_qbs_from_pbp(pbp)
    assert dict(zip(starters["team"], starters["gsis_id"], strict=True))["KC"] == QB_KC

    games_no_pbp = rec.reconcile_games(schedule, persist=False)
    assert games_no_pbp.filter(pl.col("nflverse_game_id") == "2023_01_DET_KC")["home_qb_id"][0] == 1

    qb_map = {
        (canonical_game_id_from_nflverse("2023_01_DET_KC"), "KC"): _int(QB_KC),
    }
    games = rec.reconcile_games(schedule, starting_qbs=qb_map)
    assert games.filter(pl.col("nflverse_game_id") == "2023_01_DET_KC")["home_qb_id"][0] == _int(
        QB_KC
    )

    team = rec.reconcile_team_game_logs(pbp)
    kc = team.filter(pl.col("team") == "KC").row(0, named=True)
    assert kc["off_plays"] == 4
    assert kc["off_dropbacks"] == 3
    assert kc["off_rushes"] == 1
    assert kc["off_epa_total"] == pytest.approx(-0.7)
    assert kc["off_pass_epa_total"] == pytest.approx(-0.5)
    assert kc["off_rush_epa_total"] == pytest.approx(-0.2)
    assert kc["off_success_plays"] == 2
    assert kc["off_yards"] == pytest.approx(11.0)
    assert kc["off_sacks_taken"] == 1
    assert kc["penalties"] == 1
    assert kc["penalty_yards"] == 10
    assert kc["time_of_possession_s"] == 240
    assert kc["points_for"] == 20
    assert kc["def_epa_total"] == pytest.approx(-3.8)
    assert kc["def_takeaways"] == 1
    det = team.filter(pl.col("team") == "DET").row(0, named=True)
    assert det["off_turnovers"] == 1
    assert det["def_epa_total"] == pytest.approx(kc["off_epa_total"])

    qb = rec.reconcile_qb_game_logs(pbp)
    starter = qb.filter((pl.col("team") == "KC") & pl.col("is_starter")).row(0, named=True)
    assert starter["qb_id"] == _int(QB_KC)
    assert starter["dropbacks"] == 2
    assert starter["sacks"] == 1
    assert starter["cpoe_total"] == pytest.approx(5.0)
    assert starter["cpoe_plays"] == 1
    backup = qb.filter(pl.col("qb_id") == _int(QB_KC_BACKUP)).row(0, named=True)
    assert not backup["is_starter"]


def test_player_logs_gsis_crosswalk_and_alias_join(tmp_path: Path) -> None:
    rec = FootballSilverReconciler(tmp_path / "silver")
    rec.reconcile_games(_schedule())
    logs = rec.reconcile_player_game_logs(
        _player_stats(), snap_counts_df=_snap_counts(), players_df=_players()
    )
    was = logs.filter(pl.col("player_id") == _int(QB_WAS)).row(0, named=True)
    assert was["team"] == "WAS"
    assert was["snaps"] == 60
    assert was["snap_share"] == pytest.approx(0.97)
    assert was["is_starter"] is True
    assert was["pass_attempts"] == 30

    oak = logs.filter(pl.col("player_id") == 30001).row(0, named=True)
    assert oak["team"] == "LV"
    assert oak["snap_share"] is None
    assert oak["game_id"] == make_football_game_id(2019, 1, "DEN", "LV")

    # No snap/crosswalk inputs: snap columns degrade to null rather than failing
    bare = FootballSilverReconciler(tmp_path / "bare")
    bare.reconcile_games(_schedule())
    assert bare.reconcile_player_game_logs(_player_stats())["snap_share"].null_count() == 2


def test_silver_uniqueness_and_idempotent_correction(tmp_path: Path) -> None:
    rec = FootballSilverReconciler(tmp_path / "silver")
    pbp = _pbp()
    injuries, depth, weekly = _was_availability_inputs()

    def run(schedule: pl.DataFrame) -> dict[str, pl.DataFrame]:
        return {
            "games": rec.reconcile_games(schedule),
            "team": rec.reconcile_team_game_logs(pbp),
            "qb": rec.reconcile_qb_game_logs(pbp),
            "player": rec.reconcile_player_game_logs(
                _player_stats(), snap_counts_df=_snap_counts(), players_df=_players()
            ),
            "avail": rec.reconcile_availability(
                injuries_df=injuries, depth_charts_df=depth, weekly_rosters_df=weekly
            ),
        }

    first = run(_schedule())
    keys = {
        "games": ["game_id"],
        "team": ["game_id", "team"],
        "qb": ["game_id", "team", "qb_id"],
        "player": ["game_id", "player_id"],
        "avail": ["game_id", "team", "player_id", "source", "as_of_time_utc"],
    }
    for name, frame in first.items():
        assert frame.height > 0
        assert frame.select(keys[name]).n_unique() == frame.height, name

    corrected = _schedule().with_columns(
        pl.when(pl.col("game_id") == "2023_01_ARI_WAS")
        .then(27)
        .otherwise(pl.col("home_score"))
        .alias("home_score")
    )
    second = run(corrected)
    for name in first:
        assert second[name].height == first[name].height, name
    was = second["games"].filter(pl.col("nflverse_game_id") == "2023_01_ARI_WAS").row(0, named=True)
    assert was["home_score"] == 27

    third = run(corrected)
    for name in second:
        assert_frame_equal(second[name], third[name])
    assert rec.games_path.exists()
    assert pl.read_parquet(rec.games_path).height == second["games"].height


# ----------------------------------------------------------------------------------------
# As-of resolver
# ----------------------------------------------------------------------------------------


@pytest.fixture
def was_resolver_inputs(games_silver: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    injuries, depth, weekly = _was_availability_inputs()
    availability = build_availability_snapshots(
        games_silver, injuries_df=injuries, depth_charts_df=depth, weekly_rosters_df=weekly
    )
    game_id = make_football_game_id(2023, 1, "ARI", "WAS")
    return availability, games_silver, game_id


def test_availability_snapshots_are_stamped_conservatively(
    was_resolver_inputs: tuple[pl.DataFrame, pl.DataFrame, int],
) -> None:
    availability, games, game_id = was_resolver_inputs
    assert dict(availability.schema) == SILVER_FOOTBALL_AVAILABILITY_SCHEMA
    kickoff = games.filter(pl.col("game_id") == game_id)["start_time_utc"][0]
    ina = availability.filter(pl.col("source") == "inactives")
    assert ina["as_of_time_utc"][0] == kickoff - timedelta(minutes=90)
    assert ina["is_inactive"][0]
    depth = availability.filter(pl.col("source") == "depth_chart")
    assert (depth["as_of_time_utc"] < kickoff).all()
    assert sorted(depth["depth_rank"].to_list()) == [1, 2]


def test_tier2_friday_call_never_reads_sunday_inactives(
    was_resolver_inputs: tuple[pl.DataFrame, pl.DataFrame, int],
) -> None:
    availability, games, game_id = was_resolver_inputs
    resolver = FootballStartingQBResolver(availability, games)
    friday = datetime(2023, 9, 8, 21, 0, tzinfo=UTC)

    tier2 = resolver.resolve_tier2(game_id, "WAS", friday)
    assert tier2.tier == "tier2"
    assert tier2.starter_id == _int(QB_WAS)
    probs = {c.qb_id: c.p_starts for c in tier2.candidates}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert 0.5 < probs[_int(QB_WAS)] < 0.7  # Questionable, not ruled out
    assert probs[_int(QB_WAS_BACKUP)] > 0.3
    assert tier2.source == "depth_chart+injury_report"

    # A call stamped after the Tier 1 cutoff is clamped and still ignores `inactives`
    late = resolver.resolve_tier2(game_id, "WAS", datetime(2023, 9, 10, 17, 30, tzinfo=UTC))
    assert late.as_of_time_utc == resolver.tier1_cutoff(game_id)
    assert late.starter_id == _int(QB_WAS)

    # Tier 1 at the official cutoff sees the inactives list: QB1 is out
    tier1 = resolver.resolve_tier1(game_id, "WAS")
    assert tier1.tier == "tier1"
    assert tier1.starter_id == _int(QB_WAS_BACKUP)
    assert tier1.source == "depth_chart+inactives"


def test_tier1_never_reads_data_after_kickoff_minus_90(
    was_resolver_inputs: tuple[pl.DataFrame, pl.DataFrame, int],
) -> None:
    availability, games, game_id = was_resolver_inputs
    resolver0 = FootballStartingQBResolver(availability, games)
    cutoff = resolver0.tier1_cutoff(game_id)

    leaky = pl.concat(
        [
            availability,
            pl.DataFrame(
                {
                    "as_of_time_utc": [cutoff + timedelta(minutes=30)],
                    "game_id": [game_id],
                    "season": [2023],
                    "week": [1],
                    "team": ["WAS"],
                    "player_id": [_int(QB_WAS_BACKUP)],
                    "position": ["QB"],
                    "source": ["inactives"],
                    "report_status": [None],
                    "practice_status": [None],
                    "depth_rank": [None],
                    "is_inactive": [True],
                },
                schema=SILVER_FOOTBALL_AVAILABILITY_SCHEMA,
            ),
        ]
    )
    confirmed_after = pl.DataFrame(
        {
            "game_id": [game_id],
            "team": ["WAS"],
            "qb_id": [12345],
            "as_of_time_utc": [cutoff + timedelta(minutes=1)],
        }
    ).with_columns(pl.col("as_of_time_utc").dt.replace_time_zone("UTC"))
    resolver = FootballStartingQBResolver(leaky, games, confirmed_starters=confirmed_after)

    result = resolver.resolve_tier1(game_id, "WAS", as_of_time_utc=cutoff + timedelta(hours=1))
    assert result.as_of_time_utc == cutoff
    assert result.starter_id == _int(QB_WAS_BACKUP)
    assert result.source == "depth_chart+inactives"

    on_time = confirmed_after.with_columns(
        (pl.col("as_of_time_utc") - timedelta(minutes=2)).alias("as_of_time_utc")
    )
    confirmed = FootballStartingQBResolver(leaky, games, confirmed_starters=on_time).resolve_tier1(
        game_id, "WAS"
    )
    assert confirmed.source == "confirmed_starter"
    assert confirmed.starter_id == 12345


def test_resolver_future_mutation_invariance(
    was_resolver_inputs: tuple[pl.DataFrame, pl.DataFrame, int],
) -> None:
    availability, games, game_id = was_resolver_inputs
    friday = datetime(2023, 9, 8, 21, 0, tzinfo=UTC)
    baseline = FootballStartingQBResolver(availability, games).resolve_tier2(game_id, "WAS", friday)

    mutated = availability.with_columns(
        pl.when(pl.col("as_of_time_utc") > friday)
        .then(pl.lit("Out"))
        .otherwise(pl.col("report_status"))
        .alias("report_status"),
        pl.when(pl.col("as_of_time_utc") > friday)
        .then(pl.lit(True))
        .otherwise(pl.col("is_inactive"))
        .alias("is_inactive"),
    )
    games_mutated = games.with_columns(
        (pl.col("home_score") + 30).alias("home_score"), pl.lit(1).alias("home_win")
    )
    after = FootballStartingQBResolver(mutated, games_mutated).resolve_tier2(game_id, "WAS", friday)
    assert after == baseline


# ----------------------------------------------------------------------------------------
# Client: cache, retry, availability windows, health, ingestor
# ----------------------------------------------------------------------------------------


def _parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def _csv_bytes(df: pl.DataFrame) -> bytes:
    return df.write_csv().encode("utf-8")


def _routes() -> dict[str, bytes]:
    base = "https://github.com/nflverse/nflverse-data/releases/download"
    injuries, depth, weekly = _was_availability_inputs()
    return {
        "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv": _csv_bytes(
            _schedule()
        ),
        f"{base}/pbp/play_by_play_2023.parquet": _parquet_bytes(_pbp()),
        f"{base}/player_stats/player_stats_2023.parquet": _parquet_bytes(_player_stats()),
        f"{base}/injuries/injuries_2023.parquet": _parquet_bytes(injuries),
        f"{base}/depth_charts/depth_charts_2023.parquet": _parquet_bytes(depth),
        f"{base}/weekly_rosters/roster_weekly_2023.parquet": _parquet_bytes(weekly),
        f"{base}/players/players.parquet": _parquet_bytes(_players()),
    }


def _transport(routes: dict[str, bytes], hits: list[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if hits is not None:
            hits.append(url)
        if url in routes:
            return httpx.Response(200, content=routes[url])
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_client_caches_schedule_gzip_and_parquet_assets(tmp_path: Path) -> None:
    hits: list[str] = []
    client = NflverseClient(tmp_path / "raw", transport=_transport(_routes(), hits))

    schedule = client.fetch_schedule()
    assert schedule.height == _schedule().height
    cache = tmp_path / "raw" / "schedules.csv.gz"
    assert cache.exists()
    assert cache.read_bytes()[:2] == b"\x1f\x8b"
    assert b"2023_01_DET_KC" in gzip.decompress(cache.read_bytes())

    pbp = client.fetch_asset("pbp", 2023)
    assert pbp.height == _pbp().height
    assert (tmp_path / "raw" / "pbp_2023.parquet").exists()

    n_hits = len(hits)
    client.fetch_schedule()
    client.fetch_asset("pbp", 2023)
    assert len(hits) == n_hits  # served from cache


def test_client_retries_then_succeeds_and_handles_404_and_windows(tmp_path: Path) -> None:
    calls = {"n": 0}
    payload = _parquet_bytes(_players())

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if "players.parquet" in str(request.url):
            return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, content=payload)
        return httpx.Response(404)

    client = NflverseClient(
        tmp_path / "raw",
        transport=httpx.MockTransport(flaky),
        retry_attempts=3,
        retry_min_wait=0.01,
    )
    assert client.fetch_asset("players").height == 1
    assert calls["n"] == 3

    before = calls["n"]
    with pytest.raises(NflverseAssetUnavailableError):
        client.fetch_asset("pbp", 2023)
    assert client.try_fetch_asset("pbp", 2023) is None
    assert calls["n"] == before + 2  # a 404 is never retried

    before = calls["n"]
    assert client.try_fetch_asset("injuries", 2005) is None  # injuries start ~2009
    assert client.try_fetch_asset("snap_counts", 2011) is None  # snap counts start 2012
    assert client.fetch_nextgen_stats("passing", 2015) is None  # NGS start 2016
    assert calls["n"] == before


def test_client_serves_stale_cache_when_refresh_fails(tmp_path: Path) -> None:
    routes = _routes()
    client = NflverseClient(tmp_path / "raw", transport=_transport(routes))
    client.fetch_asset("pbp", 2023, is_active_season=True)
    cached = tmp_path / "raw" / "pbp_2023.parquet"
    old = time.time() - 7 * 86400
    os.utime(cached, (old, old))

    down = NflverseClient(
        tmp_path / "raw",
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        retry_attempts=1,
    )
    assert down.fetch_asset("pbp", 2023, is_active_season=True).height == _pbp().height
    with pytest.raises(httpx.HTTPStatusError):
        down.fetch_asset("player_stats", 2023, is_active_season=True)


def test_health_probes_validate_schema_contracts(tmp_path: Path) -> None:
    client = NflverseClient(tmp_path / "probe")
    routes: dict[str, bytes] = {}
    for asset in NFLVERSE_REQUIRED_COLUMNS:
        cols = {
            (req if isinstance(req, str) else req[0]): [1]
            for req in NFLVERSE_REQUIRED_COLUMNS[asset]
        }
        spec_url = client.asset_url(asset, 2023)
        routes[spec_url] = (
            _csv_bytes(pl.DataFrame(cols))
            if asset == "schedules"
            else _parquet_bytes(pl.DataFrame(cols))
        )

    results = check_football_data_sources(
        tmp_path / "raw_ok", season=2023, transport=_transport(routes)
    )
    assert len(results) == len(NFLVERSE_REQUIRED_COLUMNS)
    assert all(r.healthy for r in results), [r.details for r in results if not r.healthy]

    broken = dict(routes)
    inj_url = client.asset_url("injuries", 2023)
    broken[inj_url] = _parquet_bytes(pl.DataFrame({"season": [1]}))
    del broken[client.asset_url("snap_counts", 2023)]
    bad = {
        r.source: r
        for r in check_football_data_sources(
            tmp_path / "raw_bad", season=2023, transport=_transport(broken)
        )
        if not r.healthy
    }
    assert "Schema contract violated" in bad["nflverse Injury Reports"].details
    assert "404" in bad["nflverse Snap Counts"].details

    with pytest.raises(ValueError, match="Only 'nfl'"):
        check_football_data_sources(tmp_path, league="ncaaf")  # type: ignore[arg-type]


def test_ingestor_end_to_end_window_flags_and_idempotency(tmp_path: Path) -> None:
    client = NflverseClient(tmp_path / "raw", transport=_transport(_routes()))
    silver = FootballSilverReconciler(tmp_path / "silver")
    ingestor = NflverseIngestor(client, silver, active_season=2026)

    with pytest.raises(ValueError, match="Specify"):
        ingestor.resolve_seasons()
    assert ingestor.resolve_seasons(from_year=2024) == [2024, 2025, 2026]
    assert ingestor.resolve_seasons(start_date=date(2024, 1, 20), end_date=date(2024, 9, 8)) == [
        2023,
        2024,
    ]
    assert derive_active_nfl_season(date(2026, 2, 8)) == 2025

    report = ingestor.ingest(from_year=2023, to_year=2023)
    assert report.seasons == (2023,)
    assert report.has_snap_counts == {2023: False}  # snap counts not published in fixture
    assert report.has_injury_report == {2023: True}
    assert report.has_ngs == {2023: False}
    assert report.team_log_rows == 4
    assert report.qb_log_rows > 0
    assert report.availability_rows > 0
    games = silver.read_games()
    kc = games.filter(pl.col("nflverse_game_id") == "2023_01_DET_KC").row(0, named=True)
    assert kc["home_qb_id"] == _int(QB_KC)  # pbp first dropback beats the schedule value

    again = ingestor.ingest(from_year=2023, to_year=2023)
    assert again.games_rows == report.games_rows
    assert again.availability_rows == report.availability_rows

    window_silver = FootballSilverReconciler(tmp_path / "window")
    window = NflverseIngestor(client, window_silver, active_season=2026).ingest(
        start_date=date(2023, 9, 10), end_date=date(2023, 9, 10)
    )
    assert window.games_rows == 1
    assert window_silver.read_games()["nflverse_game_id"].to_list() == ["2023_01_ARI_WAS"]


def test_cli_football_check_sources_and_ingest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from sports_ml import cli
    from sports_ml.sports.hockey.ingestion.health import DataSourceHealthResult

    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))
    runner = CliRunner()

    def fake_health(raw_dir: Path) -> list[DataSourceHealthResult]:
        return [
            DataSourceHealthResult(
                source="nflverse Schedules",
                endpoint="u",
                healthy=True,
                latency_ms=1.0,
                details="OK",
            )
        ]

    monkeypatch.setattr(cli, "check_football_data_sources", fake_health)
    ok = runner.invoke(cli.app, ["check-sources", "--sport", "football", "--league", "nfl"])
    assert ok.exit_code == 0
    assert "FOOTBALL" in ok.output

    bad_league = runner.invoke(cli.app, ["check-sources", "--sport", "football", "--league", "cfb"])
    assert bad_league.exit_code != 0

    no_window = runner.invoke(cli.app, ["ingest", "--sport", "football"])
    assert no_window.exit_code != 0
    offline = runner.invoke(cli.app, ["ingest", "--sport", "football", "--offline"])
    assert offline.exit_code != 0

    class FakeReport:
        seasons = (2023,)
        games_rows = 12
        team_log_rows = 4
        player_log_rows = 2
        availability_rows = 9
        missing_assets = ("snap_counts",)

    monkeypatch.setattr(cli, "_run_football_ingestion", lambda *a, **k: FakeReport())
    done = runner.invoke(cli.app, ["ingest", "--sport", "football", "--from-year", "2023"])
    assert done.exit_code == 0
    assert "silver_games=12" in done.output.replace("\n", "")
