"""Unit and anti-leakage tests for Baseball constants, adapter, IP/outs math, and Elo hooks."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from sports_ml.core.contracts import (
    SchemaValidationError,
    SportAdapter,
    StarterModifierHook,
    VenueModifierHook,
)
from sports_ml.core.ratings import PairwiseEloEngine
from sports_ml.core.temporal import pregame_exposure_rate
from sports_ml.sports.baseball import (
    ALL_BASEBALL_CLUBS,
    BASEBALL_BALLPARKS,
    MLB_CLUBS_30,
    NPB_CLUBS_12,
    BaseballAdapter,
    get_ballpark_metadata,
    get_baseball_club_metadata,
    normalize_baseball_team,
    outs_to_true_innings,
    parse_baseball_ip_to_outs,
    resolve_team_league,
)
from sports_ml.sports.baseball.constants import (
    BATTING_ORDER_SLOT_PA_WEIGHTS,
    MLB_ELO_CONFIG,
    NPB_ELO_CONFIG,
    get_league_sabermetric_priors,
)
from sports_ml.sports.baseball.features import (
    BaseballStarterEloHook,
    BaseballVenueEloHook,
)


def test_mlb_30_and_npb_12_club_registries_and_subleagues() -> None:
    """Verify all 30 MLB and 12 NPB clubs exist with valid sub-leagues and ballparks."""
    assert len(MLB_CLUBS_30) == 30
    assert len(NPB_CLUBS_12) == 12
    assert len(ALL_BASEBALL_CLUBS) == 42

    al_clubs = [c for c in MLB_CLUBS_30.values() if c.sub_league == "AL"]
    nl_clubs = [c for c in MLB_CLUBS_30.values() if c.sub_league == "NL"]
    cl_clubs = [c for c in NPB_CLUBS_12.values() if c.sub_league == "CL"]
    pl_clubs = [c for c in NPB_CLUBS_12.values() if c.sub_league == "PL"]

    assert len(al_clubs) == 15
    assert len(nl_clubs) == 15
    assert len(cl_clubs) == 6
    assert len(pl_clubs) == 6

    for code, club in ALL_BASEBALL_CLUBS.items():
        assert club.team == code
        park = get_ballpark_metadata(club.primary_venue_id)
        assert park is not None
        assert park.league == club.league
        assert -90.0 <= park.latitude <= 90.0
        assert -180.0 <= park.longitude <= 180.0
        assert park.timezone
        assert 0.70 <= park.run_factor <= 1.40
        assert 0.65 <= park.hr_factor_lhb <= 1.40
        assert 0.65 <= park.hr_factor_rhb <= 1.40


def test_npb_regional_and_mlb_international_neutral_ballparks() -> None:
    """Verify provincial NPB stadiums and MLB international series venues resolve cleanly."""
    provincial_ids = [
        "NPB_MATSUYAMA_BOTCHAN",
        "NPB_MIYAZAKI",
        "NPB_TOYAMA",
        "NPB_NAGANO",
        "NPB_NIIGATA_ECO",
        "NPB_KITAKYUSHU",
        "NPB_AKITA",
        "NPB_KORIYAMA",
        "NPB_HOTTO_MOTTO_KOBE",
    ]
    for vid in provincial_ids:
        park = BASEBALL_BALLPARKS[vid]
        assert park.league == "NPB"
        assert park.is_neutral_or_regional is True
        assert park.timezone == "Asia/Tokyo"

    assert BASEBALL_BALLPARKS["NPB_TOKYO_DOME"].is_dome_or_retractable is True
    assert BASEBALL_BALLPARKS["NPB_KOSHIEN"].is_dome_or_retractable is False
    assert BASEBALL_BALLPARKS["MLB_COL"].run_factor > 1.20
    assert BASEBALL_BALLPARKS["NPB_VANTELIN_DOME"].run_factor < 0.90


def test_tricode_and_multilingual_club_normalization() -> None:
    """Verify historical MLB codes and English/Japanese NPB names normalize accurately."""
    # MLB aliases
    assert normalize_baseball_team("OAK") == "ATH"
    assert normalize_baseball_team("Oakland Athletics") == "ATH"
    assert normalize_baseball_team("TBD") == "TBR"
    assert normalize_baseball_team("TB") == "TBR"
    assert normalize_baseball_team("FLA") == "MIA"
    assert normalize_baseball_team("MON") == "WSN"
    assert normalize_baseball_team("WSH") == "WSN"
    assert normalize_baseball_team("ANA") == "LAA"
    assert normalize_baseball_team("CWS") == "CHW"
    assert normalize_baseball_team("SD") == "SDP"
    assert normalize_baseball_team("SF") == "SFG"
    assert normalize_baseball_team("KC") == "KCR"

    # NPB aliases (English codes, full names, and Japanese kanji/katakana)
    assert normalize_baseball_team("NHF") == "FUK"
    assert normalize_baseball_team("Hokkaido Nippon-Ham Fighters") == "FUK"
    assert normalize_baseball_team("北海道日本ハムファイターズ") == "FUK"
    assert normalize_baseball_team("Yomiuri Giants") == "YOM"
    assert normalize_baseball_team("読売ジャイアンツ") == "YOM"
    assert normalize_baseball_team("Hanshin Tigers") == "HAN"
    assert normalize_baseball_team("阪神タイガース") == "HAN"
    assert normalize_baseball_team("Fukuoka SoftBank Hawks") == "SOF"
    assert normalize_baseball_team("ソフトバンク") == "SOF"
    assert normalize_baseball_team("Orix Buffaloes") == "ORX"
    assert normalize_baseball_team("BSW") == "ORX"

    assert resolve_team_league("LAD") == "MLB"
    assert resolve_team_league("OAK") == "MLB"
    assert resolve_team_league("HAN") == "NPB"
    assert resolve_team_league("NHF") == "NPB"
    assert resolve_team_league("UNKNOWN_TEAM") is None
    assert get_baseball_club_metadata("YOM") is not None


def test_slot_weights_and_league_sabermetric_priors() -> None:
    """Verify batting order slot PA weights sum to 1.0 and decline monotonically 1..9."""
    weights = [BATTING_ORDER_SLOT_PA_WEIGHTS[slot] for slot in range(1, 10)]
    assert pytest.approx(sum(weights), abs=1e-9) == 1.0
    for i in range(len(weights) - 1):
        assert weights[i] > weights[i + 1]

    mlb_p = get_league_sabermetric_priors("mlb")
    npb_p = get_league_sabermetric_priors("npb")
    assert mlb_p.lg_runs_per_team_game > npb_p.lg_runs_per_team_game
    assert mlb_p.lg_iso > npb_p.lg_iso

    with pytest.raises(ValueError, match="Unsupported baseball league"):
        get_league_sabermetric_priors("KBO")


def test_parse_baseball_ip_to_outs_never_treats_point_one_as_tenth() -> None:
    """Verify .1 -> +1 out and .2 -> +2 outs (never decimal 0.1 or 0.2 innings)."""
    assert parse_baseball_ip_to_outs(0) == 0
    assert parse_baseball_ip_to_outs("0.0") == 0
    assert parse_baseball_ip_to_outs("6.0") == 18
    assert parse_baseball_ip_to_outs("6.1") == 19
    assert parse_baseball_ip_to_outs("6.2") == 20
    assert parse_baseball_ip_to_outs(6.1) == 19
    assert parse_baseball_ip_to_outs(6.2) == 20
    assert parse_baseball_ip_to_outs(6 + 1.0 / 3.0) == 19
    assert parse_baseball_ip_to_outs(6 + 2.0 / 3.0) == 20
    assert parse_baseball_ip_to_outs(9) == 27

    assert pytest.approx(outs_to_true_innings(19), abs=1e-9) == 19.0 / 3.0
    assert pytest.approx(outs_to_true_innings(20), abs=1e-9) == 20.0 / 3.0

    with pytest.raises(ValueError):
        parse_baseball_ip_to_outs("6.3")
    with pytest.raises(ValueError):
        parse_baseball_ip_to_outs(6.5)
    with pytest.raises(ValueError):
        parse_baseball_ip_to_outs(-1)
    with pytest.raises(ValueError):
        outs_to_true_innings(-3)


def test_baseball_adapter_doubleheader_and_schema_validation() -> None:
    """Verify BaseballAdapter validates doubleheader event_seq=1,2 and rejects duplicates."""
    adapter = BaseballAdapter()
    assert isinstance(adapter, SportAdapter)
    assert adapter.sport_slug == "baseball"
    assert adapter.topology == "head_to_head"

    games_df = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "game_id": [1001, 1001, 2001],
            "event_seq": [2, 1, 1],
            "start_time_utc": [
                datetime(2025, 5, 10, 22, 10, tzinfo=UTC),
                datetime(2025, 5, 10, 17, 10, tzinfo=UTC),
                datetime(2025, 5, 11, 9, 0, tzinfo=UTC),
            ],
            "home_team": ["OAK", "OAK", "阪神"],
            "away_team": ["LAD", "LAD", "巨人"],
        }
    )

    validated = adapter.validate_event_schema(games_df)
    assert validated["event_seq"].to_list() == [1, 2, 1]
    assert validated["home_team"].to_list() == ["ATH", "ATH", "HAN"]
    assert validated["away_team"].to_list() == ["LAD", "LAD", "YOM"]
    assert validated["league"].to_list() == ["MLB", "MLB", "NPB"]

    # Reject event_seq = 3
    bad_seq_df = games_df.with_columns(pl.lit(3).alias("event_seq"))
    with pytest.raises(SchemaValidationError, match="event_seq"):
        adapter.validate_event_schema(bad_seq_df)

    # Reject duplicate (game_id, event_seq)
    dup_df = pl.concat([games_df.head(1), games_df.head(1)])
    with pytest.raises(SchemaValidationError, match="Duplicate"):
        adapter.validate_event_schema(dup_df)

    # Validate participant schema across doubleheader games
    players_df = pl.DataFrame(
        {
            "game_id": [1001, 1001],
            "event_seq": [1, 2],
            "start_time_utc": [
                datetime(2025, 5, 10, 17, 10, tzinfo=UTC),
                datetime(2025, 5, 10, 22, 10, tzinfo=UTC),
            ],
            "team": ["OAK", "OAK"],
            "player_id": [99, 99],
            "pa": [4, 5],
        }
    )
    val_players = adapter.validate_participant_schema(players_df)
    assert val_players["entity_id"].to_list() == ["ATH", "ATH"]
    assert val_players.height == 2


def test_starter_and_venue_elo_hooks_and_doubleheader_anti_leakage() -> None:
    """Verify Starter & Venue Elo hooks and prove doubleheader Game 2 never leaks into Game 1."""
    starter_hook = BaseballStarterEloHook(
        starter_xfip_by_event_entity={
            (101, "LAD"): 2.90,  # Ace starter (-1.25 below 4.15 baseline -> +40 Elo)
            (101, "SDP"): 4.15,  # Average starter (0 Elo delta)
            (102, "LAD"): 4.80,  # Spot starter in G2
            (102, "SDP"): 3.80,
        }
    )
    venue_hook = BaseballVenueEloHook(
        venue_by_event_id={
            101: "MLB_LAD",  # Primary home ballpark (0 venue delta)
            102: "MLB_GOCHEOK_SKY_DOME",  # Neutral venue (-12.0 dampening)
        }
    )
    assert isinstance(starter_hook, StarterModifierHook)
    assert isinstance(venue_hook, VenueModifierHook)
    assert starter_hook.get_starter_elo_delta(101, "LAD") > 30.0
    assert venue_hook.get_venue_elo_delta(101, "LAD", "SDP") == 0.0
    assert venue_hook.get_venue_elo_delta(102, "LAD", "SDP") == -12.0

    # Doubleheader anti-leakage test on Elo and per-9 IP rolling exposure rate
    base_games = pl.DataFrame(
        {
            "sport": ["baseball", "baseball", "baseball"],
            "season": [2025, 2025, 2025],
            "event_id": [100, 101, 102],
            "event_seq": [1, 1, 2],
            "start_time_utc": [
                datetime(2025, 5, 9, 23, 0, tzinfo=UTC),
                datetime(2025, 5, 10, 17, 0, tzinfo=UTC),
                datetime(2025, 5, 10, 22, 0, tzinfo=UTC),
            ],
            "home_team": ["LAD", "LAD", "LAD"],
            "away_team": ["SDP", "SDP", "SDP"],
            "game_state": ["FINAL", "FINAL", "FINAL"],
            "home_score": [5, 3, 2],
            "away_score": [2, 1, 9],
        }
    )

    mutated_games = base_games.with_columns(
        pl.when(pl.col("event_id") == 102)
        .then(pl.lit(25))
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when(pl.col("event_id") == 102)
        .then(pl.lit(0))
        .otherwise(pl.col("away_score"))
        .alias("away_score"),
    )

    mlb_engine = PairwiseEloEngine(
        config=MLB_ELO_CONFIG,
        starter_hook=starter_hook,
        venue_hook=venue_hook,
    )
    res_base = mlb_engine.compute_ratings(base_games)
    res_mut = mlb_engine.compute_ratings(mutated_games)

    # Game 1 of doubleheader (event_id=101, index 1) must be bit-for-bit identical
    for col in ["elo_home_pre", "elo_away_pre", "elo_diff_pre", "elo_home_win_prob"]:
        assert res_base[col][:2].to_list() == res_mut[col][:2].to_list()

    # Also test NPB 3-way Elo with 12-inning ties
    npb_games = pl.DataFrame(
        {
            "sport": ["baseball", "baseball"],
            "season": [2025, 2025],
            "event_id": [501, 502],
            "event_seq": [1, 1],
            "start_time_utc": [
                datetime(2025, 6, 1, 9, 0, tzinfo=UTC),
                datetime(2025, 6, 2, 9, 0, tzinfo=UTC),
            ],
            "home_team": ["HAN", "HAN"],
            "away_team": ["YOM", "YOM"],
            "game_state": ["FINAL", "FINAL"],
            "home_score": [2, 4],
            "away_score": [2, 1],  # Game 501 is a 12-inning NPB tie!
        }
    )
    npb_engine = PairwiseEloEngine(config=NPB_ELO_CONFIG)
    npb_res = npb_engine.compute_ratings(npb_games)
    assert "elo_draw_prob" in npb_res.columns
    for row in npb_res.iter_rows(named=True):
        total_p = row["elo_home_win_prob"] + row["elo_draw_prob"] + row["elo_away_win_prob"]
        assert pytest.approx(total_p, abs=1e-9) == 1.0
        assert 0.01 < row["elo_draw_prob"] < 0.10


def test_pitching_per_9_ip_exposure_rate_over_outs_pitched() -> None:
    """Verify per-9 IP rate uses scale_factor=27.0 over exact outs_pitched without leakage."""
    pitcher_logs = pl.DataFrame(
        {
            "player_id": [10, 10, 10],
            "start_time_utc": [
                datetime(2025, 4, 1, 20, 0, tzinfo=UTC),
                datetime(2025, 4, 7, 20, 0, tzinfo=UTC),
                datetime(2025, 4, 13, 20, 0, tzinfo=UTC),
            ],
            "event_seq": [1, 1, 1],
            "event_id": [1, 2, 3],
            "outs_pitched": [18, 21, 15],  # 6.0 IP, 7.0 IP, 5.0 IP
            "k_recorded": [8, 7, 14],
        }
    ).with_columns(
        pregame_exposure_rate(
            "k_recorded",
            "outs_pitched",
            window=5,
            group_by="player_id",
            scale_factor=27.0,
        ).alias("pregame_k_per_9")
    )

    # Start 1: no prior starts -> null
    assert pitcher_logs["pregame_k_per_9"][0] is None
    # Start 2: sees only Start 1 (8 K in 18 outs -> 27 * 8 / 18 = 12.0 K/9)
    assert pytest.approx(pitcher_logs["pregame_k_per_9"][1], abs=1e-9) == 12.0
    # Start 3: sees Starts 1 & 2 (15 K in 39 outs -> 27 * 15 / 39; Start 3's 14 K excluded!)
    assert pytest.approx(pitcher_logs["pregame_k_per_9"][2], abs=1e-6) == (27.0 * 15.0 / 39.0)
