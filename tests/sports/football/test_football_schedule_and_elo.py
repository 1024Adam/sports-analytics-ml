"""Tests for weekly-cadence schedule hooks, MOV/QB/venue Elo, and football anti-leakage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.core.contracts import StarterModifierHook, VenueModifierHook
from sports_ml.core.ratings import compute_winner_perspective_mov_multiplier
from sports_ml.sports.football.constants import FOOTBALL_ELO_CONFIG
from sports_ml.sports.football.features import (
    FootballQBEloHook,
    FootballVenueEloHook,
    build_football_schedule_features,
    build_rest_days_lookup,
    compute_football_elo_ratings,
)

K = FOOTBALL_ELO_CONFIG.k_factor
HFA = FOOTBALL_ELO_CONFIG.home_field_advantage


def _team_logs() -> pl.DataFrame:
    rows = [
        # (event_id, week, team, start_utc, venue, overtime)
        (1, 1, "BUF", datetime(2023, 9, 10, 17, 0, tzinfo=UTC), "BUF_HIGHMARK", False),
        (2, 2, "BUF", datetime(2023, 9, 15, 0, 15, tzinfo=UTC), "NYC_METLIFE", False),  # TNF
        (3, 3, "BUF", datetime(2023, 9, 24, 17, 0, tzinfo=UTC), "BUF_HIGHMARK", True),
        (4, 5, "BUF", datetime(2023, 10, 8, 13, 30, tzinfo=UTC), "LON_WEMBLEY", False),
        (5, 1, "DEN", datetime(2023, 9, 10, 20, 25, tzinfo=UTC), "DEN_MILE_HIGH", False),
        (6, 2, "DEN", datetime(2023, 9, 17, 17, 0, tzinfo=UTC), "NYC_METLIFE", False),
    ]
    return pl.DataFrame(
        rows,
        schema=["event_id", "week", "team", "start_time_utc", "venue_id", "overtime"],
        orient="row",
    ).with_columns(pl.lit(2023).alias("season"))


def _row(df: pl.DataFrame, event_id: int, team: str) -> dict[str, object]:
    return df.filter((pl.col("event_id") == event_id) & (pl.col("team") == team)).row(0, named=True)


def test_cadence_flags_on_fixture_schedule() -> None:
    feats = build_football_schedule_features(_team_logs())

    first = _row(feats, 1, "BUF")
    assert first["days_rest"] is None
    assert not first["is_short_week"]
    assert not first["is_post_bye"]

    # Sunday 1pm ET -> Thursday 8:15pm ET: 4 days 7h15m
    tnf = _row(feats, 2, "BUF")
    assert tnf["rest_hours"] == pytest.approx(4 * 24 + 7.25)
    assert tnf["days_rest"] == pytest.approx(4 + 7.25 / 24)
    assert tnf["is_short_week"]
    assert tnf["is_thursday_game"]
    assert tnf["is_thursday_short_week"]
    assert not tnf["is_monday_game"]

    # Thursday night -> following Sunday week: mini-bye (>= 9 and < 13 days)
    mini = _row(feats, 3, "BUF")
    assert mini["is_post_mini_bye"]
    assert not mini["is_short_week"]
    assert not mini["is_post_bye"]
    assert not mini["prior_game_was_overtime"]

    # Sunday -> Sunday with a bye in between: >= 13 days
    post_bye = _row(feats, 4, "BUF")
    assert post_bye["is_post_bye"]
    assert not post_bye["is_post_mini_bye"]
    assert post_bye["prior_game_was_overtime"]

    # Rest is tracked per team: DEN's second game is a normal 7-day week
    den_w2 = _row(feats, 6, "DEN")
    assert den_w2["is_short_week"] is False
    assert den_w2["days_rest"] == pytest.approx(6 + (20 + 35 / 60) / 24)


def test_international_travel_and_body_clock_hour() -> None:
    feats = build_football_schedule_features(_team_logs())
    london = _row(feats, 4, "BUF")
    assert london["is_international_travel"]
    # 13:30 UTC = 9:30 ET body clock, 14:30 BST local
    assert london["kickoff_local_hour_vs_body_clock"] == pytest.approx(9.5)
    assert london["kickoff_local_hour"] == pytest.approx(14.5)
    assert london["tz_shift_hours"] == pytest.approx(5.0)
    assert float(london["travel_km_prev_game"]) > 5000.0  # type: ignore[arg-type]

    home = _row(feats, 1, "BUF")
    assert not home["is_international_travel"]
    assert home["kickoff_local_hour"] == pytest.approx(home["kickoff_local_hour_vs_body_clock"])


def test_schedule_features_future_mutation_invariance() -> None:
    base = _team_logs()
    mutated = base.with_columns(
        pl.when(pl.col("event_id") == 4).then(True).otherwise(pl.col("overtime")).alias("overtime"),
        pl.when(pl.col("event_id") == 4)
        .then(pl.col("start_time_utc") + timedelta(days=3))
        .otherwise(pl.col("start_time_utc"))
        .alias("start_time_utc"),
    )
    before = build_football_schedule_features(base).filter(pl.col("event_id") < 4)
    after = build_football_schedule_features(mutated).filter(pl.col("event_id") < 4)
    assert_frame_equal(before.sort("event_id", "team"), after.sort("event_id", "team"))


def _elo_games() -> pl.DataFrame:
    def ts(month: int, day: int, year: int = 2023) -> datetime:
        return datetime(year, month, day, 17, 0, tzinfo=UTC)

    rows = [
        # (event_id, season, type, start, home, away, hs, as, venue)
        (1, 2023, "REG", ts(9, 10), "BUF", "MIA", 31, 10, "BUF_HIGHMARK"),
        (2, 2023, "REG", ts(9, 17), "KC", "DEN", 20, 20, "KC_ARROWHEAD"),
        (3, 2023, "REG", ts(10, 8), "JAX", "ATL", 23, 7, "LON_WEMBLEY"),
        (4, 2023, "DIV", ts(1, 20, 2024), "BUF", "KC", 24, 27, "BUF_HIGHMARK"),
        (5, 2024, "REG", ts(9, 8, 2024), "BUF", "MIA", 28, 24, "BUF_HIGHMARK"),
    ]
    return pl.DataFrame(
        rows,
        schema=[
            "event_id",
            "season",
            "season_type",
            "start_time_utc",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "venue_id",
        ],
        orient="row",
    ).with_columns(pl.lit("FINAL").alias("game_state"))


def test_elo_zero_sum_tie_neutral_and_offseason_regression() -> None:
    rated = compute_football_elo_ratings(_elo_games()).sort("event_id")

    for row in rated.iter_rows(named=True):
        home_delta = row["elo_home_post"] - row["elo_home_pre"]
        away_delta = row["elo_away_post"] - row["elo_away_pre"]
        assert home_delta + away_delta == pytest.approx(0.0, abs=1e-12)
        assert row["elo_home_win_prob"] + row["elo_away_win_prob"] == pytest.approx(1.0)

    # Game 1: both 1500, HFA only
    g1 = rated.row(0, named=True)
    assert g1["elo_diff_pre"] == pytest.approx(HFA)

    # Game 2 is a tie between fresh teams: update is K * (0.5 - p_home), no MOV multiplier
    g2 = rated.row(1, named=True)
    p_home = g2["elo_home_win_prob"]
    assert g2["elo_home_post"] - g2["elo_home_pre"] == pytest.approx(K * (0.5 - p_home))
    assert g2["elo_home_post"] < g2["elo_home_pre"]

    # Game 3: international neutral site -> no HFA between two fresh clubs
    g3 = rated.row(2, named=True)
    assert g3["elo_diff_pre"] == pytest.approx(0.0)
    assert g3["elo_home_win_prob"] == pytest.approx(0.5)

    # Offseason regression: BUF's 2024 pre-game Elo is 2/3 of its post-2023 rating + 1500/3
    buf_last_2023 = rated.filter(pl.col("event_id") == 4).row(0, named=True)["elo_home_post"]
    buf_2024 = rated.filter(pl.col("event_id") == 5).row(0, named=True)["elo_home_pre"]
    assert buf_2024 == pytest.approx(buf_last_2023 * 2 / 3 + 1500.0 / 3.0)


def test_mov_multiplier_damps_favorites_and_playoffs_scale_k() -> None:
    favorite_blowout = compute_winner_perspective_mov_multiplier(21, 150.0)
    upset_blowout = compute_winner_perspective_mov_multiplier(21, -150.0)
    assert upset_blowout > favorite_blowout

    base = _elo_games().head(1)
    reg = compute_football_elo_ratings(base).row(0, named=True)
    playoff = compute_football_elo_ratings(
        base.with_columns(pl.lit("DIV").alias("season_type"))
    ).row(0, named=True)
    reg_delta = reg["elo_home_post"] - reg["elo_home_pre"]
    playoff_delta = playoff["elo_home_post"] - playoff["elo_home_pre"]
    assert playoff_delta == pytest.approx(reg_delta * FOOTBALL_ELO_CONFIG.playoff_k_multiplier)


def test_franchise_alias_keeps_rating_history() -> None:
    games = pl.DataFrame(
        {
            "event_id": [1, 2],
            "season": [2019, 2020],
            "season_type": ["REG", "REG"],
            "start_time_utc": [
                datetime(2019, 9, 8, 17, 0, tzinfo=UTC),
                datetime(2020, 9, 13, 17, 0, tzinfo=UTC),
            ],
            "home_team": ["OAK", "LV"],
            "away_team": ["DEN", "DEN"],
            "home_score": [24, 10],
            "away_score": [16, 13],
            "game_state": ["FINAL", "FINAL"],
        }
    )
    rated = compute_football_elo_ratings(games).sort("event_id")
    first_post = rated["elo_home_post"][0]
    assert rated["elo_home_pre"][1] == pytest.approx(first_post * 2 / 3 + 500.0)


def test_qb_downgrade_lowers_pre_game_rating_by_kappa_times_delta_value() -> None:
    games = _elo_games().head(1)
    kappa = 150.0
    baseline_value = 0.20
    backup_value = 0.00
    hook = FootballQBEloHook.from_mappings(
        {(1, "BUF"): backup_value},
        {(1, "BUF"): baseline_value},
        kappa_qb=kappa,
    )
    assert isinstance(hook, StarterModifierHook)

    plain = compute_football_elo_ratings(games).row(0, named=True)
    adjusted = compute_football_elo_ratings(games, qb_hook=hook).row(0, named=True)
    expected_shift = kappa * (backup_value - baseline_value)
    assert expected_shift == pytest.approx(-30.0)
    assert adjusted["elo_diff_pre"] - plain["elo_diff_pre"] == pytest.approx(expected_shift)
    assert adjusted["elo_home_win_prob"] < plain["elo_home_win_prob"]

    # Away QB downgrade helps the home side by the same amount
    away_hook = FootballQBEloHook.from_mappings(
        {(1, "MIA"): backup_value}, {(1, "MIA"): baseline_value}, kappa_qb=kappa
    )
    away_adj = compute_football_elo_ratings(games, qb_hook=away_hook).row(0, named=True)
    assert away_adj["elo_diff_pre"] - plain["elo_diff_pre"] == pytest.approx(-expected_shift)


def test_qb_hook_accepts_callables_and_rejects_non_finite() -> None:
    hook = FootballQBEloHook(lambda e, t: 0.1, lambda e, t: 0.3, kappa_qb=100.0)
    assert hook.get_starter_elo_delta(7, "OAK") == pytest.approx(-20.0)
    assert (
        FootballQBEloHook(lambda e, t: None, lambda e, t: 0.3).get_starter_elo_delta(1, "KC") == 0.0
    )
    with pytest.raises(ValueError, match="finite"):
        FootballQBEloHook(lambda e, t: float("nan"), lambda e, t: 0.0).get_starter_elo_delta(
            1, "KC"
        )


def test_venue_hook_altitude_dome_rest_and_neutral() -> None:
    hook = FootballVenueEloHook(
        {1: "DEN_MILE_HIGH", 2: "MIN_US_BANK", 3: "LON_WEMBLEY", 4: "BUF_HIGHMARK"},
        rest_days_by_event_team={
            (4, "BUF"): 11.0,
            (4, "MIA"): 4.0,
            (3, "JAX"): 7.0,
            (3, "ATL"): 6.0,
        },
    )
    assert isinstance(hook, VenueModifierHook)

    expected_alt = 12.0 * (1609.0 - 1000.0) / 1000.0
    assert hook.get_venue_elo_delta(1, "DEN", "KC") == pytest.approx(expected_alt)
    assert hook.get_venue_elo_delta(2, "MIN", "GB") == pytest.approx(2.0)
    # Neutral venue: no environment edge, only rest differential (7 - 6 days)
    assert hook.get_venue_elo_delta(3, "JAX", "ATL") == pytest.approx(3.0)
    # Rest differential capped at +/- 7 days: min(11 - 4, 7) * 3.0
    assert hook.get_venue_elo_delta(4, "BUF", "MIA") == pytest.approx(21.0)
    assert hook.get_venue_elo_delta(99, "BUF", "MIA") == 0.0


def test_rest_lookup_feeds_venue_hook_through_engine() -> None:
    feats = build_football_schedule_features(_team_logs())
    lookup = build_rest_days_lookup(feats)
    assert (2, "BUF") in lookup
    assert (1, "BUF") not in lookup


def test_elo_future_mutation_invariance() -> None:
    games = _elo_games()
    baseline = compute_football_elo_ratings(games).sort("event_id")
    pre_cols = ["event_id", "elo_home_pre", "elo_away_pre", "elo_diff_pre", "elo_home_win_prob"]

    for t in range(1, games.height):
        mutated = games.with_columns(
            pl.when(pl.col("event_id") >= t + 1)
            .then(pl.col("home_score") + 17)
            .otherwise(pl.col("home_score"))
            .alias("home_score"),
            pl.when(pl.col("event_id") >= t + 1)
            .then(pl.col("away_score") * 0)
            .otherwise(pl.col("away_score"))
            .alias("away_score"),
        )
        after = compute_football_elo_ratings(mutated).sort("event_id")
        # Games with index <= t (0-based index t is the first mutated game): pre-game state fixed
        assert_frame_equal(
            baseline.select(pre_cols).head(t + 1), after.select(pre_cols).head(t + 1)
        )
