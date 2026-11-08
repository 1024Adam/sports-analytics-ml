"""Anti-leakage, bye-week, cold-start, QB-swap, and garbage-time tests for NFL features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.sports.football.constants import SILVER_FOOTBALL_AVAILABILITY_SCHEMA
from sports_ml.sports.football.features import (
    STAGE1_FEATURE_COLUMNS,
    TeamFeatureConfig,
    build_football_stage1_features,
    build_qb_features,
    build_roster_continuity,
    build_team_rolling_features,
    compose_qb_value,
    draft_capital_factor,
    prior_season_weight_from_continuity,
    qb_prior_rates,
)
from sports_ml.sports.football.ingestion.pbp_aggregates import aggregate_filtered_offense
from tests.sports.football.synthetic import (
    BYE_WEEK,
    TEAMS,
    SyntheticSilver,
    build_synthetic_silver,
    qb_ids,
)

K = TeamFeatureConfig().k_team_games


def _features(s: SyntheticSilver, **kwargs: object) -> pl.DataFrame:
    return build_football_stage1_features(
        s.games,
        s.team_logs,
        s.qb_logs,
        player_logs=s.player_logs,
        availability=s.availability,
        **kwargs,  # type: ignore[arg-type]
    ).sort("game_id")


def _mutate_from(s: SyntheticSilver, game_ids: set[int]) -> SyntheticSilver:
    """Corrupt every outcome-bearing input (scores, EPA, plays, QB logs) for `game_ids`."""
    in_set = pl.col("game_id").is_in(list(game_ids))
    games = s.games.with_columns(
        pl.when(in_set)
        .then(pl.col("home_score") + 31)
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when(in_set).then(0).otherwise(pl.col("away_score")).alias("away_score"),
        pl.when(in_set).then(1).otherwise(pl.col("home_win")).alias("home_win"),
        pl.when(in_set).then(True).otherwise(pl.col("overtime")).alias("overtime"),
    )
    team_logs = s.team_logs.with_columns(
        *[
            pl.when(in_set).then(pl.col(c) * -4.0 + 1.0).otherwise(pl.col(c)).alias(c)
            for c in ("f_off_epa", "f_off_pass_epa", "f_off_rush_epa")
        ],
        *[
            pl.when(in_set).then(pl.col(c) + 17).otherwise(pl.col(c)).alias(c)
            for c in (
                "f_off_plays",
                "f_off_success",
                "points_for",
                "points_against",
                "off_turnovers",
            )
        ],
        pl.when(in_set)
        .then(pl.col("penalty_yards") + 90)
        .otherwise(pl.col("penalty_yards"))
        .alias("penalty_yards"),
    )
    qb_logs = s.qb_logs.with_columns(
        pl.when(in_set)
        .then(pl.col("qb_epa_total") * -9.0)
        .otherwise(pl.col("qb_epa_total"))
        .alias("qb_epa_total"),
        pl.when(in_set)
        .then(pl.col("dropbacks") + 25)
        .otherwise(pl.col("dropbacks"))
        .alias("dropbacks"),
        pl.when(in_set)
        .then(pl.col("cpoe_total") + 400.0)
        .otherwise(pl.col("cpoe_total"))
        .alias("cpoe_total"),
    )
    return SyntheticSilver(games, team_logs, qb_logs, s.player_logs, s.availability)


def _kickoff_ordered_ids(s: SyntheticSilver) -> list[int]:
    return s.games.sort("start_time_utc", "event_seq", "game_id")["game_id"].to_list()


def test_future_mutation_invariance_across_thursday_sunday_monday() -> None:
    s = build_synthetic_silver()
    baseline = _features(s)
    ordered = _kickoff_ordered_ids(s)
    week_ids = (
        s.games.filter((pl.col("season") == 2022) & (pl.col("week") == 4))
        .sort("start_time_utc", "event_seq", "game_id")["game_id"]
        .to_list()
    )
    first = ordered.index(week_ids[0])
    # Cut at every game of the week (Thursday -> Sunday slots -> Monday) plus a later cut
    cuts = [first + i for i in range(len(week_ids))] + [ordered.index(week_ids[-1]) + 7]
    assert s.games.filter(pl.col("game_id") == week_ids[0])["start_time_utc"][0].weekday() == 3
    assert s.games.filter(pl.col("game_id") == week_ids[-1])["start_time_utc"][0].weekday() == 0

    for t in cuts:
        future = set(ordered[t:])
        mutated = _features(_mutate_from(s, future))
        past = [g for g in ordered[:t]]
        assert_frame_equal(
            baseline.filter(pl.col("game_id").is_in(past)),
            mutated.filter(pl.col("game_id").is_in(past)),
            check_exact=True,
        )
    # The mutation is genuinely visible to later games (the test is not vacuous)
    last_cut = {g for g in ordered[cuts[0] :]}
    later = _features(_mutate_from(s, last_cut)).filter(pl.col("game_id") == ordered[-1])
    assert (
        later["diff_off_epa_per_play_std"][0]
        != baseline.filter(pl.col("game_id") == ordered[-1])["diff_off_epa_per_play_std"][0]
    )


def test_rolling_windows_skip_bye_weeks_without_gaps_or_double_counting() -> None:
    s = build_synthetic_silver()
    team_feats = build_team_rolling_features(s.games, s.team_logs)
    season = s.games.filter(pl.col("season") == 2021)
    week5_teams = set(
        season.filter(pl.col("week") == BYE_WEEK)["home_team"].to_list()
        + season.filter(pl.col("week") == BYE_WEEK)["away_team"].to_list()
    )
    bye_teams = sorted(set(TEAMS) - week5_teams)
    assert len(bye_teams) == 2
    long = pl.concat(
        [
            season.select("game_id", "week", "start_time_utc", pl.col("home_team").alias("team")),
            season.select("game_id", "week", "start_time_utc", pl.col("away_team").alias("team")),
        ]
    )
    any_post_bye = False
    for team in bye_teams:
        rows = long.filter(pl.col("team") == team).sort("start_time_utc")
        assert BYE_WEEK not in rows["week"].to_list()  # no row on the bye week
        rows = rows.filter(pl.col("week") <= 8)  # regular season (week 9 is a playoff game)
        assert rows["week"].to_list() == [1, 2, 3, 4, 6, 7, 8]
        feats = rows.join(team_feats, on=["game_id", "team"]).sort("start_time_utc")
        assert feats["games_played_pre"].to_list() == [0.0, 1, 2, 3, 4, 5, 6]
        # Week 6 (first game after the bye): rolling-4 equals the 4 prior games only
        w6 = feats.filter(pl.col("week") == 6).row(0, named=True)
        prior = (
            s.team_logs.filter((pl.col("team") == team) & (pl.col("season") == 2021))
            .filter(pl.col("week") <= 4)
            .sort("start_time_utc")
        )
        curr = float(prior["f_off_epa"].sum()) / float(prior["f_off_plays"].sum())
        assert w6["off_epa_per_play_roll4"] == pytest.approx(4 * curr / (4 + K))
        assert w6["off_epa_per_play_std"] == pytest.approx(4 * curr / (4 + K))

        # Rest: exact kickoff-to-kickoff days, flagged post-bye iff >= 13
        kick = {r["week"]: r["start_time_utc"] for r in rows.iter_rows(named=True)}
        expected_days = (kick[6] - kick[4]).total_seconds() / 86400.0
        ctx_row = (
            s.games.filter(pl.col("season") == 2021)
            .filter(pl.col("week") == 6)
            .filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
        )
        side = "home" if ctx_row["home_team"][0] == team else "away"
        gid = ctx_row["game_id"][0]
        g1 = _features(s).filter(pl.col("game_id") == gid).row(0, named=True)
        assert g1[f"{side}_days_rest"] == pytest.approx(expected_days)
        assert g1[f"{side}_days_rest"] > 8.0
        assert bool(g1[f"{side}_is_post_bye"]) == (expected_days >= 13.0)
        any_post_bye |= expected_days >= 13.0
    assert any_post_bye


def test_cold_start_week1_is_finite_and_weight_rises_monotonically() -> None:
    s = build_synthetic_silver()
    feats = _features(s)
    first = feats.sort("start_time_utc").head(4)
    assert first.select(STAGE1_FEATURE_COLUMNS).null_count().to_series().sum() == 0
    assert first["home_days_rest_is_imputed"].to_list() == [1, 1, 1, 1]
    assert first["home_continuity_is_imputed"].to_list() == [1, 1, 1, 1]
    for c in STAGE1_FEATURE_COLUMNS:
        if feats.schema[c].is_float():
            assert feats[c].is_nan().sum() == 0, c

    # Constant-EPA team: shrunk value == 0.5 * n / (n + k) for the default league prior of 0
    n_games = 120
    base = datetime(2021, 9, 12, 17, 0, tzinfo=UTC)
    rows = []
    logs = []
    for i in range(n_games):
        start = base + timedelta(days=7 * i)
        gid = 2021_000_000 + i
        rows.append(
            {
                "game_id": gid,
                "event_seq": 1,
                "season": 2021,
                "season_type": "REG",
                "week": i + 1,
                "start_time_utc": start,
                "home_team": "KC",
                "away_team": "DEN",
            }
        )
        for team, opp in (("KC", "DEN"), ("DEN", "KC")):
            logs.append(
                {
                    **{c: 1 for c in s.team_logs.columns if s.team_logs.schema[c].is_integer()},
                    "game_id": gid,
                    "team": team,
                    "opponent": opp,
                    "f_off_plays": 50,
                    "f_off_epa": 25.0 if team == "KC" else 0.0,
                    "f_off_pass_epa": 0.0,
                    "f_off_rush_epa": 0.0,
                    "f_off_drive_points": 0.0,
                    "points_for": 20,
                    "points_against": 17,
                    "penalty_yards": 40,
                }
            )
    games_df = pl.DataFrame(rows).with_columns(pl.col("season").cast(pl.Int32))
    logs_df = pl.DataFrame(logs)
    kc = (
        build_team_rolling_features(
            games_df,
            logs_df.select(*[c for c in s.team_logs.columns if c in logs_df.columns]),
        )
        .filter(pl.col("team") == "KC")
        .sort("game_id")
    )
    n = kc["games_played_pre"].to_numpy()
    values = kc["off_epa_per_play_std"].to_numpy()
    assert values[0] == pytest.approx(0.0)
    assert values == pytest.approx(0.5 * n / (n + K))
    weights = values / 0.5
    assert all(b > a for a, b in zip(weights[:-1], weights[1:], strict=True))
    assert weights[-1] > 0.93  # tends to 1 as games played grows


def test_new_franchise_and_debut_qb_have_finite_features() -> None:
    s = build_synthetic_silver()
    # LAC has no 2021 history (relocated/expansion-style): drop its first season
    lac_2021 = (pl.col("season") == 2021) & (
        (pl.col("home_team") == "LAC") | (pl.col("away_team") == "LAC")
    )
    games = s.games.filter(~lac_2021)
    team_logs = s.team_logs.filter(
        ~(pl.col("game_id").is_in(s.games.filter(lac_2021)["game_id"].to_list()))
        | (pl.col("team") != "LAC")
    )
    feats = build_football_stage1_features(
        games,
        team_logs.filter(pl.col("game_id").is_in(games["game_id"].to_list())),
        s.qb_logs.filter(pl.col("game_id").is_in(games["game_id"].to_list())),
    )
    lac_first = feats.filter(
        (pl.col("season") == 2022)
        & (pl.col("week") == 1)
        & ((pl.col("home_team") == "LAC") | (pl.col("away_team") == "LAC"))
    )
    assert lac_first.height == 1
    assert lac_first.select(STAGE1_FEATURE_COLUMNS).null_count().to_series().sum() == 0

    # A debut QB with first-overall draft capital is priored above a UDFA replacement QB
    debut_id, udfa_id = 777_001, 777_002
    draft = pl.DataFrame(
        {"gsis_id": [f"00-{debut_id:07d}"], "season": [2022], "round": [1], "pick": [1]}
    )
    target = s.games.filter(pl.col("season") == 2022).sort("start_time_utc")["game_id"][0]
    team = s.games.filter(pl.col("game_id") == target)["home_team"][0]
    picked = pl.DataFrame({"game_id": [target], "team": [team], "qb_id": [debut_id]})
    udfa = pl.DataFrame({"game_id": [target], "team": [team], "qb_id": [udfa_id]})
    with_draft = build_qb_features(s.games, s.qb_logs, draft_picks=draft, starting_qbs=picked)
    without = build_qb_features(s.games, s.qb_logs, draft_picks=draft, starting_qbs=udfa)
    a = with_draft.filter((pl.col("game_id") == target) & (pl.col("team") == team)).row(
        0, named=True
    )
    b = without.filter((pl.col("game_id") == target) & (pl.col("team") == team)).row(0, named=True)
    assert a["qb_is_rookie_or_debut"] and b["qb_is_rookie_or_debut"]
    assert a["qb_career_games"] == 0
    assert a["qb_value"] > b["qb_value"]
    assert b["qb_epa_per_dropback"] == pytest.approx(qb_prior_rates(None)[0])
    assert a["qb_epa_per_dropback"] == pytest.approx(qb_prior_rates(1)[0])
    assert draft_capital_factor(1) == pytest.approx(1.0)
    assert draft_capital_factor(None) == 0.0
    assert compose_qb_value(0.1, 5.0) == pytest.approx(0.15)


def test_qb_swap_changes_only_qb_features() -> None:
    s = build_synthetic_silver(backup_starts=frozenset({(2021, 2, "BUF")}))
    target = s.games.filter((pl.col("season") == 2022) & (pl.col("week") == 3)).sort(
        "start_time_utc"
    )
    gid = target["game_id"][0]
    team = target["home_team"][0]
    backup = qb_ids(team)[1]
    swap = pl.DataFrame({"game_id": [gid], "team": [team], "qb_id": [backup]})

    base = _features(s)
    swapped = _features(s, starting_qbs=swap)
    again = _features(s, starting_qbs=swap)
    assert_frame_equal(swapped, again, check_exact=True)  # deterministic

    row_a = base.filter(pl.col("game_id") == gid).row(0, named=True)
    row_b = swapped.filter(pl.col("game_id") == gid).row(0, named=True)
    qb_changed = [
        "home_qb_value",
        "home_qb_is_backup",
        "diff_qb_value",
        "diff_qb_epa_per_dropback",
        "qb_elo_diff_pre",
        "qb_elo_home_win_prob",
    ]
    for c in qb_changed:
        assert row_a[c] != row_b[c], c
    assert row_b["home_qb_is_backup"] == 1

    # Team rolling features, plain Elo, and schedule context are untouched everywhere
    team_cols = [
        c
        for c in STAGE1_FEATURE_COLUMNS
        if c.startswith(("diff_off_", "diff_def_", "diff_adj_", "diff_pythag", "diff_rz_"))
        or c
        in ("elo_home_pre", "elo_away_pre", "diff_elo_pre", "elo_home_win_prob", "home_days_rest")
    ]
    assert_frame_equal(base.select(team_cols), swapped.select(team_cols), check_exact=True)
    assert_frame_equal(
        build_team_rolling_features(s.games, s.team_logs),
        build_team_rolling_features(s.games, s.team_logs),
        check_exact=True,
    )


def _play(
    game: str, team: str, play_id: int, play_type: str, wp: float | None, **kw: object
) -> dict[str, object]:
    base: dict[str, object] = {
        "_gid": game,
        "play_id": play_id,
        "posteam": team,
        "play_type": play_type,
        "epa": 0.5,
        "success": 1,
        "yards_gained": 5.0,
        "pass": int(play_type == "pass"),
        "rush": int(play_type == "run"),
        "qb_dropback": int(play_type == "pass"),
        "wp": wp,
        "down": 1,
        "qb_kneel": 0,
        "qb_spike": 0,
        "yardline_100": 50.0,
        "fixed_drive_result": "Punt",
        "drive": 1,
        "fumble": 0,
        "fumble_lost": 0,
    }
    base.update(kw)
    return base


def test_garbage_time_and_kneel_spike_plays_are_excluded() -> None:
    plays = [
        _play("g1", "KC", 1, "pass", 0.50, epa=1.0, yards_gained=25.0),
        _play("g1", "KC", 2, "run", 0.50, epa=-0.5, success=0, down=2, yards_gained=11.0),
        _play("g1", "KC", 3, "pass", 0.97, epa=3.0, drive=2),  # garbage time: winning big
        _play("g1", "KC", 4, "run", 0.02, epa=2.0, drive=2),  # garbage time: losing big
        _play("g1", "KC", 5, "pass", 0.50, epa=9.0, qb_spike=1, drive=3),  # spike (flagged)
        _play("g1", "KC", 6, "run", 0.50, epa=-0.2, qb_kneel=1, drive=3),  # kneel labeled run
        _play("g1", "KC", 7, "qb_kneel", 0.50, epa=-0.1, drive=3),  # kneel play type
        _play("g1", "KC", 8, "pass", None, epa=4.0, drive=3),  # unknown WP cannot be verified
        _play("g1", "KC", 9, "run", 0.05, epa=0.3, success=0, down=3, drive=4),  # band edge
        _play("g1", "KC", 10, "pass", 0.95, epa=0.7, down=3, drive=4),  # band edge
    ]
    agg = aggregate_filtered_offense(pl.DataFrame(plays))
    row = agg.filter(pl.col("posteam") == "KC").row(0, named=True)
    assert row["f_off_plays"] == 4  # plays 1, 2, 9, 10
    assert row["f_off_epa"] == pytest.approx(1.0 - 0.5 + 0.3 + 0.7)
    assert row["f_off_dropbacks"] == 2
    assert row["f_off_rushes"] == 2
    assert row["f_off_pass_epa"] == pytest.approx(1.7)
    assert row["f_off_rush_epa"] == pytest.approx(-0.2)
    assert row["f_off_success"] == 2  # plays 1 and 10
    assert row["f_off_early_plays"] == 2
    assert row["f_off_early_success"] == 1
    assert row["f_off_explosive"] == 2  # 25-yard pass, 11-yard run
    # Drive 2 begins in garbage time and drive 3 begins with a spike/kneel, so they are excluded
    assert row["f_off_drives"] == 2
    unfiltered_epa = float(pl.DataFrame(plays)["epa"].sum())
    assert row["f_off_epa"] != unfiltered_epa

    with pytest.raises(ValueError, match="missing columns"):
        aggregate_filtered_offense(pl.DataFrame(plays).drop("wp"))


def test_roster_continuity_uses_prior_usage_and_pre_game_depth_chart() -> None:
    s = build_synthetic_silver(seasons=(2021, 2022))
    game = s.games.filter(pl.col("season") == 2022).sort("start_time_utc").row(0, named=True)
    team = game["home_team"]
    kickoff = game["start_time_utc"]
    player_logs = pl.DataFrame(
        {
            "team": [team] * 3,
            "season": [2021] * 3,
            "player_id": [1, 2, 3],
            "snaps": [600.0, 300.0, 100.0],
            "pass_attempts": [None] * 3,
            "carries": [None] * 3,
            "targets": [None] * 3,
        },
        schema_overrides={"pass_attempts": pl.Int16, "carries": pl.Int16, "targets": pl.Int16},
    )

    def depth(players: list[int], stamp: datetime) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "as_of_time_utc": [stamp] * len(players),
                "game_id": [game["game_id"]] * len(players),
                "season": [2022] * len(players),
                "week": [1] * len(players),
                "team": [team] * len(players),
                "player_id": players,
                "position": ["WR"] * len(players),
                "source": ["depth_chart"] * len(players),
                "report_status": [None] * len(players),
                "practice_status": [None] * len(players),
                "depth_rank": [1] * len(players),
                "is_inactive": [False] * len(players),
            },
            schema=SILVER_FOOTBALL_AVAILABILITY_SCHEMA,
        )

    early = kickoff - timedelta(days=2)
    late = kickoff - timedelta(minutes=30)  # after the kickoff - 90 min cutoff
    availability = pl.concat([depth([1, 2], early), depth([3], late)])
    cont = build_roster_continuity(s.games, player_logs, availability)
    row = cont.filter((pl.col("game_id") == game["game_id"]) & (pl.col("team") == team)).row(
        0, named=True
    )
    assert row["roster_continuity"] == pytest.approx(900.0 / 1000.0)  # late snapshot ignored
    assert not row["continuity_is_imputed"]
    assert row["prior_season_weight"] == pytest.approx(prior_season_weight_from_continuity(0.9))

    other = cont.filter((pl.col("game_id") == game["game_id"]) & (pl.col("team") != team)).row(
        0, named=True
    )
    assert other["continuity_is_imputed"]  # no prior-season usage data

    coach = pl.DataFrame({"team": [team], "season": [2022], "head_coach_changed": [True]})
    changed = build_roster_continuity(s.games, player_logs, availability, head_coach_changed=coach)
    row2 = changed.filter((pl.col("game_id") == game["game_id"]) & (pl.col("team") == team)).row(
        0, named=True
    )
    assert row2["prior_season_weight"] < row["prior_season_weight"]
    assert prior_season_weight_from_continuity(1.0) > prior_season_weight_from_continuity(0.0)


def test_adjusted_epa_ratings_track_team_strength() -> None:
    s = build_synthetic_silver()
    feats = build_team_rolling_features(s.games, s.team_logs)
    last_week = s.games.filter((pl.col("season") == 2023) & (pl.col("week") == 8)).select(
        "game_id", "season"
    )
    wk8 = feats.join(last_week, on="game_id")
    assert wk8.height >= 6
    corr = wk8.select(pl.corr("adj_off_epa", "off_epa_per_play_std", method="spearman")).item()
    assert corr > 0.6
    first = feats.join(s.games.select("game_id", "season", "week"), on="game_id").filter(
        (pl.col("season") == 2021) & (pl.col("week") == 1)
    )
    assert first["adj_off_epa"].abs().max() == pytest.approx(0.0)
