"""Milestone 28 tests: play-by-play micro aggregation, NGS, rolling features, Stage 2 hooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from sports_ml.sports.football.constants import (
    SILVER_FOOTBALL_PLAYER_MICRO_SCHEMA,
    SILVER_FOOTBALL_TEAM_MICRO_SCHEMA,
    make_football_game_id,
)
from sports_ml.sports.football.features import (
    MICRO_FEATURE_COLUMNS,
    STAGE1_FEATURE_COLUMNS,
    STAGE1_MICRO_FEATURE_COLUMNS,
    attach_micro_features,
    build_football_stage1_features,
    build_football_stage2_features,
    build_micro_team_features,
    is_forbidden_feature_column,
)
from sports_ml.sports.football.ingestion import (
    FootballSilverReconciler,
    NeutralSituationConfig,
    aggregate_team_ngs,
    build_player_pbp_micro_logs,
    build_team_pbp_micro_logs,
    neutral_situation_mask,
)
from sports_ml.sports.football.ingestion.reconciler import build_team_game_logs
from sports_ml.sports.football.models import Stage1FootballGameModel, Stage2FootballPlayerModel
from tests.sports.football.synthetic import (
    SyntheticSilver,
    build_synthetic_player_logs,
    build_synthetic_silver,
    player_ids,
)
from tests.sports.football.synthetic_pbp import build_synthetic_ngs, build_synthetic_pbp

SEASONS = (2014, 2015, 2016, 2017)


@pytest.fixture(scope="module")
def silver() -> SyntheticSilver:
    return build_synthetic_silver(seasons=SEASONS, future_weeks=1)


@pytest.fixture(scope="module")
def pbp(silver: SyntheticSilver) -> pl.DataFrame:
    return build_synthetic_pbp(silver)


@pytest.fixture(scope="module")
def ngs(silver: SyntheticSilver) -> dict[str, pl.DataFrame]:
    return build_synthetic_ngs(silver)


@pytest.fixture(scope="module")
def micro(silver: SyntheticSilver, pbp: pl.DataFrame, ngs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return build_team_pbp_micro_logs(
        pbp,
        silver.games,
        ngs_passing=ngs["passing"],
        ngs_rushing=ngs["rushing"],
        ngs_receiving=ngs["receiving"],
    )


def _play(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "game_id": "2021_01_BUF_MIA",
        "play_id": 1,
        "posteam": "BUF",
        "play_type": "pass",
        "epa": 0.0,
        "pass": 1.0,
        "rush": 0.0,
        "qb_dropback": 1.0,
        "wp": 0.5,
        "down": 1.0,
        "yards_gained": 5.0,
        "drive": 1.0,
        "xpass": 0.5,
        "qtr": 1.0,
        "half_seconds_remaining": 1500.0,
        "game_seconds_remaining": 3400.0,
        "score_differential": 0.0,
    }
    base.update(kw)
    return base


def _tiny_games() -> pl.DataFrame:
    s = build_synthetic_silver(seasons=(2021,), future_weeks=0)
    return s.games.filter((pl.col("home_team") == "MIA") & (pl.col("away_team") == "BUF"))


# ------------------------------------------------------------------ aggregation / conservation


def test_play_level_epa_and_plays_reconcile_to_team_game_logs(
    silver: SyntheticSilver, pbp: pl.DataFrame, micro: pl.DataFrame
) -> None:
    logs = build_team_game_logs(pbp, silver.games)
    joined = micro.join(
        logs.select("game_id", "team", "off_epa_total", "off_plays"), on=["game_id", "team"]
    )
    assert joined.height == micro.height == logs.height
    assert (joined["epa_total"] - joined["off_epa_total"]).abs().max() < 1e-6  # type: ignore[operator]
    assert (joined["plays"] - joined["off_plays"]).abs().max() == 0.0
    assert dict(micro.schema) == SILVER_FOOTBALL_TEAM_MICRO_SCHEMA
    assert micro.select(pl.all().is_null().sum()).sum_horizontal().item() == 0


def test_neutral_filters_never_include_plays_outside_the_score_time_band() -> None:
    cfg = NeutralSituationConfig()
    cases: list[tuple[str, dict[str, Any], bool]] = [
        ("in band", {}, True),
        ("wp at floor", {"wp": 0.20}, True),
        ("wp below floor", {"wp": 0.19}, False),
        ("wp above cap", {"wp": 0.81}, False),
        ("score gap 14", {"score_differential": -14.0}, True),
        ("score gap 15", {"score_differential": 15.0}, False),
        ("two-minute warning", {"half_seconds_remaining": 120.0}, False),
        ("just outside two minutes", {"half_seconds_remaining": 121.0}, True),
        ("overtime", {"qtr": 5.0}, False),
        ("fourth down", {"down": 4.0}, False),
        ("third down", {"down": 3.0}, True),
        ("missing wp", {"wp": None}, False),
    ]
    frame = pl.DataFrame([_play(**kw) for _n, kw, _e in cases], infer_schema_length=None)
    mask = frame.select(neutral_situation_mask(cfg).alias("m"))["m"].to_list()
    assert mask == [expected for _n, _kw, expected in cases]


def test_micro_neutral_counts_only_use_the_neutral_band(
    silver: SyntheticSilver, pbp: pl.DataFrame, micro: pl.DataFrame
) -> None:
    cfg = NeutralSituationConfig()
    scrim = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"])
        & (pl.col("qb_kneel") == 0)
        & (pl.col("qb_spike") == 0)
    )
    inside = scrim.filter(
        (pl.col("wp").is_between(cfg.wp_min, cfg.wp_max))
        & (pl.col("score_differential").abs() <= cfg.max_abs_score_diff)
        & (pl.col("half_seconds_remaining") > cfg.min_half_seconds)
        & (pl.col("qtr") <= 4)
        & (pl.col("down") <= cfg.max_down)
    )
    assert inside.height == int(micro["neutral_plays"].sum())
    assert 0 < inside.height < scrim.height
    assert (micro["neutral_plays"] >= micro["proe_den"]).all()


def test_micro_metrics_match_hand_computed_values() -> None:
    games = _tiny_games()
    gid_str = games["nflverse_game_id"][0]
    plays = [
        _play(game_id=gid_str, play_id=1, game_seconds_remaining=3400.0, xpass=0.6),
        _play(
            game_id=gid_str,
            play_id=2,
            game_seconds_remaining=3370.0,
            xpass=0.4,
            **{
                "pass": 0.0,
                "rush": 1.0,
                "qb_dropback": 0.0,
                "play_type": "run",
                "yards_gained": -1.0,
            },
        ),
        _play(
            game_id=gid_str,
            play_id=3,
            game_seconds_remaining=3330.0,
            xpass=0.5,
            sack=1.0,
            qb_hit=1.0,
            cpoe=-10.0,
            epa=-2.0,
        ),
        _play(
            game_id=gid_str,
            play_id=4,
            game_seconds_remaining=3300.0,
            qb_kneel=1.0,
            play_type="run",
            **{"pass": 0.0, "rush": 1.0, "qb_dropback": 0.0},
        ),
        _play(
            game_id=gid_str,
            play_id=5,
            down=4.0,
            ydstogo=2.0,
            yardline_100=40.0,
            play_type="punt",
            game_seconds_remaining=3290.0,
            epa=None,
        ),
    ]
    frame = pl.DataFrame(plays, infer_schema_length=None)
    out = build_team_pbp_micro_logs(frame, games).filter(pl.col("team") == "BUF").row(0, named=True)
    assert out["plays"] == 4.0  # pass, run, pass, kneel (punt is not scrimmage)
    assert out["epa_total"] == pytest.approx(-2.0)
    assert out["neutral_plays"] == 3.0  # kneel excluded; punt is not a pass/run
    # seconds: play1->play2 30s, play2->play3 40s, play3->play4 30s
    assert out["neutral_seconds"] == pytest.approx(100.0)
    assert out["neutral_seconds_plays"] == 3.0
    assert out["proe_num"] == pytest.approx((1 - 0.6) + (0 - 0.4) + (1 - 0.5))
    assert out["proe_den"] == 3.0
    assert out["dropbacks"] == 2.0 and out["sacks"] == 1.0 and out["pressure_num"] == 1.0
    assert out["cpoe_sum"] == pytest.approx(-10.0) and out["cpoe_plays"] == 1.0
    assert out["run_stuffs"] == 1.0 and out["rush_plays"] == 1.0
    assert out["fourth_go_den"] == 1.0 and out["fourth_go_num"] == 0.0
    assert out["has_ngs"] == 0


def test_pressure_and_pace_fall_back_gracefully_when_columns_are_missing() -> None:
    games = _tiny_games()
    frame = pl.DataFrame(
        [
            _play(game_id=games["nflverse_game_id"][0])
            | {"xpass": None, "half_seconds_remaining": None}
        ]
    ).drop("score_differential", "qtr")
    row = build_team_pbp_micro_logs(frame, games).filter(pl.col("team") == "BUF").row(0, named=True)
    assert row["neutral_plays"] == 0.0 and row["proe_den"] == 0.0 and row["cpoe_plays"] == 0.0


# ------------------------------------------------------------------------------------- NGS


def test_ngs_team_aggregates_are_exposure_weighted_and_skip_season_totals(
    silver: SyntheticSilver,
) -> None:
    game = silver.games.filter(pl.col("season") == 2016).sort("start_time_utc").row(0, named=True)
    team = str(game["home_team"])
    week = game["week"]
    passing = pl.DataFrame(
        {
            "season": [2016, 2016, 2016],
            "week": [week, week, 0],
            "team_abbr": [team, team, team],
            "attempts": [30.0, 10.0, 500.0],
            "avg_time_to_throw": [2.5, 3.5, 9.0],
            "completion_percentage_above_expectation": [4.0, -2.0, 50.0],
        }
    )
    out = aggregate_team_ngs(silver.games, passing=passing).filter(pl.col("team") == team)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["ngs_pass_attempts"] == 40.0  # the week-0 season row is ignored
    assert row["ngs_time_to_throw_sum"] == pytest.approx(2.5 * 30 + 3.5 * 10)
    assert row["ngs_cpoe_sum"] == pytest.approx(4.0 * 30 - 2.0 * 10)
    assert row["ngs_rush_attempts"] == 0.0


def test_has_ngs_flag_follows_source_availability(micro: pl.DataFrame) -> None:
    by_season = micro.group_by("season").agg(pl.col("has_ngs").mean()).sort("season")
    assert by_season["has_ngs"].to_list() == [0.0, 0.0, 1.0, 1.0]
    pre = micro.filter(pl.col("season") < 2016)
    assert pre["ngs_pass_attempts"].sum() == 0.0 and pre["ngs_ryoe_sum"].sum() == 0.0


def test_player_micro_logs_cover_receivers_rushers_and_ngs(
    silver: SyntheticSilver, pbp: pl.DataFrame, ngs: dict[str, pl.DataFrame]
) -> None:
    players = build_player_pbp_micro_logs(
        pbp, silver.games, ngs_rushing=ngs["rushing"], ngs_receiving=ngs["receiving"]
    )
    assert dict(players.schema) == SILVER_FOOTBALL_PLAYER_MICRO_SCHEMA
    assert players.select(["game_id", "team", "player_id"]).is_unique().all()
    rb = player_ids("BUF")["RB1"]
    rb_rows = players.filter(pl.col("player_id") == rb)
    assert rb_rows["rush_attempts"].sum() > 0 and rb_rows["ngs_ryoe_sum"].abs().sum() > 0
    assert set(players.filter(pl.col("targets") > 0)["player_id"].to_list()) <= {
        v
        for t in ("BUF", "MIA", "NE", "NYJ", "KC", "DEN", "LV", "LAC")
        for v in player_ids(t).values()
    }
    pre = players.filter(pl.col("season") < 2016)
    assert pre["has_ngs"].sum() == 0


def test_reconciler_upserts_micro_logs_idempotently(
    tmp_path: Any, silver: SyntheticSilver, pbp: pl.DataFrame, ngs: dict[str, pl.DataFrame]
) -> None:
    rec = FootballSilverReconciler(tmp_path / "silver")
    silver.games.write_parquet(rec.games_path)
    first = rec.reconcile_team_pbp_micro_logs(pbp, ngs_passing=ngs["passing"])
    again = rec.reconcile_team_pbp_micro_logs(pbp, ngs_passing=ngs["passing"])
    assert first.height == again.height
    assert again.select(["game_id", "team"]).is_unique().all()
    players = rec.reconcile_player_pbp_micro_logs(pbp, ngs_rushing=ngs["rushing"])
    assert rec.player_micro_path.exists() and players.height > 0


# -------------------------------------------------------------------- rolling features / Gold


def _gold(silver: SyntheticSilver, micro: pl.DataFrame | None) -> pl.DataFrame:
    return build_football_stage1_features(
        silver.games, silver.team_logs, silver.qb_logs, micro_logs=micro
    )


def _full(silver: SyntheticSilver, micro: pl.DataFrame | None) -> pl.DataFrame:
    """Every micro column (shipped and experimental) per game, before Stage 1 column selection."""
    base = silver.games.select("game_id", "season", "start_time_utc", "home_team", "away_team")
    return attach_micro_features(base, build_micro_team_features(silver.games, micro))


def test_micro_columns_pass_the_firewall_and_join_stage1(
    silver: SyntheticSilver, micro: pl.DataFrame
) -> None:
    assert not [c for c in MICRO_FEATURE_COLUMNS if is_forbidden_feature_column(c)]
    assert set(STAGE1_MICRO_FEATURE_COLUMNS) <= set(STAGE1_FEATURE_COLUMNS)
    assert set(STAGE1_MICRO_FEATURE_COLUMNS) <= set(MICRO_FEATURE_COLUMNS)
    gold = _gold(silver, micro)
    full = _full(silver, micro)
    for frame, cols in ((gold, STAGE1_MICRO_FEATURE_COLUMNS), (full, MICRO_FEATURE_COLUMNS)):
        for c in cols:
            series = frame[c].cast(pl.Float64)
            assert series.null_count() == 0, c
            assert series.is_finite().all(), c
    assert "pace_total_neutral" not in gold.columns  # experimental columns are not in Gold
    assert full["pace_total_neutral"].std() > 0.0  # type: ignore[operator]


def test_pre_2016_games_have_finite_imputed_ngs_features(
    silver: SyntheticSilver, micro: pl.DataFrame
) -> None:
    gold = _full(silver, micro)
    pre = gold.filter(pl.col("season") < 2016)
    assert (pre["home_ngs_is_imputed"] == 1).all() and (pre["away_ngs_is_imputed"] == 1).all()
    for c in ("diff_time_to_throw", "diff_ryoe_per_att", "diff_separation"):
        assert pre[c].is_finite().all() and (pre[c] == 0.0).all()  # league prior vs league prior
    first_2016 = gold.filter(pl.col("season") == 2016).sort("start_time_utc").head(1)
    assert first_2016["home_ngs_is_imputed"][0] == 1
    late = gold.filter(pl.col("season") == 2017).sort("start_time_utc").tail(1)
    assert late["home_ngs_is_imputed"][0] == 0 and late["away_ngs_is_imputed"][0] == 0
    # play-by-play micro features exist before 2016: only the first-ever games are imputed
    pbp_flags = pre.sort("start_time_utc")
    assert pbp_flags["home_micro_is_imputed"].sum() <= 16
    assert pbp_flags.tail(8)["home_micro_is_imputed"].sum() == 0


def test_no_micro_logs_resolves_to_priors_and_flags_everything_imputed(
    silver: SyntheticSilver,
) -> None:
    gold = _gold(silver, None)
    assert (gold["home_micro_is_imputed"] == 1).all() and (gold["away_micro_is_imputed"] == 1).all()
    assert gold["diff_proe"].abs().max() == 0.0
    full = _full(silver, None)
    assert (full["away_ngs_is_imputed"] == 1).all()
    assert full["pace_total_neutral"].unique().to_list() == [59.0]


def test_rolling_micro_features_are_strictly_pre_game(
    silver: SyntheticSilver, pbp: pl.DataFrame, ngs: dict[str, pl.DataFrame], micro: pl.DataFrame
) -> None:
    target = silver.games.filter(pl.col("game_state") == "FINAL").sort("start_time_utc")[100]
    t0 = target["start_time_utc"][0]
    late = pl.col("game_id").is_in(
        silver.games.filter(pl.col("start_time_utc") >= t0)["nflverse_game_id"].to_list()
    )
    mutated_pbp = pbp.with_columns(
        pl.when(late).then(pl.col("epa") + 3.0).otherwise(pl.col("epa")).alias("epa"),
        pl.when(late).then(pl.lit(1.0)).otherwise(pl.col("pass")).alias("pass"),
        pl.when(late).then(pl.col("xpass") * 0.1).otherwise(pl.col("xpass")).alias("xpass"),
        pl.when(late).then(pl.col("cpoe") + 40.0).otherwise(pl.col("cpoe")).alias("cpoe"),
        pl.when(late).then(pl.lit(1.0)).otherwise(pl.col("qb_hit")).alias("qb_hit"),
        pl.when(late)
        .then(pl.col("game_seconds_remaining") - 20.0)
        .otherwise(pl.col("game_seconds_remaining"))
        .alias("game_seconds_remaining"),
    )
    late_games = silver.games.filter(pl.col("start_time_utc") >= t0)["game_id"].to_list()
    t_season, t_week = int(target["season"][0]), int(target["week"][0])
    scale = ("attempts", "avg_time_to_throw", "rush_yards_over_expected", "avg_separation")
    after_t = (pl.col("season") > t_season) | (
        (pl.col("season") == t_season) & (pl.col("week") >= t_week)
    )
    mutated_ngs = {
        k: v.with_columns(
            pl.when(after_t).then(pl.col(c) * 5.0).otherwise(pl.col(c)).alias(c)
            for c in v.columns
            if c in scale
        )
        for k, v in ngs.items()
    }
    changed_micro = build_team_pbp_micro_logs(
        mutated_pbp,
        silver.games,
        ngs_passing=mutated_ngs["passing"],
        ngs_rushing=mutated_ngs["rushing"],
        ngs_receiving=mutated_ngs["receiving"],
    )
    mutated_rows = micro.filter(pl.col("game_id").is_in(late_games))
    assert not mutated_rows.equals(
        changed_micro.filter(pl.col("game_id").is_in(late_games))
    )  # control

    base = _gold(silver, micro)
    after = _gold(silver, changed_micro)
    keep = ["game_id", *STAGE1_MICRO_FEATURE_COLUMNS]
    full_base, full_after = _full(silver, micro), _full(silver, changed_micro)
    full_keep = ["game_id", *MICRO_FEATURE_COLUMNS]
    for a, b in ((full_base, full_after),):
        left = a.filter(pl.col("start_time_utc") <= t0).select(full_keep).sort("game_id")
        right = b.filter(pl.col("start_time_utc") <= t0).select(full_keep).sort("game_id")
        assert left.equals(right)  # every micro column, shipped or not
    early = base.filter(pl.col("start_time_utc") <= t0).select(keep).sort("game_id")
    early_after = after.filter(pl.col("start_time_utc") <= t0).select(keep).sort("game_id")
    assert early.height > 0
    assert early.equals(early_after)  # bit-for-bit, including the game at t0 itself
    assert not base.select(keep).sort("game_id").equals(after.select(keep).sort("game_id"))


def test_micro_features_respond_to_history_in_the_expected_direction(
    silver: SyntheticSilver, micro: pl.DataFrame
) -> None:
    games = silver.games
    team = "BUF"
    target = (
        games.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
        .filter(pl.col("season") == 2017)
        .sort("start_time_utc")[6]
    )
    prior = micro.filter(
        (pl.col("team") == team) & (pl.col("start_time_utc") < target["start_time_utc"][0])
    )
    boosted = micro.with_columns(
        pl.when((pl.col("team") == team) & pl.col("game_id").is_in(prior["game_id"].to_list()))
        .then(pl.col("proe_num") + pl.col("proe_den") * 0.2)
        .otherwise(pl.col("proe_num"))
        .alias("proe_num")
    )
    base_val = build_micro_team_features(games, micro).filter(
        (pl.col("game_id") == target["game_id"][0]) & (pl.col("team") == team)
    )["proe"][0]
    high_val = build_micro_team_features(games, boosted).filter(
        (pl.col("game_id") == target["game_id"][0]) & (pl.col("team") == team)
    )["proe"][0]
    assert high_val > base_val
    assert high_val < base_val + 0.2  # shrinkage keeps a +0.2 per-play shock below its raw size


def test_baseline_model_trains_with_micro_columns(
    silver: SyntheticSilver, micro: pl.DataFrame
) -> None:
    gold = _gold(silver, micro)
    done = gold.filter(pl.col("game_state") == "FINAL")
    model = Stage1FootballGameModel(min_calibration_samples=30).fit(done)
    assert set(STAGE1_MICRO_FEATURE_COLUMNS) <= set(model.feature_columns)
    preds = model.predict(gold)
    assert ((preds["home_win_prob"] > 0) & (preds["home_win_prob"] < 1)).all()


# ----------------------------------------------------------------------------------- Stage 2


@pytest.fixture(scope="module")
def player_micro(
    silver: SyntheticSilver, pbp: pl.DataFrame, ngs: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    return build_player_pbp_micro_logs(
        pbp, silver.games, ngs_rushing=ngs["rushing"], ngs_receiving=ngs["receiving"]
    )


def test_stage2_yprr_and_ryoe_are_pre_game_and_flag_missing_ngs(
    silver: SyntheticSilver, player_micro: pl.DataFrame
) -> None:
    logs = build_synthetic_player_logs(silver)
    rows = build_football_stage2_features(
        silver.games, logs, silver.team_logs, player_micro=player_micro
    )
    pre = rows.filter(pl.col("season") < 2016)
    assert (pre["ngs_is_imputed"] == 1).all()
    assert (pre["rush_yards_over_expected_per_att"] == 0.0).all()
    receivers = rows.filter(pl.col("position").is_in(["WR", "TE"]) & (pl.col("season") == 2017))
    assert receivers["yprr"].is_finite().all() and float(receivers["yprr"].min()) > 0.0  # type: ignore[arg-type]
    qb = rows.filter(pl.col("position") == "QB")
    assert (qb["yprr"] == 0.0).all()
    late_rb = rows.filter((pl.col("position") == "RB") & (pl.col("season") == 2017))
    assert (late_rb["ngs_is_imputed"] == 0).any()
    assert float(late_rb["rush_yards_over_expected_per_att"].abs().max()) > 0.0  # type: ignore[arg-type]

    target = silver.games.filter(pl.col("game_state") == "FINAL").sort("start_time_utc")[110]
    t0 = target["start_time_utc"][0]
    mutated_micro = player_micro.with_columns(
        pl.when(pl.col("start_time_utc") >= t0)
        .then(pl.col("ngs_ryoe_sum") + 500.0)
        .otherwise(pl.col("ngs_ryoe_sum"))
        .alias("ngs_ryoe_sum")
    )
    mutated_logs = logs.with_columns(
        pl.when(pl.col("start_time_utc") >= t0)
        .then(pl.col("rec_yards") * 4.0)
        .otherwise(pl.col("rec_yards"))
        .alias("rec_yards"),
        pl.when(pl.col("start_time_utc") >= t0)
        .then(pl.col("snaps").fill_null(40) * 2)
        .otherwise(pl.col("snaps"))
        .cast(pl.Int16)
        .alias("snaps"),
    )
    changed = build_football_stage2_features(
        silver.games, mutated_logs, silver.team_logs, player_micro=mutated_micro
    )
    cols = ["game_id", "player_id", "yprr", "rush_yards_over_expected_per_att", "ngs_is_imputed"]
    a = rows.filter(pl.col("start_time_utc") <= t0).select(cols).sort("game_id", "player_id")
    b = changed.filter(pl.col("start_time_utc") <= t0).select(cols).sort("game_id", "player_id")
    assert a.equals(b)


def test_stage2_projections_respond_to_ryoe_and_yprr(
    silver: SyntheticSilver, player_micro: pl.DataFrame
) -> None:
    logs = build_synthetic_player_logs(silver)
    gold = build_football_stage1_features(silver.games, silver.team_logs, silver.qb_logs)
    stage1 = Stage1FootballGameModel(min_calibration_samples=30).fit(
        gold.filter(pl.col("game_state") == "FINAL")
    )
    preds = stage1.predict(gold)
    rows = build_football_stage2_features(
        silver.games, logs, silver.team_logs, stage1_df=gold, player_micro=player_micro
    )
    model = Stage2FootballPlayerModel().fit(rows, preds)
    upcoming = rows.filter(pl.col("game_state") != "FINAL")
    game = int(upcoming["game_id"][0])
    team = str(upcoming.filter(pl.col("game_id") == game)["team"][0])
    sub = upcoming.filter((pl.col("game_id") == game) & (pl.col("team") == team))

    def stat(frame: pl.DataFrame, pos: str, col: str) -> float:
        out = model.predict(frame, preds, with_probabilities=False)
        return float(out.filter(pl.col("position") == pos)[col].sum())

    hot = sub.with_columns(
        pl.when(pl.col("position") == "RB")
        .then(2.0)
        .otherwise(pl.col("rush_yards_over_expected_per_att"))
        .alias("rush_yards_over_expected_per_att")
    )
    cold = sub.with_columns(
        pl.when(pl.col("position") == "RB")
        .then(-2.0)
        .otherwise(pl.col("rush_yards_over_expected_per_att"))
        .alias("rush_yards_over_expected_per_att")
    )
    assert stat(hot, "RB", "proj_rush_yds") > stat(cold, "RB", "proj_rush_yds")
    fast = sub.with_columns((pl.col("yprr") * 1.4).alias("yprr"))
    slow = sub.with_columns((pl.col("yprr") * 0.6).alias("yprr"))
    assert stat(fast, "WR", "proj_rec_yds") > stat(slow, "WR", "proj_rec_yds")
    assert np.isfinite(stat(sub, "QB", "proj_pass_yds"))


def test_micro_game_ids_are_canonical(silver: SyntheticSilver, micro: pl.DataFrame) -> None:
    assert set(micro["game_id"].to_list()) <= set(silver.games["game_id"].to_list())
    g = silver.games.filter(pl.col("game_state") == "FINAL").row(0, named=True)
    gid = make_football_game_id(
        int(g["season"]), int(g["week"]), str(g["away_team"]), str(g["home_team"])
    )
    assert gid == g["game_id"]
    assert micro.filter(pl.col("game_id") == gid).height == 2
