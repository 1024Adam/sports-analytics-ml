"""Stage 2 football tests: features, projections, reconciliation, predictor, ledger."""

from __future__ import annotations

from datetime import UTC, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sports_ml.config import get_settings
from sports_ml.core.registry import ModelRegistry
from sports_ml.sports.football.features import (
    STAGE2_FEATURE_COLUMNS,
    STAGE2_PLAYER_FEATURE_COLUMNS,
    STAGE2_TARGET_COLUMNS,
    assert_zero_firewall_contamination,
    build_and_persist_football_gold_features,
    build_and_persist_football_stage2_features,
    build_football_stage1_features,
    build_football_stage2_features,
)
from sports_ml.sports.football.models import (
    FootballMatchupPredictor,
    Stage1FootballGameModel,
    Stage2FootballPlayerModel,
    evaluate_football_stage2_holdout,
    load_football_stage1_bundle,
    score_football_prediction_ledger,
    train_football_stage1_bundle,
)
from sports_ml.sports.football.models.stage2_player import (
    PROBABILITY_COLUMNS,
    PROJECTION_COLUMNS,
    team_context_from_stage1,
)
from tests.sports.football.synthetic import (
    SyntheticSilver,
    build_synthetic_player_logs,
    build_synthetic_silver,
    qb_ids,
    write_synthetic_silver,
)

KEYS = ["game_id", "team", "player_id"]


@pytest.fixture(scope="module")
def silver() -> SyntheticSilver:
    return build_synthetic_silver(future_weeks=1)


@pytest.fixture(scope="module")
def player_logs(silver: SyntheticSilver) -> pl.DataFrame:
    return build_synthetic_player_logs(silver)


@pytest.fixture(scope="module")
def gold(silver: SyntheticSilver) -> pl.DataFrame:
    return build_football_stage1_features(silver.games, silver.team_logs, silver.qb_logs)


@pytest.fixture(scope="module")
def stage2_rows(
    silver: SyntheticSilver, player_logs: pl.DataFrame, gold: pl.DataFrame
) -> pl.DataFrame:
    return build_football_stage2_features(
        silver.games, player_logs, silver.team_logs, stage1_df=gold
    )


@pytest.fixture(scope="module")
def stage1_model(gold: pl.DataFrame) -> Stage1FootballGameModel:
    done = gold.filter(pl.col("game_state") == "FINAL")
    return Stage1FootballGameModel(min_calibration_samples=30).fit(done)


@pytest.fixture(scope="module")
def stage1_preds(stage1_model: Stage1FootballGameModel, gold: pl.DataFrame) -> pl.DataFrame:
    return stage1_model.predict(gold)


@pytest.fixture(scope="module")
def stage2_model(
    stage2_rows: pl.DataFrame, stage1_preds: pl.DataFrame
) -> Stage2FootballPlayerModel:
    return Stage2FootballPlayerModel().fit(stage2_rows, stage1_preds)


def _future_rows(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.filter(pl.col("game_state") != "FINAL")


# ------------------------------------------------------------------------ features


def test_stage2_feature_contract_is_firewalled_and_clean(stage2_rows: pl.DataFrame) -> None:
    assert_zero_firewall_contamination(STAGE2_FEATURE_COLUMNS)
    assert not set(STAGE2_FEATURE_COLUMNS) & set(STAGE2_TARGET_COLUMNS)
    assert set(STAGE2_FEATURE_COLUMNS) == set(STAGE2_PLAYER_FEATURE_COLUMNS)
    for col in STAGE2_FEATURE_COLUMNS:
        series = stage2_rows[col].cast(pl.Float64)
        assert series.null_count() == 0, col
        assert not series.is_nan().any(), col
    p = stage2_rows["p_active"]
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0  # type: ignore[arg-type]


def test_upcoming_games_cover_all_positions_without_labels(stage2_rows: pl.DataFrame) -> None:
    fut = _future_rows(stage2_rows)
    assert not fut.is_empty()
    assert set(fut["position"].unique().to_list()) == {"QB", "RB", "WR", "TE"}
    assert fut["actual_targets"].null_count() == fut.height
    done = stage2_rows.filter(pl.col("game_state") == "FINAL")
    assert done["actual_targets"].null_count() == 0


def test_stage2_features_future_mutation_invariance(
    silver: SyntheticSilver, player_logs: pl.DataFrame, gold: pl.DataFrame
) -> None:
    base = build_football_stage2_features(
        silver.games, player_logs, silver.team_logs, stage1_df=gold
    )
    target = silver.games.filter(pl.col("game_state") == "FINAL").sort("start_time_utc")["game_id"][
        60
    ]
    t0 = silver.games.filter(pl.col("game_id") == target)["start_time_utc"][0]
    later = pl.col("start_time_utc") >= t0
    mutated_players = player_logs.with_columns(
        *[
            pl.when(later).then(pl.col(c) * 3.0 + 7.0).otherwise(pl.col(c)).alias(c)
            for c in ("pass_attempts", "carries", "targets", "rec_yards", "rush_yards")
        ],
        pl.when(later)
        .then(pl.col("snap_share") * 0.2)
        .otherwise(pl.col("snap_share"))
        .alias("snap_share"),
    )
    mutated_teams = silver.team_logs.with_columns(
        pl.when(later)
        .then(pl.col("off_plays") * 2)
        .otherwise(pl.col("off_plays"))
        .alias("off_plays")
    )
    changed = build_football_stage2_features(
        silver.games, mutated_players, mutated_teams, stage1_df=gold
    )
    cols = [*KEYS, *STAGE2_PLAYER_FEATURE_COLUMNS]
    a = base.filter(pl.col("game_id") == target).select(cols).sort(KEYS)
    b = changed.filter(pl.col("game_id") == target).select(cols).sort(KEYS)
    assert a.height > 0
    assert a.equals(b)


# ---------------------------------------------------------------------- projections


def test_projections_are_valid_and_monotone_in_thresholds(
    stage2_model: Stage2FootballPlayerModel, stage2_rows: pl.DataFrame, stage1_preds: pl.DataFrame
) -> None:
    proj = stage2_model.predict(_future_rows(stage2_rows), stage1_preds)
    assert proj.height > 0
    for col in PROJECTION_COLUMNS:
        assert float(proj[col].min()) >= 0.0, col  # type: ignore[arg-type]
    for col in PROBABILITY_COLUMNS:
        assert float(proj[col].min()) >= 0.0 and float(proj[col].max()) <= 1.0, col  # type: ignore[arg-type]
    assert (proj["prob_100plus_rush_yds"] <= proj["prob_75plus_rush_yds"] + 1e-12).all()
    assert (proj["prob_100plus_rec_yds"] <= proj["prob_75plus_rec_yds"] + 1e-12).all()
    assert (proj["prob_300plus_pass_yds"] <= proj["prob_250plus_pass_yds"] + 1e-12).all()
    # non-QBs never throw; non-RB/QB never carry the ball in this position model
    assert float(proj.filter(pl.col("position") != "QB")["proj_pass_att"].max()) == 0.0  # type: ignore[arg-type]
    assert float(proj.filter(pl.col("position").is_in(["WR", "TE"]))["proj_carries"].max()) == 0.0  # type: ignore[arg-type]


def test_inactive_player_gets_zero_opportunity_and_team_totals_are_conserved(
    stage2_model: Stage2FootballPlayerModel, stage2_rows: pl.DataFrame, stage1_preds: pl.DataFrame
) -> None:
    fut = _future_rows(stage2_rows)
    game = int(fut["game_id"][0])
    team = str(fut.filter(pl.col("game_id") == game)["team"][0])
    rows = fut.filter((pl.col("game_id") == game) & (pl.col("team") == team))
    wr = rows.filter(pl.col("position") == "WR").sort("player_id")["player_id"][0]
    out_rows = rows.with_columns(
        pl.when(pl.col("player_id") == wr).then(0.0).otherwise(pl.col("p_active")).alias("p_active")
    )
    base = stage2_model.predict(rows, stage1_preds)
    proj = stage2_model.predict(out_rows, stage1_preds)
    zeroed = proj.filter(pl.col("player_id") == wr)
    for col in (*PROJECTION_COLUMNS, "prob_anytime_td"):
        assert float(zeroed[col][0]) == 0.0, col

    ctx = stage2_model.project_team_opportunity(rows, stage1_preds)
    pass_att = float(ctx["team_pass_att"][0])
    rush_att = float(ctx["team_rush_att"][0])
    assert float(proj["proj_pass_att"].sum()) == pytest.approx(pass_att, rel=1e-6)
    assert float(proj["proj_targets"].sum()) <= pass_att + 1e-6
    assert float(proj.filter(pl.col("position") == "RB")["proj_carries"].sum()) <= rush_att + 1e-6
    # the missing WR's volume is redistributed, never created
    assert float(proj["proj_targets"].sum()) <= float(base["proj_targets"].sum()) + 1e-6


def test_game_script_and_role_monotonicity(
    stage2_model: Stage2FootballPlayerModel, stage2_rows: pl.DataFrame, stage1_preds: pl.DataFrame
) -> None:
    fut = _future_rows(stage2_rows)
    game = int(fut["game_id"][0])
    home = str(stage1_preds.filter(pl.col("game_id") == game)["home_team"][0])
    rows = fut.filter((pl.col("game_id") == game) & (pl.col("team") == home))

    def shifted(delta: float) -> pl.DataFrame:
        return stage1_preds.with_columns(
            pl.when(pl.col("game_id") == game)
            .then(pl.col("expected_home_score") + delta)
            .otherwise(pl.col("expected_home_score"))
            .alias("expected_home_score")
        )

    lose, win = (stage2_model.predict(rows, shifted(d)) for d in (-8.0, 8.0))

    def total(frame: pl.DataFrame, pos: str, col: str) -> float:
        return float(frame.filter(pl.col("position") == pos)[col].sum())

    assert total(win, "RB", "proj_carries") > total(lose, "RB", "proj_carries")
    assert total(win, "QB", "proj_pass_att") < total(lose, "QB", "proj_pass_att")
    assert float(win["prob_anytime_td"].sum()) > float(lose["prob_anytime_td"].sum())

    wr = rows.filter(pl.col("position") == "WR").sort("player_id")["player_id"][0]
    bumped = rows.with_columns(
        pl.when(pl.col("player_id") == wr)
        .then(pl.col("target_share") + 0.10)
        .otherwise(pl.col("target_share"))
        .alias("target_share")
    )
    before = stage2_model.predict(rows, stage1_preds).filter(pl.col("player_id") == wr)
    after = stage2_model.predict(bumped, stage1_preds).filter(pl.col("player_id") == wr)
    assert float(after["proj_targets"][0]) > float(before["proj_targets"][0])


def test_team_context_and_holdout_evaluation(
    gold: pl.DataFrame, stage2_rows: pl.DataFrame, stage1_preds: pl.DataFrame
) -> None:
    ctx = team_context_from_stage1(stage1_preds)
    assert ctx.height == 2 * stage1_preds.height
    metrics = evaluate_football_stage2_holdout(gold, stage2_rows, n_holdout_seasons=1)
    assert metrics["mae_wr_proj_targets"] < metrics["baseline_mae_wr_proj_targets"] * 1.5
    assert all(np.isfinite(v) for v in metrics.values())


# -------------------------------------------------------------------- bundle + predictor


@pytest.fixture
def seeded_root(tmp_path: Path, silver: SyntheticSilver) -> Path:
    write_synthetic_silver(tmp_path, silver)
    settings = get_settings(root_dir=tmp_path)
    gold = build_and_persist_football_gold_features(settings)
    stage2 = build_and_persist_football_stage2_features(settings, stage1_df=gold)
    assert (settings.paths.gold_dir / "football" / "stage2_features.parquet").exists()
    result = train_football_stage1_bundle(
        gold, settings.paths.artifacts_dir, model_version="t1", stage2_df=stage2
    )
    assert result.stage2_model is not None and result.n_stage2_rows > 0
    return tmp_path


def _predictor(root: Path) -> FootballMatchupPredictor:
    settings = get_settings(root_dir=root)
    return FootballMatchupPredictor.from_artifacts(
        settings.paths.artifacts_dir / "football" / "latest", settings=settings
    )


def test_bundle_round_trip_validates_stage2_schema(seeded_root: Path) -> None:
    settings = get_settings(root_dir=seeded_root)
    bundle = load_football_stage1_bundle(settings.paths.artifacts_dir)
    assert bundle.stage2_model is not None
    stage2 = pl.read_parquet(settings.paths.gold_dir / "football" / "stage2_features.parquet")
    ModelRegistry.validate_features(stage2, bundle.metadata, stage=2)
    assert bundle.metadata.stage2_schema_sha256


def test_predictor_stage1_probabilities_and_ledger(
    seeded_root: Path, silver: SyntheticSilver
) -> None:
    predictor = _predictor(seeded_root)
    upcoming = predictor.upcoming_week()
    assert upcoming is not None
    preds = predictor.predict_week(*upcoming)
    assert len(preds) == silver.games.filter(pl.col("game_state") != "FINAL").height
    for p in preds:
        assert 0.0 < p.home_win_prob < 1.0
        assert p.home_win_prob + p.away_win_prob == pytest.approx(1.0)
        assert p.expected_total == pytest.approx(p.expected_home_score + p.expected_away_score)
        assert p.player_projections is None
        assert not {"STAGE1_ONLY_NO_PLAYER_DATA"} & set(p.warnings)
    ledger = pl.read_parquet(
        get_settings(root_dir=seeded_root).paths.predictions_dir / "football_predictions.parquet"
    )
    assert ledger.height == len(preds)
    assert ledger["model_version"].unique().to_list() == ["t1"]


def test_predictor_player_projections_and_next_games(seeded_root: Path) -> None:
    predictor = _predictor(seeded_root)
    upcoming = predictor.upcoming_week()
    assert upcoming is not None
    pred = predictor.predict_matchup(upcoming[0], upcoming[1], "KC", "BUF", include_players=True)
    assert pred.player_projections is not None
    home, away = pred.player_projections.home, pred.player_projections.away
    assert set(home["team"].unique().to_list()) == {"BUF"}
    assert set(away["team"].unique().to_list()) == {"KC"}
    assert {"QB", "RB", "WR", "TE"} <= set(home["position"].to_list())
    assert float(home["prob_anytime_td"].max()) <= 1.0  # type: ignore[arg-type]
    nxt = predictor.predict_team_next_games("KC", n_games=1, include_players=False)
    assert [g.game_id for g in nxt] == [pred.game_id]
    assert pred.home_qb.team == "BUF" and pred.home_qb.qb_id == qb_ids("BUF")[0]
    player_ledger = pl.read_parquet(
        get_settings(root_dir=seeded_root).paths.predictions_dir
        / "football_player_predictions.parquet"
    )
    assert player_ledger.height == home.height + away.height


def test_predictor_without_stage2_flags_stage1_only(seeded_root: Path) -> None:
    predictor = _predictor(seeded_root)
    predictor.stage2_model = None
    upcoming = predictor.upcoming_week()
    assert upcoming is not None
    pred = predictor.predict_week(
        upcoming[0], upcoming[1], include_players=True, persist_ledger=False
    )[0]
    assert "STAGE1_ONLY_NO_PLAYER_DATA" in pred.warnings
    assert pred.player_projections is None


def test_predictor_rejects_unknown_matchup(seeded_root: Path) -> None:
    predictor = _predictor(seeded_root)
    with pytest.raises(ValueError, match="No scheduled game"):
        predictor.predict_matchup(2023, 1, "KC", "KC")
    with pytest.raises(ValueError, match="AWAY@HOME"):
        predictor.predict_week(2023, 1, matchup_filter="KCBUF")


def test_resolver_starters_drive_qb_status_and_player_rows(
    tmp_path: Path, silver: SyntheticSilver, player_logs: pl.DataFrame
) -> None:
    fut = silver.games.filter(pl.col("game_state") != "FINAL").row(0, named=True)
    home, away, gid = str(fut["home_team"]), str(fut["away_team"]), int(fut["game_id"])
    starter, backup = qb_ids(home)
    seen = fut["start_time_utc"] - timedelta(days=2)

    def snap(team: str, pid: int, source: str, **kw: object) -> dict[str, object]:
        base: dict[str, object] = {
            "as_of_time_utc": seen,
            "game_id": gid,
            "season": fut["season"],
            "week": fut["week"],
            "team": team,
            "player_id": pid,
            "position": "QB",
            "source": source,
            "report_status": None,
            "practice_status": None,
            "depth_rank": None,
            "is_inactive": None,
        }
        base.update(kw)
        return base

    rows = [
        snap(home, starter, "depth_chart", depth_rank=1),
        snap(home, backup, "depth_chart", depth_rank=2),
        snap(home, starter, "injury_report", report_status="Out"),
        snap(away, qb_ids(away)[0], "depth_chart", depth_rank=1),
    ]
    availability = pl.DataFrame(rows, schema=silver.availability.schema)
    write_synthetic_silver(tmp_path, silver, availability=availability, player_logs=player_logs)
    settings = get_settings(root_dir=tmp_path)
    gold = build_and_persist_football_gold_features(settings)
    stage2 = build_and_persist_football_stage2_features(settings, stage1_df=gold)
    train_football_stage1_bundle(gold, settings.paths.artifacts_dir, stage2_df=stage2)
    predictor = _predictor(tmp_path)
    pred = predictor.predict_matchup(
        int(fut["season"]), int(fut["week"]), away, home, include_players=True, persist_ledger=False
    )
    assert pred.home_qb.qb_id == backup  # starter ruled Out -> backup
    assert pred.home_qb.status in {"PROBABLE_QB", "QB_UNCERTAIN"}
    assert pred.away_qb.qb_id == qb_ids(away)[0]
    assert pred.player_projections is not None
    qb_rows = pred.player_projections.home.filter(pl.col("position") == "QB")
    by_id = {int(r["player_id"]): float(r["proj_pass_att"]) for r in qb_rows.iter_rows(named=True)}
    assert by_id.get(starter, 0.0) == 0.0
    assert by_id[backup] > 15.0


def test_predictions_for_completed_games_ignore_later_data_and_flag_backups(
    tmp_path: Path,
) -> None:
    backup_game = (2022, 3, "BUF")
    base = build_synthetic_silver(future_weeks=1, backup_starts=frozenset({backup_game}))
    roots = {}
    for name, mutate in (("clean", False), ("mutated", True)):
        root = tmp_path / name
        games, team_logs, qb_logs, players = (
            base.games,
            base.team_logs,
            base.qb_logs,
            build_synthetic_player_logs(base),
        )
        if mutate:
            cut = base.games.filter((pl.col("season") == 2022) & (pl.col("week") == 3))[
                "start_time_utc"
            ].max()
            later = pl.col("start_time_utc") > cut
            games = games.with_columns(
                pl.when(later)
                .then(pl.col("home_score") + 17)
                .otherwise(pl.col("home_score"))
                .alias("home_score")
            )
            team_logs = team_logs.with_columns(
                pl.when(later)
                .then(pl.col("off_plays") + 30)
                .otherwise(pl.col("off_plays"))
                .alias("off_plays")
            )
        silver = SyntheticSilver(games, team_logs, qb_logs, players, base.availability)
        write_synthetic_silver(root, silver, player_logs=players)
        settings = get_settings(root_dir=root)
        gold = build_and_persist_football_gold_features(settings)
        train_football_stage1_bundle(gold, settings.paths.artifacts_dir, stage2_df=None)
        roots[name] = _predictor(root)

    season, week, backup_team = backup_game
    game = base.games.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & ((pl.col("home_team") == backup_team) | (pl.col("away_team") == backup_team))
    ).row(0, named=True)
    home = str(game["home_team"])
    clean = roots["clean"].gold_features().filter(pl.col("game_id") == game["game_id"])
    mutated = roots["mutated"].gold_features().filter(pl.col("game_id") == game["game_id"])
    assert clean.equals(mutated)  # Gold for game T is blind to anything after T

    pred = roots["clean"].predict_matchup(
        season, week, str(game["away_team"]), home, persist_ledger=False
    )
    assert "BACKUP_QB" in pred.warnings
    side = pred.home_qb if home == backup_team else pred.away_qb
    assert side.is_backup and side.status == "CONFIRMED_QB"
    assert pred.game_state == "FINAL"
    assert pred.start_time_utc.astimezone(UTC) == game["start_time_utc"]


def test_ledger_scoring_matches_completed_games(seeded_root: Path) -> None:
    predictor = _predictor(seeded_root)
    done = predictor._load_silver().games.filter(pl.col("game_state") == "FINAL")
    row = done.sort("start_time_utc").tail(1).row(0, named=True)
    predictor.predict_matchup(
        int(row["season"]),
        int(row["week"]),
        str(row["away_team"]),
        str(row["home_team"]),
        include_players=True,
    )
    score = score_football_prediction_ledger(get_settings(root_dir=seeded_root))
    assert score.n_games == 1
    assert score.n_player_rows > 0
    assert all(np.isfinite(v) for v in score.player_mae.values())
    assert 0.0 <= score.report.model.brier_score <= 1.0
