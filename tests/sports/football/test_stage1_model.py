"""Stage 1 football tests: probabilities, ties, walk-forward ordering, baselines, bundles."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sports_ml.config import RANDOM_SEED
from sports_ml.core.calibration import BinaryProbabilityCalibrator
from sports_ml.core.registry import SchemaMismatchError
from sports_ml.sports.football.constants import FOOTBALL_ELO_CONFIG
from sports_ml.sports.football.features import (
    STAGE1_FEATURE_COLUMNS,
    build_football_stage1_features,
)
from sports_ml.sports.football.models import (
    Stage1FootballGameModel,
    load_football_stage1_bundle,
    run_football_stage1_walk_forward_backtest,
    swap_home_away_features,
    train_football_stage1_bundle,
)
from tests.sports.football.synthetic import build_synthetic_silver

WEEK_SIZES = [15] * 18 + [6, 4, 2, 1]
PLAYOFF_TYPES = ["REG"] * 18 + ["WC", "DIV", "CON", "SB"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def synthetic_gold(n_seasons: int = 8, *, seed: int = RANDOM_SEED) -> pl.DataFrame:
    """Directional synthetic Gold matrix: noisy views of a latent strength gap drive outcomes."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for si in range(n_seasons):
        season = 2012 + si
        base = datetime(season, 9, 8, 17, 0, tzinfo=UTC)
        gid = 0
        for wi, (n_games, stype) in enumerate(zip(WEEK_SIZES, PLAYOFF_TYPES, strict=True)):
            week = wi + 1
            for g in range(n_games):
                gid += 1
                neutral = stype == "SB"
                d = float(rng.normal())
                hfa = 0.0 if neutral else 2.0
                margin = hfa + 6.5 * d + float(rng.normal(0.0, 13.5))
                hs = max(0, round(22.0 + margin / 2.0 + float(rng.normal(0.0, 5.0))))
                as_ = max(0, round(22.0 - margin / 2.0 + float(rng.normal(0.0, 5.0))))
                tie = hs == as_
                elo_diff = 30.0 * d + float(rng.normal(0.0, 90.0))
                hfa_elo = 0.0 if neutral else 48.0
                qb_diff = elo_diff + float(rng.normal(0.0, 25.0))
                rows.append(
                    {
                        "game_id": season * 10_000 + week * 100 + g,
                        "event_seq": 1,
                        "league": "NFL",
                        "season": season,
                        "season_type": stype,
                        "week": week,
                        "start_time_utc": base + timedelta(days=7 * wi, hours=g % 4),
                        "home_team": "KC",
                        "away_team": "BUF",
                        "game_state": "FINAL",
                        "home_score": hs,
                        "away_score": as_,
                        "home_win": None if tie else int(hs > as_),
                        "is_tie": int(tie),
                        "is_neutral_site": int(neutral),
                        "diff_elo_pre": elo_diff,
                        "qb_elo_diff_pre": qb_diff + hfa_elo,
                        "elo_home_win_prob": float(
                            _sigmoid(np.array((elo_diff + hfa_elo) / 173.7))
                        ),
                        "qb_elo_home_win_prob": float(
                            _sigmoid(np.array((qb_diff + hfa_elo) / 173.7))
                        ),
                        "diff_off_epa_per_play_std": 0.06 * d + float(rng.normal(0.0, 0.06)),
                        "diff_def_epa_per_play_std": -0.05 * d + float(rng.normal(0.0, 0.06)),
                        "diff_adj_epa_net": 0.08 * d + float(rng.normal(0.0, 0.07)),
                        "diff_qb_value": 0.05 * d + float(rng.normal(0.0, 0.05)),
                        "diff_qb_epa_per_dropback": 0.05 * d + float(rng.normal(0.0, 0.05)),
                        "home_qb_value": 0.02 + 0.02 * d,
                        "away_qb_value": 0.02 - 0.02 * d,
                        "home_days_rest": 7.0,
                        "away_days_rest": 7.0,
                        "is_playoff": int(stype != "REG"),
                    }
                )
    return pl.DataFrame(rows).with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int16)
    )


@pytest.fixture(scope="module")
def gold() -> pl.DataFrame:
    return synthetic_gold()


@pytest.fixture(scope="module")
def fitted(gold: pl.DataFrame) -> Stage1FootballGameModel:
    return Stage1FootballGameModel().fit(gold.filter(pl.col("season") < 2018))


def test_probability_and_score_contracts(
    gold: pl.DataFrame, fitted: Stage1FootballGameModel
) -> None:
    preds = fitted.predict(gold.filter(pl.col("season") >= 2018))
    p = preds["home_win_prob"].to_numpy()
    assert preds.height > 400
    assert ((p > 0.0) & (p < 1.0)).all()
    assert np.abs(preds["away_win_prob"].to_numpy() - (1.0 - p)).max() < 1e-9
    assert (preds["draw_prob"].to_numpy() == 0.0).all()
    for col in ("expected_home_score", "expected_away_score", "expected_total"):
        v = preds[col].to_numpy()
        assert np.isfinite(v).all() and (v >= 0.0).all()
    assert np.allclose(
        preds["expected_margin"].to_numpy(),
        preds["expected_home_score"].to_numpy() - preds["expected_away_score"].to_numpy(),
    )
    assert np.allclose(
        preds["expected_total"].to_numpy(),
        preds["expected_home_score"].to_numpy() + preds["expected_away_score"].to_numpy(),
    )
    assert 11.0 <= fitted.sigma_margin <= 16.0
    assert 20.0 < float(preds["expected_total"].to_numpy().mean()) / 2.0 < 25.0  # league anchor
    with pytest.raises(RuntimeError):
        Stage1FootballGameModel().predict(gold)


def test_neutral_site_games_are_symmetric_under_relabeling(
    gold: pl.DataFrame, fitted: Stage1FootballGameModel
) -> None:
    sample = (
        gold.filter(pl.col("season") == 2018)
        .head(60)
        .with_columns(pl.lit(1).alias("is_neutral_site"))
    )
    swapped = swap_home_away_features(sample, fitted.feature_columns)
    a = fitted.predict(sample).sort("game_id")
    b = fitted.predict(swapped).sort("game_id")
    assert np.allclose(
        a["home_win_prob"].to_numpy(), 1.0 - b["home_win_prob"].to_numpy(), atol=1e-9
    )
    assert np.allclose(
        a["expected_home_score"].to_numpy(), b["expected_away_score"].to_numpy(), atol=1e-9
    )
    assert np.allclose(a["expected_margin"].to_numpy(), -b["expected_margin"].to_numpy(), atol=1e-9)

    # Home-field games are *not* forced symmetric (the model keeps its home edge)
    home_field = sample.with_columns(pl.lit(0).alias("is_neutral_site"))
    c = fitted.predict(home_field).sort("game_id")
    d = fitted.predict(swap_home_away_features(home_field, fitted.feature_columns)).sort("game_id")
    assert not np.allclose(
        c["home_win_prob"].to_numpy(), 1.0 - d["home_win_prob"].to_numpy(), atol=1e-6
    )


def test_expected_margin_is_monotone_in_qb_value_gap(
    gold: pl.DataFrame, fitted: Stage1FootballGameModel
) -> None:
    base = gold.filter(pl.col("season") == 2018).head(1)
    grid = pl.concat(
        [
            base.with_columns(
                pl.lit(v).alias("diff_qb_value"), pl.lit(i).alias("game_id")
            ).with_columns(pl.col("game_id").cast(pl.Int64))
            for i, v in enumerate(np.linspace(-0.4, 0.4, 25))
        ]
    )
    out = fitted.predict(grid).sort("game_id")
    assert (np.diff(out["expected_margin"].to_numpy()) >= -1e-9).all()
    assert (np.diff(out["expected_home_score"].to_numpy()) >= -1e-9).all()
    assert (np.diff(out["expected_away_score"].to_numpy()) <= 1e-9).all()


def test_ties_are_excluded_from_classifier_and_calibration_but_kept_for_scores(
    gold: pl.DataFrame,
) -> None:
    train = gold.filter(pl.col("season") < 2018)
    n_ties = int(train["is_tie"].sum())
    assert n_ties > 0
    no_ties = train.filter(pl.col("is_tie") == 0)

    with_ties_model = Stage1FootballGameModel().fit(train)
    no_ties_model = Stage1FootballGameModel().fit(no_ties)

    # Classifier and calibrator are fit on the identical non-tie rows
    assert with_ties_model._logit_clf is not None and no_ties_model._logit_clf is not None
    assert np.allclose(with_ties_model._logit_clf.coef_, no_ties_model._logit_clf.coef_)
    cal_rows = train.filter(pl.col("season") >= 2016)
    assert with_ties_model.calibrator.n_fit_samples == cal_rows.filter(pl.col("is_tie") == 0).height
    assert Stage1FootballGameModel.classifier_mask(train) is not None
    assert (
        train.filter(Stage1FootballGameModel.classifier_mask(train)).height == train.height - n_ties
    )

    # Ties still inform the score regressors
    probe = gold.filter(pl.col("season") == 2018).head(20)
    a = with_ties_model.predict(probe)["expected_total"].to_numpy()
    b = no_ties_model.predict(probe)["expected_total"].to_numpy()
    assert not np.allclose(a, b)


def test_ties_update_elo_at_half_point() -> None:
    s = build_synthetic_silver(seasons=(2021,))
    first = s.games.sort("start_time_utc").row(0, named=True)
    tied = s.games.with_columns(
        pl.when(pl.col("game_id") == first["game_id"])
        .then(20)
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when(pl.col("game_id") == first["game_id"])
        .then(20)
        .otherwise(pl.col("away_score"))
        .alias("away_score"),
        pl.when(pl.col("game_id") == first["game_id"])
        .then(None)
        .otherwise(pl.col("home_win"))
        .alias("home_win"),
        pl.when(pl.col("game_id") == first["game_id"])
        .then(1)
        .otherwise(pl.col("is_tie"))
        .alias("is_tie"),
    )
    feats = build_football_stage1_features(tied, s.team_logs, s.qb_logs).sort("start_time_utc")
    row = feats.row(0, named=True)
    assert row["is_tie"] == 1 and row["home_win"] is None
    expected_shift = FOOTBALL_ELO_CONFIG.k_factor * (0.5 - row["elo_home_win_prob"])
    home = first["home_team"]
    nxt = feats.filter(
        (pl.col("start_time_utc") > first["start_time_utc"])
        & ((pl.col("home_team") == home) | (pl.col("away_team") == home))
    ).row(0, named=True)
    after = nxt["elo_home_pre"] if nxt["home_team"] == home else nxt["elo_away_pre"]
    assert after == pytest.approx(1500.0 + expected_shift)
    assert expected_shift < 0  # the home favorite lost ground on a tie


def test_walk_forward_ordering_and_no_future_training(gold: pl.DataFrame) -> None:
    result = run_football_stage1_walk_forward_backtest(gold, min_train_seasons=3, n_bootstrap=50)
    assert len(result.fold_audits) == 5
    for audit in result.fold_audits:
        assert audit.gbdt_train_max_time < audit.calibrator_min_time
        assert audit.calibrator_min_time <= audit.calibrator_max_time
        assert audit.calibrator_max_time < audit.val_min_time
    oof = result.oof_predictions
    first_test = int(oof.select(pl.col("season").min()).item())
    assert first_test == 2015
    # Season s is only ever predicted by a model fit on seasons < s
    for fold, audit in enumerate(result.fold_audits):
        seasons_in_fold = oof.filter(pl.col("fold_index") == fold)["season"].unique().to_list()
        assert len(seasons_in_fold) == 1
        assert audit.gbdt_train_max_time.year <= seasons_in_fold[0]
        fold_start = (
            oof.filter(pl.col("fold_index") == fold).select(pl.col("start_time_utc").min()).item()
        )
        assert audit.calibrator_max_time < fold_start

    blocks = run_football_stage1_walk_forward_backtest(
        gold.filter(pl.col("season") <= 2016),
        min_train_seasons=3,
        test_unit="week_block",
        week_block_size=8,
        min_calibration_samples=200,
        n_bootstrap=20,
    )
    assert len(blocks.fold_audits) >= 3
    for audit in blocks.fold_audits:
        assert audit.gbdt_train_max_time < audit.calibrator_min_time
        assert audit.calibrator_max_time < audit.val_min_time

    with pytest.raises(ValueError, match="Temporal leakage"):
        Stage1FootballGameModel().fit(
            gold.filter(pl.col("season") < 2018), gold.filter(pl.col("season") == 2016)
        )
    with pytest.raises(ValueError, match="more than"):
        run_football_stage1_walk_forward_backtest(
            gold.filter(pl.col("season") <= 2013), min_train_seasons=3
        )


def test_calibrated_model_beats_home_prior_and_elo_baselines(gold: pl.DataFrame) -> None:
    result = run_football_stage1_walk_forward_backtest(gold, min_train_seasons=3, n_bootstrap=300)
    report = result.benchmark_report
    assert report.model.log_loss < report.home_win_rate_baseline.log_loss
    assert report.model.log_loss < report.coin_flip_baseline.log_loss
    assert report.elo_baseline is not None
    assert report.model.log_loss < report.elo_baseline.log_loss
    assert report.model.brier_score < report.elo_baseline.brier_score
    assert report.model.ece < 0.08
    assert report.beats_home_baseline_brier

    ci_prior, ci_elo = result.log_loss_vs_home_prior, result.log_loss_vs_qb_elo
    assert ci_elo is not None
    for ci in (ci_prior, ci_elo):
        assert ci.n_units == 5
        assert ci.lower <= ci.upper
        assert ci.mean < 0.0
    assert result.plain_elo_metrics is not None

    sm = result.score_metrics
    assert sm.margin_mae < sm.baseline_margin_mae
    assert sm.total_mae > 0.0 and sm.home_rmse >= sm.home_mae

    assert set(result.slices) == {"playoffs", "weeks_1_4"}
    assert result.slices["playoffs"].n_games >= 5
    assert result.slices["weeks_1_4"].n_games > 100
    assert set(result.era_home_win_rates) == {"2011-2019"}
    assert all(0.4 < r < 0.8 for r in result.era_home_win_rates.values())
    # Tie rows are scored for expected scores but never counted as probability targets
    oof = result.oof_predictions
    assert oof.filter(pl.col("is_tie") == 1).height > 0
    assert result.score_metrics.n_games == oof.height


def test_calibration_size_guard_falls_back_to_identity(gold: pl.DataFrame) -> None:
    train = gold.filter(pl.col("season") < 2018)
    guarded = Stage1FootballGameModel(min_calibration_samples=100_000).fit(train)
    assert guarded.calibrator.is_identity
    assert guarded.calibrator.fallback_reason == "insufficient_samples"
    preds = guarded.predict(
        gold.filter(pl.col("season") == 2018).with_columns(pl.lit(0).alias("is_neutral_site"))
    )
    assert np.allclose(preds["home_win_prob"].to_numpy(), preds["raw_home_win_prob"].to_numpy())

    enough = Stage1FootballGameModel(min_calibration_samples=400).fit(train)
    assert enough.calibrator.n_fit_samples >= 400
    assert enough.calibrator.fallback_reason in (None, "non_positive_slope")

    backtest = run_football_stage1_walk_forward_backtest(
        gold, min_train_seasons=3, min_calibration_samples=100_000, n_bootstrap=10
    )
    assert backtest.calibration_identity_folds == [0, 1, 2, 3, 4]


def test_binary_calibrator_platt_temperature_and_guards() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    true_logit = rng.normal(0.0, 1.0, 3000)
    y = (rng.random(3000) < _sigmoid(true_logit)).astype(np.int64)
    overconfident = _sigmoid(2.5 * true_logit)

    def logloss(p: np.ndarray) -> float:
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    for method in ("platt", "temperature"):
        cal = BinaryProbabilityCalibrator(method=method, min_samples=400).fit(overconfident, y)
        out = cal.predict_proba(overconfident)
        assert not cal.is_identity
        assert logloss(out) < logloss(overconfident)
        assert ((out > 0.0) & (out < 1.0)).all()
    temp = BinaryProbabilityCalibrator(method="temperature").fit(overconfident, y)
    assert temp.temperature == pytest.approx(2.5, rel=0.2)

    small = BinaryProbabilityCalibrator(min_samples=400).fit(overconfident[:100], y[:100])
    assert small.is_identity and small.fallback_reason == "insufficient_samples"
    assert np.allclose(small.predict_proba(overconfident[:5]), overconfident[:5])
    one_class = BinaryProbabilityCalibrator(min_samples=10).fit(
        overconfident[:50], np.ones(50, dtype=np.int64)
    )
    assert one_class.fallback_reason == "single_class"
    inverted = BinaryProbabilityCalibrator(min_samples=100).fit(1.0 - overconfident, y)
    assert inverted.fallback_reason == "non_positive_slope"
    with pytest.raises(RuntimeError):
        BinaryProbabilityCalibrator().predict_proba(overconfident)
    with pytest.raises(ValueError):
        BinaryProbabilityCalibrator(min_samples=1)


def test_bundle_training_persists_fingerprinted_metadata(
    tmp_path: Path, gold: pl.DataFrame
) -> None:
    from sports_ml.core.registry import ModelRegistry

    features = gold.filter(pl.col("season") <= 2016)
    result = train_football_stage1_bundle(features, tmp_path / "artifacts", model_version="v1")
    assert result.n_training_games == features.height
    assert set(result.validation_metrics) >= {"log_loss", "brier_score", "ece", "accuracy"}
    assert (tmp_path / "artifacts" / "football" / "v1" / "metadata.json").exists()
    assert (tmp_path / "artifacts" / "football" / "latest" / "stage1_model.pkl").exists()

    loaded = load_football_stage1_bundle(tmp_path / "artifacts")
    meta = loaded.metadata
    assert meta.sport == "football"
    assert len(meta.schema_sha256) == 64
    assert meta.stage1_feature_columns == result.model.feature_columns
    ModelRegistry.validate_features(features, meta)

    original = result.model.predict(gold.filter(pl.col("season") == 2017))
    restored = loaded.stage1_model.predict(gold.filter(pl.col("season") == 2017))
    assert np.allclose(original["home_win_prob"].to_numpy(), restored["home_win_prob"].to_numpy())

    with pytest.raises(SchemaMismatchError):
        ModelRegistry.validate_features(features.drop(meta.stage1_feature_columns[0]), meta)
    with pytest.raises(SchemaMismatchError):
        ModelRegistry.validate_features(
            features.with_columns(pl.col(meta.stage1_feature_columns[0]).cast(pl.Float32)), meta
        )
    with pytest.raises(FileNotFoundError):
        load_football_stage1_bundle(tmp_path / "artifacts", "missing")
    with pytest.raises(ValueError, match="No completed"):
        train_football_stage1_bundle(
            features.with_columns(pl.lit(None, dtype=pl.Int64).alias("home_score")),
            tmp_path / "x",
        )

    # Short histories fall back to a chronological holdout for validation metrics
    short = train_football_stage1_bundle(
        gold.filter(pl.col("season") <= 2014), tmp_path / "short", model_version="v0"
    )
    assert short.validation_metrics["log_loss"] > 0.0


def test_stage1_runs_on_real_pipeline_features_including_upcoming_games() -> None:
    s = build_synthetic_silver(future_weeks=1)
    features = build_football_stage1_features(s.games, s.team_logs, s.qb_logs)
    model = Stage1FootballGameModel(min_calibration_samples=50).fit(
        features.filter(pl.col("game_state") == "FINAL")
    )
    assert set(model.feature_columns).issubset(STAGE1_FEATURE_COLUMNS)
    preds = model.predict(features)
    upcoming = preds.filter(pl.col("game_state") == "FUT")
    assert upcoming.height > 0
    p = preds["home_win_prob"].to_numpy()
    assert ((p > 0) & (p < 1)).all() and np.isfinite(preds["expected_total"].to_numpy()).all()
    with pytest.raises(ValueError, match="empty"):
        Stage1FootballGameModel().fit(features.head(0))
    with pytest.raises(ValueError, match="No Stage 1"):
        Stage1FootballGameModel.resolve_feature_columns(features.select("game_id"))
