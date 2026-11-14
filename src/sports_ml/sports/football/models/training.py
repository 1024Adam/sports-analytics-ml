"""Football Stage 1 model-bundle training and persistence (`models/training.py`).

Fits the production `Stage1FootballGameModel` on all completed games and saves it with a
`metadata.json` SHA-256 schema fingerprint under `artifacts/football/<version>/` (and `latest/`)
through the shared `ModelRegistry`, consistent with the other sports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from sports_ml.config import RANDOM_SEED
from sports_ml.core.evaluation import evaluate_probabilities
from sports_ml.core.registry import LoadedModelBundle, ModelRegistry
from sports_ml.core.temporal import sort_events_chronologically
from sports_ml.sports.football.models.stage1_game import (
    Stage1FootballGameModel,
    run_football_stage1_walk_forward_backtest,
)
from sports_ml.sports.football.models.stage2_player import Stage2FootballPlayerModel

FOOTBALL_SPORT_SLUG = "football"
MIN_BACKTEST_SEASONS = 4


@dataclass(frozen=True)
class FootballTrainingResult:
    """Outcome of `train_football_stage1_bundle`."""

    model: Stage1FootballGameModel
    bundle_dir: Path
    validation_metrics: dict[str, float]
    n_training_games: int
    stage2_model: Stage2FootballPlayerModel | None = None
    n_stage2_rows: int = 0


def _completed_games(features_df: pl.DataFrame) -> pl.DataFrame:
    return sort_events_chronologically(
        features_df.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    )


def compute_validation_metrics(
    completed: pl.DataFrame, *, random_seed: int = RANDOM_SEED
) -> dict[str, float]:
    """Out-of-sample metrics: walk-forward backtest when possible, else a chronological holdout."""
    n_seasons = completed["season"].n_unique()
    if n_seasons >= MIN_BACKTEST_SEASONS:
        result = run_football_stage1_walk_forward_backtest(
            completed,
            min_train_seasons=n_seasons - 1,
            n_bootstrap=200,
            random_seed=random_seed,
        )
        m = result.benchmark_report.model
        return {
            "log_loss": m.log_loss,
            "brier_score": m.brier_score,
            "ece": m.ece,
            "accuracy": m.accuracy,
            "score_margin_mae": result.score_metrics.margin_mae,
            "score_total_mae": result.score_metrics.total_mae,
        }

    times = completed["start_time_utc"].unique().sort().to_list()
    if len(times) < 10:
        raise ValueError("Not enough completed games to compute validation metrics.")
    cutoff = times[int(len(times) * 0.8)]
    train = completed.filter(pl.col("start_time_utc") < cutoff)
    test = completed.filter(pl.col("start_time_utc") >= cutoff)
    preds = Stage1FootballGameModel(random_seed=random_seed).fit(train).predict(test)
    clf = preds.filter(Stage1FootballGameModel.classifier_mask(preds))
    if clf.is_empty():
        raise ValueError("Holdout has no non-tie games to evaluate.")
    metrics = evaluate_probabilities(
        np.asarray(clf["home_win_prob"].to_numpy(), dtype=np.float64),
        Stage1FootballGameModel.binary_target(clf),
    )
    return {
        "log_loss": metrics.log_loss,
        "brier_score": metrics.brier_score,
        "ece": metrics.ece,
        "accuracy": metrics.accuracy,
    }


def train_football_stage1_bundle(
    features_df: pl.DataFrame,
    artifacts_dir: Path,
    *,
    model_version: str = "v1",
    stage2_df: pl.DataFrame | None = None,
    random_seed: int = RANDOM_SEED,
) -> FootballTrainingResult:
    """Fit Stage 1 (and optional Stage 2 from `stage2_df`) on completed games and persist a bundle.

    Stage 2 is conditioned on in-sample Stage 1 expected scores (documented mild optimism).
    """
    completed = _completed_games(features_df)
    if completed.is_empty():
        raise ValueError("No completed football games available to train on.")
    metrics = compute_validation_metrics(completed, random_seed=random_seed)
    model = Stage1FootballGameModel(random_seed=random_seed).fit(completed)
    stage2_model: Stage2FootballPlayerModel | None = None
    stage2_train: pl.DataFrame | None = None
    if stage2_df is not None and not stage2_df.is_empty():
        stage2_train = stage2_df.filter(pl.col("actual_targets").is_not_null())
        if not stage2_train.is_empty():
            stage2_model = Stage2FootballPlayerModel(random_seed=random_seed).fit(
                stage2_train, model.predict(completed)
            )
    bundle_dir = ModelRegistry(artifacts_dir=artifacts_dir).save_bundle(
        sport=FOOTBALL_SPORT_SLUG,
        model_version=model_version,
        stage1_model=model,
        stage1_train_df=completed,
        stage1_feature_columns=model.feature_columns,
        validation_metrics=metrics,
        stage2_model=stage2_model,
        stage2_train_df=stage2_train if stage2_model is not None else None,
        stage2_feature_columns=stage2_model.feature_columns if stage2_model else None,
        update_latest=True,
    )
    return FootballTrainingResult(
        model=model,
        bundle_dir=bundle_dir,
        validation_metrics=metrics,
        n_training_games=completed.height,
        stage2_model=stage2_model,
        n_stage2_rows=stage2_train.height
        if stage2_model is not None and stage2_train is not None
        else 0,
    )


def evaluate_football_stage2_holdout(
    features_df: pl.DataFrame,
    stage2_df: pl.DataFrame,
    *,
    n_holdout_seasons: int = 1,
    random_seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Fit Stage 1 + Stage 2 on earlier seasons, then score the newest `n_holdout_seasons`.

    The holdout is the newest `n_holdout_seasons` that have completed player rows. Stage 2 trains
    against in-sample Stage 1 scores but is evaluated against the out-of-sample Stage 1 scores
    of the held-out seasons (MAE vs. a position-mean baseline).
    """
    completed = _completed_games(features_df)
    done = stage2_df.filter(pl.col("actual_targets").is_not_null())
    seasons = sorted(done["season"].unique().to_list())
    if len(seasons) <= n_holdout_seasons:
        raise ValueError("Need more seasons with player rows than held-out seasons for Stage 2.")
    cutoff = seasons[-n_holdout_seasons]
    train_games = completed.filter(pl.col("season") < cutoff)
    stage1 = Stage1FootballGameModel(random_seed=random_seed).fit(train_games)
    train_rows = done.filter(pl.col("season") < cutoff)
    test_rows = done.filter(pl.col("season") >= cutoff)
    if train_rows.is_empty() or test_rows.is_empty():
        raise ValueError("Stage 2 backtest needs player rows on both sides of the holdout split.")
    stage2 = Stage2FootballPlayerModel(random_seed=random_seed).fit(
        train_rows, stage1.predict(train_games)
    )
    return stage2.evaluate(test_rows, stage1.predict(completed.filter(pl.col("season") >= cutoff)))


def load_football_stage1_bundle(
    artifacts_dir: Path, model_version: str = "latest"
) -> LoadedModelBundle:
    """Load a saved football bundle (`artifacts/football/<version>/`)."""
    return ModelRegistry.load_bundle(artifacts_dir / FOOTBALL_SPORT_SLUG / model_version)
