"""Temporal firewall and future-mutation invariance tests (`test_temporal_leakage.py`)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.config import RANDOM_SEED, Settings
from sports_ml.core.contracts import (
    CANONICAL_EVENT_KEY_COLS,
    CANONICAL_PARTICIPANT_KEY_COLS,
    CANONICAL_TEMPORAL_SORT_COLS,
    SchemaValidationError,
    validate_canonical_event_frame,
    validate_canonical_participant_frame,
)
from sports_ml.core.shrinkage import (
    bayesian_shrinkage,
    compute_regressed_prior,
    shrink_scalar,
)
from sports_ml.core.temporal import (
    pregame_ewma,
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_rolling_sum,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.hockey.adapter import HockeyAdapter


def _build_temporal_features(logs: pl.DataFrame) -> pl.DataFrame:
    """Apply all core temporal and shrinkage helpers after strict chronological sort."""
    sorted_logs = sort_events_chronologically(logs, extra_tiebreakers=["team"])
    with_counts = sorted_logs.with_columns(
        pregame_rolling_mean("goals_for", window=3, group_by="team").alias("gf_roll_mean_3"),
        pregame_rolling_sum("goals_for", window=3, group_by="team").alias("gf_roll_sum_3"),
        pregame_ewma("xgf_5v5", alpha=0.25, group_by="team").alias("xgf_ewma"),
        pregame_exposure_rate(
            "xgf_5v5",
            "toi_5v5_minutes",
            window=3,
            group_by="team",
            scale_factor=60.0,
        ).alias("xgf_per_60_3"),
        pregame_sample_count("xgf_5v5", group_by="team").alias("n_prior_games"),
    )
    return with_counts.with_columns(
        bayesian_shrinkage(
            current_rate_col="xgf_per_60_3",
            sample_size_col="n_prior_games",
            prior_col=2.50,
            pseudo_count_k=8.0,
        ).alias("xgf_per_60_shrunk"),
    )


def test_framework_seed_and_canonical_keys(
    temp_settings: Settings,
    sample_chronological_logs: pl.DataFrame,
) -> None:
    """Verify deterministic seed configuration and canonical temporal sort columns."""
    assert RANDOM_SEED == 42
    assert temp_settings.random_seed == 42
    assert temp_settings.paths.silver_dir.exists()
    assert CANONICAL_TEMPORAL_SORT_COLS == ("start_time_utc", "event_seq", "event_id")
    assert "event_id" in CANONICAL_EVENT_KEY_COLS
    assert "participant_id" in CANONICAL_PARTICIPANT_KEY_COLS
    assert set(CANONICAL_TEMPORAL_SORT_COLS).issubset(set(sample_chronological_logs.columns))


def test_future_mutation_invariance(sample_chronological_logs: pl.DataFrame) -> None:
    """Mutating events at >= cutoff_time must produce zero change in features at <= cutoff_time."""
    baseline_features = _build_temporal_features(sample_chronological_logs)

    unique_times = baseline_features["start_time_utc"].unique().sort()
    cutoff_time = unique_times[len(unique_times) // 2]

    corrupted_logs = sample_chronological_logs.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_time)
        .then(pl.lit(999.0))
        .otherwise(pl.col(col_name))
        .alias(col_name)
        for col_name in ["goals_for", "goals_against", "xgf_5v5", "xga_5v5", "toi_5v5_minutes"]
    )

    mutated_features = _build_temporal_features(corrupted_logs)

    feature_cols = [
        "sport",
        "season",
        "event_id",
        "event_seq",
        "start_time_utc",
        "team",
        "gf_roll_mean_3",
        "gf_roll_sum_3",
        "xgf_ewma",
        "xgf_per_60_3",
        "n_prior_games",
        "xgf_per_60_shrunk",
    ]
    past_baseline = baseline_features.filter(pl.col("start_time_utc") <= cutoff_time).select(
        feature_cols
    )
    past_mutated = mutated_features.filter(pl.col("start_time_utc") <= cutoff_time).select(
        feature_cols
    )

    assert_frame_equal(past_baseline, past_mutated)


def test_same_day_doubleheader_and_multi_round_leakage_invariance() -> None:
    """Mutating Game 2 (event_seq=2) must never leak into Game 1 (event_seq=1) on the same day."""
    doubleheader_logs = pl.DataFrame(
        {
            "sport": ["baseball"] * 4,
            "season": [2025] * 4,
            "event_id": [1001, 1002, 1003, 1004],
            "event_seq": [1, 1, 2, 1],
            "game_date": [
                date(2025, 6, 1),
                date(2025, 6, 5),
                date(2025, 6, 5),
                date(2025, 6, 6),
            ],
            "start_time_utc": [
                datetime(2025, 6, 1, 17, 0, tzinfo=UTC),
                datetime(2025, 6, 5, 17, 0, tzinfo=UTC),
                datetime(2025, 6, 5, 17, 0, tzinfo=UTC),  # Same timestamp, ordered by event_seq=2
                datetime(2025, 6, 6, 23, 0, tzinfo=UTC),
            ],
            "team": ["NYY", "NYY", "NYY", "NYY"],
            "earned_runs": [2.0, 3.0, 1.0, 4.0],
            "innings_pitched": [9.0, 9.0, 9.0, 9.0],
        }
    )

    def _compute_era(df: pl.DataFrame) -> pl.DataFrame:
        sorted_df = sort_events_chronologically(df)
        return sorted_df.with_columns(
            pregame_exposure_rate(
                "earned_runs",
                "innings_pitched",
                window=2,
                group_by="team",
                scale_factor=9.0,
            ).alias("pregame_era_2")
        )

    base = _compute_era(doubleheader_logs)

    # Corrupt Game 2 of the doubleheader (event_seq=2 on 2025-06-05) and later games
    corrupted = doubleheader_logs.with_columns(
        pl.when((pl.col("game_date") == date(2025, 6, 5)) & (pl.col("event_seq") == 2))
        .then(pl.lit(50.0))
        .otherwise(pl.col("earned_runs"))
        .alias("earned_runs")
    )
    mutated = _compute_era(corrupted)

    # Game 1 of doubleheader (event_id=1002) and Game 2 pre-game feature (event_id=1003) must match
    assert_frame_equal(
        base.filter(pl.col("event_id").is_in([1001, 1002, 1003])).select(
            ["event_id", "event_seq", "pregame_era_2"]
        ),
        mutated.filter(pl.col("event_id").is_in([1001, 1002, 1003])).select(
            ["event_id", "event_seq", "pregame_era_2"]
        ),
    )


def test_cold_start_and_bayesian_shrinkage() -> None:
    """Verify Game 1 (n=0) and rookie debut evaluate cleanly to prior with zero NaNs."""
    df = (
        pl.DataFrame(
            {
                "player_id": [1, 1, 2],
                "prev_season_rate": [1.20, 1.20, None],  # Player 2 is a rookie with no prior season
                "league_mean": [1.00, 1.00, 1.00],
                "curr_rate": [None, 2.00, None],
                "n_games": [0, 8, 0],
            }
        )
        .with_columns(
            compute_regressed_prior(
                "prev_season_rate", "league_mean", prev_season_weight=0.65
            ).alias("regressed_prior")
        )
        .with_columns(
            bayesian_shrinkage(
                current_rate_col="curr_rate",
                sample_size_col="n_games",
                prior_col="regressed_prior",
                pseudo_count_k=8.0,
            ).alias("shrunk_rate")
        )
    )

    assert df["shrunk_rate"].null_count() == 0
    # Player 1 Game 1 (n=0): 0.65 * 1.20 + 0.35 * 1.00 = 1.13
    assert df["shrunk_rate"][0] == pytest.approx(1.13)
    # Player 1 after 8 games (n=8, k=8): halfway between 2.00 and 1.13 -> 1.565
    assert df["shrunk_rate"][1] == pytest.approx((8.0 * 2.0 + 8.0 * 1.13) / 16.0)
    # Rookie Game 1 (n=0, prev_season=None): falls back to league_mean = 1.00
    assert df["shrunk_rate"][2] == pytest.approx(1.00)

    assert shrink_scalar(None, 0, 1.5, 10.0) == pytest.approx(1.5)
    assert shrink_scalar(2.5, 10, 1.5, 10.0) == pytest.approx(2.0)


def test_contracts_and_validation_errors(sample_chronological_logs: pl.DataFrame) -> None:
    """Verify schema validation rules and parameter guards."""
    adapter = HockeyAdapter()
    validated = adapter.validate_event_schema(
        sample_chronological_logs.drop("event_seq").rename({"event_id": "game_id"})
    )
    assert "event_seq" in validated.columns
    assert "event_id" in validated.columns

    participants = pl.DataFrame(
        {
            "game_id": [2024020001, 2024020001],
            "team": ["BOS", "TOR"],
            "player_id": [8478402, 8479318],
        }
    )
    valid_part = adapter.validate_participant_schema(participants)
    assert set(CANONICAL_PARTICIPANT_KEY_COLS).issubset(set(valid_part.columns))

    with pytest.raises(SchemaValidationError):
        validate_canonical_event_frame(sample_chronological_logs.drop("start_time_utc"))

    with pytest.raises(SchemaValidationError):
        validate_canonical_participant_frame(participants.drop("player_id"))

    with pytest.raises(ValueError, match="window must be >= 1"):
        pregame_rolling_mean("goals_for", window=0, group_by="team")

    with pytest.raises(ValueError, match="alpha must be in"):
        pregame_ewma("goals_for", alpha=1.5, group_by="team")

    with pytest.raises(ValueError, match="pseudo_count_k must be > 0"):
        bayesian_shrinkage("curr", "n", 1.0, pseudo_count_k=0.0)
