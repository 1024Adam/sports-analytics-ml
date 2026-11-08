"""Odds / observed-outcome firewall tests for football feature lists and Gold output."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sports_ml.config import get_settings
from sports_ml.sports.football.features import (
    FOOTBALL_STAGE1_CANDIDATE_FEATURE_COLS,
    FOOTBALL_STAGE2_CANDIDATE_FEATURE_COLS,
    STAGE1_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    assert_zero_firewall_contamination,
    build_and_persist_football_gold_features,
    build_football_stage1_features,
    filter_forbidden_feature_columns,
    is_forbidden_feature_column,
    strip_forbidden_columns,
)
from tests.sports.football.synthetic import SyntheticSilver, build_synthetic_silver


def _build(s: SyntheticSilver) -> pl.DataFrame:
    return build_football_stage1_features(s.games, s.team_logs, s.qb_logs).sort("game_id")


def test_feature_lists_contain_no_market_weather_or_odds_like_columns() -> None:
    for columns in (STAGE1_FEATURE_COLUMNS, STAGE2_FEATURE_COLUMNS):
        assert columns
        for c in columns:
            lower = c.lower()
            assert not lower.startswith(("market_", "obs_weather_")), c
            assert not any(tok in lower for tok in ("odds", "spread", "closing")), c
            assert not is_forbidden_feature_column(c)
    assert set(STAGE1_FEATURE_COLUMNS) == set(FOOTBALL_STAGE1_CANDIDATE_FEATURE_COLS)
    assert set(STAGE2_FEATURE_COLUMNS) == set(FOOTBALL_STAGE2_CANDIDATE_FEATURE_COLS)
    assert len(set(STAGE1_FEATURE_COLUMNS)) == len(STAGE1_FEATURE_COLUMNS)


def test_firewall_detects_and_filters_contaminated_columns() -> None:
    bad = [
        "market_prob_home",
        "obs_weather_wind",
        "closing_total",
        "home_spread",
        "odds_ratio",
        "Market_Spread_Home",
    ]
    for c in bad:
        assert is_forbidden_feature_column(c), c
    assert not is_forbidden_feature_column("diff_off_epa_per_play_std")
    assert filter_forbidden_feature_columns(["a", *bad, "b"]) == ("a", "b")
    with pytest.raises(ValueError, match="contamination"):
        assert_zero_firewall_contamination(["diff_qb_value", "market_prob_home"])
    frame = pl.DataFrame({"x": [1], **{c: [1.0] for c in bad}})
    assert strip_forbidden_columns(frame).columns == ["x"]


def test_mutating_market_and_weather_columns_changes_zero_features() -> None:
    s = build_synthetic_silver()
    baseline = _build(s)

    market_noise = s.games.with_columns(
        pl.lit(7.77).alias("market_prob_home"),
        pl.lit(0.01).alias("market_prob_away"),
        pl.lit(99.0).alias("market_spread_home"),
        pl.lit(250.0).alias("market_total_points"),
        pl.lit(1.01).alias("market_home_decimal_odds"),
        pl.lit(-40.0).alias("obs_weather_temp"),
        pl.lit(90.0).alias("obs_weather_wind"),
        pl.lit("dome").alias("obs_weather_roof"),
        pl.lit(123.0).alias("closing_line"),
        pl.lit(5.0).alias("spread_line"),
    )
    mutated = _build(
        SyntheticSilver(market_noise, s.team_logs, s.qb_logs, s.player_logs, s.availability)
    )
    assert_frame_equal(baseline, mutated, check_exact=True)
    assert not [c for c in mutated.columns if is_forbidden_feature_column(c)]

    without = _build(
        SyntheticSilver(
            strip_forbidden_columns(s.games), s.team_logs, s.qb_logs, s.player_logs, s.availability
        )
    )
    assert_frame_equal(baseline, without, check_exact=True)


def test_gold_parquet_is_zero_nan_flagged_and_firewalled(tmp_path: Path) -> None:
    s = build_synthetic_silver(future_weeks=1)
    settings = get_settings(root_dir=tmp_path)
    settings.paths.ensure_dirs(sport="football")
    silver = settings.paths.silver_dir / "football"
    s.games.write_parquet(silver / "games.parquet")
    s.team_logs.write_parquet(silver / "team_game_logs.parquet")
    s.qb_logs.write_parquet(silver / "qb_game_logs.parquet")

    gold = build_and_persist_football_gold_features(settings)
    path = settings.paths.gold_dir / "football" / "stage1_features.parquet"
    assert path.exists()
    stored = pl.read_parquet(path)
    assert stored.height == s.games.height
    assert not [c for c in stored.columns if is_forbidden_feature_column(c)]
    assert set(STAGE1_FEATURE_COLUMNS).issubset(stored.columns)
    for c in STAGE1_FEATURE_COLUMNS:
        assert stored[c].null_count() == 0, c
        if stored.schema[c].is_float():
            assert stored[c].is_nan().sum() == 0, c
        assert stored.schema[c] != pl.Boolean
    assert_frame_equal(gold, stored)

    # Explicit imputation flags: season openers have no previous kickoff to measure rest from
    opener = stored.sort("start_time_utc").row(0, named=True)
    assert opener["home_days_rest_is_imputed"] == 1
    assert opener["home_continuity_is_imputed"] == 1
    assert opener["home_qb_is_imputed"] in (0, 1)
    upcoming = stored.filter(pl.col("game_state") == "FUT")
    assert upcoming.height > 0
    assert upcoming["home_win"].null_count() == upcoming.height  # labels stay on the label side

    empty = get_settings(root_dir=tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        build_and_persist_football_gold_features(empty)


def test_cli_features_build_football(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sports_ml import cli

    s = build_synthetic_silver()
    settings = get_settings(root_dir=tmp_path)
    settings.paths.ensure_dirs(sport="football")
    silver = settings.paths.silver_dir / "football"
    s.games.write_parquet(silver / "games.parquet")
    s.team_logs.write_parquet(silver / "team_game_logs.parquet")
    s.qb_logs.write_parquet(silver / "qb_game_logs.parquet")
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))

    runner = CliRunner()
    ok = runner.invoke(cli.app, ["features", "build", "--sport", "football", "--league", "nfl"])
    assert ok.exit_code == 0, ok.output
    assert "Stage 1 rows=96" in ok.output.replace("\n", "")
    assert (settings.paths.gold_dir / "football" / "stage1_features.parquet").exists()

    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path / "missing"))
    missing = runner.invoke(cli.app, ["features", "build", "--sport", "football"])
    assert missing.exit_code != 0
