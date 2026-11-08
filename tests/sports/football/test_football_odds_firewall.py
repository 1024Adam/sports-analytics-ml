"""Odds / observed-outcome firewall tests for football feature lists and Gold output."""

from __future__ import annotations

import gzip
from datetime import timedelta
from pathlib import Path

import httpx
import numpy as np
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
from sports_ml.sports.football.ingestion import (
    FootballOddsIngestor,
    merge_odds_into_games,
    parse_historical_odds_csv,
    parse_the_odds_api_payload,
)
from sports_ml.sports.football.models import (
    FootballMatchupPredictor,
    Stage1FootballGameModel,
    evaluate_football_market,
    score_football_prediction_ledger,
    train_football_stage1_bundle,
)
from tests.sports.football.synthetic import (
    SyntheticSilver,
    build_synthetic_silver,
    write_synthetic_silver,
)


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


# ------------------------------------------------------------------ Milestone 29: market firewall


def _market_games(s: SyntheticSilver) -> pl.DataFrame:
    """Silver games whose closing lines vary by game (so ATS buckets are non-trivial)."""
    rng = np.random.default_rng(7)
    n = s.games.height
    spread = np.round(rng.uniform(-9.0, 9.0, n) * 2.0) / 2.0
    total = np.round(rng.uniform(38.0, 52.0, n) * 2.0) / 2.0
    p_home = np.clip(0.5 - spread / 28.0, 0.1, 0.9)
    return s.games.with_columns(
        pl.Series("market_spread_home", spread),
        pl.Series("market_total_points", total),
        pl.Series("market_prob_home", p_home),
        pl.Series("market_prob_away", 1.0 - p_home),
        pl.Series("market_home_decimal_odds", 1.0 / np.clip(p_home * 1.04, 0.05, 0.95)),
        pl.Series("market_away_decimal_odds", 1.0 / np.clip((1.0 - p_home) * 1.04, 0.05, 0.95)),
    )


def test_no_market_like_names_in_feature_lists_including_spread_and_total_prefixes() -> None:
    for columns in (STAGE1_FEATURE_COLUMNS, STAGE2_FEATURE_COLUMNS):
        for c in columns:
            lower = c.lower()
            assert not lower.startswith(("market_", "spread", "total", "closing")), c
            assert not any(tok in lower for tok in ("odds", "spread", "closing", "moneyline")), c


def test_mutating_market_columns_changes_zero_predictions_but_updates_market_report() -> None:
    s = build_synthetic_silver(future_weeks=1)
    games = _market_games(s)
    gold = build_football_stage1_features(games, s.team_logs, s.qb_logs)
    shuffled = games.with_columns(
        (pl.col("market_spread_home") * -1.0 + 3.0).alias("market_spread_home"),
        (pl.col("market_total_points") + 9.0).alias("market_total_points"),
        (1.0 - pl.col("market_prob_home")).alias("market_prob_home"),
    )
    gold_mutated = build_football_stage1_features(shuffled, s.team_logs, s.qb_logs)
    assert_frame_equal(gold, gold_mutated, check_exact=True)

    done = gold.filter(pl.col("game_state") == "FINAL")
    model = Stage1FootballGameModel(min_calibration_samples=30).fit(done)
    preds = model.predict(done)
    preds_mutated = model.predict(gold_mutated.filter(pl.col("game_state") == "FINAL"))
    for col in ("home_win_prob", "expected_home_score", "expected_away_score", "expected_margin"):
        assert (
            preds[col].to_numpy().tolist() == preds_mutated[col].to_numpy().tolist()
        )  # 0.0 change

    base = evaluate_football_market(preds, games)
    other = evaluate_football_market(preds, shuffled)
    assert base is not None and other is not None
    assert base.moneyline is not None and other.moneyline is not None
    assert base.moneyline.market_closing_baseline != other.moneyline.market_closing_baseline
    assert base.lines.spread is not None and other.lines.spread is not None
    assert base.lines.spread.buckets != other.lines.spread.buckets
    assert base.lines.total is not None and other.lines.total is not None
    assert base.lines.total.buckets != other.lines.total.buckets
    assert (
        evaluate_football_market(preds, games.drop("market_spread_home", "market_total_points"))
        is not None
    )
    assert (
        evaluate_football_market(
            preds,
            games.with_columns(
                *[
                    pl.lit(None, dtype=pl.Float64).alias(c)
                    for c in games.columns
                    if c.startswith("market_")
                ]
            ),
        )
        is None
    )


def test_historical_csv_parses_moneylines_spread_conventions_and_aliases() -> None:
    csv_text = (
        "nflverse_game_id,game_date,home_team,away_team,home_moneyline,away_moneyline,spread_home,total_points\n"
        "2023_01_DET_KC,2023-09-07,Kansas City Chiefs,Detroit Lions,-190,160,-4,53.5\n"
        ",2023-09-10,WAS,ARI,-150,130,,\n"
        ",,,,,,,\n"
    )
    odds = parse_historical_odds_csv(csv_text)
    assert odds.height == 2
    kc = odds.filter(pl.col("nflverse_game_id") == "2023_01_DET_KC").row(0, named=True)
    assert (kc["home_team"], kc["away_team"]) == ("KC", "DET")
    assert kc["market_spread_home"] == -4.0 and kc["market_total_points"] == 53.5
    assert kc["market_prob_home"] + kc["market_prob_away"] == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < kc["market_prob_away"] < kc["market_prob_home"] < 1.0
    assert kc["market_home_decimal_odds"] == pytest.approx(1.0 + 100.0 / 190.0)

    nflverse_style = parse_historical_odds_csv(
        "game_date,home_team,away_team,spread_line,total_line\n2023-09-10,WAS,ARI,3.5,40\n"
    )
    row = nflverse_style.row(0, named=True)
    assert row["market_spread_home"] == -3.5 and row["market_total_points"] == 40.0
    assert row["market_prob_home"] is None  # no moneyline -> no probability


def test_merge_never_overwrites_schedule_values_and_matches_within_one_day() -> None:
    s = build_synthetic_silver(future_weeks=1)
    g = s.games.sort("start_time_utc")
    keep, fill, shifted, final_only = (g.row(i, named=True) for i in range(4))
    games = g.with_columns(
        *[pl.lit(None, dtype=pl.Float64).alias(c) for c in g.columns if c.startswith("market_")]
    ).with_columns(
        pl.when(pl.col("game_id") == keep["game_id"])
        .then(2.5)
        .otherwise(None)
        .alias("market_total_points")
    )

    def row(rec: dict[str, object], **kw: object) -> dict[str, object]:
        base = {
            "nflverse_game_id": None,
            "game_date": rec["game_date"],
            "home_team": rec["home_team"],
            "away_team": rec["away_team"],
            "market_home_decimal_odds": 1.8,
            "market_away_decimal_odds": 2.1,
            "market_prob_home": 0.55,
            "market_prob_away": 0.45,
            "market_spread_home": -2.0,
            "market_total_points": 47.0,
        }
        return base | kw

    odds = pl.DataFrame(
        [
            row(keep),
            row(fill, nflverse_game_id=str(fill["nflverse_game_id"])),
            row(shifted, game_date=shifted["game_date"] + timedelta(days=1)),
            row(final_only, game_date=final_only["game_date"] + timedelta(days=5)),
        ],
        schema=parse_historical_odds_csv("").schema,
    )
    merged = merge_odds_into_games(games, odds).sort("start_time_utc")
    r = [merged.row(i, named=True) for i in range(4)]
    assert r[0]["market_total_points"] == 2.5  # the schedule value survives
    assert r[0]["market_spread_home"] == -2.0  # ... while its missing columns are filled
    assert r[1]["market_prob_home"] == 0.55  # id match
    assert r[2]["market_prob_home"] == 0.55  # +1 day tolerance
    assert r[3]["market_prob_home"] is None  # 5 days away: no match
    two_sided = (
        "market_home_decimal_odds",
        "market_away_decimal_odds",
        "market_prob_home",
        "market_prob_away",
    )
    partial = games.with_columns(pl.lit(1.9).alias("market_home_decimal_odds"))
    kept = merge_odds_into_games(partial, odds).sort("start_time_utc").row(1, named=True)
    assert kept["market_prob_home"] is None  # a quote is taken whole or not at all
    assert all(kept[c] is None or c == "market_home_decimal_odds" for c in two_sided)
    unfinished = merge_odds_into_games(games, odds, only_unfinished=True)
    assert unfinished.filter(pl.col("game_state") == "FINAL")["market_prob_home"].null_count() == (
        unfinished.filter(pl.col("game_state") == "FINAL").height
    )


def test_ingestor_reads_archive_skips_live_without_key_and_persists_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    s = build_synthetic_silver(future_weeks=1)
    g0 = s.games.sort("start_time_utc").row(0, named=True)
    s = SyntheticSilver(
        s.games.with_columns(
            *[
                pl.lit(None, dtype=pl.Float64).alias(c)
                for c in s.games.columns
                if c.startswith("market_")
            ]
        ),
        s.team_logs,
        s.qb_logs,
        s.player_logs,
        s.availability,
    )
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    s.games.write_parquet(silver_dir / "games.parquet")
    ingestor = FootballOddsIngestor(tmp_path / "odds")
    assert ingestor.persist_to_silver(silver_dir) == 0  # no archive, no key: nothing to add
    assert ingestor.fetch_live_odds_snapshot().is_empty()

    csv_text = (
        "game_date,home_team,away_team,home_moneyline,away_moneyline,spread_home,total_points\n"
        f"{g0['game_date']},{g0['home_team']},{g0['away_team']},-200,170,-5.5,45.5\n"
    )
    with gzip.open(ingestor.historical_archive_path, "wb") as fh:
        fh.write(csv_text.encode())
    assert ingestor.persist_to_silver(silver_dir) == 1
    first = pl.read_parquet(silver_dir / "games.parquet")
    assert ingestor.persist_to_silver(silver_dir) == 1
    assert_frame_equal(first, pl.read_parquet(silver_dir / "games.parquet"))
    row = first.filter(pl.col("game_id") == g0["game_id"]).row(0, named=True)
    assert row["market_spread_home"] == -5.5 and 0.0 < row["market_prob_home"] < 1.0


def test_live_snapshot_parses_events_fills_only_unfinished_and_failures_are_silent(
    tmp_path: Path,
) -> None:
    s = build_synthetic_silver(future_weeks=1)
    fut = s.games.filter(pl.col("game_state") != "FINAL").row(0, named=True)
    kickoff = fut["start_time_utc"].isoformat()
    names = {"BUF": "Buffalo Bills", "KC": "Kansas City Chiefs", "MIA": "Miami Dolphins"}
    home, away = str(fut["home_team"]), str(fut["away_team"])
    payload = [
        {
            "home_team": names.get(home, home),
            "away_team": names.get(away, away),
            "commence_time": kickoff,
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": names.get(home, home), "price": 1.80},
                                {"name": names.get(away, away), "price": 2.10},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": names.get(home, home), "price": 1.91, "point": -3.5}
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [{"name": "Over", "price": 1.9, "point": 44.5}],
                        },
                    ]
                }
            ],
        }
    ]
    parsed = parse_the_odds_api_payload(payload)
    assert parsed.height == 1
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(200, json=payload)

    ing = FootballOddsIngestor(
        tmp_path / "odds", api_key="k", transport=httpx.MockTransport(handler)
    )
    snap = ing.fetch_live_odds_snapshot()
    assert len(hits) == 1 and "markets=h2h%2Cspreads%2Ctotals" in hits[0]
    assert ing.fetch_live_odds_snapshot(offline=True).is_empty() and len(hits) == 1
    assert snap.height == 1
    merged = ing.apply_to_games(
        s.games.with_columns(
            *[
                pl.lit(None, dtype=pl.Float64).alias(c)
                for c in s.games.columns
                if c.startswith("market_")
            ]
        )
    )
    assert merged.filter(pl.col("game_state") == "FINAL")["market_prob_home"].null_count() == (
        merged.filter(pl.col("game_state") == "FINAL").height
    )
    got = merged.filter(pl.col("game_id") == fut["game_id"]).row(0, named=True)
    assert got["market_spread_home"] == -3.5 and got["market_total_points"] == 44.5
    assert got["market_prob_home"] + got["market_prob_away"] == pytest.approx(1.0, abs=1e-9)

    broken = FootballOddsIngestor(
        tmp_path / "odds2",
        api_key="k",
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    assert broken.fetch_live_odds_snapshot().is_empty()


def test_predictor_ledger_records_market_context_and_score_reports_ats_and_clv(
    tmp_path: Path,
) -> None:
    s = build_synthetic_silver(future_weeks=1)
    games = _market_games(s)
    write_synthetic_silver(
        tmp_path, SyntheticSilver(games, s.team_logs, s.qb_logs, s.player_logs, s.availability)
    )
    settings = get_settings(root_dir=tmp_path)
    gold = build_and_persist_football_gold_features(settings)
    train_football_stage1_bundle(gold, settings.paths.artifacts_dir)
    predictor = FootballMatchupPredictor.from_artifacts(
        settings.paths.artifacts_dir / "football" / "latest", settings=settings
    )
    done = games.filter(pl.col("game_state") == "FINAL").sort("start_time_utc").tail(24)
    preds = predictor.predict_games(done["game_id"].to_list())
    assert all(p.market_spread_home is not None for p in preds)
    ledger = pl.read_parquet(settings.paths.predictions_dir / "football_predictions.parquet")
    assert {"market_spread_home_at_pred", "market_total_at_pred"} <= set(ledger.columns)

    # lines moved after the prediction: closing = taken - 1 for the home side -> CLV for home bets
    moved = games.with_columns((pl.col("market_spread_home") - 1.0).alias("market_spread_home"))
    moved.write_parquet(settings.paths.silver_dir / "football" / "games.parquet")
    score = score_football_prediction_ledger(settings)
    assert score.market is not None and score.market.lines.spread is not None
    bucket = score.market.lines.spread.get_bucket(1.0)
    assert bucket is not None and bucket.mean_clv is not None and bucket.n_bets > 0
    assert score.market.moneyline is not None and score.market.n_with_moneyline == score.n_games


def test_football_cli_shows_market_benchmark_when_odds_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rich.console import Console
    from typer.testing import CliRunner

    from sports_ml import cli

    s = build_synthetic_silver(future_weeks=1)
    games = _market_games(s)
    write_synthetic_silver(
        tmp_path, SyntheticSilver(games, s.team_logs, s.qb_logs, s.player_logs, s.availability)
    )
    monkeypatch.setenv("SPORTS_ML_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "console", Console(width=300))
    runner = CliRunner()

    def run(*args: str) -> str:
        res = runner.invoke(cli.app, list(args))
        assert res.exit_code == 0, f"{args}: {res.output}{res.exception!r}"
        return res.output

    run("features", "build", "--sport", "football")
    run("train", "--sport", "football")
    out = run("backtest", "--sport", "football", "--stage", "1", "--splits", "1")
    assert "De-Vigged Market Closing Line" in out
    assert "Fundamental vs. Market Disagreement" in out
    assert "Against the Spread (ATS)" in out and "Over / Under vs. Closing Total" in out
    assert "break-even 52.38%" in out

    run("predict", "--sport", "football", "--offline", "--season", "2023", "--week", "3")
    scored = run("score-predictions", "--sport", "football")
    assert "Against the Spread (ATS)" in scored and "CLV" in scored

    # without odds the CLI degrades gracefully
    stripped = games.with_columns(
        *[pl.lit(None, dtype=pl.Float64).alias(c) for c in games.columns if c.startswith("market_")]
    )
    stripped.write_parquet(tmp_path / "data" / "silver" / "football" / "games.parquet")
    assert "market benchmark skipped" in run(
        "backtest", "--sport", "football", "--stage", "1", "--splits", "1"
    )
