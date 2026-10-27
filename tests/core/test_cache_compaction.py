"""Tests for raw-cache compression and compaction (`test_cache_compaction.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from sports_ml.cli import app
from sports_ml.core.cache import (
    compact_raw_cache_directory,
    read_cached_bytes,
    read_cached_json,
    resolve_cache_read_path,
    write_cached_json,
    write_gzip_bytes,
)


def test_gzip_bytes_json_and_polars_csv_roundtrip(tmp_path: Path) -> None:
    """Verify `write_gzip_bytes`, `write_cached_json`, and `pl.read_csv(gz_path)` round-trip."""
    csv_raw = b"gameId,season,team,xGoalsFor\n2024020001,2024,BOS,2.85\n2024020001,2024,TOR,1.95\n"
    csv_gz_path = tmp_path / "moneypuck" / "skaters_2024.csv.gz"

    written_path = write_gzip_bytes(csv_gz_path, csv_raw)
    assert written_path == csv_gz_path
    assert csv_gz_path.exists()
    assert read_cached_bytes(csv_gz_path) == csv_raw

    # Polars natively decompresses `.csv.gz`
    df = pl.read_csv(csv_gz_path)
    assert df.height == 2
    assert df["team"].to_list() == ["BOS", "TOR"]

    # JSON round-trip with TTL check
    json_payload = {"id": 2024020001, "gameState": "FINAL", "teams": ["BOS", "TOR"]}
    json_gz_path = tmp_path / "nhl_api" / "boxscores" / "2024020001.json.gz"
    write_cached_json(json_gz_path, json_payload)
    assert json_gz_path.exists()
    assert read_cached_json(json_gz_path) == json_payload
    assert read_cached_json(json_gz_path, ttl_seconds=3600.0) == json_payload
    assert read_cached_json(json_gz_path, ttl_seconds=-1.0) is None


def test_legacy_uncompressed_fallback_and_cleanup_on_write(tmp_path: Path) -> None:
    """Verify `resolve_cache_read_path` falls back to legacy files and cleans up on write."""
    legacy_json = tmp_path / "schedule_now.json"
    preferred_gz = tmp_path / "schedule_now.json.gz"
    payload = {"gameWeek": [{"date": "2024-10-10", "games": []}]}
    legacy_json.write_text(json.dumps(payload), encoding="utf-8")

    # Fallback resolves legacy uncompressed file when `.gz` does not exist yet
    assert resolve_cache_read_path(preferred_gz) == legacy_json
    assert read_cached_json(preferred_gz) == payload

    # Writing `.json.gz` atomically replaces and removes the legacy uncompressed sibling
    updated = {"gameWeek": [{"date": "2024-10-11", "games": []}]}
    write_cached_json(preferred_gz, updated)
    assert preferred_gz.exists()
    assert not legacy_json.exists()
    assert resolve_cache_read_path(preferred_gz) == preferred_gz
    assert read_cached_json(preferred_gz) == updated


def test_compact_raw_cache_directory_deduplicates_teams_and_supports_dry_run(
    tmp_path: Path,
) -> None:
    """Verify `compact_raw_cache_directory` deduplicates `teams_YYYY.csv` and compacts files."""
    raw_dir = tmp_path / "data" / "raw"
    mp_dir = raw_dir / "moneypuck"
    nhl_sched_dir = raw_dir / "nhl_api" / "schedules"
    mp_dir.mkdir(parents=True)
    nhl_sched_dir.mkdir(parents=True)

    # Create 3 redundant cumulative MoneyPuck `teams_YYYY.csv` files + `.meta.json` sidecars
    cumulative_csv = (
        "gameId,season,team,situation,xGoalsFor,xGoalsAgainst\n"
        + "\n".join(
            f"202{yr}0200{i:02d},202{yr},BOS,5on5,2.75,1.85"
            for yr in (3, 4, 5)
            for i in range(1, 30)
        )
        + "\n"
    ).encode("utf-8")

    for yr in (2023, 2024, 2025):
        (mp_dir / f"teams_{yr}.csv").write_bytes(cumulative_csv)
        (mp_dir / f"teams_{yr}.meta.json").write_text(
            json.dumps({"etag": f"etag-{yr}", "source_url": "https://moneypuck.com/all_teams.csv"}),
            encoding="utf-8",
        )

    skaters_csv = (
        "gameId,season,playerId,playerTeam,situation,I_F_xGoals\n"
        + "\n".join(f"20240200{i:02d},2024,8477956,BOS,all,0.85" for i in range(1, 30))
        + "\n"
    ).encode("utf-8")
    (mp_dir / "skaters_2024.csv").write_bytes(skaters_csv)
    (mp_dir / "skaters_2024.meta.json").write_text('{"etag": "sk-2024"}', encoding="utf-8")

    sched_json = json.dumps(
        {"gameWeek": [{"date": "2024-10-10", "games": [{"id": i} for i in range(50)]}]},
        indent=2,
    )
    (nhl_sched_dir / "schedule_2024-10-10.json").write_text(sched_json, encoding="utf-8")

    # 1. Dry run: reports compaction & deduplication without mutating files on disk
    dry_report = compact_raw_cache_directory(raw_dir, dry_run=True)
    assert dry_report.files_compacted == 3  # teams_all + skaters_2024 + schedule_2024-10-10
    assert dry_report.duplicates_removed == 2  # 2 redundant teams_YYYY.csv files
    assert dry_report.bytes_saved > 0
    assert (mp_dir / "teams_2023.csv").exists()
    assert not (mp_dir / "teams_all.csv.gz").exists()

    # 2. Live compaction in place
    report = compact_raw_cache_directory(raw_dir, dry_run=False)
    assert report.files_compacted == 3
    assert report.duplicates_removed == 2
    assert report.bytes_after < report.bytes_before
    assert report.bytes_saved == report.bytes_before - report.bytes_after

    # Verify deduplicated `teams_all.csv.gz` and preserved `.meta.json`
    assert (mp_dir / "teams_all.csv.gz").exists()
    assert (mp_dir / "teams_all.meta.json").exists()
    for yr in (2023, 2024, 2025):
        assert not (mp_dir / f"teams_{yr}.csv").exists()
        assert not (mp_dir / f"teams_{yr}.meta.json").exists()

    # Verify `skaters_2024.csv.gz` and `schedule_2024-10-10.json.gz`
    assert (mp_dir / "skaters_2024.csv.gz").exists()
    assert not (mp_dir / "skaters_2024.csv").exists()
    assert (mp_dir / "skaters_2024.meta.json").exists()
    assert (nhl_sched_dir / "schedule_2024-10-10.json.gz").exists()
    assert not (nhl_sched_dir / "schedule_2024-10-10.json").exists()

    # Verify Polars can read compacted `teams_all.csv.gz` directly
    teams_df = pl.read_csv(mp_dir / "teams_all.csv.gz")
    assert set(teams_df["season"].unique().to_list()) == {2023, 2024, 2025}

    # 3. Verify `sports-ml compact-cache` CLI command works cleanly (idempotent second pass)
    runner = CliRunner()
    cli_res = runner.invoke(app, ["compact-cache", "--data-dir", str(tmp_path / "data")])
    assert cli_res.exit_code == 0, cli_res.stdout
    assert "Raw Cache Storage Compaction" in cli_res.stdout
