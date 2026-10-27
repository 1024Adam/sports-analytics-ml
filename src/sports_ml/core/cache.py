"""Sport-agnostic raw-cache compression, fallback reading, and directory compaction (`cache.py`).

Provides transparent Gzip compression (`.csv.gz` and `.json.gz`) with automatic
fallback to legacy uncompressed `.csv` / `.json` files on disk, plus an offline
directory compactor (`compact_raw_cache_directory`) that deduplicates cumulative
MoneyPuck `teams_*.csv` files and compresses existing raw caches in place.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict


class CacheCompactionReport(BaseModel):
    """Summary metrics from running `compact_raw_cache_directory`."""

    model_config = ConfigDict(frozen=True)

    files_compacted: int
    duplicates_removed: int
    bytes_before: int
    bytes_after: int
    bytes_saved: int


def _ensure_gz_path(path: Path) -> Path:
    """Return `path` with a `.gz` suffix if not already present."""
    if path.suffix == ".gz":
        return path
    return Path(f"{path}.gz")


def _legacy_uncompressed_path(gz_path: Path) -> Path:
    """Return the uncompressed sibling path for a `.gz` path (or `gz_path` itself)."""
    if gz_path.suffix == ".gz":
        return gz_path.with_suffix("")
    return gz_path


def resolve_cache_read_path(preferred_gz_path: Path) -> Path | None:
    """Resolve `*.gz` first and fall back to the legacy uncompressed sibling path if present."""
    gz_candidate = _ensure_gz_path(preferred_gz_path)
    if gz_candidate.exists():
        return gz_candidate
    legacy_candidate = _legacy_uncompressed_path(gz_candidate)
    if legacy_candidate.exists():
        return legacy_candidate
    return None


def write_gzip_bytes(
    gz_path: Path,
    raw_bytes: bytes,
    *,
    compresslevel: int = 6,
) -> Path:
    """Atomically write Gzip-compressed bytes and remove any legacy uncompressed sibling file."""
    target_gz = _ensure_gz_path(gz_path)
    target_gz.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_gz.with_name(f".{target_gz.name}.tmp")
    compressed = gzip.compress(raw_bytes, compresslevel=compresslevel)
    tmp_path.write_bytes(compressed)
    tmp_path.replace(target_gz)

    legacy_path = _legacy_uncompressed_path(target_gz)
    if legacy_path != target_gz and legacy_path.exists():
        legacy_path.unlink(missing_ok=True)

    return target_gz


def read_cached_bytes(preferred_gz_path: Path) -> bytes:
    """Read and decompress `*.gz` or read the legacy uncompressed file transparently."""
    resolved = resolve_cache_read_path(preferred_gz_path)
    if resolved is None:
        raise FileNotFoundError(f"Cached file not found for '{preferred_gz_path}'.")
    if resolved.suffix == ".gz":
        return gzip.decompress(resolved.read_bytes())
    return resolved.read_bytes()


def write_cached_json(
    gz_path: Path,
    payload: Any,
    *,
    compresslevel: int = 6,
) -> Path:
    """Serialize `payload` as compact UTF-8 JSON and write Gzip-compressed bytes to `*.json.gz`."""
    raw_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return write_gzip_bytes(gz_path, raw_bytes, compresslevel=compresslevel)


def read_cached_json(
    preferred_gz_path: Path,
    *,
    ttl_seconds: float | None = None,
) -> Any | None:
    """Read JSON from `*.json.gz` (or legacy `.json`) with optional `mtime` TTL check."""
    resolved = resolve_cache_read_path(preferred_gz_path)
    if resolved is None:
        return None
    if ttl_seconds is not None:
        age = time.time() - resolved.stat().st_mtime
        if age > ttl_seconds:
            return None
    raw_bytes = (
        gzip.decompress(resolved.read_bytes())
        if resolved.suffix == ".gz"
        else resolved.read_bytes()
    )
    return json.loads(raw_bytes.decode("utf-8"))


def list_cached_files(directory: Path, pattern: str) -> list[Path]:
    """List logical cached files matching `pattern` (e.g. `'*.json'` or `'skaters_*.csv'`).

    Deduplicates when both `<name>` and `<name>.gz` exist on disk, preferring `<name>.gz`,
    and excludes `*.meta.json` sidecar files when matching `*.json`.
    """
    if not directory.exists():
        return []

    by_base_name: dict[str, Path] = {}
    for candidate in sorted(directory.glob(pattern)):
        if candidate.name.endswith(".meta.json") or candidate.name.startswith("."):
            continue
        by_base_name[candidate.name] = candidate

    gz_pattern = pattern if pattern.endswith(".gz") else f"{pattern}.gz"
    for gz_candidate in sorted(directory.glob(gz_pattern)):
        if gz_candidate.name.startswith("."):
            continue
        base_name = gz_candidate.name[:-3]
        if base_name.endswith(".meta.json"):
            continue
        by_base_name[base_name] = gz_candidate

    return [by_base_name[k] for k in sorted(by_base_name)]


def cache_stem(path: Path) -> str:
    """Return the logical filename stem stripping `.csv.gz`, `.json.gz`, `.csv`, or `.json`."""
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).stem


_LEGACY_MP_TEAMS_RE = re.compile(r"^teams_(\d{4})\.csv(\.gz)?$")


def compact_raw_cache_directory(
    raw_dir: Path,
    *,
    dry_run: bool = False,
) -> CacheCompactionReport:
    """Recursively compact `raw_dir` in place with zero network calls.

    1. Consolidates legacy `moneypuck/teams_YYYY.csv` (and `.csv.gz`) files into a single
       `moneypuck/teams_all.csv.gz` (preserving the newest/largest cumulative payload and
       its `.meta.json` sidecar) and removes redundant per-season copies.
    2. Compresses all remaining uncompressed `.csv` -> `.csv.gz` and `.json` (excluding
       `*.meta.json` ETag sidecars) -> `.json.gz`, verifying decompression/parse integrity
       before unlinking the uncompressed original.
    """
    if not raw_dir.exists():
        return CacheCompactionReport(
            files_compacted=0,
            duplicates_removed=0,
            bytes_before=0,
            bytes_after=0,
            bytes_saved=0,
        )

    all_initial_files = [
        p for p in raw_dir.rglob("*") if p.is_file() and not p.name.startswith(".")
    ]
    bytes_before = sum(p.stat().st_size for p in all_initial_files)

    files_compacted = 0
    duplicates_removed = 0
    estimated_bytes_after = bytes_before

    # Step 1: Deduplicate legacy MoneyPuck `teams_YYYY.csv[.gz]` into `teams_all.csv.gz`
    mp_dirs: set[Path] = set()
    if (raw_dir / "moneypuck").is_dir():
        mp_dirs.add(raw_dir / "moneypuck")
    if raw_dir.name == "moneypuck":
        mp_dirs.add(raw_dir)
    for candidate_mp in raw_dir.rglob("moneypuck"):
        if candidate_mp.is_dir():
            mp_dirs.add(candidate_mp)

    handled_paths: set[Path] = set()

    for mp_dir in sorted(mp_dirs):
        team_files = [
            p
            for p in sorted(mp_dir.iterdir())
            if p.is_file() and _LEGACY_MP_TEAMS_RE.match(p.name) is not None
        ]
        if not team_files:
            continue

        target_all_gz = mp_dir / "teams_all.csv.gz"
        target_all_meta = mp_dir / "teams_all.meta.json"
        existing_all = resolve_cache_read_path(target_all_gz)

        # Pick the newest/largest file as canonical cumulative source
        best_source = max(
            team_files,
            key=lambda p: (p.stat().st_size, p.stat().st_mtime),
        )
        raw_payload = (
            gzip.decompress(best_source.read_bytes())
            if best_source.suffix == ".gz"
            else best_source.read_bytes()
        )
        if existing_all is not None:
            existing_bytes = read_cached_bytes(target_all_gz)
            if len(existing_bytes) > len(raw_payload):
                raw_payload = existing_bytes

        compressed_payload = gzip.compress(raw_payload, compresslevel=6)
        best_stem = cache_stem(best_source)
        best_meta = mp_dir / f"{best_stem}.meta.json"

        # Calculate size delta for dry-run and reporting
        old_group_bytes = sum(p.stat().st_size for p in team_files)
        if existing_all is not None and existing_all not in team_files:
            old_group_bytes += existing_all.stat().st_size

        meta_removed_bytes = 0
        for tf in team_files:
            stem = cache_stem(tf)
            meta_file = mp_dir / f"{stem}.meta.json"
            if meta_file.exists():
                meta_removed_bytes += meta_file.stat().st_size
                handled_paths.add(meta_file)
            handled_paths.add(tf)

        new_meta_bytes = 0
        if target_all_meta.exists():
            new_meta_bytes = target_all_meta.stat().st_size
        elif best_meta.exists():
            new_meta_bytes = best_meta.stat().st_size

        estimated_bytes_after += (
            len(compressed_payload) + new_meta_bytes - old_group_bytes - meta_removed_bytes
        )

        # Count compacted vs removed duplicates
        if existing_all is not None and existing_all.suffix == ".gz":
            duplicates_removed += len(team_files)
        else:
            files_compacted += 1
            duplicates_removed += max(0, len(team_files) - 1)

        if not dry_run:
            write_gzip_bytes(target_all_gz, raw_payload, compresslevel=6)
            # Verify readable by Polars before deleting legacy files
            _ = pl.read_csv(target_all_gz, n_rows=5)
            if not target_all_meta.exists() and best_meta.exists():
                target_all_meta.write_bytes(best_meta.read_bytes())
            for tf in team_files:
                tf.unlink(missing_ok=True)
                stem = cache_stem(tf)
                meta_file = mp_dir / f"{stem}.meta.json"
                if meta_file != target_all_meta and meta_file.exists():
                    meta_file.unlink(missing_ok=True)

    # Step 2: Compress all remaining uncompressed `.csv` and `.json` (excluding `*.meta.json`)
    remaining_files = [
        p
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file()
        and p not in handled_paths
        and not p.name.startswith(".")
        and not p.name.endswith(".meta.json")
        and p.suffix in {".csv", ".json"}
    ]

    for file_path in remaining_files:
        gz_target = _ensure_gz_path(file_path)
        raw_bytes = file_path.read_bytes()
        old_size = file_path.stat().st_size

        if gz_target.exists():
            # Already has a `.gz` sibling -> redundant uncompressed copy
            duplicates_removed += 1
            estimated_bytes_after -= old_size
            if not dry_run:
                file_path.unlink(missing_ok=True)
            continue

        compressed_bytes = gzip.compress(raw_bytes, compresslevel=6)
        files_compacted += 1
        estimated_bytes_after += len(compressed_bytes) - old_size

        if not dry_run:
            write_gzip_bytes(gz_target, raw_bytes, compresslevel=6)
            # Verify integrity of written `.gz`
            decompressed = read_cached_bytes(gz_target)
            if decompressed != raw_bytes:
                raise RuntimeError(f"Gzip verification failed for '{gz_target}'.")

    if dry_run:
        bytes_after = max(0, estimated_bytes_after)
    else:
        final_files = [p for p in raw_dir.rglob("*") if p.is_file() and not p.name.startswith(".")]
        bytes_after = sum(p.stat().st_size for p in final_files)

    bytes_saved = max(0, bytes_before - bytes_after)
    return CacheCompactionReport(
        files_compacted=files_compacted,
        duplicates_removed=duplicates_removed,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        bytes_saved=bytes_saved,
    )
