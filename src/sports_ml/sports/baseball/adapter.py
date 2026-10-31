"""MLB & NPB Baseball implementation of `SportAdapter` (`sports/baseball/adapter.py`)."""

from __future__ import annotations

import math

import polars as pl

from sports_ml.core.contracts import (
    CompetitionTopology,
    SchemaValidationError,
    validate_canonical_event_frame,
    validate_canonical_participant_frame,
)
from sports_ml.sports.baseball.constants import (
    normalize_baseball_team,
    resolve_team_league,
)


def parse_baseball_ip_to_outs(ip_value: str | int | float) -> int:
    """Convert traditional baseball innings-pitched notation (`6.1`, `'6.2'`) to exact outs.

    In baseball boxscores, `.1` denotes 1/3 of an inning (1 out) and `.2` denotes
    2/3 of an inning (2 outs). True floating-point thirds (`6.333...`, `6.666...`)
    are also supported. Any invalid fractional digit (e.g., `.4` or `.5`) raises
    `ValueError`.
    """
    if isinstance(ip_value, int):
        if ip_value < 0:
            raise ValueError(f"Innings pitched cannot be negative, got {ip_value}.")
        return ip_value * 3

    if isinstance(ip_value, str):
        cleaned = ip_value.strip()
        if not cleaned:
            raise ValueError("Empty innings pitched string.")
        if "." not in cleaned:
            whole = int(cleaned)
            if whole < 0:
                raise ValueError(f"Innings pitched cannot be negative, got '{ip_value}'.")
            return whole * 3
        whole_str, frac_str = cleaned.split(".", maxsplit=1)
        whole = int(whole_str) if whole_str else 0
        if whole < 0:
            raise ValueError(f"Innings pitched cannot be negative, got '{ip_value}'.")
        if frac_str in {"0", "00"}:
            return whole * 3
        if frac_str == "1":
            return whole * 3 + 1
        if frac_str == "2":
            return whole * 3 + 2
        # Check if string is a decimal representation of 1/3 or 2/3
        val = float(cleaned)
        return _float_ip_to_outs(val)

    if isinstance(ip_value, float):
        return _float_ip_to_outs(ip_value)

    raise TypeError(f"Unsupported innings pitched type: {type(ip_value)!r}")


def _float_ip_to_outs(val: float) -> int:
    if math.isnan(val) or math.isinf(val) or val < 0.0:
        raise ValueError(f"Invalid innings pitched float: {val}")
    whole = int(math.floor(val + 1e-9))
    rem = val - whole
    if abs(rem) < 1e-4:
        return whole * 3
    if abs(rem - 0.1) < 1e-4 or abs(rem - (1.0 / 3.0)) < 1e-2:
        return whole * 3 + 1
    if abs(rem - 0.2) < 1e-4 or abs(rem - (2.0 / 3.0)) < 1e-2:
        return whole * 3 + 2
    raise ValueError(
        f"Invalid baseball innings-pitched value {val}: fractional part must be "
        ".0, .1 (1 out), .2 (2 outs), 1/3, or 2/3."
    )


def outs_to_true_innings(outs_pitched: int | float) -> float:
    """Convert integer `outs_pitched` to true mathematical innings (`outs / 3.0`)."""
    if outs_pitched < 0:
        raise ValueError(f"outs_pitched cannot be negative, got {outs_pitched}.")
    return float(outs_pitched) / 3.0


class BaseballAdapter:
    """Dual-league (MLB & NPB) Baseball adapter implementing `SportAdapter`."""

    @property
    def sport_slug(self) -> str:
        return "baseball"

    @property
    def topology(self) -> CompetitionTopology:
        return "head_to_head"

    def validate_event_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize an MLB/NPB games DataFrame against canonical event rules."""
        working = df
        if "sport" not in working.columns:
            working = working.with_columns(pl.lit(self.sport_slug).alias("sport"))
        if "event_id" not in working.columns and "game_id" in working.columns:
            working = working.with_columns(pl.col("game_id").alias("event_id"))
        if "home_team" in working.columns:
            working = working.with_columns(
                pl.col("home_team")
                .map_elements(normalize_baseball_team, return_dtype=pl.Utf8)
                .alias("home_team")
            )
        if "away_team" in working.columns:
            working = working.with_columns(
                pl.col("away_team")
                .map_elements(normalize_baseball_team, return_dtype=pl.Utf8)
                .alias("away_team")
            )
        if "league" not in working.columns and "home_team" in working.columns:
            working = working.with_columns(
                pl.col("home_team")
                .map_elements(lambda t: resolve_team_league(t) or "MLB", return_dtype=pl.Utf8)
                .alias("league")
            )
        validated = validate_canonical_event_frame(
            working,
            allow_default_event_seq=True,
            check_unique_event_key=True,
        )
        invalid_seq = validated.filter(~pl.col("event_seq").is_in([1, 2]))
        if invalid_seq.height > 0:
            bad_vals = invalid_seq["event_seq"].unique().to_list()
            raise SchemaValidationError(
                "Baseball 'event_seq' must be 1 or 2 (single game or doubleheader), "
                f"got {bad_vals}."
            )
        if "league" in validated.columns:
            invalid_lg = validated.filter(~pl.col("league").is_in(["MLB", "NPB"]))
            if invalid_lg.height > 0:
                bad_lgs = invalid_lg["league"].unique().to_list()
                raise SchemaValidationError(
                    f"Baseball 'league' must be 'MLB' or 'NPB', got {bad_lgs}."
                )
        return validated

    def validate_participant_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize a pitcher or batter appearance DataFrame."""
        working = df
        if "event_id" not in working.columns and "game_id" in working.columns:
            working = working.with_columns(pl.col("game_id").alias("event_id"))
        if "entity_id" not in working.columns and "team" in working.columns:
            working = working.with_columns(
                pl.col("team")
                .map_elements(normalize_baseball_team, return_dtype=pl.Utf8)
                .alias("entity_id")
            )
        if "participant_id" not in working.columns and "player_id" in working.columns:
            working = working.with_columns(pl.col("player_id").alias("participant_id"))
        if "event_seq" not in working.columns:
            working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))

        validated = validate_canonical_participant_frame(
            working,
            check_unique_participant_key=False,
        )
        # In doubleheaders, uniqueness is on (event_id, event_seq, entity_id, participant_id)
        key_cols = ["event_id", "event_seq", "entity_id", "participant_id"]
        n_rows = validated.height
        n_unique = validated.select(key_cols).n_unique()
        if n_unique != n_rows:
            raise SchemaValidationError(
                "Duplicate (event_id, event_seq, entity_id, participant_id) keys detected: "
                f"{n_rows - n_unique} duplicates."
            )
        return validated
