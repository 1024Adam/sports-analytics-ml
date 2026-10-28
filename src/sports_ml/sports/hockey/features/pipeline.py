"""NHL feature pipeline and hard market-odds anti-contamination firewall (`features/pipeline.py`).

Enforces that no `market_*` or `*odds*` columns ever enter `STAGE1_FEATURE_COLUMNS` or
`STAGE2_FEATURE_COLUMNS`, guaranteeing the NHL models remain 100% first-principles
fundamental models while preserving `market_*` columns for evaluation-only benchmarking.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from sports_ml.core.market import is_forbidden_market_column
from sports_ml.sports.hockey.features.goalie import (
    build_goalie_rolling_features,
    resolve_team_starting_goalie_features,
)
from sports_ml.sports.hockey.features.player import build_stage2_player_features
from sports_ml.sports.hockey.features.roster import (
    build_roster_strength_features,
    build_skater_rolling_features,
)
from sports_ml.sports.hockey.features.team import (
    build_stage1_game_features,
    build_team_rolling_features,
    compute_special_teams_net_edge,
)
from sports_ml.sports.hockey.models.stage1_game import STAGE1_CANDIDATE_FEATURE_COLS
from sports_ml.sports.hockey.models.stage2_player import STAGE2_SKATER_FEATURE_COLS


def filter_forbidden_market_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Strip any `market_*` or `*odds*` column names from a candidate feature list."""
    return tuple(c for c in columns if not is_forbidden_market_column(c))


def assert_zero_odds_contamination(columns: Sequence[str]) -> None:
    """Raise `ValueError` if any column in `columns` violates the market-odds firewall."""
    contaminated = [c for c in columns if is_forbidden_market_column(c)]
    if contaminated:
        raise ValueError(
            "Market/odds contamination detected in feature columns: "
            f"{contaminated}. Market columns are strictly evaluation-only."
        )


def strip_market_columns_from_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return a copy of `df` with all `market_*` and `*odds*` columns removed."""
    drop_cols = [c for c in df.columns if is_forbidden_market_column(c)]
    return df.drop(drop_cols) if drop_cols else df


STAGE1_FEATURE_COLUMNS: Final[tuple[str, ...]] = filter_forbidden_market_columns(
    STAGE1_CANDIDATE_FEATURE_COLS
)
STAGE2_FEATURE_COLUMNS: Final[tuple[str, ...]] = filter_forbidden_market_columns(
    STAGE2_SKATER_FEATURE_COLS
)

assert_zero_odds_contamination(STAGE1_FEATURE_COLUMNS)
assert_zero_odds_contamination(STAGE2_FEATURE_COLUMNS)

__all__ = [
    "STAGE1_FEATURE_COLUMNS",
    "STAGE2_FEATURE_COLUMNS",
    "assert_zero_odds_contamination",
    "build_goalie_rolling_features",
    "build_roster_strength_features",
    "build_skater_rolling_features",
    "build_stage1_game_features",
    "build_stage2_player_features",
    "build_team_rolling_features",
    "compute_special_teams_net_edge",
    "filter_forbidden_market_columns",
    "resolve_team_starting_goalie_features",
    "strip_market_columns_from_frame",
]
