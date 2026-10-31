"""Baseball feature pipeline and market-odds anti-contamination firewall (`pipeline.py`).

Enforces that no `market_*` or `*odds*` columns ever enter Stage 1 or Stage 2
feature lists (`BASEBALL_STAGE1_FEATURE_COLUMNS`, `BASEBALL_STAGE2_PITCHER_FEATURE_COLUMNS`,
`BASEBALL_STAGE2_BATTER_FEATURE_COLUMNS`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from sports_ml.core.market import is_forbidden_market_column
from sports_ml.sports.baseball.features.bullpen import build_bullpen_pregame_features
from sports_ml.sports.baseball.features.lineup import (
    build_batter_rolling_features,
    build_platoon_lineup_strength_features,
)
from sports_ml.sports.baseball.features.pitcher import (
    build_starting_pitcher_rolling_features,
    resolve_game_starting_pitcher_features,
)
from sports_ml.sports.baseball.features.player import (
    build_baseball_stage2_batter_features,
    build_baseball_stage2_pitcher_features,
)
from sports_ml.sports.baseball.features.team import (
    build_baseball_stage1_game_features,
    build_baseball_team_rolling_features,
)

BASEBALL_STAGE1_CANDIDATE_FEATURE_COLS: Final[tuple[str, ...]] = (
    "elo_home_pre",
    "elo_away_pre",
    "diff_elo_pre",
    "elo_home_win_prob",
    "home_sp_xfip_shrunk",
    "away_sp_xfip_shrunk",
    "diff_sp_xfip_edge",
    "diff_sp_fip_edge",
    "home_sp_k_minus_bb_pct_shrunk",
    "away_sp_k_minus_bb_pct_shrunk",
    "diff_sp_k_minus_bb_pct",
    "home_sp_gb_pct_shrunk",
    "away_sp_gb_pct_shrunk",
    "home_sp_expected_outs_ceiling",
    "away_sp_expected_outs_ceiling",
    "diff_sp_expected_outs",
    "home_sp_days_rest",
    "away_sp_days_rest",
    "home_sp_xwoba_against_shrunk",
    "away_sp_xwoba_against_shrunk",
    "home_sp_swstr_pct_shrunk",
    "away_sp_swstr_pct_shrunk",
    "home_sp_velo_delta_mph",
    "away_sp_velo_delta_mph",
    "diff_sp_velo_delta",
    "diff_sp_swstr_pct",
    "home_bp_xfip_shrunk",
    "away_bp_xfip_shrunk",
    "diff_bp_xfip_edge",
    "home_bp_k_minus_bb_pct_shrunk",
    "away_bp_k_minus_bb_pct_shrunk",
    "home_bp_pitches_last_1d",
    "away_bp_pitches_last_1d",
    "home_bp_pitches_last_3d",
    "away_bp_pitches_last_3d",
    "diff_bp_pitches_last_3d",
    "home_high_leverage_arms_unavailable_count",
    "away_high_leverage_arms_unavailable_count",
    "diff_hl_arms_unavailable",
    "home_lineup_platoon_woba",
    "away_lineup_platoon_woba",
    "diff_lineup_platoon_woba",
    "home_lineup_platoon_iso",
    "away_lineup_platoon_iso",
    "diff_lineup_platoon_iso",
    "home_lineup_platoon_k_pct",
    "away_lineup_platoon_k_pct",
    "diff_lineup_platoon_k_pct",
    "home_lineup_same_hand_share",
    "away_lineup_same_hand_share",
    "home_missing_regular_pa_woba",
    "away_missing_regular_pa_woba",
    "diff_missing_regular_woba",
    "diff_lineup_xwoba",
    "diff_woba_minus_xwoba_regression",
    "home_team_runs_for_per_g_shrunk",
    "away_team_runs_for_per_g_shrunk",
    "home_team_runs_against_per_g_shrunk",
    "away_team_runs_against_per_g_shrunk",
    "diff_team_run_diff",
    "diff_team_woba",
    "diff_pythag_luck",
    "diff_baseruns_luck",
    "venue_run_factor",
    "venue_hr_factor_matchup",
    "is_dome_flag",
    "is_neutral_or_regional_flag",
    "is_koshien_summer_road_trip_flag",
    "is_interleague_flag",
    "uses_dh_flag",
    "is_doubleheader_g2",
    "diff_rest_days",
    "away_travel_km",
)

BASEBALL_STAGE2_PITCHER_CANDIDATE_FEATURE_COLS: Final[tuple[str, ...]] = (
    "is_home_flag",
    "sp_is_lhp",
    "sp_xfip_shrunk",
    "sp_fip_shrunk",
    "sp_era_shrunk",
    "sp_k_pct_shrunk",
    "sp_bb_pct_shrunk",
    "sp_k_minus_bb_pct_shrunk",
    "sp_whip_shrunk",
    "sp_hr_per_9_shrunk",
    "sp_gb_pct_shrunk",
    "sp_days_rest",
    "sp_rolling_pitches_per_start",
    "sp_expected_outs_ceiling",
    "sp_xwoba_against_shrunk",
    "sp_swstr_pct_shrunk",
    "sp_velo_delta_mph",
    "opp_lineup_platoon_woba",
    "opp_lineup_platoon_k_pct",
    "opp_lineup_platoon_bb_pct",
    "venue_run_factor",
    "is_dome_flag",
    "stage1_team_win_prob",
    "stage1_opp_expected_runs",
)

BASEBALL_STAGE2_BATTER_CANDIDATE_FEATURE_COLS: Final[tuple[str, ...]] = (
    "is_home_flag",
    "is_starter_flag",
    "batting_slot_num",
    "slot_pa_weight",
    "batter_woba_shrunk",
    "batter_platoon_woba",
    "batter_iso_shrunk",
    "batter_k_pct_shrunk",
    "batter_bb_pct_shrunk",
    "batter_hr_per_pa_shrunk",
    "batter_xwoba_shrunk",
    "batter_barrel_rate_shrunk",
    "is_opposite_hand_matchup",
    "opp_sp_xfip_shrunk",
    "opp_sp_k_pct_shrunk",
    "opp_sp_bb_pct_shrunk",
    "opp_bp_xfip_shrunk",
    "venue_run_factor",
    "venue_hr_factor_batter",
    "stage1_team_win_prob",
    "stage1_team_expected_runs",
)


def filter_forbidden_market_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Strip any `market_*` or `*odds*` column names from a candidate feature list."""
    return tuple(c for c in columns if not is_forbidden_market_column(c))


def assert_zero_odds_contamination(columns: Sequence[str]) -> None:
    """Raise `ValueError` if any column in `columns` violates the market-odds firewall."""
    contaminated = [c for c in columns if is_forbidden_market_column(c)]
    if contaminated:
        raise ValueError(
            "Market/odds contamination detected in baseball feature columns: "
            f"{contaminated}. Market columns are strictly evaluation-only."
        )


def strip_market_columns_from_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return a copy of `df` with all `market_*` and `*odds*` columns removed."""
    drop_cols = [c for c in df.columns if is_forbidden_market_column(c)]
    return df.drop(drop_cols) if drop_cols else df


STAGE1_FEATURE_COLUMNS: Final[tuple[str, ...]] = filter_forbidden_market_columns(
    BASEBALL_STAGE1_CANDIDATE_FEATURE_COLS
)
STAGE2_PITCHER_FEATURE_COLUMNS: Final[tuple[str, ...]] = filter_forbidden_market_columns(
    BASEBALL_STAGE2_PITCHER_CANDIDATE_FEATURE_COLS
)
STAGE2_BATTER_FEATURE_COLUMNS: Final[tuple[str, ...]] = filter_forbidden_market_columns(
    BASEBALL_STAGE2_BATTER_CANDIDATE_FEATURE_COLS
)

assert_zero_odds_contamination(STAGE1_FEATURE_COLUMNS)
assert_zero_odds_contamination(STAGE2_PITCHER_FEATURE_COLUMNS)
assert_zero_odds_contamination(STAGE2_BATTER_FEATURE_COLUMNS)

__all__ = [
    "BASEBALL_STAGE1_CANDIDATE_FEATURE_COLS",
    "BASEBALL_STAGE2_BATTER_CANDIDATE_FEATURE_COLS",
    "BASEBALL_STAGE2_PITCHER_CANDIDATE_FEATURE_COLS",
    "STAGE1_FEATURE_COLUMNS",
    "STAGE2_BATTER_FEATURE_COLUMNS",
    "STAGE2_PITCHER_FEATURE_COLUMNS",
    "assert_zero_odds_contamination",
    "build_baseball_stage1_game_features",
    "build_baseball_stage2_batter_features",
    "build_baseball_stage2_pitcher_features",
    "build_baseball_team_rolling_features",
    "build_batter_rolling_features",
    "build_bullpen_pregame_features",
    "build_platoon_lineup_strength_features",
    "build_starting_pitcher_rolling_features",
    "filter_forbidden_market_columns",
    "resolve_game_starting_pitcher_features",
    "strip_market_columns_from_frame",
]
