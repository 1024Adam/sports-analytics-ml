"""Stage 2 player feature/label column contracts (constants only; no feature code)."""

from __future__ import annotations

from typing import Final

STAGE2_POSITIONS: Final[tuple[str, ...]] = ("QB", "RB", "WR", "TE")
POSITION_ALIASES: Final[dict[str, str]] = {"FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}

STAGE2_IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "event_seq",
    "season",
    "season_type",
    "week",
    "start_time_utc",
    "game_state",
    "team",
    "opponent",
    "player_id",
    "player_name",
    "position",
)

STAGE2_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "actual_pass_att",
    "actual_completions",
    "actual_pass_yds",
    "actual_pass_td",
    "actual_int",
    "actual_carries",
    "actual_rush_yds",
    "actual_rush_td",
    "actual_targets",
    "actual_rec",
    "actual_rec_yds",
    "actual_rec_td",
)

STAGE2_PLAYER_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "is_home_flag",
    "pos_is_qb",
    "pos_is_rb",
    "pos_is_wr",
    "pos_is_te",
    "p_active",
    "p_active_is_imputed",
    "is_debut_flag",
    "career_games_pre",
    "draft_capital_factor",
    # Role / opportunity shares (shrunk, decayed, strictly pre-game)
    "snap_share",
    "target_share",
    "air_yards_share",
    "wopr",
    "route_participation",
    "rz_target_share",
    "carry_share",
    "gl_carry_share",
    "rz_carry_share",
    "pass_att_share",
    # Shrunk per-opportunity rates
    "qb_comp_rate",
    "qb_yds_per_att",
    "qb_td_per_att",
    "qb_int_per_att",
    "qb_rush_att_per_game",
    "qb_rush_ypc",
    "qb_rush_td_per_carry",
    "rush_ypc",
    "rush_td_per_carry",
    "catch_rate",
    "rec_yds_per_target",
    "rec_td_per_target",
    # Opposing defense vs position (index, 1.0 = league average)
    "opp_def_pass_yds_idx",
    "opp_def_rush_yds_idx",
    "opp_def_rec_yds_idx",
    # Team context (pre-game Elo-based; Stage 1 projections are model inputs)
    "team_elo_win_prob",
    "team_pace_plays_pg",
    "team_pass_rate",
)

STAGE2_FEATURE_COLUMNS_CANDIDATE: Final[tuple[str, ...]] = STAGE2_PLAYER_FEATURE_COLUMNS
