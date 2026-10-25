"""Dynamic pairwise Elo and multi-entrant skill rating engines (`core/ratings.py`).

Records pre-event ratings strictly before applying post-event updates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import polars as pl
from pydantic import BaseModel, ConfigDict

from sports_ml.core.contracts import StarterModifierHook, VenueModifierHook
from sports_ml.core.temporal import sort_events_chronologically


class EloConfig(BaseModel):
    """Configuration for pairwise Elo rating updates."""

    model_config = ConfigDict(frozen=True)

    base_rating: float = 1500.0
    home_field_advantage: float = 35.0
    k_factor: float = 6.0
    shootout_winner_score: float = 0.60
    offseason_retention: float = 0.67
    use_mov_multiplier: bool = True


def compute_elo_win_probability(elo_diff: float) -> float:
    """Compute analytical logistic Elo win probability `1 / (1 + 10 ** (-elo_diff / 400))`."""
    return 1.0 / (1.0 + math.pow(10.0, -elo_diff / 400.0))


def compute_mov_multiplier(goal_diff: int | float, elo_diff: float) -> float:
    """Compute MOV multiplier `ln(|GoalDiff| + 1) * 2.2 / (0.001 * |elo_diff| + 2.2)`."""
    eff_diff = max(abs(float(goal_diff)), 1.0)
    return math.log(eff_diff + 1.0) * (2.2 / (0.001 * abs(elo_diff) + 2.2))


class PairwiseEloEngine:
    """Dynamic pairwise Elo engine supporting HFA, MOV scaling, OT/SO dampening, and regression."""

    def __init__(
        self,
        config: EloConfig | None = None,
        starter_hook: StarterModifierHook | None = None,
        venue_hook: VenueModifierHook | None = None,
    ) -> None:
        self.config = config or EloConfig()
        self.starter_hook = starter_hook
        self.venue_hook = venue_hook

    def regress_rating(self, rating: float) -> float:
        """Apply between-season regression toward `base_rating` (`0.67 * E + 0.33 * 1500`)."""
        ret = self.config.offseason_retention
        return ret * rating + (1.0 - ret) * self.config.base_rating

    def regress_ratings_offseason(self, ratings: Mapping[str, float]) -> dict[str, float]:
        """Regress a dictionary of entity Elo ratings toward `base_rating` between seasons."""
        return {entity: self.regress_rating(rating) for entity, rating in ratings.items()}

    def compute_ratings(
        self,
        games_df: pl.DataFrame,
        *,
        initial_ratings: Mapping[str, float] | None = None,
    ) -> pl.DataFrame:
        """Compute leak-free pre-game Elo ratings, win probabilities, and post-game updates.

        Input DataFrame must contain `start_time_utc`, `season`, `home_team` (or `home_entity`),
        `away_team` (or `away_entity`), and `event_id` (or `game_id`).
        Outcome columns (`home_goals`/`away_goals` or `home_score`/`away_score` or `home_win`)
        may be null for upcoming (`FUT`) games, in which case pre-game ratings are emitted
        without mutating post-game ratings.
        """
        sorted_df = sort_events_chronologically(games_df)
        if sorted_df.is_empty():
            return sorted_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("elo_home_pre"),
                pl.lit(None, dtype=pl.Float64).alias("elo_away_pre"),
                pl.lit(None, dtype=pl.Float64).alias("elo_diff_pre"),
                pl.lit(None, dtype=pl.Float64).alias("elo_home_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("elo_away_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("elo_home_post"),
                pl.lit(None, dtype=pl.Float64).alias("elo_away_post"),
            )

        home_col = "home_team" if "home_team" in sorted_df.columns else "home_entity"
        away_col = "away_team" if "away_team" in sorted_df.columns else "away_entity"
        home_score_col = "home_goals" if "home_goals" in sorted_df.columns else "home_score"
        away_score_col = "away_goals" if "away_goals" in sorted_df.columns else "away_score"

        ratings: dict[str, float] = dict(initial_ratings) if initial_ratings is not None else {}
        current_season: int | None = None

        elo_home_pre_list: list[float] = []
        elo_away_pre_list: list[float] = []
        elo_diff_pre_list: list[float] = []
        elo_home_prob_list: list[float] = []
        elo_away_prob_list: list[float] = []
        elo_home_post_list: list[float] = []
        elo_away_post_list: list[float] = []

        for row in sorted_df.iter_rows(named=True):
            season_val = int(row["season"])
            if current_season is not None and season_val != current_season:
                ratings = self.regress_ratings_offseason(ratings)
            current_season = season_val

            event_id = int(row["event_id"])
            home_entity = str(row[home_col])
            away_entity = str(row[away_col])

            # 1. Record strictly pre-event ratings
            r_home_pre = ratings.get(home_entity, self.config.base_rating)
            r_away_pre = ratings.get(away_entity, self.config.base_rating)

            starter_delta_home = float(row.get("home_starter_elo_delta") or 0.0)
            starter_delta_away = float(row.get("away_starter_elo_delta") or 0.0)
            if self.starter_hook is not None:
                starter_delta_home += self.starter_hook.get_starter_elo_delta(event_id, home_entity)
                starter_delta_away += self.starter_hook.get_starter_elo_delta(event_id, away_entity)

            venue_delta = float(row.get("venue_elo_delta") or 0.0)
            if self.venue_hook is not None:
                venue_delta += self.venue_hook.get_venue_elo_delta(
                    event_id, home_entity, away_entity
                )

            is_neutral = bool(row.get("is_neutral_site", False))
            hfa = 0.0 if is_neutral else self.config.home_field_advantage

            elo_diff = (
                (r_home_pre + starter_delta_home)
                + hfa
                + venue_delta
                - (r_away_pre + starter_delta_away)
            )
            p_home = compute_elo_win_probability(elo_diff)
            p_away = 1.0 - p_home

            elo_home_pre_list.append(r_home_pre)
            elo_away_pre_list.append(r_away_pre)
            elo_diff_pre_list.append(elo_diff)
            elo_home_prob_list.append(p_home)
            elo_away_prob_list.append(p_away)

            # 2. Check if game is completed before applying post-event update
            game_state = str(row.get("game_state") or "FINAL").upper()
            h_score = row.get(home_score_col)
            a_score = row.get(away_score_col)
            home_win_flag = row.get("home_win")

            is_completed = game_state in {"FINAL", "OFF"} and (
                (h_score is not None and a_score is not None) or home_win_flag is not None
            )

            if not is_completed:
                elo_home_post_list.append(r_home_pre)
                elo_away_post_list.append(r_away_pre)
                continue

            finish_type = str(row.get("finish_period_type") or "REG").upper()
            if h_score is not None and a_score is not None:
                h_goals = float(h_score)
                a_goals = float(a_score)
                goal_diff = abs(h_goals - a_goals)
                home_won = h_goals > a_goals
                is_tie = h_goals == a_goals
            else:
                goal_diff = 1.0
                home_won = bool(home_win_flag is not None and int(home_win_flag) == 1)
                is_tie = False

            if is_tie:
                actual_s_home = 0.5
            elif finish_type == "SO":
                actual_s_home = (
                    self.config.shootout_winner_score
                    if home_won
                    else (1.0 - self.config.shootout_winner_score)
                )
            else:
                actual_s_home = 1.0 if home_won else 0.0

            mov_mult = (
                compute_mov_multiplier(goal_diff, elo_diff)
                if (self.config.use_mov_multiplier and not is_tie)
                else 1.0
            )

            delta = self.config.k_factor * mov_mult * (actual_s_home - p_home)
            r_home_post = r_home_pre + delta
            r_away_post = r_away_pre - delta

            ratings[home_entity] = r_home_post
            ratings[away_entity] = r_away_post

            elo_home_post_list.append(r_home_post)
            elo_away_post_list.append(r_away_post)

        return sorted_df.with_columns(
            pl.Series("elo_home_pre", elo_home_pre_list, dtype=pl.Float64),
            pl.Series("elo_away_pre", elo_away_pre_list, dtype=pl.Float64),
            pl.Series("elo_diff_pre", elo_diff_pre_list, dtype=pl.Float64),
            pl.Series("elo_home_win_prob", elo_home_prob_list, dtype=pl.Float64),
            pl.Series("elo_away_win_prob", elo_away_prob_list, dtype=pl.Float64),
            pl.Series("elo_home_post", elo_home_post_list, dtype=pl.Float64),
            pl.Series("elo_away_post", elo_away_post_list, dtype=pl.Float64),
        )


class FieldSkillConfig(BaseModel):
    """Configuration for multi-entrant field skill ratings (e.g., PGA tournaments)."""

    model_config = ConfigDict(frozen=True)

    base_skill: float = 0.0
    learning_rate: float = 0.15
    temperature: float = 1.0
    offseason_retention: float = 0.75


class MultiEntrantSkillEngine:
    """Extensible field-adjusted skill rating engine for `multi_entrant_field` events."""

    def __init__(self, config: FieldSkillConfig | None = None) -> None:
        self.config = config or FieldSkillConfig()

    def compute_ratings(
        self,
        field_df: pl.DataFrame,
        *,
        entity_col: str = "entity_id",
        performance_col: str = "strokes_gained",
    ) -> pl.DataFrame:
        """Compute leak-free pre-event skill ratings and Plackett-Luce field win probabilities."""
        sorted_df = sort_events_chronologically(field_df, extra_tiebreakers=[entity_col])
        if sorted_df.is_empty():
            return sorted_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("skill_rating_pre"),
                pl.lit(None, dtype=pl.Float64).alias("field_win_prob"),
                pl.lit(None, dtype=pl.Float64).alias("skill_rating_post"),
            )

        skills: dict[str, float] = {}
        current_season: int | None = None

        rows = sorted_df.to_dicts()
        n_rows = len(rows)
        skill_pre_out = [0.0] * n_rows
        win_prob_out = [0.0] * n_rows
        skill_post_out = [0.0] * n_rows

        idx = 0
        while idx < n_rows:
            season_val = int(rows[idx]["season"])
            if current_season is not None and season_val != current_season:
                ret = self.config.offseason_retention
                base = self.config.base_skill
                skills = {k: ret * v + (1.0 - ret) * base for k, v in skills.items()}
            current_season = season_val

            # Group all entrants sharing the same (start_time_utc, event_seq, event_id)
            group_key = (
                rows[idx]["start_time_utc"],
                rows[idx]["event_seq"],
                rows[idx]["event_id"],
            )
            end_idx = idx + 1
            while (
                end_idx < n_rows
                and (
                    rows[end_idx]["start_time_utc"],
                    rows[end_idx]["event_seq"],
                    rows[end_idx]["event_id"],
                )
                == group_key
            ):
                end_idx += 1

            group_indices = list(range(idx, end_idx))
            group_pre = [
                skills.get(str(rows[i][entity_col]), self.config.base_skill) for i in group_indices
            ]
            max_skill = max(group_pre)
            temp = max(self.config.temperature, 1e-6)
            exp_weights = [math.exp((s - max_skill) / temp) for s in group_pre]
            total_weight = sum(exp_weights)

            for local_i, row_i in enumerate(group_indices):
                skill_pre_out[row_i] = group_pre[local_i]
                win_prob_out[row_i] = exp_weights[local_i] / total_weight

            # Update skills strictly after recording pre-event state for the entire field
            perf_vals = [rows[i].get(performance_col) for i in group_indices]
            valid_perfs = [float(v) for v in perf_vals if v is not None]
            field_mean_perf = sum(valid_perfs) / len(valid_perfs) if valid_perfs else 0.0

            lr = self.config.learning_rate
            for local_i, row_i in enumerate(group_indices):
                ent = str(rows[row_i][entity_col])
                raw_perf = rows[row_i].get(performance_col)
                if raw_perf is None:
                    skill_post_out[row_i] = group_pre[local_i]
                else:
                    rel_perf = float(raw_perf) - field_mean_perf
                    new_skill = (1.0 - lr) * group_pre[local_i] + lr * rel_perf
                    skills[ent] = new_skill
                    skill_post_out[row_i] = new_skill

            idx = end_idx

        return sorted_df.with_columns(
            pl.Series("skill_rating_pre", skill_pre_out, dtype=pl.Float64),
            pl.Series("field_win_prob", win_prob_out, dtype=pl.Float64),
            pl.Series("skill_rating_post", skill_post_out, dtype=pl.Float64),
        )
