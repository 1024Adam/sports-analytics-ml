"""Starting-QB pre-game features and the QB value feeding the Elo hook (`features/qb.py`).

QB identity is first-class. Each quarterback carries a **player-level** prior that follows him
across teams: decayed career sums (strictly from games before `t`) shrunk toward a
replacement-level prior, which for rookies/debutants is lifted by draft capital
(`draft_round`, `draft_pick`). The composite `qb_value = EPA/dropback + w * CPOE` is the
quantity `FootballQBEloHook` turns into an Elo adjustment against the team's baseline.

The starter for game `t` is the Tier-1/Tier-2 resolved QB (an explicit `starting_qbs` override),
else `games.{home,away}_qb_id`; for unplayed games without a starter the team incumbent is
projected and `qb_is_imputed` is set. Features use only each QB's games before `t`.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import polars as pl

from sports_ml.core.shrinkage import shrink_scalar
from sports_ml.sports.football.adapter import gsis_id_to_int
from sports_ml.sports.football.constants import normalize_football_team

QB_GAME_DECAY: Final[float] = 0.96
QB_K_DROPBACKS: Final[float] = 250.0
QB_K_ATTEMPTS: Final[float] = 200.0
QB_K_GAMES: Final[float] = 6.0
CPOE_EPA_WEIGHT: Final[float] = 0.01
REPLACEMENT_QB_EPA: Final[float] = -0.10
REPLACEMENT_QB_CPOE: Final[float] = -2.0
DRAFT_EPA_BONUS: Final[float] = 0.15
DRAFT_CPOE_BONUS: Final[float] = 3.0
DRAFT_PICK_DECAY: Final[float] = 40.0
LEAGUE_SACK_RATE: Final[float] = 0.07
LEAGUE_AIR_YARDS_PER_ATTEMPT: Final[float] = 7.5
INCUMBENT_LOOKBACK_GAMES: Final[int] = 17
TEAM_BASELINE_ALPHA: Final[float] = 0.15

QB_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "qb_epa_per_dropback",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_air_yards_per_attempt",
    "qb_rush_epa",
    "qb_games_with_team",
    "qb_career_games",
    "qb_is_rookie_or_debut",
    "qb_is_backup",
    "qb_is_imputed",
    "qb_value",
    "team_qb_baseline_value",
    "qb_value_over_baseline",
)


def draft_capital_factor(draft_pick: float | None) -> float:
    """Return `exp(-(pick - 1) / 40)` in `(0, 1]` (0.0 for undrafted / unknown picks)."""
    if draft_pick is None or not math.isfinite(float(draft_pick)) or draft_pick < 1:
        return 0.0
    return math.exp(-(float(draft_pick) - 1.0) / DRAFT_PICK_DECAY)


def qb_prior_rates(draft_pick: float | None) -> tuple[float, float]:
    """Return `(prior_epa_per_dropback, prior_cpoe)` for a QB with the given draft pick."""
    f = draft_capital_factor(draft_pick)
    return (
        REPLACEMENT_QB_EPA + DRAFT_EPA_BONUS * f,
        REPLACEMENT_QB_CPOE + DRAFT_CPOE_BONUS * f,
    )


def compose_qb_value(epa_per_dropback: float, cpoe: float) -> float:
    """Composite `EPA/dropback + CPOE_EPA_WEIGHT * CPOE` feeding the Elo hook."""
    return float(epa_per_dropback) + CPOE_EPA_WEIGHT * float(cpoe)


@dataclass
class _QBState:
    epa: float = 0.0
    dropbacks: float = 0.0
    cpoe: float = 0.0
    cpoe_n: float = 0.0
    sacks: float = 0.0
    attempts: float = 0.0
    air_yards: float = 0.0
    rush_epa: float = 0.0
    games: float = 0.0
    career_games: int = 0
    games_with_team: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def decay(self, g: float) -> None:
        self.epa *= g
        self.dropbacks *= g
        self.cpoe *= g
        self.cpoe_n *= g
        self.sacks *= g
        self.attempts *= g
        self.air_yards *= g
        self.rush_epa *= g
        self.games *= g


def _draft_lookup(draft_picks: pl.DataFrame | None) -> dict[int, tuple[int | None, float | None]]:
    """Map `qb_id -> (draft_season, draft_pick)` from an nflverse `draft_picks` frame."""
    if draft_picks is None or draft_picks.is_empty():
        return {}
    out: dict[int, tuple[int | None, float | None]] = {}
    for r in draft_picks.iter_rows(named=True):
        raw = r.get("qb_id", r.get("gsis_id"))
        if raw is None:
            continue
        try:
            qid = gsis_id_to_int(raw)
        except (TypeError, ValueError):
            continue
        season = r.get("season")
        pick = r.get("pick")
        out[qid] = (
            int(season) if season is not None else None,
            float(pick) if pick is not None else None,
        )
    return out


def _starter_overrides(starting_qbs: pl.DataFrame | None) -> dict[tuple[int, str], int]:
    if starting_qbs is None or starting_qbs.is_empty():
        return {}
    return {
        (int(r["game_id"]), normalize_football_team(str(r["team"]))): int(r["qb_id"])
        for r in starting_qbs.iter_rows(named=True)
        if r["qb_id"] is not None
    }


def build_qb_features(
    games_df: pl.DataFrame,
    qb_logs: pl.DataFrame,
    *,
    draft_picks: pl.DataFrame | None = None,
    starting_qbs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return per-`(game_id, team)` starting-QB features (`qb_id` + `QB_FEATURE_COLUMNS`)."""
    draft = _draft_lookup(draft_picks)
    overrides = _starter_overrides(starting_qbs)

    logs_by_game: dict[int, list[dict[str, object]]] = defaultdict(list)
    for r in qb_logs.iter_rows(named=True):
        logs_by_game[int(r["game_id"])].append(r)

    ordered = games_df.sort("start_time_utc", "event_seq", "game_id")
    states: dict[int, _QBState] = defaultdict(_QBState)
    team_recent: dict[str, deque[dict[int, float]]] = defaultdict(
        lambda: deque(maxlen=INCUMBENT_LOOKBACK_GAMES)
    )
    team_baseline: dict[str, float] = {}
    out: list[dict[str, object]] = []

    def incumbent(team: str) -> int | None:
        totals: dict[int, float] = defaultdict(float)
        for game_totals in team_recent[team]:
            for qid, db in game_totals.items():
                totals[qid] += db
        if not totals:
            return None
        return max(totals, key=lambda q: (totals[q], -q))

    def features(qid: int | None, team: str, season: int, *, imputed: bool) -> dict[str, object]:
        draft_season, pick = draft.get(qid, (None, None)) if qid is not None else (None, None)
        prior_epa, prior_cpoe = qb_prior_rates(pick)
        st = states[qid] if qid is not None else _QBState()
        epa = shrink_scalar(
            st.epa / st.dropbacks if st.dropbacks > 0 else None,
            st.dropbacks,
            prior_epa,
            QB_K_DROPBACKS,
        )
        cpoe = shrink_scalar(
            st.cpoe / st.cpoe_n if st.cpoe_n > 0 else None, st.cpoe_n, prior_cpoe, QB_K_DROPBACKS
        )
        sack = shrink_scalar(
            st.sacks / st.dropbacks if st.dropbacks > 0 else None,
            st.dropbacks,
            LEAGUE_SACK_RATE,
            QB_K_DROPBACKS,
        )
        air = shrink_scalar(
            st.air_yards / st.attempts if st.attempts > 0 else None,
            st.attempts,
            LEAGUE_AIR_YARDS_PER_ATTEMPT,
            QB_K_ATTEMPTS,
        )
        rush = shrink_scalar(
            st.rush_epa / st.games if st.games > 0 else None, st.games, 0.0, QB_K_GAMES
        )
        value = compose_qb_value(epa, cpoe)
        base = team_baseline.get(team, value)
        inc = incumbent(team)
        return {
            "qb_id": qid,
            "qb_epa_per_dropback": epa,
            "qb_cpoe": cpoe,
            "qb_sack_rate": sack,
            "qb_air_yards_per_attempt": air,
            "qb_rush_epa": rush,
            "qb_games_with_team": st.games_with_team.get(team, 0),
            "qb_career_games": st.career_games,
            "qb_is_rookie_or_debut": st.career_games == 0
            or (draft_season is not None and draft_season == season),
            "qb_is_backup": bool(inc is not None and qid is not None and qid != inc),
            "qb_is_imputed": imputed,
            "qb_value": value,
            "team_qb_baseline_value": base,
            "qb_value_over_baseline": value - base,
        }

    for g in ordered.iter_rows(named=True):
        gid = int(g["game_id"])
        season = int(g["season"])
        resolved: list[tuple[str, dict[str, object]]] = []
        for team, col in ((g["home_team"], "home_qb_id"), (g["away_team"], "away_qb_id")):
            team = str(team)
            qid = overrides.get((gid, team))
            if qid is None and g.get(col) is not None:
                qid = int(g[col])
            imputed = False
            if qid is None:
                qid = incumbent(team)
                imputed = True
            row = {"game_id": gid, "team": team, **features(qid, team, season, imputed=imputed)}
            resolved.append((team, row))
            out.append(row)

        # Update state strictly after features for both teams are recorded
        if str(g["game_state"]).upper() == "FINAL":
            for team, row in resolved:
                if row["qb_id"] is None:
                    continue
                value = float(row["qb_value"])  # type: ignore[arg-type]
                prev = team_baseline.get(team)
                team_baseline[team] = (
                    value
                    if prev is None
                    else TEAM_BASELINE_ALPHA * value + (1.0 - TEAM_BASELINE_ALPHA) * prev
                )
        played: dict[str, dict[int, float]] = defaultdict(dict)
        for log in logs_by_game.get(gid, []):
            qid = int(log["qb_id"])  # type: ignore[call-overload]
            team = normalize_football_team(str(log["team"]))
            st = states[qid]
            st.decay(QB_GAME_DECAY)
            st.epa += float(log["qb_epa_total"] or 0.0)  # type: ignore[arg-type]
            st.dropbacks += float(log["dropbacks"] or 0.0)  # type: ignore[arg-type]
            st.cpoe += float(log["cpoe_total"] or 0.0)  # type: ignore[arg-type]
            st.cpoe_n += float(log["cpoe_plays"] or 0.0)  # type: ignore[arg-type]
            st.sacks += float(log["sacks"] or 0.0)  # type: ignore[arg-type]
            st.attempts += float(log["pass_attempts"] or 0.0)  # type: ignore[arg-type]
            st.air_yards += float(log["air_yards_total"] or 0.0)  # type: ignore[arg-type]
            st.rush_epa += float(log.get("rush_epa_total") or 0.0)  # type: ignore[arg-type]
            st.games += 1.0
            st.career_games += 1
            st.games_with_team[team] += 1
            played[team][qid] = played[team].get(qid, 0.0) + float(log["dropbacks"] or 0.0)  # type: ignore[arg-type]
        for team, dropbacks_by_qb in played.items():
            team_recent[team].append(dropbacks_by_qb)

    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        "game_id": pl.Int64,
        "team": pl.Utf8,
        "qb_id": pl.Int64,
        "qb_epa_per_dropback": pl.Float64,
        "qb_cpoe": pl.Float64,
        "qb_sack_rate": pl.Float64,
        "qb_air_yards_per_attempt": pl.Float64,
        "qb_rush_epa": pl.Float64,
        "qb_games_with_team": pl.Int32,
        "qb_career_games": pl.Int32,
        "qb_is_rookie_or_debut": pl.Boolean,
        "qb_is_backup": pl.Boolean,
        "qb_is_imputed": pl.Boolean,
        "qb_value": pl.Float64,
        "team_qb_baseline_value": pl.Float64,
        "qb_value_over_baseline": pl.Float64,
    }
    return pl.DataFrame(out, schema=schema).sort("game_id", "team")


def build_qb_elo_value_maps(
    qb_features: pl.DataFrame,
) -> tuple[Mapping[tuple[int, str], float], Mapping[tuple[int, str], float]]:
    """Return `(starter_values, baseline_values)` keyed `(game_id, team)` for the Elo QB hook."""
    starters: dict[tuple[int, str], float] = {}
    baselines: dict[tuple[int, str], float] = {}
    for r in qb_features.select(
        "game_id", "team", "qb_value", "team_qb_baseline_value", "qb_is_imputed"
    ).iter_rows(named=True):
        key = (int(r["game_id"]), str(r["team"]))
        starters[key] = float(r["qb_value"])
        baselines[key] = float(r["team_qb_baseline_value"])
    return starters, baselines
