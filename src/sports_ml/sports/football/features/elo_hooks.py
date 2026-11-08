"""Starting-QB and venue Elo modifier hooks plus the margin-of-victory Elo driver.

Implements `StarterModifierHook` and `VenueModifierHook` from `core/contracts.py` for NFL
games and wires them into `PairwiseEloEngine` with `FOOTBALL_ELO_CONFIG`
(MOV multiplier `ln(|PD| + 1) * 2.2 / (0.001 * dElo_winner + 2.2)`, ties update at 0.5,
playoff K multiplier, neutral-site HFA = 0, ~1/3 offseason regression).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping
from typing import Final

import polars as pl

from sports_ml.core.ratings import PairwiseEloEngine
from sports_ml.sports.football.constants import (
    ALTITUDE_ELO_PER_KM,
    ALTITUDE_THRESHOLD_M,
    DOME_HOME_ELO_EDGE,
    FOOTBALL_ELO_CONFIG,
    MAX_REST_DIFF_DAYS,
    PLAYOFF_SEASON_TYPES,
    REST_ELO_PER_DAY,
    get_stadium_metadata,
    is_neutral_game,
    normalize_football_team,
)

# Elo points per 1.0 unit of QBValue (shrunk rolling EPA+CPOE composite, Milestone 24).
DEFAULT_KAPPA_QB: Final[float] = 200.0
MAX_QB_ELO_DELTA: Final[float] = 250.0

QBValueFn = Callable[[int, str], float | None]


def _finite_or_none(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return number


def compute_qb_elo_delta(
    starter_value: float,
    baseline_value: float,
    *,
    kappa_qb: float = DEFAULT_KAPPA_QB,
    max_abs_delta: float = MAX_QB_ELO_DELTA,
) -> float:
    """Return `clip(kappa_qb * (QBValue_starter - QBValue_baseline), +/- max_abs_delta)`."""
    raw = kappa_qb * (float(starter_value) - float(baseline_value))
    return float(min(max(raw, -max_abs_delta), max_abs_delta))


class FootballQBEloHook:
    """`StarterModifierHook`: `dE_QB = kappa_QB * (QBValue_starter - QBValue_team_baseline)`.

    `starter_value_fn(event_id, team)` and `baseline_value_fn(event_id, team)` may be any
    callables returning a finite float (or None when unknown, yielding a 0.0 adjustment),
    so the hook is testable with stubs before Milestone 24 supplies real QB values.
    """

    def __init__(
        self,
        starter_value_fn: QBValueFn,
        baseline_value_fn: QBValueFn,
        *,
        kappa_qb: float = DEFAULT_KAPPA_QB,
        max_abs_delta: float = MAX_QB_ELO_DELTA,
    ) -> None:
        self._starter_value_fn = starter_value_fn
        self._baseline_value_fn = baseline_value_fn
        self.kappa_qb = kappa_qb
        self.max_abs_delta = max_abs_delta

    @classmethod
    def from_mappings(
        cls,
        starter_values: Mapping[tuple[int, str], float],
        baseline_values: Mapping[tuple[int, str], float],
        *,
        kappa_qb: float = DEFAULT_KAPPA_QB,
        max_abs_delta: float = MAX_QB_ELO_DELTA,
    ) -> FootballQBEloHook:
        """Build a hook from `(event_id, team) -> value` lookup tables."""
        starters = {(int(e), normalize_football_team(t)): v for (e, t), v in starter_values.items()}
        baselines = {
            (int(e), normalize_football_team(t)): v for (e, t), v in baseline_values.items()
        }
        return cls(
            lambda e, t: starters.get((e, t)),
            lambda e, t: baselines.get((e, t)),
            kappa_qb=kappa_qb,
            max_abs_delta=max_abs_delta,
        )

    def get_starter_elo_delta(self, event_id: int, entity_id: str) -> float:
        """Return the pre-game Elo adjustment for `entity_id`'s starting quarterback."""
        team = normalize_football_team(entity_id)
        starter = _finite_or_none(self._starter_value_fn(int(event_id), team), "QBValue starter")
        baseline = _finite_or_none(self._baseline_value_fn(int(event_id), team), "QBValue baseline")
        if starter is None or baseline is None:
            return 0.0
        return compute_qb_elo_delta(
            starter, baseline, kappa_qb=self.kappa_qb, max_abs_delta=self.max_abs_delta
        )


class FootballVenueEloHook:
    """`VenueModifierHook`: altitude/dome environment edges and rest-differential adjustment.

    Neutral-site HFA removal is done by the engine through the `is_neutral_site` column
    (see `prepare_football_elo_games`); for neutral events this hook zeroes the
    home-team-specific altitude/dome edges and still applies the rest differential.
    """

    def __init__(
        self,
        venue_by_event_id: Mapping[int, str] | None = None,
        *,
        neutral_event_ids: Collection[int] = (),
        rest_days_by_event_team: Mapping[tuple[int, str], float] | None = None,
        rest_elo_per_day: float = REST_ELO_PER_DAY,
        max_rest_diff_days: float = MAX_REST_DIFF_DAYS,
        altitude_elo_per_km: float = ALTITUDE_ELO_PER_KM,
        dome_home_edge: float = DOME_HOME_ELO_EDGE,
    ) -> None:
        self._venue_by_event = {
            int(e): str(v).strip().upper() for e, v in (venue_by_event_id or {}).items()
        }
        self._neutral_events = {int(e) for e in neutral_event_ids}
        self._rest_days = {
            (int(e), normalize_football_team(t)): float(d)
            for (e, t), d in (rest_days_by_event_team or {}).items()
        }
        self.rest_elo_per_day = rest_elo_per_day
        self.max_rest_diff_days = max_rest_diff_days
        self.altitude_elo_per_km = altitude_elo_per_km
        self.dome_home_edge = dome_home_edge

    def _environment_delta(self, venue_id: str, home: str, away: str) -> float:
        venue = get_stadium_metadata(venue_id)
        if venue is None or home not in venue.primary_teams:
            return 0.0
        away_home_venue = get_stadium_metadata(away)

        home_excess = max(0.0, venue.altitude_m - ALTITUDE_THRESHOLD_M) / 1000.0
        away_excess = (
            max(0.0, away_home_venue.altitude_m - ALTITUDE_THRESHOLD_M) / 1000.0
            if away_home_venue is not None
            else 0.0
        )
        delta = self.altitude_elo_per_km * max(0.0, home_excess - away_excess)

        if venue.is_covered and away_home_venue is not None and not away_home_venue.is_covered:
            delta += self.dome_home_edge
        return delta

    def get_venue_elo_delta(self, event_id: int, home_entity: str, away_entity: str) -> float:
        """Return the pre-game Elo adjustment (home perspective) for venue and rest effects."""
        home = normalize_football_team(home_entity)
        away = normalize_football_team(away_entity)
        eid = int(event_id)
        delta = 0.0

        venue_id = self._venue_by_event.get(eid)
        neutral = eid in self._neutral_events
        if venue_id is not None and not neutral:
            venue = get_stadium_metadata(venue_id)
            neutral = venue is not None and venue.is_neutral_site
        if venue_id is not None and not neutral:
            delta += self._environment_delta(venue_id, home, away)

        home_rest = self._rest_days.get((eid, home))
        away_rest = self._rest_days.get((eid, away))
        if home_rest is not None and away_rest is not None:
            diff = max(
                -self.max_rest_diff_days, min(self.max_rest_diff_days, home_rest - away_rest)
            )
            delta += self.rest_elo_per_day * diff
        return float(delta)


def build_rest_days_lookup(schedule_features: pl.DataFrame) -> dict[tuple[int, str], float]:
    """Map `(event_id, team) -> days_rest` from `build_football_schedule_features` output."""
    lookup: dict[tuple[int, str], float] = {}
    for row in schedule_features.select("event_id", "team", "days_rest").iter_rows(named=True):
        if row["days_rest"] is not None:
            lookup[(int(row["event_id"]), str(row["team"]))] = float(row["days_rest"])
    return lookup


def prepare_football_elo_games(games_df: pl.DataFrame) -> pl.DataFrame:
    """Normalize franchises and add the `is_neutral_site` / `is_playoff` flags the engine reads."""
    working = games_df
    if "event_id" not in working.columns:
        working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    working = working.with_columns(
        pl.col("home_team").map_elements(normalize_football_team, return_dtype=pl.Utf8),
        pl.col("away_team").map_elements(normalize_football_team, return_dtype=pl.Utf8),
    )
    season_types = [str(s).upper() for s in working["season_type"].to_list()]
    venue_ids: list[str | None] = (
        working["venue_id"].to_list() if "venue_id" in working.columns else [None] * working.height
    )
    return working.with_columns(
        pl.Series(
            "is_neutral_site",
            [is_neutral_game(st, v) for st, v in zip(season_types, venue_ids, strict=True)],
            dtype=pl.Boolean,
        ),
        pl.Series(
            "is_playoff",
            [st in PLAYOFF_SEASON_TYPES for st in season_types],
            dtype=pl.Boolean,
        ),
    )


def compute_football_elo_ratings(
    games_df: pl.DataFrame,
    *,
    qb_hook: FootballQBEloHook | None = None,
    venue_hook: FootballVenueEloHook | None = None,
    initial_ratings: Mapping[str, float] | None = None,
) -> pl.DataFrame:
    """Compute leak-free pre-game NFL Elo with MOV, 0.5 ties, QB, and venue adjustments."""
    engine = PairwiseEloEngine(FOOTBALL_ELO_CONFIG, starter_hook=qb_hook, venue_hook=venue_hook)
    return engine.compute_ratings(
        prepare_football_elo_games(games_df), initial_ratings=initial_ratings
    )
