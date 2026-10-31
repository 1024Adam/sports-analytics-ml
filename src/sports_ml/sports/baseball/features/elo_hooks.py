"""Starting pitcher and ballpark Elo modifier hooks (`sports/baseball/features/elo_hooks.py`).

Implements `StarterModifierHook` and `VenueModifierHook` from `core/contracts.py`
for MLB and NPB games.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from sports_ml.sports.baseball.constants import (
    get_ballpark_metadata,
    get_baseball_club_metadata,
    get_league_sabermetric_priors,
    normalize_baseball_team,
)

# Elo points per 1.00 run of starting pitcher xFIP better than team/league baseline
DEFAULT_SP_XFIP_ELO_SCALE: Final[float] = 32.0
MAX_STARTER_ELO_DELTA: Final[float] = 45.0

# Provincial / regional NPB neutral stadium or displaced home venue HFA dampening
DEFAULT_REGIONAL_VENUE_HFA_DAMPENING: Final[float] = -12.0


def compute_starter_xfip_elo_delta(
    starter_xfip: float,
    baseline_xfip: float,
    *,
    kappa_sp: float = DEFAULT_SP_XFIP_ELO_SCALE,
    max_abs_delta: float = MAX_STARTER_ELO_DELTA,
) -> float:
    """Compute pre-game Elo adjustment for today's starting pitcher.

    Formula: `clip(kappa_sp * (baseline_xfip - starter_xfip), -max_abs_delta, +max_abs_delta)`.
    A starter with lower (better) xFIP than the rotation baseline yields a positive
    Elo delta.
    """
    raw_delta = kappa_sp * (float(baseline_xfip) - float(starter_xfip))
    return float(min(max(raw_delta, -max_abs_delta), max_abs_delta))


class BaseballStarterEloHook:
    """`StarterModifierHook` mapping `(event_id, team)` to pre-game starting pitcher Elo delta."""

    def __init__(
        self,
        starter_xfip_by_event_entity: Mapping[tuple[int, str], float] | None = None,
        team_rotation_baseline_xfip: Mapping[tuple[int, str], float] | None = None,
        *,
        kappa_sp: float = DEFAULT_SP_XFIP_ELO_SCALE,
        max_abs_delta: float = MAX_STARTER_ELO_DELTA,
    ) -> None:
        self._starter_xfip: dict[tuple[int, str], float] = {
            (int(ev), normalize_baseball_team(team)): float(val)
            for (ev, team), val in (starter_xfip_by_event_entity or {}).items()
        }
        self._team_baseline: dict[tuple[int, str], float] = {
            (int(ev), normalize_baseball_team(team)): float(val)
            for (ev, team), val in (team_rotation_baseline_xfip or {}).items()
        }
        self.kappa_sp = kappa_sp
        self.max_abs_delta = max_abs_delta

    def get_starter_elo_delta(self, event_id: int, entity_id: str) -> float:
        """Return pre-event Elo point adjustment for `entity_id`'s starting pitcher."""
        norm_team = normalize_baseball_team(entity_id)
        key = (int(event_id), norm_team)
        sp_xfip = self._starter_xfip.get(key)
        if sp_xfip is None:
            return 0.0

        baseline = self._team_baseline.get(key)
        if baseline is None:
            club = get_baseball_club_metadata(norm_team)
            lg_code = club.league if club is not None else "MLB"
            baseline = get_league_sabermetric_priors(lg_code).lg_sp_xfip

        return compute_starter_xfip_elo_delta(
            sp_xfip,
            baseline,
            kappa_sp=self.kappa_sp,
            max_abs_delta=self.max_abs_delta,
        )


class BaseballVenueEloHook:
    """`VenueModifierHook` adjusting pre-game Elo for NPB regional venues & displaced home games."""

    def __init__(
        self,
        venue_by_event_id: Mapping[int, str] | None = None,
        *,
        regional_hfa_dampening: float = DEFAULT_REGIONAL_VENUE_HFA_DAMPENING,
    ) -> None:
        self._venue_by_event: dict[int, str] = {
            int(ev): str(venue_id).strip().upper()
            for ev, venue_id in (venue_by_event_id or {}).items()
        }
        self.regional_hfa_dampening = regional_hfa_dampening

    def get_venue_elo_delta(self, event_id: int, home_entity: str, away_entity: str) -> float:
        """Return pre-event Elo adjustment for venue displacement / regional neutral stadiums."""
        _ = away_entity
        venue_id = self._venue_by_event.get(int(event_id))
        if venue_id is None:
            return 0.0

        park = get_ballpark_metadata(venue_id)
        if park is None:
            return 0.0

        norm_home = normalize_baseball_team(home_entity)
        club = get_baseball_club_metadata(norm_home)
        is_displaced_home = club is not None and park.venue_id != club.primary_venue_id

        if park.is_neutral_or_regional or is_displaced_home:
            return float(self.regional_hfa_dampening)
        return 0.0
