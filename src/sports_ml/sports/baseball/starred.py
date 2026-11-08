"""Starred-teams resolver for MLB / NPB baseball (`TeamResolver` implementation)."""

from __future__ import annotations

from sports_ml.core.starred import ResolvedTeam, StarredConfigError
from sports_ml.sports.baseball.constants import (
    get_baseball_club_metadata,
    normalize_baseball_team,
)


class BaseballTeamResolver:
    """Validate and normalize MLB / NPB baseball team codes/names."""

    sport = "baseball"

    def resolve(self, team: str, league: str | None) -> ResolvedTeam:
        if league not in (None, "mlb", "npb"):
            raise StarredConfigError(
                f"Unsupported baseball league '{league}'. Use 'mlb' or 'npb' (or omit it)."
            )
        code = normalize_baseball_team(team)
        meta = get_baseball_club_metadata(code)
        if meta is None:
            raise StarredConfigError(f"Unknown MLB/NPB club '{team}'.")
        resolved = str(meta.league).lower()
        if league is not None and league != resolved:
            raise StarredConfigError(
                f"Club '{code}' plays in {resolved.upper()}, not {league.upper()}."
            )
        return ResolvedTeam(sport=self.sport, league=resolved, team=code, full_name=meta.team_name)
