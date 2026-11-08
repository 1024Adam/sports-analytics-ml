"""Starred-teams resolver for EPL soccer (`TeamResolver` implementation)."""

from __future__ import annotations

from sports_ml.core.starred import ResolvedTeam, StarredConfigError
from sports_ml.sports.soccer.constants import get_stadium_metadata, normalize_soccer_club


class SoccerTeamResolver:
    """Validate and normalize EPL soccer team codes/names."""

    sport = "soccer"

    def resolve(self, team: str, league: str | None) -> ResolvedTeam:
        if league not in (None, "epl"):
            raise StarredConfigError(f"Unsupported soccer league '{league}'. Only 'epl' exists.")
        code = normalize_soccer_club(team)
        meta = get_stadium_metadata(code)
        if meta is None:
            raise StarredConfigError(f"Unknown EPL club '{team}'.")
        return ResolvedTeam(sport=self.sport, league="epl", team=code, full_name=meta.team_name)
