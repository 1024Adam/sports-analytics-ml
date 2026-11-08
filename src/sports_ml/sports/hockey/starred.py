"""Starred-teams resolver for NHL hockey (`TeamResolver` implementation)."""

from __future__ import annotations

from sports_ml.core.starred import ResolvedTeam, StarredConfigError
from sports_ml.sports.hockey.constants import get_arena_metadata, normalize_tricode


class HockeyTeamResolver:
    """Validate and normalize NHL hockey team codes/names."""

    sport = "hockey"

    def resolve(self, team: str, league: str | None) -> ResolvedTeam:
        if league not in (None, "nhl"):
            raise StarredConfigError(f"Unsupported hockey league '{league}'. Only 'nhl' exists.")
        code = normalize_tricode(team)
        meta = get_arena_metadata(code)
        if meta is None:
            raise StarredConfigError(f"Unknown NHL team '{team}'.")
        return ResolvedTeam(sport=self.sport, league="nhl", team=code, full_name=meta.team_name)
