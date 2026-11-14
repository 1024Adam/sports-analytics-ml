"""Starred-teams resolver for NFL football (`TeamResolver` implementation)."""

from __future__ import annotations

from sports_ml.core.starred import ResolvedTeam, StarredConfigError
from sports_ml.sports.football.constants import get_football_team_metadata, normalize_football_team


class FootballTeamResolver:
    """Validate and normalize NFL football team codes/names."""

    sport = "football"

    def resolve(self, team: str, league: str | None) -> ResolvedTeam:
        if league not in (None, "nfl"):
            raise StarredConfigError(
                f"Unsupported football league '{league}'. Only 'nfl' is implemented."
            )
        code = normalize_football_team(team)
        meta = get_football_team_metadata(code)
        if meta is None:
            raise StarredConfigError(f"Unknown NFL team '{team}'.")
        return ResolvedTeam(sport=self.sport, league="nfl", team=code, full_name=meta.team_name)
