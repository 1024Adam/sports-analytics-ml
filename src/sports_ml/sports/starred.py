"""Registers every sport plugin's starred-teams resolver with `core.starred`."""

from __future__ import annotations

from sports_ml.core.starred import register_team_resolver
from sports_ml.sports.baseball.starred import BaseballTeamResolver
from sports_ml.sports.football.starred import FootballTeamResolver
from sports_ml.sports.hockey.starred import HockeyTeamResolver
from sports_ml.sports.soccer.starred import SoccerTeamResolver


def register_default_team_resolvers() -> None:
    """Idempotently register the hockey, soccer, baseball, and football resolvers."""
    for resolver in (
        HockeyTeamResolver(),
        SoccerTeamResolver(),
        BaseballTeamResolver(),
        FootballTeamResolver(),
    ):
        register_team_resolver(resolver.sport, resolver)


__all__ = ["register_default_team_resolvers"]
