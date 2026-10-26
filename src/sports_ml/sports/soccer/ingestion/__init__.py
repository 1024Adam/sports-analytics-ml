"""EPL Soccer ingestion package (`src/sports_ml/sports/soccer/ingestion/`)."""

from sports_ml.sports.soccer.ingestion.football_data import (
    FootballDataClient,
    parse_football_data_csv,
    season_to_football_data_code,
)
from sports_ml.sports.soccer.ingestion.fpl_api import (
    FPLAPIClient,
    derive_active_epl_season,
    make_canonical_match_id,
    parse_bootstrap_squad_df,
    parse_bootstrap_teams_map,
    parse_element_summary_payload,
    parse_fixtures_payload,
)
from sports_ml.sports.soccer.ingestion.health import (
    check_football_data_epl_source,
    check_fpl_bootstrap_and_fixtures_source,
    check_soccer_data_sources,
    check_understat_epl_source,
)
from sports_ml.sports.soccer.ingestion.lineups import (
    ResolvedSoccerTeamLineup,
    SoccerLineupStatus,
    SoccerPreMatchLineupResolver,
    normalize_soccer_player_name,
)
from sports_ml.sports.soccer.ingestion.reconciler import SoccerSilverReconciler
from sports_ml.sports.soccer.ingestion.understat import (
    UnderstatClient,
    parse_understat_player_matches_payload,
    parse_understat_team_matches_payload,
)

__all__ = [
    "FPLAPIClient",
    "FootballDataClient",
    "ResolvedSoccerTeamLineup",
    "SoccerLineupStatus",
    "SoccerPreMatchLineupResolver",
    "SoccerSilverReconciler",
    "UnderstatClient",
    "check_football_data_epl_source",
    "check_fpl_bootstrap_and_fixtures_source",
    "check_soccer_data_sources",
    "check_understat_epl_source",
    "derive_active_epl_season",
    "make_canonical_match_id",
    "normalize_soccer_player_name",
    "parse_bootstrap_squad_df",
    "parse_bootstrap_teams_map",
    "parse_element_summary_payload",
    "parse_fixtures_payload",
    "parse_football_data_csv",
    "parse_understat_player_matches_payload",
    "parse_understat_team_matches_payload",
    "season_to_football_data_code",
]
