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
from sports_ml.sports.soccer.ingestion.non_league_fixtures import (
    CUP_COMPETITION_TYPES,
    EUROPEAN_COMPETITION_TYPES,
    VALID_COMPETITION_TYPES,
    NonLeagueFixturesIngestor,
    build_epl_schedule_rows_from_matches,
    normalize_competition_type,
    parse_non_league_fixtures_csv,
    parse_non_league_fixtures_records,
    resolve_fixture_venue_coords,
)
from sports_ml.sports.soccer.ingestion.reconciler import SoccerSilverReconciler
from sports_ml.sports.soccer.ingestion.understat import (
    UnderstatClient,
    parse_understat_player_matches_payload,
    parse_understat_team_matches_payload,
)

__all__ = [
    "CUP_COMPETITION_TYPES",
    "EUROPEAN_COMPETITION_TYPES",
    "FPLAPIClient",
    "FootballDataClient",
    "NonLeagueFixturesIngestor",
    "ResolvedSoccerTeamLineup",
    "SoccerLineupStatus",
    "SoccerPreMatchLineupResolver",
    "SoccerSilverReconciler",
    "UnderstatClient",
    "VALID_COMPETITION_TYPES",
    "build_epl_schedule_rows_from_matches",
    "check_football_data_epl_source",
    "check_fpl_bootstrap_and_fixtures_source",
    "check_soccer_data_sources",
    "check_understat_epl_source",
    "derive_active_epl_season",
    "make_canonical_match_id",
    "normalize_competition_type",
    "normalize_soccer_club",
    "normalize_soccer_player_name",
    "parse_bootstrap_squad_df",
    "parse_bootstrap_teams_map",
    "parse_element_summary_payload",
    "parse_fixtures_payload",
    "parse_football_data_csv",
    "parse_non_league_fixtures_csv",
    "parse_non_league_fixtures_records",
    "parse_understat_player_matches_payload",
    "parse_understat_team_matches_payload",
    "resolve_fixture_venue_coords",
    "season_to_football_data_code",
]
