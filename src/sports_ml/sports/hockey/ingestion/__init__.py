"""NHL data ingestion and Silver lakehouse reconciliation (`sports/hockey/ingestion/`)."""

from sports_ml.sports.hockey.ingestion.health import (
    DataSourceHealthResult,
    check_dailyfaceoff_source,
    check_hockey_data_sources,
    check_moneypuck_source,
    check_nhl_roster_source,
    check_nhl_schedule_source,
)
from sports_ml.sports.hockey.ingestion.lineups import (
    GoalieConfirmationStatus,
    PreGameLineupResolver,
    ResolvedTeamLineup,
    normalize_player_name,
)
from sports_ml.sports.hockey.ingestion.moneypuck import (
    MoneyPuckClient,
    parse_moneypuck_goalies_csv,
    parse_moneypuck_skaters_csv,
    parse_moneypuck_teams_csv,
)
from sports_ml.sports.hockey.ingestion.nhl_api import (
    NHL_WEB_API_BASE_URL,
    NHLWebAPIClient,
    derive_active_nhl_season,
    parse_boxscore_payload,
    parse_roster_payload,
    parse_schedule_payload,
    parse_toi_minutes,
)
from sports_ml.sports.hockey.ingestion.reconciler import HockeySilverReconciler

__all__ = [
    "DataSourceHealthResult",
    "GoalieConfirmationStatus",
    "HockeySilverReconciler",
    "MoneyPuckClient",
    "NHLWebAPIClient",
    "NHL_WEB_API_BASE_URL",
    "PreGameLineupResolver",
    "ResolvedTeamLineup",
    "check_dailyfaceoff_source",
    "check_hockey_data_sources",
    "check_moneypuck_source",
    "check_nhl_roster_source",
    "check_nhl_schedule_source",
    "derive_active_nhl_season",
    "normalize_player_name",
    "parse_boxscore_payload",
    "parse_moneypuck_goalies_csv",
    "parse_moneypuck_skaters_csv",
    "parse_moneypuck_teams_csv",
    "parse_roster_payload",
    "parse_schedule_payload",
    "parse_toi_minutes",
]
