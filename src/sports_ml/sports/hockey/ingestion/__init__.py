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
from sports_ml.sports.hockey.ingestion.odds import (
    NHL_ODDS_SCHEMA,
    NHLOddsIngestor,
    parse_espn_core_odds_items,
    parse_nhl_historical_odds_csv,
    parse_the_odds_api_payload,
)
from sports_ml.sports.hockey.ingestion.pbp_shots import (
    NHLPbpShotsIngestor,
    aggregate_shot_events_to_micro_logs,
    compute_5v5_score_venue_weight,
    compute_shot_geometry,
    estimate_shot_xg_from_coords,
    parse_moneypuck_shots_csv,
    parse_nhl_pbp_payload,
)
from sports_ml.sports.hockey.ingestion.reconciler import HockeySilverReconciler
from sports_ml.sports.hockey.ingestion.shifts import (
    compute_shift_overlap_qoc_qot,
    enrich_skater_micro_logs_with_shifts,
)

__all__ = [
    "DataSourceHealthResult",
    "GoalieConfirmationStatus",
    "HockeySilverReconciler",
    "MoneyPuckClient",
    "NHLOddsIngestor",
    "NHLPbpShotsIngestor",
    "NHLWebAPIClient",
    "NHL_ODDS_SCHEMA",
    "NHL_WEB_API_BASE_URL",
    "PreGameLineupResolver",
    "ResolvedTeamLineup",
    "aggregate_shot_events_to_micro_logs",
    "check_dailyfaceoff_source",
    "check_hockey_data_sources",
    "check_moneypuck_source",
    "check_nhl_roster_source",
    "check_nhl_schedule_source",
    "compute_5v5_score_venue_weight",
    "compute_shift_overlap_qoc_qot",
    "compute_shot_geometry",
    "derive_active_nhl_season",
    "enrich_skater_micro_logs_with_shifts",
    "estimate_shot_xg_from_coords",
    "normalize_player_name",
    "parse_boxscore_payload",
    "parse_espn_core_odds_items",
    "parse_moneypuck_goalies_csv",
    "parse_moneypuck_shots_csv",
    "parse_moneypuck_skaters_csv",
    "parse_moneypuck_teams_csv",
    "parse_nhl_historical_odds_csv",
    "parse_nhl_pbp_payload",
    "parse_roster_payload",
    "parse_schedule_payload",
    "parse_the_odds_api_payload",
    "parse_toi_minutes",
]
