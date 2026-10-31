"""MLB & NPB Baseball ingestion package (`sports/baseball/ingestion/`)."""

from __future__ import annotations

from sports_ml.sports.baseball.ingestion.health import (
    check_baseball_data_sources,
    check_mlb_statsapi_source,
    check_npb_source,
    check_statcast_source,
)
from sports_ml.sports.baseball.ingestion.lineups import (
    BaseballLineupStatus,
    BaseballPreGameLineupResolver,
    BaseballStarterStatus,
    ResolvedBaseballTeamLineup,
)
from sports_ml.sports.baseball.ingestion.mlb_api import (
    MLB_STATSAPI_BASE_URL,
    MLBStatsAPIClient,
    derive_active_baseball_season,
    parse_mlb_boxscore_payload,
    parse_mlb_roster_payload,
    parse_mlb_schedule_payload,
    parse_mlb_schedule_team_logs,
)
from sports_ml.sports.baseball.ingestion.npb_client import (
    NPB_BIS_BASE_URL,
    NPB_GAME_ID_OFFSET,
    NPBClient,
    build_npb_team_logs_from_games,
    namespace_npb_game_id,
    parse_npb_boxscore_payload,
    parse_npb_calendar_html,
    parse_npb_roster_payload,
    parse_npb_schedule_payload,
    resolve_npb_sub_league_and_dh,
    resolve_npb_venue,
)
from sports_ml.sports.baseball.ingestion.npb_micro import (
    compute_npb_dips_and_tactical_micro_features,
    compute_npb_venue_park_interaction,
)
from sports_ml.sports.baseball.ingestion.odds import (
    BASEBALL_ODDS_SCHEMA,
    BaseballOddsIngestor,
    parse_baseball_historical_odds_csv,
)
from sports_ml.sports.baseball.ingestion.reconciler import BaseballSilverReconciler
from sports_ml.sports.baseball.ingestion.statcast import (
    STATCAST_BASE_URL,
    StatcastClient,
    parse_statcast_batter_metrics_csv,
    parse_statcast_pitcher_metrics_csv,
    parse_statcast_team_metrics_csv,
)
from sports_ml.sports.baseball.ingestion.statcast_micro import (
    SILVER_BATTER_MICRO_LOGS_SCHEMA,
    SILVER_PITCHER_MICRO_LOGS_SCHEMA,
    aggregate_statcast_pitches_to_micro_logs,
    compute_pregame_batter_woba_minus_xwoba_luck,
    compute_pregame_pitcher_velo_delta,
)

__all__ = [
    "BASEBALL_ODDS_SCHEMA",
    "BaseballLineupStatus",
    "BaseballOddsIngestor",
    "BaseballPreGameLineupResolver",
    "BaseballSilverReconciler",
    "BaseballStarterStatus",
    "MLBStatsAPIClient",
    "MLB_STATSAPI_BASE_URL",
    "NPBClient",
    "NPB_BIS_BASE_URL",
    "NPB_GAME_ID_OFFSET",
    "ResolvedBaseballTeamLineup",
    "SILVER_BATTER_MICRO_LOGS_SCHEMA",
    "SILVER_PITCHER_MICRO_LOGS_SCHEMA",
    "STATCAST_BASE_URL",
    "StatcastClient",
    "aggregate_statcast_pitches_to_micro_logs",
    "build_npb_team_logs_from_games",
    "check_baseball_data_sources",
    "check_mlb_statsapi_source",
    "check_npb_source",
    "check_statcast_source",
    "compute_npb_dips_and_tactical_micro_features",
    "compute_npb_venue_park_interaction",
    "compute_pregame_batter_woba_minus_xwoba_luck",
    "compute_pregame_pitcher_velo_delta",
    "derive_active_baseball_season",
    "namespace_npb_game_id",
    "parse_baseball_historical_odds_csv",
    "parse_mlb_boxscore_payload",
    "parse_mlb_roster_payload",
    "parse_mlb_schedule_payload",
    "parse_mlb_schedule_team_logs",
    "parse_npb_boxscore_payload",
    "parse_npb_calendar_html",
    "parse_npb_roster_payload",
    "parse_npb_schedule_payload",
    "parse_statcast_batter_metrics_csv",
    "parse_statcast_pitcher_metrics_csv",
    "parse_statcast_team_metrics_csv",
    "resolve_npb_sub_league_and_dh",
    "resolve_npb_venue",
]
