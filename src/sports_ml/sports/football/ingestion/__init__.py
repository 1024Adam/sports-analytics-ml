"""NFL ingestion package (`src/sports_ml/sports/football/ingestion/`)."""

from __future__ import annotations

from sports_ml.sports.football.ingestion.health import (
    NFLVERSE_REQUIRED_COLUMNS,
    check_football_data_sources,
    check_nflverse_asset,
    missing_required_columns,
)
from sports_ml.sports.football.ingestion.ingestor import (
    FootballIngestionReport,
    NflverseIngestor,
)
from sports_ml.sports.football.ingestion.lineups import (
    FootballStartingQBResolver,
    QBCandidate,
    ResolvedStartingQB,
    derive_starting_qbs_from_pbp,
)
from sports_ml.sports.football.ingestion.nflverse_client import (
    NFLVERSE_ASSETS,
    NflverseAsset,
    NflverseAssetUnavailableError,
    NflverseClient,
    asset_available_for_season,
    derive_active_nfl_season,
)
from sports_ml.sports.football.ingestion.reconciler import (
    QUARANTINED_SCHEDULE_COLUMNS,
    FootballSilverReconciler,
    assert_games_quarantined,
    build_availability_snapshots,
    build_games_frame,
    build_player_game_logs,
    build_qb_game_logs,
    build_team_game_logs,
)

__all__ = [
    "FootballIngestionReport",
    "FootballSilverReconciler",
    "FootballStartingQBResolver",
    "NFLVERSE_ASSETS",
    "NFLVERSE_REQUIRED_COLUMNS",
    "NflverseAsset",
    "NflverseAssetUnavailableError",
    "NflverseClient",
    "NflverseIngestor",
    "QBCandidate",
    "QUARANTINED_SCHEDULE_COLUMNS",
    "ResolvedStartingQB",
    "asset_available_for_season",
    "assert_games_quarantined",
    "build_availability_snapshots",
    "build_games_frame",
    "build_player_game_logs",
    "build_qb_game_logs",
    "build_team_game_logs",
    "check_football_data_sources",
    "check_nflverse_asset",
    "derive_active_nfl_season",
    "derive_starting_qbs_from_pbp",
    "missing_required_columns",
]
