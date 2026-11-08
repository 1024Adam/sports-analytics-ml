"""Season/date-window sync orchestrator for nflverse -> Silver (`football/ingestion/ingestor.py`).

Supports `--from-year`, `--current-season`, and date-window syncs. Re-running a window upserts
by canonical keys (idempotent), so late stat corrections simply overwrite earlier rows. Sources
that lag or predate their availability window degrade to `has_*` flags instead of failing.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sports_ml.sports.football.constants import (
    canonical_game_id_from_nflverse,
    normalize_football_team,
)
from sports_ml.sports.football.ingestion.lineups import derive_starting_qbs_from_pbp
from sports_ml.sports.football.ingestion.nflverse_client import (
    NflverseClient,
    derive_active_nfl_season,
)
from sports_ml.sports.football.ingestion.reconciler import FootballSilverReconciler


class FootballIngestionReport(BaseModel):
    """Summary of one `NflverseIngestor.ingest` run."""

    model_config = ConfigDict(frozen=True)

    seasons: tuple[int, ...]
    games_rows: int = 0
    team_log_rows: int = 0
    qb_log_rows: int = 0
    player_log_rows: int = 0
    availability_rows: int = 0
    has_snap_counts: dict[int, bool] = Field(default_factory=dict)
    has_ngs: dict[int, bool] = Field(default_factory=dict)
    has_injury_report: dict[int, bool] = Field(default_factory=dict)
    missing_assets: tuple[str, ...] = ()


def _present(frame: pl.DataFrame | None) -> bool:
    return frame is not None and not frame.is_empty()


class NflverseIngestor:
    """Orchestrates `NflverseClient` downloads into `FootballSilverReconciler` upserts."""

    def __init__(
        self,
        client: NflverseClient,
        reconciler: FootballSilverReconciler,
        *,
        active_season: int | None = None,
    ) -> None:
        self.client = client
        self.reconciler = reconciler
        self.active_season = active_season or derive_active_nfl_season()

    def resolve_seasons(
        self,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        current_season: bool = False,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[int]:
        """Resolve the sorted set of seasons implied by the sync options."""
        seasons: set[int] = set()
        if from_year is not None:
            last = to_year if to_year is not None else self.active_season
            seasons.update(range(from_year, last + 1))
        if current_season:
            seasons.add(self.active_season)
        if start_date is not None or end_date is not None:
            lo = derive_active_nfl_season(start_date or end_date)
            hi = derive_active_nfl_season(end_date or start_date)
            seasons.update(range(lo, hi + 1))
        if not seasons:
            raise ValueError(
                "Specify from_year, current_season, or a start_date/end_date window to ingest."
            )
        return sorted(seasons)

    def ingest(
        self,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        current_season: bool = False,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> FootballIngestionReport:
        """Sync the requested seasons/window and upsert all Silver tables."""
        seasons = self.resolve_seasons(
            from_year=from_year,
            to_year=to_year,
            current_season=current_season,
            start_date=start_date,
            end_date=end_date,
        )
        windowed = start_date is not None or end_date is not None
        schedule = self.client.fetch_schedule(ttl_seconds=21600.0 if current_season else None)
        schedule = schedule.filter(pl.col("season").is_in(seasons))
        if windowed:
            schedule = schedule.filter(
                (
                    pl.col("gameday").cast(pl.Utf8).str.slice(0, 10)
                    >= (start_date or date.min).isoformat()
                )
                & (
                    pl.col("gameday").cast(pl.Utf8).str.slice(0, 10)
                    <= (end_date or date.max).isoformat()
                )
            )

        players = self.client.try_fetch_asset("players")
        missing: set[str] = set() if players is not None else {"players"}
        counts = {"games": 0, "team": 0, "qb": 0, "player": 0, "avail": 0}
        has_snaps: dict[int, bool] = {}
        has_ngs: dict[int, bool] = {}
        has_inj: dict[int, bool] = {}

        for season in seasons:
            active = season == self.active_season
            season_sched = schedule.filter(pl.col("season") == season)
            if season_sched.is_empty():
                continue
            pbp = self.client.try_fetch_asset("pbp", season, is_active_season=active)
            if pbp is not None and windowed:
                ids = set(season_sched["game_id"].to_list())
                pbp = pbp.filter(pl.col("game_id").is_in(list(ids)))
            starting_qbs = self._starting_qbs(pbp) if _present(pbp) else None

            counts["games"] = self.reconciler.reconcile_games(
                season_sched, starting_qbs=starting_qbs
            ).height
            if _present(pbp) and pbp is not None:
                counts["team"] = self.reconciler.reconcile_team_game_logs(pbp).height
                counts["qb"] = self.reconciler.reconcile_qb_game_logs(pbp).height
            else:
                missing.add("pbp")

            stats = self.client.try_fetch_asset("player_stats", season, is_active_season=active)
            snaps = self.client.try_fetch_asset("snap_counts", season, is_active_season=active)
            injuries = self.client.try_fetch_asset("injuries", season, is_active_season=active)
            depth = self.client.try_fetch_asset("depth_charts", season, is_active_season=active)
            weekly = self.client.try_fetch_asset("rosters_weekly", season, is_active_season=active)
            ngs = self.client.fetch_nextgen_stats("passing", season)

            has_snaps[season] = _present(snaps)
            has_inj[season] = _present(injuries)
            has_ngs[season] = _present(ngs)

            if _present(stats) and stats is not None:
                counts["player"] = self.reconciler.reconcile_player_game_logs(
                    stats, snap_counts_df=snaps, players_df=players, pbp_df=pbp
                ).height
            else:
                missing.add("player_stats")
            counts["avail"] = self.reconciler.reconcile_availability(
                injuries_df=injuries, depth_charts_df=depth, weekly_rosters_df=weekly
            ).height

        return FootballIngestionReport(
            seasons=tuple(seasons),
            games_rows=counts["games"],
            team_log_rows=counts["team"],
            qb_log_rows=counts["qb"],
            player_log_rows=counts["player"],
            availability_rows=counts["avail"],
            has_snap_counts=has_snaps,
            has_ngs=has_ngs,
            has_injury_report=has_inj,
            missing_assets=tuple(sorted(missing)),
        )

    @staticmethod
    def _starting_qbs(pbp: pl.DataFrame | None) -> dict[tuple[int, str], int] | None:
        if pbp is None:
            return None
        starters = derive_starting_qbs_from_pbp(pbp)
        out: dict[tuple[int, str], int] = {}
        for row in starters.iter_rows(named=True):
            gsis = str(row["gsis_id"])
            digits = gsis.split("-")[-1]
            if digits.isdigit():
                key = (
                    canonical_game_id_from_nflverse(str(row["nflverse_game_id"])),
                    normalize_football_team(str(row["team"])),
                )
                out[key] = int(digits)
        return out
