"""Reconciler joining FPL, Understat, and Football-Data into Silver Parquet (`reconciler.py`).

Produces `matches.parquet`, `team_match_logs.parquet`, `outfield_match_logs.parquet`,
and `gk_match_logs.parquet` under `data/silver/soccer/` with graceful fallback
(`has_understat_xg = False`) when Understat lags post-match.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sports_ml.sports.soccer.constants import (
    SILVER_GK_MATCH_LOGS_SCHEMA,
    SILVER_MATCHES_SCHEMA,
    SILVER_OUTFIELD_MATCH_LOGS_SCHEMA,
    SILVER_TEAM_MATCH_LOGS_SCHEMA,
    normalize_soccer_club,
)
from sports_ml.sports.soccer.ingestion.lineups import normalize_soccer_player_name


def _cast_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema=schema)
    exprs = [pl.col(col).cast(dtype) for col, dtype in schema.items()]
    return df.select(exprs)


class SoccerSilverReconciler:
    """Reconciles FPL API, Understat, and Football-Data records into canonical Silver Parquet."""

    def __init__(self, silver_dir: Path) -> None:
        self.silver_dir = silver_dir
        self.silver_dir.mkdir(parents=True, exist_ok=True)

    @property
    def matches_path(self) -> Path:
        return self.silver_dir / "matches.parquet"

    @property
    def team_logs_path(self) -> Path:
        return self.silver_dir / "team_match_logs.parquet"

    @property
    def outfield_logs_path(self) -> Path:
        return self.silver_dir / "outfield_match_logs.parquet"

    @property
    def gk_logs_path(self) -> Path:
        return self.silver_dir / "gk_match_logs.parquet"

    @staticmethod
    def _upsert_frame(
        existing: pl.DataFrame | None,
        incoming: pl.DataFrame,
        *,
        unique_keys: list[str],
        sort_keys: list[str],
        schema: dict[str, pl.DataType],
    ) -> pl.DataFrame:
        incoming_typed = _cast_to_schema(incoming, schema)
        if existing is None or existing.is_empty():
            combined = incoming_typed
        elif incoming_typed.is_empty():
            combined = _cast_to_schema(existing, schema)
        else:
            combined = pl.concat(
                [_cast_to_schema(existing, schema), incoming_typed],
                how="vertical",
            )
        return combined.unique(subset=unique_keys, keep="last").sort(sort_keys)

    def reconcile_matches(
        self,
        matches_df: pl.DataFrame,
        *,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Normalize club tricodes and upsert `matches.parquet` on `(match_id)`."""
        if matches_df.is_empty():
            normalized = pl.DataFrame(schema=SILVER_MATCHES_SCHEMA)
        else:
            normalized = matches_df.with_columns(
                pl.col("home_team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
                pl.col("away_team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
            )
        existing = pl.read_parquet(self.matches_path) if self.matches_path.exists() else None
        merged = self._upsert_frame(
            existing,
            normalized,
            unique_keys=["match_id"],
            sort_keys=["match_date", "start_time_utc", "event_seq", "match_id"],
            schema=SILVER_MATCHES_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.matches_path, compression="zstd")
        return merged

    def reconcile_team_match_logs(
        self,
        matches_df: pl.DataFrame,
        *,
        football_data_box_df: pl.DataFrame | None = None,
        understat_teams_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Build and upsert `team_match_logs.parquet` on `(match_id, team)`.

        Synthesizes 2 rows per completed `FINAL` match (Home & Away), enriches boxscore
        counts (`shots`, `sot`, `corners`, `fouls`, `cards`) from `football_data_box_df`,
        and joins `npxG`, `xG`, `PPDA`, and `deep_completions` from `understat_teams_df`.
        When Understat lags post-match, falls back gracefully with `has_understat_xg = False`.
        """
        final_matches = matches_df.filter(
            (pl.col("game_state") == "FINAL")
            & pl.col("home_goals").is_not_null()
            & pl.col("away_goals").is_not_null()
        )
        if final_matches.is_empty():
            existing = (
                pl.read_parquet(self.team_logs_path) if self.team_logs_path.exists() else None
            )
            return (
                _cast_to_schema(existing, SILVER_TEAM_MATCH_LOGS_SCHEMA)
                if existing is not None
                else pl.DataFrame(schema=SILVER_TEAM_MATCH_LOGS_SCHEMA)
            )

        home_base = final_matches.select(
            pl.col("match_id").cast(pl.Int64),
            pl.col("match_date").cast(pl.Date),
            pl.col("start_time_utc").dt.cast_time_unit("us"),
            pl.col("season").cast(pl.Int32),
            pl.col("home_team")
            .map_elements(normalize_soccer_club, return_dtype=pl.Utf8)
            .alias("team"),
            pl.col("away_team")
            .map_elements(normalize_soccer_club, return_dtype=pl.Utf8)
            .alias("opponent"),
            pl.lit(True).alias("is_home"),
            pl.col("home_goals").cast(pl.Int16).alias("goals_for"),
            pl.col("away_goals").cast(pl.Int16).alias("goals_against"),
        )
        away_base = final_matches.select(
            pl.col("match_id").cast(pl.Int64),
            pl.col("match_date").cast(pl.Date),
            pl.col("start_time_utc").dt.cast_time_unit("us"),
            pl.col("season").cast(pl.Int32),
            pl.col("away_team")
            .map_elements(normalize_soccer_club, return_dtype=pl.Utf8)
            .alias("team"),
            pl.col("home_team")
            .map_elements(normalize_soccer_club, return_dtype=pl.Utf8)
            .alias("opponent"),
            pl.lit(False).alias("is_home"),
            pl.col("away_goals").cast(pl.Int16).alias("goals_for"),
            pl.col("home_goals").cast(pl.Int16).alias("goals_against"),
        )
        base = pl.concat([home_base, away_base], how="vertical")

        # 1. Join Football-Data boxscore counts if provided
        if football_data_box_df is not None and not football_data_box_df.is_empty():
            fd_norm = football_data_box_df.with_columns(
                pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
                pl.col("match_date").cast(pl.Date),
                pl.col("season").cast(pl.Int32),
            ).select(
                [
                    "season",
                    "match_date",
                    "team",
                    pl.col("shots_for").alias("_fd_shots_for"),
                    pl.col("shots_against").alias("_fd_shots_against"),
                    pl.col("sot_for").alias("_fd_sot_for"),
                    pl.col("sot_against").alias("_fd_sot_against"),
                    pl.col("corners_for").alias("_fd_corners_for"),
                    pl.col("corners_against").alias("_fd_corners_against"),
                    pl.col("fouls_for").alias("_fd_fouls_for"),
                    pl.col("fouls_against").alias("_fd_fouls_against"),
                    pl.col("yellow_cards").alias("_fd_yellow_cards"),
                    pl.col("red_cards").alias("_fd_red_cards"),
                ]
            )
            base = base.join(fd_norm, on=["season", "match_date", "team"], how="left")
        else:
            base = base.with_columns(
                pl.lit(None, dtype=pl.Int16).alias("_fd_shots_for"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_shots_against"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_sot_for"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_sot_against"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_corners_for"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_corners_against"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_fouls_for"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_fouls_against"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_yellow_cards"),
                pl.lit(None, dtype=pl.Int16).alias("_fd_red_cards"),
            )

        base = base.with_columns(
            pl.coalesce(
                [
                    pl.col("_fd_shots_for"),
                    (pl.col("goals_for") * 4 + 8).cast(pl.Int16),
                ]
            ).alias("shots_for"),
            pl.coalesce(
                [
                    pl.col("_fd_shots_against"),
                    (pl.col("goals_against") * 4 + 8).cast(pl.Int16),
                ]
            ).alias("shots_against"),
            pl.coalesce(
                [
                    pl.col("_fd_sot_for"),
                    (pl.col("goals_for") * 2 + 2).cast(pl.Int16),
                ]
            ).alias("sot_for"),
            pl.coalesce(
                [
                    pl.col("_fd_sot_against"),
                    (pl.col("goals_against") * 2 + 2).cast(pl.Int16),
                ]
            ).alias("sot_against"),
            pl.coalesce([pl.col("_fd_corners_for"), pl.lit(5, dtype=pl.Int16)]).alias(
                "corners_for"
            ),
            pl.coalesce([pl.col("_fd_corners_against"), pl.lit(5, dtype=pl.Int16)]).alias(
                "corners_against"
            ),
            pl.coalesce([pl.col("_fd_fouls_for"), pl.lit(11, dtype=pl.Int16)]).alias("fouls_for"),
            pl.coalesce([pl.col("_fd_fouls_against"), pl.lit(11, dtype=pl.Int16)]).alias(
                "fouls_against"
            ),
            pl.coalesce([pl.col("_fd_yellow_cards"), pl.lit(2, dtype=pl.Int16)]).alias(
                "yellow_cards"
            ),
            pl.coalesce([pl.col("_fd_red_cards"), pl.lit(0, dtype=pl.Int16)]).alias("red_cards"),
        )

        # 2. Join Understat shot-quality & pressing metrics if provided
        if understat_teams_df is not None and not understat_teams_df.is_empty():
            us_norm = understat_teams_df.with_columns(
                pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
                pl.col("match_date").cast(pl.Date),
                pl.col("season").cast(pl.Int32),
            ).select(
                [
                    "season",
                    "match_date",
                    "team",
                    pl.col("xgf").alias("_us_xgf"),
                    pl.col("xga").alias("_us_xga"),
                    pl.col("npxgf").alias("_us_npxgf"),
                    pl.col("npxga").alias("_us_npxga"),
                    pl.col("ppda").alias("_us_ppda"),
                    pl.col("opp_ppda").alias("_us_opp_ppda"),
                    pl.col("deep_completions").alias("_us_deep"),
                    pl.col("opp_deep_completions").alias("_us_opp_deep"),
                ]
            )
            base = base.join(us_norm, on=["season", "match_date", "team"], how="left")
        else:
            base = base.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("_us_xgf"),
                pl.lit(None, dtype=pl.Float64).alias("_us_xga"),
                pl.lit(None, dtype=pl.Float64).alias("_us_npxgf"),
                pl.lit(None, dtype=pl.Float64).alias("_us_npxga"),
                pl.lit(None, dtype=pl.Float64).alias("_us_ppda"),
                pl.lit(None, dtype=pl.Float64).alias("_us_opp_ppda"),
                pl.lit(None, dtype=pl.Int16).alias("_us_deep"),
                pl.lit(None, dtype=pl.Int16).alias("_us_opp_deep"),
            )

        # Fallback shot-based xG estimate when Understat has not updated yet
        fallback_xgf = (
            pl.col("sot_for").cast(pl.Float64) * 0.24
            + (pl.col("shots_for") - pl.col("sot_for")).clip(lower_bound=0).cast(pl.Float64) * 0.05
        )
        fallback_xga = (
            pl.col("sot_against").cast(pl.Float64) * 0.24
            + (pl.col("shots_against") - pl.col("sot_against")).clip(lower_bound=0).cast(pl.Float64)
            * 0.05
        )

        enriched = base.with_columns(
            pl.col("_us_xgf").is_not_null().alias("has_understat_xg"),
            pl.coalesce([pl.col("_us_xgf"), fallback_xgf]).alias("xgf"),
            pl.coalesce([pl.col("_us_xga"), fallback_xga]).alias("xga"),
            pl.coalesce([pl.col("_us_npxgf"), pl.col("_us_xgf"), fallback_xgf]).alias("npxgf"),
            pl.coalesce([pl.col("_us_npxga"), pl.col("_us_xga"), fallback_xga]).alias("npxga"),
            pl.coalesce([pl.col("_us_ppda"), pl.lit(11.5, dtype=pl.Float64)]).alias("ppda"),
            pl.coalesce([pl.col("_us_opp_ppda"), pl.lit(11.5, dtype=pl.Float64)]).alias("opp_ppda"),
            pl.coalesce([pl.col("_us_deep"), pl.lit(6, dtype=pl.Int16)]).alias("deep_completions"),
            pl.coalesce([pl.col("_us_opp_deep"), pl.lit(6, dtype=pl.Int16)]).alias(
                "opp_deep_completions"
            ),
        )

        existing = pl.read_parquet(self.team_logs_path) if self.team_logs_path.exists() else None
        merged = self._upsert_frame(
            existing,
            enriched,
            unique_keys=["match_id", "team"],
            sort_keys=["match_date", "start_time_utc", "match_id", "team"],
            schema=SILVER_TEAM_MATCH_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.team_logs_path, compression="zstd")
        return merged

    def reconcile_outfield_match_logs(
        self,
        fpl_outfield_df: pl.DataFrame,
        understat_players_df: pl.DataFrame | None = None,
        *,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Join FPL outfield player logs with Understat shot/possession metrics and upsert."""
        if fpl_outfield_df.is_empty():
            existing = (
                pl.read_parquet(self.outfield_logs_path)
                if self.outfield_logs_path.exists()
                else None
            )
            return (
                _cast_to_schema(existing, SILVER_OUTFIELD_MATCH_LOGS_SCHEMA)
                if existing is not None
                else pl.DataFrame(schema=SILVER_OUTFIELD_MATCH_LOGS_SCHEMA)
            )

        working = fpl_outfield_df.with_columns(
            pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
            pl.col("opponent").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
            pl.col("player_name")
            .map_elements(normalize_soccer_player_name, return_dtype=pl.Utf8)
            .alias("_norm_name"),
        )

        if understat_players_df is not None and not understat_players_df.is_empty():
            us_norm = understat_players_df.with_columns(
                pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
                pl.col("match_date").cast(pl.Date),
                pl.col("season").cast(pl.Int32),
                pl.col("player_name")
                .map_elements(normalize_soccer_player_name, return_dtype=pl.Utf8)
                .alias("_norm_name"),
            ).select(
                [
                    "season",
                    "match_date",
                    "team",
                    "_norm_name",
                    pl.col("shots").alias("_us_shots"),
                    pl.col("sot").alias("_us_sot"),
                    pl.col("key_passes").alias("_us_key_passes"),
                    pl.col("xg").alias("_us_xg"),
                    pl.col("npxg").alias("_us_npxg"),
                    pl.col("xa").alias("_us_xa"),
                    pl.col("xg_chain").alias("_us_xg_chain"),
                    pl.col("xg_buildup").alias("_us_xg_buildup"),
                ]
            )
            working = working.join(
                us_norm,
                on=["season", "match_date", "team", "_norm_name"],
                how="left",
            ).with_columns(
                pl.coalesce([pl.col("_us_shots"), pl.col("shots")]).alias("shots"),
                pl.coalesce([pl.col("_us_sot"), pl.col("sot")]).alias("sot"),
                pl.coalesce([pl.col("_us_key_passes"), pl.col("key_passes")]).alias("key_passes"),
                pl.coalesce([pl.col("_us_xg"), pl.col("xg")]).alias("xg"),
                pl.coalesce([pl.col("_us_npxg"), pl.col("npxg")]).alias("npxg"),
                pl.coalesce([pl.col("_us_xa"), pl.col("xa")]).alias("xa"),
                pl.coalesce([pl.col("_us_xg_chain"), pl.col("xg_chain")]).alias("xg_chain"),
                pl.coalesce([pl.col("_us_xg_buildup"), pl.col("xg_buildup")]).alias("xg_buildup"),
            )

        existing = (
            pl.read_parquet(self.outfield_logs_path) if self.outfield_logs_path.exists() else None
        )
        merged = self._upsert_frame(
            existing,
            working,
            unique_keys=["match_id", "player_id"],
            sort_keys=["match_date", "start_time_utc", "match_id", "player_id"],
            schema=SILVER_OUTFIELD_MATCH_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.outfield_logs_path, compression="zstd")
        return merged

    def reconcile_gk_match_logs(
        self,
        fpl_gk_df: pl.DataFrame,
        *,
        team_logs_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Enrich goalkeeper logs with opponent Understat `xGA` faced (`gsaa`) and upsert."""
        if fpl_gk_df.is_empty():
            existing = pl.read_parquet(self.gk_logs_path) if self.gk_logs_path.exists() else None
            return (
                _cast_to_schema(existing, SILVER_GK_MATCH_LOGS_SCHEMA)
                if existing is not None
                else pl.DataFrame(schema=SILVER_GK_MATCH_LOGS_SCHEMA)
            )

        working = fpl_gk_df.with_columns(
            pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
            pl.col("opponent").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
        )

        if team_logs_df is not None and not team_logs_df.is_empty():
            team_xga = team_logs_df.select(
                [
                    "match_id",
                    "team",
                    pl.col("xga").alias("_team_xga"),
                    pl.col("sot_against").alias("_team_sot_against"),
                ]
            )
            working = working.join(team_xga, on=["match_id", "team"], how="left").with_columns(
                pl.when(pl.col("_team_xga").is_not_null())
                .then(pl.col("_team_xga") * (pl.col("minutes") / 90.0).clip(0.0, 1.0))
                .otherwise(pl.col("xga_faced"))
                .alias("xga_faced"),
                pl.coalesce([pl.col("_team_sot_against"), pl.col("shots_on_target_against")]).alias(
                    "shots_on_target_against"
                ),
            )
            working = working.with_columns(
                (pl.col("xga_faced") - pl.col("goals_against").cast(pl.Float64)).alias("gsaa")
            )

        existing = pl.read_parquet(self.gk_logs_path) if self.gk_logs_path.exists() else None
        merged = self._upsert_frame(
            existing,
            working,
            unique_keys=["match_id", "player_id"],
            sort_keys=["match_date", "start_time_utc", "match_id", "player_id"],
            schema=SILVER_GK_MATCH_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.gk_logs_path, compression="zstd")
        return merged
