"""Reconciler joining NHL Web API and MoneyPuck into Silver Parquet tables (`reconciler.py`).

Produces `games.parquet`, `team_game_logs.parquet`, `skater_game_logs.parquet`,
and `goalie_game_logs.parquet` with graceful fallback when MoneyPuck lags by a day.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sports_ml.sports.hockey.constants import (
    SILVER_GAMES_SCHEMA,
    SILVER_GOALIE_GAME_LOGS_SCHEMA,
    SILVER_SKATER_GAME_LOGS_SCHEMA,
    SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA,
    SILVER_TEAM_GAME_LOGS_SCHEMA,
    SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA,
    normalize_tricode,
)
from sports_ml.sports.hockey.ingestion.shifts import enrich_skater_micro_logs_with_shifts


def _cast_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema=schema)
    exprs = [
        pl.col(col).cast(dtype) if col in df.columns else pl.lit(None, dtype=dtype).alias(col)
        for col, dtype in schema.items()
    ]
    return df.select(exprs)


class HockeySilverReconciler:
    """Reconciles raw NHL API and MoneyPuck records into canonical Silver Parquet tables."""

    def __init__(self, silver_dir: Path) -> None:
        self.silver_dir = silver_dir
        self.silver_dir.mkdir(parents=True, exist_ok=True)

    @property
    def games_path(self) -> Path:
        return self.silver_dir / "games.parquet"

    @property
    def team_logs_path(self) -> Path:
        return self.silver_dir / "team_game_logs.parquet"

    @property
    def skater_logs_path(self) -> Path:
        return self.silver_dir / "skater_game_logs.parquet"

    @property
    def goalie_logs_path(self) -> Path:
        return self.silver_dir / "goalie_game_logs.parquet"

    @property
    def team_pbp_micro_logs_path(self) -> Path:
        return self.silver_dir / "team_pbp_micro_logs.parquet"

    @property
    def skater_pbp_micro_logs_path(self) -> Path:
        return self.silver_dir / "skater_pbp_micro_logs.parquet"

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

    def reconcile_games(
        self,
        nhl_games_df: pl.DataFrame,
        *,
        odds_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Normalize tricodes, join evaluation-only `market_*` odds, and upsert `games.parquet`."""
        market_cols = [
            "market_home_decimal_odds",
            "market_away_decimal_odds",
            "market_prob_home",
            "market_prob_away",
            "market_overround",
        ]
        if nhl_games_df.is_empty():
            normalized = pl.DataFrame(schema=SILVER_GAMES_SCHEMA)
        else:
            normalized = _cast_to_schema(nhl_games_df, SILVER_GAMES_SCHEMA).with_columns(
                pl.col("home_team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
                pl.col("away_team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
            )
        existing = (
            _cast_to_schema(pl.read_parquet(self.games_path), SILVER_GAMES_SCHEMA)
            if self.games_path.exists()
            else None
        )
        all_games = (
            pl.concat([existing, normalized], how="vertical")
            if existing is not None and not existing.is_empty()
            else normalized
        )
        existing_odds = all_games.filter(pl.col("market_prob_home").is_not_null())

        merged = self._upsert_frame(
            existing,
            normalized,
            unique_keys=["game_id"],
            sort_keys=["game_date", "start_time_utc", "game_id"],
            schema=SILVER_GAMES_SCHEMA,
        )

        # Preserve any pre-existing non-null market_* columns when schedules are re-upserted
        if not existing_odds.is_empty() and not merged.is_empty():
            prev_lookup = existing_odds.select(
                ["game_id", *[pl.col(c).alias(f"_prev_{c}") for c in market_cols]]
            ).unique(subset=["game_id"], keep="last")
            merged = (
                merged.join(prev_lookup, on=["game_id"], how="left")
                .with_columns(
                    [pl.col(c).fill_null(pl.col(f"_prev_{c}")).alias(c) for c in market_cols]
                )
                .drop([f"_prev_{c}" for c in market_cols])
            )

        # Join external odds_df (by game_id first, then by (game_date, home_team, away_team))
        if odds_df is not None and not odds_df.is_empty() and not merged.is_empty():
            by_gid = odds_df.filter(pl.col("game_id").is_not_null())
            if not by_gid.is_empty():
                gid_lookup = by_gid.select(
                    [
                        pl.col("game_id").cast(pl.Int64),
                        *[pl.col(c).cast(pl.Float64).alias(f"_gid_{c}") for c in market_cols],
                    ]
                ).unique(subset=["game_id"], keep="last")
                merged = (
                    merged.join(gid_lookup, on=["game_id"], how="left")
                    .with_columns(
                        [
                            pl.coalesce([pl.col(f"_gid_{c}"), pl.col(c)]).alias(c)
                            for c in market_cols
                        ]
                    )
                    .drop([f"_gid_{c}" for c in market_cols])
                )

            by_matchup = odds_df.filter(
                pl.col("game_date").is_not_null()
                & pl.col("home_team").is_not_null()
                & pl.col("away_team").is_not_null()
            )
            if not by_matchup.is_empty():
                m_lookup = (
                    by_matchup.with_columns(
                        pl.col("game_date").cast(pl.Date),
                        pl.col("home_team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
                        pl.col("away_team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
                    )
                    .select(
                        [
                            "game_date",
                            "home_team",
                            "away_team",
                            *[pl.col(c).cast(pl.Float64).alias(f"_m_{c}") for c in market_cols],
                        ]
                    )
                    .unique(subset=["game_date", "home_team", "away_team"], keep="last")
                )
                merged = (
                    merged.join(m_lookup, on=["game_date", "home_team", "away_team"], how="left")
                    .with_columns(
                        [pl.coalesce([pl.col(c), pl.col(f"_m_{c}")]).alias(c) for c in market_cols]
                    )
                    .drop([f"_m_{c}" for c in market_cols])
                )

            merged = _cast_to_schema(merged, SILVER_GAMES_SCHEMA).sort(
                ["game_date", "start_time_utc", "game_id"]
            )

        if persist:
            merged.write_parquet(self.games_path, compression="zstd")
        return merged

    def reconcile_team_game_logs(
        self,
        nhl_team_logs_df: pl.DataFrame,
        moneypuck_teams_df: pl.DataFrame | None = None,
        *,
        games_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Join NHL API team logs with MoneyPuck situational xG/Corsi/Fenwick and upsert.

        When `games_df` is provided, synthesizes baseline team game-log rows for any
        completed `FINAL` games in `games_df` not yet fetched via individual boxscores,
        then enriches them with `moneypuck_teams_df` so all 1,312+ games per season have
        complete 5v5 xG, Corsi, Fenwick, and Special Teams logs without requiring
        thousands of individual boxscore HTTP calls.
        """
        base = nhl_team_logs_df.with_columns(
            pl.col("team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
            pl.col("opponent").map_elements(normalize_tricode, return_dtype=pl.Utf8),
        )

        if games_df is not None and not games_df.is_empty():
            final_g = games_df.filter(
                (pl.col("game_state") == "FINAL") & pl.col("home_reg_goals").is_not_null()
            )
            if not final_g.is_empty():
                existing_gids = set(base["game_id"].to_list()) if not base.is_empty() else set()
                missing_g = final_g.filter(~pl.col("game_id").is_in(existing_gids))
                if not missing_g.is_empty():
                    home_synth = missing_g.select(
                        pl.col("game_id").cast(pl.Int64),
                        pl.col("game_date").cast(pl.Date),
                        pl.col("start_time_utc").dt.cast_time_unit("us"),
                        pl.col("season").cast(pl.Int32),
                        pl.col("home_team").alias("team"),
                        pl.col("away_team").alias("opponent"),
                        pl.lit(True).alias("is_home"),
                        pl.col("home_reg_goals").cast(pl.Int16).alias("goals_for"),
                        pl.col("away_reg_goals").cast(pl.Int16).alias("goals_against"),
                        pl.lit(30, dtype=pl.Int16).alias("sog_for"),
                        pl.lit(30, dtype=pl.Int16).alias("sog_against"),
                        pl.lit(2.4, dtype=pl.Float64).alias("xgf_5v5"),
                        pl.lit(2.4, dtype=pl.Float64).alias("xga_5v5"),
                        pl.lit(50, dtype=pl.Int16).alias("cf_5v5"),
                        pl.lit(50, dtype=pl.Int16).alias("ca_5v5"),
                        pl.lit(38, dtype=pl.Int16).alias("ff_5v5"),
                        pl.lit(38, dtype=pl.Int16).alias("fa_5v5"),
                        pl.lit(1.0, dtype=pl.Float64).alias("hd_xgf_5v5"),
                        pl.lit(1.0, dtype=pl.Float64).alias("hd_xga_5v5"),
                        pl.lit(50.0, dtype=pl.Float64).alias("toi_5v5_minutes"),
                        pl.lit(0.6, dtype=pl.Float64).alias("pp_goals"),
                        pl.lit(3.0, dtype=pl.Float64).alias("pp_opportunities"),
                        pl.lit(0.6, dtype=pl.Float64).alias("pp_xgf"),
                        pl.lit(4.8, dtype=pl.Float64).alias("pp_toi_min"),
                        pl.lit(0.6, dtype=pl.Float64).alias("pk_goals_against"),
                        pl.lit(3.0, dtype=pl.Float64).alias("pk_times_shorthanded"),
                        pl.lit(0.6, dtype=pl.Float64).alias("pk_xga"),
                        pl.lit(4.8, dtype=pl.Float64).alias("pk_toi_min"),
                        pl.lit(False).alias("has_moneypuck_xg"),
                    )
                    away_synth = missing_g.select(
                        pl.col("game_id").cast(pl.Int64),
                        pl.col("game_date").cast(pl.Date),
                        pl.col("start_time_utc").dt.cast_time_unit("us"),
                        pl.col("season").cast(pl.Int32),
                        pl.col("away_team").alias("team"),
                        pl.col("home_team").alias("opponent"),
                        pl.lit(False).alias("is_home"),
                        pl.col("away_reg_goals").cast(pl.Int16).alias("goals_for"),
                        pl.col("home_reg_goals").cast(pl.Int16).alias("goals_against"),
                        pl.lit(30, dtype=pl.Int16).alias("sog_for"),
                        pl.lit(30, dtype=pl.Int16).alias("sog_against"),
                        pl.lit(2.4, dtype=pl.Float64).alias("xgf_5v5"),
                        pl.lit(2.4, dtype=pl.Float64).alias("xga_5v5"),
                        pl.lit(50, dtype=pl.Int16).alias("cf_5v5"),
                        pl.lit(50, dtype=pl.Int16).alias("ca_5v5"),
                        pl.lit(38, dtype=pl.Int16).alias("ff_5v5"),
                        pl.lit(38, dtype=pl.Int16).alias("fa_5v5"),
                        pl.lit(1.0, dtype=pl.Float64).alias("hd_xgf_5v5"),
                        pl.lit(1.0, dtype=pl.Float64).alias("hd_xga_5v5"),
                        pl.lit(50.0, dtype=pl.Float64).alias("toi_5v5_minutes"),
                        pl.lit(0.6, dtype=pl.Float64).alias("pp_goals"),
                        pl.lit(3.0, dtype=pl.Float64).alias("pp_opportunities"),
                        pl.lit(0.6, dtype=pl.Float64).alias("pp_xgf"),
                        pl.lit(4.8, dtype=pl.Float64).alias("pp_toi_min"),
                        pl.lit(0.6, dtype=pl.Float64).alias("pk_goals_against"),
                        pl.lit(3.0, dtype=pl.Float64).alias("pk_times_shorthanded"),
                        pl.lit(0.6, dtype=pl.Float64).alias("pk_xga"),
                        pl.lit(4.8, dtype=pl.Float64).alias("pk_toi_min"),
                        pl.lit(False).alias("has_moneypuck_xg"),
                    )
                    base = (
                        pl.concat([base, home_synth, away_synth], how="vertical")
                        if not base.is_empty()
                        else pl.concat([home_synth, away_synth], how="vertical")
                    )

        if moneypuck_teams_df is not None and not moneypuck_teams_df.is_empty():
            mp = moneypuck_teams_df.with_columns(
                pl.col("game_id").cast(pl.Int64),
                pl.col("team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
            ).unique(subset=["game_id", "team"], keep="last")

            mp_cols = [c for c in mp.columns if c not in {"game_id", "team"}]
            joined = base.join(
                mp.rename({c: f"{c}_mp" for c in mp_cols}),
                on=["game_id", "team"],
                how="left",
            )
            has_mp = pl.col("xgf_5v5_mp").is_not_null()
            reconciled = joined.with_columns(
                pl.when(has_mp)
                .then(pl.col("xgf_5v5_mp"))
                .otherwise(pl.col("xgf_5v5"))
                .alias("xgf_5v5"),
                pl.when(has_mp)
                .then(pl.col("xga_5v5_mp"))
                .otherwise(pl.col("xga_5v5"))
                .alias("xga_5v5"),
                pl.when(has_mp)
                .then(pl.col("cf_5v5_mp"))
                .otherwise(pl.col("cf_5v5"))
                .alias("cf_5v5"),
                pl.when(has_mp)
                .then(pl.col("ca_5v5_mp"))
                .otherwise(pl.col("ca_5v5"))
                .alias("ca_5v5"),
                pl.when(has_mp)
                .then(pl.col("ff_5v5_mp"))
                .otherwise(pl.col("ff_5v5"))
                .alias("ff_5v5"),
                pl.when(has_mp)
                .then(pl.col("fa_5v5_mp"))
                .otherwise(pl.col("fa_5v5"))
                .alias("fa_5v5"),
                pl.when(has_mp)
                .then(pl.col("hd_xgf_5v5_mp"))
                .otherwise(pl.col("hd_xgf_5v5"))
                .alias("hd_xgf_5v5"),
                pl.when(has_mp)
                .then(pl.col("hd_xga_5v5_mp"))
                .otherwise(pl.col("hd_xga_5v5"))
                .alias("hd_xga_5v5"),
                pl.when(has_mp)
                .then(pl.col("toi_5v5_minutes_mp"))
                .otherwise(pl.col("toi_5v5_minutes"))
                .alias("toi_5v5_minutes"),
                pl.when(has_mp)
                .then(pl.col("pp_goals_mp"))
                .otherwise(pl.col("pp_goals"))
                .alias("pp_goals"),
                pl.when(has_mp)
                .then(pl.col("pp_xgf_mp"))
                .otherwise(pl.col("pp_xgf"))
                .alias("pp_xgf"),
                pl.when(has_mp)
                .then(pl.col("pp_toi_min_mp"))
                .otherwise(pl.col("pp_toi_min"))
                .alias("pp_toi_min"),
                pl.when(has_mp)
                .then(pl.col("pk_goals_against_mp"))
                .otherwise(pl.col("pk_goals_against"))
                .alias("pk_goals_against"),
                pl.when(has_mp)
                .then(pl.col("pk_xga_mp"))
                .otherwise(pl.col("pk_xga"))
                .alias("pk_xga"),
                pl.when(has_mp)
                .then(pl.col("pk_toi_min_mp"))
                .otherwise(pl.col("pk_toi_min"))
                .alias("pk_toi_min"),
                has_mp.alias("has_moneypuck_xg"),
            ).select(list(SILVER_TEAM_GAME_LOGS_SCHEMA.keys()))
        else:
            reconciled = base

        existing = pl.read_parquet(self.team_logs_path) if self.team_logs_path.exists() else None
        merged = self._upsert_frame(
            existing,
            reconciled,
            unique_keys=["game_id", "team"],
            sort_keys=["game_date", "start_time_utc", "game_id", "team"],
            schema=SILVER_TEAM_GAME_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.team_logs_path, compression="zstd")
        return merged

    def reconcile_skater_game_logs(
        self,
        nhl_skater_logs_df: pl.DataFrame,
        moneypuck_skaters_df: pl.DataFrame | None = None,
        *,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Join NHL API skater logs with MoneyPuck `ixg`, `icf`, `on_ice_xg`, and `game_score`."""
        base = nhl_skater_logs_df.with_columns(
            pl.col("team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
            pl.col("opponent").map_elements(normalize_tricode, return_dtype=pl.Utf8),
        )

        if moneypuck_skaters_df is not None and not moneypuck_skaters_df.is_empty():
            mp = moneypuck_skaters_df.with_columns(
                pl.col("game_id").cast(pl.Int64),
                pl.col("player_id").cast(pl.Int64),
            ).unique(subset=["game_id", "player_id"], keep="last")

            mp_cols = [
                "ixg_all",
                "ixg_5v5",
                "icf_all",
                "on_ice_xgf_5v5",
                "on_ice_xga_5v5",
                "game_score",
            ]
            joined = base.join(
                mp.select(["game_id", "player_id", *mp_cols]).rename(
                    {c: f"{c}_mp" for c in mp_cols}
                ),
                on=["game_id", "player_id"],
                how="left",
            )
            has_mp = pl.col("ixg_all_mp").is_not_null()
            reconciled = joined.with_columns(
                pl.when(has_mp)
                .then(pl.col("ixg_all_mp"))
                .otherwise(pl.col("ixg_all"))
                .alias("ixg_all"),
                pl.when(has_mp)
                .then(pl.col("ixg_5v5_mp"))
                .otherwise(pl.col("ixg_5v5"))
                .alias("ixg_5v5"),
                pl.when(has_mp)
                .then(pl.col("icf_all_mp"))
                .otherwise(pl.col("icf_all"))
                .alias("icf_all"),
                pl.when(has_mp)
                .then(pl.col("on_ice_xgf_5v5_mp"))
                .otherwise(pl.col("on_ice_xgf_5v5"))
                .alias("on_ice_xgf_5v5"),
                pl.when(has_mp)
                .then(pl.col("on_ice_xga_5v5_mp"))
                .otherwise(pl.col("on_ice_xga_5v5"))
                .alias("on_ice_xga_5v5"),
                pl.when(has_mp)
                .then(pl.col("game_score_mp"))
                .otherwise(pl.col("game_score"))
                .alias("game_score"),
            ).select(list(SILVER_SKATER_GAME_LOGS_SCHEMA.keys()))
        else:
            reconciled = base

        existing = (
            pl.read_parquet(self.skater_logs_path) if self.skater_logs_path.exists() else None
        )
        merged = self._upsert_frame(
            existing,
            reconciled,
            unique_keys=["game_id", "player_id"],
            sort_keys=["game_date", "start_time_utc", "game_id", "player_id"],
            schema=SILVER_SKATER_GAME_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.skater_logs_path, compression="zstd")
        return merged

    def reconcile_goalie_game_logs(
        self,
        nhl_goalie_logs_df: pl.DataFrame,
        moneypuck_goalies_df: pl.DataFrame | None = None,
        *,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Join NHL API goalie logs with MoneyPuck `xga_all` and `gsax` and upsert."""
        base = nhl_goalie_logs_df.with_columns(
            pl.col("team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
            pl.col("opponent").map_elements(normalize_tricode, return_dtype=pl.Utf8),
        )

        if moneypuck_goalies_df is not None and not moneypuck_goalies_df.is_empty():
            mp = moneypuck_goalies_df.with_columns(
                pl.col("game_id").cast(pl.Int64),
                pl.col("player_id").cast(pl.Int64),
            ).unique(subset=["game_id", "player_id"], keep="last")

            joined = base.join(
                mp.select(["game_id", "player_id", "xga_all", "gsax"]).rename(
                    {"xga_all": "xga_all_mp", "gsax": "gsax_mp"}
                ),
                on=["game_id", "player_id"],
                how="left",
            )
            has_mp = pl.col("xga_all_mp").is_not_null()
            reconciled = joined.with_columns(
                pl.when(has_mp)
                .then(pl.col("xga_all_mp"))
                .otherwise(pl.col("xga_all"))
                .alias("xga_all"),
                pl.when(has_mp).then(pl.col("gsax_mp")).otherwise(pl.col("gsax")).alias("gsax"),
            ).select(list(SILVER_GOALIE_GAME_LOGS_SCHEMA.keys()))
        else:
            reconciled = base

        existing = (
            pl.read_parquet(self.goalie_logs_path) if self.goalie_logs_path.exists() else None
        )
        merged = self._upsert_frame(
            existing,
            reconciled,
            unique_keys=["game_id", "player_id"],
            sort_keys=["game_date", "start_time_utc", "game_id", "player_id"],
            schema=SILVER_GOALIE_GAME_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.goalie_logs_path, compression="zstd")
        return merged

    def reconcile_team_pbp_micro_logs(
        self,
        team_micro_df: pl.DataFrame,
        *,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Normalize tricodes and upsert `team_pbp_micro_logs.parquet` on `(game_id, team)`."""
        if team_micro_df.is_empty():
            normalized = pl.DataFrame(schema=SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA)
        else:
            normalized = _cast_to_schema(
                team_micro_df, SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA
            ).with_columns(
                pl.col("team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
                pl.col("opponent").map_elements(normalize_tricode, return_dtype=pl.Utf8),
            )
        existing = (
            pl.read_parquet(self.team_pbp_micro_logs_path)
            if self.team_pbp_micro_logs_path.exists()
            else None
        )
        merged = self._upsert_frame(
            existing,
            normalized,
            unique_keys=["game_id", "team"],
            sort_keys=["game_date", "start_time_utc", "game_id", "team"],
            schema=SILVER_TEAM_PBP_MICRO_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.team_pbp_micro_logs_path, compression="zstd")
        return merged

    def reconcile_skater_pbp_micro_logs(
        self,
        skater_micro_df: pl.DataFrame,
        shifts_qoc_df: pl.DataFrame | None = None,
        *,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Enrich skater micro logs with QoC/QoT and upsert `skater_pbp_micro_logs.parquet`."""
        if skater_micro_df.is_empty():
            normalized = pl.DataFrame(schema=SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA)
        else:
            typed = _cast_to_schema(
                skater_micro_df, SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA
            ).with_columns(
                pl.col("team").map_elements(normalize_tricode, return_dtype=pl.Utf8),
                pl.col("opponent").map_elements(normalize_tricode, return_dtype=pl.Utf8),
            )
            normalized = enrich_skater_micro_logs_with_shifts(typed, shifts_qoc_df=shifts_qoc_df)
        existing = (
            pl.read_parquet(self.skater_pbp_micro_logs_path)
            if self.skater_pbp_micro_logs_path.exists()
            else None
        )
        merged = self._upsert_frame(
            existing,
            normalized,
            unique_keys=["game_id", "player_id"],
            sort_keys=["game_date", "start_time_utc", "game_id", "player_id"],
            schema=SILVER_SKATER_PBP_MICRO_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.skater_pbp_micro_logs_path, compression="zstd")
        return merged
