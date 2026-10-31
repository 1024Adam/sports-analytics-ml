"""Silver Lakehouse Reconciler for MLB & NPB Baseball (`baseball/ingestion/reconciler.py`).

Upserts and joins MLB StatsAPI, Baseball Savant Statcast, and NPB records into
canonical `zstd`-compressed Silver Parquet tables under `data/silver/baseball/`:
- `games.parquet`
- `team_game_logs.parquet`
- `pitcher_game_logs.parquet`
- `batter_game_logs.parquet`
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sports_ml.sports.baseball.constants import (
    SILVER_BASEBALL_BATTER_LOGS_SCHEMA,
    SILVER_BASEBALL_GAMES_SCHEMA,
    SILVER_BASEBALL_PITCHER_LOGS_SCHEMA,
    SILVER_BASEBALL_TEAM_LOGS_SCHEMA,
    normalize_baseball_team,
)


def _cast_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema=schema)
    exprs = [
        pl.col(col).cast(dtype) if col in df.columns else pl.lit(None, dtype=dtype).alias(col)
        for col, dtype in schema.items()
    ]
    return df.select(exprs)


class BaseballSilverReconciler:
    """Reconciles MLB and NPB feeds into `zstd`-compressed canonical Silver Parquet tables."""

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
    def pitcher_logs_path(self) -> Path:
        return self.silver_dir / "pitcher_game_logs.parquet"

    @property
    def batter_logs_path(self) -> Path:
        return self.silver_dir / "batter_game_logs.parquet"

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
        games_df: pl.DataFrame,
        *,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Normalize club codes, preserve `market_*` columns, and upsert `games.parquet`."""
        market_cols = [
            "market_home_decimal_odds",
            "market_draw_decimal_odds",
            "market_away_decimal_odds",
            "market_prob_home",
            "market_prob_draw",
            "market_prob_away",
        ]
        if games_df.is_empty():
            normalized = pl.DataFrame(schema=SILVER_BASEBALL_GAMES_SCHEMA)
        else:
            normalized = _cast_to_schema(games_df, SILVER_BASEBALL_GAMES_SCHEMA).with_columns(
                pl.col("home_team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
                pl.col("away_team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
            )

        existing = (
            _cast_to_schema(pl.read_parquet(self.games_path), SILVER_BASEBALL_GAMES_SCHEMA)
            if self.games_path.exists()
            else None
        )
        all_rows = (
            pl.concat([existing, normalized], how="vertical")
            if existing is not None and not existing.is_empty()
            else normalized
        )
        odds_source = all_rows.filter(pl.col("market_prob_home").is_not_null())

        merged = self._upsert_frame(
            existing,
            normalized,
            unique_keys=["game_id", "event_seq"],
            sort_keys=["game_date", "start_time_utc", "event_seq", "game_id"],
            schema=SILVER_BASEBALL_GAMES_SCHEMA,
        )

        if not odds_source.is_empty() and not merged.is_empty():
            odds_lookup = (
                odds_source.select(["game_id", "event_seq", *market_cols])
                .unique(subset=["game_id", "event_seq"], keep="last")
                .rename({c: f"{c}_saved" for c in market_cols})
            )
            merged = (
                merged.join(odds_lookup, on=["game_id", "event_seq"], how="left")
                .with_columns(
                    [pl.coalesce([pl.col(c), pl.col(f"{c}_saved")]).alias(c) for c in market_cols]
                )
                .drop([f"{c}_saved" for c in market_cols])
            )
            merged = _cast_to_schema(merged, SILVER_BASEBALL_GAMES_SCHEMA).sort(
                ["game_date", "start_time_utc", "event_seq", "game_id"]
            )

        if persist:
            merged.write_parquet(self.games_path, compression="zstd")
        return merged

    def reconcile_team_logs(
        self,
        team_logs_df: pl.DataFrame,
        *,
        statcast_team_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Join optional Statcast `team_xwoba` / `opp_xwoba` and upsert `team_game_logs.parquet`."""
        if team_logs_df.is_empty():
            working = pl.DataFrame(schema=SILVER_BASEBALL_TEAM_LOGS_SCHEMA)
        else:
            working = _cast_to_schema(team_logs_df, SILVER_BASEBALL_TEAM_LOGS_SCHEMA).with_columns(
                pl.col("team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
                pl.col("opponent").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
            )

        if (
            statcast_team_df is not None
            and not statcast_team_df.is_empty()
            and not working.is_empty()
        ):
            sc = (
                statcast_team_df.with_columns(
                    pl.col("team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8)
                )
                .unique(subset=["game_id", "event_seq", "team"], keep="last")
                .rename({"team_xwoba": "sc_team_xwoba", "opp_xwoba": "sc_opp_xwoba"})
            )
            working = (
                working.join(sc, on=["game_id", "event_seq", "team"], how="left")
                .with_columns(
                    pl.coalesce([pl.col("sc_team_xwoba"), pl.col("team_xwoba")]).alias(
                        "team_xwoba"
                    ),
                    pl.coalesce([pl.col("sc_opp_xwoba"), pl.col("opp_xwoba")]).alias("opp_xwoba"),
                )
                .with_columns(pl.col("team_xwoba").is_not_null().alias("has_statcast_metrics"))
                .drop(["sc_team_xwoba", "sc_opp_xwoba"])
            )

        existing = (
            _cast_to_schema(pl.read_parquet(self.team_logs_path), SILVER_BASEBALL_TEAM_LOGS_SCHEMA)
            if self.team_logs_path.exists()
            else None
        )
        merged = self._upsert_frame(
            existing,
            working,
            unique_keys=["game_id", "event_seq", "team"],
            sort_keys=["game_date", "start_time_utc", "event_seq", "game_id", "team"],
            schema=SILVER_BASEBALL_TEAM_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.team_logs_path, compression="zstd")
        return merged

    def reconcile_pitcher_logs(
        self,
        pitcher_logs_df: pl.DataFrame,
        *,
        statcast_pitcher_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Join optional Statcast pitcher metrics and upsert `pitcher_game_logs.parquet`."""
        if pitcher_logs_df.is_empty():
            working = pl.DataFrame(schema=SILVER_BASEBALL_PITCHER_LOGS_SCHEMA)
        else:
            working = _cast_to_schema(
                pitcher_logs_df, SILVER_BASEBALL_PITCHER_LOGS_SCHEMA
            ).with_columns(
                pl.col("team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
                pl.col("opponent").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
            )

        if (
            statcast_pitcher_df is not None
            and not statcast_pitcher_df.is_empty()
            and not working.is_empty()
        ):
            sc_cols = ["xwoba_against", "barrel_pct_against", "swstr_pct", "avg_fastball_velo"]
            sc = statcast_pitcher_df.unique(
                subset=["game_id", "event_seq", "player_id"], keep="last"
            ).rename({c: f"sc_{c}" for c in sc_cols if c in statcast_pitcher_df.columns})
            working = working.join(sc, on=["game_id", "event_seq", "player_id"], how="left")
            for c in sc_cols:
                sc_c = f"sc_{c}"
                if sc_c in working.columns:
                    working = working.with_columns(
                        pl.coalesce([pl.col(sc_c), pl.col(c)]).alias(c)
                    ).drop(sc_c)

        existing = (
            _cast_to_schema(
                pl.read_parquet(self.pitcher_logs_path), SILVER_BASEBALL_PITCHER_LOGS_SCHEMA
            )
            if self.pitcher_logs_path.exists()
            else None
        )
        merged = self._upsert_frame(
            existing,
            working,
            unique_keys=["game_id", "event_seq", "team", "player_id"],
            sort_keys=[
                "game_date",
                "start_time_utc",
                "event_seq",
                "game_id",
                "team",
                "player_id",
            ],
            schema=SILVER_BASEBALL_PITCHER_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.pitcher_logs_path, compression="zstd")
        return merged

    def reconcile_batter_logs(
        self,
        batter_logs_df: pl.DataFrame,
        *,
        statcast_batter_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Join optional Statcast batter metrics and upsert `batter_game_logs.parquet`."""
        if batter_logs_df.is_empty():
            working = pl.DataFrame(schema=SILVER_BASEBALL_BATTER_LOGS_SCHEMA)
        else:
            working = _cast_to_schema(
                batter_logs_df, SILVER_BASEBALL_BATTER_LOGS_SCHEMA
            ).with_columns(
                pl.col("team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
                pl.col("opponent").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
            )

        if (
            statcast_batter_df is not None
            and not statcast_batter_df.is_empty()
            and not working.is_empty()
        ):
            sc_cols = ["xwoba", "hard_hit_rate", "barrel_rate"]
            sc = statcast_batter_df.unique(
                subset=["game_id", "event_seq", "player_id"], keep="last"
            ).rename({c: f"sc_{c}" for c in sc_cols if c in statcast_batter_df.columns})
            working = working.join(sc, on=["game_id", "event_seq", "player_id"], how="left")
            for c in sc_cols:
                sc_c = f"sc_{c}"
                if sc_c in working.columns:
                    working = working.with_columns(
                        pl.coalesce([pl.col(sc_c), pl.col(c)]).alias(c)
                    ).drop(sc_c)

        existing = (
            _cast_to_schema(
                pl.read_parquet(self.batter_logs_path), SILVER_BASEBALL_BATTER_LOGS_SCHEMA
            )
            if self.batter_logs_path.exists()
            else None
        )
        merged = self._upsert_frame(
            existing,
            working,
            unique_keys=["game_id", "event_seq", "team", "player_id"],
            sort_keys=[
                "game_date",
                "start_time_utc",
                "event_seq",
                "game_id",
                "team",
                "player_id",
            ],
            schema=SILVER_BASEBALL_BATTER_LOGS_SCHEMA,
        )
        if persist:
            merged.write_parquet(self.batter_logs_path, compression="zstd")
        return merged
