"""Unified two-stage NHL matchup predictor service (`models/predictor.py`).

Exposes `HockeyMatchupPredictor.predict_matchup(..., include_players=...)` and
`predict_date(...)`, surfaces pre-prediction confidence warnings (starting goalie
confirmation status, data freshness, MoneyPuck fallback, and Bayesian shrinkage
flags), appends to `data/predictions/hockey_predictions.parquet`, and scores the
live prediction ledger against completed Silver games.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sports_ml.config import Settings, get_settings
from sports_ml.core.evaluation import BenchmarkComparisonReport, compare_against_baselines
from sports_ml.core.registry import ModelMetadata, ModelRegistry
from sports_ml.sports.hockey.constants import normalize_tricode
from sports_ml.sports.hockey.features.goalie import (
    GOALIE_SHRINKAGE_K,
    build_goalie_rolling_features,
    resolve_team_starting_goalie_features,
)
from sports_ml.sports.hockey.features.player import build_stage2_player_features
from sports_ml.sports.hockey.features.team import (
    TEAM_SHRINKAGE_K,
    build_stage1_game_features,
    build_team_rolling_features,
)
from sports_ml.sports.hockey.ingestion.lineups import (
    fetch_dailyfaceoff_starting_goalies,
)
from sports_ml.sports.hockey.ingestion.nhl_api import (
    NHLWebAPIClient,
    derive_active_nhl_season,
    parse_roster_payload,
)
from sports_ml.sports.hockey.models.stage1_game import Stage1GameOutcomeModel
from sports_ml.sports.hockey.models.stage2_player import Stage2PlayerProjectionModel


class PlayerMatchupProjections(BaseModel):
    """Container for Stage 2 skater and goalie projection DataFrames."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    home_skaters: pl.DataFrame
    away_skaters: pl.DataFrame
    goalies: pl.DataFrame


class MatchupPrediction(BaseModel):
    """Structured Stage 1 (mandatory) and Stage 2 (optional) matchup prediction output."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    game_id: int | None = None
    game_date: date
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    home_goalie_status: str = "PROJECTED_HEURISTIC"
    away_goalie_status: str = "PROJECTED_HEURISTIC"
    latest_silver_game_date: str | None = None
    has_moneypuck_xg: bool = True
    warnings: list[str] = Field(default_factory=list)
    player_projections: PlayerMatchupProjections | None = None


def build_and_persist_gold_features(
    settings: Settings,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build and write Gold Stage 1 and Stage 2 feature matrices from Silver Parquet tables."""
    silver_dir = settings.paths.silver_dir / "hockey"
    gold_dir = settings.paths.gold_dir / "hockey"
    gold_dir.mkdir(parents=True, exist_ok=True)

    games_path = silver_dir / "games.parquet"
    teams_path = silver_dir / "team_game_logs.parquet"
    skaters_path = silver_dir / "skater_game_logs.parquet"
    goalies_path = silver_dir / "goalie_game_logs.parquet"

    if not games_path.exists() or not teams_path.exists():
        raise FileNotFoundError(
            f"Silver Parquet tables not found in '{silver_dir}'. Run 'sports-ml ingest' first."
        )

    games_df = pl.read_parquet(games_path)
    team_logs_df = pl.read_parquet(teams_path)
    skater_logs_df = pl.read_parquet(skaters_path) if skaters_path.exists() else pl.DataFrame()
    goalie_logs_df = pl.read_parquet(goalies_path) if goalies_path.exists() else pl.DataFrame()

    team_feat_df = build_team_rolling_features(team_logs_df)
    goalie_feat_df = (
        build_goalie_rolling_features(goalie_logs_df)
        if not goalie_logs_df.is_empty()
        else pl.DataFrame()
    )
    stage1_df = build_stage1_game_features(
        games_df,
        team_logs_df,
        skater_logs_df if not skater_logs_df.is_empty() else None,
        goalie_logs_df if not goalie_logs_df.is_empty() else None,
    )
    stage2_df = (
        build_stage2_player_features(skater_logs_df, team_feat_df, goalie_feat_df)
        if not skater_logs_df.is_empty()
        else pl.DataFrame()
    )

    stage1_df.write_parquet(gold_dir / "stage1_game_features.parquet")
    if not stage2_df.is_empty():
        stage2_df.write_parquet(gold_dir / "stage2_player_features.parquet")

    return stage1_df, stage2_df, team_feat_df, goalie_feat_df


class HockeyMatchupPredictor:
    """Unified two-stage inference service for NHL matchups."""

    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        stage1_model: Stage1GameOutcomeModel,
        stage2_model: Stage2PlayerProjectionModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.metadata = metadata
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.settings = settings or get_settings()

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        *,
        settings: Settings | None = None,
    ) -> HockeyMatchupPredictor:
        """Load a fingerprinted model bundle from `artifact_dir`."""
        bundle = ModelRegistry.load_bundle(artifact_dir)
        resolved_settings = settings or get_settings(root_dir=bundle.bundle_dir.parents[2])
        return cls(
            metadata=bundle.metadata,
            stage1_model=bundle.stage1_model,
            stage2_model=bundle.stage2_model,
            settings=resolved_settings,
        )

    def _synthesize_matchup_row_if_missing(
        self,
        stage1_df: pl.DataFrame,
        home_code: str,
        away_code: str,
        game_date: date,
    ) -> pl.DataFrame:
        """Synthesize a pre-game matchup feature row from each team's latest state if absent."""
        exact = stage1_df.filter(
            (pl.col("game_date") == game_date)
            & (pl.col("home_team") == home_code)
            & (pl.col("away_team") == away_code)
        )
        if not exact.is_empty():
            return exact.tail(1)

        from sports_ml.core.ratings import compute_elo_win_probability

        def _latest_team_val(t_code: str, col_base: str, default_val: float) -> float:
            t_rows = stage1_df.filter(
                (pl.col("home_team") == t_code) | (pl.col("away_team") == t_code)
            )
            if t_rows.is_empty():
                return default_val
            last_r = t_rows.row(-1, named=True)
            side = "home" if last_r["home_team"] == t_code else "away"
            if col_base == "elo":
                val = last_r.get(f"elo_{side}_post") or last_r.get(f"elo_{side}_pre")
                return float(val) if val is not None else default_val
            val = last_r.get(f"{col_base}_{side}")
            return float(val) if val is not None else default_val

        elo_h = _latest_team_val(home_code, "elo", 1500.0)
        elo_a = _latest_team_val(away_code, "elo", 1500.0)
        elo_diff = elo_h + 35.0 - elo_a
        elo_prob = compute_elo_win_probability(elo_diff)

        xgf_h = _latest_team_val(home_code, "xgf_pct_5v5_w10", 0.50)
        xgf_a = _latest_team_val(away_code, "xgf_pct_5v5_w10", 0.50)
        cf_h = _latest_team_val(home_code, "cf_pct_5v5_w10", 0.50)
        cf_a = _latest_team_val(away_code, "cf_pct_5v5_w10", 0.50)
        gsax_h = _latest_team_val(home_code, "gsax_per_60_shrunk", 0.0)
        gsax_a = _latest_team_val(away_code, "gsax_per_60_shrunk", 0.0)
        fwd_h = _latest_team_val(home_code, "forward_weighted_ixg_per_60", 0.75)
        fwd_a = _latest_team_val(away_code, "forward_weighted_ixg_per_60", 0.75)
        def_h = _latest_team_val(home_code, "defense_weighted_xgd_per_60", 0.0)
        def_a = _latest_team_val(away_code, "defense_weighted_xgd_per_60", 0.0)

        latest_row = stage1_df.tail(1)
        synth_id = int(f"{game_date.year}029999")
        return latest_row.with_columns(
            pl.lit(synth_id, dtype=pl.Int64).alias("game_id"),
            pl.lit(synth_id, dtype=pl.Int64).alias("event_id"),
            pl.lit(game_date, dtype=pl.Date).alias("game_date"),
            pl.lit(home_code, dtype=pl.Utf8).alias("home_team"),
            pl.lit(away_code, dtype=pl.Utf8).alias("away_team"),
            pl.lit("FUT", dtype=pl.Utf8).alias("game_state"),
            pl.lit(elo_h, dtype=pl.Float64).alias("elo_home_pre"),
            pl.lit(elo_a, dtype=pl.Float64).alias("elo_away_pre"),
            pl.lit(elo_diff, dtype=pl.Float64).alias("elo_diff_pre"),
            pl.lit(elo_prob, dtype=pl.Float64).alias("elo_home_win_prob"),
            pl.lit(xgf_h, dtype=pl.Float64).alias("xgf_pct_5v5_w10_home"),
            pl.lit(xgf_a, dtype=pl.Float64).alias("xgf_pct_5v5_w10_away"),
            pl.lit(xgf_h - xgf_a, dtype=pl.Float64).alias("xgf_pct_5v5_w10_diff"),
            pl.lit(cf_h, dtype=pl.Float64).alias("cf_pct_5v5_w10_home"),
            pl.lit(cf_a, dtype=pl.Float64).alias("cf_pct_5v5_w10_away"),
            pl.lit(cf_h - cf_a, dtype=pl.Float64).alias("cf_pct_5v5_w10_diff"),
            pl.lit(gsax_h, dtype=pl.Float64).alias("gsax_per_60_shrunk_home"),
            pl.lit(gsax_a, dtype=pl.Float64).alias("gsax_per_60_shrunk_away"),
            pl.lit(gsax_h - gsax_a, dtype=pl.Float64).alias("goalie_gsax_per_60_diff"),
            pl.lit(fwd_h, dtype=pl.Float64).alias("forward_weighted_ixg_per_60_home"),
            pl.lit(fwd_a, dtype=pl.Float64).alias("forward_weighted_ixg_per_60_away"),
            pl.lit(fwd_h - fwd_a, dtype=pl.Float64).alias("forward_ixg_per_60_diff"),
            pl.lit(def_h, dtype=pl.Float64).alias("defense_weighted_xgd_per_60_home"),
            pl.lit(def_a, dtype=pl.Float64).alias("defense_weighted_xgd_per_60_away"),
            pl.lit(def_h - def_a, dtype=pl.Float64).alias("defense_xgd_per_60_diff"),
        )

    def _collect_confidence_warnings(
        self,
        team_feat_df: pl.DataFrame,
        goalie_feat_df: pl.DataFrame,
        home_code: str,
        away_code: str,
        home_goalie_status: str,
        away_goalie_status: str,
    ) -> tuple[list[str], str | None, bool]:
        warnings: list[str] = []

        if "CONFIRMED" not in home_goalie_status.upper():
            warnings.append(
                f"Home starting goalie ({home_code}) is {home_goalie_status} (not yet confirmed)."
            )
        if "CONFIRMED" not in away_goalie_status.upper():
            warnings.append(
                f"Away starting goalie ({away_code}) is {away_goalie_status} (not yet confirmed)."
            )

        latest_date_str: str | None = None
        has_mp = True
        if not team_feat_df.is_empty():
            if "game_date" in team_feat_df.columns:
                latest_date_str = str(team_feat_df["game_date"].max())
            if "has_moneypuck_xg" in team_feat_df.columns:
                has_mp = bool(team_feat_df["has_moneypuck_xg"].tail(2).all())
                if not has_mp:
                    warnings.append(
                        "MoneyPuck xG lag detected on latest games; using NHL API fallback."
                    )

            for t_code in (home_code, away_code):
                t_rows = team_feat_df.filter(pl.col("team") == t_code)
                if not t_rows.is_empty() and "team_prior_games_n" in t_rows.columns:
                    n_g = int(t_rows["team_prior_games_n"][-1])
                    if n_g < TEAM_SHRINKAGE_K:
                        warnings.append(
                            f"Early-season shrinkage active for {t_code} "
                            f"({n_g} prior games < k={int(TEAM_SHRINKAGE_K)}; >50% prior weight)."
                        )

        if not goalie_feat_df.is_empty():
            for t_code in (home_code, away_code):
                g_rows = goalie_feat_df.filter(pl.col("team") == t_code)
                if not g_rows.is_empty() and "goalie_prior_games_n" in g_rows.columns:
                    n_starts = int(g_rows["goalie_prior_games_n"][-1])
                    if n_starts < GOALIE_SHRINKAGE_K:
                        warnings.append(
                            f"Low-sample goalie shrinkage active for {t_code} starter "
                            f"({n_starts} prior starts < k={int(GOALIE_SHRINKAGE_K)})."
                        )

        return warnings, latest_date_str, has_mp

    def _append_to_prediction_ledger(self, prediction: MatchupPrediction) -> Path:
        """Append prediction record to `data/predictions/hockey_predictions.parquet`."""
        ledger_dir = self.settings.paths.predictions_dir
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "hockey_predictions.parquet"

        new_row = pl.DataFrame(
            {
                "predicted_at_utc": [datetime.now(tz=UTC)],
                "sport": ["hockey"],
                "model_version": [self.metadata.model_version],
                "game_id": [int(prediction.game_id or -1)],
                "game_date": [prediction.game_date],
                "home_team": [prediction.home_team],
                "away_team": [prediction.away_team],
                "home_win_prob": [prediction.home_win_prob],
                "away_win_prob": [prediction.away_win_prob],
                "expected_home_goals": [prediction.expected_home_goals],
                "expected_away_goals": [prediction.expected_away_goals],
                "home_goalie_status": [prediction.home_goalie_status],
                "away_goalie_status": [prediction.away_goalie_status],
                "has_moneypuck_xg": [prediction.has_moneypuck_xg],
                "included_stage2_players": [prediction.player_projections is not None],
            }
        )

        if ledger_path.exists():
            existing = pl.read_parquet(ledger_path)
            combined = pl.concat([existing, new_row], how="vertical")
        else:
            combined = new_row

        combined.write_parquet(ledger_path)
        return ledger_path

    def _resolve_matchup_goalie_statuses(
        self,
        game_id_val: int | None,
        game_date: date,
        home_code: str,
        away_code: str,
        goalie_feat_df: pl.DataFrame,
        *,
        home_goalie_status_override: str | None = None,
        away_goalie_status_override: str | None = None,
        offline: bool = False,
    ) -> tuple[str, str]:
        """Resolve starting goalie confirmation statuses via Tier 1 & Tier 2 waterfall."""
        if home_goalie_status_override and away_goalie_status_override:
            return home_goalie_status_override, away_goalie_status_override

        h_status = home_goalie_status_override or "PROJECTED_HEURISTIC"
        a_status = away_goalie_status_override or "PROJECTED_HEURISTIC"

        # Tier 1a: Check if starter is already recorded in Silver goalie_game_logs for this game_id
        if game_id_val is not None and not goalie_feat_df.is_empty():
            g_ev_col = "game_id" if "game_id" in goalie_feat_df.columns else "event_id"
            game_goalies = goalie_feat_df.filter(
                (pl.col(g_ev_col) == game_id_val) & pl.col("is_starter")
            )
            if not game_goalies.is_empty():
                teams_with_starter = set(game_goalies["team"].to_list())
                if home_code in teams_with_starter and not home_goalie_status_override:
                    h_status = "CONFIRMED_NHL_OFFICIAL"
                if away_code in teams_with_starter and not away_goalie_status_override:
                    a_status = "CONFIRMED_NHL_OFFICIAL"

        if "CONFIRMED" in h_status and "CONFIRMED" in a_status:
            return h_status, a_status

        # Tier 1b: Check cached raw boxscore JSON if present on disk
        if game_id_val is not None:
            box_path = self.settings.paths.raw_dir / "nhl_api" / "boxscores" / f"{game_id_val}.json"
            if box_path.exists():
                import json

                box_payload = json.loads(box_path.read_text(encoding="utf-8"))
                pbg = box_payload.get("playerByGameStats") or {}
                if (pbg.get("homeTeam") or {}).get("goalies") and not home_goalie_status_override:
                    h_status = "CONFIRMED_NHL_OFFICIAL"
                if (pbg.get("awayTeam") or {}).get("goalies") and not away_goalie_status_override:
                    a_status = "CONFIRMED_NHL_OFFICIAL"

        if "CONFIRMED" in h_status and "CONFIRMED" in a_status:
            return h_status, a_status

        # Tier 2: Query DailyFaceoff morning-skate feed for target date when online
        if not offline:
            df_starters = fetch_dailyfaceoff_starting_goalies(game_date)
            for code_side, is_home_side in ((home_code, True), (away_code, False)):
                curr_st = h_status if is_home_side else a_status
                if "CONFIRMED" in curr_st:
                    continue
                entry = df_starters.get(code_side)
                if entry is not None:
                    raw_strength = str(entry.get("goalie_status", "Expected")).upper()
                    mapped = (
                        "CONFIRMED_MORNING_SKATE"
                        if "CONFIRM" in raw_strength
                        else "EXPECTED_MORNING_SKATE"
                    )
                    if is_home_side and not home_goalie_status_override:
                        h_status = mapped
                    elif not is_home_side and not away_goalie_status_override:
                        a_status = mapped

        return h_status, a_status

    def _select_active_team_skaters(
        self,
        stage2_df: pl.DataFrame,
        team_code: str,
        opp_code: str,
        is_home: bool,
        game_id_val: int | None,
        game_date: date,
        *,
        offline: bool = False,
    ) -> pl.DataFrame:
        """Select active roster skaters for `team_code` on `game_date` without historical ghosts."""
        target_season = derive_active_nhl_season(game_date)
        team_hist = stage2_df.filter(pl.col("team") == team_code)

        # Restrict to the target season (or latest available season for this team)
        season_hist = team_hist.filter(pl.col("season") == target_season)
        if season_hist.is_empty() and not team_hist.is_empty():
            max_season_val = team_hist["season"].cast(pl.Int32).to_list()
            max_season = max(max_season_val) if max_season_val else target_season
            season_hist = team_hist.filter(pl.col("season") == max_season)

        # When predicting a current-season game, filter to Official NHL active roster if available
        active_season_now = derive_active_nhl_season()
        if target_season == active_season_now:
            roster_ids: set[int] = set()
            raw_nhl_dir = self.settings.paths.raw_dir / "nhl_api"
            cached_roster_path = raw_nhl_dir / "rosters" / f"{team_code}_current.json"
            try:
                if not offline:
                    nhl_client = NHLWebAPIClient(cache_dir=raw_nhl_dir)
                    roster_payload = nhl_client.fetch_team_roster(team_code)
                    roster_df = parse_roster_payload(team_code, roster_payload)
                    roster_ids = set(
                        roster_df.filter(pl.col("position_group") != "G")["player_id"].to_list()
                    )
                elif cached_roster_path.exists():
                    import json

                    roster_payload = json.loads(cached_roster_path.read_text(encoding="utf-8"))
                    roster_df = parse_roster_payload(team_code, roster_payload)
                    roster_ids = set(
                        roster_df.filter(pl.col("position_group") != "G")["player_id"].to_list()
                    )
            except Exception:
                roster_ids = set()

            if roster_ids:
                filtered_by_roster = season_hist.filter(pl.col("player_id").is_in(roster_ids))
                if not filtered_by_roster.is_empty():
                    season_hist = filtered_by_roster

        latest_skaters = (
            season_hist.sort(["start_time_utc", "game_id"])
            .unique(subset=["player_id"], keep="last")
            .tail(20)
        )
        return latest_skaters.with_columns(
            pl.lit(game_id_val or -1, dtype=pl.Int64).alias("game_id"),
            pl.lit(is_home).alias("is_home"),
            pl.lit(opp_code).alias("opponent"),
        )

    def predict_matchup(
        self,
        home_team: str,
        away_team: str,
        game_date: date,
        include_players: bool = False,
        *,
        home_goalie_status: str | None = None,
        away_goalie_status: str | None = None,
        offline: bool = False,
        persist_ledger: bool = True,
    ) -> MatchupPrediction:
        """Predict Stage 1 calibrated win probability and optional Stage 2 player projections."""
        home_code = normalize_tricode(home_team)
        away_code = normalize_tricode(away_team)

        stage1_df, stage2_df, team_feat_df, goalie_feat_df = build_and_persist_gold_features(
            self.settings
        )

        matchup_row_df = self._synthesize_matchup_row_if_missing(
            stage1_df, home_code, away_code, game_date
        )

        # Validate live input feature schema against saved metadata.json SHA-256 fingerprint
        ModelRegistry.validate_features(
            matchup_row_df,
            self.metadata,
            stage=1,
            enforce_column_order=True,
        )

        s1_pred_df = self.stage1_model.predict(matchup_row_df)
        s1_row = s1_pred_df.row(0, named=True)
        game_id_val = int(s1_row["game_id"]) if s1_row.get("game_id") is not None else None
        home_prob = float(s1_row["home_win_prob"])
        away_prob = float(s1_row["away_win_prob"])
        exp_home_g = float(s1_row["expected_home_goals"])
        exp_away_g = float(s1_row["expected_away_goals"])

        resolved_home_status, resolved_away_status = self._resolve_matchup_goalie_statuses(
            game_id_val,
            game_date,
            home_code,
            away_code,
            goalie_feat_df,
            home_goalie_status_override=home_goalie_status,
            away_goalie_status_override=away_goalie_status,
            offline=offline,
        )

        warnings, latest_silver_date, has_mp = self._collect_confidence_warnings(
            team_feat_df,
            goalie_feat_df,
            home_code,
            away_code,
            resolved_home_status,
            resolved_away_status,
        )

        player_projections: PlayerMatchupProjections | None = None
        if include_players:
            if self.stage2_model is None or not self.stage2_model.is_fitted:
                raise RuntimeError(
                    "Stage 2 player projections requested (--include-players), "
                    "but no Stage 2 model is present in the bundle. "
                    "Run 'sports-ml train --sport hockey --include-stage2'."
                )
            if not stage2_df.is_empty():
                matchup_skaters = stage2_df.filter(
                    (pl.col("game_id") == game_id_val)
                    & pl.col("team").is_in([home_code, away_code])
                )
                if matchup_skaters.is_empty():
                    # Use each team's active current-season roster skaters conditioned on Stage 1
                    home_latest = self._select_active_team_skaters(
                        stage2_df,
                        home_code,
                        away_code,
                        True,
                        game_id_val,
                        game_date,
                        offline=offline,
                    )
                    away_latest = self._select_active_team_skaters(
                        stage2_df,
                        away_code,
                        home_code,
                        False,
                        game_id_val,
                        game_date,
                        offline=offline,
                    )
                    matchup_skaters = pl.concat([home_latest, away_latest], how="vertical")

                conditioned_skaters = matchup_skaters.with_columns(
                    pl.when(pl.col("team") == home_code)
                    .then(pl.lit(home_prob))
                    .otherwise(pl.lit(away_prob))
                    .alias("stage1_team_win_prob"),
                    pl.when(pl.col("team") == home_code)
                    .then(pl.lit(exp_home_g))
                    .otherwise(pl.lit(exp_away_g))
                    .alias("stage1_expected_team_goals"),
                )

                ModelRegistry.validate_features(
                    conditioned_skaters,
                    self.metadata,
                    stage=2,
                    enforce_column_order=True,
                )

                skater_preds = self.stage2_model.predict(conditioned_skaters)
                home_skaters = skater_preds.filter(pl.col("team") == home_code).sort(
                    "proj_points", descending=True
                )
                away_skaters = skater_preds.filter(pl.col("team") == away_code).sort(
                    "proj_points", descending=True
                )

                if not goalie_feat_df.is_empty():
                    starters = resolve_team_starting_goalie_features(goalie_feat_df)
                    matchup_g = starters.filter(
                        (pl.col("game_id") == game_id_val)
                        & pl.col("team").is_in([home_code, away_code])
                    )
                    if matchup_g.is_empty():
                        matchup_g = (
                            starters.filter(pl.col("team").is_in([home_code, away_code]))
                            .unique(subset=["team"], keep="last")
                            .with_columns(
                                pl.lit(game_id_val or -1, dtype=pl.Int64).alias("game_id"),
                                (pl.col("team") == home_code).alias("is_home"),
                                pl.when(pl.col("team") == home_code)
                                .then(pl.lit(away_code))
                                .otherwise(pl.lit(home_code))
                                .alias("opponent"),
                            )
                        )
                    goalies_pred = self.stage2_model.predict_goalies(
                        matchup_g,
                        skater_projections_df=skater_preds,
                        stage1_predictions_df=s1_pred_df,
                    )
                else:
                    goalies_pred = pl.DataFrame()

                player_projections = PlayerMatchupProjections(
                    home_skaters=home_skaters,
                    away_skaters=away_skaters,
                    goalies=goalies_pred,
                )

        pred_obj = MatchupPrediction(
            game_id=game_id_val,
            game_date=game_date,
            home_team=home_code,
            away_team=away_code,
            home_win_prob=home_prob,
            away_win_prob=away_prob,
            expected_home_goals=exp_home_g,
            expected_away_goals=exp_away_g,
            home_goalie_status=resolved_home_status,
            away_goalie_status=resolved_away_status,
            latest_silver_game_date=latest_silver_date,
            has_moneypuck_xg=has_mp,
            warnings=warnings,
            player_projections=player_projections,
        )

        if persist_ledger:
            self._append_to_prediction_ledger(pred_obj)

        return pred_obj

    def predict_date(
        self,
        game_date: date,
        *,
        include_players: bool = False,
        matchup_filter: str | None = None,
        offline: bool = False,
        persist_ledger: bool = True,
    ) -> list[MatchupPrediction]:
        """Predict all scheduled matchups on `game_date` (or a specific `AWAY@HOME` filter)."""
        if matchup_filter:
            cleaned = matchup_filter.upper().replace(" ", "")
            if "@" in cleaned:
                away_part, home_part = cleaned.split("@", maxsplit=1)
            elif "VS" in cleaned:
                home_part, away_part = cleaned.split("VS", maxsplit=1)
            else:
                raise ValueError(
                    f"Invalid --matchup format '{matchup_filter}'. Expected 'AWAY@HOME'."
                )
            return [
                self.predict_matchup(
                    home_team=home_part,
                    away_team=away_part,
                    game_date=game_date,
                    include_players=include_players,
                    offline=offline,
                    persist_ledger=persist_ledger,
                )
            ]

        games_path = self.settings.paths.silver_dir / "hockey" / "games.parquet"
        if not games_path.exists():
            raise FileNotFoundError(
                f"Silver games table not found at '{games_path}'. Run 'sports-ml ingest' first."
            )

        games_df = pl.read_parquet(games_path)
        slate_df = games_df.filter(pl.col("game_date") == game_date)
        if slate_df.is_empty():
            slate_df = games_df.tail(1)

        results: list[MatchupPrediction] = []
        for row in slate_df.iter_rows(named=True):
            results.append(
                self.predict_matchup(
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    game_date=row["game_date"],
                    include_players=include_players,
                    offline=offline,
                    persist_ledger=persist_ledger,
                )
            )
        return results


def score_prediction_ledger(settings: Settings) -> BenchmarkComparisonReport:
    """Join `data/predictions/hockey_predictions.parquet` against completed `FINAL` Silver games."""
    ledger_path = settings.paths.predictions_dir / "hockey_predictions.parquet"
    games_path = settings.paths.silver_dir / "hockey" / "games.parquet"

    if not ledger_path.exists():
        raise FileNotFoundError(
            f"Prediction ledger not found at '{ledger_path}'. Run 'sports-ml predict' first."
        )
    if not games_path.exists():
        raise FileNotFoundError(f"Silver games table not found at '{games_path}'.")

    ledger_df = pl.read_parquet(ledger_path).unique(
        subset=["game_date", "home_team", "away_team"], keep="last"
    )
    completed_games = (
        pl.read_parquet(games_path)
        .filter((pl.col("game_state") == "FINAL") & pl.col("home_win").is_not_null())
        .select(["game_date", "home_team", "away_team", "home_win"])
    )

    joined = ledger_df.join(
        completed_games, on=["game_date", "home_team", "away_team"], how="inner"
    )
    if joined.is_empty():
        raise ValueError("No predictions in the ledger match completed FINAL games yet.")

    probs = np.asarray(joined["home_win_prob"].to_numpy(), dtype=np.float64)
    y_true = np.asarray(joined["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)
    return compare_against_baselines(probs, y_true)
