"""Unified two-stage EPL soccer matchup predictor service (`soccer/models/predictor.py`).

Exposes `SoccerMatchupPredictor.predict_matchup(..., include_players=...)` and
`predict_date(...)`, surfaces pre-match confidence warnings (unconfirmed starting XI
vs. confirmed lineup, doubtful players with `chance_of_playing < 75%`, promoted-club
early-season prior, Understat lag fallback), appends to
`data/predictions/soccer_predictions.parquet`, and scores the live prediction ledger
against completed Silver EPL matches (`1X2` RPS, Log Loss, Brier Score, ECE).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sports_ml.config import Settings, get_settings
from sports_ml.core.cache import resolve_cache_read_path
from sports_ml.core.evaluation import (
    MulticlassBenchmarkReport,
    compare_multiclass_against_baselines,
)
from sports_ml.core.ratings import compute_elo_1x2_probabilities
from sports_ml.core.registry import ModelMetadata, ModelRegistry
from sports_ml.sports.soccer.constants import normalize_soccer_club
from sports_ml.sports.soccer.features.goalkeeper import (
    GK_SHRINKAGE_K,
    build_goalkeeper_rolling_features,
    resolve_team_starting_gk_features,
)
from sports_ml.sports.soccer.features.player import build_soccer_stage2_player_features
from sports_ml.sports.soccer.features.team import (
    SOCCER_TEAM_SHRINKAGE_K,
    build_soccer_stage1_match_features,
    build_soccer_team_rolling_features,
)
from sports_ml.sports.soccer.ingestion.fpl_api import (
    FPLAPIClient,
    parse_bootstrap_squad_df,
)
from sports_ml.sports.soccer.ingestion.lineups import SoccerPreMatchLineupResolver
from sports_ml.sports.soccer.ingestion.odds import fetch_espn_epl_date_odds
from sports_ml.sports.soccer.models.stage1_match import Stage1SoccerMatchModel
from sports_ml.sports.soccer.models.stage2_player import Stage2SoccerPlayerModel


class SoccerPlayerMatchupProjections(BaseModel):
    """Container for Stage 2 outfield and goalkeeper projection DataFrames."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    home_outfield: pl.DataFrame
    away_outfield: pl.DataFrame
    goalkeepers: pl.DataFrame


class SoccerMatchupPrediction(BaseModel):
    """Structured Stage 1 (`1X2` + goals) and Stage 2 (optional) EPL matchup prediction."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    match_id: int | None = None
    match_date: date
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    market_prob_home: float | None = None
    market_prob_draw: float | None = None
    market_prob_away: float | None = None
    market_home_decimal_odds: float | None = None
    market_draw_decimal_odds: float | None = None
    market_away_decimal_odds: float | None = None
    home_lineup_status: str = "HEURISTIC_XI"
    away_lineup_status: str = "HEURISTIC_XI"
    latest_silver_match_date: str | None = None
    has_understat_xg: bool = True
    warnings: list[str] = Field(default_factory=list)
    player_projections: SoccerPlayerMatchupProjections | None = None


def build_and_persist_soccer_gold_features(
    settings: Settings,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build and write Gold EPL Stage 1 and Stage 2 feature matrices from Silver Parquet tables."""
    silver_dir = settings.paths.silver_dir / "soccer"
    gold_dir = settings.paths.gold_dir / "soccer"
    gold_dir.mkdir(parents=True, exist_ok=True)

    matches_path = silver_dir / "matches.parquet"
    teams_path = silver_dir / "team_match_logs.parquet"
    outfield_path = silver_dir / "outfield_match_logs.parquet"
    gk_path = silver_dir / "gk_match_logs.parquet"
    all_comp_path = silver_dir / "club_all_competitions_schedule.parquet"
    shot_micro_path = silver_dir / "team_shot_micro_logs.parquet"

    if not matches_path.exists() or not teams_path.exists():
        raise FileNotFoundError(
            f"Silver Parquet tables not found in '{silver_dir}'. "
            "Run 'sports-ml ingest --sport soccer' first."
        )

    matches_df = pl.read_parquet(matches_path)
    team_logs_df = pl.read_parquet(teams_path)
    outfield_logs_df = pl.read_parquet(outfield_path) if outfield_path.exists() else pl.DataFrame()
    gk_logs_df = pl.read_parquet(gk_path) if gk_path.exists() else pl.DataFrame()
    all_comp_df = pl.read_parquet(all_comp_path) if all_comp_path.exists() else None
    shot_micro_df = pl.read_parquet(shot_micro_path) if shot_micro_path.exists() else None

    # Load cached FPL bootstrap squad metadata if available
    squad_meta_df: pl.DataFrame | None = None
    bootstrap_cache = resolve_cache_read_path(
        settings.paths.raw_dir / "fpl_api" / "bootstrap_static.json.gz"
    )
    if bootstrap_cache is not None:
        try:
            fpl_client = FPLAPIClient(cache_dir=settings.paths.raw_dir / "fpl_api")
            bootstrap = fpl_client.fetch_bootstrap_static(ttl_seconds=None)
            squad_meta_df = parse_bootstrap_squad_df(bootstrap)
        except Exception:
            squad_meta_df = None

    team_feat_df = build_soccer_team_rolling_features(
        team_logs_df,
        all_comp_schedule_df=all_comp_df,
        team_shot_micro_logs=shot_micro_df,
    )
    gk_feat_df = (
        build_goalkeeper_rolling_features(gk_logs_df)
        if not gk_logs_df.is_empty()
        else pl.DataFrame()
    )
    stage1_df = build_soccer_stage1_match_features(
        matches_df,
        team_logs_df,
        outfield_logs_df if not outfield_logs_df.is_empty() else None,
        gk_logs_df if not gk_logs_df.is_empty() else None,
        all_comp_schedule_df=all_comp_df,
        team_shot_micro_logs=shot_micro_df,
    )
    stage2_df = (
        build_soccer_stage2_player_features(
            outfield_logs_df,
            team_feat_df,
            gk_feat_df,
            squad_metadata_df=squad_meta_df,
        )
        if not outfield_logs_df.is_empty()
        else pl.DataFrame()
    )

    stage1_df.write_parquet(gold_dir / "stage1_match_features.parquet", compression="zstd")
    if not stage2_df.is_empty():
        stage2_df.write_parquet(gold_dir / "stage2_player_features.parquet", compression="zstd")

    return stage1_df, stage2_df, team_feat_df, gk_feat_df


class SoccerMatchupPredictor:
    """Unified two-stage inference service for EPL soccer matchups."""

    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        stage1_model: Stage1SoccerMatchModel,
        stage2_model: Stage2SoccerPlayerModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.metadata = metadata
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.settings = settings or get_settings()
        self._live_odds_cache: dict[date, pl.DataFrame] = {}

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        *,
        settings: Settings | None = None,
    ) -> SoccerMatchupPredictor:
        """Load a fingerprinted EPL model bundle from `artifact_dir`."""
        bundle = ModelRegistry.load_bundle(artifact_dir)
        resolved_settings = settings or get_settings(root_dir=bundle.bundle_dir.parents[2])
        return cls(
            metadata=bundle.metadata,
            stage1_model=bundle.stage1_model,
            stage2_model=bundle.stage2_model,
            settings=resolved_settings,
        )

    def _load_cached_squad_df(self) -> pl.DataFrame | None:
        bootstrap_cache = resolve_cache_read_path(
            self.settings.paths.raw_dir / "fpl_api" / "bootstrap_static.json.gz"
        )
        if bootstrap_cache is None:
            return None
        try:
            fpl_client = FPLAPIClient(cache_dir=self.settings.paths.raw_dir / "fpl_api")
            bootstrap = fpl_client.fetch_bootstrap_static(ttl_seconds=None)
            return parse_bootstrap_squad_df(bootstrap)
        except Exception:
            return None

    def _synthesize_matchup_row_if_missing(
        self,
        stage1_df: pl.DataFrame,
        home_code: str,
        away_code: str,
        match_date: date,
    ) -> pl.DataFrame:
        date_col = "match_date" if "match_date" in stage1_df.columns else "game_date"
        exact = stage1_df.filter(
            (pl.col(date_col) == match_date)
            & (pl.col("home_team") == home_code)
            & (pl.col("away_team") == away_code)
        )
        if not exact.is_empty():
            return exact.tail(1)

        def _latest_val(t_code: str, col_base: str, default_val: float) -> float:
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

        elo_h = _latest_val(home_code, "elo", 1500.0)
        elo_a = _latest_val(away_code, "elo", 1500.0)
        elo_diff = elo_h + 48.0 - elo_a
        p_h_elo, p_d_elo, p_a_elo = compute_elo_1x2_probabilities(elo_diff, draw_param=0.26)

        npxgd_h = _latest_val(home_code, "npxgd_per_90_w10", 0.0)
        npxgd_a = _latest_val(away_code, "npxgd_per_90_w10", 0.0)
        npxgf_pct_h = _latest_val(home_code, "npxgf_pct_w10", 0.50)
        npxgf_pct_a = _latest_val(away_code, "npxgf_pct_w10", 0.50)
        ppda_h = _latest_val(home_code, "ppda_w10", 11.5)
        ppda_a = _latest_val(away_code, "ppda_w10", 11.5)
        deep_h = _latest_val(home_code, "deep_completions_per_90_w10", 6.5)
        deep_a = _latest_val(away_code, "deep_completions_per_90_w10", 6.5)
        gsaa_h = _latest_val(home_code, "gsaa_per_90_shrunk", 0.0)
        gsaa_a = _latest_val(away_code, "gsaa_per_90_shrunk", 0.0)
        fwd_h = _latest_val(home_code, "fwd_weighted_npxg_per_90", 0.36)
        fwd_a = _latest_val(away_code, "fwd_weighted_npxg_per_90", 0.36)
        mid_h = _latest_val(home_code, "mid_weighted_xg_chain_per_90", 0.52)
        mid_a = _latest_val(away_code, "mid_weighted_xg_chain_per_90", 0.52)

        latest_row = stage1_df.tail(1)
        synth_id = int(f"{match_date.year}99999")
        return latest_row.with_columns(
            pl.lit(synth_id, dtype=pl.Int64).alias("match_id"),
            pl.lit(synth_id, dtype=pl.Int64).alias("event_id"),
            pl.lit(match_date, dtype=pl.Date).alias("match_date"),
            pl.lit(home_code, dtype=pl.Utf8).alias("home_team"),
            pl.lit(away_code, dtype=pl.Utf8).alias("away_team"),
            pl.lit(elo_h, dtype=pl.Float64).alias("elo_home_pre"),
            pl.lit(elo_a, dtype=pl.Float64).alias("elo_away_pre"),
            pl.lit(elo_diff, dtype=pl.Float64).alias("elo_diff_pre"),
            pl.lit(p_h_elo, dtype=pl.Float64).alias("elo_home_win_prob"),
            pl.lit(p_d_elo, dtype=pl.Float64).alias("elo_draw_prob"),
            pl.lit(p_a_elo, dtype=pl.Float64).alias("elo_away_win_prob"),
            pl.lit(npxgd_h, dtype=pl.Float64).alias("npxgd_per_90_w10_home"),
            pl.lit(npxgd_a, dtype=pl.Float64).alias("npxgd_per_90_w10_away"),
            pl.lit(npxgd_h - npxgd_a, dtype=pl.Float64).alias("npxgd_per_90_w10_diff"),
            pl.lit(npxgf_pct_h, dtype=pl.Float64).alias("npxgf_pct_w10_home"),
            pl.lit(npxgf_pct_a, dtype=pl.Float64).alias("npxgf_pct_w10_away"),
            pl.lit(npxgf_pct_h - npxgf_pct_a, dtype=pl.Float64).alias("npxgf_pct_w10_diff"),
            pl.lit(ppda_h, dtype=pl.Float64).alias("ppda_w10_home"),
            pl.lit(ppda_a, dtype=pl.Float64).alias("ppda_w10_away"),
            pl.lit(ppda_a - ppda_h, dtype=pl.Float64).alias("pressing_intensity_diff"),
            pl.lit(deep_h, dtype=pl.Float64).alias("deep_completions_per_90_w10_home"),
            pl.lit(deep_a, dtype=pl.Float64).alias("deep_completions_per_90_w10_away"),
            pl.lit(deep_h - deep_a, dtype=pl.Float64).alias("deep_completions_per_90_w10_diff"),
            pl.lit(gsaa_h, dtype=pl.Float64).alias("gsaa_per_90_shrunk_home"),
            pl.lit(gsaa_a, dtype=pl.Float64).alias("gsaa_per_90_shrunk_away"),
            pl.lit(gsaa_h - gsaa_a, dtype=pl.Float64).alias("gk_gsaa_per_90_diff"),
            pl.lit(fwd_h, dtype=pl.Float64).alias("fwd_weighted_npxg_per_90_home"),
            pl.lit(fwd_a, dtype=pl.Float64).alias("fwd_weighted_npxg_per_90_away"),
            pl.lit(fwd_h - fwd_a, dtype=pl.Float64).alias("fwd_npxg_per_90_diff"),
            pl.lit(mid_h, dtype=pl.Float64).alias("mid_weighted_xg_chain_per_90_home"),
            pl.lit(mid_a, dtype=pl.Float64).alias("mid_weighted_xg_chain_per_90_away"),
            pl.lit(mid_h - mid_a, dtype=pl.Float64).alias("mid_xg_chain_per_90_diff"),
        )

    def _collect_confidence_warnings(
        self,
        team_feat_df: pl.DataFrame,
        gk_feat_df: pl.DataFrame,
        squad_df: pl.DataFrame | None,
        home_code: str,
        away_code: str,
        home_lineup_status: str,
        away_lineup_status: str,
        home_doubtful_ids: list[int],
        away_doubtful_ids: list[int],
    ) -> tuple[list[str], str | None, bool]:
        warnings: list[str] = []
        latest_date_str: str | None = None
        has_understat = True

        if not team_feat_df.is_empty():
            date_col = "match_date" if "match_date" in team_feat_df.columns else "game_date"
            max_d = team_feat_df[date_col].max()
            if max_d is not None:
                latest_date_str = str(max_d)[:10]

            for t_code in (home_code, away_code):
                t_rows = team_feat_df.filter(pl.col("team") == t_code)
                if not t_rows.is_empty():
                    last_row = t_rows.row(-1, named=True)
                    if not bool(last_row.get("has_understat_xg", True)):
                        has_understat = False
                        warnings.append(
                            f"{t_code}: Understat xG/PPDA not yet published for most recent match "
                            "(using shot-based fallback)."
                        )
                    if bool(last_row.get("is_promoted_team", False)):
                        warnings.append(
                            f"{t_code}: Promoted EFL Championship club prior applied "
                            "(promoted_team_rating=1390.0)."
                        )
                    prior_n = int(last_row.get("team_prior_matches_n", 0))
                    if prior_n < SOCCER_TEAM_SHRINKAGE_K:
                        warnings.append(
                            f"{t_code}: Early-season team sample "
                            f"(n={prior_n} < k={int(SOCCER_TEAM_SHRINKAGE_K)}); "
                            "Bayesian prior shrinkage active."
                        )

        if home_lineup_status != "CONFIRMED_OFFICIAL_XI":
            warnings.append(
                f"{home_code}: Starting XI not yet officially confirmed ({home_lineup_status})."
            )
        if away_lineup_status != "CONFIRMED_OFFICIAL_XI":
            warnings.append(
                f"{away_code}: Starting XI not yet officially confirmed ({away_lineup_status})."
            )

        if squad_df is not None and not squad_df.is_empty():
            for t_code, d_ids in ((home_code, home_doubtful_ids), (away_code, away_doubtful_ids)):
                if d_ids:
                    d_names = (
                        squad_df.filter(pl.col("player_id").is_in(d_ids))["web_name"].to_list()
                        if "web_name" in squad_df.columns
                        else [str(i) for i in d_ids]
                    )
                    warnings.append(
                        f"{t_code}: Doubtful squad players (<100% chance of playing): "
                        f"{', '.join(d_names[:4])}."
                    )

        if not gk_feat_df.is_empty():
            starters = resolve_team_starting_gk_features(gk_feat_df)
            for t_code in (home_code, away_code):
                g_rows = starters.filter(pl.col("team") == t_code)
                if not g_rows.is_empty():
                    last_g = g_rows.row(-1, named=True)
                    g_n = int(last_g.get("gk_prior_matches_n", 0))
                    if g_n < GK_SHRINKAGE_K:
                        warnings.append(
                            f"{t_code}: Starting GK has low sample size "
                            f"(n={g_n} < k={int(GK_SHRINKAGE_K)}); shrunk toward league mean."
                        )

        return warnings, latest_date_str, has_understat

    def _select_active_team_outfield(
        self,
        stage2_df: pl.DataFrame,
        team_code: str,
        opp_code: str,
        is_home: bool,
        match_id: int,
        match_date: date,
        active_outfield_ids: list[int],
    ) -> pl.DataFrame:
        team_rows = stage2_df.filter(pl.col("team") == team_code)
        if team_rows.is_empty():
            return team_rows

        latest_per_player = (
            team_rows.sort(["start_time_utc", "player_id"])
            .unique(subset=["player_id"], keep="last")
            .filter(pl.col("availability_weight") > 0.0)
        )
        if active_outfield_ids:
            filtered = latest_per_player.filter(pl.col("player_id").is_in(active_outfield_ids))
            if not filtered.is_empty():
                latest_per_player = filtered

        latest_per_player = latest_per_player.sort("rolling_minutes", descending=True).head(15)
        return latest_per_player.with_columns(
            pl.lit(match_id, dtype=pl.Int64).alias("match_id"),
            pl.lit(match_id, dtype=pl.Int64).alias("event_id"),
            pl.lit(match_date, dtype=pl.Date).alias("match_date"),
            pl.lit(opp_code, dtype=pl.Utf8).alias("opponent"),
            pl.lit(is_home, dtype=pl.Boolean).alias("is_home"),
        )

    def _select_active_team_gk(
        self,
        starters_df: pl.DataFrame,
        team_code: str,
        opp_code: str,
        is_home: bool,
        match_id: int,
        match_date: date,
        starting_gk_id: int,
    ) -> pl.DataFrame:
        team_gks = starters_df.filter(pl.col("team") == team_code)
        if team_gks.is_empty():
            return team_gks
        exact_gk = team_gks.filter(pl.col("player_id") == starting_gk_id)
        chosen = exact_gk.tail(1) if not exact_gk.is_empty() else team_gks.tail(1)
        return chosen.with_columns(
            pl.lit(match_id, dtype=pl.Int64).alias("match_id"),
            pl.lit(match_id, dtype=pl.Int64).alias("event_id"),
            pl.lit(match_date, dtype=pl.Date).alias("match_date"),
            pl.lit(opp_code, dtype=pl.Utf8).alias("opponent"),
            pl.lit(is_home, dtype=pl.Boolean).alias("is_home"),
        )

    def _append_to_prediction_ledger(self, pred: SoccerMatchupPrediction) -> Path:
        ledger_dir = self.settings.paths.predictions_dir
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "soccer_predictions.parquet"

        new_row = pl.DataFrame(
            {
                "predicted_at_utc": [datetime.now(tz=UTC)],
                "model_version": [self.metadata.model_version],
                "schema_sha256": [self.metadata.schema_sha256],
                "match_id": [int(pred.match_id) if pred.match_id is not None else -1],
                "match_date": [pred.match_date],
                "home_team": [pred.home_team],
                "away_team": [pred.away_team],
                "home_win_prob": [pred.home_win_prob],
                "draw_prob": [pred.draw_prob],
                "away_win_prob": [pred.away_win_prob],
                "expected_home_goals": [pred.expected_home_goals],
                "expected_away_goals": [pred.expected_away_goals],
                "market_prob_home": [pred.market_prob_home],
                "market_prob_draw": [pred.market_prob_draw],
                "market_prob_away": [pred.market_prob_away],
                "market_home_decimal_odds": [pred.market_home_decimal_odds],
                "market_draw_decimal_odds": [pred.market_draw_decimal_odds],
                "market_away_decimal_odds": [pred.market_away_decimal_odds],
                "home_lineup_status": [pred.home_lineup_status],
                "away_lineup_status": [pred.away_lineup_status],
                "included_stage2": [pred.player_projections is not None],
            }
        )
        if ledger_path.exists():
            existing = pl.read_parquet(ledger_path)
            combined = pl.concat([existing, new_row], how="diagonal_relaxed").unique(
                subset=["match_date", "home_team", "away_team", "model_version"],
                keep="last",
            )
        else:
            combined = new_row
        combined.write_parquet(ledger_path, compression="zstd")
        return ledger_path

    def _resolve_matchup_market_odds(
        self,
        p_row: dict[str, object],
        home_code: str,
        away_code: str,
        match_date: date,
        *,
        offline: bool = False,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
        """Resolve evaluation-only de-vigged `1X2` market probabilities and decimal odds."""
        raw_ph = p_row.get("market_prob_home")
        raw_pd = p_row.get("market_prob_draw")
        raw_pa = p_row.get("market_prob_away")
        raw_dh = p_row.get("market_home_decimal_odds")
        raw_dd = p_row.get("market_draw_decimal_odds")
        raw_da = p_row.get("market_away_decimal_odds")
        if (
            isinstance(raw_ph, (int, float))
            and isinstance(raw_pd, (int, float))
            and isinstance(raw_pa, (int, float))
        ):
            return (
                float(raw_ph),
                float(raw_pd),
                float(raw_pa),
                float(raw_dh) if isinstance(raw_dh, (int, float)) else None,
                float(raw_dd) if isinstance(raw_dd, (int, float)) else None,
                float(raw_da) if isinstance(raw_da, (int, float)) else None,
            )

        if offline:
            return (None, None, None, None, None, None)

        if match_date not in self._live_odds_cache:
            self._live_odds_cache[match_date] = fetch_espn_epl_date_odds(match_date, offline=False)

        day_odds = self._live_odds_cache[match_date]
        if day_odds.is_empty():
            return (None, None, None, None, None, None)

        matched = day_odds.filter(
            (pl.col("home_team") == home_code) & (pl.col("away_team") == away_code)
        )
        if matched.is_empty():
            return (None, None, None, None, None, None)

        r = matched.row(-1, named=True)
        return (
            float(r["market_prob_home"]) if r.get("market_prob_home") is not None else None,
            float(r["market_prob_draw"]) if r.get("market_prob_draw") is not None else None,
            float(r["market_prob_away"]) if r.get("market_prob_away") is not None else None,
            float(r["market_home_decimal_odds"])
            if r.get("market_home_decimal_odds") is not None
            else None,
            float(r["market_draw_decimal_odds"])
            if r.get("market_draw_decimal_odds") is not None
            else None,
            float(r["market_away_decimal_odds"])
            if r.get("market_away_decimal_odds") is not None
            else None,
        )

    def predict_matchup(
        self,
        home_team: str,
        away_team: str,
        match_date: date,
        *,
        include_players: bool = False,
        offline: bool = False,
        persist_ledger: bool = True,
        _gold_bundle: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame] | None = None,
    ) -> SoccerMatchupPrediction:
        """Predict calibrated Stage 1 `1X2` probabilities and optional Stage 2 projections."""
        _ = offline
        home_code = normalize_soccer_club(home_team)
        away_code = normalize_soccer_club(away_team)

        stage1_df, stage2_df, team_feat_df, gk_feat_df = (
            _gold_bundle
            if _gold_bundle is not None
            else build_and_persist_soccer_gold_features(self.settings)
        )
        matchup_row = self._synthesize_matchup_row_if_missing(
            stage1_df, home_code, away_code, match_date
        )

        squad_df = self._load_cached_squad_df()
        resolver = SoccerPreMatchLineupResolver()
        home_lineup = resolver.resolve_team_lineup(
            home_code,
            squad_df=squad_df,
            historical_outfield_logs=stage2_df if not stage2_df.is_empty() else None,
            historical_gk_logs=gk_feat_df if not gk_feat_df.is_empty() else None,
        )
        away_lineup = resolver.resolve_team_lineup(
            away_code,
            squad_df=squad_df,
            historical_outfield_logs=stage2_df if not stage2_df.is_empty() else None,
            historical_gk_logs=gk_feat_df if not gk_feat_df.is_empty() else None,
        )

        ModelRegistry.validate_features(
            matchup_row,
            self.metadata,
            stage=1,
            enforce_column_order=True,
        )

        s1_pred_df = self.stage1_model.predict(matchup_row)
        p_row = s1_pred_df.row(0, named=True)
        ev_col = "match_id" if "match_id" in s1_pred_df.columns else "event_id"
        match_id_val = int(p_row[ev_col]) if p_row.get(ev_col) is not None else None
        home_prob = float(p_row["home_win_prob"])
        draw_prob = float(p_row["draw_prob"])
        away_prob = float(p_row["away_win_prob"])
        exp_home_g = float(p_row["expected_home_goals"])
        exp_away_g = float(p_row["expected_away_goals"])

        warnings, latest_silver_date, has_us = self._collect_confidence_warnings(
            team_feat_df,
            gk_feat_df,
            squad_df,
            home_code,
            away_code,
            home_lineup.lineup_status,
            away_lineup.lineup_status,
            home_lineup.doubtful_player_ids,
            away_lineup.doubtful_player_ids,
        )

        player_projections: SoccerPlayerMatchupProjections | None = None
        if include_players:
            if self.stage2_model is None or not self.stage2_model.is_fitted:
                raise RuntimeError(
                    "Stage 2 player projections requested (--include-players), "
                    "but no Stage 2 model is present in the bundle. "
                    "Run 'sports-ml train --sport soccer --include-stage2'."
                )
            if not stage2_df.is_empty() and match_id_val is not None:
                s2_ev_col = "match_id" if "match_id" in stage2_df.columns else "event_id"
                matchup_outfield = stage2_df.filter(
                    (pl.col(s2_ev_col) == match_id_val)
                    & pl.col("team").is_in([home_code, away_code])
                )
                if matchup_outfield.is_empty():
                    home_out = self._select_active_team_outfield(
                        stage2_df,
                        home_code,
                        away_code,
                        True,
                        match_id_val,
                        match_date,
                        home_lineup.active_outfield_ids,
                    )
                    away_out = self._select_active_team_outfield(
                        stage2_df,
                        away_code,
                        home_code,
                        False,
                        match_id_val,
                        match_date,
                        away_lineup.active_outfield_ids,
                    )
                    matchup_outfield = pl.concat([home_out, away_out], how="vertical")

                conditioned_outfield = matchup_outfield.with_columns(
                    pl.when(pl.col("team") == home_code)
                    .then(pl.lit(home_prob))
                    .otherwise(pl.lit(away_prob))
                    .alias("stage1_team_win_prob"),
                    pl.lit(draw_prob, dtype=pl.Float64).alias("stage1_draw_prob"),
                    pl.when(pl.col("team") == home_code)
                    .then(pl.lit(exp_home_g))
                    .otherwise(pl.lit(exp_away_g))
                    .alias("stage1_expected_team_goals"),
                )

                ModelRegistry.validate_features(
                    conditioned_outfield,
                    self.metadata,
                    stage=2,
                    enforce_column_order=True,
                )

                outfield_preds = self.stage2_model.predict(conditioned_outfield)
                home_outfield = outfield_preds.filter(pl.col("team") == home_code).sort(
                    "proj_goal_involvements", descending=True
                )
                away_outfield = outfield_preds.filter(pl.col("team") == away_code).sort(
                    "proj_goal_involvements", descending=True
                )

                if not gk_feat_df.is_empty():
                    starters = resolve_team_starting_gk_features(gk_feat_df)
                    g_ev_col = "match_id" if "match_id" in starters.columns else "event_id"
                    matchup_gk = starters.filter(
                        (pl.col(g_ev_col) == match_id_val)
                        & pl.col("team").is_in([home_code, away_code])
                    )
                    if matchup_gk.is_empty():
                        h_gk = self._select_active_team_gk(
                            starters,
                            home_code,
                            away_code,
                            True,
                            match_id_val,
                            match_date,
                            home_lineup.starting_gk_id,
                        )
                        a_gk = self._select_active_team_gk(
                            starters,
                            away_code,
                            home_code,
                            False,
                            match_id_val,
                            match_date,
                            away_lineup.starting_gk_id,
                        )
                        matchup_gk = pl.concat([h_gk, a_gk], how="vertical")
                    gks_pred = self.stage2_model.predict_goalkeepers(
                        matchup_gk,
                        outfield_projections_df=outfield_preds,
                        stage1_predictions_df=s1_pred_df,
                    )
                else:
                    gks_pred = pl.DataFrame()

                player_projections = SoccerPlayerMatchupProjections(
                    home_outfield=home_outfield,
                    away_outfield=away_outfield,
                    goalkeepers=gks_pred,
                )

        (
            mkt_p_home,
            mkt_p_draw,
            mkt_p_away,
            mkt_d_home,
            mkt_d_draw,
            mkt_d_away,
        ) = self._resolve_matchup_market_odds(
            p_row,
            home_code,
            away_code,
            match_date,
            offline=offline,
        )

        pred_obj = SoccerMatchupPrediction(
            match_id=match_id_val,
            match_date=match_date,
            home_team=home_code,
            away_team=away_code,
            home_win_prob=home_prob,
            draw_prob=draw_prob,
            away_win_prob=away_prob,
            expected_home_goals=exp_home_g,
            expected_away_goals=exp_away_g,
            market_prob_home=mkt_p_home,
            market_prob_draw=mkt_p_draw,
            market_prob_away=mkt_p_away,
            market_home_decimal_odds=mkt_d_home,
            market_draw_decimal_odds=mkt_d_draw,
            market_away_decimal_odds=mkt_d_away,
            home_lineup_status=home_lineup.lineup_status,
            away_lineup_status=away_lineup.lineup_status,
            latest_silver_match_date=latest_silver_date,
            has_understat_xg=has_us,
            warnings=warnings,
            player_projections=player_projections,
        )

        if persist_ledger:
            self._append_to_prediction_ledger(pred_obj)

        return pred_obj

    def predict_date(
        self,
        match_date: date,
        *,
        include_players: bool = False,
        matchup_filter: str | None = None,
        offline: bool = False,
        persist_ledger: bool = True,
    ) -> list[SoccerMatchupPrediction]:
        """Predict all scheduled EPL matches on `match_date` (or a specific `AWAY@HOME` filter)."""
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
                    match_date=match_date,
                    include_players=include_players,
                    offline=offline,
                    persist_ledger=persist_ledger,
                )
            ]

        matches_path = self.settings.paths.silver_dir / "soccer" / "matches.parquet"
        if not matches_path.exists():
            raise FileNotFoundError(
                f"Silver matches table not found at '{matches_path}'. "
                "Run 'sports-ml ingest --sport soccer' first."
            )

        matches_df = pl.read_parquet(matches_path)
        slate_df = matches_df.filter(pl.col("match_date") == match_date)
        if slate_df.is_empty():
            slate_df = matches_df.tail(1)

        gold_bundle = build_and_persist_soccer_gold_features(self.settings)
        results: list[SoccerMatchupPrediction] = []
        for row in slate_df.iter_rows(named=True):
            results.append(
                self.predict_matchup(
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    match_date=row["match_date"],
                    include_players=include_players,
                    offline=offline,
                    persist_ledger=persist_ledger,
                    _gold_bundle=gold_bundle,
                )
            )
        return results

    def predict_team_next_games(
        self,
        team: str,
        n_games: int = 1,
        *,
        from_date: date | None = None,
        include_players: bool = False,
        offline: bool = False,
        persist_ledger: bool = True,
    ) -> list[SoccerMatchupPrediction]:
        """Predict the next `n_games` scheduled EPL matches for `team` in chronological order."""
        if n_games < 1:
            raise ValueError(f"n_games must be >= 1, got {n_games}.")

        team_code = normalize_soccer_club(team)
        matches_path = self.settings.paths.silver_dir / "soccer" / "matches.parquet"
        if not matches_path.exists():
            raise FileNotFoundError(
                f"Silver matches table not found at '{matches_path}'. "
                "Run 'sports-ml ingest --sport soccer' first."
            )

        matches_df = pl.read_parquet(matches_path).sort(
            ["match_date", "start_time_utc", "match_id"]
        )
        team_matches_df = matches_df.filter(
            (pl.col("home_team") == team_code) | (pl.col("away_team") == team_code)
        )
        if team_matches_df.is_empty():
            raise ValueError(f"No scheduled matches found in Silver table for team '{team_code}'.")

        if from_date is not None:
            on_or_after = team_matches_df.filter(pl.col("match_date") >= from_date)
            if on_or_after.is_empty():
                raise ValueError(
                    f"No scheduled matches found for team '{team_code}' on or after "
                    f"{from_date.isoformat()}."
                )
            target_df = on_or_after.head(n_games)
        else:
            today_utc = datetime.now(tz=UTC).date()
            fut_matches = team_matches_df.filter(pl.col("game_state") != "FINAL")
            fut_upcoming = fut_matches.filter(pl.col("match_date") >= today_utc)
            if not fut_upcoming.is_empty():
                target_df = fut_upcoming.head(n_games)
            elif not fut_matches.is_empty():
                target_df = fut_matches.head(n_games)
            else:
                target_df = team_matches_df.tail(n_games)

        gold_bundle = build_and_persist_soccer_gold_features(self.settings)
        results: list[SoccerMatchupPrediction] = []
        for row in target_df.iter_rows(named=True):
            results.append(
                self.predict_matchup(
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    match_date=row["match_date"],
                    include_players=include_players,
                    offline=offline,
                    persist_ledger=persist_ledger,
                    _gold_bundle=gold_bundle,
                )
            )
        return results


def score_soccer_prediction_ledger(settings: Settings) -> MulticlassBenchmarkReport:
    """Join `data/predictions/soccer_predictions.parquet` against completed `FINAL` EPL matches."""
    ledger_path = settings.paths.predictions_dir / "soccer_predictions.parquet"
    matches_path = settings.paths.silver_dir / "soccer" / "matches.parquet"

    if not ledger_path.exists():
        raise FileNotFoundError(f"No soccer prediction ledger found at '{ledger_path}'.")
    if not matches_path.exists():
        raise FileNotFoundError(f"No Silver matches table found at '{matches_path}'.")

    ledger_df = pl.read_parquet(ledger_path)
    matches_df = pl.read_parquet(matches_path).filter(
        (pl.col("game_state") == "FINAL") & pl.col("outcome_1x2").is_not_null()
    )
    sel_cols = ["match_date", "home_team", "away_team", "outcome_1x2"]
    for m_col in (
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
        "market_home_decimal_odds",
        "market_draw_decimal_odds",
        "market_away_decimal_odds",
    ):
        if m_col in matches_df.columns:
            sel_cols.append(m_col)

    joined = ledger_df.join(
        matches_df.select(sel_cols),
        on=["match_date", "home_team", "away_team"],
        how="inner",
    )
    if joined.is_empty():
        raise ValueError("No completed FINAL matches matched the soccer prediction ledger yet.")

    for m_col in (
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
        "market_home_decimal_odds",
        "market_draw_decimal_odds",
        "market_away_decimal_odds",
    ):
        right_col = f"{m_col}_right"
        if right_col in joined.columns:
            joined = joined.with_columns(
                pl.coalesce([pl.col(m_col), pl.col(right_col)]).alias(m_col)
            ).drop(right_col)

    model_probs = np.column_stack(
        [
            joined["home_win_prob"].to_numpy(),
            joined["draw_prob"].to_numpy(),
            joined["away_win_prob"].to_numpy(),
        ]
    )
    y_true = np.asarray(joined["outcome_1x2"].cast(pl.Int64).to_numpy(), dtype=np.int64)

    market_probs = (
        np.column_stack(
            [
                joined["market_prob_home"].cast(pl.Float64).to_numpy(),
                joined["market_prob_draw"].cast(pl.Float64).to_numpy(),
                joined["market_prob_away"].cast(pl.Float64).to_numpy(),
            ]
        )
        if {"market_prob_home", "market_prob_draw", "market_prob_away"}.issubset(joined.columns)
        else None
    )
    market_odds = (
        np.column_stack(
            [
                joined["market_home_decimal_odds"].cast(pl.Float64).to_numpy(),
                joined["market_draw_decimal_odds"].cast(pl.Float64).to_numpy(),
                joined["market_away_decimal_odds"].cast(pl.Float64).to_numpy(),
            ]
        )
        if {
            "market_home_decimal_odds",
            "market_draw_decimal_odds",
            "market_away_decimal_odds",
        }.issubset(joined.columns)
        else None
    )
    return compare_multiclass_against_baselines(
        model_probs,
        y_true,
        market_probs=market_probs,
        market_decimal_odds=market_odds,
    )
