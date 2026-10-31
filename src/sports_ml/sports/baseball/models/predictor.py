"""Unified two-stage MLB & NPB Baseball matchup predictor (`baseball/models/predictor.py`).

Exposes `BaseballMatchupPredictor` supporting `predict_matchup(...)`, `predict_date(...)`,
and `predict_team_next_games(...)` across MLB (`2-way`) and NPB (`3-way 12-inning + DNB`),
surfaces pre-game confidence warnings (`CONFIRMED_SP` vs `PROBABLE_SP` vs
`ROTATION_PROJECTED_SP`, `CONFIRMED_LINEUP_1_9` vs `PLATOON_PROJECTED_LINEUP`,
`CROSS_LEAGUE_TRANSLATION_PRIOR`, `NPB_NEUTRAL_PROVINCIAL_STADIUM`, `STATCAST_FALLBACK`),
appends to `data/predictions/baseball_predictions.parquet`, and scores the live ledger.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sports_ml.config import Settings, get_settings
from sports_ml.core.evaluation import (
    BenchmarkComparisonReport,
    MulticlassBenchmarkReport,
    compare_against_baselines,
    compare_multiclass_against_baselines,
)
from sports_ml.core.ratings import (
    compute_elo_1x2_probabilities,
    compute_elo_win_probability,
)
from sports_ml.core.registry import ModelMetadata, ModelRegistry
from sports_ml.sports.baseball.constants import (
    BaseballLeague,
    get_ballpark_metadata,
    get_league_sabermetric_priors,
    normalize_baseball_team,
    resolve_team_league,
)
from sports_ml.sports.baseball.features.player import (
    build_baseball_stage2_batter_features,
    build_baseball_stage2_pitcher_features,
)
from sports_ml.sports.baseball.features.team import (
    build_baseball_stage1_game_features,
)
from sports_ml.sports.baseball.ingestion.lineups import (
    BaseballPreGameLineupResolver,
)
from sports_ml.sports.baseball.models.stage1_game import Stage1BaseballGameModel
from sports_ml.sports.baseball.models.stage2_player import Stage2BaseballPlayerModel


class BaseballPlayerMatchupProjections(BaseModel):
    """Container for Stage 2 Starting Pitcher (`SP`) and 1-9 Lineup Batter projections."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    home_batters: pl.DataFrame
    away_batters: pl.DataFrame
    starting_pitchers: pl.DataFrame


class BaseballMatchupPrediction(BaseModel):
    """Structured Stage 1 (`2-way MLB` / `3-way NPB`) and Stage 2 baseball prediction."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    game_id: int | None = None
    event_seq: int = 1
    league: BaseballLeague = "MLB"
    game_date: date
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float = 0.0
    away_win_prob: float
    home_dnb_prob: float
    away_dnb_prob: float
    expected_home_runs: float
    expected_away_runs: float
    home_sp_id: int | None = None
    away_sp_id: int | None = None
    home_sp_hand: str = "R"
    away_sp_hand: str = "R"
    home_starter_status: str = "ROTATION_PROJECTED_SP"
    away_starter_status: str = "ROTATION_PROJECTED_SP"
    home_lineup_status: str = "PLATOON_PROJECTED_LINEUP"
    away_lineup_status: str = "PLATOON_PROJECTED_LINEUP"
    latest_silver_game_date: str | None = None
    has_statcast_metrics: bool = True
    warnings: list[str] = Field(default_factory=list)
    player_projections: BaseballPlayerMatchupProjections | None = None


def build_and_persist_baseball_gold_features(
    settings: Settings,
    *,
    league: str = "all",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build and persist Gold Baseball Stage 1 and Stage 2 feature matrices from Silver Parquet."""
    silver_dir = settings.paths.silver_dir / "baseball"
    gold_dir = settings.paths.gold_dir / "baseball"
    gold_dir.mkdir(parents=True, exist_ok=True)

    games_path = silver_dir / "games.parquet"
    teams_path = silver_dir / "team_game_logs.parquet"
    pitchers_path = silver_dir / "pitcher_game_logs.parquet"
    batters_path = silver_dir / "batter_game_logs.parquet"

    if not games_path.exists() or not teams_path.exists():
        raise FileNotFoundError(
            f"Baseball Silver Parquet tables not found in '{silver_dir}'. "
            "Run 'sports-ml ingest --sport baseball' first."
        )

    games_df = pl.read_parquet(games_path)
    team_logs_df = pl.read_parquet(teams_path)
    pitcher_logs_df = pl.read_parquet(pitchers_path) if pitchers_path.exists() else pl.DataFrame()
    batter_logs_df = pl.read_parquet(batters_path) if batters_path.exists() else pl.DataFrame()

    lg_filter = league.strip().upper()
    if lg_filter in {"MLB", "NPB"}:
        games_df = games_df.filter(pl.col("league") == lg_filter)
        team_logs_df = team_logs_df.filter(pl.col("league") == lg_filter)
        if not pitcher_logs_df.is_empty() and "league" in pitcher_logs_df.columns:
            pitcher_logs_df = pitcher_logs_df.filter(pl.col("league") == lg_filter)
        if not batter_logs_df.is_empty() and "league" in batter_logs_df.columns:
            batter_logs_df = batter_logs_df.filter(pl.col("league") == lg_filter)

    stage1_df = build_baseball_stage1_game_features(
        games_df,
        team_logs_df,
        pitcher_logs_df if not pitcher_logs_df.is_empty() else None,
        batter_logs_df if not batter_logs_df.is_empty() else None,
    )
    stage2_batters_df = (
        build_baseball_stage2_batter_features(
            games_df,
            batter_logs_df,
            stage1_df,
        )
        if not batter_logs_df.is_empty()
        else pl.DataFrame()
    )
    stage2_pitchers_df = (
        build_baseball_stage2_pitcher_features(
            games_df,
            pitcher_logs_df,
            stage1_df,
        )
        if not pitcher_logs_df.is_empty()
        else pl.DataFrame()
    )

    stage1_df.write_parquet(gold_dir / "stage1_game_features.parquet", compression="zstd")
    if not stage2_batters_df.is_empty():
        stage2_batters_df.write_parquet(
            gold_dir / "stage2_batter_features.parquet", compression="zstd"
        )
    if not stage2_pitchers_df.is_empty():
        stage2_pitchers_df.write_parquet(
            gold_dir / "stage2_pitcher_features.parquet", compression="zstd"
        )

    return stage1_df, stage2_batters_df, stage2_pitchers_df


class BaseballMatchupPredictor:
    """Unified two-stage inference service for MLB & NPB baseball matchups."""

    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        stage1_model: Stage1BaseballGameModel,
        stage2_model: Stage2BaseballPlayerModel | None = None,
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
    ) -> BaseballMatchupPredictor:
        """Load a fingerprinted Baseball model bundle from `artifact_dir`."""
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
        league: BaseballLeague,
        event_seq: int = 1,
    ) -> pl.DataFrame:
        exact = stage1_df.filter(
            (pl.col("game_date") == game_date)
            & (pl.col("home_team") == home_code)
            & (pl.col("away_team") == away_code)
            & (pl.col("event_seq") == event_seq)
        )
        if not exact.is_empty():
            return exact.tail(1)

        def _latest_val(t_code: str, col_suffix: str, default_val: float) -> float:
            t_rows = stage1_df.filter(
                (pl.col("home_team") == t_code) | (pl.col("away_team") == t_code)
            )
            if t_rows.is_empty():
                return default_val
            last_r = t_rows.row(-1, named=True)
            side = "home" if last_r["home_team"] == t_code else "away"
            if col_suffix == "elo":
                val = last_r.get(f"elo_{side}_post") or last_r.get(f"elo_{side}_pre")
                return float(val) if val is not None else default_val
            val = last_r.get(f"{side}_{col_suffix}")
            return float(val) if val is not None else default_val

        lg_priors = get_league_sabermetric_priors(league)
        park = get_ballpark_metadata(home_code)
        v_rf = float(park.run_factor) if park else 1.0
        v_hr = float(0.5 * (park.hr_factor_lhb + park.hr_factor_rhb)) if park else 1.0
        is_dome = 1.0 if (park and park.is_dome_or_retractable) else 0.0

        elo_h = _latest_val(home_code, "elo", 1500.0)
        elo_a = _latest_val(away_code, "elo", 1500.0)
        hfa = 26.0 if league == "NPB" else 24.0
        elo_diff = elo_h + hfa - elo_a
        if league == "NPB":
            p_h_elo, p_d_elo, p_a_elo = compute_elo_1x2_probabilities(elo_diff, draw_param=0.032)
        else:
            p_h_elo = compute_elo_win_probability(elo_diff)
            p_d_elo = 0.0
            p_a_elo = 1.0 - p_h_elo

        h_sp_xfip = _latest_val(home_code, "sp_xfip_shrunk", lg_priors.lg_sp_xfip)
        a_sp_xfip = _latest_val(away_code, "sp_xfip_shrunk", lg_priors.lg_sp_xfip)
        h_woba = _latest_val(home_code, "lineup_platoon_woba", lg_priors.lg_woba)
        a_woba = _latest_val(away_code, "lineup_platoon_woba", lg_priors.lg_woba)
        h_bp_xfip = _latest_val(home_code, "bp_xfip_shrunk", lg_priors.lg_bp_xfip)
        a_bp_xfip = _latest_val(away_code, "bp_xfip_shrunk", lg_priors.lg_bp_xfip)

        latest_row = stage1_df.tail(1)
        synth_id = int(f"{game_date.year}9999{event_seq}")
        return latest_row.with_columns(
            pl.lit(synth_id, dtype=pl.Int64).alias("game_id"),
            pl.lit(synth_id, dtype=pl.Int64).alias("event_id"),
            pl.lit(event_seq, dtype=pl.Int32).alias("event_seq"),
            pl.lit(league, dtype=pl.Utf8).alias("league"),
            pl.lit(game_date, dtype=pl.Date).alias("game_date"),
            pl.lit(home_code, dtype=pl.Utf8).alias("home_team"),
            pl.lit(away_code, dtype=pl.Utf8).alias("away_team"),
            pl.lit(elo_h, dtype=pl.Float64).alias("elo_home_pre"),
            pl.lit(elo_a, dtype=pl.Float64).alias("elo_away_pre"),
            pl.lit(elo_diff, dtype=pl.Float64).alias("diff_elo_pre"),
            pl.lit(p_h_elo, dtype=pl.Float64).alias("elo_home_win_prob"),
            pl.lit(p_d_elo, dtype=pl.Float64).alias("elo_draw_prob"),
            pl.lit(p_a_elo, dtype=pl.Float64).alias("elo_away_win_prob"),
            pl.lit(h_sp_xfip, dtype=pl.Float64).alias("home_sp_xfip_shrunk"),
            pl.lit(a_sp_xfip, dtype=pl.Float64).alias("away_sp_xfip_shrunk"),
            pl.lit(a_sp_xfip - h_sp_xfip, dtype=pl.Float64).alias("diff_sp_xfip_edge"),
            pl.lit(h_bp_xfip, dtype=pl.Float64).alias("home_bp_xfip_shrunk"),
            pl.lit(a_bp_xfip, dtype=pl.Float64).alias("away_bp_xfip_shrunk"),
            pl.lit(a_bp_xfip - h_bp_xfip, dtype=pl.Float64).alias("diff_bp_xfip_edge"),
            pl.lit(h_woba, dtype=pl.Float64).alias("home_lineup_platoon_woba"),
            pl.lit(a_woba, dtype=pl.Float64).alias("away_lineup_platoon_woba"),
            pl.lit(h_woba - a_woba, dtype=pl.Float64).alias("diff_lineup_platoon_woba"),
            pl.lit(v_rf, dtype=pl.Float64).alias("venue_run_factor"),
            pl.lit(v_hr, dtype=pl.Float64).alias("venue_hr_factor_matchup"),
            pl.lit(is_dome, dtype=pl.Float64).alias("is_dome_flag"),
        )

    def _append_to_prediction_ledger(self, pred: BaseballMatchupPrediction) -> None:
        pred_dir = self.settings.paths.predictions_dir
        pred_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = pred_dir / "baseball_predictions.parquet"

        new_row = pl.DataFrame(
            {
                "predicted_at_utc": [datetime.now(tz=UTC)],
                "game_id": [int(pred.game_id) if pred.game_id is not None else -1],
                "event_seq": [int(pred.event_seq)],
                "league": [pred.league],
                "game_date": [pred.game_date],
                "home_team": [pred.home_team],
                "away_team": [pred.away_team],
                "model_version": [self.metadata.model_version],
                "home_win_prob": [pred.home_win_prob],
                "draw_prob": [pred.draw_prob],
                "away_win_prob": [pred.away_win_prob],
                "home_dnb_prob": [pred.home_dnb_prob],
                "away_dnb_prob": [pred.away_dnb_prob],
                "expected_home_runs": [pred.expected_home_runs],
                "expected_away_runs": [pred.expected_away_runs],
                "home_starter_status": [pred.home_starter_status],
                "away_starter_status": [pred.away_starter_status],
                "home_lineup_status": [pred.home_lineup_status],
                "away_lineup_status": [pred.away_lineup_status],
            }
        )

        if ledger_path.exists():
            existing = pl.read_parquet(ledger_path)
            combined = pl.concat([existing, new_row], how="diagonal")
        else:
            combined = new_row
        combined.write_parquet(ledger_path, compression="zstd")

    def predict_matchup(
        self,
        home_team: str,
        away_team: str,
        game_date: date,
        *,
        league: str | None = None,
        event_seq: int = 1,
        include_players: bool = False,
        offline: bool = False,
        persist_ledger: bool = True,
    ) -> BaseballMatchupPrediction:
        """Predict a single MLB or NPB matchup (Stage 1 + optional Stage 2 player projections)."""
        _ = offline
        home_code = normalize_baseball_team(home_team)
        away_code = normalize_baseball_team(away_team)
        resolved_lg: BaseballLeague = (
            "NPB"
            if (league and league.strip().upper() == "NPB")
            or resolve_team_league(home_code) == "NPB"
            else "MLB"
        )

        stage1_df, stage2_bat_df, stage2_pit_df = build_and_persist_baseball_gold_features(
            self.settings, league="all"
        )
        silver_dir = self.settings.paths.silver_dir / "baseball"
        games_df = (
            pl.read_parquet(silver_dir / "games.parquet")
            if (silver_dir / "games.parquet").exists()
            else pl.DataFrame()
        )
        pitcher_logs_df = (
            pl.read_parquet(silver_dir / "pitcher_game_logs.parquet")
            if (silver_dir / "pitcher_game_logs.parquet").exists()
            else pl.DataFrame()
        )
        batter_logs_df = (
            pl.read_parquet(silver_dir / "batter_game_logs.parquet")
            if (silver_dir / "batter_game_logs.parquet").exists()
            else pl.DataFrame()
        )

        match_row_df = self._synthesize_matchup_row_if_missing(
            stage1_df,
            home_code,
            away_code,
            game_date,
            resolved_lg,
            event_seq=event_seq,
        )

        ModelRegistry.validate_features(
            match_row_df,
            self.metadata,
            stage=1,
            enforce_column_order=False,
        )
        s1_pred_df = self.stage1_model.predict(match_row_df)
        p_row = s1_pred_df.row(0, named=True)

        gid = int(p_row["game_id"]) if p_row.get("game_id") is not None else None
        cutoff_utc = datetime(game_date.year, game_date.month, game_date.day, 23, 59, tzinfo=UTC)

        # Resolve 2-Tier pre-game starters and 1-9 lineups
        h_prob_sp = int(p_row["home_sp_id"]) if p_row.get("home_sp_id") is not None else None
        a_prob_sp = int(p_row["away_sp_id"]) if p_row.get("away_sp_id") is not None else None
        h_prob_hand = str(p_row.get("home_sp_hand") or "R").upper()
        a_prob_hand = str(p_row.get("away_sp_hand") or "R").upper()

        home_resolved = BaseballPreGameLineupResolver.resolve_team_lineup(
            home_code,
            opposing_sp_hand=a_prob_hand,
            probable_sp_id=h_prob_sp,
            probable_sp_hand=h_prob_hand,
            pitcher_logs_df=pitcher_logs_df,
            batter_logs_df=batter_logs_df,
            games_df=games_df,
            before_time_utc=cutoff_utc,
        )
        away_resolved = BaseballPreGameLineupResolver.resolve_team_lineup(
            away_code,
            opposing_sp_hand=home_resolved.starting_pitcher_hand,
            probable_sp_id=a_prob_sp,
            probable_sp_hand=a_prob_hand,
            pitcher_logs_df=pitcher_logs_df,
            batter_logs_df=batter_logs_df,
            games_df=games_df,
            before_time_utc=cutoff_utc,
        )

        warnings: list[str] = []
        if home_resolved.starter_status == "ROTATION_PROJECTED_SP":
            warnings.append(
                f"{home_code}: Starting pitcher projected from rotation cycle "
                f"(SP ID {home_resolved.starting_pitcher_id})."
            )
        if away_resolved.starter_status == "ROTATION_PROJECTED_SP":
            warnings.append(
                f"{away_code}: Starting pitcher projected from rotation cycle "
                f"(SP ID {away_resolved.starting_pitcher_id})."
            )
        if home_resolved.lineup_status != "CONFIRMED_LINEUP_1_9":
            warnings.append(
                f"{home_code}: Batting order 1-9 using {home_resolved.lineup_status} "
                f"vs {away_resolved.starting_pitcher_hand}HP."
            )
        if away_resolved.lineup_status != "CONFIRMED_LINEUP_1_9":
            warnings.append(
                f"{away_code}: Batting order 1-9 using {away_resolved.lineup_status} "
                f"vs {home_resolved.starting_pitcher_hand}HP."
            )
        if float(p_row.get("is_neutral_or_regional_flag") or 0.0) >= 0.5:
            warnings.append(
                f"Venue notice: Neutral/provincial or displaced stadium "
                f"({p_row.get('venue_id', 'REGIONAL')})."
            )
        if (
            float(p_row.get("home_sp_is_cross_league_prior") or 0.0) >= 0.5
            or float(p_row.get("away_sp_is_cross_league_prior") or 0.0) >= 0.5
        ):
            warnings.append(
                "Cross-league translation prior (NPB <-> MLB) active for starting pitcher."
            )

        latest_date_str = (
            str(games_df.filter(pl.col("game_state") == "FINAL")["game_date"].max())[:10]
            if not games_df.is_empty() and "game_state" in games_df.columns
            else None
        )

        player_proj: BaseballPlayerMatchupProjections | None = None
        if include_players:
            if self.stage2_model is None or not self.stage2_model.is_fitted:
                raise ValueError(
                    "Stage 2 player projections requested (--include-players), but the loaded "
                    "model bundle was trained without --include-stage2."
                )

            # Build Stage 2 projections for the active 1-9 batters and starting pitchers
            exp_h_runs = float(p_row["expected_home_runs"])
            exp_a_runs = float(p_row["expected_away_runs"])
            p_h_win = float(p_row["home_win_prob"])
            p_a_win = float(p_row["away_win_prob"])

            def _team_batter_preds(
                t_code: str,
                opp_code: str,
                is_home_flag: bool,
                resolved_lineup_ids: list[int],
                team_exp_runs: float,
                team_win_p: float,
            ) -> pl.DataFrame:
                if stage2_bat_df.is_empty():
                    return pl.DataFrame()
                t_hist = stage2_bat_df.filter(pl.col("team") == t_code)
                if t_hist.is_empty():
                    return pl.DataFrame()
                latest_per_p = t_hist.unique(subset=["player_id"], keep="last")
                if resolved_lineup_ids:
                    sub = latest_per_p.filter(pl.col("player_id").is_in(resolved_lineup_ids))
                    if sub.height < 9:
                        sub = latest_per_p.head(9)
                else:
                    sub = latest_per_p.head(9)

                slots = list(range(1, sub.height + 1))
                staged = sub.with_columns(
                    pl.lit(gid or 99999, dtype=pl.Int64).alias("game_id"),
                    pl.lit(event_seq, dtype=pl.Int32).alias("event_seq"),
                    pl.lit(t_code, dtype=pl.Utf8).alias("team"),
                    pl.lit(opp_code, dtype=pl.Utf8).alias("opponent"),
                    pl.lit(is_home_flag, dtype=pl.Boolean).alias("is_home"),
                    pl.lit(1.0 if is_home_flag else 0.0, dtype=pl.Float64).alias("is_home_flag"),
                    pl.lit(1.0, dtype=pl.Float64).alias("is_starter_flag"),
                    pl.Series("batting_order_slot", slots, dtype=pl.Int8),
                    pl.Series("batting_slot_num", [float(s) for s in slots], dtype=pl.Float64),
                    pl.lit(team_exp_runs, dtype=pl.Float64).alias("stage1_team_expected_runs"),
                    pl.lit(team_win_p, dtype=pl.Float64).alias("stage1_team_win_prob"),
                )
                assert self.stage2_model is not None
                return self.stage2_model.predict_batters(staged).sort("batting_order_slot")

            home_bat_preds = _team_batter_preds(
                home_code,
                away_code,
                True,
                home_resolved.starting_batter_ids,
                exp_h_runs,
                p_h_win,
            )
            away_bat_preds = _team_batter_preds(
                away_code,
                home_code,
                False,
                away_resolved.starting_batter_ids,
                exp_a_runs,
                p_a_win,
            )

            sp_frames: list[pl.DataFrame] = []
            if not stage2_pit_df.is_empty():
                for t_code, opp_code, is_h, sp_id, opp_runs, win_p in (
                    (
                        away_code,
                        home_code,
                        False,
                        away_resolved.starting_pitcher_id,
                        exp_h_runs,
                        p_a_win,
                    ),
                    (
                        home_code,
                        away_code,
                        True,
                        home_resolved.starting_pitcher_id,
                        exp_a_runs,
                        p_h_win,
                    ),
                ):
                    p_sub = stage2_pit_df.filter(pl.col("player_id") == sp_id).tail(1)
                    if p_sub.is_empty():
                        p_sub = stage2_pit_df.filter(pl.col("team") == t_code).tail(1)
                    if not p_sub.is_empty():
                        p_staged = p_sub.with_columns(
                            pl.lit(gid or 99999, dtype=pl.Int64).alias("game_id"),
                            pl.lit(event_seq, dtype=pl.Int32).alias("event_seq"),
                            pl.lit(t_code, dtype=pl.Utf8).alias("team"),
                            pl.lit(opp_code, dtype=pl.Utf8).alias("opponent"),
                            pl.lit(is_h, dtype=pl.Boolean).alias("is_home"),
                            pl.lit(1.0 if is_h else 0.0, dtype=pl.Float64).alias("is_home_flag"),
                            pl.lit(opp_runs, dtype=pl.Float64).alias("stage1_opp_expected_runs"),
                            pl.lit(win_p, dtype=pl.Float64).alias("stage1_team_win_prob"),
                        )
                        sp_frames.append(self.stage2_model.predict_pitchers(p_staged))

            sp_preds_df = pl.concat(sp_frames, how="vertical") if sp_frames else pl.DataFrame()
            player_proj = BaseballPlayerMatchupProjections(
                home_batters=home_bat_preds,
                away_batters=away_bat_preds,
                starting_pitchers=sp_preds_df,
            )

        prediction = BaseballMatchupPrediction(
            game_id=gid,
            event_seq=event_seq,
            league=resolved_lg,
            game_date=game_date,
            home_team=home_code,
            away_team=away_code,
            home_win_prob=float(p_row["home_win_prob"]),
            draw_prob=float(p_row["draw_prob"]),
            away_win_prob=float(p_row["away_win_prob"]),
            home_dnb_prob=float(p_row["home_dnb_prob"]),
            away_dnb_prob=float(p_row["away_dnb_prob"]),
            expected_home_runs=float(p_row["expected_home_runs"]),
            expected_away_runs=float(p_row["expected_away_runs"]),
            home_sp_id=home_resolved.starting_pitcher_id,
            away_sp_id=away_resolved.starting_pitcher_id,
            home_sp_hand=home_resolved.starting_pitcher_hand,
            away_sp_hand=away_resolved.starting_pitcher_hand,
            home_starter_status=home_resolved.starter_status,
            away_starter_status=away_resolved.starter_status,
            home_lineup_status=home_resolved.lineup_status,
            away_lineup_status=away_resolved.lineup_status,
            latest_silver_game_date=latest_date_str,
            has_statcast_metrics=resolved_lg == "MLB",
            warnings=warnings,
            player_projections=player_proj,
        )

        if persist_ledger:
            self._append_to_prediction_ledger(prediction)
        return prediction

    def predict_date(
        self,
        game_date: date,
        *,
        league: str = "all",
        include_players: bool = False,
        matchup_filter: str | None = None,
        event_seq: int | None = None,
        offline: bool = False,
        persist_ledger: bool = True,
    ) -> list[BaseballMatchupPrediction]:
        """Predict all scheduled MLB and/or NPB games on `game_date` (or a specific `AWAY@HOME`)."""
        if matchup_filter:
            if "@" not in matchup_filter:
                raise ValueError(
                    f"Invalid matchup format '{matchup_filter}'. "
                    "Expected 'AWAY@HOME' (e.g., SDP@LAD or YOM@HAN)."
                )
            away_raw, home_raw = matchup_filter.split("@", maxsplit=1)
            return [
                self.predict_matchup(
                    home_team=home_raw,
                    away_team=away_raw,
                    game_date=game_date,
                    league=league if league.lower() != "all" else None,
                    event_seq=event_seq or 1,
                    include_players=include_players,
                    offline=offline,
                    persist_ledger=persist_ledger,
                )
            ]

        silver_games_path = self.settings.paths.silver_dir / "baseball" / "games.parquet"
        if not silver_games_path.exists():
            return []

        games_df = pl.read_parquet(silver_games_path).filter(pl.col("game_date") == game_date)
        lg_up = league.strip().upper()
        if lg_up in {"MLB", "NPB"}:
            games_df = games_df.filter(pl.col("league") == lg_up)
        if event_seq is not None:
            games_df = games_df.filter(pl.col("event_seq") == int(event_seq))

        results: list[BaseballMatchupPrediction] = []
        for r in games_df.sort(["start_time_utc", "event_seq", "game_id"]).iter_rows(named=True):
            results.append(
                self.predict_matchup(
                    home_team=str(r["home_team"]),
                    away_team=str(r["away_team"]),
                    game_date=game_date,
                    league=str(r["league"]),
                    event_seq=int(r.get("event_seq") or 1),
                    include_players=include_players,
                    offline=offline,
                    persist_ledger=persist_ledger,
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
    ) -> list[BaseballMatchupPrediction]:
        """Predict the next `n_games` scheduled games for an MLB or NPB club."""
        norm_team = normalize_baseball_team(team)
        silver_games_path = self.settings.paths.silver_dir / "baseball" / "games.parquet"
        if not silver_games_path.exists():
            raise ValueError(
                "No Silver baseball games found. Run 'sports-ml ingest --sport baseball'."
            )

        games_df = pl.read_parquet(silver_games_path)
        team_games = games_df.filter(
            (pl.col("home_team") == norm_team) | (pl.col("away_team") == norm_team)
        ).sort(["game_date", "start_time_utc", "event_seq", "game_id"])

        if from_date is not None:
            upcoming = team_games.filter(pl.col("game_date") >= from_date)
        else:
            fut = team_games.filter(pl.col("game_state") == "FUT")
            upcoming = fut if not fut.is_empty() else team_games.tail(n_games)

        selected = upcoming.head(max(1, n_games))
        return [
            self.predict_matchup(
                home_team=str(r["home_team"]),
                away_team=str(r["away_team"]),
                game_date=r["game_date"],
                league=str(r["league"]),
                event_seq=int(r.get("event_seq") or 1),
                include_players=include_players,
                offline=offline,
                persist_ledger=persist_ledger,
            )
            for r in selected.iter_rows(named=True)
        ]


def score_baseball_prediction_ledger(
    settings: Settings,
) -> tuple[BenchmarkComparisonReport | None, MulticlassBenchmarkReport | None]:
    """Score `data/predictions/baseball_predictions.parquet` against completed Silver games."""
    ledger_path = settings.paths.predictions_dir / "baseball_predictions.parquet"
    silver_games_path = settings.paths.silver_dir / "baseball" / "games.parquet"

    if not ledger_path.exists():
        raise FileNotFoundError(f"Prediction ledger not found at '{ledger_path}'.")
    if not silver_games_path.exists():
        raise FileNotFoundError(f"Silver games not found at '{silver_games_path}'.")

    ledger_df = pl.read_parquet(ledger_path).unique(
        subset=["game_date", "home_team", "away_team", "event_seq"], keep="last"
    )
    completed_df = pl.read_parquet(silver_games_path).filter(
        (pl.col("game_state") == "FINAL") & (pl.col("home_runs").is_not_null())
    )

    joined = ledger_df.join(
        completed_df.select(
            [
                "game_date",
                "home_team",
                "away_team",
                "event_seq",
                "home_runs",
                "away_runs",
                "outcome_1x2",
                "home_win",
                "market_prob_home",
                "market_prob_draw",
                "market_prob_away",
                "market_home_decimal_odds",
                "market_draw_decimal_odds",
                "market_away_decimal_odds",
            ]
        ),
        on=["game_date", "home_team", "away_team", "event_seq"],
        how="inner",
    )
    if joined.is_empty():
        raise ValueError("No completed games matched entries in baseball_predictions.parquet.")

    mlb_part = joined.filter(pl.col("league") == "MLB")
    npb_part = joined.filter(pl.col("league") == "NPB")

    mlb_rep: BenchmarkComparisonReport | None = None
    if not mlb_part.is_empty():
        p_h = np.asarray(mlb_part["home_win_prob"].to_numpy(), dtype=np.float64)
        y_h = np.asarray(mlb_part["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        mlb_rep = compare_against_baselines(
            p_h,
            y_h,
            train_home_win_rate=float(np.mean(y_h)),
        )

    npb_rep: MulticlassBenchmarkReport | None = None
    if not npb_part.is_empty():
        p_1x2 = np.column_stack(
            [
                npb_part["home_win_prob"].to_numpy(),
                npb_part["draw_prob"].to_numpy(),
                npb_part["away_win_prob"].to_numpy(),
            ]
        )
        y_1x2 = np.asarray(npb_part["outcome_1x2"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        npb_rep = compare_multiclass_against_baselines(p_1x2, y_1x2)

    return mlb_rep, npb_rep
