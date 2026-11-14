"""Unified two-stage NFL matchup predictor (`football/models/predictor.py`).

`FootballMatchupPredictor` exposes `predict_matchup(season, week, away, home, ...)`,
`predict_week`, `predict_date`, and `predict_team_next_games`. Stage 1 gives calibrated
`P(Home)` / `P(Away)` plus expected scores; the optional Stage 2 layer projects QB / RB / WR / TE
stat lines reconciled to the Stage 1 team totals. Pre-game confidence warnings
(`CONFIRMED_QB`, `PROBABLE_QB`, `QB_UNCERTAIN`, `BACKUP_QB`, `EARLY_SEASON_SHRINKAGE`,
`DEBUT_OR_NEW_QB_PRIOR`, `NEUTRAL_OR_INTERNATIONAL_SITE`, `KEY_OFFENSIVE_PLAYER_OUT`,
`STAGE1_ONLY_NO_PLAYER_DATA`) are surfaced, and every prediction is appended to
`data/predictions/football_predictions.parquet` (player lines to
`data/predictions/football_player_predictions.parquet`) for later live scoring.

Predictions made for games that are already FINAL are retrospective: Stage 2 rows then use the
team's recent contributors (never the realized participants), but ratings and rolling state are
whatever the Gold builder recorded strictly pre-game.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from sports_ml.config import Settings, get_settings
from sports_ml.core.evaluation import BenchmarkComparisonReport, compare_against_baselines
from sports_ml.core.registry import ModelMetadata, ModelRegistry
from sports_ml.sports.football.constants import normalize_football_team
from sports_ml.sports.football.features.environment import build_environment_features
from sports_ml.sports.football.features.pipeline import (
    build_football_stage1_features,
    load_football_game_weather,
    strip_forbidden_columns,
)
from sports_ml.sports.football.features.player import build_football_stage2_features
from sports_ml.sports.football.ingestion.lineups import (
    FootballStartingQBResolver,
    ResolvedStartingQB,
)
from sports_ml.sports.football.models.market_eval import (
    FootballMarketEvaluation,
    evaluate_football_market,
)
from sports_ml.sports.football.models.stage1_game import Stage1FootballGameModel
from sports_ml.sports.football.models.stage2_player import (
    PROJECTION_COLUMNS,
    Stage2FootballPlayerModel,
)

TEAM_LEDGER_NAME: Final[str] = "football_predictions.parquet"
PLAYER_LEDGER_NAME: Final[str] = "football_player_predictions.parquet"
CONFIRMED_SOURCES: Final[frozenset[str]] = frozenset({"confirmed_starter", "depth_chart+inactives"})
PROBABLE_QB_MIN_P: Final[float] = 0.80
EARLY_SEASON_GAMES: Final[int] = 4
OUT_P_ACTIVE: Final[float] = 0.5
KEY_TARGET_SHARE: Final[float] = 0.12
KEY_CARRY_SHARE: Final[float] = 0.30
KEY_PASS_SHARE: Final[float] = 0.50

_PLAYER_LEDGER_KEYS: Final[tuple[str, ...]] = (
    "game_id",
    "team",
    "player_id",
    "player_name",
    "position",
    "p_active",
    *PROJECTION_COLUMNS,
    "prob_anytime_td",
)


class FootballQBCard(BaseModel):
    """Starting-QB summary for one side of a matchup."""

    model_config = ConfigDict(frozen=True)

    team: str
    qb_id: int | None = None
    qb_name: str | None = None
    status: str = "QB_UNCERTAIN"
    p_starts: float | None = None
    source: str = "none"
    is_backup: bool = False
    is_debut: bool = False
    qb_value: float | None = None


class FootballPlayerMatchupProjections(BaseModel):
    """Stage 2 projections for both teams (QB / RB / WR / TE rows)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    home: pl.DataFrame
    away: pl.DataFrame


class FootballMatchupPrediction(BaseModel):
    """Structured Stage 1 (2-way win probability + expected score) and Stage 2 prediction."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    game_id: int
    season: int
    week: int
    season_type: str
    game_date: date
    start_time_utc: datetime
    game_state: str
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    expected_home_score: float
    expected_away_score: float
    expected_margin: float
    expected_total: float
    home_qb: FootballQBCard
    away_qb: FootballQBCard
    warnings: list[str] = Field(default_factory=list)
    player_projections: FootballPlayerMatchupProjections | None = None
    # Evaluation-only market context at prediction time (never a model input)
    market_spread_home: float | None = None
    market_total_points: float | None = None
    market_prob_home: float | None = None


@dataclass
class _SilverBundle:
    games: pl.DataFrame
    team_logs: pl.DataFrame
    qb_logs: pl.DataFrame
    player_logs: pl.DataFrame | None
    availability: pl.DataFrame | None
    draft_picks: pl.DataFrame | None
    names: dict[int, str]
    incumbents: dict[str, int]
    snap_logs: pl.DataFrame | None
    weather: pl.DataFrame
    micro_logs: pl.DataFrame | None
    player_micro: pl.DataFrame | None


def _read_optional(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class FootballMatchupPredictor:
    """Two-stage inference service for NFL matchups."""

    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        stage1_model: Stage1FootballGameModel,
        stage2_model: Stage2FootballPlayerModel | None = None,
        settings: Settings | None = None,
        as_of_time_utc: datetime | None = None,
    ) -> None:
        self.metadata = metadata
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.settings = settings or get_settings()
        self.as_of_time_utc = as_of_time_utc
        self._silver: _SilverBundle | None = None
        self._gold: pl.DataFrame | None = None
        self._resolved: dict[tuple[int, str], ResolvedStartingQB] = {}

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        *,
        settings: Settings | None = None,
        as_of_time_utc: datetime | None = None,
    ) -> FootballMatchupPredictor:
        """Load a fingerprinted football bundle from `artifact_dir`."""
        bundle = ModelRegistry.load_bundle(artifact_dir)
        resolved_settings = settings or get_settings(root_dir=bundle.bundle_dir.parents[2])
        return cls(
            metadata=bundle.metadata,
            stage1_model=bundle.stage1_model,
            stage2_model=bundle.stage2_model,
            settings=resolved_settings,
            as_of_time_utc=as_of_time_utc,
        )

    # ------------------------------------------------------------------ data

    def _load_silver(self) -> _SilverBundle:
        if self._silver is not None:
            return self._silver
        silver = self.settings.paths.silver_dir / "football"
        games_path = silver / "games.parquet"
        if not games_path.exists():
            raise FileNotFoundError(
                f"No Silver football games found at '{games_path}'. "
                "Run 'sports-ml ingest --sport football --league nfl' first."
            )
        games = pl.read_parquet(games_path)
        team_logs = _read_optional(silver / "team_game_logs.parquet")
        qb_logs = _read_optional(silver / "qb_game_logs.parquet")
        if team_logs is None or qb_logs is None:
            raise FileNotFoundError("Silver team/QB game logs are missing; re-run ingestion.")
        player_logs = _read_optional(silver / "player_game_logs.parquet")
        names: dict[int, str] = {}
        if player_logs is not None and not player_logs.is_empty():
            for r in (
                player_logs.filter(pl.col("player_id").is_not_null())
                .select("player_id", "player_name")
                .unique(subset=["player_id"], keep="last")
                .iter_rows(named=True)
            ):
                names[int(r["player_id"])] = str(r["player_name"] or "")
        latest = (
            qb_logs.join(
                games.filter(pl.col("game_state") == "FINAL").select("game_id", "start_time_utc"),
                on="game_id",
            )
            .sort("start_time_utc", "dropbacks")
            .group_by("team", maintain_order=True)
            .last()
        )
        incumbents = {str(r["team"]): int(r["qb_id"]) for r in latest.iter_rows(named=True)}
        draft_path = self.settings.paths.raw_dir / "football" / "nflverse" / "draft_picks.parquet"
        self._silver = _SilverBundle(
            games=games,
            team_logs=team_logs,
            qb_logs=qb_logs,
            player_logs=player_logs,
            availability=_read_optional(silver / "team_week_availability.parquet"),
            draft_picks=_read_optional(draft_path),
            names=names,
            incumbents=incumbents,
            snap_logs=_read_optional(silver / "snap_game_logs.parquet"),
            weather=load_football_game_weather(self.settings, games),
            micro_logs=_read_optional(silver / "team_pbp_micro_logs.parquet"),
            player_micro=_read_optional(silver / "player_pbp_micro_logs.parquet"),
        )
        return self._silver

    def _resolve_starters(self, silver: _SilverBundle) -> pl.DataFrame | None:
        """Tier 1 -> Tier 2 starter for every not-yet-final team-game (injected into Gold)."""
        if silver.availability is None or silver.availability.is_empty():
            return None
        resolver = FootballStartingQBResolver(silver.availability, silver.games)
        as_of = self.as_of_time_utc or _utc_now()
        rows: list[dict[str, object]] = []
        pending = silver.games.filter(pl.col("game_state") != "FINAL")
        for g in pending.iter_rows(named=True):
            gid = int(g["game_id"])
            for team in (str(g["home_team"]), str(g["away_team"])):
                res = resolver.resolve_tier1(gid, team, as_of)
                self._resolved[(gid, team)] = res
                if res.starter_id is not None:
                    rows.append({"game_id": gid, "team": team, "qb_id": res.starter_id})
        if not rows:
            return None
        return pl.DataFrame(rows, schema={"game_id": pl.Int64, "team": pl.Utf8, "qb_id": pl.Int64})

    def gold_features(self) -> pl.DataFrame:
        """Stage 1 Gold matrix rebuilt from Silver (with resolver starters for upcoming games)."""
        if self._gold is not None:
            return self._gold
        silver = self._load_silver()
        self._gold = build_football_stage1_features(
            silver.games,
            silver.team_logs,
            silver.qb_logs,
            player_logs=silver.player_logs,
            availability=silver.availability,
            draft_picks=silver.draft_picks,
            starting_qbs=self._resolve_starters(silver),
            snap_logs=silver.snap_logs,
            weather=silver.weather,
            as_of_time_utc=self.as_of_time_utc,
            micro_logs=silver.micro_logs,
        )
        ModelRegistry.validate_features(self._gold, self.metadata, stage=1)
        return self._gold

    # ------------------------------------------------------------- QB and flags

    def _qb_card(self, row: dict[str, object], side: str, silver: _SilverBundle) -> FootballQBCard:
        team = str(row[f"{side}_team"])
        gid = int(row["game_id"])  # type: ignore[call-overload]
        final = str(row["game_state"]).upper() == "FINAL"
        res = self._resolved.get((gid, team))
        game = silver.games.filter(pl.col("game_id") == gid)
        qb_id: int | None = res.starter_id if res is not None else None
        if qb_id is None and not game.is_empty():
            raw = game[f"{side}_qb_id"][0]
            qb_id = int(raw) if raw is not None else None
        if qb_id is None:
            qb_id = silver.incumbents.get(team)
        p_starts: float | None = None
        if res is not None:
            for cand in res.candidates:
                if cand.qb_id == qb_id:
                    p_starts = cand.p_starts
        imputed = bool(row.get(f"{side}_qb_is_imputed"))
        if final:
            status, p_starts = "CONFIRMED_QB", 1.0
        elif res is not None and res.source in CONFIRMED_SOURCES:
            status = "CONFIRMED_QB"
        elif res is not None and res.starter_id is not None:
            ok = (p_starts or 0.0) >= PROBABLE_QB_MIN_P
            status = "PROBABLE_QB" if ok else "QB_UNCERTAIN"
        else:
            status = "QB_UNCERTAIN" if imputed else "PROBABLE_QB"
        qb_value_raw = row.get(f"{side}_qb_value")
        return FootballQBCard(
            team=team,
            qb_id=qb_id,
            qb_name=silver.names.get(qb_id) if qb_id is not None else None,
            status=status,
            p_starts=p_starts,
            source=res.source if res is not None else ("actual" if final else "incumbent"),
            is_backup=bool(row.get(f"{side}_qb_is_backup")),
            is_debut=bool(row.get(f"{side}_qb_is_rookie_or_debut"))
            or int(row.get(f"{side}_qb_games_with_team") or 0) == 0,  # type: ignore[call-overload]
            qb_value=None if qb_value_raw is None else float(qb_value_raw),  # type: ignore[arg-type]
        )

    @staticmethod
    def _warnings(
        row: dict[str, object],
        cards: tuple[FootballQBCard, FootballQBCard],
        *,
        include_players: bool,
        player_rows_available: bool,
        key_player_out: bool,
    ) -> list[str]:
        out: list[str] = []
        statuses = {c.status for c in cards}
        for code in ("QB_UNCERTAIN", "PROBABLE_QB", "CONFIRMED_QB"):
            if code in statuses:
                out.append(code)
                break
        if any(c.is_backup for c in cards):
            out.append("BACKUP_QB")
        early = min(
            int(row.get("home_games_played_pre") or 0),  # type: ignore[call-overload]
            int(row.get("away_games_played_pre") or 0),  # type: ignore[call-overload]
        )
        if early < EARLY_SEASON_GAMES:
            out.append("EARLY_SEASON_SHRINKAGE")
        if any(c.is_debut for c in cards):
            out.append("DEBUT_OR_NEW_QB_PRIOR")
        if bool(row.get("is_neutral_site")) or bool(row.get("is_international")):
            out.append("NEUTRAL_OR_INTERNATIONAL_SITE")
        if key_player_out:
            out.append("KEY_OFFENSIVE_PLAYER_OUT")
        if include_players and not player_rows_available:
            out.append("STAGE1_ONLY_NO_PLAYER_DATA")
        return out

    # ------------------------------------------------------------------ Stage 2

    def _stage2_projection(
        self, gold: pl.DataFrame, preds: pl.DataFrame, game_ids: list[int]
    ) -> tuple[pl.DataFrame | None, set[tuple[int, str]]]:
        silver = self._load_silver()
        model = self.stage2_model
        if (
            model is None
            or not model.is_fitted
            or silver.player_logs is None
            or silver.player_logs.is_empty()
        ):
            return None, set()
        # Dropping the target games' realized participants makes them behave as upcoming games.
        logs = silver.player_logs.filter(~pl.col("game_id").is_in(game_ids))
        rows = build_football_stage2_features(
            silver.games,
            logs,
            silver.team_logs,
            availability=silver.availability,
            draft_picks=silver.draft_picks,
            stage1_df=gold,
            as_of_time_utc=self.as_of_time_utc,
            starting_qbs=self._starting_qb_frame(game_ids),
            player_micro=silver.player_micro,
            environment=build_environment_features(
                strip_forbidden_columns(silver.games), silver.weather
            ),
        ).filter(pl.col("game_id").is_in(game_ids))
        if rows.is_empty():
            return None, set()

        key_out = {
            (int(r["game_id"]), str(r["team"]))
            for r in rows.filter(
                (pl.col("p_active") < OUT_P_ACTIVE)
                & (
                    (pl.col("target_share") >= KEY_TARGET_SHARE)
                    | (pl.col("carry_share") >= KEY_CARRY_SHARE)
                    | ((pl.col("position") == "QB") & (pl.col("pass_att_share") >= KEY_PASS_SHARE))
                )
            ).iter_rows(named=True)
        }
        rows = self._apply_qb_start_probabilities(rows)
        proj = model.predict(rows, preds).with_columns(
            pl.when(pl.col("player_name").fill_null("") == "")
            .then(
                pl.col("player_id").map_elements(
                    lambda pid: silver.names.get(int(pid)) or f"Player {int(pid)}",
                    return_dtype=pl.Utf8,
                )
            )
            .otherwise(pl.col("player_name"))
            .alias("player_name")
        )
        return proj, key_out

    def _starting_qb_frame(self, game_ids: list[int]) -> pl.DataFrame | None:
        wanted = set(game_ids)
        rows = [
            {"game_id": gid, "team": team, "qb_id": cand.qb_id}
            for (gid, team), res in self._resolved.items()
            if gid in wanted
            for cand in res.candidates
        ]
        if not rows:
            return None
        return pl.DataFrame(rows, schema={"game_id": pl.Int64, "team": pl.Utf8, "qb_id": pl.Int64})

    def _apply_qb_start_probabilities(self, rows: pl.DataFrame) -> pl.DataFrame:
        """Scale each QB row's `p_active` by the resolver's `P(starts)` for that team-game."""
        probs: dict[tuple[int, str, int], float] = {}
        for (gid, team), res in self._resolved.items():
            for cand in res.candidates:
                probs[(gid, team, cand.qb_id)] = cand.p_starts
        if not probs:
            return rows
        qb_ids_in_frame: dict[tuple[int, str], set[int]] = {}
        for r in rows.filter(pl.col("position") == "QB").iter_rows(named=True):
            qb_ids_in_frame.setdefault((int(r["game_id"]), str(r["team"])), set()).add(
                int(r["player_id"])
            )
        gid_col, team_col, pid_col, pos_col, act_col = (
            rows["game_id"].to_list(),
            rows["team"].to_list(),
            rows["player_id"].to_list(),
            rows["position"].to_list(),
            rows["p_active"].to_list(),
        )
        adjusted = list(act_col)
        for i, (gid, team, pid, pos) in enumerate(
            zip(gid_col, team_col, pid_col, pos_col, strict=True)
        ):
            if pos != "QB":
                continue
            known = qb_ids_in_frame.get((int(gid), str(team)), set())
            if not any((int(gid), str(team), q) in probs for q in known):
                continue
            adjusted[i] = float(act_col[i]) * probs.get((int(gid), str(team), int(pid)), 0.0)
        return rows.with_columns(pl.Series("p_active", adjusted, dtype=pl.Float64))

    # --------------------------------------------------------------- prediction

    def predict_games(
        self,
        game_ids: Iterable[int],
        *,
        include_players: bool = False,
        persist_ledger: bool = True,
    ) -> list[FootballMatchupPrediction]:
        """Predict the listed Silver `game_id`s (Stage 1 and optional Stage 2)."""
        ids = sorted({int(g) for g in game_ids})
        if not ids:
            return []
        silver = self._load_silver()
        gold = self.gold_features()
        target = gold.filter(pl.col("game_id").is_in(ids))
        if target.is_empty():
            return []
        preds = self.stage1_model.predict(target)

        proj: pl.DataFrame | None = None
        key_out: set[tuple[int, str]] = set()
        if include_players:
            proj, key_out = self._stage2_projection(gold, preds, ids)

        market_by_game = {
            int(r["game_id"]): r
            for r in silver.games.select(
                "game_id",
                *[
                    c
                    for c in ("market_spread_home", "market_total_points", "market_prob_home")
                    if c in silver.games.columns
                ],
            ).iter_rows(named=True)
        }
        game_dates = {
            int(r["game_id"]): r["game_date"]
            for r in silver.games.select("game_id", "game_date").iter_rows(named=True)
        }
        merged = target.join(
            preds.select(
                "game_id",
                "home_win_prob",
                "away_win_prob",
                "expected_home_score",
                "expected_away_score",
                "expected_margin",
                "expected_total",
            ),
            on="game_id",
        ).sort("start_time_utc", "event_seq", "game_id")

        results: list[FootballMatchupPrediction] = []
        for row in merged.iter_rows(named=True):
            gid = int(row["game_id"])
            home, away = str(row["home_team"]), str(row["away_team"])
            cards = (self._qb_card(row, "home", silver), self._qb_card(row, "away", silver))
            players: FootballPlayerMatchupProjections | None = None
            if proj is not None:
                game_proj = proj.filter(pl.col("game_id") == gid)
                players = FootballPlayerMatchupProjections(
                    home=game_proj.filter(pl.col("team") == home),
                    away=game_proj.filter(pl.col("team") == away),
                )
            has_rows = players is not None and (not players.home.is_empty())
            warnings = self._warnings(
                row,
                cards,
                include_players=include_players,
                player_rows_available=has_rows,
                key_player_out=bool({(gid, home), (gid, away)} & key_out),
            )
            results.append(
                FootballMatchupPrediction(
                    game_id=gid,
                    season=int(row["season"]),
                    week=int(row["week"]),
                    season_type=str(row["season_type"]),
                    game_date=game_dates[gid],
                    start_time_utc=row["start_time_utc"],
                    game_state=str(row["game_state"]),
                    home_team=home,
                    away_team=away,
                    home_win_prob=float(row["home_win_prob"]),
                    away_win_prob=float(row["away_win_prob"]),
                    expected_home_score=float(row["expected_home_score"]),
                    expected_away_score=float(row["expected_away_score"]),
                    expected_margin=float(row["expected_margin"]),
                    expected_total=float(row["expected_total"]),
                    home_qb=cards[0],
                    away_qb=cards[1],
                    warnings=warnings,
                    player_projections=players,
                    market_spread_home=market_by_game.get(gid, {}).get("market_spread_home"),
                    market_total_points=market_by_game.get(gid, {}).get("market_total_points"),
                    market_prob_home=market_by_game.get(gid, {}).get("market_prob_home"),
                )
            )
        if persist_ledger:
            self._append_to_prediction_ledger(results)
        return results

    def _append_to_prediction_ledger(self, predictions: list[FootballMatchupPrediction]) -> None:
        if not predictions:
            return
        pred_dir = self.settings.paths.predictions_dir
        pred_dir.mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        team_rows = pl.DataFrame(
            {
                "predicted_at_utc": [now] * len(predictions),
                "game_id": [p.game_id for p in predictions],
                "season": [p.season for p in predictions],
                "week": [p.week for p in predictions],
                "season_type": [p.season_type for p in predictions],
                "game_date": [p.game_date for p in predictions],
                "home_team": [p.home_team for p in predictions],
                "away_team": [p.away_team for p in predictions],
                "model_version": [self.metadata.model_version] * len(predictions),
                "home_win_prob": [p.home_win_prob for p in predictions],
                "away_win_prob": [p.away_win_prob for p in predictions],
                "expected_home_score": [p.expected_home_score for p in predictions],
                "expected_away_score": [p.expected_away_score for p in predictions],
                "expected_margin": [p.expected_margin for p in predictions],
                "expected_total": [p.expected_total for p in predictions],
                "home_qb_id": [p.home_qb.qb_id for p in predictions],
                "away_qb_id": [p.away_qb.qb_id for p in predictions],
                "home_qb_status": [p.home_qb.status for p in predictions],
                "away_qb_status": [p.away_qb.status for p in predictions],
                "warnings": [",".join(p.warnings) for p in predictions],
                "market_spread_home_at_pred": [p.market_spread_home for p in predictions],
                "market_total_at_pred": [p.market_total_points for p in predictions],
                "market_prob_home_at_pred": [p.market_prob_home for p in predictions],
            },
            schema_overrides={
                "market_spread_home_at_pred": pl.Float64,
                "market_total_at_pred": pl.Float64,
                "market_prob_home_at_pred": pl.Float64,
                "home_qb_id": pl.Int64,
                "away_qb_id": pl.Int64,
                "predicted_at_utc": pl.Datetime("us", "UTC"),
            },
        )
        self._append_parquet(pred_dir / TEAM_LEDGER_NAME, team_rows)

        player_frames = [
            side.select([c for c in _PLAYER_LEDGER_KEYS if c in side.columns])
            for p in predictions
            if p.player_projections is not None
            for side in (p.player_projections.home, p.player_projections.away)
            if not side.is_empty()
        ]
        if player_frames:
            players = pl.concat(player_frames, how="diagonal").with_columns(
                pl.lit(now).cast(pl.Datetime("us", "UTC")).alias("predicted_at_utc"),
                pl.lit(self.metadata.model_version).alias("model_version"),
            )
            self._append_parquet(pred_dir / PLAYER_LEDGER_NAME, players)

    @staticmethod
    def _append_parquet(path: Path, new_rows: pl.DataFrame) -> None:
        if path.exists():
            new_rows = pl.concat([pl.read_parquet(path), new_rows], how="diagonal_relaxed")
        new_rows.write_parquet(path, compression="zstd")

    # ------------------------------------------------------------- entry points

    def _select_games(self, predicate: pl.Expr) -> pl.DataFrame:
        return (
            self._load_silver()
            .games.filter(predicate)
            .sort("start_time_utc", "event_seq", "game_id")
        )

    def upcoming_week(self) -> tuple[int, int] | None:
        """Return `(season, week)` of the earliest not-yet-final scheduled game, if any."""
        pending = self._select_games(pl.col("game_state") != "FINAL")
        if pending.is_empty():
            return None
        return int(pending["season"][0]), int(pending["week"][0])

    def predict_matchup(
        self,
        season: int,
        week: int,
        away_team: str,
        home_team: str,
        *,
        include_players: bool = False,
        persist_ledger: bool = True,
    ) -> FootballMatchupPrediction:
        """Predict one `AWAY@HOME` game in `(season, week)`."""
        away, home = normalize_football_team(away_team), normalize_football_team(home_team)
        games = self._select_games(
            (pl.col("season") == season)
            & (pl.col("week") == week)
            & (pl.col("home_team") == home)
            & (pl.col("away_team") == away)
        )
        if games.is_empty():
            raise ValueError(f"No scheduled game {away}@{home} in season {season} week {week}.")
        return self.predict_games(
            [int(games["game_id"][0])],
            include_players=include_players,
            persist_ledger=persist_ledger,
        )[0]

    def predict_week(
        self,
        season: int,
        week: int,
        *,
        include_players: bool = False,
        persist_ledger: bool = True,
        matchup_filter: str | None = None,
    ) -> list[FootballMatchupPrediction]:
        """Predict every game in `(season, week)` (optionally a single `AWAY@HOME`)."""
        if matchup_filter:
            away, home = parse_matchup(matchup_filter)
            return [
                self.predict_matchup(
                    season,
                    week,
                    away,
                    home,
                    include_players=include_players,
                    persist_ledger=persist_ledger,
                )
            ]
        games = self._select_games((pl.col("season") == season) & (pl.col("week") == week))
        return self.predict_games(
            games["game_id"].to_list(),
            include_players=include_players,
            persist_ledger=persist_ledger,
        )

    def predict_date(
        self,
        game_date: date,
        *,
        include_players: bool = False,
        matchup_filter: str | None = None,
        persist_ledger: bool = True,
    ) -> list[FootballMatchupPrediction]:
        """Predict every game on `game_date` (optionally a single `AWAY@HOME`)."""
        games = self._select_games(pl.col("game_date") == game_date)
        if matchup_filter:
            away, home = parse_matchup(matchup_filter)
            games = games.filter((pl.col("home_team") == home) & (pl.col("away_team") == away))
        return self.predict_games(
            games["game_id"].to_list(),
            include_players=include_players,
            persist_ledger=persist_ledger,
        )

    def predict_team_next_games(
        self,
        team: str,
        n_games: int = 1,
        *,
        from_date: date | None = None,
        include_players: bool = False,
        persist_ledger: bool = True,
    ) -> list[FootballMatchupPrediction]:
        """Predict the next `n_games` scheduled games (or most recent, if none upcoming)."""
        norm = normalize_football_team(team)
        mine = self._select_games((pl.col("home_team") == norm) | (pl.col("away_team") == norm))
        if mine.is_empty():
            raise ValueError(f"No Silver football games found for team '{team}'.")
        if from_date is not None:
            chosen = mine.filter(pl.col("game_date") >= from_date)
        else:
            upcoming = mine.filter(pl.col("game_state") != "FINAL")
            chosen = upcoming if not upcoming.is_empty() else mine.tail(max(1, n_games))
        ids = chosen.head(max(1, n_games))["game_id"].to_list()
        return self.predict_games(
            ids, include_players=include_players, persist_ledger=persist_ledger
        )


def parse_matchup(raw: str) -> tuple[str, str]:
    """Parse `AWAY@HOME` into normalized `(away, home)` team codes."""
    if "@" not in raw:
        raise ValueError(f"Invalid matchup format '{raw}'. Expected 'AWAY@HOME' (e.g., KC@BUF).")
    away, home = raw.split("@", maxsplit=1)
    return normalize_football_team(away.strip()), normalize_football_team(home.strip())


@dataclass(frozen=True)
class FootballLedgerScore:
    """Live-ledger scoring result: game-level benchmark report and player projection MAE."""

    report: BenchmarkComparisonReport
    n_games: int
    player_mae: dict[str, float]
    n_player_rows: int
    market: FootballMarketEvaluation | None = None


def score_football_prediction_ledger(settings: Settings) -> FootballLedgerScore:
    """Score the football ledger(s) against completed Silver games / player logs."""
    ledger_path = settings.paths.predictions_dir / TEAM_LEDGER_NAME
    games_path = settings.paths.silver_dir / "football" / "games.parquet"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Prediction ledger not found at '{ledger_path}'.")
    if not games_path.exists():
        raise FileNotFoundError(f"Silver games not found at '{games_path}'.")

    ledger = pl.read_parquet(ledger_path).unique(subset=["game_id"], keep="last")
    completed = pl.read_parquet(games_path).filter(
        (pl.col("game_state") == "FINAL")
        & pl.col("home_score").is_not_null()
        & (pl.col("is_tie").fill_null(0) == 0)
    )
    joined = ledger.join(
        completed.select("game_id", "home_win", "market_prob_home"), on="game_id", how="inner"
    )
    if joined.is_empty():
        raise ValueError("No completed games matched entries in football_predictions.parquet.")
    p_home = np.asarray(joined["home_win_prob"].to_numpy(), dtype=np.float64)
    y = np.asarray(joined["home_win"].cast(pl.Int64).to_numpy(), dtype=np.int64)
    market = joined["market_prob_home"]
    market_probs = (
        np.asarray(market.to_numpy(), dtype=np.float64) if market.null_count() == 0 else None
    )
    report = compare_against_baselines(
        p_home,
        y,
        train_home_win_rate=float(np.mean(y)),
        market_probs=market_probs,
    )
    player_mae, n_players = _score_player_ledger(settings)
    market_eval = evaluate_football_market(ledger, pl.read_parquet(games_path))
    return FootballLedgerScore(
        report=report,
        n_games=joined.height,
        player_mae=player_mae,
        n_player_rows=n_players,
        market=market_eval,
    )


def _score_player_ledger(settings: Settings) -> tuple[dict[str, float], int]:
    path = settings.paths.predictions_dir / PLAYER_LEDGER_NAME
    logs_path = settings.paths.silver_dir / "football" / "player_game_logs.parquet"
    if not path.exists() or not logs_path.exists():
        return {}, 0
    ledger = pl.read_parquet(path).unique(subset=["game_id", "team", "player_id"], keep="last")
    actual = pl.read_parquet(logs_path).select(
        "game_id",
        "team",
        "player_id",
        pl.col("pass_attempts").alias("actual_pass_att"),
        pl.col("pass_yards").alias("actual_pass_yds"),
        pl.col("carries").alias("actual_carries"),
        pl.col("rush_yards").alias("actual_rush_yds"),
        pl.col("targets").alias("actual_targets"),
        pl.col("rec_yards").alias("actual_rec_yds"),
    )
    joined = ledger.join(actual, on=["game_id", "team", "player_id"], how="inner")
    if joined.is_empty():
        return {}, 0
    mae: dict[str, float] = {}
    for pos, proj_col, act_col in (
        ("QB", "proj_pass_att", "actual_pass_att"),
        ("QB", "proj_pass_yds", "actual_pass_yds"),
        ("RB", "proj_carries", "actual_carries"),
        ("RB", "proj_rush_yds", "actual_rush_yds"),
        ("WR", "proj_targets", "actual_targets"),
        ("WR", "proj_rec_yds", "actual_rec_yds"),
        ("TE", "proj_targets", "actual_targets"),
        ("TE", "proj_rec_yds", "actual_rec_yds"),
    ):
        sub = joined.filter(pl.col("position") == pos).drop_nulls([proj_col, act_col])
        if not sub.is_empty():
            diff = sub[proj_col].cast(pl.Float64) - sub[act_col].cast(pl.Float64)
            mae[f"{pos}_{proj_col}"] = float(diff.abs().mean())  # type: ignore[arg-type]
    return mae, joined.height


__all__ = [
    "FootballLedgerScore",
    "FootballMatchupPrediction",
    "FootballMatchupPredictor",
    "FootballPlayerMatchupProjections",
    "FootballQBCard",
    "parse_matchup",
    "score_football_prediction_ledger",
]
