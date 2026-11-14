"""Football Gold feature pipeline and the hard odds / observed-outcome firewall.

Builds `data/gold/football/stage1_features.parquet`: one row per game (completed and upcoming)
with home-minus-away `diff_*` columns, QB, schedule, and Elo features. The **Hard Odds &
Observed-Outcome Firewall** guarantees that no `market_*`, `obs_weather_*`, or odds/spread/
closing-like column ever enters `STAGE1_FEATURE_COLUMNS` / `STAGE2_FEATURE_COLUMNS`, and the
builder strips such columns from its inputs before any feature code runs.

Missingness is explicit (`*_is_imputed` flags); feature columns contain zero `NaN`/null.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import polars as pl

from sports_ml.config import Settings
from sports_ml.sports.football.adapter import FootballAdapter
from sports_ml.sports.football.features.availability import (
    AVAILABILITY_FEATURE_COLUMNS,
    attach_availability_features,
    build_availability_features,
)
from sports_ml.sports.football.features.elo_hooks import FootballQBEloHook
from sports_ml.sports.football.features.environment import (
    ENVIRONMENT_FEATURE_COLUMNS,
    build_environment_features,
)
from sports_ml.sports.football.features.micro import (
    STAGE1_MICRO_FEATURE_COLUMNS,
    attach_micro_features,
    build_micro_team_features,
)
from sports_ml.sports.football.features.player import build_football_stage2_features
from sports_ml.sports.football.features.player_columns import STAGE2_PLAYER_FEATURE_COLUMNS
from sports_ml.sports.football.features.qb import (
    QB_FEATURE_COLUMNS,
    build_qb_elo_value_maps,
    build_qb_features,
)
from sports_ml.sports.football.features.schedule_context import (
    ELO_COLUMNS,
    build_elo_context,
    build_schedule_context,
)
from sports_ml.sports.football.features.team import (
    TEAM_METRICS,
    TeamFeatureConfig,
    build_roster_continuity,
    build_team_rolling_features,
    metric_sides,
    team_feature_column_names,
)
from sports_ml.sports.football.ingestion.weather import build_game_weather, load_cached_forecasts

FORBIDDEN_FEATURE_PREFIXES: Final[tuple[str, ...]] = ("market_", "obs_weather_")
FORBIDDEN_FEATURE_TOKENS: Final[tuple[str, ...]] = ("odds", "spread", "closing")

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "event_seq",
    "league",
    "season",
    "season_type",
    "week",
    "start_time_utc",
    "home_team",
    "away_team",
    "game_state",
)
LABEL_COLUMNS: Final[tuple[str, ...]] = ("home_score", "away_score", "home_win", "is_tie")

_CORE_VARIANT_METRICS: Final[tuple[str, ...]] = ("epa_per_play",)
_DIFF_LUCK_COLUMNS: Final[tuple[str, ...]] = (
    "adj_off_epa",
    "adj_def_epa",
    "adj_epa_net",
    "turnover_diff_pg",
    "turnover_regression_signal",
    "fumble_luck_pg",
    "close_game_win_pct",
    "pythag_win_pct",
    "pythag_luck",
    "rz_td_rate",
    "roster_continuity",
)
_QB_DIFF_COLUMNS: Final[tuple[str, ...]] = (
    "qb_value",
    "qb_epa_per_dropback",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_air_yards_per_attempt",
    "qb_rush_epa",
    "qb_value_over_baseline",
)
_QB_SIDE_COLUMNS: Final[tuple[str, ...]] = (
    "qb_value",
    "qb_value_over_baseline",
    "qb_games_with_team",
    "qb_is_rookie_or_debut",
    "qb_is_backup",
    "qb_is_imputed",
)
_TEAM_SIDE_COLUMNS: Final[tuple[str, ...]] = (
    "games_played_pre",
    "prior_season_weight",
    "roster_continuity",
)
_SCHEDULE_SIDE_COLUMNS: Final[tuple[str, ...]] = (
    "days_rest",
    "is_short_week",
    "is_post_bye",
    "is_post_mini_bye",
    "prior_game_was_overtime",
    "travel_km",
    "timezone_shift",
    "consecutive_road_games",
    "kickoff_body_clock_hour",
    "days_rest_is_imputed",
)
_SCHEDULE_GAME_COLUMNS: Final[tuple[str, ...]] = (
    "rest_diff",
    "is_thursday_game",
    "is_monday_game",
    "is_international",
    "is_neutral_site",
    "is_division_game",
    "is_conference_game",
    "is_playoff",
    "playoff_round",
)


def is_forbidden_feature_column(column_name: str) -> bool:
    """Return True for `market_*`, `obs_weather_*`, and odds/spread/closing-like names."""
    lower = column_name.strip().lower()
    return lower.startswith(FORBIDDEN_FEATURE_PREFIXES) or any(
        token in lower for token in FORBIDDEN_FEATURE_TOKENS
    )


def filter_forbidden_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Drop any firewall-violating names from a candidate feature list."""
    return tuple(c for c in columns if not is_forbidden_feature_column(c))


def assert_zero_firewall_contamination(columns: Sequence[str]) -> None:
    """Raise `ValueError` if any column violates the odds / observed-outcome firewall."""
    bad = [c for c in columns if is_forbidden_feature_column(c)]
    if bad:
        raise ValueError(
            f"Odds/observed-outcome contamination in football feature columns: {bad}. "
            "market_*, obs_weather_*, odds, spread, and closing columns are evaluation-only."
        )


def strip_forbidden_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Return `df` without any firewall-violating columns."""
    drop = [c for c in df.columns if is_forbidden_feature_column(c)]
    return df.drop(drop) if drop else df


def _stage1_candidate_columns() -> tuple[str, ...]:
    cols: list[str] = []
    for m in TEAM_METRICS:
        for side in metric_sides(m):
            variants = ["std"]
            if m.name in _CORE_VARIANT_METRICS:
                variants += ["roll4", "roll8", "ewma"]
            cols.extend(f"diff_{side}_{m.name}_{v}" for v in variants)
    cols.extend(f"diff_{c}" for c in _DIFF_LUCK_COLUMNS)
    for c in _TEAM_SIDE_COLUMNS:
        cols.extend([f"home_{c}", f"away_{c}"])
    cols.extend(["home_continuity_is_imputed", "away_continuity_is_imputed"])
    cols.extend(f"diff_{c}" for c in _QB_DIFF_COLUMNS)
    for c in _QB_SIDE_COLUMNS:
        cols.extend([f"home_{c}", f"away_{c}"])
    for c in _SCHEDULE_SIDE_COLUMNS:
        cols.extend([f"home_{c}", f"away_{c}"])
    cols.append("diff_travel_km")
    cols.extend(_SCHEDULE_GAME_COLUMNS)
    cols.extend(ELO_COLUMNS)
    cols.extend(ENVIRONMENT_FEATURE_COLUMNS)
    cols.extend(AVAILABILITY_FEATURE_COLUMNS)
    cols.extend(STAGE1_MICRO_FEATURE_COLUMNS)
    seen: dict[str, None] = {}
    for c in cols:
        seen.setdefault(c, None)
    return tuple(seen)


FOOTBALL_STAGE1_CANDIDATE_FEATURE_COLS: Final[tuple[str, ...]] = _stage1_candidate_columns()

FOOTBALL_STAGE2_CANDIDATE_FEATURE_COLS: Final[tuple[str, ...]] = STAGE2_PLAYER_FEATURE_COLUMNS

STAGE1_FEATURE_COLUMNS: Final[tuple[str, ...]] = filter_forbidden_feature_columns(
    FOOTBALL_STAGE1_CANDIDATE_FEATURE_COLS
)
STAGE2_FEATURE_COLUMNS: Final[tuple[str, ...]] = filter_forbidden_feature_columns(
    FOOTBALL_STAGE2_CANDIDATE_FEATURE_COLS
)
assert_zero_firewall_contamination(STAGE1_FEATURE_COLUMNS)
assert_zero_firewall_contamination(STAGE2_FEATURE_COLUMNS)


def _rename_side(df: pl.DataFrame, cols: Sequence[str], prefix: str) -> pl.DataFrame:
    return df.rename({c: f"{prefix}_{c}" for c in cols})


def build_football_stage1_features(
    games_df: pl.DataFrame,
    team_logs: pl.DataFrame,
    qb_logs: pl.DataFrame,
    *,
    player_logs: pl.DataFrame | None = None,
    availability: pl.DataFrame | None = None,
    draft_picks: pl.DataFrame | None = None,
    starting_qbs: pl.DataFrame | None = None,
    head_coach_changed: pl.DataFrame | None = None,
    config: TeamFeatureConfig | None = None,
    snap_logs: pl.DataFrame | None = None,
    weather: pl.DataFrame | None = None,
    as_of_time_utc: datetime | None = None,
    micro_logs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the Stage 1 Gold matrix (identifiers, labels, `STAGE1_FEATURE_COLUMNS`).

    `micro_logs` is Silver `team_pbp_micro_logs`; without it the micro columns resolve to their
    league priors and are flagged `*_is_imputed = 1`.

    `weather` is the per-game frame from `build_game_weather` (forecast or flagged observed
    proxy); without it outdoor weather is neutral and flagged `weather_is_imputed`. `snap_logs`
    and `availability` feed the position-group availability indices as of `as_of_time_utc`.
    """
    games = FootballAdapter().validate_event_schema(strip_forbidden_columns(games_df))
    games = games.drop("event_id", "sport", strict=False)
    games = games.with_columns(pl.col("game_id").cast(pl.Int64))

    continuity = build_roster_continuity(
        games,
        player_logs,
        availability,
        head_coach_changed=head_coach_changed,
        config=config,
    )
    team = build_team_rolling_features(
        games, team_logs, roster_continuity=continuity, config=config
    ).join(
        continuity.select("game_id", "team", "continuity_is_imputed"),
        on=["game_id", "team"],
        how="left",
    )
    qb = build_qb_features(games, qb_logs, draft_picks=draft_picks, starting_qbs=starting_qbs)
    schedule = build_schedule_context(games)
    qb_hook = FootballQBEloHook.from_mappings(*build_qb_elo_value_maps(qb))
    elo = build_elo_context(games, qb_hook=qb_hook, schedule_context=schedule)

    team_cols = [*team_feature_column_names(), "continuity_is_imputed"]
    qb_cols = list(QB_FEATURE_COLUMNS)
    base = games.select(*KEY_COLUMNS, *LABEL_COLUMNS)

    def attach(
        joined: pl.DataFrame, frame: pl.DataFrame, cols: list[str], side_team: str, prefix: str
    ) -> pl.DataFrame:
        side = _rename_side(frame.select("game_id", "team", *cols), cols, prefix).rename(
            {"team": side_team}
        )
        return joined.join(side, on=["game_id", side_team], how="left")

    base_joined = base
    base_joined = attach(base_joined, team, team_cols, "home_team", "home")
    base_joined = attach(base_joined, team, team_cols, "away_team", "away")
    base_joined = attach(base_joined, qb, qb_cols, "home_team", "home")
    base_joined = attach(base_joined, qb, qb_cols, "away_team", "away")
    environment = build_environment_features(games, weather)
    availability_frame = build_availability_features(
        games, snap_logs, availability, as_of_time_utc=as_of_time_utc
    )
    out = (
        base_joined.join(schedule, on="game_id", how="left")
        .join(elo, on="game_id", how="left")
        .join(environment, on="game_id", how="left")
        .sort("start_time_utc", "event_seq", "game_id")
    )
    out = attach_availability_features(out, availability_frame)
    out = attach_micro_features(out, build_micro_team_features(games, micro_logs))

    diff_exprs: list[pl.Expr] = []
    for m in TEAM_METRICS:
        for side in metric_sides(m):
            variants = ["std"]
            if m.name in _CORE_VARIANT_METRICS:
                variants += ["roll4", "roll8", "ewma"]
            for v in variants:
                col = f"{side}_{m.name}_{v}"
                diff_exprs.append(
                    (pl.col(f"home_{col}") - pl.col(f"away_{col}")).alias(f"diff_{col}")
                )
    for c in (*_DIFF_LUCK_COLUMNS, *_QB_DIFF_COLUMNS):
        diff_exprs.append((pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"diff_{c}"))
    diff_exprs.append((pl.col("home_travel_km") - pl.col("away_travel_km")).alias("diff_travel_km"))
    out = out.with_columns(*diff_exprs)

    for c in STAGE1_FEATURE_COLUMNS:
        if out.schema[c] == pl.Boolean:
            out = out.with_columns(pl.col(c).cast(pl.Int8))
    out = out.select(*KEY_COLUMNS, *LABEL_COLUMNS, *STAGE1_FEATURE_COLUMNS)
    assert_feature_matrix_clean(out)
    return out


def assert_feature_matrix_clean(df: pl.DataFrame) -> None:
    """Raise if any Stage 1 feature column contains null or NaN values."""
    for c in STAGE1_FEATURE_COLUMNS:
        nulls = df[c].null_count()
        nans = int(df[c].is_nan().sum()) if df.schema[c].is_float() else 0
        if nulls or nans:
            raise ValueError(f"Feature column '{c}' has {nulls} nulls and {nans} NaNs.")


def load_football_game_weather(settings: Settings, games_df: pl.DataFrame) -> pl.DataFrame:
    """Per-game weather from the cached forecasts (flagged observed proxy where uncovered)."""
    cache = settings.paths.raw_dir / "football" / "weather"
    return build_game_weather(games_df, load_cached_forecasts(cache))


def build_and_persist_football_gold_features(
    settings: Settings,
    *,
    persist: bool = True,
) -> pl.DataFrame:
    """Read Silver football tables, build Stage 1 Gold features, and write the parquet."""
    silver = settings.paths.silver_dir / "football"
    games_path = silver / "games.parquet"
    if not games_path.exists():
        raise FileNotFoundError(
            f"Silver football games not found at {games_path}; run "
            "`sports-ml ingest --sport football --league nfl` first."
        )

    def read(name: str) -> pl.DataFrame | None:
        path = silver / name
        return pl.read_parquet(path) if path.exists() else None

    team_logs = read("team_game_logs.parquet")
    qb_logs = read("qb_game_logs.parquet")
    if team_logs is None or qb_logs is None:
        raise FileNotFoundError("Silver team/QB game logs are missing; re-run football ingestion.")
    draft_path: Path = settings.paths.raw_dir / "football" / "nflverse" / "draft_picks.parquet"
    draft = pl.read_parquet(draft_path) if draft_path.exists() else None

    games = pl.read_parquet(games_path)
    features = build_football_stage1_features(
        games,
        team_logs,
        qb_logs,
        player_logs=read("player_game_logs.parquet"),
        availability=read("team_week_availability.parquet"),
        draft_picks=draft,
        snap_logs=read("snap_game_logs.parquet"),
        weather=load_football_game_weather(settings, games),
        micro_logs=read("team_pbp_micro_logs.parquet"),
    )
    if persist:
        gold = settings.paths.gold_dir / "football"
        gold.mkdir(parents=True, exist_ok=True)
        features.write_parquet(gold / "stage1_features.parquet", compression="zstd")
    return features


def build_and_persist_football_stage2_features(
    settings: Settings,
    *,
    stage1_df: pl.DataFrame | None = None,
    persist: bool = True,
) -> pl.DataFrame:
    """Build the Stage 2 player Gold matrix from Silver (empty frame if no player logs exist)."""
    silver = settings.paths.silver_dir / "football"
    players_path = silver / "player_game_logs.parquet"
    if not (silver / "games.parquet").exists():
        raise FileNotFoundError(
            f"Silver football games not found under {silver}; run "
            "`sports-ml ingest --sport football --league nfl` first."
        )
    games = pl.read_parquet(silver / "games.parquet")
    team_logs = pl.read_parquet(silver / "team_game_logs.parquet")
    player_logs = pl.read_parquet(players_path) if players_path.exists() else None
    avail_path = silver / "team_week_availability.parquet"
    draft_path: Path = settings.paths.raw_dir / "football" / "nflverse" / "draft_picks.parquet"
    if player_logs is None or player_logs.is_empty():
        return pl.DataFrame()
    frame = build_football_stage2_features(
        games,
        player_logs,
        team_logs,
        environment=build_environment_features(
            strip_forbidden_columns(games), load_football_game_weather(settings, games)
        ),
        availability=pl.read_parquet(avail_path) if avail_path.exists() else None,
        draft_picks=pl.read_parquet(draft_path) if draft_path.exists() else None,
        stage1_df=stage1_df,
        player_micro=(
            pl.read_parquet(silver / "player_pbp_micro_logs.parquet")
            if (silver / "player_pbp_micro_logs.parquet").exists()
            else None
        ),
    )
    if persist:
        gold = settings.paths.gold_dir / "football"
        gold.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(gold / "stage2_features.parquet", compression="zstd")
    return frame
