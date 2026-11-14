"""Silver lakehouse reconciler for NFL data (`sports/football/ingestion/reconciler.py`).

Normalizes franchise aliases and reconciles nflverse releases into canonical `zstd` Parquet
tables under `data/silver/football/`:

* `games.parquet`, `team_game_logs.parquet`, `player_game_logs.parquet`
* `qb_game_logs.parquet`, `team_week_availability.parquet`

**Leakage quarantine**: schedule-embedded odds (`spread_line`, `total_line`, moneylines) are
split into the evaluation-only `market_*` namespace and observed weather/roof/surface into
`obs_weather_*` at reconcile time, so no downstream feature builder can read them by accident.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import polars as pl

from sports_ml.core.market import american_to_decimal, devig_probabilities
from sports_ml.sports.football.adapter import gsis_id_to_int
from sports_ml.sports.football.constants import (
    FOOTBALL_STADIUMS,
    MODEL_SEASON_TYPES,
    NFL_TEAMS_32,
    NFLVERSE_STADIUM_ID_TO_VENUE,
    SILVER_FOOTBALL_AVAILABILITY_SCHEMA,
    SILVER_FOOTBALL_GAMES_SCHEMA,
    SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA,
    SILVER_FOOTBALL_PLAYER_MICRO_SCHEMA,
    SILVER_FOOTBALL_QB_LOGS_SCHEMA,
    SILVER_FOOTBALL_SNAP_LOGS_SCHEMA,
    SILVER_FOOTBALL_TEAM_LOGS_SCHEMA,
    SILVER_FOOTBALL_TEAM_MICRO_SCHEMA,
    canonical_game_id_from_nflverse,
    is_neutral_game,
    make_football_game_id,
    normalize_football_team,
)
from sports_ml.sports.football.ingestion.lineups import (
    SOURCE_DEPTH_CHART,
    SOURCE_INACTIVES,
    SOURCE_INJURY_REPORT,
    TIER1_CUTOFF_MINUTES,
    derive_starting_qbs_from_pbp,
)
from sports_ml.sports.football.ingestion.pbp_aggregates import (
    aggregate_filtered_offense,
    aggregate_player_pbp_usage,
)
from sports_ml.sports.football.ingestion.pbp_micro import (
    build_player_pbp_micro_logs,
    build_team_pbp_micro_logs,
)

_EASTERN = ZoneInfo("America/New_York")
DEFAULT_KICKOFF_ET: Final[str] = "13:00"
# Conservative as-of stamps for snapshots that lack a publication timestamp: a later stamp
# can only hide information from a prediction, never leak it.
UNTIMED_INJURY_REPORT_LEAD: Final[timedelta] = timedelta(hours=24)
UNTIMED_DEPTH_CHART_LEAD: Final[timedelta] = timedelta(days=2)

QUARANTINED_SCHEDULE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
        "temp",
        "wind",
        "roof",
        "surface",
    }
)


def assert_games_quarantined(games_df: pl.DataFrame) -> None:
    """Raise if raw odds/weather schedule columns leaked into a Silver `games` frame."""
    leaked = sorted(QUARANTINED_SCHEDULE_COLUMNS.intersection(games_df.columns))
    if leaked:
        raise ValueError(f"Quarantined schedule columns present in Silver games: {leaked}.")


def _cast_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema=schema)
    exprs = [
        pl.col(col).cast(dtype) if col in df.columns else pl.lit(None, dtype=dtype).alias(col)
        for col, dtype in schema.items()
    ]
    return df.select(exprs)


def _gsis_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return gsis_id_to_int(value)
    except (ValueError, TypeError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if number == number else None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _kickoff_utc(gameday: str, gametime: str | None) -> datetime:
    day = date.fromisoformat(str(gameday)[:10])
    hh, mm = (gametime or DEFAULT_KICKOFF_ET).strip().split(":")[:2]
    local = datetime(day.year, day.month, day.day, int(hh), int(mm), tzinfo=_EASTERN)
    return local.astimezone(UTC)


def _resolve_venue_id(
    stadium_id: str | None, home_team: str, is_neutral_location: bool
) -> str | None:
    if stadium_id is not None and stadium_id in NFLVERSE_STADIUM_ID_TO_VENUE:
        return NFLVERSE_STADIUM_ID_TO_VENUE[stadium_id]
    if is_neutral_location:
        return None
    # Legacy venues outside the registry fall back to the host club's current home venue
    # as a same-metro proxy for timezone/coordinates.
    return NFL_TEAMS_32[home_team].primary_venue_id if home_team in NFL_TEAMS_32 else None


def _market_fields(row: dict[str, Any]) -> dict[str, float | None]:
    home_ml = _optional_float(row.get("home_moneyline"))
    away_ml = _optional_float(row.get("away_moneyline"))
    out: dict[str, float | None] = {
        "market_home_decimal_odds": None,
        "market_away_decimal_odds": None,
        "market_prob_home": None,
        "market_prob_away": None,
    }
    if home_ml is not None and away_ml is not None and abs(home_ml) >= 100 and abs(away_ml) >= 100:
        dec_home = float(american_to_decimal(home_ml))
        dec_away = float(american_to_decimal(away_ml))
        probs = devig_probabilities([dec_home, dec_away], method="power")
        out.update(
            market_home_decimal_odds=dec_home,
            market_away_decimal_odds=dec_away,
            market_prob_home=float(probs[0]),
            market_prob_away=float(probs[1]),
        )
    spread = _optional_float(row.get("spread_line"))
    # nflverse `spread_line` is positive when the home team is favored; betting convention
    # (home line) is negative for a home favorite.
    out["market_spread_home"] = -spread if spread is not None else None
    out["market_total_points"] = _optional_float(row.get("total_line"))
    return out


def build_games_frame(
    schedule_df: pl.DataFrame,
    *,
    starting_qbs: dict[tuple[int, str], int] | None = None,
) -> pl.DataFrame:
    """Build canonical Silver `games` rows from an nflverse schedule (`games.csv`) frame.

    `starting_qbs` maps `(game_id, team) -> gsis int` (first-dropback QBs from play-by-play) and
    takes precedence over the schedule's `home_qb_id` / `away_qb_id`.
    """
    rows: list[dict[str, Any]] = []
    for raw in schedule_df.iter_rows(named=True):
        season_type = str(raw.get("game_type") or "").upper()
        if season_type not in MODEL_SEASON_TYPES:
            continue
        season = int(raw["season"])
        week = int(raw["week"])
        home = normalize_football_team(str(raw["home_team"]))
        away = normalize_football_team(str(raw["away_team"]))
        game_id = make_football_game_id(season, week, away, home)
        start_utc = _kickoff_utc(str(raw["gameday"]), _clean_str(raw.get("gametime")))

        neutral_location = str(raw.get("location") or "").strip().lower() == "neutral"
        venue_id = _resolve_venue_id(_clean_str(raw.get("stadium_id")), home, neutral_location)
        venue = FOOTBALL_STADIUMS.get(venue_id) if venue_id is not None else None

        home_score = _optional_int(raw.get("home_score"))
        away_score = _optional_int(raw.get("away_score"))
        completed = home_score is not None and away_score is not None
        home_win: int | None = None
        is_tie: int | None = None
        if home_score is not None and away_score is not None:
            is_tie = int(home_score == away_score)
            home_win = None if is_tie else int(home_score > away_score)
        overtime_raw = _optional_int(raw.get("overtime"))

        home_qb = (starting_qbs or {}).get((game_id, home)) or _gsis_or_none(raw.get("home_qb_id"))
        away_qb = (starting_qbs or {}).get((game_id, away)) or _gsis_or_none(raw.get("away_qb_id"))

        row: dict[str, Any] = {
            "sport": "football",
            "league": "NFL",
            "season": season,
            "season_type": season_type,
            "week": week,
            "game_id": game_id,
            "event_seq": 1,
            "game_date": date.fromisoformat(str(raw["gameday"])[:10]),
            "start_time_utc": start_utc,
            "home_team": home,
            "away_team": away,
            "venue_id": venue_id,
            "venue_timezone": venue.timezone if venue is not None else None,
            "is_neutral_site": neutral_location or is_neutral_game(season_type, venue_id),
            "is_international": bool(venue.is_international) if venue is not None else False,
            "game_state": "FINAL" if completed else "FUT",
            "home_qb_id": home_qb,
            "away_qb_id": away_qb,
            "overtime": bool(overtime_raw) if completed and overtime_raw is not None else None,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": home_win,
            "is_tie": is_tie,
            "obs_weather_temp": _optional_float(raw.get("temp")),
            "obs_weather_wind": _optional_float(raw.get("wind")),
            "obs_weather_roof": _clean_str(raw.get("roof")),
            "obs_weather_surface": _clean_str(raw.get("surface")),
            "nflverse_game_id": _clean_str(raw.get("game_id")),
            **_market_fields(raw),
        }
        rows.append(row)
    if not rows:
        return pl.DataFrame(schema=SILVER_FOOTBALL_GAMES_SCHEMA)
    frame = pl.DataFrame(rows, schema=SILVER_FOOTBALL_GAMES_SCHEMA)
    assert_games_quarantined(frame)
    return frame.sort("start_time_utc", "event_seq", "game_id")


def _team_game_lookup(games_df: pl.DataFrame) -> pl.DataFrame:
    """Long-form `(game_id, team, opponent, is_home, ...)` view of a games frame."""
    base = ["game_id", "event_seq", "game_date", "start_time_utc", "season", "season_type", "week"]
    home = games_df.select(
        *base,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        "venue_id",
        "overtime",
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against"),
    )
    away = games_df.select(
        *base,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        "venue_id",
        "overtime",
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against"),
    )
    return pl.concat([home, away], how="vertical")


def _with_canonical_game_id(df: pl.DataFrame, source_col: str = "game_id") -> pl.DataFrame:
    ids = df[source_col].unique().drop_nulls().to_list()
    mapping = {gid: canonical_game_id_from_nflverse(gid) for gid in ids}
    return df.with_columns(
        pl.col(source_col)
        .replace_strict(mapping, default=None, return_dtype=pl.Int64)
        .alias("_gid")
    ).filter(pl.col("_gid").is_not_null())


def build_team_game_logs(pbp_df: pl.DataFrame, games_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate play-by-play into one Silver row per team-game (raw totals, no filters)."""
    if pbp_df.is_empty() or games_df.is_empty():
        return pl.DataFrame(schema=SILVER_FOOTBALL_TEAM_LOGS_SCHEMA)
    pbp = _with_canonical_game_id(pbp_df).with_columns(
        pl.col("posteam").map_elements(normalize_football_team, return_dtype=pl.Utf8),
        pl.col("penalty_team").map_elements(normalize_football_team, return_dtype=pl.Utf8),
    )

    scrimmage = pbp.filter(
        pl.col("posteam").is_not_null() & pl.col("play_type").is_in(["pass", "run"])
    )
    offense = scrimmage.group_by("_gid", "posteam").agg(
        pl.len().alias("off_plays"),
        (pl.col("qb_dropback") == 1).sum().alias("off_dropbacks"),
        (pl.col("rush") == 1).sum().alias("off_rushes"),
        pl.col("epa").sum().alias("off_epa_total"),
        pl.col("epa").filter(pl.col("pass") == 1).sum().alias("off_pass_epa_total"),
        pl.col("epa").filter(pl.col("rush") == 1).sum().alias("off_rush_epa_total"),
        (pl.col("success") == 1).sum().alias("off_success_plays"),
        pl.col("yards_gained").sum().alias("off_yards"),
        ((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1)).sum().alias("off_turnovers"),
        (pl.col("sack") == 1).sum().alias("off_sacks_taken"),
    )
    penalties = (
        pbp.filter((pl.col("penalty") == 1) & pl.col("penalty_team").is_not_null())
        .group_by("_gid", "penalty_team")
        .agg(pl.len().alias("penalties"), pl.col("penalty_yards").sum().alias("penalty_yards"))
        .rename({"penalty_team": "posteam"})
    )
    drive_seconds = (
        pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("drive").is_not_null()
            & pl.col("drive_time_of_possession").is_not_null()
        )
        .group_by("_gid", "posteam", "drive")
        .agg(pl.col("drive_time_of_possession").first().alias("_top"))
        .with_columns(
            (
                pl.col("_top").str.split(":").list.get(0).cast(pl.Int32, strict=False) * 60
                + pl.col("_top").str.split(":").list.get(1).cast(pl.Int32, strict=False)
            ).alias("_secs")
        )
        .group_by("_gid", "posteam")
        .agg(pl.col("_secs").sum().alias("time_of_possession_s"))
    )
    filtered = aggregate_filtered_offense(pbp)
    stats = (
        offense.join(filtered, on=["_gid", "posteam"], how="left")
        .join(penalties, on=["_gid", "posteam"], how="left")
        .join(drive_seconds, on=["_gid", "posteam"], how="left")
        .rename({"_gid": "game_id", "posteam": "team"})
    )
    opp_cols = {
        "off_plays": "def_plays",
        "off_epa_total": "def_epa_total",
        "off_success_plays": "def_success_plays",
        "off_yards": "def_yards",
        "off_turnovers": "def_takeaways",
        "off_sacks_taken": "def_sacks",
    }

    lookup = _team_game_lookup(games_df)
    joined = lookup.join(stats, on=["game_id", "team"], how="inner")
    opp = stats.select(
        "game_id",
        pl.col("team").alias("opponent"),
        *[pl.col(src).alias(dst) for src, dst in opp_cols.items()],
    )
    out = joined.join(opp, on=["game_id", "opponent"], how="left").with_columns(
        pl.lit("NFL").alias("league"),
        pl.col("penalties").fill_null(0),
        pl.col("penalty_yards").fill_null(0),
    )
    return _cast_to_schema(out, SILVER_FOOTBALL_TEAM_LOGS_SCHEMA).sort(
        "start_time_utc", "event_seq", "game_id", "team"
    )


def build_qb_game_logs(pbp_df: pl.DataFrame, games_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate dropback-level play-by-play into one Silver row per `(game, team, QB)`."""
    if pbp_df.is_empty() or games_df.is_empty():
        return pl.DataFrame(schema=SILVER_FOOTBALL_QB_LOGS_SCHEMA)
    starters = derive_starting_qbs_from_pbp(pbp_df).select(
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("team").map_elements(normalize_football_team, return_dtype=pl.Utf8),
        pl.col("gsis_id").alias("starter_gsis"),
    )
    pbp = _with_canonical_game_id(pbp_df).with_columns(
        pl.col("posteam").map_elements(normalize_football_team, return_dtype=pl.Utf8)
    )
    dropbacks = pbp.filter(
        (pl.col("qb_dropback") == 1) & pl.col("posteam").is_not_null() & pl.col("id").is_not_null()
    )
    qb = dropbacks.group_by("_gid", "posteam", "id").agg(
        pl.len().alias("dropbacks"),
        (pl.col("pass_attempt") == 1).sum().alias("pass_attempts"),
        pl.col("epa").sum().alias("qb_epa_total"),
        pl.col("cpoe").sum().alias("cpoe_total"),
        pl.col("cpoe").count().alias("cpoe_plays"),
        (pl.col("sack") == 1).sum().alias("sacks"),
        pl.col("air_yards").sum().alias("air_yards_total"),
        (pl.col("qb_scramble") == 1).sum().alias("scrambles"),
    )
    qb_rushes = (
        pbp.filter(
            (pl.col("play_type") == "run")
            & (pl.col("qb_dropback").fill_null(0) != 1)
            & pl.col("posteam").is_not_null()
            & pl.col("id").is_not_null()
        )
        .group_by("_gid", "posteam", "id")
        .agg(
            pl.len().alias("rush_attempts"),
            pl.col("epa").sum().alias("rush_epa_total"),
        )
    )
    qb = qb.join(qb_rushes, on=["_gid", "posteam", "id"], how="left").with_columns(
        pl.col("rush_attempts").fill_null(0), pl.col("rush_epa_total").fill_null(0.0)
    )
    starter_ids = _with_canonical_game_id(starters).select(
        pl.col("_gid").alias("game_id"), "team", "starter_gsis"
    )
    qb = (
        qb.rename({"_gid": "game_id", "posteam": "team", "id": "gsis_id"})
        .join(starter_ids, on=["game_id", "team"], how="left")
        .with_columns(
            (pl.col("gsis_id") == pl.col("starter_gsis")).alias("is_starter"),
            pl.col("gsis_id").map_elements(_gsis_or_none, return_dtype=pl.Int64).alias("qb_id"),
        )
        .filter(pl.col("qb_id").is_not_null())
    )
    lookup = _team_game_lookup(games_df).select(
        "game_id",
        "event_seq",
        "game_date",
        "start_time_utc",
        "season",
        "season_type",
        "week",
        "team",
        "opponent",
    )
    out = qb.join(lookup, on=["game_id", "team"], how="inner")
    return _cast_to_schema(out, SILVER_FOOTBALL_QB_LOGS_SCHEMA).sort(
        "start_time_utc", "event_seq", "game_id", "team", "qb_id"
    )


def _first_present(df: pl.DataFrame, *candidates: str) -> pl.Expr:
    for name in candidates:
        if name in df.columns:
            return pl.col(name)
    return pl.lit(None)


def build_player_game_logs(
    player_stats_df: pl.DataFrame,
    games_df: pl.DataFrame,
    *,
    snap_counts_df: pl.DataFrame | None = None,
    players_df: pl.DataFrame | None = None,
    pbp_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build Silver player-game rows from weekly stats (+ snap share and play-level usage)."""
    if player_stats_df.is_empty() or games_df.is_empty():
        return pl.DataFrame(schema=SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA)
    stats = player_stats_df.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int16),
        _first_present(player_stats_df, "recent_team", "team")
        .map_elements(normalize_football_team, return_dtype=pl.Utf8)
        .alias("team"),
        _first_present(player_stats_df, "player_id")
        .map_elements(_gsis_or_none, return_dtype=pl.Int64)
        .alias("player_id"),
        _first_present(player_stats_df, "player_display_name", "player_name").alias("player_name"),
        _first_present(player_stats_df, "position").alias("position"),
        _first_present(player_stats_df, "attempts").alias("pass_attempts"),
        _first_present(player_stats_df, "completions").alias("completions"),
        _first_present(player_stats_df, "passing_air_yards").alias("pass_air_yards"),
        _first_present(player_stats_df, "receiving_air_yards").alias("rec_air_yards"),
        _first_present(player_stats_df, "passing_yards").alias("pass_yards"),
        _first_present(player_stats_df, "passing_tds").alias("pass_tds"),
        _first_present(player_stats_df, "passing_interceptions", "interceptions").alias(
            "interceptions"
        ),
        _first_present(player_stats_df, "carries").alias("carries"),
        _first_present(player_stats_df, "rushing_yards").alias("rush_yards"),
        _first_present(player_stats_df, "rushing_tds").alias("rush_tds"),
        _first_present(player_stats_df, "targets").alias("targets"),
        _first_present(player_stats_df, "receptions").alias("receptions"),
        _first_present(player_stats_df, "receiving_yards").alias("rec_yards"),
        _first_present(player_stats_df, "receiving_tds").alias("rec_tds"),
    ).filter(pl.col("player_id").is_not_null())

    lookup = (
        _team_game_lookup(games_df)
        .select(
            "game_id",
            "event_seq",
            "game_date",
            "start_time_utc",
            "season",
            "season_type",
            "week",
            "team",
            "opponent",
        )
        .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int16))
    )
    out = stats.join(lookup, on=["season", "week", "team"], how="inner")

    snaps = _snap_share_frame(snap_counts_df, players_df)
    if snaps is not None:
        out = out.join(snaps, on=["game_id", "player_id"], how="left")
    else:
        out = out.with_columns(
            pl.lit(None, dtype=pl.Int16).alias("snaps"),
            pl.lit(None, dtype=pl.Float64).alias("snap_share"),
        )
    usage = _player_pbp_usage_frame(pbp_df)
    if usage is not None:
        out = out.join(usage, on=["game_id", "player_id"], how="left")
        out = out.with_columns(pl.col("rz_targets", "rz_carries", "gl_carries").fill_null(0))
    out = out.with_columns(
        pl.when(pl.col("snap_share").is_not_null())
        .then(pl.col("snap_share") >= 0.5)
        .otherwise(None)
        .alias("is_starter")
    )
    return _cast_to_schema(out, SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA).sort(
        "start_time_utc", "event_seq", "game_id", "team", "player_id"
    )


def _player_pbp_usage_frame(pbp_df: pl.DataFrame | None) -> pl.DataFrame | None:
    if pbp_df is None or pbp_df.is_empty():
        return None
    usage = aggregate_player_pbp_usage(_with_canonical_game_id(pbp_df))
    if usage.is_empty():
        return None
    return usage.select(
        pl.col("_gid").alias("game_id"),
        pl.col("gsis_id").map_elements(_gsis_or_none, return_dtype=pl.Int64).alias("player_id"),
        "rz_targets",
        "rz_carries",
        "gl_carries",
    ).filter(pl.col("player_id").is_not_null())


def _snap_share_frame(
    snap_counts_df: pl.DataFrame | None, players_df: pl.DataFrame | None
) -> pl.DataFrame | None:
    if snap_counts_df is None or snap_counts_df.is_empty():
        return None
    if players_df is None or "pfr_id" not in players_df.columns:
        return None
    crosswalk = (
        players_df.select(pl.col("pfr_id"), pl.col("gsis_id"))
        .drop_nulls()
        .unique(subset=["pfr_id"], keep="first")
    )
    snaps = _with_canonical_game_id(snap_counts_df).join(
        crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner"
    )
    return (
        snaps.select(
            pl.col("_gid").alias("game_id"),
            pl.col("gsis_id").map_elements(_gsis_or_none, return_dtype=pl.Int64).alias("player_id"),
            pl.col("offense_snaps").cast(pl.Int16).alias("snaps"),
            pl.col("offense_pct").cast(pl.Float64).alias("snap_share"),
        )
        .filter(pl.col("player_id").is_not_null())
        .unique(subset=["game_id", "player_id"], keep="last")
    )


def build_snap_game_logs(
    snap_counts_df: pl.DataFrame | None,
    players_df: pl.DataFrame | None,
    games_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build all-position offensive/defensive snap shares per `(game_id, player_id)`.

    Unlike `player_game_logs` (stat-sheet players only), this keeps linemen, linebackers and
    defensive backs so position-group availability can be weighted by prior snap share.
    """
    empty = pl.DataFrame(schema=SILVER_FOOTBALL_SNAP_LOGS_SCHEMA)
    if snap_counts_df is None or snap_counts_df.is_empty() or games_df.is_empty():
        return empty
    if players_df is None or "pfr_id" not in players_df.columns:
        return empty
    crosswalk = (
        players_df.select(pl.col("pfr_id"), pl.col("gsis_id"))
        .drop_nulls()
        .unique(subset=["pfr_id"], keep="first")
    )
    games = games_df.select("game_id", "event_seq", "start_time_utc", "season", "week")
    snaps = (
        _with_canonical_game_id(snap_counts_df)
        .join(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .select(
            pl.col("_gid").alias("game_id"),
            pl.col("team")
            .map_elements(normalize_football_team, return_dtype=pl.Utf8)
            .alias("team"),
            pl.col("gsis_id").map_elements(_gsis_or_none, return_dtype=pl.Int64).alias("player_id"),
            pl.col("position").cast(pl.Utf8).str.to_uppercase().alias("position"),
            pl.col("offense_snaps").cast(pl.Int16, strict=False).alias("offense_snaps"),
            pl.col("offense_pct").cast(pl.Float64, strict=False).alias("offense_pct"),
            pl.col("defense_snaps").cast(pl.Int16, strict=False).alias("defense_snaps"),
            pl.col("defense_pct").cast(pl.Float64, strict=False).alias("defense_pct"),
        )
        .filter(pl.col("player_id").is_not_null())
        .join(games, on="game_id", how="inner")
        .unique(subset=["game_id", "team", "player_id"], keep="last")
    )
    return snaps


def _availability_rows(
    source: str,
    frame: pl.DataFrame,
    lookup: pl.DataFrame,
    *,
    team_col: str,
) -> pl.DataFrame:
    games = lookup.select("game_id", "season", "week", "team", "start_time_utc").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int16)
    )
    base = frame.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int16),
        pl.col(team_col).map_elements(normalize_football_team, return_dtype=pl.Utf8).alias("team"),
        pl.col("gsis_id").map_elements(_gsis_or_none, return_dtype=pl.Int64).alias("player_id"),
        _first_present(frame, "position", "depth_position").alias("position"),
        *[
            pl.col(c)
            for c in (
                "report_status",
                "practice_status",
                "date_modified",
                "depth_team",
                "dt",
                "formation",
            )
            if c in frame.columns
        ],
    ).filter(pl.col("player_id").is_not_null())
    return base.join(games, on=["season", "week", "team"], how="inner").with_columns(
        pl.lit(source).alias("source")
    )


def _normalize_depth_chart(depth_df: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    """Map the 2025+ nflverse depth-chart layout onto the legacy `(season, week, club_code, ...)`.

    The newer feed has timestamped snapshots (`dt`, `team`, `pos_abb`, `pos_rank`) but no
    `season`/`week`; each snapshot is assigned to the team's next scheduled game at or after `dt`.
    """
    if "season" in depth_df.columns or not {"dt", "team", "pos_abb"} <= set(depth_df.columns):
        return depth_df
    snaps = (
        depth_df.with_columns(
            pl.col("dt")
            .cast(pl.Utf8)
            .str.to_datetime(time_zone="UTC", time_unit="us", strict=False)
            .alias("_dt"),
            pl.col("team")
            .map_elements(normalize_football_team, return_dtype=pl.Utf8)
            .alias("_team"),
        )
        .filter(pl.col("_dt").is_not_null() & pl.col("gsis_id").is_not_null())
        .sort("_dt")
    )
    games = lookup.select(
        pl.col("team").alias("_team"),
        pl.col("season"),
        pl.col("week"),
        pl.col("start_time_utc").alias("_next_kickoff"),
    ).sort("_next_kickoff")
    joined = snaps.join_asof(
        games,
        left_on="_dt",
        right_on="_next_kickoff",
        by="_team",
        strategy="forward",
        check_sortedness=False,
    ).filter(pl.col("season").is_not_null())
    return joined.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("_team").alias("club_code"),
        pl.col("gsis_id"),
        pl.col("pos_abb").alias("position"),
        pl.col("pos_abb").alias("depth_position"),
        pl.col("pos_rank").cast(pl.Utf8).alias("depth_team"),
        pl.col("_dt").alias("dt"),
    )


def build_availability_snapshots(
    games_df: pl.DataFrame,
    *,
    injuries_df: pl.DataFrame | None = None,
    depth_charts_df: pl.DataFrame | None = None,
    weekly_rosters_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build as-of availability snapshots (injury report, depth chart, official inactives)."""
    if games_df.is_empty():
        return pl.DataFrame(schema=SILVER_FOOTBALL_AVAILABILITY_SCHEMA)
    lookup = _team_game_lookup(games_df)
    parts: list[pl.DataFrame] = []

    if injuries_df is not None and not injuries_df.is_empty():
        inj = _availability_rows(SOURCE_INJURY_REPORT, injuries_df, lookup, team_col="team")
        if "date_modified" not in inj.columns:
            inj = inj.with_columns(
                pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("date_modified")
            )
        parts.append(
            inj.with_columns(
                pl.coalesce(
                    pl.col("date_modified").cast(pl.Datetime("us", "UTC")),
                    pl.col("start_time_utc") - UNTIMED_INJURY_REPORT_LEAD,
                ).alias("as_of_time_utc"),
                pl.col("report_status").str.strip_chars().replace("", None).alias("report_status"),
                pl.col("practice_status")
                .str.strip_chars()
                .replace("", None)
                .alias("practice_status"),
                pl.lit(None, dtype=pl.Int8).alias("depth_rank"),
                pl.lit(False).alias("is_inactive"),
            )
        )

    if depth_charts_df is not None and not depth_charts_df.is_empty():
        # Offense, defense and special teams are all kept: Milestone 27 weights position-group
        # availability (OL, pass rush, secondary) from the full depth chart.
        depth = _normalize_depth_chart(depth_charts_df, lookup)
        dc = _availability_rows(SOURCE_DEPTH_CHART, depth, lookup, team_col="club_code")
        as_of = pl.col("start_time_utc") - UNTIMED_DEPTH_CHART_LEAD
        if "dt" in dc.columns:
            as_of = pl.coalesce(pl.col("dt").cast(pl.Datetime("us", "UTC"), strict=False), as_of)
        parts.append(
            dc.with_columns(
                as_of.alias("as_of_time_utc"),
                pl.col("depth_team").cast(pl.Int8, strict=False).alias("depth_rank"),
                pl.lit(None, dtype=pl.Utf8).alias("report_status"),
                pl.lit(None, dtype=pl.Utf8).alias("practice_status"),
                pl.lit(False).alias("is_inactive"),
            )
        )

    if weekly_rosters_df is not None and not weekly_rosters_df.is_empty():
        inactive = weekly_rosters_df.filter(pl.col("status") == "INA")
        if not inactive.is_empty():
            ina = _availability_rows(SOURCE_INACTIVES, inactive, lookup, team_col="team")
            parts.append(
                ina.with_columns(
                    (pl.col("start_time_utc") - timedelta(minutes=TIER1_CUTOFF_MINUTES)).alias(
                        "as_of_time_utc"
                    ),
                    pl.lit(None, dtype=pl.Utf8).alias("report_status"),
                    pl.lit(None, dtype=pl.Utf8).alias("practice_status"),
                    pl.lit(None, dtype=pl.Int8).alias("depth_rank"),
                    pl.lit(True).alias("is_inactive"),
                )
            )

    if not parts:
        return pl.DataFrame(schema=SILVER_FOOTBALL_AVAILABILITY_SCHEMA)
    combined = pl.concat([_cast_to_schema(p, SILVER_FOOTBALL_AVAILABILITY_SCHEMA) for p in parts])
    return combined.unique(
        subset=["game_id", "team", "player_id", "source", "as_of_time_utc"], keep="last"
    ).sort("as_of_time_utc", "game_id", "team", "source", "player_id")


class FootballSilverReconciler:
    """Reconciles nflverse feeds into `zstd` Silver Parquet tables with idempotent upserts."""

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
    def player_logs_path(self) -> Path:
        return self.silver_dir / "player_game_logs.parquet"

    @property
    def qb_logs_path(self) -> Path:
        return self.silver_dir / "qb_game_logs.parquet"

    @property
    def snap_logs_path(self) -> Path:
        return self.silver_dir / "snap_game_logs.parquet"

    @property
    def team_micro_path(self) -> Path:
        return self.silver_dir / "team_pbp_micro_logs.parquet"

    @property
    def player_micro_path(self) -> Path:
        return self.silver_dir / "player_pbp_micro_logs.parquet"

    @property
    def availability_path(self) -> Path:
        return self.silver_dir / "team_week_availability.parquet"

    @staticmethod
    def _upsert(
        path: Path,
        incoming: pl.DataFrame,
        *,
        schema: dict[str, pl.DataType],
        unique_keys: list[str],
        sort_keys: list[str],
        persist: bool,
    ) -> pl.DataFrame:
        incoming_typed = _cast_to_schema(incoming, schema)
        if path.exists():
            existing = _cast_to_schema(pl.read_parquet(path), schema)
            combined = pl.concat([existing, incoming_typed], how="vertical")
        else:
            combined = incoming_typed
        merged = combined.unique(subset=unique_keys, keep="last", maintain_order=True).sort(
            sort_keys
        )
        if persist:
            merged.write_parquet(path, compression="zstd")
        return merged

    def read_games(self) -> pl.DataFrame:
        """Return the persisted Silver `games` table (empty frame if not yet built)."""
        if not self.games_path.exists():
            return pl.DataFrame(schema=SILVER_FOOTBALL_GAMES_SCHEMA)
        return _cast_to_schema(pl.read_parquet(self.games_path), SILVER_FOOTBALL_GAMES_SCHEMA)

    def reconcile_games(
        self,
        schedule_df: pl.DataFrame,
        *,
        starting_qbs: dict[tuple[int, str], int] | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Normalize franchises, quarantine odds/weather, and upsert `games.parquet`."""
        incoming = build_games_frame(schedule_df, starting_qbs=starting_qbs)
        return self._upsert(
            self.games_path,
            incoming,
            schema=SILVER_FOOTBALL_GAMES_SCHEMA,
            unique_keys=["game_id"],
            sort_keys=["start_time_utc", "event_seq", "game_id"],
            persist=persist,
        )

    def reconcile_team_game_logs(
        self, pbp_df: pl.DataFrame, *, persist: bool = True
    ) -> pl.DataFrame:
        """Aggregate play-by-play into `team_game_logs.parquet`."""
        incoming = build_team_game_logs(pbp_df, self.read_games())
        return self._upsert(
            self.team_logs_path,
            incoming,
            schema=SILVER_FOOTBALL_TEAM_LOGS_SCHEMA,
            unique_keys=["game_id", "team"],
            sort_keys=["start_time_utc", "event_seq", "game_id", "team"],
            persist=persist,
        )

    def reconcile_qb_game_logs(self, pbp_df: pl.DataFrame, *, persist: bool = True) -> pl.DataFrame:
        """Aggregate dropbacks into `qb_game_logs.parquet`."""
        incoming = build_qb_game_logs(pbp_df, self.read_games())
        return self._upsert(
            self.qb_logs_path,
            incoming,
            schema=SILVER_FOOTBALL_QB_LOGS_SCHEMA,
            unique_keys=["game_id", "team", "qb_id"],
            sort_keys=["start_time_utc", "event_seq", "game_id", "team", "qb_id"],
            persist=persist,
        )

    def reconcile_player_game_logs(
        self,
        player_stats_df: pl.DataFrame,
        *,
        snap_counts_df: pl.DataFrame | None = None,
        players_df: pl.DataFrame | None = None,
        pbp_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Build `player_game_logs.parquet` from weekly stats (+ snap share, pbp usage)."""
        incoming = build_player_game_logs(
            player_stats_df,
            self.read_games(),
            snap_counts_df=snap_counts_df,
            players_df=players_df,
            pbp_df=pbp_df,
        )
        return self._upsert(
            self.player_logs_path,
            incoming,
            schema=SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA,
            unique_keys=["game_id", "player_id"],
            sort_keys=["start_time_utc", "event_seq", "game_id", "team", "player_id"],
            persist=persist,
        )

    def reconcile_team_pbp_micro_logs(
        self,
        pbp_df: pl.DataFrame,
        *,
        ngs_passing: pl.DataFrame | None = None,
        ngs_rushing: pl.DataFrame | None = None,
        ngs_receiving: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Build `team_pbp_micro_logs.parquet` (pace, PROE, CPOE, pressure proxy, NGS totals)."""
        incoming = build_team_pbp_micro_logs(
            pbp_df,
            self.read_games(),
            ngs_passing=ngs_passing,
            ngs_rushing=ngs_rushing,
            ngs_receiving=ngs_receiving,
        )
        return self._upsert(
            self.team_micro_path,
            incoming,
            schema=SILVER_FOOTBALL_TEAM_MICRO_SCHEMA,
            unique_keys=["game_id", "team"],
            sort_keys=["start_time_utc", "event_seq", "game_id", "team"],
            persist=persist,
        )

    def reconcile_player_pbp_micro_logs(
        self,
        pbp_df: pl.DataFrame,
        *,
        ngs_rushing: pl.DataFrame | None = None,
        ngs_receiving: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Build `player_pbp_micro_logs.parquet` (receiving/rushing micro usage, NGS totals)."""
        incoming = build_player_pbp_micro_logs(
            pbp_df, self.read_games(), ngs_rushing=ngs_rushing, ngs_receiving=ngs_receiving
        )
        return self._upsert(
            self.player_micro_path,
            incoming,
            schema=SILVER_FOOTBALL_PLAYER_MICRO_SCHEMA,
            unique_keys=["game_id", "team", "player_id"],
            sort_keys=["start_time_utc", "event_seq", "game_id", "team", "player_id"],
            persist=persist,
        )

    def reconcile_snap_game_logs(
        self,
        snap_counts_df: pl.DataFrame | None,
        *,
        players_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Build `snap_game_logs.parquet` (all-position snap shares) from nflverse snap counts."""
        incoming = build_snap_game_logs(snap_counts_df, players_df, self.read_games())
        return self._upsert(
            self.snap_logs_path,
            incoming,
            schema=SILVER_FOOTBALL_SNAP_LOGS_SCHEMA,
            unique_keys=["game_id", "team", "player_id"],
            sort_keys=["start_time_utc", "game_id", "team", "player_id"],
            persist=persist,
        )

    def reconcile_availability(
        self,
        *,
        injuries_df: pl.DataFrame | None = None,
        depth_charts_df: pl.DataFrame | None = None,
        weekly_rosters_df: pl.DataFrame | None = None,
        persist: bool = True,
    ) -> pl.DataFrame:
        """Build `team_week_availability.parquet` as-of snapshots."""
        incoming = build_availability_snapshots(
            self.read_games(),
            injuries_df=injuries_df,
            depth_charts_df=depth_charts_df,
            weekly_rosters_df=weekly_rosters_df,
        )
        return self._upsert(
            self.availability_path,
            incoming,
            schema=SILVER_FOOTBALL_AVAILABILITY_SCHEMA,
            unique_keys=["game_id", "team", "player_id", "source", "as_of_time_utc"],
            sort_keys=["as_of_time_utc", "game_id", "team", "source", "player_id"],
            persist=persist,
        )
