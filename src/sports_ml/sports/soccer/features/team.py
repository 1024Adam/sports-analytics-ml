"""EPL pre-match team rolling, pressing (PPDA), regression, and schedule features (`team.py`).

Computes strictly leak-free pre-match rolling (5, 10, 19 matches) and EWMA (alpha=0.14)
features for:
- Underlying Shot & Chance Quality: `npxgf_per_90`, `npxga_per_90`, `npxgd_per_90`,
  `npxgf_pct`, `xgf_per_90`, `xga_per_90`, `shots_per_90`, `sot_pct`.
- Tactical & Territorial Control: `ppda`, `opp_ppda`, `deep_completions_per_90`, `corner_share`.
- Finishing & Defensive Luck / Regression Signals: `finishing_above_xg_w10` (`GF - xGF`) and
  `defensive_ga_above_xga_w10` (`GA - xGA`).
- Schedule Congestion & Promoted Status: `rest_days`, `is_midweek_turnaround` (`rest_days <= 3`),
  `matches_last_14d`, geodesic stadium travel distance, and `is_promoted_team`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Final

import polars as pl

from sports_ml.core.ratings import EloConfig, PairwiseEloEngine
from sports_ml.core.schedule import compute_schedule_features, haversine_distance_km
from sports_ml.core.shrinkage import bayesian_shrinkage
from sports_ml.core.temporal import (
    pregame_ewma,
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_rolling_sum,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.soccer.constants import get_stadium_metadata, normalize_soccer_club
from sports_ml.sports.soccer.ingestion.non_league_fixtures import (
    EUROPEAN_COMPETITION_TYPES,
    normalize_competition_type,
    resolve_fixture_venue_coords,
)

SOCCER_TEAM_SHRINKAGE_K: Final[float] = 6.0
DEFAULT_SOCCER_TEAM_WINDOWS: Final[tuple[int, ...]] = (5, 10, 19)
DEFAULT_SOCCER_EWMA_ALPHA: Final[float] = 0.14

SOCCER_ELO_CONFIG: Final[EloConfig] = EloConfig(
    base_rating=1500.0,
    home_field_advantage=48.0,
    k_factor=18.0,
    offseason_retention=0.75,
    use_mov_multiplier=True,
    draw_param=0.26,
    promoted_team_rating=1390.0,
)

# Causal EPL priors (established top-flight vs. newly promoted Championship club priors)
PRIOR_NPXGF_PER_90: Final[float] = 1.35
PRIOR_NPXGA_PER_90: Final[float] = 1.35
PRIOR_PROMOTED_NPXGF_PER_90: Final[float] = 1.05
PRIOR_PROMOTED_NPXGA_PER_90: Final[float] = 1.68

PRIOR_XGF_PER_90: Final[float] = 1.45
PRIOR_XGA_PER_90: Final[float] = 1.45
PRIOR_PROMOTED_XGF_PER_90: Final[float] = 1.12
PRIOR_PROMOTED_XGA_PER_90: Final[float] = 1.78

PRIOR_SHARE_50: Final[float] = 0.50
PRIOR_PROMOTED_SHARE: Final[float] = 0.39
PRIOR_SHOTS_PER_90: Final[float] = 12.8
PRIOR_SOT_AGAINST_PER_90: Final[float] = 4.5
PRIOR_SOT_PCT: Final[float] = 0.34
PRIOR_PPDA: Final[float] = 11.5
PRIOR_DEEP_PER_90: Final[float] = 6.5
PRIOR_OPEN_PLAY_XGF_PER_90: Final[float] = 1.00
PRIOR_OPEN_PLAY_XGA_PER_90: Final[float] = 1.00
PRIOR_SET_PIECE_XGF_PER_90: Final[float] = 0.35
PRIOR_SET_PIECE_XGA_PER_90: Final[float] = 0.35
PRIOR_BOX_SHOT_SHARE: Final[float] = 0.62
PRIOR_THROUGHBALL_XGF_PER_90: Final[float] = 0.14


def _ensure_soccer_team_log_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Populate optional team match-log columns with defaults so minimal fixtures work cleanly."""
    working = df
    if "match_id" in working.columns and "event_id" not in working.columns:
        working = working.with_columns(pl.col("match_id").cast(pl.Int64).alias("event_id"))
    if "match_date" in working.columns and "game_date" not in working.columns:
        working = working.with_columns(pl.col("match_date").cast(pl.Date).alias("game_date"))

    defaults: list[tuple[str, float | int, pl.DataType]] = [
        ("shots_for", 12, pl.Int16()),
        ("shots_against", 12, pl.Int16()),
        ("sot_for", 4, pl.Int16()),
        ("sot_against", 4, pl.Int16()),
        ("xgf", 1.40, pl.Float64()),
        ("xga", 1.40, pl.Float64()),
        ("npxgf", 1.30, pl.Float64()),
        ("npxga", 1.30, pl.Float64()),
        ("ppda", 11.5, pl.Float64()),
        ("opp_ppda", 11.5, pl.Float64()),
        ("deep_completions", 6, pl.Int16()),
        ("opp_deep_completions", 6, pl.Int16()),
        ("corners_for", 5, pl.Int16()),
        ("corners_against", 5, pl.Int16()),
        ("fouls_for", 11, pl.Int16()),
        ("fouls_against", 11, pl.Int16()),
        ("yellow_cards", 2, pl.Int16()),
        ("red_cards", 0, pl.Int16()),
    ]
    missing_exprs = [
        pl.lit(val, dtype=dtype).alias(col)
        for col, val, dtype in defaults
        if col not in working.columns
    ]
    return working.with_columns(missing_exprs) if missing_exprs else working


def _annotate_promoted_and_matches_14d(sorted_df: pl.DataFrame) -> pl.DataFrame:
    """Compute leak-free `is_promoted_team` and `matches_last_14d` per `(team, season)`."""
    if sorted_df.is_empty():
        return sorted_df.with_columns(
            pl.lit(False, dtype=pl.Boolean).alias("is_promoted_team"),
            pl.lit(1, dtype=pl.Int32).alias("matches_last_14d"),
            pl.lit(None, dtype=pl.Float64).alias("venue_lat"),
            pl.lit(None, dtype=pl.Float64).alias("venue_lon"),
            pl.lit("Europe/London", dtype=pl.Utf8).alias("venue_timezone"),
        )

    rows = sorted_df.to_dicts()
    n_rows = len(rows)

    promoted_out = [False] * n_rows
    matches_14d_out = [1] * n_rows
    lat_out: list[float | None] = [None] * n_rows
    lon_out: list[float | None] = [None] * n_rows
    tz_out: list[str] = ["Europe/London"] * n_rows

    current_season: int | None = None
    prev_season_teams: set[str] = set()
    active_season_teams: set[str] = set()
    has_prior_season = False

    date_hist: dict[tuple[str, int], list[date]] = {}

    for idx, row in enumerate(rows):
        season = int(row["season"])
        if current_season is not None and season != current_season:
            prev_season_teams = set(active_season_teams)
            active_season_teams = set()
            has_prior_season = True
        current_season = season

        team = normalize_soccer_club(str(row["team"]))
        opp = normalize_soccer_club(str(row.get("opponent") or ""))
        is_home = bool(row.get("is_home", True))
        active_season_teams.add(team)

        explicit_prom = row.get("is_promoted_team")
        if explicit_prom is not None:
            promoted_out[idx] = bool(explicit_prom)
        elif has_prior_season:
            promoted_out[idx] = team not in prev_season_teams
        else:
            promoted_out[idx] = False

        # Resolve stadium coordinates of the match venue (home team's stadium)
        venue_club = team if is_home else (opp or team)
        meta = get_stadium_metadata(venue_club)
        if meta is not None:
            lat_out[idx] = meta.latitude
            lon_out[idx] = meta.longitude
            tz_out[idx] = meta.timezone

        start_utc: datetime = row["start_time_utc"]
        ev_date: date = (
            row["match_date"]
            if "match_date" in row and row["match_date"] is not None
            else start_utc.date()
        )
        t_hist = date_hist.setdefault((team, season), [])
        t_hist.append(ev_date)
        matches_14d_out[idx] = sum(1 for d in t_hist if 0 <= (ev_date - d).days <= 13)

    return sorted_df.with_columns(
        pl.Series("is_promoted_team", promoted_out, dtype=pl.Boolean),
        pl.Series("matches_last_14d", matches_14d_out, dtype=pl.Int32),
        pl.Series("venue_lat", lat_out, dtype=pl.Float64),
        pl.Series("venue_lon", lon_out, dtype=pl.Float64),
        pl.Series("venue_timezone", tz_out, dtype=pl.Utf8),
    )


def _pregame_share_expr(
    for_col: str,
    against_col: str,
    window: int,
    group_by: str,
) -> pl.Expr:
    f_sum = pregame_rolling_sum(for_col, window=window, group_by=group_by)
    a_sum = pregame_rolling_sum(against_col, window=window, group_by=group_by)
    denom = f_sum + a_sum
    return f_sum / pl.when(denom > 0.0).then(denom).otherwise(None)


def compute_cross_competition_fatigue_features(
    target_df: pl.DataFrame,
    all_comp_schedule_df: pl.DataFrame | None = None,
    *,
    max_rest_days: float = 7.0,
) -> pl.DataFrame:
    """Annotate EPL team rows with cross-competition rest, UEFA travel, and congestion features.

    Queries the unified all-competition timeline (`strictly < T_match`) before every
    EPL fixture to compute:
    - `true_all_comp_rest_days`: Calendar days since the club's last competitive match in
      any competition (capped at `max_rest_days`, default `7.0`).
    - `had_midweek_european_match`: Binary indicator (`1` if the club played a `UCL` /
      `UEL` / `UECL` fixture within the last 5 days).
    - `had_midweek_away_european_match`: Binary indicator (`1` if the club played an away
      `UCL` / `UEL` / `UECL` fixture within the last 5 days).
    - `european_travel_km_last_7d`: Total round-trip geodesic distance (km) traveled for
      away European fixtures in the prior 7 days.
    - `all_comp_matches_last_14d` & `all_comp_matches_last_28d`: True multi-competition
      fixture density (including the current match, matching `matches_last_14d` semantics).
    - `cup_extra_time_last_7d`: Binary indicator (`1` if the club played extra time in any
      competition within the prior 7 days).
    """
    if target_df.is_empty():
        return target_df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("true_all_comp_rest_days"),
            pl.lit(None, dtype=pl.Int8).alias("had_midweek_european_match"),
            pl.lit(None, dtype=pl.Int8).alias("had_midweek_away_european_match"),
            pl.lit(None, dtype=pl.Float64).alias("european_travel_km_last_7d"),
            pl.lit(None, dtype=pl.Int32).alias("all_comp_matches_last_14d"),
            pl.lit(None, dtype=pl.Int32).alias("all_comp_matches_last_28d"),
            pl.lit(None, dtype=pl.Int8).alias("cup_extra_time_last_7d"),
        )

    # Build unified per-(team, season) timeline of all competitive appearances
    # Each entry: (start_time_utc, match_date, comp_type, is_home, venue_lat, venue_lon, extra_time)
    timeline_by_team: dict[
        tuple[str, int],
        list[tuple[datetime, date, str, bool, float, float, bool]],
    ] = {}

    def _add_entry(
        team_raw: str,
        opp_raw: str,
        season_val: int,
        start_utc: datetime,
        m_date: date,
        comp_raw: str,
        is_home_val: bool,
        v_lat: float | None,
        v_lon: float | None,
        et_val: bool,
    ) -> None:
        t_code = normalize_soccer_club(team_raw)
        o_code = normalize_soccer_club(opp_raw)
        c_type = normalize_competition_type(comp_raw)
        lat_f, lon_f = resolve_fixture_venue_coords(
            t_code, o_code, c_type, is_home_val, v_lat, v_lon
        )
        utc_norm = start_utc.astimezone(UTC) if start_utc.tzinfo else start_utc.replace(tzinfo=UTC)
        timeline_by_team.setdefault((t_code, season_val), []).append(
            (utc_norm, m_date, c_type, is_home_val, lat_f, lon_f, et_val)
        )

    for r in target_df.iter_rows(named=True):
        s_utc: datetime = r["start_time_utc"]
        d_val: date = r.get("match_date") or r.get("game_date") or s_utc.date()
        _add_entry(
            str(r["team"]),
            str(r.get("opponent") or ""),
            int(r["season"]),
            s_utc,
            d_val,
            str(r.get("competition_type") or "EPL"),
            bool(r.get("is_home", True)),
            float(r["venue_lat"]) if r.get("venue_lat") is not None else None,
            float(r["venue_lon"]) if r.get("venue_lon") is not None else None,
            bool(r.get("extra_time_played", False)),
        )

    if all_comp_schedule_df is not None and not all_comp_schedule_df.is_empty():
        for r in all_comp_schedule_df.iter_rows(named=True):
            s_utc = r["start_time_utc"]
            d_val = r.get("match_date") or r.get("game_date") or s_utc.date()
            _add_entry(
                str(r["team"]),
                str(r.get("opponent") or ""),
                int(r.get("season") or (d_val.year if d_val.month >= 7 else d_val.year - 1)),
                s_utc,
                d_val,
                str(r.get("competition_type") or "EPL"),
                bool(r.get("is_home", True)),
                float(r["venue_lat"]) if r.get("venue_lat") is not None else None,
                float(r["venue_lon"]) if r.get("venue_lon") is not None else None,
                bool(r.get("extra_time_played", False)),
            )

    # Deduplicate (team, season) timelines by (match_date, comp_type) keeping latest metadata
    dedup_timelines: dict[
        tuple[str, int],
        list[tuple[datetime, date, str, bool, float, float, bool]],
    ] = {}
    for key, entries in timeline_by_team.items():
        by_date_comp: dict[
            tuple[date, str], tuple[datetime, date, str, bool, float, float, bool]
        ] = {}
        for item in sorted(entries, key=lambda x: x[0]):
            by_date_comp[(item[1], item[2])] = item
        dedup_timelines[key] = sorted(by_date_comp.values(), key=lambda x: (x[0], x[1]))

    n_rows = target_df.height
    true_rest_out = [float(max_rest_days)] * n_rows
    midweek_euro_out = [0] * n_rows
    midweek_away_euro_out = [0] * n_rows
    euro_travel_7d_out = [0.0] * n_rows
    all_comp_14d_out = [1] * n_rows
    all_comp_28d_out = [1] * n_rows
    cup_et_7d_out = [0] * n_rows

    for idx, r in enumerate(target_df.iter_rows(named=True)):
        t_code = normalize_soccer_club(str(r["team"]))
        season_val = int(r["season"])
        s_utc = r["start_time_utc"]
        curr_utc = s_utc.astimezone(UTC) if s_utc.tzinfo else s_utc.replace(tzinfo=UTC)
        curr_date: date = r.get("match_date") or r.get("game_date") or curr_utc.date()

        home_meta = get_stadium_metadata(t_code)
        home_lat = home_meta.latitude if home_meta is not None else 51.5074
        home_lon = home_meta.longitude if home_meta is not None else -0.1278

        t_entries = dedup_timelines.get((t_code, season_val), [])
        # Strictly prior matches (< curr_utc, and not the same EPL fixture on curr_date)
        prior_entries = [
            e for e in t_entries if e[0] < curr_utc and not (e[1] == curr_date and e[2] == "EPL")
        ]

        if prior_entries:
            last_utc, last_date, _, _, _, _, _ = prior_entries[-1]
            raw_days = float((curr_date - last_date).days)
            if raw_days <= 0.0:
                raw_days = max(0.0, (curr_utc - last_utc).total_seconds() / 86400.0)
            true_rest_out[idx] = min(float(max_rest_days), raw_days)

            has_euro_5d = False
            has_away_euro_5d = False
            euro_km_7d = 0.0
            has_et_7d = False

            for _, p_date, p_comp, p_home, p_lat, p_lon, p_et in prior_entries:
                days_ago = (curr_date - p_date).days
                if 0 < days_ago <= 5 and p_comp in EUROPEAN_COMPETITION_TYPES:
                    has_euro_5d = True
                    if not p_home:
                        has_away_euro_5d = True
                if 0 < days_ago <= 7:
                    if p_et:
                        has_et_7d = True
                    if p_comp in EUROPEAN_COMPETITION_TYPES and not p_home:
                        # Round-trip distance from club's home stadium to European away venue
                        one_way = haversine_distance_km(home_lat, home_lon, p_lat, p_lon)
                        euro_km_7d += 2.0 * one_way

            midweek_euro_out[idx] = 1 if has_euro_5d else 0
            midweek_away_euro_out[idx] = 1 if has_away_euro_5d else 0
            euro_travel_7d_out[idx] = float(euro_km_7d)
            cup_et_7d_out[idx] = 1 if has_et_7d else 0

        # Window counts include current match (+1) plus prior matches in [curr_date - 13d]
        prior_14d = sum(1 for e in prior_entries if 0 <= (curr_date - e[1]).days <= 13)
        prior_28d = sum(1 for e in prior_entries if 0 <= (curr_date - e[1]).days <= 27)
        all_comp_14d_out[idx] = 1 + prior_14d
        all_comp_28d_out[idx] = 1 + prior_28d

    return target_df.with_columns(
        pl.Series("true_all_comp_rest_days", true_rest_out, dtype=pl.Float64),
        pl.Series("had_midweek_european_match", midweek_euro_out, dtype=pl.Int8),
        pl.Series("had_midweek_away_european_match", midweek_away_euro_out, dtype=pl.Int8),
        pl.Series("european_travel_km_last_7d", euro_travel_7d_out, dtype=pl.Float64),
        pl.Series("all_comp_matches_last_14d", all_comp_14d_out, dtype=pl.Int32),
        pl.Series("all_comp_matches_last_28d", all_comp_28d_out, dtype=pl.Int32),
        pl.Series("cup_extra_time_last_7d", cup_et_7d_out, dtype=pl.Int8),
    )


def build_soccer_team_rolling_features(
    team_match_logs: pl.DataFrame,
    *,
    all_comp_schedule_df: pl.DataFrame | None = None,
    team_shot_micro_logs: pl.DataFrame | None = None,
    windows: Sequence[int] = DEFAULT_SOCCER_TEAM_WINDOWS,
    ewma_alpha: float = DEFAULT_SOCCER_EWMA_ALPHA,
    shrinkage_k: float = SOCCER_TEAM_SHRINKAGE_K,
) -> pl.DataFrame:
    """Compute leak-free pre-match EPL team rolling, tactical (PPDA), and regression features.

    Tactical Isolation Guarantee:
    If `team_match_logs` contains non-EPL rows (`competition_type != "EPL"`), they are
    routed into the cross-competition schedule timeline and strictly filtered out before
    computing rolling `npxGF/90`, `npxGA/90`, `PPDA`, or finishing regression.
    """
    extra_sched_rows: pl.DataFrame | None = all_comp_schedule_df
    working_logs = team_match_logs
    if "competition_type" in working_logs.columns:
        non_epl_logs = working_logs.filter(
            pl.col("competition_type").map_elements(
                normalize_competition_type, return_dtype=pl.Utf8
            )
            != "EPL"
        )
        if not non_epl_logs.is_empty():
            extra_sched_rows = (
                pl.concat([extra_sched_rows, non_epl_logs], how="diagonal_relaxed")
                if extra_sched_rows is not None and not extra_sched_rows.is_empty()
                else non_epl_logs
            )
        working_logs = working_logs.filter(
            pl.col("competition_type").map_elements(
                normalize_competition_type, return_dtype=pl.Utf8
            )
            == "EPL"
        )

    prepared = _ensure_soccer_team_log_columns(working_logs).with_columns(
        pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
        pl.col("opponent").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
    )
    event_col = "match_id" if "match_id" in prepared.columns else "event_id"

    if team_shot_micro_logs is not None and not team_shot_micro_logs.is_empty():
        sm_ev_col = "match_id" if "match_id" in team_shot_micro_logs.columns else "event_id"
        sm_cols = [
            "open_play_xgf",
            "open_play_xga",
            "set_piece_xgf",
            "set_piece_xga",
            "tied_state_npxgd_per_90",
            "box_shot_share",
            "throughball_xgf",
        ]
        avail_sm = [c for c in sm_cols if c in team_shot_micro_logs.columns]
        sm_lookup = (
            team_shot_micro_logs.with_columns(
                pl.col("team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8)
            )
            .select(
                [
                    pl.col(sm_ev_col).cast(pl.Int64).alias(event_col),
                    pl.col("team"),
                    *[pl.col(c).cast(pl.Float64).alias(f"_sm_{c}") for c in avail_sm],
                ]
            )
            .unique(subset=[event_col, "team"], keep="last")
        )
        prepared = prepared.join(sm_lookup, on=[event_col, "team"], how="left")
        for c in avail_sm:
            if c in prepared.columns:
                prepared = prepared.with_columns(
                    pl.coalesce([pl.col(f"_sm_{c}"), pl.col(c).cast(pl.Float64)]).alias(c)
                )
            else:
                prepared = prepared.with_columns(pl.col(f"_sm_{c}").alias(c))
        prepared = prepared.drop([f"_sm_{c}" for c in avail_sm])

    shot_micro_defaults: list[tuple[str, pl.Expr]] = [
        ("open_play_xgf", 0.74 * pl.col("npxgf").cast(pl.Float64)),
        ("open_play_xga", 0.74 * pl.col("npxga").cast(pl.Float64)),
        ("set_piece_xgf", 0.26 * pl.col("npxgf").cast(pl.Float64)),
        ("set_piece_xga", 0.26 * pl.col("npxga").cast(pl.Float64)),
        (
            "tied_state_npxgd_per_90",
            pl.col("npxgf").cast(pl.Float64) - pl.col("npxga").cast(pl.Float64),
        ),
        ("box_shot_share", pl.lit(PRIOR_BOX_SHOT_SHARE, dtype=pl.Float64)),
        ("throughball_xgf", 0.10 * pl.col("npxgf").cast(pl.Float64)),
    ]
    for col_name, fallback_expr in shot_micro_defaults:
        if col_name in prepared.columns:
            prepared = prepared.with_columns(
                pl.col(col_name).cast(pl.Float64).fill_null(fallback_expr).alias(col_name)
            )
        else:
            prepared = prepared.with_columns(fallback_expr.alias(col_name))

    sorted_df = sort_events_chronologically(prepared, extra_tiebreakers=["team"])
    annotated = _annotate_promoted_and_matches_14d(sorted_df)
    with_sched = compute_schedule_features(annotated, entity_col="team", date_col="game_date")
    with_sched = compute_cross_competition_fatigue_features(
        with_sched, extra_sched_rows
    ).with_columns(
        pl.col("true_all_comp_rest_days").round(0).cast(pl.Int32).alias("rest_days"),
        pl.col("all_comp_matches_last_14d").cast(pl.Int32).alias("matches_last_14d"),
    )

    is_prom = pl.col("is_promoted_team")
    with_derived = with_sched.with_columns(
        (pl.col("rest_days") <= 3).alias("is_midweek_turnaround"),
        (pl.col("npxgf").cast(pl.Float64) - pl.col("npxga").cast(pl.Float64)).alias("_match_npxgd"),
        (
            pl.col("npxgf").cast(pl.Float64)
            / pl.when((pl.col("npxgf") + pl.col("npxga")) > 0.0)
            .then(pl.col("npxgf") + pl.col("npxga"))
            .otherwise(0.50)
        ).alias("_match_npxgf_share"),
        (pl.col("goals_for").cast(pl.Float64) - pl.col("xgf").cast(pl.Float64)).alias(
            "_match_finishing_above_xg"
        ),
        (pl.col("goals_against").cast(pl.Float64) - pl.col("xga").cast(pl.Float64)).alias(
            "_match_ga_above_xga"
        ),
        pl.when(is_prom)
        .then(PRIOR_PROMOTED_NPXGF_PER_90)
        .otherwise(PRIOR_NPXGF_PER_90)
        .alias("_p_npxgf"),
        pl.when(is_prom)
        .then(PRIOR_PROMOTED_NPXGA_PER_90)
        .otherwise(PRIOR_NPXGA_PER_90)
        .alias("_p_npxga"),
        pl.when(is_prom)
        .then(PRIOR_PROMOTED_XGF_PER_90)
        .otherwise(PRIOR_XGF_PER_90)
        .alias("_p_xgf"),
        pl.when(is_prom)
        .then(PRIOR_PROMOTED_XGA_PER_90)
        .otherwise(PRIOR_XGA_PER_90)
        .alias("_p_xga"),
        pl.when(is_prom).then(PRIOR_PROMOTED_SHARE).otherwise(PRIOR_SHARE_50).alias("_p_share"),
        pregame_sample_count("npxgf", group_by="team").alias("team_prior_matches_n"),
    )

    exprs: list[pl.Expr] = []
    for w in windows:
        n_w = pregame_sample_count("npxgf", group_by="team", window=w)
        r_npxgf = pregame_rolling_mean("npxgf", window=w, group_by="team")
        r_npxga = pregame_rolling_mean("npxga", window=w, group_by="team")
        r_xgf = pregame_rolling_mean("xgf", window=w, group_by="team")
        r_xga = pregame_rolling_mean("xga", window=w, group_by="team")
        r_npxgf_pct = _pregame_share_expr("npxgf", "npxga", window=w, group_by="team")
        r_shots = pregame_rolling_mean("shots_for", window=w, group_by="team")
        r_sot_against = pregame_rolling_mean("sot_against", window=w, group_by="team")
        r_sot_pct = pregame_exposure_rate(
            "sot_for", "shots_for", window=w, group_by="team", scale_factor=1.0
        )
        r_ppda = pregame_rolling_mean("ppda", window=w, group_by="team")
        r_opp_ppda = pregame_rolling_mean("opp_ppda", window=w, group_by="team")
        r_deep = pregame_rolling_mean("deep_completions", window=w, group_by="team")
        r_corner_share = _pregame_share_expr(
            "corners_for", "corners_against", window=w, group_by="team"
        )
        r_finishing = pregame_rolling_mean("_match_finishing_above_xg", window=w, group_by="team")
        r_ga_luck = pregame_rolling_mean("_match_ga_above_xga", window=w, group_by="team")
        r_op_xgf = pregame_rolling_mean("open_play_xgf", window=w, group_by="team")
        r_op_xga = pregame_rolling_mean("open_play_xga", window=w, group_by="team")
        r_sp_xgf = pregame_rolling_mean("set_piece_xgf", window=w, group_by="team")
        r_sp_xga = pregame_rolling_mean("set_piece_xga", window=w, group_by="team")
        r_tied_npxgd = pregame_rolling_mean("tied_state_npxgd_per_90", window=w, group_by="team")
        r_box_share = pregame_rolling_mean("box_shot_share", window=w, group_by="team")
        r_tb_xgf = pregame_rolling_mean("throughball_xgf", window=w, group_by="team")

        shrunk_npxgf = bayesian_shrinkage(r_npxgf, n_w, pl.col("_p_npxgf"), shrinkage_k)
        shrunk_npxga = bayesian_shrinkage(r_npxga, n_w, pl.col("_p_npxga"), shrinkage_k)
        shrunk_op_xgf = bayesian_shrinkage(r_op_xgf, n_w, PRIOR_OPEN_PLAY_XGF_PER_90, shrinkage_k)
        shrunk_op_xga = bayesian_shrinkage(r_op_xga, n_w, PRIOR_OPEN_PLAY_XGA_PER_90, shrinkage_k)
        shrunk_sp_xgf = bayesian_shrinkage(r_sp_xgf, n_w, PRIOR_SET_PIECE_XGF_PER_90, shrinkage_k)
        shrunk_sp_xga = bayesian_shrinkage(r_sp_xga, n_w, PRIOR_SET_PIECE_XGA_PER_90, shrinkage_k)

        exprs.extend(
            [
                shrunk_npxgf.alias(f"npxgf_per_90_w{w}"),
                shrunk_npxga.alias(f"npxga_per_90_w{w}"),
                (shrunk_npxgf - shrunk_npxga).alias(f"npxgd_per_90_w{w}"),
                shrunk_op_xgf.alias(f"open_play_xgf_per_90_w{w}"),
                shrunk_op_xga.alias(f"open_play_xga_per_90_w{w}"),
                (shrunk_op_xgf - shrunk_op_xga).alias(f"open_play_npxgd_per_90_w{w}"),
                shrunk_sp_xgf.alias(f"set_piece_xgf_per_90_w{w}"),
                shrunk_sp_xga.alias(f"set_piece_xga_per_90_w{w}"),
                bayesian_shrinkage(r_tied_npxgd, n_w, 0.0, shrinkage_k).alias(
                    f"tied_state_npxgd_per_90_w{w}"
                ),
                bayesian_shrinkage(r_box_share, n_w, PRIOR_BOX_SHOT_SHARE, shrinkage_k).alias(
                    f"box_shot_share_w{w}"
                ),
                bayesian_shrinkage(r_tb_xgf, n_w, PRIOR_THROUGHBALL_XGF_PER_90, shrinkage_k).alias(
                    f"throughball_xgf_per_90_w{w}"
                ),
                bayesian_shrinkage(r_npxgf_pct, n_w, pl.col("_p_share"), shrinkage_k).alias(
                    f"npxgf_pct_w{w}"
                ),
                bayesian_shrinkage(r_xgf, n_w, pl.col("_p_xgf"), shrinkage_k).alias(
                    f"xgf_per_90_w{w}"
                ),
                bayesian_shrinkage(r_xga, n_w, pl.col("_p_xga"), shrinkage_k).alias(
                    f"xga_per_90_w{w}"
                ),
                bayesian_shrinkage(r_shots, n_w, PRIOR_SHOTS_PER_90, shrinkage_k).alias(
                    f"shots_per_90_w{w}"
                ),
                bayesian_shrinkage(r_sot_against, n_w, PRIOR_SOT_AGAINST_PER_90, shrinkage_k).alias(
                    f"sot_against_per_90_w{w}"
                ),
                bayesian_shrinkage(r_sot_pct, n_w, PRIOR_SOT_PCT, shrinkage_k).alias(
                    f"sot_pct_w{w}"
                ),
                bayesian_shrinkage(r_ppda, n_w, PRIOR_PPDA, shrinkage_k).alias(f"ppda_w{w}"),
                bayesian_shrinkage(r_opp_ppda, n_w, PRIOR_PPDA, shrinkage_k).alias(
                    f"opp_ppda_w{w}"
                ),
                bayesian_shrinkage(r_deep, n_w, PRIOR_DEEP_PER_90, shrinkage_k).alias(
                    f"deep_completions_per_90_w{w}"
                ),
                bayesian_shrinkage(r_corner_share, n_w, PRIOR_SHARE_50, shrinkage_k).alias(
                    f"corner_share_w{w}"
                ),
                bayesian_shrinkage(r_finishing, n_w, 0.0, shrinkage_k).alias(
                    f"finishing_above_xg_w{w}"
                ),
                bayesian_shrinkage(r_ga_luck, n_w, 0.0, shrinkage_k).alias(
                    f"defensive_ga_above_xga_w{w}"
                ),
            ]
        )

    n_tot = pl.col("team_prior_matches_n")
    raw_ewma_npxgf_share = pregame_ewma("_match_npxgf_share", alpha=ewma_alpha, group_by="team")
    raw_ewma_npxgd = pregame_ewma("_match_npxgd", alpha=ewma_alpha, group_by="team")
    exprs.extend(
        [
            bayesian_shrinkage(raw_ewma_npxgf_share, n_tot, pl.col("_p_share"), shrinkage_k).alias(
                "npxgf_pct_ewma"
            ),
            bayesian_shrinkage(
                raw_ewma_npxgd,
                n_tot,
                pl.col("_p_npxgf") - pl.col("_p_npxga"),
                shrinkage_k,
            ).alias("npxgd_per_90_ewma"),
        ]
    )

    ref_w = 10 if 10 in windows else windows[0]
    result = with_derived.with_columns(exprs).with_columns(
        pl.col(f"npxgf_per_90_w{ref_w}").alias("npxgf_per_90"),
        pl.col(f"npxga_per_90_w{ref_w}").alias("npxga_per_90"),
        pl.col(f"npxgd_per_90_w{ref_w}").alias("npxgd_per_90"),
        pl.col(f"xga_per_90_w{ref_w}").alias("xga_per_90"),
        pl.col(f"sot_against_per_90_w{ref_w}").alias("sot_against_per_90"),
        pl.col(f"ppda_w{ref_w}").alias("ppda_rolling"),
        pl.col(f"opp_ppda_w{ref_w}").alias("opp_ppda_rolling"),
    )

    return result.drop(
        [
            "_match_npxgd",
            "_match_npxgf_share",
            "_match_finishing_above_xg",
            "_match_ga_above_xga",
            "_p_npxgf",
            "_p_npxga",
            "_p_xgf",
            "_p_xga",
            "_p_share",
            "open_play_xgf",
            "open_play_xga",
            "set_piece_xgf",
            "set_piece_xga",
            "tied_state_npxgd_per_90",
            "box_shot_share",
            "throughball_xgf",
        ]
    )


def _compute_postmatch_team_features(team_feat: pl.DataFrame) -> pl.DataFrame:
    """Compute post-match rolling state so upcoming `FUT` fixtures inherit latest team form."""
    sorted_t = sort_events_chronologically(team_feat, extra_tiebreakers=["team"])
    w = 10
    f_npxg = (
        pl.col("npxgf").cast(pl.Float64).rolling_mean(window_size=w, min_samples=1).over("team")
    )
    a_npxg = (
        pl.col("npxga").cast(pl.Float64).rolling_mean(window_size=w, min_samples=1).over("team")
    )
    f_xg = pl.col("xgf").cast(pl.Float64).rolling_mean(window_size=w, min_samples=1).over("team")
    a_xg = pl.col("xga").cast(pl.Float64).rolling_mean(window_size=w, min_samples=1).over("team")
    n_post = (
        pl.col("npxgf")
        .is_not_null()
        .cast(pl.Int32)
        .rolling_sum(window_size=w, min_samples=1)
        .over("team")
    )
    post_npxgf = bayesian_shrinkage(f_npxg, n_post, PRIOR_NPXGF_PER_90, SOCCER_TEAM_SHRINKAGE_K)
    post_npxga = bayesian_shrinkage(a_npxg, n_post, PRIOR_NPXGA_PER_90, SOCCER_TEAM_SHRINKAGE_K)
    return sorted_t.with_columns(
        post_npxgf.alias("_post_npxgf_per_90_w10"),
        post_npxga.alias("_post_npxga_per_90_w10"),
        (post_npxgf - post_npxga).alias("_post_npxgd_per_90_w10"),
        bayesian_shrinkage(
            f_npxg / pl.when((f_npxg + a_npxg) > 0.0).then(f_npxg + a_npxg).otherwise(1.0),
            n_post,
            PRIOR_SHARE_50,
            SOCCER_TEAM_SHRINKAGE_K,
        ).alias("_post_npxgf_pct_w10"),
        bayesian_shrinkage(f_xg, n_post, PRIOR_XGF_PER_90, SOCCER_TEAM_SHRINKAGE_K).alias(
            "_post_xgf_per_90_w10"
        ),
        bayesian_shrinkage(a_xg, n_post, PRIOR_XGA_PER_90, SOCCER_TEAM_SHRINKAGE_K).alias(
            "_post_xga_per_90_w10"
        ),
    )


def build_soccer_stage1_match_features(
    matches_df: pl.DataFrame,
    team_match_logs: pl.DataFrame,
    outfield_match_logs: pl.DataFrame | None = None,
    gk_match_logs: pl.DataFrame | None = None,
    *,
    all_comp_schedule_df: pl.DataFrame | None = None,
    team_shot_micro_logs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Assemble the complete Stage 1 pre-match EPL feature matrix (`Home`, `Away`, and `Diff`).

    Combines 3-way (`1X2`) Elo with promotion/relegation priors, team rolling `npxG`/`xG`/`PPDA`/
    deep completions/finishing regression, open-play vs. set-piece xG splits, cross-competition
    schedule congestion/UEFA travel, starting GK `GSAA/90`, and minutes-weighted squad strength.
    """
    from sports_ml.sports.soccer.features.goalkeeper import (
        PRIOR_GK_CLEAN_SHEET_RATE,
        PRIOR_GK_GSAA_PER_90,
        PRIOR_GK_SAVE_PCT,
        GoalkeeperStarterEloHook,
        build_goalkeeper_rolling_features,
        resolve_team_starting_gk_features,
    )
    from sports_ml.sports.soccer.features.roster import (
        PRIOR_NPXG_PER_90_FWD,
        PRIOR_NPXG_PER_90_MID,
        PRIOR_XA_PER_90_FWD,
        PRIOR_XA_PER_90_MID,
        PRIOR_XGBUILDUP_PER_90_DEF,
        PRIOR_XGCHAIN_PER_90_MID,
        build_soccer_roster_strength_features,
    )

    working_matches = matches_df
    if "competition_type" in working_matches.columns:
        working_matches = working_matches.filter(
            pl.col("competition_type").map_elements(
                normalize_competition_type, return_dtype=pl.Utf8
            )
            == "EPL"
        )
    working_matches = working_matches.with_columns(
        pl.col("home_team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
        pl.col("away_team").map_elements(normalize_soccer_club, return_dtype=pl.Utf8),
    )
    if "match_id" in working_matches.columns and "event_id" not in working_matches.columns:
        working_matches = working_matches.with_columns(
            pl.col("match_id").cast(pl.Int64).alias("event_id")
        )
    event_col = "match_id" if "match_id" in working_matches.columns else "event_id"

    gk_feat = (
        build_goalkeeper_rolling_features(gk_match_logs)
        if gk_match_logs is not None and not gk_match_logs.is_empty()
        else pl.DataFrame()
    )
    starter_hook = GoalkeeperStarterEloHook(gk_feat) if not gk_feat.is_empty() else None
    elo_df = PairwiseEloEngine(config=SOCCER_ELO_CONFIG, starter_hook=starter_hook).compute_ratings(
        working_matches
    )

    team_feat = _compute_postmatch_team_features(
        build_soccer_team_rolling_features(
            team_match_logs,
            all_comp_schedule_df=all_comp_schedule_df,
            team_shot_micro_logs=team_shot_micro_logs,
        )
    )
    t_ev_col = "match_id" if "match_id" in team_feat.columns else "event_id"

    core_team_cols = [
        "npxgf_per_90_w10",
        "npxga_per_90_w10",
        "npxgd_per_90_w10",
        "open_play_xgf_per_90_w10",
        "open_play_xga_per_90_w10",
        "open_play_npxgd_per_90_w10",
        "set_piece_xgf_per_90_w10",
        "set_piece_xga_per_90_w10",
        "tied_state_npxgd_per_90_w10",
        "box_shot_share_w10",
        "throughball_xgf_per_90_w10",
        "npxgf_pct_w10",
        "npxgf_pct_ewma",
        "npxgd_per_90_ewma",
        "xgf_per_90_w10",
        "xga_per_90_w10",
        "shots_per_90_w10",
        "sot_pct_w10",
        "ppda_w10",
        "opp_ppda_w10",
        "deep_completions_per_90_w10",
        "corner_share_w10",
        "finishing_above_xg_w10",
        "defensive_ga_above_xga_w10",
        "rest_days",
        "true_all_comp_rest_days",
        "is_midweek_turnaround",
        "had_midweek_european_match",
        "had_midweek_away_european_match",
        "european_travel_km_last_7d",
        "matches_last_14d",
        "all_comp_matches_last_14d",
        "all_comp_matches_last_28d",
        "cup_extra_time_last_7d",
        "travel_km_prev_game",
        "is_promoted_team",
    ]
    avail_cols = [c for c in core_team_cols if c in team_feat.columns]

    home_lookup = team_feat.select(
        [
            pl.col(t_ev_col).alias(event_col),
            pl.col("team").alias("home_team"),
            *[pl.col(c).alias(f"{c}_home") for c in avail_cols],
        ]
    ).unique(subset=[event_col, "home_team"], keep="last")
    away_lookup = team_feat.select(
        [
            pl.col(t_ev_col).alias(event_col),
            pl.col("team").alias("away_team"),
            *[pl.col(c).alias(f"{c}_away") for c in avail_cols],
        ]
    ).unique(subset=[event_col, "away_team"], keep="last")

    matchup = elo_df.join(home_lookup, on=[event_col, "home_team"], how="left").join(
        away_lookup, on=[event_col, "away_team"], how="left"
    )

    # Causal backward join_asof for upcoming FUT fixtures not yet in team_match_logs
    if not team_feat.is_empty():
        asof_source = team_feat.sort("start_time_utc").select(
            [
                pl.col("start_time_utc"),
                pl.col("team"),
                pl.col("_post_npxgf_per_90_w10").alias("npxgf_per_90_w10"),
                pl.col("_post_npxga_per_90_w10").alias("npxga_per_90_w10"),
                pl.col("_post_npxgd_per_90_w10").alias("npxgd_per_90_w10"),
                pl.col("_post_npxgf_pct_w10").alias("npxgf_pct_w10"),
                pl.col("npxgf_pct_ewma"),
                pl.col("npxgd_per_90_ewma"),
                pl.col("_post_xgf_per_90_w10").alias("xgf_per_90_w10"),
                pl.col("_post_xga_per_90_w10").alias("xga_per_90_w10"),
                pl.col("shots_per_90_w10"),
                pl.col("sot_pct_w10"),
                pl.col("ppda_w10"),
                pl.col("opp_ppda_w10"),
                pl.col("deep_completions_per_90_w10"),
                pl.col("corner_share_w10"),
                pl.col("finishing_above_xg_w10"),
                pl.col("defensive_ga_above_xga_w10"),
                pl.col("is_promoted_team"),
            ]
        )
        asof_cols = [c for c in asof_source.columns if c not in {"start_time_utc", "team"}]
        matchup = (
            matchup.sort(["home_team", "start_time_utc"])
            .join_asof(
                asof_source.sort(["team", "start_time_utc"]).rename(
                    {"team": "home_team", **{c: f"_asof_{c}_home" for c in asof_cols}}
                ),
                on="start_time_utc",
                by="home_team",
                strategy="backward",
                check_sortedness=False,
            )
            .sort(["away_team", "start_time_utc"])
            .join_asof(
                asof_source.sort(["team", "start_time_utc"]).rename(
                    {"team": "away_team", **{c: f"_asof_{c}_away" for c in asof_cols}}
                ),
                on="start_time_utc",
                by="away_team",
                strategy="backward",
                check_sortedness=False,
            )
            .sort("start_time_utc")
        )
        fill_exprs: list[pl.Expr] = []
        for c in asof_cols:
            for side in ("home", "away"):
                col_s = f"{c}_{side}"
                asof_s = f"_asof_{c}_{side}"
                if col_s in matchup.columns and asof_s in matchup.columns:
                    fill_exprs.append(pl.col(col_s).fill_null(pl.col(asof_s)).alias(col_s))
        matchup = matchup.with_columns(fill_exprs).drop(
            [f"_asof_{c}_{side}" for c in asof_cols for side in ("home", "away")]
        )

    # Ensure non-null defaults on cold start
    default_team_fills: list[tuple[str, float | int | bool]] = [
        ("npxgf_per_90_w10", PRIOR_NPXGF_PER_90),
        ("npxga_per_90_w10", PRIOR_NPXGA_PER_90),
        ("npxgd_per_90_w10", 0.0),
        ("open_play_xgf_per_90_w10", PRIOR_OPEN_PLAY_XGF_PER_90),
        ("open_play_xga_per_90_w10", PRIOR_OPEN_PLAY_XGA_PER_90),
        ("open_play_npxgd_per_90_w10", 0.0),
        ("set_piece_xgf_per_90_w10", PRIOR_SET_PIECE_XGF_PER_90),
        ("set_piece_xga_per_90_w10", PRIOR_SET_PIECE_XGA_PER_90),
        ("tied_state_npxgd_per_90_w10", 0.0),
        ("box_shot_share_w10", PRIOR_BOX_SHOT_SHARE),
        ("throughball_xgf_per_90_w10", PRIOR_THROUGHBALL_XGF_PER_90),
        ("npxgf_pct_w10", PRIOR_SHARE_50),
        ("npxgf_pct_ewma", PRIOR_SHARE_50),
        ("npxgd_per_90_ewma", 0.0),
        ("xgf_per_90_w10", PRIOR_XGF_PER_90),
        ("xga_per_90_w10", PRIOR_XGA_PER_90),
        ("shots_per_90_w10", PRIOR_SHOTS_PER_90),
        ("sot_pct_w10", PRIOR_SOT_PCT),
        ("ppda_w10", PRIOR_PPDA),
        ("opp_ppda_w10", PRIOR_PPDA),
        ("deep_completions_per_90_w10", PRIOR_DEEP_PER_90),
        ("corner_share_w10", PRIOR_SHARE_50),
        ("finishing_above_xg_w10", 0.0),
        ("defensive_ga_above_xga_w10", 0.0),
        ("rest_days", 6),
        ("true_all_comp_rest_days", 7.0),
        ("is_midweek_turnaround", False),
        ("had_midweek_european_match", 0),
        ("had_midweek_away_european_match", 0),
        ("european_travel_km_last_7d", 0.0),
        ("matches_last_14d", 2),
        ("all_comp_matches_last_14d", 2),
        ("all_comp_matches_last_28d", 3),
        ("cup_extra_time_last_7d", 0),
        ("travel_km_prev_game", 0.0),
        ("is_promoted_team", False),
    ]
    for base_c, default_v in default_team_fills:
        for side in ("home", "away"):
            col_name = f"{base_c}_{side}"
            if col_name in matchup.columns:
                matchup = matchup.with_columns(pl.col(col_name).fill_null(default_v))
            else:
                matchup = matchup.with_columns(pl.lit(default_v).alias(col_name))

    # Join starting goalkeeper features
    if not gk_feat.is_empty():
        starters = resolve_team_starting_gk_features(gk_feat)
        g_ev_col = "match_id" if "match_id" in starters.columns else "event_id"
        gk_cols = [
            "gsaa_per_90_shrunk",
            "save_pct_shrunk",
            "clean_sheet_rate_shrunk",
            "gk_days_rest",
        ]
        h_gk = starters.select(
            [
                pl.col(g_ev_col).alias(event_col),
                pl.col("team").alias("home_team"),
                *[pl.col(c).alias(f"{c}_home") for c in gk_cols],
            ]
        )
        a_gk = starters.select(
            [
                pl.col(g_ev_col).alias(event_col),
                pl.col("team").alias("away_team"),
                *[pl.col(c).alias(f"{c}_away") for c in gk_cols],
            ]
        )
        matchup = matchup.join(h_gk, on=[event_col, "home_team"], how="left").join(
            a_gk, on=[event_col, "away_team"], how="left"
        )
    else:
        for side in ("home", "away"):
            matchup = matchup.with_columns(
                pl.lit(PRIOR_GK_GSAA_PER_90).alias(f"gsaa_per_90_shrunk_{side}"),
                pl.lit(PRIOR_GK_SAVE_PCT).alias(f"save_pct_shrunk_{side}"),
                pl.lit(PRIOR_GK_CLEAN_SHEET_RATE).alias(f"clean_sheet_rate_shrunk_{side}"),
                pl.lit(7, dtype=pl.Int32).alias(f"gk_days_rest_{side}"),
            )

    matchup = matchup.with_columns(
        pl.col("gsaa_per_90_shrunk_home").fill_null(PRIOR_GK_GSAA_PER_90),
        pl.col("gsaa_per_90_shrunk_away").fill_null(PRIOR_GK_GSAA_PER_90),
        pl.col("save_pct_shrunk_home").fill_null(PRIOR_GK_SAVE_PCT),
        pl.col("save_pct_shrunk_away").fill_null(PRIOR_GK_SAVE_PCT),
        pl.col("clean_sheet_rate_shrunk_home").fill_null(PRIOR_GK_CLEAN_SHEET_RATE),
        pl.col("clean_sheet_rate_shrunk_away").fill_null(PRIOR_GK_CLEAN_SHEET_RATE),
        pl.col("gk_days_rest_home").fill_null(7),
        pl.col("gk_days_rest_away").fill_null(7),
    )

    # Join active squad minutes-weighted features
    if outfield_match_logs is not None and not outfield_match_logs.is_empty():
        roster_df = build_soccer_roster_strength_features(
            outfield_match_logs, all_comp_schedule_df=all_comp_schedule_df
        )
        r_ev_col = "match_id" if "match_id" in roster_df.columns else "event_id"
        r_cols = [
            "fwd_weighted_npxg_per_90",
            "fwd_weighted_xa_per_90",
            "mid_weighted_npxg_per_90",
            "mid_weighted_xa_per_90",
            "mid_weighted_xg_chain_per_90",
            "def_weighted_xg_buildup_per_90",
            "missing_regular_minutes",
            "active_xi_14d_all_comp_minutes",
        ]
        h_ros = roster_df.select(
            [
                pl.col(r_ev_col).alias(event_col),
                pl.col("team").alias("home_team"),
                *[pl.col(c).alias(f"{c}_home") for c in r_cols],
            ]
        )
        a_ros = roster_df.select(
            [
                pl.col(r_ev_col).alias(event_col),
                pl.col("team").alias("away_team"),
                *[pl.col(c).alias(f"{c}_away") for c in r_cols],
            ]
        )
        matchup = matchup.join(h_ros, on=[event_col, "home_team"], how="left").join(
            a_ros, on=[event_col, "away_team"], how="left"
        )
    else:
        for side in ("home", "away"):
            matchup = matchup.with_columns(
                pl.lit(PRIOR_NPXG_PER_90_FWD).alias(f"fwd_weighted_npxg_per_90_{side}"),
                pl.lit(PRIOR_XA_PER_90_FWD).alias(f"fwd_weighted_xa_per_90_{side}"),
                pl.lit(PRIOR_NPXG_PER_90_MID).alias(f"mid_weighted_npxg_per_90_{side}"),
                pl.lit(PRIOR_XA_PER_90_MID).alias(f"mid_weighted_xa_per_90_{side}"),
                pl.lit(PRIOR_XGCHAIN_PER_90_MID).alias(f"mid_weighted_xg_chain_per_90_{side}"),
                pl.lit(PRIOR_XGBUILDUP_PER_90_DEF).alias(f"def_weighted_xg_buildup_per_90_{side}"),
                pl.lit(0.0).alias(f"missing_regular_minutes_{side}"),
                pl.lit(0.0).alias(f"active_xi_14d_all_comp_minutes_{side}"),
            )

    matchup = matchup.with_columns(
        pl.col("fwd_weighted_npxg_per_90_home").fill_null(PRIOR_NPXG_PER_90_FWD),
        pl.col("fwd_weighted_npxg_per_90_away").fill_null(PRIOR_NPXG_PER_90_FWD),
        pl.col("fwd_weighted_xa_per_90_home").fill_null(PRIOR_XA_PER_90_FWD),
        pl.col("fwd_weighted_xa_per_90_away").fill_null(PRIOR_XA_PER_90_FWD),
        pl.col("mid_weighted_npxg_per_90_home").fill_null(PRIOR_NPXG_PER_90_MID),
        pl.col("mid_weighted_npxg_per_90_away").fill_null(PRIOR_NPXG_PER_90_MID),
        pl.col("mid_weighted_xa_per_90_home").fill_null(PRIOR_XA_PER_90_MID),
        pl.col("mid_weighted_xa_per_90_away").fill_null(PRIOR_XA_PER_90_MID),
        pl.col("mid_weighted_xg_chain_per_90_home").fill_null(PRIOR_XGCHAIN_PER_90_MID),
        pl.col("mid_weighted_xg_chain_per_90_away").fill_null(PRIOR_XGCHAIN_PER_90_MID),
        pl.col("def_weighted_xg_buildup_per_90_home").fill_null(PRIOR_XGBUILDUP_PER_90_DEF),
        pl.col("def_weighted_xg_buildup_per_90_away").fill_null(PRIOR_XGBUILDUP_PER_90_DEF),
        pl.col("missing_regular_minutes_home").fill_null(0.0),
        pl.col("missing_regular_minutes_away").fill_null(0.0),
        pl.col("active_xi_14d_all_comp_minutes_home").fill_null(0.0),
        pl.col("active_xi_14d_all_comp_minutes_away").fill_null(0.0),
    )

    # If all_comp_schedule_df is provided, recompute cross-competition fatigue directly on
    # `working_matches` for both home and away sides so upcoming FUT matches (or minimal
    # test fixtures) always reflect exact cross-competition rest, UEFA travel, and density.
    if all_comp_schedule_df is not None and not all_comp_schedule_df.is_empty():
        h_sched_in = working_matches.select(
            pl.col(event_col),
            pl.col("season"),
            pl.col("match_date")
            if "match_date" in working_matches.columns
            else pl.col("start_time_utc").dt.date().alias("match_date"),
            pl.col("start_time_utc"),
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent"),
            pl.lit(True).alias("is_home"),
        )
        a_sched_in = working_matches.select(
            pl.col(event_col),
            pl.col("season"),
            pl.col("match_date")
            if "match_date" in working_matches.columns
            else pl.col("start_time_utc").dt.date().alias("match_date"),
            pl.col("start_time_utc"),
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent"),
            pl.lit(False).alias("is_home"),
        )
        cc_all = compute_cross_competition_fatigue_features(
            pl.concat([h_sched_in, a_sched_in], how="vertical"),
            all_comp_schedule_df,
        )
        cc_cols = [
            "true_all_comp_rest_days",
            "had_midweek_european_match",
            "had_midweek_away_european_match",
            "european_travel_km_last_7d",
            "all_comp_matches_last_14d",
            "all_comp_matches_last_28d",
            "cup_extra_time_last_7d",
        ]
        h_cc = cc_all.select(
            [
                pl.col(event_col),
                pl.col("team").alias("home_team"),
                *[pl.col(c).alias(f"_cc_{c}_home") for c in cc_cols],
            ]
        ).unique(subset=[event_col, "home_team"], keep="last")
        a_cc = cc_all.select(
            [
                pl.col(event_col),
                pl.col("team").alias("away_team"),
                *[pl.col(c).alias(f"_cc_{c}_away") for c in cc_cols],
            ]
        ).unique(subset=[event_col, "away_team"], keep="last")
        matchup = (
            matchup.join(h_cc, on=[event_col, "home_team"], how="left")
            .join(a_cc, on=[event_col, "away_team"], how="left")
            .with_columns(
                [
                    pl.coalesce([pl.col(f"_cc_{c}_{s}"), pl.col(f"{c}_{s}")]).alias(f"{c}_{s}")
                    for c in cc_cols
                    for s in ("home", "away")
                ]
            )
            .drop([f"_cc_{c}_{s}" for c in cc_cols for s in ("home", "away")])
            .with_columns(
                pl.col("true_all_comp_rest_days_home")
                .round(0)
                .cast(pl.Int32)
                .alias("rest_days_home"),
                pl.col("true_all_comp_rest_days_away")
                .round(0)
                .cast(pl.Int32)
                .alias("rest_days_away"),
                (pl.col("true_all_comp_rest_days_home") <= 3.5).alias("is_midweek_turnaround_home"),
                (pl.col("true_all_comp_rest_days_away") <= 3.5).alias("is_midweek_turnaround_away"),
                pl.col("all_comp_matches_last_14d_home").alias("matches_last_14d_home"),
                pl.col("all_comp_matches_last_14d_away").alias("matches_last_14d_away"),
            )
        )

    # Matchup differentials (Home - Away)
    matchup = matchup.with_columns(
        (pl.col("npxgd_per_90_w10_home") - pl.col("npxgd_per_90_w10_away")).alias(
            "npxgd_per_90_w10_diff"
        ),
        (
            pl.col("open_play_npxgd_per_90_w10_home") - pl.col("open_play_npxgd_per_90_w10_away")
        ).alias("diff_open_play_npxgd_90"),
        (
            (pl.col("set_piece_xgf_per_90_w10_home") - pl.col("set_piece_xga_per_90_w10_away"))
            - (pl.col("set_piece_xgf_per_90_w10_away") - pl.col("set_piece_xga_per_90_w10_home"))
        ).alias("diff_set_piece_xg_edge"),
        (pl.col("npxgf_pct_w10_home") - pl.col("npxgf_pct_w10_away")).alias("npxgf_pct_w10_diff"),
        (pl.col("ppda_w10_away") - pl.col("ppda_w10_home")).alias("pressing_intensity_diff"),
        (
            pl.col("deep_completions_per_90_w10_home") - pl.col("deep_completions_per_90_w10_away")
        ).alias("deep_completions_per_90_w10_diff"),
        (pl.col("rest_days_home") - pl.col("rest_days_away")).alias("rest_advantage"),
        (pl.col("gsaa_per_90_shrunk_home") - pl.col("gsaa_per_90_shrunk_away")).alias(
            "gk_gsaa_per_90_diff"
        ),
        (pl.col("fwd_weighted_npxg_per_90_home") - pl.col("fwd_weighted_npxg_per_90_away")).alias(
            "fwd_npxg_per_90_diff"
        ),
        (
            pl.col("mid_weighted_xg_chain_per_90_home")
            - pl.col("mid_weighted_xg_chain_per_90_away")
        ).alias("mid_xg_chain_per_90_diff"),
    )

    return sort_events_chronologically(matchup)
