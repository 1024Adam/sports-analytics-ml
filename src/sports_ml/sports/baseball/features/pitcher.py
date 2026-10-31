"""Starting pitcher pre-game rolling, shrunk, and workload features (`features/pitcher.py`).

Enforces a strict temporal firewall over `(start_time_utc, event_seq, game_id)` so
that even in a same-day doubleheader (`event_seq = 1, 2`), Game 1 and Game 2 never
leak current or future pitching stats. Supports cross-league `NPB <-> MLB` priors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Final

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage, shrink_scalar
from sports_ml.core.temporal import (
    pregame_ewma,
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_rolling_sum,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.baseball.constants import (
    SHRINKAGE_K_SP_BF,
    BaseballLeague,
    get_league_sabermetric_priors,
    resolve_team_league,
)
from sports_ml.sports.baseball.features.translation import (
    TranslatedPitcherPrior,
    get_rookie_pitcher_prior,
)

DEFAULT_SP_WINDOWS: Final[tuple[int, ...]] = (5, 10)
MAX_SP_REST_DAYS: Final[int] = 14
DEFAULT_SP_PITCH_CEILING: Final[float] = 94.0
DEFAULT_SP_OUTS_CEILING: Final[float] = 17.5


def _ensure_pitcher_log_columns(df: pl.DataFrame) -> pl.DataFrame:
    working = df
    if "game_id" in working.columns and "event_id" not in working.columns:
        working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
    if "league" not in working.columns and "team" in working.columns:
        working = working.with_columns(
            pl.col("team")
            .map_elements(lambda t: resolve_team_league(t) or "MLB", return_dtype=pl.Utf8)
            .alias("league")
        )

    defaults: list[tuple[str, Any, pl.DataType]] = [
        ("pitch_hand", "R", pl.Utf8()),
        ("is_starter", True, pl.Boolean()),
        ("outs_pitched", 18, pl.Int16()),
        ("batters_faced", 24, pl.Int16()),
        ("pitches_thrown", 92, pl.Int16()),
        ("strikes_thrown", 60, pl.Int16()),
        ("hits_allowed", 5, pl.Int16()),
        ("runs_allowed", 3, pl.Int16()),
        ("er_allowed", 3, pl.Int16()),
        ("hr_allowed", 1, pl.Int16()),
        ("bb_allowed", 2, pl.Int16()),
        ("ibb_allowed", 0, pl.Int16()),
        ("hbp_allowed", 0, pl.Int16()),
        ("k_recorded", 6, pl.Int16()),
        ("gb_count", 7, pl.Int16()),
        ("fb_count", 6, pl.Int16()),
        ("ld_count", 3, pl.Int16()),
        ("xwoba_against", None, pl.Float64()),
        ("swstr_pct", None, pl.Float64()),
        ("avg_fastball_velo", None, pl.Float64()),
    ]
    missing_exprs = [
        pl.lit(val, dtype=dtype).alias(col)
        for col, val, dtype in defaults
        if col not in working.columns
    ]
    return working.with_columns(missing_exprs) if missing_exprs else working


def _compute_sp_rest_days_column(
    sorted_df: pl.DataFrame,
    *,
    max_rest_days: int = MAX_SP_REST_DAYS,
) -> pl.DataFrame:
    if sorted_df.is_empty():
        return sorted_df.with_columns(pl.lit(None, dtype=pl.Int32).alias("sp_days_rest"))

    rows = sorted_df.to_dicts()
    rest_out = [max_rest_days] * len(rows)
    last_seen: dict[tuple[int, int], date] = {}

    for idx, row in enumerate(rows):
        pid = int(row["player_id"])
        season = int(row.get("season") or 0)
        start_utc: datetime = row["start_time_utc"]
        ev_date: date = (
            row["game_date"]
            if "game_date" in row and row["game_date"] is not None
            else start_utc.date()
        )
        key = (pid, season)
        prev_d = last_seen.get(key)
        if prev_d is None:
            rest_out[idx] = max_rest_days
        else:
            rest_out[idx] = min(max(0, (ev_date - prev_d).days), max_rest_days)
        last_seen[key] = ev_date

    return sorted_df.with_columns(pl.Series("sp_days_rest", rest_out, dtype=pl.Int32))


def build_starting_pitcher_rolling_features(
    pitcher_logs_df: pl.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_SP_WINDOWS,
    shrinkage_k_bf: float = SHRINKAGE_K_SP_BF,
    cross_league_priors: Mapping[int, TranslatedPitcherPrior] | None = None,
) -> pl.DataFrame:
    """Build strictly pre-game `.shift(1)` rolling and shrunk starting pitcher features."""
    prepared = _ensure_pitcher_log_columns(pitcher_logs_df)
    sorted_df = sort_events_chronologically(prepared, extra_tiebreakers=["player_id"])
    with_rest = _compute_sp_rest_days_column(sorted_df)

    # Resolve per-player priors (cross-league translated or league baseline)
    rows_prior_fip: list[float] = []
    rows_prior_xfip: list[float] = []
    rows_prior_k_pct: list[float] = []
    rows_prior_bb_pct: list[float] = []
    rows_prior_gb_pct: list[float] = []
    rows_fip_const: list[float] = []
    rows_hr_fb_prior: list[float] = []

    for row in with_rest.iter_rows(named=True):
        pid = int(row["player_id"])
        lg_str = str(row.get("league") or "MLB").upper()
        lg_priors = get_league_sabermetric_priors("NPB" if lg_str == "NPB" else "MLB")
        custom_p = cross_league_priors.get(pid) if cross_league_priors else None
        if custom_p is not None:
            rows_prior_fip.append(custom_p.fip_prior)
            rows_prior_xfip.append(custom_p.xfip_prior)
            rows_prior_k_pct.append(custom_p.k_pct_prior)
            rows_prior_bb_pct.append(custom_p.bb_pct_prior)
            rows_prior_gb_pct.append(custom_p.gb_pct_prior)
        else:
            rows_prior_fip.append(lg_priors.lg_sp_fip)
            rows_prior_xfip.append(lg_priors.lg_sp_xfip)
            rows_prior_k_pct.append(lg_priors.lg_sp_k_pct)
            rows_prior_bb_pct.append(lg_priors.lg_sp_bb_pct)
            rows_prior_gb_pct.append(lg_priors.lg_sp_gb_pct)
        rows_fip_const.append(lg_priors.fip_constant)
        rows_hr_fb_prior.append(lg_priors.lg_hr_per_fb)

    enriched = with_rest.with_columns(
        pl.Series("_prior_fip", rows_prior_fip, dtype=pl.Float64),
        pl.Series("_prior_xfip", rows_prior_xfip, dtype=pl.Float64),
        pl.Series("_prior_k_pct", rows_prior_k_pct, dtype=pl.Float64),
        pl.Series("_prior_bb_pct", rows_prior_bb_pct, dtype=pl.Float64),
        pl.Series("_prior_gb_pct", rows_prior_gb_pct, dtype=pl.Float64),
        pl.Series("_fip_const", rows_fip_const, dtype=pl.Float64),
        pl.Series("_hr_fb_prior", rows_hr_fb_prior, dtype=pl.Float64),
        (
            pl.col("bb_allowed").cast(pl.Float64)
            - pl.col("ibb_allowed").cast(pl.Float64).fill_null(0.0)
            + pl.col("hbp_allowed").cast(pl.Float64).fill_null(0.0)
        )
        .clip(lower_bound=0.0)
        .alias("_ubb_hbp"),
        (pl.col("hits_allowed").cast(pl.Float64) + pl.col("bb_allowed").cast(pl.Float64)).alias(
            "_wh_num"
        ),
        (
            pl.col("gb_count").cast(pl.Float64)
            + pl.col("fb_count").cast(pl.Float64)
            + pl.col("ld_count").cast(pl.Float64)
        ).alias("_batted_balls"),
    ).with_columns(
        # Per-appearance FIP numerator * 3 (since IP = outs / 3):
        # FIP = 3 * (13*HR + 3*uBB_HBP - 2*K) / outs + c_FIP
        (
            13.0 * pl.col("hr_allowed").cast(pl.Float64)
            + 3.0 * pl.col("_ubb_hbp")
            - 2.0 * pl.col("k_recorded").cast(pl.Float64)
        ).alias("_fip_core_num"),
        (
            13.0 * (pl.col("fb_count").cast(pl.Float64) * pl.col("_hr_fb_prior"))
            + 3.0 * pl.col("_ubb_hbp")
            - 2.0 * pl.col("k_recorded").cast(pl.Float64)
        ).alias("_xfip_core_num"),
    )

    max_w = max(windows) if windows else 10
    cum_bf = pregame_rolling_sum("batters_faced", window=30, group_by="player_id")
    cum_outs = pregame_rolling_sum("outs_pitched", window=30, group_by="player_id")

    raw_fip_30 = pregame_exposure_rate(
        "_fip_core_num",
        "outs_pitched",
        window=30,
        group_by="player_id",
        scale_factor=3.0,
    ) + pl.col("_fip_const")
    raw_xfip_30 = pregame_exposure_rate(
        "_xfip_core_num",
        "outs_pitched",
        window=30,
        group_by="player_id",
        scale_factor=3.0,
    ) + pl.col("_fip_const")
    raw_era_30 = pregame_exposure_rate(
        "er_allowed",
        "outs_pitched",
        window=30,
        group_by="player_id",
        scale_factor=27.0,
    )
    raw_whip_30 = pregame_exposure_rate(
        "_wh_num",
        "outs_pitched",
        window=30,
        group_by="player_id",
        scale_factor=3.0,
    )
    raw_hr9_30 = pregame_exposure_rate(
        "hr_allowed",
        "outs_pitched",
        window=30,
        group_by="player_id",
        scale_factor=27.0,
    )
    raw_k_pct_30 = pregame_exposure_rate(
        "k_recorded",
        "batters_faced",
        window=30,
        group_by="player_id",
        scale_factor=1.0,
    )
    raw_bb_pct_30 = pregame_exposure_rate(
        "_ubb_hbp",
        "batters_faced",
        window=30,
        group_by="player_id",
        scale_factor=1.0,
    )
    raw_gb_pct_30 = pregame_exposure_rate(
        "gb_count",
        "_batted_balls",
        window=30,
        group_by="player_id",
        scale_factor=1.0,
    )

    exprs: list[pl.Expr] = [
        pregame_sample_count("outs_pitched", group_by="player_id").alias("sp_prior_starts_n"),
        cum_bf.fill_null(0.0).alias("sp_rolling_bf_30"),
        cum_outs.fill_null(0.0).alias("sp_rolling_outs_30"),
        bayesian_shrinkage(raw_fip_30, cum_bf, "_prior_fip", shrinkage_k_bf).alias("sp_fip_shrunk"),
        bayesian_shrinkage(raw_xfip_30, cum_bf, "_prior_xfip", shrinkage_k_bf).alias(
            "sp_xfip_shrunk"
        ),
        bayesian_shrinkage(raw_era_30, cum_bf, "_prior_fip", shrinkage_k_bf).alias("sp_era_shrunk"),
        bayesian_shrinkage(raw_k_pct_30, cum_bf, "_prior_k_pct", shrinkage_k_bf).alias(
            "sp_k_pct_shrunk"
        ),
        bayesian_shrinkage(raw_bb_pct_30, cum_bf, "_prior_bb_pct", shrinkage_k_bf).alias(
            "sp_bb_pct_shrunk"
        ),
        bayesian_shrinkage(raw_whip_30, cum_bf, 1.28, shrinkage_k_bf).alias("sp_whip_shrunk"),
        bayesian_shrinkage(raw_hr9_30, cum_bf, 1.10, shrinkage_k_bf).alias("sp_hr_per_9_shrunk"),
        bayesian_shrinkage(raw_gb_pct_30, cum_bf, "_prior_gb_pct", shrinkage_k_bf).alias(
            "sp_gb_pct_shrunk"
        ),
        bayesian_shrinkage(
            pregame_rolling_mean("pitches_thrown", window=max_w, group_by="player_id"),
            pregame_sample_count("pitches_thrown", group_by="player_id", window=max_w),
            DEFAULT_SP_PITCH_CEILING,
            3.0,
        ).alias("sp_rolling_pitches_per_start"),
        bayesian_shrinkage(
            pregame_rolling_mean("outs_pitched", window=max_w, group_by="player_id"),
            pregame_sample_count("outs_pitched", group_by="player_id", window=max_w),
            DEFAULT_SP_OUTS_CEILING,
            3.0,
        ).alias("sp_expected_outs_ceiling"),
        bayesian_shrinkage(
            pregame_rolling_mean("xwoba_against", window=max_w, group_by="player_id"),
            pregame_sample_count("xwoba_against", group_by="player_id", window=max_w),
            0.312,
            4.0,
        ).alias("sp_xwoba_against_shrunk"),
        bayesian_shrinkage(
            pregame_rolling_mean("swstr_pct", window=max_w, group_by="player_id"),
            pregame_sample_count("swstr_pct", group_by="player_id", window=max_w),
            0.110,
            4.0,
        ).alias("sp_swstr_pct_shrunk"),
        pregame_ewma("pitches_thrown", alpha=0.35, group_by="player_id")
        .fill_null(DEFAULT_SP_PITCH_CEILING)
        .alias("sp_ewma_pitches"),
        (
            pregame_rolling_mean("avg_fastball_velo", window=2, group_by="player_id")
            - pregame_rolling_mean("avg_fastball_velo", window=12, group_by="player_id")
        )
        .fill_null(0.0)
        .alias("sp_velo_delta_mph"),
    ]

    for w in windows:
        bf_w = pregame_rolling_sum("batters_faced", window=w, group_by="player_id")
        xfip_w = pregame_exposure_rate(
            "_xfip_core_num",
            "outs_pitched",
            window=w,
            group_by="player_id",
            scale_factor=3.0,
        ) + pl.col("_fip_const")
        k_w = pregame_exposure_rate(
            "k_recorded",
            "batters_faced",
            window=w,
            group_by="player_id",
            scale_factor=1.0,
        )
        bb_w = pregame_exposure_rate(
            "_ubb_hbp",
            "batters_faced",
            window=w,
            group_by="player_id",
            scale_factor=1.0,
        )
        exprs.extend(
            [
                bayesian_shrinkage(xfip_w, bf_w, "_prior_xfip", shrinkage_k_bf).alias(
                    f"sp_xfip_w{w}"
                ),
                (
                    bayesian_shrinkage(k_w, bf_w, "_prior_k_pct", shrinkage_k_bf)
                    - bayesian_shrinkage(bb_w, bf_w, "_prior_bb_pct", shrinkage_k_bf)
                ).alias(f"sp_k_minus_bb_pct_w{w}"),
            ]
        )

    out = enriched.with_columns(exprs).with_columns(
        (pl.col("sp_k_pct_shrunk") - pl.col("sp_bb_pct_shrunk")).alias("sp_k_minus_bb_pct_shrunk")
    )
    drop_temp = [
        "_prior_fip",
        "_prior_xfip",
        "_prior_k_pct",
        "_prior_bb_pct",
        "_prior_gb_pct",
        "_fip_const",
        "_hr_fb_prior",
        "_ubb_hbp",
        "_wh_num",
        "_batted_balls",
        "_fip_core_num",
        "_xfip_core_num",
    ]
    return out.drop([c for c in drop_temp if c in out.columns])


def resolve_game_starting_pitcher_features(
    games_df: pl.DataFrame,
    pitcher_logs_df: pl.DataFrame,
    *,
    cross_league_priors: Mapping[int, TranslatedPitcherPrior] | None = None,
) -> pl.DataFrame:
    """Resolve `home_sp_*` and `away_sp_*` pre-game features for every game in `games_df`.

    Strictly filters pitcher history to appearances completed at `(start_time_utc, event_seq)`
    strictly before each game's `(start_time_utc, event_seq)`.
    """
    if games_df.is_empty():
        return games_df

    sp_logs = (
        build_starting_pitcher_rolling_features(
            pitcher_logs_df, cross_league_priors=cross_league_priors
        )
        if not pitcher_logs_df.is_empty()
        else pl.DataFrame()
    )

    # Index pitcher appearances by player_id in chronological order for fast causal lookup
    sp_history_by_pid: dict[int, list[dict[str, Any]]] = {}
    if not sp_logs.is_empty():
        for r in sp_logs.iter_rows(named=True):
            pid = int(r["player_id"])
            sp_history_by_pid.setdefault(pid, []).append(r)

    out_rows: list[dict[str, Any]] = []
    for g in games_df.iter_rows(named=True):
        gid = int(g.get("game_id") or g.get("event_id") or 0)
        g_seq = int(g.get("event_seq") or 1)
        g_utc: datetime = g["start_time_utc"]
        g_date: date = g.get("game_date") or g_utc.date()
        lg_str = str(g.get("league") or "MLB").upper()
        lg_code: BaseballLeague = "NPB" if lg_str == "NPB" else "MLB"
        lg_priors = get_league_sabermetric_priors(lg_code)
        rookie_p = get_rookie_pitcher_prior(lg_code)

        row_dict: dict[str, Any] = {"game_id": gid, "event_seq": g_seq}

        for side in ("home", "away"):
            sp_id_raw = g.get(f"{side}_sp_id")
            sp_hand_raw = str(g.get(f"{side}_sp_hand") or "R").upper()
            sp_id = int(sp_id_raw) if sp_id_raw is not None else -1

            custom_p = cross_league_priors.get(sp_id) if cross_league_priors else None
            default_p = custom_p if custom_p is not None else rookie_p

            prior_apps = [
                app
                for app in sp_history_by_pid.get(sp_id, [])
                if (app["start_time_utc"], int(app.get("event_seq") or 1)) < (g_utc, g_seq)
            ]

            if not prior_apps:
                row_dict[f"{side}_sp_xfip_shrunk"] = default_p.xfip_prior
                row_dict[f"{side}_sp_fip_shrunk"] = default_p.fip_prior
                row_dict[f"{side}_sp_era_shrunk"] = default_p.fip_prior
                row_dict[f"{side}_sp_k_pct_shrunk"] = default_p.k_pct_prior
                row_dict[f"{side}_sp_bb_pct_shrunk"] = default_p.bb_pct_prior
                row_dict[f"{side}_sp_k_minus_bb_pct_shrunk"] = default_p.k_minus_bb_pct_prior
                row_dict[f"{side}_sp_whip_shrunk"] = 1.28
                row_dict[f"{side}_sp_hr_per_9_shrunk"] = 1.10
                row_dict[f"{side}_sp_gb_pct_shrunk"] = default_p.gb_pct_prior
                row_dict[f"{side}_sp_days_rest"] = MAX_SP_REST_DAYS
                row_dict[f"{side}_sp_rolling_pitches_per_start"] = DEFAULT_SP_PITCH_CEILING
                row_dict[f"{side}_sp_expected_outs_ceiling"] = DEFAULT_SP_OUTS_CEILING
                row_dict[f"{side}_sp_xwoba_against_shrunk"] = 0.312
                row_dict[f"{side}_sp_swstr_pct_shrunk"] = 0.110
                row_dict[f"{side}_sp_velo_delta_mph"] = 0.0
                row_dict[f"{side}_sp_is_lhp"] = 1.0 if sp_hand_raw == "L" else 0.0
                row_dict[f"{side}_sp_is_cross_league_prior"] = (
                    1.0 if default_p.is_cross_league_transfer else 0.0
                )
            else:
                # Recompute cumulative pre-game rates through the latest completed start
                recent = prior_apps[-30:]
                tot_bf = sum(int(a.get("batters_faced") or 0) for a in recent)
                tot_outs = sum(int(a.get("outs_pitched") or 0) for a in recent)
                tot_hr = sum(int(a.get("hr_allowed") or 0) for a in recent)
                tot_ubb_hbp = sum(
                    max(
                        0,
                        int(a.get("bb_allowed") or 0)
                        - int(a.get("ibb_allowed") or 0)
                        + int(a.get("hbp_allowed") or 0),
                    )
                    for a in recent
                )
                tot_k = sum(int(a.get("k_recorded") or 0) for a in recent)
                tot_er = sum(int(a.get("er_allowed") or 0) for a in recent)
                tot_wh = sum(
                    int(a.get("hits_allowed") or 0) + int(a.get("bb_allowed") or 0) for a in recent
                )
                tot_gb = sum(int(a.get("gb_count") or 0) for a in recent)
                tot_fb = sum(int(a.get("fb_count") or 0) for a in recent)
                tot_bb_bip = sum(
                    int(a.get("gb_count") or 0)
                    + int(a.get("fb_count") or 0)
                    + int(a.get("ld_count") or 0)
                    for a in recent
                )

                ip_30 = max(tot_outs / 3.0, 1.0 / 3.0)
                raw_fip = (
                    13.0 * tot_hr + 3.0 * tot_ubb_hbp - 2.0 * tot_k
                ) / ip_30 + lg_priors.fip_constant
                raw_xfip = (
                    13.0 * (tot_fb * lg_priors.lg_hr_per_fb) + 3.0 * tot_ubb_hbp - 2.0 * tot_k
                ) / ip_30 + lg_priors.fip_constant
                raw_era = 9.0 * tot_er / ip_30
                raw_whip = tot_wh / ip_30
                raw_hr9 = 9.0 * tot_hr / ip_30
                raw_k_pct = tot_k / max(tot_bf, 1)
                raw_bb_pct = tot_ubb_hbp / max(tot_bf, 1)
                raw_gb_pct = tot_gb / tot_bb_bip if tot_bb_bip > 0 else default_p.gb_pct_prior

                fip_s = shrink_scalar(raw_fip, tot_bf, default_p.fip_prior, SHRINKAGE_K_SP_BF)
                xfip_s = shrink_scalar(raw_xfip, tot_bf, default_p.xfip_prior, SHRINKAGE_K_SP_BF)
                era_s = shrink_scalar(raw_era, tot_bf, default_p.fip_prior, SHRINKAGE_K_SP_BF)
                k_s = shrink_scalar(raw_k_pct, tot_bf, default_p.k_pct_prior, SHRINKAGE_K_SP_BF)
                bb_s = shrink_scalar(raw_bb_pct, tot_bf, default_p.bb_pct_prior, SHRINKAGE_K_SP_BF)
                whip_s = shrink_scalar(raw_whip, tot_bf, 1.28, SHRINKAGE_K_SP_BF)
                hr9_s = shrink_scalar(raw_hr9, tot_bf, 1.10, SHRINKAGE_K_SP_BF)
                gb_s = shrink_scalar(raw_gb_pct, tot_bf, default_p.gb_pct_prior, SHRINKAGE_K_SP_BF)

                last_app = prior_apps[-1]
                last_date: date = last_app.get("game_date") or last_app["start_time_utc"].date()
                rest_days = min(max(0, (g_date - last_date).days), MAX_SP_REST_DAYS)

                last_10 = prior_apps[-10:]
                mean_pitches = sum(float(a.get("pitches_thrown") or 90.0) for a in last_10) / len(
                    last_10
                )
                mean_outs = sum(float(a.get("outs_pitched") or 17.5) for a in last_10) / len(
                    last_10
                )
                pitches_s = shrink_scalar(mean_pitches, len(last_10), DEFAULT_SP_PITCH_CEILING, 3.0)
                outs_ceil = shrink_scalar(mean_outs, len(last_10), DEFAULT_SP_OUTS_CEILING, 3.0)
                if rest_days <= 3:
                    outs_ceil = max(9.0, outs_ceil * 0.82)

                xwoba_vals = [
                    float(a["xwoba_against"]) for a in last_10 if a.get("xwoba_against") is not None
                ]
                swstr_vals = [
                    float(a["swstr_pct"]) for a in last_10 if a.get("swstr_pct") is not None
                ]
                xwoba_s = shrink_scalar(
                    sum(xwoba_vals) / len(xwoba_vals) if xwoba_vals else None,
                    len(xwoba_vals),
                    0.312,
                    4.0,
                )
                swstr_s = shrink_scalar(
                    sum(swstr_vals) / len(swstr_vals) if swstr_vals else None,
                    len(swstr_vals),
                    0.110,
                    4.0,
                )
                velo_all = [
                    float(a["avg_fastball_velo"])
                    for a in prior_apps[-12:]
                    if a.get("avg_fastball_velo") is not None
                ]
                if len(velo_all) >= 2:
                    v_short = sum(velo_all[-2:]) / len(velo_all[-2:])
                    v_base = sum(velo_all) / len(velo_all)
                    velo_delta = v_short - v_base
                else:
                    velo_delta = 0.0
                hand_eff = str(last_app.get("pitch_hand") or sp_hand_raw).upper()

                row_dict[f"{side}_sp_xfip_shrunk"] = xfip_s
                row_dict[f"{side}_sp_fip_shrunk"] = fip_s
                row_dict[f"{side}_sp_era_shrunk"] = era_s
                row_dict[f"{side}_sp_k_pct_shrunk"] = k_s
                row_dict[f"{side}_sp_bb_pct_shrunk"] = bb_s
                row_dict[f"{side}_sp_k_minus_bb_pct_shrunk"] = k_s - bb_s
                row_dict[f"{side}_sp_whip_shrunk"] = whip_s
                row_dict[f"{side}_sp_hr_per_9_shrunk"] = hr9_s
                row_dict[f"{side}_sp_gb_pct_shrunk"] = gb_s
                row_dict[f"{side}_sp_days_rest"] = rest_days
                row_dict[f"{side}_sp_rolling_pitches_per_start"] = pitches_s
                row_dict[f"{side}_sp_expected_outs_ceiling"] = outs_ceil
                row_dict[f"{side}_sp_xwoba_against_shrunk"] = xwoba_s
                row_dict[f"{side}_sp_swstr_pct_shrunk"] = swstr_s
                row_dict[f"{side}_sp_velo_delta_mph"] = velo_delta
                row_dict[f"{side}_sp_is_lhp"] = 1.0 if hand_eff == "L" else 0.0
                row_dict[f"{side}_sp_is_cross_league_prior"] = (
                    1.0 if default_p.is_cross_league_transfer else 0.0
                )

        out_rows.append(row_dict)

    sp_df = pl.DataFrame(out_rows)
    join_keys = ["game_id", "event_seq"] if "event_seq" in games_df.columns else ["game_id"]
    if "event_seq" not in games_df.columns:
        sp_df = sp_df.drop("event_seq")
    return games_df.join(sp_df, on=join_keys, how="left")


__all__ = [
    "DEFAULT_SP_OUTS_CEILING",
    "DEFAULT_SP_PITCH_CEILING",
    "DEFAULT_SP_WINDOWS",
    "MAX_SP_REST_DAYS",
    "build_starting_pitcher_rolling_features",
    "resolve_game_starting_pitcher_features",
]
