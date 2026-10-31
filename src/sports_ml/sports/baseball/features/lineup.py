"""Platoon-adjusted 1-9 batting order aggregation (`features/lineup.py`).

Weights starting batters `1..9` by expected slot Plate Appearance share
(`BATTING_ORDER_SLOT_PA_WEIGHTS`) and adjusts each batter's shrunk `wOBA`, `ISO`,
`K%`, and `BB%` for the opposing starting pitcher's handedness (`opp_sp_hand in {'L', 'R'}`)
using heavy platoon shrinkage (`k_LHB = 1000 PA`, `k_RHB = 2200 PA`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import polars as pl

from sports_ml.core.shrinkage import bayesian_shrinkage, shrink_scalar
from sports_ml.core.temporal import (
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_rolling_sum,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.baseball.constants import (
    BATTING_ORDER_SLOT_PA_WEIGHTS,
    DEFAULT_OPPOSITE_HAND_WOBA_BOOST,
    DEFAULT_SAME_HAND_WOBA_PENALTY,
    SHRINKAGE_K_BATTER_PA,
    SHRINKAGE_K_PLATOON_LHB_PA,
    SHRINKAGE_K_PLATOON_RHB_PA,
    get_league_sabermetric_priors,
    normalize_baseball_team,
    resolve_team_league,
)
from sports_ml.sports.baseball.features.translation import (
    TranslatedBatterPrior,
    get_rookie_batter_prior,
)


def _ensure_batter_log_columns(df: pl.DataFrame) -> pl.DataFrame:
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
        ("bat_side", "R", pl.Utf8()),
        ("batting_order_slot", 5, pl.Int8()),
        ("position_group", "INF", pl.Utf8()),
        ("is_starter", True, pl.Boolean()),
        ("pa", 4, pl.Int16()),
        ("ab", 4, pl.Int16()),
        ("hits", 1, pl.Int16()),
        ("singles", 1, pl.Int16()),
        ("doubles", 0, pl.Int16()),
        ("triples", 0, pl.Int16()),
        ("hr", 0, pl.Int16()),
        ("total_bases", 1, pl.Int16()),
        ("runs", 0, pl.Int16()),
        ("rbi", 0, pl.Int16()),
        ("bb", 0, pl.Int16()),
        ("ibb", 0, pl.Int16()),
        ("hbp", 0, pl.Int16()),
        ("k", 1, pl.Int16()),
        ("sf", 0, pl.Int16()),
        ("sb", 0, pl.Int16()),
        ("cs", 0, pl.Int16()),
        ("xwoba", None, pl.Float64()),
        ("hard_hit_rate", None, pl.Float64()),
        ("barrel_rate", None, pl.Float64()),
    ]
    missing_exprs = [
        pl.lit(val, dtype=dtype).alias(col)
        for col, val, dtype in defaults
        if col not in working.columns
    ]
    if missing_exprs:
        working = working.with_columns(missing_exprs)

    if "woba_num" not in working.columns:
        working = working.with_columns(
            (
                0.690 * (pl.col("bb") - pl.col("ibb")).clip(lower_bound=0)
                + 0.722 * pl.col("hbp")
                + 0.888 * pl.col("singles")
                + 1.271 * pl.col("doubles")
                + 1.616 * pl.col("triples")
                + 2.101 * pl.col("hr")
            )
            .cast(pl.Float64)
            .alias("woba_num")
        )
    return working


def compute_platoon_adjusted_woba(
    overall_shrunk_woba: float,
    *,
    bat_side: str,
    opposing_sp_hand: str,
    split_woba: float | None = None,
    split_pa: int | float = 0.0,
) -> float:
    """Compute a batter's platoon-adjusted wOBA against `opposing_sp_hand` (`'L'` or `'R'`).

    Applies empirical handedness platoon prior (`+0.014` opposite-hand, `-0.010` same-hand,
    `+0.004` switch-hitter) and blends with observed split wOBA using heavy platoon
    shrinkage (`k = 1000 PA` for LHB, `k = 2200 PA` for RHB).
    """
    b_side = bat_side.strip().upper()
    p_hand = opposing_sp_hand.strip().upper()

    if b_side == "S":
        platoon_delta_prior = 0.004
        k_platoon = SHRINKAGE_K_PLATOON_LHB_PA
    elif b_side != p_hand:
        platoon_delta_prior = DEFAULT_OPPOSITE_HAND_WOBA_BOOST
        k_platoon = SHRINKAGE_K_PLATOON_LHB_PA if b_side == "L" else SHRINKAGE_K_PLATOON_RHB_PA
    else:
        platoon_delta_prior = DEFAULT_SAME_HAND_WOBA_PENALTY
        k_platoon = SHRINKAGE_K_PLATOON_LHB_PA if b_side == "L" else SHRINKAGE_K_PLATOON_RHB_PA

    prior_split_woba = float(overall_shrunk_woba) + platoon_delta_prior
    return shrink_scalar(split_woba, split_pa, prior_split_woba, k_platoon)


def build_batter_rolling_features(
    batter_logs_df: pl.DataFrame,
    *,
    window: int = 35,
    shrinkage_k_pa: float = SHRINKAGE_K_BATTER_PA,
    cross_league_priors: Mapping[int, TranslatedBatterPrior] | None = None,
) -> pl.DataFrame:
    """Build strictly pre-game `.shift(1)` rolling and shrunk batter rate features."""
    prepared = _ensure_batter_log_columns(batter_logs_df)
    sorted_df = sort_events_chronologically(prepared, extra_tiebreakers=["player_id"])

    rows_woba_p: list[float] = []
    rows_iso_p: list[float] = []
    rows_k_p: list[float] = []
    rows_bb_p: list[float] = []

    for row in sorted_df.iter_rows(named=True):
        pid = int(row["player_id"])
        lg_str = str(row.get("league") or "MLB").upper()
        lg_priors = get_league_sabermetric_priors("NPB" if lg_str == "NPB" else "MLB")
        custom_p = cross_league_priors.get(pid) if cross_league_priors else None
        if custom_p is not None:
            rows_woba_p.append(custom_p.woba_prior)
            rows_iso_p.append(custom_p.iso_prior)
            rows_k_p.append(custom_p.k_pct_prior)
            rows_bb_p.append(custom_p.bb_pct_prior)
        else:
            rows_woba_p.append(lg_priors.lg_woba)
            rows_iso_p.append(lg_priors.lg_iso)
            rows_k_p.append(lg_priors.lg_batter_k_pct)
            rows_bb_p.append(lg_priors.lg_batter_bb_pct)

    enriched = sorted_df.with_columns(
        pl.Series("_prior_woba", rows_woba_p, dtype=pl.Float64),
        pl.Series("_prior_iso", rows_iso_p, dtype=pl.Float64),
        pl.Series("_prior_k_pct", rows_k_p, dtype=pl.Float64),
        pl.Series("_prior_bb_pct", rows_bb_p, dtype=pl.Float64),
        (pl.col("total_bases").cast(pl.Float64) - pl.col("hits").cast(pl.Float64))
        .clip(lower_bound=0.0)
        .alias("_iso_num"),
        (pl.col("bb").cast(pl.Float64) - pl.col("ibb").cast(pl.Float64).fill_null(0.0))
        .clip(lower_bound=0.0)
        .alias("_ubb"),
    )

    cum_pa = pregame_rolling_sum("pa", window=window, group_by="player_id")
    cum_ab = pregame_rolling_sum("ab", window=window, group_by="player_id")

    raw_woba = pregame_exposure_rate(
        "woba_num", "pa", window=window, group_by="player_id", scale_factor=1.0
    )
    raw_iso = pregame_exposure_rate(
        "_iso_num", "ab", window=window, group_by="player_id", scale_factor=1.0
    )
    raw_k_pct = pregame_exposure_rate(
        "k", "pa", window=window, group_by="player_id", scale_factor=1.0
    )
    raw_bb_pct = pregame_exposure_rate(
        "_ubb", "pa", window=window, group_by="player_id", scale_factor=1.0
    )
    raw_hr_pa = pregame_exposure_rate(
        "hr", "pa", window=window, group_by="player_id", scale_factor=1.0
    )

    out = enriched.with_columns(
        pregame_sample_count("pa", group_by="player_id").alias("batter_prior_games_n"),
        cum_pa.fill_null(0.0).alias("batter_rolling_pa"),
        bayesian_shrinkage(raw_woba, cum_pa, "_prior_woba", shrinkage_k_pa).alias(
            "batter_woba_shrunk"
        ),
        bayesian_shrinkage(raw_iso, cum_ab, "_prior_iso", shrinkage_k_pa).alias(
            "batter_iso_shrunk"
        ),
        bayesian_shrinkage(raw_k_pct, cum_pa, "_prior_k_pct", shrinkage_k_pa).alias(
            "batter_k_pct_shrunk"
        ),
        bayesian_shrinkage(raw_bb_pct, cum_pa, "_prior_bb_pct", shrinkage_k_pa).alias(
            "batter_bb_pct_shrunk"
        ),
        bayesian_shrinkage(raw_hr_pa, cum_pa, 0.031, shrinkage_k_pa).alias(
            "batter_hr_per_pa_shrunk"
        ),
        bayesian_shrinkage(
            pregame_rolling_mean("xwoba", window=window, group_by="player_id"),
            pregame_sample_count("xwoba", group_by="player_id", window=window),
            "_prior_woba",
            15.0,
        ).alias("batter_xwoba_shrunk"),
        bayesian_shrinkage(
            pregame_rolling_mean("barrel_rate", window=window, group_by="player_id"),
            pregame_sample_count("barrel_rate", group_by="player_id", window=window),
            0.068,
            15.0,
        ).alias("batter_barrel_rate_shrunk"),
    )

    drop_temp = ["_prior_woba", "_prior_iso", "_prior_k_pct", "_prior_bb_pct", "_iso_num", "_ubb"]
    return out.drop([c for c in drop_temp if c in out.columns])


def build_platoon_lineup_strength_features(
    games_df: pl.DataFrame,
    batter_logs_df: pl.DataFrame,
    *,
    cross_league_priors: Mapping[int, TranslatedBatterPrior] | None = None,
    custom_lineups_by_game_team: Mapping[tuple[int, int, str], Sequence[int]] | None = None,
) -> pl.DataFrame:
    """Compute PA-weighted, handedness platoon-adjusted 1-9 lineup features for `games_df`."""
    if games_df.is_empty():
        return games_df

    b_work = (
        _ensure_batter_log_columns(batter_logs_df)
        if not batter_logs_df.is_empty()
        else pl.DataFrame()
    )

    # Join opposing starter hand from games_df onto batter_logs to track platoon splits
    if not b_work.is_empty() and {"home_sp_hand", "away_sp_hand"}.issubset(games_df.columns):
        g_hands = games_df.select(
            [
                pl.col("game_id" if "game_id" in games_df.columns else "event_id")
                .cast(pl.Int64)
                .alias("game_id"),
                (
                    pl.col("event_seq").cast(pl.Int32)
                    if "event_seq" in games_df.columns
                    else pl.lit(1, dtype=pl.Int32).alias("event_seq")
                ),
                pl.col("home_sp_hand"),
                pl.col("away_sp_hand"),
            ]
        ).unique(subset=["game_id", "event_seq"], keep="last")
        b_work = b_work.join(g_hands, on=["game_id", "event_seq"], how="left").with_columns(
            pl.when(pl.col("is_home"))
            .then(pl.col("away_sp_hand"))
            .otherwise(pl.col("home_sp_hand"))
            .fill_null("R")
            .alias("_faced_sp_hand")
        )
    elif not b_work.is_empty():
        b_work = b_work.with_columns(pl.lit("R").alias("_faced_sp_hand"))

    b_sorted = (
        sort_events_chronologically(b_work, extra_tiebreakers=["player_id"])
        if not b_work.is_empty()
        else pl.DataFrame()
    )

    hist_by_club: dict[str, list[dict[str, Any]]] = {}
    actual_starters_by_game_team: dict[tuple[int, int, str], list[int]] = {}
    if not b_sorted.is_empty():
        for r in b_sorted.iter_rows(named=True):
            c = normalize_baseball_team(str(r["team"]))
            hist_by_club.setdefault(c, []).append(r)
            if bool(r.get("is_starter", False)) and 1 <= int(r.get("batting_order_slot") or 0) <= 9:
                g_key = (int(r["game_id"]), int(r.get("event_seq") or 1), c)
                actual_starters_by_game_team.setdefault(g_key, []).append(int(r["player_id"]))

    out_rows: list[dict[str, Any]] = []
    for g in games_df.iter_rows(named=True):
        gid = int(g.get("game_id") or g.get("event_id") or 0)
        g_seq = int(g.get("event_seq") or 1)
        g_utc: datetime = g["start_time_utc"]

        row_out: dict[str, Any] = {"game_id": gid, "event_seq": g_seq}

        for side, opp_side in (("home", "away"), ("away", "home")):
            team_code = normalize_baseball_team(str(g[f"{side}_team"]))
            lg_code = resolve_team_league(team_code) or str(g.get("league") or "MLB").upper()
            lg_priors = get_league_sabermetric_priors("NPB" if lg_code == "NPB" else "MLB")
            rookie_b = get_rookie_batter_prior("NPB" if lg_code == "NPB" else "MLB")

            opp_sp_hand = str(g.get(f"{opp_side}_sp_hand") or "R").upper()
            if opp_sp_hand not in {"L", "R"}:
                opp_sp_hand = "R"

            # Strictly pre-game batter logs for this team
            prior_logs = [
                bl
                for bl in hist_by_club.get(team_code, [])
                if (bl["start_time_utc"], int(bl.get("event_seq") or 1)) < (g_utc, g_seq)
            ]

            # Build per-batter pre-game shrunk stats
            by_pid: dict[int, list[dict[str, Any]]] = {}
            for bl in prior_logs:
                by_pid.setdefault(int(bl["player_id"]), []).append(bl)

            batter_stats: dict[int, dict[str, float | str]] = {}
            for pid, p_logs in by_pid.items():
                recent = p_logs[-45:]
                tot_pa = sum(int(x.get("pa") or 0) for x in recent)
                tot_ab = sum(int(x.get("ab") or 0) for x in recent)
                tot_woba_num = sum(float(x.get("woba_num") or 0.0) for x in recent)
                tot_iso_num = sum(
                    max(0, int(x.get("total_bases") or 0) - int(x.get("hits") or 0)) for x in recent
                )
                tot_k = sum(int(x.get("k") or 0) for x in recent)
                tot_ubb = sum(
                    max(0, int(x.get("bb") or 0) - int(x.get("ibb") or 0)) for x in recent
                )

                custom_p = cross_league_priors.get(pid) if cross_league_priors else None
                base_p = custom_p if custom_p is not None else rookie_b

                raw_w = tot_woba_num / max(tot_pa, 1) if tot_pa > 0 else None
                raw_iso = tot_iso_num / max(tot_ab, 1) if tot_ab > 0 else None
                raw_k = tot_k / max(tot_pa, 1) if tot_pa > 0 else None
                raw_bb = tot_ubb / max(tot_pa, 1) if tot_pa > 0 else None

                woba_s = shrink_scalar(raw_w, tot_pa, base_p.woba_prior, SHRINKAGE_K_BATTER_PA)
                iso_s = shrink_scalar(raw_iso, tot_ab, base_p.iso_prior, SHRINKAGE_K_BATTER_PA)
                k_s = shrink_scalar(raw_k, tot_pa, base_p.k_pct_prior, SHRINKAGE_K_BATTER_PA)
                bb_s = shrink_scalar(raw_bb, tot_pa, base_p.bb_pct_prior, SHRINKAGE_K_BATTER_PA)

                split_logs = [
                    x for x in recent if str(x.get("_faced_sp_hand") or "R").upper() == opp_sp_hand
                ]
                split_pa = sum(int(x.get("pa") or 0) for x in split_logs)
                split_woba = (
                    sum(float(x.get("woba_num") or 0.0) for x in split_logs) / split_pa
                    if split_pa > 0
                    else None
                )
                b_side = str(recent[-1].get("bat_side") or "R").upper()
                platoon_woba = compute_platoon_adjusted_woba(
                    woba_s,
                    bat_side=b_side,
                    opposing_sp_hand=opp_sp_hand,
                    split_woba=split_woba,
                    split_pa=split_pa,
                )
                # Adjust ISO and K% modestly for platoon handedness advantage
                is_opp_hand = b_side == "S" or (b_side in {"L", "R"} and b_side != opp_sp_hand)
                platoon_iso = iso_s + (0.010 if is_opp_hand else -0.008)
                platoon_k = max(0.04, k_s + (-0.012 if is_opp_hand else 0.012))

                xwoba_vals = [float(x["xwoba"]) for x in recent if x.get("xwoba") is not None]
                xwoba_s = shrink_scalar(
                    sum(xwoba_vals) / len(xwoba_vals) if xwoba_vals else None,
                    len(xwoba_vals),
                    woba_s,
                    12.0,
                )

                batter_stats[pid] = {
                    "bat_side": b_side,
                    "tot_pa": float(tot_pa),
                    "woba_shrunk": woba_s,
                    "xwoba_shrunk": xwoba_s,
                    "woba_minus_xwoba_luck": woba_s - xwoba_s,
                    "platoon_woba": platoon_woba,
                    "platoon_iso": platoon_iso,
                    "platoon_k_pct": platoon_k,
                    "bb_pct_shrunk": bb_s,
                    "same_hand": 1.0 if (b_side in {"L", "R"} and b_side == opp_sp_hand) else 0.0,
                }

            # Top-9 regular batters by prior Plate Appearances (for missing_regular_pa_woba)
            top_regular_pids = sorted(
                batter_stats.keys(),
                key=lambda pid: float(batter_stats[pid]["tot_pa"]),
                reverse=True,
            )[:9]

            # Determine active 1-9 lineup for this game:
            # 1) Custom override, 2) Known starting 9, or 3) Platoon-projected top 9
            g_team_key = (gid, g_seq, team_code)
            if custom_lineups_by_game_team and g_team_key in custom_lineups_by_game_team:
                lineup_pids = list(custom_lineups_by_game_team[g_team_key])[:9]
            elif (
                g_team_key in actual_starters_by_game_team
                and len(actual_starters_by_game_team[g_team_key]) >= 9
            ):
                lineup_pids = actual_starters_by_game_team[g_team_key][:9]
            else:
                # Platoon-projected 9 starters: rank available batters by PA + opposite-hand bonus
                lineup_pids = sorted(
                    batter_stats.keys(),
                    key=lambda pid: (
                        float(batter_stats[pid]["tot_pa"])
                        + (40.0 if float(batter_stats[pid]["same_hand"]) == 0.0 else 0.0)
                    ),
                    reverse=True,
                )[:9]

            if not lineup_pids:
                # Opening Day cold-start fallback
                is_lhp_opp = opp_sp_hand == "L"
                row_out[f"{side}_lineup_platoon_woba"] = lg_priors.lg_woba + (
                    0.004 if is_lhp_opp else 0.002
                )
                row_out[f"{side}_lineup_platoon_iso"] = lg_priors.lg_iso
                row_out[f"{side}_lineup_platoon_k_pct"] = lg_priors.lg_batter_k_pct
                row_out[f"{side}_lineup_platoon_bb_pct"] = lg_priors.lg_batter_bb_pct
                row_out[f"{side}_lineup_same_hand_share"] = 0.25 if is_lhp_opp else 0.55
                row_out[f"{side}_missing_regular_pa_woba"] = 0.0
                row_out[f"{side}_lineup_xwoba"] = lg_priors.lg_woba
                row_out[f"{side}_woba_minus_xwoba_luck"] = 0.0
                continue

            # Compute slot-weighted 1..9 lineup metrics
            w_sum = 0.0
            woba_acc = 0.0
            xwoba_acc = 0.0
            luck_acc = 0.0
            iso_acc = 0.0
            k_acc = 0.0
            bb_acc = 0.0
            same_hand_acc = 0.0

            for slot_idx, pid in enumerate(lineup_pids[:9], start=1):
                w_slot = BATTING_ORDER_SLOT_PA_WEIGHTS.get(slot_idx, 1.0 / 9.0)
                st = batter_stats.get(pid)
                if st is None:
                    custom_p = cross_league_priors.get(pid) if cross_league_priors else None
                    base_p = custom_p if custom_p is not None else rookie_b
                    p_woba = base_p.woba_prior
                    p_xwoba = base_p.woba_prior
                    p_luck = 0.0
                    p_iso = base_p.iso_prior
                    p_k = base_p.k_pct_prior
                    p_bb = base_p.bb_pct_prior
                    s_hand = 0.5
                else:
                    p_woba = float(st["platoon_woba"])
                    p_xwoba = float(st["xwoba_shrunk"])
                    p_luck = float(st["woba_minus_xwoba_luck"])
                    p_iso = float(st["platoon_iso"])
                    p_k = float(st["platoon_k_pct"])
                    p_bb = float(st["bb_pct_shrunk"])
                    s_hand = float(st["same_hand"])

                w_sum += w_slot
                woba_acc += w_slot * p_woba
                xwoba_acc += w_slot * p_xwoba
                luck_acc += w_slot * p_luck
                iso_acc += w_slot * p_iso
                k_acc += w_slot * p_k
                bb_acc += w_slot * p_bb
                same_hand_acc += w_slot * s_hand

            active_set = set(lineup_pids[:9])
            missing_reg_woba = sum(
                float(batter_stats[r_pid]["woba_shrunk"]) * (1.0 / 9.0)
                for r_pid in top_regular_pids
                if r_pid not in active_set
            )

            row_out[f"{side}_lineup_platoon_woba"] = woba_acc / max(w_sum, 1e-9)
            row_out[f"{side}_lineup_platoon_iso"] = iso_acc / max(w_sum, 1e-9)
            row_out[f"{side}_lineup_platoon_k_pct"] = k_acc / max(w_sum, 1e-9)
            row_out[f"{side}_lineup_platoon_bb_pct"] = bb_acc / max(w_sum, 1e-9)
            row_out[f"{side}_lineup_same_hand_share"] = same_hand_acc / max(w_sum, 1e-9)
            row_out[f"{side}_missing_regular_pa_woba"] = float(missing_reg_woba)
            row_out[f"{side}_lineup_xwoba"] = xwoba_acc / max(w_sum, 1e-9)
            row_out[f"{side}_woba_minus_xwoba_luck"] = luck_acc / max(w_sum, 1e-9)

        out_rows.append(row_out)

    lineup_df = pl.DataFrame(out_rows)
    join_keys = ["game_id", "event_seq"] if "event_seq" in games_df.columns else ["game_id"]
    if "event_seq" not in games_df.columns:
        lineup_df = lineup_df.drop("event_seq")
    return games_df.join(lineup_df, on=join_keys, how="left")
