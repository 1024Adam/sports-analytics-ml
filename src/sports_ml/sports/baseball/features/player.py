"""Stage 2 Starting Pitcher & Lineup Batter matchup feature builder (`features/player.py`).

Builds individual pre-game matchup feature matrices for:
1. Starting Pitchers (`SP`): Conditioned on starter skill/ceiling, opposing lineup platoon
   `wOBA` & `K%`, ballpark factor, and Stage 1 opponent expected runs (`stage1_opp_expected_runs`).
2. Lineup Batters (`1..9`): Conditioned on batting order slot PA weight (`1..9`), home/away
   bottom-9th effect, batter platoon-adjusted `wOBA`/`ISO`/`K%`, opposing `SP` `xFIP`/`K%`,
   ballpark LHB/RHB HR factor, and Stage 1 team expected runs (`stage1_team_expected_runs`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from sports_ml.sports.baseball.constants import (
    BATTING_ORDER_SLOT_PA_WEIGHTS,
    get_ballpark_metadata,
    get_league_sabermetric_priors,
    normalize_baseball_team,
    resolve_team_league,
)
from sports_ml.sports.baseball.features.lineup import (
    build_batter_rolling_features,
    compute_platoon_adjusted_woba,
)
from sports_ml.sports.baseball.features.pitcher import (
    build_starting_pitcher_rolling_features,
)
from sports_ml.sports.baseball.features.translation import (
    TranslatedBatterPrior,
    TranslatedPitcherPrior,
)


def build_baseball_stage2_pitcher_features(
    games_df: pl.DataFrame,
    pitcher_logs_df: pl.DataFrame,
    stage1_features_df: pl.DataFrame | None = None,
    *,
    cross_league_pitcher_priors: Mapping[int, TranslatedPitcherPrior] | None = None,
) -> pl.DataFrame:
    """Build Stage 2 Starting Pitcher matchup features conditioned on Stage 1 environment."""
    if pitcher_logs_df.is_empty():
        return pl.DataFrame()

    sp_only = pitcher_logs_df.filter(pl.col("is_starter") == pl.lit(True))
    if sp_only.is_empty():
        return pl.DataFrame()

    rolled = build_starting_pitcher_rolling_features(
        sp_only,
        cross_league_priors=cross_league_pitcher_priors,
    )

    # Index game-level context from stage1_features_df or games_df
    ctx_source = stage1_features_df if stage1_features_df is not None else games_df
    ctx_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if not ctx_source.is_empty():
        for r in ctx_source.iter_rows(named=True):
            gid = int(r.get("game_id") or r.get("event_id") or 0)
            seq = int(r.get("event_seq") or 1)
            ctx_by_key[(gid, seq)] = r

    opp_woba_list: list[float] = []
    opp_k_list: list[float] = []
    opp_bb_list: list[float] = []
    venue_rf_list: list[float] = []
    dome_list: list[float] = []
    s1_win_prob_list: list[float] = []
    s1_opp_runs_list: list[float] = []

    for r in rolled.iter_rows(named=True):
        gid = int(r["game_id"])
        seq = int(r.get("event_seq") or 1)
        is_home = bool(r.get("is_home", True))
        team_code = normalize_baseball_team(str(r["team"]))
        lg_code = resolve_team_league(team_code) or str(r.get("league") or "MLB").upper()
        lg_priors = get_league_sabermetric_priors("NPB" if lg_code == "NPB" else "MLB")

        ctx = ctx_by_key.get((gid, seq), {})
        opp_side = "away" if is_home else "home"
        my_side = "home" if is_home else "away"

        opp_woba = float(ctx.get(f"{opp_side}_lineup_platoon_woba") or lg_priors.lg_woba)
        opp_k = float(ctx.get(f"{opp_side}_lineup_platoon_k_pct") or lg_priors.lg_batter_k_pct)
        opp_bb = float(ctx.get(f"{opp_side}_lineup_platoon_bb_pct") or lg_priors.lg_batter_bb_pct)

        park = get_ballpark_metadata(str(ctx.get("venue_id") or team_code))
        v_rf = float(ctx.get("venue_run_factor") or (park.run_factor if park else 1.0))
        raw_dome = ctx.get("is_dome_flag")
        is_dome = (
            float(raw_dome)
            if raw_dome is not None
            else (1.0 if (park and park.is_dome_or_retractable) else 0.0)
        )

        win_p = float(
            ctx.get(f"stage1_{my_side}_win_prob")
            or ctx.get(f"elo_{my_side}_win_prob")
            or (0.53 if is_home else 0.47)
        )
        opp_runs = float(
            ctx.get(f"stage1_expected_{opp_side}_runs")
            or ctx.get(f"{opp_side}_team_runs_for_per_g_shrunk")
            or lg_priors.lg_runs_per_team_game
        )

        opp_woba_list.append(opp_woba)
        opp_k_list.append(opp_k)
        opp_bb_list.append(opp_bb)
        venue_rf_list.append(v_rf)
        dome_list.append(is_dome)
        s1_win_prob_list.append(win_p)
        s1_opp_runs_list.append(opp_runs)

    return rolled.with_columns(
        pl.col("is_home").cast(pl.Float64).alias("is_home_flag"),
        (pl.col("pitch_hand") == "L").cast(pl.Float64).alias("sp_is_lhp"),
        pl.Series("opp_lineup_platoon_woba", opp_woba_list, dtype=pl.Float64),
        pl.Series("opp_lineup_platoon_k_pct", opp_k_list, dtype=pl.Float64),
        pl.Series("opp_lineup_platoon_bb_pct", opp_bb_list, dtype=pl.Float64),
        pl.Series("venue_run_factor", venue_rf_list, dtype=pl.Float64),
        pl.Series("is_dome_flag", dome_list, dtype=pl.Float64),
        pl.Series("stage1_team_win_prob", s1_win_prob_list, dtype=pl.Float64),
        pl.Series("stage1_opp_expected_runs", s1_opp_runs_list, dtype=pl.Float64),
    )


def build_baseball_stage2_batter_features(
    games_df: pl.DataFrame,
    batter_logs_df: pl.DataFrame,
    stage1_features_df: pl.DataFrame | None = None,
    *,
    cross_league_batter_priors: Mapping[int, TranslatedBatterPrior] | None = None,
) -> pl.DataFrame:
    """Build Stage 2 Lineup Batter matchup features conditioned on Stage 1 environment."""
    if batter_logs_df.is_empty():
        return pl.DataFrame()

    rolled = build_batter_rolling_features(
        batter_logs_df,
        cross_league_priors=cross_league_batter_priors,
    )

    ctx_source = stage1_features_df if stage1_features_df is not None else games_df
    ctx_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if not ctx_source.is_empty():
        for r in ctx_source.iter_rows(named=True):
            gid = int(r.get("game_id") or r.get("event_id") or 0)
            seq = int(r.get("event_seq") or 1)
            ctx_by_key[(gid, seq)] = r

    slot_w_list: list[float] = []
    platoon_woba_list: list[float] = []
    opp_hand_matchup_list: list[float] = []
    opp_sp_xfip_list: list[float] = []
    opp_sp_k_list: list[float] = []
    opp_sp_bb_list: list[float] = []
    opp_bp_xfip_list: list[float] = []
    venue_rf_list: list[float] = []
    venue_hr_list: list[float] = []
    s1_win_prob_list: list[float] = []
    s1_team_runs_list: list[float] = []

    for r in rolled.iter_rows(named=True):
        gid = int(r["game_id"])
        seq = int(r.get("event_seq") or 1)
        is_home = bool(r.get("is_home", True))
        slot = int(r.get("batting_order_slot") or 5)
        b_side = str(r.get("bat_side") or "R").upper()
        team_code = normalize_baseball_team(str(r["team"]))
        lg_code = resolve_team_league(team_code) or str(r.get("league") or "MLB").upper()
        lg_priors = get_league_sabermetric_priors("NPB" if lg_code == "NPB" else "MLB")

        ctx = ctx_by_key.get((gid, seq), {})
        opp_side = "away" if is_home else "home"
        my_side = "home" if is_home else "away"

        opp_sp_hand = str(ctx.get(f"{opp_side}_sp_hand") or "R").upper()
        woba_s = float(r["batter_woba_shrunk"])
        plat_woba = compute_platoon_adjusted_woba(
            woba_s, bat_side=b_side, opposing_sp_hand=opp_sp_hand
        )
        is_opp_hand = (
            1.0 if (b_side == "S" or (b_side in {"L", "R"} and b_side != opp_sp_hand)) else 0.0
        )

        opp_sp_xfip = float(ctx.get(f"{opp_side}_sp_xfip_shrunk") or lg_priors.lg_sp_xfip)
        opp_sp_k = float(ctx.get(f"{opp_side}_sp_k_pct_shrunk") or lg_priors.lg_sp_k_pct)
        opp_sp_bb = float(ctx.get(f"{opp_side}_sp_bb_pct_shrunk") or lg_priors.lg_sp_bb_pct)
        opp_bp_xfip = float(ctx.get(f"{opp_side}_bp_xfip_shrunk") or lg_priors.lg_bp_xfip)

        park = get_ballpark_metadata(str(ctx.get("venue_id") or team_code))
        v_rf = float(ctx.get("venue_run_factor") or (park.run_factor if park else 1.0))
        if b_side == "L" or (b_side == "S" and opp_sp_hand == "R"):
            v_hr = float(ctx.get("venue_hr_factor_lhb") or (park.hr_factor_lhb if park else 1.0))
        else:
            v_hr = float(ctx.get("venue_hr_factor_rhb") or (park.hr_factor_rhb if park else 1.0))

        win_p = float(
            ctx.get(f"stage1_{my_side}_win_prob")
            or ctx.get(f"elo_{my_side}_win_prob")
            or (0.53 if is_home else 0.47)
        )
        team_runs = float(
            ctx.get(f"stage1_expected_{my_side}_runs")
            or ctx.get(f"{my_side}_team_runs_for_per_g_shrunk")
            or lg_priors.lg_runs_per_team_game
        )

        slot_w_list.append(BATTING_ORDER_SLOT_PA_WEIGHTS.get(slot, 0.095))
        platoon_woba_list.append(plat_woba)
        opp_hand_matchup_list.append(is_opp_hand)
        opp_sp_xfip_list.append(opp_sp_xfip)
        opp_sp_k_list.append(opp_sp_k)
        opp_sp_bb_list.append(opp_sp_bb)
        opp_bp_xfip_list.append(opp_bp_xfip)
        venue_rf_list.append(v_rf)
        venue_hr_list.append(v_hr)
        s1_win_prob_list.append(win_p)
        s1_team_runs_list.append(team_runs)

    return rolled.with_columns(
        pl.col("is_home").cast(pl.Float64).alias("is_home_flag"),
        pl.col("is_starter").cast(pl.Float64).alias("is_starter_flag"),
        pl.col("batting_order_slot").cast(pl.Float64).alias("batting_slot_num"),
        pl.Series("slot_pa_weight", slot_w_list, dtype=pl.Float64),
        pl.Series("batter_platoon_woba", platoon_woba_list, dtype=pl.Float64),
        pl.Series("is_opposite_hand_matchup", opp_hand_matchup_list, dtype=pl.Float64),
        pl.Series("opp_sp_xfip_shrunk", opp_sp_xfip_list, dtype=pl.Float64),
        pl.Series("opp_sp_k_pct_shrunk", opp_sp_k_list, dtype=pl.Float64),
        pl.Series("opp_sp_bb_pct_shrunk", opp_sp_bb_list, dtype=pl.Float64),
        pl.Series("opp_bp_xfip_shrunk", opp_bp_xfip_list, dtype=pl.Float64),
        pl.Series("venue_run_factor", venue_rf_list, dtype=pl.Float64),
        pl.Series("venue_hr_factor_batter", venue_hr_list, dtype=pl.Float64),
        pl.Series("stage1_team_win_prob", s1_win_prob_list, dtype=pl.Float64),
        pl.Series("stage1_team_expected_runs", s1_team_runs_list, dtype=pl.Float64),
    )
