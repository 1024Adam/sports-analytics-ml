"""Team rolling efficiency, Pythagorean/BaseRuns luck, ballpark, and Stage 1 features (`team.py`).

Combines league-partitioned Starter- & Venue-Adjusted Elo, starting pitcher skill/rest,
bullpen 3-day fatigue, handedness platoon lineups, ballpark run/HR factors, Koshien /
provincial neutral venue context, and schedule density into the Gold Stage 1 feature matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Final

import polars as pl

from sports_ml.core.ratings import PairwiseEloEngine
from sports_ml.core.schedule import haversine_distance_km
from sports_ml.core.shrinkage import bayesian_shrinkage, shrink_scalar
from sports_ml.core.temporal import (
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_rolling_sum,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.sports.baseball.constants import (
    BASEBALL_BALLPARKS,
    MLB_ELO_CONFIG,
    NPB_ELO_CONFIG,
    SHRINKAGE_K_TEAM_GAMES,
    get_ballpark_metadata,
    get_baseball_club_metadata,
    get_league_sabermetric_priors,
    normalize_baseball_team,
    resolve_team_league,
)
from sports_ml.sports.baseball.features.bullpen import build_bullpen_pregame_features
from sports_ml.sports.baseball.features.elo_hooks import (
    BaseballStarterEloHook,
    BaseballVenueEloHook,
)
from sports_ml.sports.baseball.features.lineup import (
    build_platoon_lineup_strength_features,
)
from sports_ml.sports.baseball.features.pitcher import (
    resolve_game_starting_pitcher_features,
)
from sports_ml.sports.baseball.features.translation import (
    TranslatedBatterPrior,
    TranslatedPitcherPrior,
)

PYTHAG_EXPONENT: Final[float] = 1.83


def build_baseball_team_rolling_features(
    team_logs_df: pl.DataFrame,
    *,
    window: int = 30,
    shrinkage_k: float = SHRINKAGE_K_TEAM_GAMES,
) -> pl.DataFrame:
    """Build strictly pre-game `.shift(1)` team rolling offensive, defensive, and luck features."""
    working = team_logs_df
    if "game_id" in working.columns and "event_id" not in working.columns:
        working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))

    sorted_df = sort_events_chronologically(working, extra_tiebreakers=["team"])

    rows_prior_runs: list[float] = []
    rows_prior_woba: list[float] = []
    for r in sorted_df.iter_rows(named=True):
        lg = str(r.get("league") or resolve_team_league(str(r["team"])) or "MLB").upper()
        priors = get_league_sabermetric_priors("NPB" if lg == "NPB" else "MLB")
        rows_prior_runs.append(priors.lg_runs_per_team_game)
        rows_prior_woba.append(priors.lg_woba)

    enriched = sorted_df.with_columns(
        pl.Series("_prior_runs", rows_prior_runs, dtype=pl.Float64),
        pl.Series("_prior_woba", rows_prior_woba, dtype=pl.Float64),
        (pl.col("runs_for") > pl.col("runs_against")).cast(pl.Float64).alias("_won_game"),
    )

    n_w = pregame_sample_count("runs_for", group_by="team", window=window)
    rf_mean = pregame_rolling_mean("runs_for", window=window, group_by="team")
    ra_mean = pregame_rolling_mean("runs_against", window=window, group_by="team")
    woba_mean = pregame_rolling_mean("team_woba", window=window, group_by="team")
    opp_woba_mean = pregame_rolling_mean("opp_woba", window=window, group_by="team")
    win_pct = pregame_rolling_mean("_won_game", window=window, group_by="team")

    rf_sum = pregame_rolling_sum("runs_for", window=window, group_by="team").clip(lower_bound=1.0)
    ra_sum = pregame_rolling_sum("runs_against", window=window, group_by="team").clip(
        lower_bound=1.0
    )
    pythag_pct = (rf_sum**PYTHAG_EXPONENT) / ((rf_sum**PYTHAG_EXPONENT) + (ra_sum**PYTHAG_EXPONENT))

    out = enriched.with_columns(
        bayesian_shrinkage(rf_mean, n_w, "_prior_runs", shrinkage_k).alias(
            "team_runs_for_per_g_shrunk"
        ),
        bayesian_shrinkage(ra_mean, n_w, "_prior_runs", shrinkage_k).alias(
            "team_runs_against_per_g_shrunk"
        ),
        bayesian_shrinkage(woba_mean, n_w, "_prior_woba", shrinkage_k).alias("team_woba_shrunk"),
        bayesian_shrinkage(opp_woba_mean, n_w, "_prior_woba", shrinkage_k).alias("opp_woba_shrunk"),
        bayesian_shrinkage(
            pregame_exposure_rate("k_for", "pa_for", window=window, group_by="team"),
            n_w,
            0.215,
            shrinkage_k,
        ).alias("team_k_pct_shrunk"),
        bayesian_shrinkage(
            pregame_exposure_rate("bb_for", "pa_for", window=window, group_by="team"),
            n_w,
            0.082,
            shrinkage_k,
        ).alias("team_bb_pct_shrunk"),
        (win_pct.fill_null(0.5) - pythag_pct.fill_null(0.5)).alias("team_pythag_luck_w30"),
    ).with_columns(
        (pl.col("team_runs_for_per_g_shrunk") - pl.col("team_runs_against_per_g_shrunk")).alias(
            "team_run_diff_per_g_shrunk"
        )
    )
    return out.drop(["_prior_runs", "_prior_woba", "_won_game"])


def _compute_team_history_features_for_games(
    games_df: pl.DataFrame,
    team_logs_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute strictly pre-game team efficiency, Pythagorean/BaseRuns luck, and travel/rest."""
    if games_df.is_empty():
        return games_df

    hist_by_club: dict[str, list[dict[str, Any]]] = {}
    if not team_logs_df.is_empty():
        t_work = team_logs_df
        if "game_id" in t_work.columns and "event_id" not in t_work.columns:
            t_work = t_work.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
        if "event_seq" not in t_work.columns:
            t_work = t_work.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
        t_sorted = sort_events_chronologically(t_work, extra_tiebreakers=["team"])
        for r in t_sorted.iter_rows(named=True):
            c = normalize_baseball_team(str(r["team"]))
            hist_by_club.setdefault(c, []).append(r)

    # Also index games_df chronologically by team for rest & travel distance calculation
    g_work = games_df
    if "game_id" in g_work.columns and "event_id" not in g_work.columns:
        g_work = g_work.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    if "event_seq" not in g_work.columns:
        g_work = g_work.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
    g_sorted = sort_events_chronologically(g_work)

    sched_by_club: dict[str, list[dict[str, Any]]] = {}
    for g in g_sorted.iter_rows(named=True):
        for side in ("home", "away"):
            c = normalize_baseball_team(str(g[f"{side}_team"]))
            sched_by_club.setdefault(c, []).append(g)

    out_rows: list[dict[str, Any]] = []
    for g in games_df.iter_rows(named=True):
        gid = int(g.get("game_id") or g.get("event_id") or 0)
        g_seq = int(g.get("event_seq") or 1)
        g_utc: datetime = g["start_time_utc"]
        g_date: date = g.get("game_date") or g_utc.date()

        home_code = normalize_baseball_team(str(g["home_team"]))
        away_code = normalize_baseball_team(str(g["away_team"]))
        lg_str = str(g.get("league") or resolve_team_league(home_code) or "MLB").upper()
        lg_code = "NPB" if lg_str == "NPB" else "MLB"
        lg_priors = get_league_sabermetric_priors(lg_code)

        raw_vid = str(g.get("venue_id") or "")
        park = (
            BASEBALL_BALLPARKS.get(raw_vid.upper())
            or get_ballpark_metadata(home_code)
            or BASEBALL_BALLPARKS["MLB_LAD"]
        )

        # Hanshin Tigers August Summer Koshien road-trip displacement indicator
        is_koshien_road_trip = (
            1.0
            if (
                lg_code == "NPB"
                and g_date.month == 8
                and (away_code == "HAN" or (home_code == "HAN" and park.venue_id != "NPB_KOSHIEN"))
            )
            else 0.0
        )

        sub_lg = str(g.get("sub_league") or "").upper()
        h_club = get_baseball_club_metadata(home_code)
        a_club = get_baseball_club_metadata(away_code)
        is_interleague = (
            1.0
            if (
                sub_lg in {"INTERLEAGUE_MLB", "KORYUSEN"}
                or (
                    h_club is not None
                    and a_club is not None
                    and h_club.sub_league != a_club.sub_league
                )
            )
            else 0.0
        )
        uses_dh = (
            1.0
            if bool(
                g.get("uses_dh")
                if g.get("uses_dh") is not None
                else (lg_code == "MLB" or (h_club and h_club.sub_league == "PL"))
            )
            else 0.0
        )

        row_out: dict[str, Any] = {
            "game_id": gid,
            "event_seq": g_seq,
            "venue_run_factor": float(park.run_factor),
            "venue_hr_factor_lhb": float(park.hr_factor_lhb),
            "venue_hr_factor_rhb": float(park.hr_factor_rhb),
            "venue_hr_factor_matchup": float(0.5 * (park.hr_factor_lhb + park.hr_factor_rhb)),
            "is_dome_flag": 1.0
            if bool(g.get("is_dome_or_retractable", park.is_dome_or_retractable))
            else 0.0,
            "is_neutral_or_regional_flag": 1.0
            if bool(g.get("is_neutral_or_regional_venue", park.is_neutral_or_regional))
            else 0.0,
            "is_koshien_summer_road_trip_flag": is_koshien_road_trip,
            "is_interleague_flag": is_interleague,
            "uses_dh_flag": uses_dh,
            "is_doubleheader_g1": 1.0
            if g_seq == 1 and int(g.get("is_doubleheader") or 0) == 1
            else 0.0,
            "is_doubleheader_g2": 1.0 if g_seq == 2 else 0.0,
        }

        for side, club_code in (("home", home_code), ("away", away_code)):
            prior_t = [
                tl
                for tl in hist_by_club.get(club_code, [])
                if (tl["start_time_utc"], int(tl.get("event_seq") or 1)) < (g_utc, g_seq)
            ][-30:]

            if not prior_t:
                row_out[f"{side}_team_runs_for_per_g_shrunk"] = lg_priors.lg_runs_per_team_game
                row_out[f"{side}_team_runs_against_per_g_shrunk"] = lg_priors.lg_runs_per_team_game
                row_out[f"{side}_team_run_diff_per_g_shrunk"] = 0.0
                row_out[f"{side}_team_woba_shrunk"] = lg_priors.lg_woba
                row_out[f"{side}_opp_woba_shrunk"] = lg_priors.lg_woba
                row_out[f"{side}_pythag_luck_w30"] = 0.0
                row_out[f"{side}_baseruns_luck_w20"] = 0.0
            else:
                n_g = len(prior_t)
                rf_avg = sum(float(x.get("runs_for") or 0.0) for x in prior_t) / n_g
                ra_avg = sum(float(x.get("runs_against") or 0.0) for x in prior_t) / n_g
                woba_avg = (
                    sum(float(x.get("team_woba") or lg_priors.lg_woba) for x in prior_t) / n_g
                )
                opp_woba_avg = (
                    sum(float(x.get("opp_woba") or lg_priors.lg_woba) for x in prior_t) / n_g
                )

                rf_s = shrink_scalar(
                    rf_avg, n_g, lg_priors.lg_runs_per_team_game, SHRINKAGE_K_TEAM_GAMES
                )
                ra_s = shrink_scalar(
                    ra_avg, n_g, lg_priors.lg_runs_per_team_game, SHRINKAGE_K_TEAM_GAMES
                )
                woba_s = shrink_scalar(woba_avg, n_g, lg_priors.lg_woba, SHRINKAGE_K_TEAM_GAMES)
                opp_w_s = shrink_scalar(
                    opp_woba_avg, n_g, lg_priors.lg_woba, SHRINKAGE_K_TEAM_GAMES
                )

                wins = sum(
                    1.0 if int(x.get("runs_for") or 0) > int(x.get("runs_against") or 0) else 0.0
                    for x in prior_t
                )
                win_pct = wins / n_g
                rf_tot = max(1.0, sum(float(x.get("runs_for") or 0.0) for x in prior_t))
                ra_tot = max(1.0, sum(float(x.get("runs_against") or 0.0) for x in prior_t))
                pythag_w = (rf_tot**PYTHAG_EXPONENT) / (
                    (rf_tot**PYTHAG_EXPONENT) + (ra_tot**PYTHAG_EXPONENT)
                )

                # BaseRuns / wOBA-implied runs luck over last 20 games
                last_20 = prior_t[-20:]
                rf_20 = sum(float(x.get("runs_for") or 0.0) for x in last_20) / len(last_20)
                woba_20 = sum(
                    float(x.get("team_woba") or lg_priors.lg_woba) for x in last_20
                ) / len(last_20)
                implied_runs_20 = lg_priors.lg_runs_per_team_game * (
                    (woba_20 / max(lg_priors.lg_woba, 0.200)) ** 1.65
                )

                row_out[f"{side}_team_runs_for_per_g_shrunk"] = rf_s
                row_out[f"{side}_team_runs_against_per_g_shrunk"] = ra_s
                row_out[f"{side}_team_run_diff_per_g_shrunk"] = rf_s - ra_s
                row_out[f"{side}_team_woba_shrunk"] = woba_s
                row_out[f"{side}_opp_woba_shrunk"] = opp_w_s
                row_out[f"{side}_pythag_luck_w30"] = win_pct - pythag_w
                row_out[f"{side}_baseruns_luck_w20"] = rf_20 - implied_runs_20

            # Rest days and travel distance from prior scheduled/completed game
            prior_sched = [
                sg
                for sg in sched_by_club.get(club_code, [])
                if (sg["start_time_utc"], int(sg.get("event_seq") or 1)) < (g_utc, g_seq)
            ]
            if not prior_sched:
                row_out[f"{side}_rest_days"] = 5.0
                if side == "away":
                    home_park = get_ballpark_metadata(away_code) or park
                    row_out["away_travel_km"] = haversine_distance_km(
                        home_park.latitude,
                        home_park.longitude,
                        park.latitude,
                        park.longitude,
                    )
            else:
                prev_g = prior_sched[-1]
                prev_d: date = prev_g.get("game_date") or prev_g["start_time_utc"].date()
                row_out[f"{side}_rest_days"] = float(min(max(0, (g_date - prev_d).days), 7))
                if side == "away":
                    prev_vid = str(prev_g.get("venue_id") or "")
                    prev_home = normalize_baseball_team(str(prev_g["home_team"]))
                    prev_park = (
                        BASEBALL_BALLPARKS.get(prev_vid.upper())
                        or get_ballpark_metadata(prev_home)
                        or park
                    )
                    row_out["away_travel_km"] = haversine_distance_km(
                        prev_park.latitude,
                        prev_park.longitude,
                        park.latitude,
                        park.longitude,
                    )

        out_rows.append(row_out)

    t_df = pl.DataFrame(out_rows)
    join_keys = ["game_id", "event_seq"] if "event_seq" in games_df.columns else ["game_id"]
    if "event_seq" not in games_df.columns:
        t_df = t_df.drop("event_seq")
    return games_df.join(t_df, on=join_keys, how="left")


def build_baseball_stage1_game_features(
    games_df: pl.DataFrame,
    team_logs_df: pl.DataFrame,
    pitcher_logs_df: pl.DataFrame | None = None,
    batter_logs_df: pl.DataFrame | None = None,
    *,
    cross_league_pitcher_priors: Mapping[int, TranslatedPitcherPrior] | None = None,
    cross_league_batter_priors: Mapping[int, TranslatedBatterPrior] | None = None,
    custom_lineups_by_game_team: Mapping[tuple[int, int, str], Sequence[int]] | None = None,
) -> pl.DataFrame:
    """Build the complete leak-free Gold Stage 1 baseball feature matrix (`MLB` & `NPB`)."""
    if games_df.is_empty():
        return games_df

    working = games_df
    if "sport" not in working.columns:
        working = working.with_columns(pl.lit("baseball").alias("sport"))
    if "event_id" not in working.columns and "game_id" in working.columns:
        working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    if "game_id" not in working.columns and "event_id" in working.columns:
        working = working.with_columns(pl.col("event_id").cast(pl.Int64).alias("game_id"))
    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))

    working = working.with_columns(
        pl.col("home_team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
        pl.col("away_team").map_elements(normalize_baseball_team, return_dtype=pl.Utf8),
    )
    if "league" not in working.columns:
        working = working.with_columns(
            pl.col("home_team")
            .map_elements(lambda t: resolve_team_league(t) or "MLB", return_dtype=pl.Utf8)
            .alias("league")
        )
    if "home_score" not in working.columns and "home_runs" in working.columns:
        working = working.with_columns(
            pl.col("home_runs").alias("home_score"),
            pl.col("away_runs").alias("away_score"),
        )

    sorted_games = sort_events_chronologically(working)

    # 1. Starting Pitcher Features
    p_logs = pitcher_logs_df if pitcher_logs_df is not None else pl.DataFrame()
    with_sp = resolve_game_starting_pitcher_features(
        sorted_games,
        p_logs,
        cross_league_priors=cross_league_pitcher_priors,
    )

    # 2. Compute Starter- & Venue-Adjusted Elo per league (MLB vs NPB)
    starter_xfip_map: dict[tuple[int, str], float] = {}
    venue_map: dict[int, str] = {}
    for r in with_sp.iter_rows(named=True):
        ev_id = int(r["event_id"])
        starter_xfip_map[(ev_id, str(r["home_team"]))] = float(r["home_sp_xfip_shrunk"])
        starter_xfip_map[(ev_id, str(r["away_team"]))] = float(r["away_sp_xfip_shrunk"])
        if r.get("venue_id"):
            venue_map[ev_id] = str(r["venue_id"])

    s_hook = BaseballStarterEloHook(starter_xfip_by_event_entity=starter_xfip_map)
    v_hook = BaseballVenueEloHook(venue_by_event_id=venue_map)

    mlb_part = with_sp.filter(pl.col("league") == "MLB")
    npb_part = with_sp.filter(pl.col("league") == "NPB")

    elo_frames: list[pl.DataFrame] = []
    if not mlb_part.is_empty():
        mlb_engine = PairwiseEloEngine(
            config=MLB_ELO_CONFIG, starter_hook=s_hook, venue_hook=v_hook
        )
        mlb_elo = mlb_engine.compute_ratings(mlb_part).with_columns(
            pl.lit(0.0, dtype=pl.Float64).alias("elo_draw_prob")
        )
        elo_frames.append(mlb_elo)

    if not npb_part.is_empty():
        npb_engine = PairwiseEloEngine(
            config=NPB_ELO_CONFIG, starter_hook=s_hook, venue_hook=v_hook
        )
        npb_elo = npb_engine.compute_ratings(npb_part)
        elo_frames.append(npb_elo)

    with_elo = (
        sort_events_chronologically(pl.concat(elo_frames, how="diagonal"))
        if elo_frames
        else with_sp
    )

    # 3. Bullpen 3-Day Fatigue & Quality Features
    with_bp = build_bullpen_pregame_features(with_elo, team_logs_df, p_logs)

    # 4. Platoon-Adjusted 1-9 Batting Order Features
    b_logs = batter_logs_df if batter_logs_df is not None else pl.DataFrame()
    with_lineup = build_platoon_lineup_strength_features(
        with_bp,
        b_logs,
        cross_league_priors=cross_league_batter_priors,
        custom_lineups_by_game_team=custom_lineups_by_game_team,
    )

    # 5. Team Rolling Efficiency, Pythagorean/BaseRuns Luck & Ballpark/Schedule Features
    with_team = _compute_team_history_features_for_games(with_lineup, team_logs_df)

    # 6. Matchup Differential Features (`Home - Away`)
    return with_team.with_columns(
        pl.col("elo_diff_pre").alias("diff_elo_pre"),
        # Note: Lower xFIP is better, so (away_sp_xfip - home_sp_xfip) > 0 favors Home
        (pl.col("away_sp_xfip_shrunk") - pl.col("home_sp_xfip_shrunk")).alias("diff_sp_xfip_edge"),
        (pl.col("away_sp_fip_shrunk") - pl.col("home_sp_fip_shrunk")).alias("diff_sp_fip_edge"),
        (pl.col("home_sp_k_minus_bb_pct_shrunk") - pl.col("away_sp_k_minus_bb_pct_shrunk")).alias(
            "diff_sp_k_minus_bb_pct"
        ),
        (pl.col("home_sp_expected_outs_ceiling") - pl.col("away_sp_expected_outs_ceiling")).alias(
            "diff_sp_expected_outs"
        ),
        (pl.col("away_bp_xfip_shrunk") - pl.col("home_bp_xfip_shrunk")).alias("diff_bp_xfip_edge"),
        (pl.col("home_bp_pitches_last_3d") - pl.col("away_bp_pitches_last_3d")).alias(
            "diff_bp_pitches_last_3d"
        ),
        (
            pl.col("home_high_leverage_arms_unavailable_count")
            - pl.col("away_high_leverage_arms_unavailable_count")
        ).alias("diff_hl_arms_unavailable"),
        (pl.col("home_lineup_platoon_woba") - pl.col("away_lineup_platoon_woba")).alias(
            "diff_lineup_platoon_woba"
        ),
        (pl.col("home_lineup_platoon_iso") - pl.col("away_lineup_platoon_iso")).alias(
            "diff_lineup_platoon_iso"
        ),
        (pl.col("home_lineup_platoon_k_pct") - pl.col("away_lineup_platoon_k_pct")).alias(
            "diff_lineup_platoon_k_pct"
        ),
        (pl.col("home_team_woba_shrunk") - pl.col("away_team_woba_shrunk")).alias("diff_team_woba"),
        (
            pl.col("home_team_run_diff_per_g_shrunk") - pl.col("away_team_run_diff_per_g_shrunk")
        ).alias("diff_team_run_diff"),
        (pl.col("home_pythag_luck_w30") - pl.col("away_pythag_luck_w30")).alias("diff_pythag_luck"),
        (pl.col("home_baseruns_luck_w20") - pl.col("away_baseruns_luck_w20")).alias(
            "diff_baseruns_luck"
        ),
        (pl.col("home_missing_regular_pa_woba") - pl.col("away_missing_regular_pa_woba")).alias(
            "diff_missing_regular_woba"
        ),
        (pl.col("home_rest_days") - pl.col("away_rest_days")).alias("diff_rest_days"),
        (pl.col("home_sp_velo_delta_mph") - pl.col("away_sp_velo_delta_mph")).alias(
            "diff_sp_velo_delta"
        ),
        (pl.col("home_sp_swstr_pct_shrunk") - pl.col("away_sp_swstr_pct_shrunk")).alias(
            "diff_sp_swstr_pct"
        ),
        (pl.col("home_lineup_xwoba") - pl.col("away_lineup_xwoba")).alias("diff_lineup_xwoba"),
        (pl.col("home_woba_minus_xwoba_luck") - pl.col("away_woba_minus_xwoba_luck")).alias(
            "diff_woba_minus_xwoba_regression"
        ),
    )
