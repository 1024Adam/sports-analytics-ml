"""NPB DIPS, Batted-Ball & Koshien/Provincial Micro-Context (`baseball/ingestion/npb_micro.py`).

Computes NPB-specific tactical and venue micro-features from game/player logs:
- Defense-Independent Pitching (`npb_dips_xwoba_proxy`) when radar tracking is absent.
- Small-ball / sacrifice bunt tactical propensity (`sac_bunt_rate`).
- Stadium park-interaction & provincial neutral-surface adjustment scores.
"""

from __future__ import annotations

import polars as pl

from sports_ml.core.temporal import pregame_rolling_mean, sort_events_chronologically
from sports_ml.sports.baseball.constants import (
    NPB_SABERMETRIC_PRIORS,
    get_ballpark_metadata,
    normalize_baseball_team,
)


def compute_npb_dips_and_tactical_micro_features(
    team_logs_df: pl.DataFrame,
    *,
    window: int = 20,
) -> pl.DataFrame:
    """Compute strictly pre-game `.shift(1)` NPB DIPS `xwOBA` proxy and small-ball features."""
    if team_logs_df.is_empty():
        return team_logs_df

    working = team_logs_df
    if "game_id" in working.columns and "event_id" not in working.columns:
        working = working.with_columns(pl.col("game_id").cast(pl.Int64).alias("event_id"))
    if "event_seq" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int32).alias("event_seq"))
    if "sac_bunts" not in working.columns:
        working = working.with_columns(pl.lit(1, dtype=pl.Int16).alias("sac_bunts"))

    sorted_df = sort_events_chronologically(working, extra_tiebreakers=["team"])
    priors = NPB_SABERMETRIC_PRIORS

    # DIPS xwOBA proxy from K%, BB%, HR/PA, and league BABIP (~0.295 in NPB)
    pa_safe = pl.col("pa_for").cast(pl.Float64).clip(lower_bound=1.0)
    k_rate = pl.col("k_for").cast(pl.Float64) / pa_safe
    bb_rate = pl.col("bb_for").cast(pl.Float64) / pa_safe
    hr_rate = pl.col("hr_for").cast(pl.Float64) / pa_safe
    bip_rate = (1.0 - k_rate - bb_rate - hr_rate).clip(lower_bound=0.10)
    dips_xwoba = (
        priors.w_ubb * bb_rate
        + priors.w_hr * hr_rate
        + bip_rate * 0.295 * (0.78 * priors.w_1b + 0.19 * priors.w_2b + 0.03 * priors.w_3b)
    )
    bunt_rate = pl.col("sac_bunts").cast(pl.Float64) / pa_safe

    enriched = sorted_df.with_columns(
        pl.coalesce([pl.col("team_xwoba").cast(pl.Float64), dips_xwoba]).alias("_eff_xwoba"),
        (pl.col("team_woba").cast(pl.Float64) - pl.coalesce([pl.col("team_xwoba"), dips_xwoba]))
        .fill_null(0.0)
        .alias("_woba_minus_xwoba"),
        bunt_rate.alias("_sac_bunt_rate"),
    )

    return enriched.with_columns(
        pregame_rolling_mean("_eff_xwoba", window=window, group_by="team")
        .fill_null(priors.lg_woba)
        .alias("pregame_team_xwoba"),
        pregame_rolling_mean("_woba_minus_xwoba", window=window, group_by="team")
        .fill_null(0.0)
        .alias("pregame_woba_minus_xwoba_luck"),
        pregame_rolling_mean("_sac_bunt_rate", window=window, group_by="team")
        .fill_null(0.022)
        .alias("pregame_sac_bunt_rate"),
    ).drop(["_eff_xwoba", "_woba_minus_xwoba", "_sac_bunt_rate"])


def compute_npb_venue_park_interaction(
    venue_id_or_team: str,
    *,
    lineup_iso: float,
    sp_gb_pct: float,
) -> float:
    """Compute park-interaction edge for high-HR (Jingu) vs groundball parks (Koshien)."""
    park = get_ballpark_metadata(normalize_baseball_team(venue_id_or_team))
    if park is None:
        park = get_ballpark_metadata(venue_id_or_team)
    if park is None:
        return 0.0
    avg_hr_factor = 0.5 * (park.hr_factor_lhb + park.hr_factor_rhb)
    # Power lineups benefit in high-HR stadiums; groundball pitchers thrive at Koshien/Nagoya
    return float((avg_hr_factor - 1.0) * (lineup_iso - 0.120) * 5.0 - (sp_gb_pct - 0.45) * 0.15)
