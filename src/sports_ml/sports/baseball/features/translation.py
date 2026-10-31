"""Cross-league player translation priors (`NPB <-> MLB`) (`features/translation.py`).

Translates prior-season batting and pitching rate statistics when a player moves
between NPB and MLB (e.g., NPB posting to MLB or MLB/AAA foreign signing in NPB)
before applying Bayesian shrinkage (`core/shrinkage.py`).
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

from sports_ml.core.shrinkage import shrink_scalar
from sports_ml.sports.baseball.constants import (
    MLB_SABERMETRIC_PRIORS,
    NPB_SABERMETRIC_PRIORS,
    SHRINKAGE_K_BATTER_PA,
    SHRINKAGE_K_SP_BF,
    BaseballLeague,
    get_league_sabermetric_priors,
)

# Rookie replacement-level penalty relative to league average when a player has 0 prior history
ROOKIE_BATTER_WOBA_PENALTY: Final[float] = -0.020
ROOKIE_BATTER_ISO_PENALTY: Final[float] = -0.022
ROOKIE_BATTER_K_PCT_PENALTY: Final[float] = 0.028
ROOKIE_BATTER_BB_PCT_PENALTY: Final[float] = -0.010

ROOKIE_PITCHER_FIP_PENALTY: Final[float] = 0.40
ROOKIE_PITCHER_K_PCT_PENALTY: Final[float] = -0.022
ROOKIE_PITCHER_BB_PCT_PENALTY: Final[float] = 0.012


class TranslatedBatterPrior(BaseModel):
    """Translated pre-season / pre-game batting rate priors in the target league."""

    model_config = ConfigDict(frozen=True)

    source_league: BaseballLeague | None
    target_league: BaseballLeague
    is_cross_league_transfer: bool
    woba_prior: float
    iso_prior: float
    k_pct_prior: float
    bb_pct_prior: float


class TranslatedPitcherPrior(BaseModel):
    """Translated pre-season / pre-game pitching rate priors in the target league."""

    model_config = ConfigDict(frozen=True)

    source_league: BaseballLeague | None
    target_league: BaseballLeague
    is_cross_league_transfer: bool
    fip_prior: float
    xfip_prior: float
    k_pct_prior: float
    bb_pct_prior: float
    k_minus_bb_pct_prior: float
    gb_pct_prior: float


def get_rookie_batter_prior(target_league: BaseballLeague) -> TranslatedBatterPrior:
    """Return conservative rookie replacement-level batting prior for `target_league`."""
    lg = get_league_sabermetric_priors(target_league)
    return TranslatedBatterPrior(
        source_league=None,
        target_league=target_league,
        is_cross_league_transfer=False,
        woba_prior=max(0.220, lg.lg_woba + ROOKIE_BATTER_WOBA_PENALTY),
        iso_prior=max(0.050, lg.lg_iso + ROOKIE_BATTER_ISO_PENALTY),
        k_pct_prior=min(0.400, lg.lg_batter_k_pct + ROOKIE_BATTER_K_PCT_PENALTY),
        bb_pct_prior=max(0.030, lg.lg_batter_bb_pct + ROOKIE_BATTER_BB_PCT_PENALTY),
    )


def get_rookie_pitcher_prior(target_league: BaseballLeague) -> TranslatedPitcherPrior:
    """Return conservative rookie replacement-level pitching prior for `target_league`."""
    lg = get_league_sabermetric_priors(target_league)
    k_pct = max(0.100, lg.lg_sp_k_pct + ROOKIE_PITCHER_K_PCT_PENALTY)
    bb_pct = min(0.160, lg.lg_sp_bb_pct + ROOKIE_PITCHER_BB_PCT_PENALTY)
    return TranslatedPitcherPrior(
        source_league=None,
        target_league=target_league,
        is_cross_league_transfer=False,
        fip_prior=lg.lg_sp_fip + ROOKIE_PITCHER_FIP_PENALTY,
        xfip_prior=lg.lg_sp_xfip + ROOKIE_PITCHER_FIP_PENALTY,
        k_pct_prior=k_pct,
        bb_pct_prior=bb_pct,
        k_minus_bb_pct_prior=k_pct - bb_pct,
        gb_pct_prior=lg.lg_sp_gb_pct,
    )


def translate_npb_to_mlb_batter_prior(
    *,
    npb_woba: float,
    npb_iso: float,
    npb_k_pct: float,
    npb_bb_pct: float,
    npb_pa: int | float = 500.0,
) -> TranslatedBatterPrior:
    """Translate an NPB batter's prior-season rates into MLB equivalency priors.

    First shrinks the raw NPB rates toward the NPB league average by `npb_pa`
    (with `k = 240 PA`) so small-sample NPB call-ups do not extrapolate wildly,
    then applies empirical NPB -> MLB translation multipliers:
      - `wOBA_MLB = mu_MLB + 0.78 * (wOBA_NPB - mu_NPB) - 0.018`
      - `K%_MLB = 1.24 * K%_NPB`
      - `BB%_MLB = 0.86 * BB%_NPB`
      - `ISO_MLB = 0.82 * ISO_NPB`
    """
    npb_lg = NPB_SABERMETRIC_PRIORS
    mlb_lg = MLB_SABERMETRIC_PRIORS

    shrunk_npb_woba = shrink_scalar(npb_woba, npb_pa, npb_lg.lg_woba, SHRINKAGE_K_BATTER_PA)
    shrunk_npb_iso = shrink_scalar(npb_iso, npb_pa, npb_lg.lg_iso, SHRINKAGE_K_BATTER_PA)
    shrunk_npb_k = shrink_scalar(npb_k_pct, npb_pa, npb_lg.lg_batter_k_pct, SHRINKAGE_K_BATTER_PA)
    shrunk_npb_bb = shrink_scalar(
        npb_bb_pct, npb_pa, npb_lg.lg_batter_bb_pct, SHRINKAGE_K_BATTER_PA
    )

    mlb_woba = mlb_lg.lg_woba + 0.78 * (shrunk_npb_woba - npb_lg.lg_woba) - 0.018
    mlb_iso = 0.82 * shrunk_npb_iso
    mlb_k_pct = 1.24 * shrunk_npb_k
    mlb_bb_pct = 0.86 * shrunk_npb_bb

    return TranslatedBatterPrior(
        source_league="NPB",
        target_league="MLB",
        is_cross_league_transfer=True,
        woba_prior=float(min(max(mlb_woba, 0.220), 0.450)),
        iso_prior=float(min(max(mlb_iso, 0.040), 0.360)),
        k_pct_prior=float(min(max(mlb_k_pct, 0.060), 0.450)),
        bb_pct_prior=float(min(max(mlb_bb_pct, 0.025), 0.240)),
    )


def translate_mlb_to_npb_batter_prior(
    *,
    mlb_woba: float,
    mlb_iso: float,
    mlb_k_pct: float,
    mlb_bb_pct: float,
    mlb_pa: int | float = 400.0,
) -> TranslatedBatterPrior:
    """Translate an MLB batter's prior-season rates into NPB equivalency priors."""
    mlb_lg = MLB_SABERMETRIC_PRIORS
    npb_lg = NPB_SABERMETRIC_PRIORS

    shrunk_mlb_woba = shrink_scalar(mlb_woba, mlb_pa, mlb_lg.lg_woba, SHRINKAGE_K_BATTER_PA)
    shrunk_mlb_iso = shrink_scalar(mlb_iso, mlb_pa, mlb_lg.lg_iso, SHRINKAGE_K_BATTER_PA)
    shrunk_mlb_k = shrink_scalar(mlb_k_pct, mlb_pa, mlb_lg.lg_batter_k_pct, SHRINKAGE_K_BATTER_PA)
    shrunk_mlb_bb = shrink_scalar(
        mlb_bb_pct, mlb_pa, mlb_lg.lg_batter_bb_pct, SHRINKAGE_K_BATTER_PA
    )

    npb_woba = npb_lg.lg_woba + 0.85 * (shrunk_mlb_woba - mlb_lg.lg_woba) + 0.016
    npb_iso = 0.92 * shrunk_mlb_iso
    npb_k_pct = 0.85 * shrunk_mlb_k
    npb_bb_pct = 1.10 * shrunk_mlb_bb

    return TranslatedBatterPrior(
        source_league="MLB",
        target_league="NPB",
        is_cross_league_transfer=True,
        woba_prior=float(min(max(npb_woba, 0.210), 0.440)),
        iso_prior=float(min(max(npb_iso, 0.035), 0.340)),
        k_pct_prior=float(min(max(npb_k_pct, 0.050), 0.420)),
        bb_pct_prior=float(min(max(npb_bb_pct, 0.025), 0.240)),
    )


def translate_npb_to_mlb_pitcher_prior(
    *,
    npb_fip: float,
    npb_xfip: float | None = None,
    npb_k_pct: float,
    npb_bb_pct: float,
    npb_gb_pct: float = 0.455,
    npb_bf: int | float = 600.0,
) -> TranslatedPitcherPrior:
    """Translate an NPB pitcher's prior-season rates into MLB equivalency priors.

    Applies sample-size shrinkage against NPB baseline (`k = 180 BF`), then
    maps into the MLB run environment:
      - `K%_MLB = 0.91 * K%_NPB`
      - `BB%_MLB = 1.16 * BB%_NPB`
      - `FIP_MLB = mu_MLB_FIP + 0.85 * (FIP_NPB - mu_NPB_FIP) + 0.35`
    """
    npb_lg = NPB_SABERMETRIC_PRIORS
    mlb_lg = MLB_SABERMETRIC_PRIORS
    raw_xfip = npb_xfip if npb_xfip is not None else npb_fip

    shrunk_fip = shrink_scalar(npb_fip, npb_bf, npb_lg.lg_sp_fip, SHRINKAGE_K_SP_BF)
    shrunk_xfip = shrink_scalar(raw_xfip, npb_bf, npb_lg.lg_sp_xfip, SHRINKAGE_K_SP_BF)
    shrunk_k = shrink_scalar(npb_k_pct, npb_bf, npb_lg.lg_sp_k_pct, SHRINKAGE_K_SP_BF)
    shrunk_bb = shrink_scalar(npb_bb_pct, npb_bf, npb_lg.lg_sp_bb_pct, SHRINKAGE_K_SP_BF)
    shrunk_gb = shrink_scalar(npb_gb_pct, npb_bf, npb_lg.lg_sp_gb_pct, SHRINKAGE_K_SP_BF)

    mlb_k_pct = float(min(max(0.91 * shrunk_k, 0.100), 0.420))
    mlb_bb_pct = float(min(max(1.16 * shrunk_bb, 0.030), 0.180))
    mlb_fip = float(
        min(max(mlb_lg.lg_sp_fip + 0.85 * (shrunk_fip - npb_lg.lg_sp_fip) + 0.35, 2.20), 6.20)
    )
    mlb_xfip = float(
        min(max(mlb_lg.lg_sp_xfip + 0.85 * (shrunk_xfip - npb_lg.lg_sp_xfip) + 0.35, 2.30), 6.20)
    )
    mlb_gb_pct = float(min(max(0.96 * shrunk_gb, 0.250), 0.680))

    return TranslatedPitcherPrior(
        source_league="NPB",
        target_league="MLB",
        is_cross_league_transfer=True,
        fip_prior=mlb_fip,
        xfip_prior=mlb_xfip,
        k_pct_prior=mlb_k_pct,
        bb_pct_prior=mlb_bb_pct,
        k_minus_bb_pct_prior=mlb_k_pct - mlb_bb_pct,
        gb_pct_prior=mlb_gb_pct,
    )


def translate_mlb_to_npb_pitcher_prior(
    *,
    mlb_fip: float,
    mlb_xfip: float | None = None,
    mlb_k_pct: float,
    mlb_bb_pct: float,
    mlb_gb_pct: float = 0.430,
    mlb_bf: int | float = 400.0,
) -> TranslatedPitcherPrior:
    """Translate an MLB pitcher's prior-season rates into NPB equivalency priors."""
    mlb_lg = MLB_SABERMETRIC_PRIORS
    npb_lg = NPB_SABERMETRIC_PRIORS
    raw_xfip = mlb_xfip if mlb_xfip is not None else mlb_fip

    shrunk_fip = shrink_scalar(mlb_fip, mlb_bf, mlb_lg.lg_sp_fip, SHRINKAGE_K_SP_BF)
    shrunk_xfip = shrink_scalar(raw_xfip, mlb_bf, mlb_lg.lg_sp_xfip, SHRINKAGE_K_SP_BF)
    shrunk_k = shrink_scalar(mlb_k_pct, mlb_bf, mlb_lg.lg_sp_k_pct, SHRINKAGE_K_SP_BF)
    shrunk_bb = shrink_scalar(mlb_bb_pct, mlb_bf, mlb_lg.lg_sp_bb_pct, SHRINKAGE_K_SP_BF)
    shrunk_gb = shrink_scalar(mlb_gb_pct, mlb_bf, mlb_lg.lg_sp_gb_pct, SHRINKAGE_K_SP_BF)

    npb_k_pct = float(min(max(1.04 * shrunk_k, 0.100), 0.400))
    npb_bb_pct = float(min(max(0.92 * shrunk_bb, 0.030), 0.160))
    npb_fip = float(
        min(max(npb_lg.lg_sp_fip + 0.80 * (shrunk_fip - mlb_lg.lg_sp_fip) - 0.25, 1.80), 5.50)
    )
    npb_xfip = float(
        min(max(npb_lg.lg_sp_xfip + 0.80 * (shrunk_xfip - mlb_lg.lg_sp_xfip) - 0.25, 1.90), 5.50)
    )
    npb_gb_pct = float(min(max(1.03 * shrunk_gb, 0.250), 0.700))

    return TranslatedPitcherPrior(
        source_league="MLB",
        target_league="NPB",
        is_cross_league_transfer=True,
        fip_prior=npb_fip,
        xfip_prior=npb_xfip,
        k_pct_prior=npb_k_pct,
        bb_pct_prior=npb_bb_pct,
        k_minus_bb_pct_prior=npb_k_pct - npb_bb_pct,
        gb_pct_prior=npb_gb_pct,
    )


def resolve_batter_prior(
    *,
    target_league: BaseballLeague,
    prev_league: BaseballLeague | None = None,
    prev_woba: float | None = None,
    prev_iso: float | None = None,
    prev_k_pct: float | None = None,
    prev_bb_pct: float | None = None,
    prev_pa: int | float = 0.0,
) -> TranslatedBatterPrior:
    """Resolve a batter's pre-season prior (same-league, cross-league translated, or rookie)."""
    if (
        prev_league is None
        or prev_pa <= 0
        or prev_woba is None
        or prev_iso is None
        or prev_k_pct is None
        or prev_bb_pct is None
    ):
        return get_rookie_batter_prior(target_league)

    if prev_league == "NPB" and target_league == "MLB":
        return translate_npb_to_mlb_batter_prior(
            npb_woba=prev_woba,
            npb_iso=prev_iso,
            npb_k_pct=prev_k_pct,
            npb_bb_pct=prev_bb_pct,
            npb_pa=prev_pa,
        )
    if prev_league == "MLB" and target_league == "NPB":
        return translate_mlb_to_npb_batter_prior(
            mlb_woba=prev_woba,
            mlb_iso=prev_iso,
            mlb_k_pct=prev_k_pct,
            mlb_bb_pct=prev_bb_pct,
            mlb_pa=prev_pa,
        )

    # Same-league returning player: regress prior season toward target league mean
    lg = get_league_sabermetric_priors(target_league)
    return TranslatedBatterPrior(
        source_league=prev_league,
        target_league=target_league,
        is_cross_league_transfer=False,
        woba_prior=shrink_scalar(prev_woba, prev_pa, lg.lg_woba, SHRINKAGE_K_BATTER_PA),
        iso_prior=shrink_scalar(prev_iso, prev_pa, lg.lg_iso, SHRINKAGE_K_BATTER_PA),
        k_pct_prior=shrink_scalar(prev_k_pct, prev_pa, lg.lg_batter_k_pct, SHRINKAGE_K_BATTER_PA),
        bb_pct_prior=shrink_scalar(
            prev_bb_pct, prev_pa, lg.lg_batter_bb_pct, SHRINKAGE_K_BATTER_PA
        ),
    )


def resolve_pitcher_prior(
    *,
    target_league: BaseballLeague,
    prev_league: BaseballLeague | None = None,
    prev_fip: float | None = None,
    prev_xfip: float | None = None,
    prev_k_pct: float | None = None,
    prev_bb_pct: float | None = None,
    prev_gb_pct: float | None = None,
    prev_bf: int | float = 0.0,
) -> TranslatedPitcherPrior:
    """Resolve a pitcher's pre-season prior (same-league, cross-league translated, or rookie)."""
    if (
        prev_league is None
        or prev_bf <= 0
        or prev_fip is None
        or prev_k_pct is None
        or prev_bb_pct is None
    ):
        return get_rookie_pitcher_prior(target_league)

    lg_src = get_league_sabermetric_priors(prev_league)
    eff_xfip = prev_xfip if prev_xfip is not None else prev_fip
    eff_gb = prev_gb_pct if prev_gb_pct is not None else lg_src.lg_sp_gb_pct

    if prev_league == "NPB" and target_league == "MLB":
        return translate_npb_to_mlb_pitcher_prior(
            npb_fip=prev_fip,
            npb_xfip=eff_xfip,
            npb_k_pct=prev_k_pct,
            npb_bb_pct=prev_bb_pct,
            npb_gb_pct=eff_gb,
            npb_bf=prev_bf,
        )
    if prev_league == "MLB" and target_league == "NPB":
        return translate_mlb_to_npb_pitcher_prior(
            mlb_fip=prev_fip,
            mlb_xfip=eff_xfip,
            mlb_k_pct=prev_k_pct,
            mlb_bb_pct=prev_bb_pct,
            mlb_gb_pct=eff_gb,
            mlb_bf=prev_bf,
        )

    # Same-league returning pitcher
    lg = get_league_sabermetric_priors(target_league)
    fip_p = shrink_scalar(prev_fip, prev_bf, lg.lg_sp_fip, SHRINKAGE_K_SP_BF)
    xfip_p = shrink_scalar(eff_xfip, prev_bf, lg.lg_sp_xfip, SHRINKAGE_K_SP_BF)
    k_p = shrink_scalar(prev_k_pct, prev_bf, lg.lg_sp_k_pct, SHRINKAGE_K_SP_BF)
    bb_p = shrink_scalar(prev_bb_pct, prev_bf, lg.lg_sp_bb_pct, SHRINKAGE_K_SP_BF)
    gb_p = shrink_scalar(eff_gb, prev_bf, lg.lg_sp_gb_pct, SHRINKAGE_K_SP_BF)
    return TranslatedPitcherPrior(
        source_league=prev_league,
        target_league=target_league,
        is_cross_league_transfer=False,
        fip_prior=fip_p,
        xfip_prior=xfip_p,
        k_pct_prior=k_p,
        bb_pct_prior=bb_p,
        k_minus_bb_pct_prior=k_p - bb_p,
        gb_pct_prior=gb_p,
    )
