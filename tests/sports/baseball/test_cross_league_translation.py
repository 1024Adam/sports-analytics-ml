"""Tests for NPB <-> MLB Cross-League Player Translation Priors (`translation.py`)."""

from __future__ import annotations

import pytest

from sports_ml.core.shrinkage import shrink_scalar
from sports_ml.sports.baseball.constants import (
    SHRINKAGE_K_BATTER_PA,
    SHRINKAGE_K_SP_BF,
)
from sports_ml.sports.baseball.features.translation import (
    get_rookie_batter_prior,
    get_rookie_pitcher_prior,
    resolve_batter_prior,
    resolve_pitcher_prior,
    translate_mlb_to_npb_batter_prior,
    translate_mlb_to_npb_pitcher_prior,
    translate_npb_to_mlb_batter_prior,
)


def test_npb_ace_posted_to_mlb_beats_generic_rookie_prior() -> None:
    """Verify an NPB ace pitcher posted to MLB receives a stronger prior than an MLB rookie."""
    mlb_rookie = get_rookie_pitcher_prior("MLB")

    # Elite NPB starter (e.g., 1.85 FIP, 29% K%, 5.5% BB% over 700 BF in NPB)
    posted_ace = resolve_pitcher_prior(
        target_league="MLB",
        prev_league="NPB",
        prev_fip=1.85,
        prev_xfip=2.05,
        prev_k_pct=0.290,
        prev_bb_pct=0.055,
        prev_gb_pct=0.480,
        prev_bf=700,
    )

    assert posted_ace.is_cross_league_transfer is True
    assert posted_ace.source_league == "NPB"
    assert posted_ace.target_league == "MLB"
    assert posted_ace.fip_prior < mlb_rookie.fip_prior - 0.80
    assert posted_ace.xfip_prior < mlb_rookie.xfip_prior - 0.70
    assert posted_ace.k_pct_prior > mlb_rookie.k_pct_prior
    assert posted_ace.bb_pct_prior < mlb_rookie.bb_pct_prior
    assert posted_ace.k_minus_bb_pct_prior > mlb_rookie.k_minus_bb_pct_prior


def test_npb_slugger_posted_to_mlb_and_small_sample_shrinkage_guard() -> None:
    """Verify NPB power hitter translation and guard against tiny NPB sample sizes."""
    mlb_rookie = get_rookie_batter_prior("MLB")

    # Full-season NPB star (600 PA, .410 wOBA, .280 ISO, 18% K%, 14% BB%)
    full_season_star = translate_npb_to_mlb_batter_prior(
        npb_woba=0.410,
        npb_iso=0.280,
        npb_k_pct=0.180,
        npb_bb_pct=0.140,
        npb_pa=600,
    )
    # Tiny-sample NPB call-up with identical raw rates over only 15 PA
    tiny_sample_callup = translate_npb_to_mlb_batter_prior(
        npb_woba=0.410,
        npb_iso=0.280,
        npb_k_pct=0.180,
        npb_bb_pct=0.140,
        npb_pa=15,
    )

    assert full_season_star.woba_prior > mlb_rookie.woba_prior + 0.045
    assert full_season_star.iso_prior > mlb_rookie.iso_prior + 0.050
    # Tiny sample is shrunk toward NPB mean first, so its MLB prior stays much closer to average
    assert full_season_star.woba_prior > tiny_sample_callup.woba_prior + 0.030


def test_mlb_to_npb_foreign_signing_translations() -> None:
    """Verify MLB -> NPB batter and pitcher translations for foreign signings (suketto)."""
    npb_rookie_b = get_rookie_batter_prior("NPB")
    npb_rookie_p = get_rookie_pitcher_prior("NPB")

    mlb_batter_to_npb = translate_mlb_to_npb_batter_prior(
        mlb_woba=0.335,
        mlb_iso=0.200,
        mlb_k_pct=0.230,
        mlb_bb_pct=0.095,
        mlb_pa=450,
    )
    assert mlb_batter_to_npb.is_cross_league_transfer is True
    assert mlb_batter_to_npb.target_league == "NPB"
    assert mlb_batter_to_npb.woba_prior > npb_rookie_b.woba_prior
    assert mlb_batter_to_npb.iso_prior > npb_rookie_b.iso_prior

    mlb_pitcher_to_npb = translate_mlb_to_npb_pitcher_prior(
        mlb_fip=3.80,
        mlb_xfip=3.85,
        mlb_k_pct=0.240,
        mlb_bb_pct=0.075,
        mlb_bf=500,
    )
    assert mlb_pitcher_to_npb.is_cross_league_transfer is True
    assert mlb_pitcher_to_npb.target_league == "NPB"
    assert mlb_pitcher_to_npb.fip_prior < npb_rookie_p.fip_prior


def test_bayesian_shrinkage_decay_as_current_season_exposure_grows() -> None:
    """Verify translated prior weight decays smoothly toward 0 as current-league BF/PA -> inf."""
    prior = resolve_pitcher_prior(
        target_league="MLB",
        prev_league="NPB",
        prev_fip=2.10,
        prev_k_pct=0.280,
        prev_bb_pct=0.060,
        prev_bf=650,
    )

    # Opening Day (0 BF in MLB): evaluates strictly to the translated prior
    day1_fip = shrink_scalar(None, 0, prior.fip_prior, SHRINKAGE_K_SP_BF)
    assert pytest.approx(day1_fip, abs=1e-9) == prior.fip_prior

    # Suppose the pitcher struggles in MLB with a 4.80 FIP:
    # After 90 BF (~3.5 starts), blended between prior and 4.80
    mid_fip = shrink_scalar(4.80, 90, prior.fip_prior, SHRINKAGE_K_SP_BF)
    # After 700 BF (full season), much closer to 4.80
    full_fip = shrink_scalar(4.80, 700, prior.fip_prior, SHRINKAGE_K_SP_BF)
    # After 50,000 BF, converges to 4.80
    asymptotic_fip = shrink_scalar(4.80, 50_000, prior.fip_prior, SHRINKAGE_K_SP_BF)

    assert prior.fip_prior < mid_fip < full_fip < asymptotic_fip <= 4.80
    assert pytest.approx(asymptotic_fip, abs=0.02) == 4.80

    # Same-league returning batter check
    returning_mlb = resolve_batter_prior(
        target_league="MLB",
        prev_league="MLB",
        prev_woba=0.370,
        prev_iso=0.230,
        prev_k_pct=0.190,
        prev_bb_pct=0.110,
        prev_pa=600,
    )
    assert returning_mlb.is_cross_league_transfer is False
    assert returning_mlb.woba_prior > 0.340
    day1_woba = shrink_scalar(None, 0, returning_mlb.woba_prior, SHRINKAGE_K_BATTER_PA)
    assert pytest.approx(day1_woba, abs=1e-9) == returning_mlb.woba_prior
