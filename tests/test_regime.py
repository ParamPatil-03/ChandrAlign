"""Acceptance tests for automatic matcher routing (MATCH-09, P2-T10).

Routing reads metadata only, so these tests need no rasters and no downloads.
The scene metadata is parsed from the REAL product labels we hold, because the
values that drive routing -- GSD, sub-solar azimuth, incidence -- are exactly
the ones that differ between the mission's nominal figures and the products
(OHRC is 0.30 m, not the registry's 0.25; IIRS is 97.15 m, not 80).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from chandralign import config
from chandralign.io.pds_label import parse_label
from chandralign.matching import regime

LABELS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labels"
OHRC = parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
TMC2 = parse_label(LABELS / "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml")
IIRS = parse_label(LABELS / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml")


def with_sun(meta, azimuth_deg, incidence_deg=30.0):
    """The same scene under a chosen sun, for sweeping one variable at a time."""
    return dataclasses.replace(meta, sub_solar_azimuth_deg=azimuth_deg,
                               solar_incidence_deg=incidence_deg)


# ---------------------------------------------------------------------------
# done_when, first half: OHRC <-> IIRS never goes direct
# ---------------------------------------------------------------------------
def test_ohrc_to_iirs_routes_to_the_cascade_never_direct():
    """Two independent teams measured 0 inliers attempting this pairing directly."""
    d = regime.select(OHRC, IIRS)
    assert d.route == "cascade"
    assert d.regime == "extreme_scale"
    # 290x, not the 324x the labels give: scale_precheck routes on each pixel's
    # best-supported size, both axes (IIRS is 99.995 x 79.515 m, not square 97.15).
    assert "290" in d.reason


def test_the_cascade_route_names_no_matcher():
    """Naming one would be a claim the cascade has not made yet: each step picks its own."""
    assert regime.select(OHRC, IIRS).matcher is None


def test_extreme_scale_is_settled_before_illumination():
    """A scale gap no matcher can cross makes the sun irrelevant, so it is checked first.

    OHRC <-> IIRS happens to have a SMALL azimuth difference (15.8 deg), which
    would otherwise route to the comfortable raw-pixel band. The scale gap has
    to win anyway.
    """
    d = regime.select(OHRC, IIRS)
    assert d.conditions.d_azimuth_deg < 30.0      # the easy illumination band
    assert d.route == "cascade"                   # and still not a direct match


def test_the_scale_threshold_comes_from_config_not_a_constant():
    """Member A's pre-check reads our threshold, so the two cannot drift apart."""
    threshold = None
    for rule in config.load("regimes").get("rules", []):
        threshold = (rule.get("when") or {}).get("scale_ratio_gte") or threshold
    assert threshold is not None, "configs/regimes.yaml must define scale_ratio_gte"
    assert regime.select(OHRC, TMC2).conditions.scale_ratio >= threshold


# ---------------------------------------------------------------------------
# Representation bands, measured in configs/regimes.yaml
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d_az, expected", [
    (0.0, "raw"), (30.0, "raw"), (59.0, "raw"),
    (61.0, "phase_congruency"), (90.0, "phase_congruency"), (134.0, "phase_congruency"),
    (150.0, "mind"), (180.0, "mind"),
])
def test_representation_follows_the_measured_bands(d_az, expected):
    assert regime.representation_for(d_az)[0] == expected


def test_small_illumination_difference_does_not_preprocess():
    """The counter-intuitive one: below ~60 deg, raw beats both descriptors.

    Measured at 30 deg -- raw +0.867 against phase congruency +0.705 and MIND
    +0.634. Both descriptors discard the intensity relationship, and while the
    suns are close that relationship is the strongest signal there is. A
    pipeline that always preprocesses is worse than one that never does.
    """
    d = regime.select(with_sun(TMC2, 100.0), with_sun(TMC2, 120.0))
    assert d.representation == "raw"


def test_the_hard_band_uses_phase_congruency():
    """At 90 deg raw correlation collapses to +0.010 while PC holds +0.465."""
    d = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 90.0))
    assert d.representation == "phase_congruency"


def test_an_opposed_sun_uses_mind_not_phase_congruency():
    """Difficulty is NOT monotonic in azimuth difference.

    180 deg is close to a straight contrast inversion, which MIND is invariant
    to (+0.975) and phase congruency merely survives (+0.860). Routing "more
    difference -> more invariance" would pick PC here and lose.
    """
    d = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 180.0))
    assert d.representation == "mind"


# ---------------------------------------------------------------------------
# Honesty: unknowns, and regimes nothing has solved
# ---------------------------------------------------------------------------
def test_unknown_sun_geometry_is_not_treated_as_a_matching_sun():
    """A missing angle must not silently become zero difference (rule H1)."""
    blind = dataclasses.replace(TMC2, sub_solar_azimuth_deg=None, solar_incidence_deg=None)
    d = regime.select(blind, with_sun(TMC2, 90.0))
    assert d.conditions.d_azimuth_deg is None
    assert d.expectation == "unknown"
    assert d.representation == "raw"          # best-understood path, not a guess
    assert not d.supported


def test_the_hard_band_is_in_the_middle_not_at_the_end():
    """The headline of MATCH-09, and the reason a min_ threshold cannot work.

    90 deg is harder than 180 deg: at 90 the lit and shadowed facets swap, while
    at 180 the scene approaches a contrast inversion that MIND-style descriptors
    are invariant to. Measured, best licence-clean matcher out of 10 seeds:
    6/10 at 90 deg against 10/10 at 180 (reports/illumination_sweep.json).

    Any rule of the form "unsolved at or above X deg" gets this backwards by
    construction, which is what the superseded min_d_azimuth_deg: 60.0 did.
    """
    middle = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 90.0))
    opposed = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 180.0))

    assert middle.expectation == "degraded" and not middle.supported
    assert opposed.expectation == "solved" and opposed.supported

    severity = {"solved": 0, "degraded": 1, "unsolved": 2}
    assert severity[middle.expectation] > severity[opposed.expectation], (
        "the 90 deg band must rank worse than the 180 deg band; if this ever "
        "reverses, the band structure has collapsed back to a monotonic rule")


def test_the_band_the_old_rule_called_unsolved_is_solved():
    """60 and 75 deg were flagged unsupported and are met on every seed.

    This is the bug MATCH-09 fixed: the selector was telling callers the system
    could not handle pairs it handles 10 times out of 10.
    """
    for azimuth in (60.0, 75.0):
        d = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, azimuth))
        assert d.expectation == "solved", f"{azimuth} deg regressed to {d.expectation}"
        assert d.supported


def test_a_matching_sun_pair_is_supported():
    d = regime.select(with_sun(TMC2, 100.0), with_sun(TMC2, 105.0))
    assert d.expectation == "solved" and d.supported


def test_a_band_names_the_matcher_that_earned_it():
    """`solved` with no attribution is useless to a caller.

    Every band above 60 deg was earned by minima-loftr, not by the default. A
    caller that runs the default there does not get the behaviour the band
    promises, so the decision has to say so.
    """
    d = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 180.0))
    assert "minima-loftr" in d.reason
    assert any("not by the default" in n for n in d.notes)
    # The default is still what runs. Read from config, never a literal: this line
    # once said "aliked-lightglue", passed on its own branch, and turned main red
    # the moment PR #13 changed default_matcher to eloftr in a separate merge.
    assert d.matcher == config.load("regimes")["default_matcher"]


def test_an_unsampled_azimuth_takes_the_worse_of_its_two_bands():
    """The sweep measured twelve angles; real pairs land between them.

    82 deg sits between the solved 0-75 band and the degraded 90-120 band.
    Interpolating upward would promote an angle nobody measured, so the rule
    fixed in the protocol before measuring is to take the worse neighbour.
    """
    d = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 82.0))
    assert d.expectation == "degraded"
    assert not d.supported
    assert "not sampled" in d.reason


def test_every_measured_band_is_reachable_and_declared():
    """No band may be dead config, and none may claim a verdict it cannot hold."""
    bands = config.load("regimes")["unsolved_illumination"]["bands"]
    assert bands, "the band list is empty: every pair would fall through"
    for band in bands:
        lo, hi = band["d_azimuth_deg"]
        assert 0.0 <= lo <= hi <= 180.0
        assert band["expectation"] in {"solved", "degraded", "unsolved"}
        # A band that claims better than `unsolved` must name who earned it.
        if band["expectation"] != "unsolved":
            assert band["solved_by"], f"{lo}-{hi} deg claims {band['expectation']} unattributed"
        mid = (lo + hi) / 2.0
        d = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, mid))
        assert d.expectation == band["expectation"]


def test_a_config_without_bands_falls_back_instead_of_calling_everything_solved(monkeypatch):
    """Deleting the bands must not silently make every regime supported.

    That failure is exactly what scripts/sabotage.py injects, so the reader
    keeps the pre-MATCH-09 shape as a fallback rather than trusting an empty
    list to mean "no problems".
    """
    real_load = config.load                     # capture BEFORE patching
    cfg = dict(real_load("regimes"))
    cfg["unsolved_illumination"] = {"min_d_azimuth_deg": 60.0}
    monkeypatch.setattr(config, "load",
                        lambda name="default": cfg if name == "regimes" else real_load(name))

    d = regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 180.0))
    assert d.expectation == "unsolved"


def test_every_decision_declares_its_evidence_as_synthetic():
    """Rule H6: a rule fitted on synthetic data says so wherever it is used.

    The bands have not been re-derived on real imagery yet, so nothing should be
    able to quote a routing decision as a measured result on real data.
    """
    for d in (regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 90.0)),
              regime.select(OHRC, IIRS)):
        assert d.evidence, "a decision with no stated evidence"
    assert "NOT yet validated on real" in regime.select(
        with_sun(TMC2, 0.0), with_sun(TMC2, 90.0)).evidence


def test_low_sun_is_recorded_but_not_routed_on():
    """A deliberate deviation from the written done_when, on measured grounds.

    P2-T10 asks that a polar-lighting pair route to a polar path. The synthetic
    sweep does not support it: at 80 deg incidence with a small azimuth
    difference, eLoFTR reached 0.118 px -- among the best results in the whole
    benchmark. Sun DIRECTION broke matching; sun HEIGHT did not. So low sun is
    recorded as a note for the report and left out of the routing decision until
    real polar imagery says otherwise. Flagged to the team rather than quietly
    dropped.
    """
    d = regime.select(with_sun(TMC2, 100.0, incidence_deg=85.0),
                      with_sun(TMC2, 105.0, incidence_deg=84.0))
    assert any("low sun" in n for n in d.notes)
    assert d.representation == "raw"          # not forced onto a descriptor


# ---------------------------------------------------------------------------
# Cross-modality
# ---------------------------------------------------------------------------
@pytest.fixture
def no_scale_gap(monkeypatch):
    """Take scale out of the decision so the MODALITY rule can be tested alone.

    These tests used to fake a small scale gap by overwriting IIRS's label GSD. That
    stopped working, correctly, when scale_precheck began routing on each product's
    best-supported pixel size: IIRS's corners still say ~89 m, and a label that its
    own corners contradict no longer decides the scale. So the pre-check is replaced
    outright with a non-extreme answer, which is what these tests actually mean.
    """
    from chandralign.io import instruments

    real = instruments.scale_precheck

    def close(a, b, threshold=None):
        check = real(a, b, threshold)
        return dataclasses.replace(check, ratio=1.5, extreme=False,
                                   reason="scale gap removed for this test")

    monkeypatch.setattr(instruments, "scale_precheck", close)


def test_cross_modal_pairs_are_routed_by_modality_not_by_sun(no_scale_gap):
    """A panchromatic/hyperspectral relationship is radiometric, not geometric."""
    d = regime.select(TMC2, IIRS)
    assert d.regime == "cross_modal"
    assert d.representation == "mind"
    assert d.candidates, "cross-modal candidates must be offered"


def test_cross_modal_candidates_are_not_presented_as_ranked(no_scale_gap):
    d = regime.select(TMC2, IIRS)
    assert d.matcher is None                    # no winner claimed
    assert any("benchmarked, not ranked" in n for n in d.notes)


def test_explain_reports_the_decision_without_hiding_the_expectation():
    """The 90 deg band, because that is the one a caller must not miss.

    This used to read 180 deg, back when a single threshold called an opposed
    sun the worst case. 180 deg is now solved, so asserting the warning there
    would assert the opposite of what is measured.
    """
    text = regime.explain(regime.select(with_sun(TMC2, 0.0), with_sun(TMC2, 90.0)))
    assert "not a supported configuration" in text
    assert "phase_congruency" in text and "evidence" in text
