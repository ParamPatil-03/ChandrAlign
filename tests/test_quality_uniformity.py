"""Tests for spatial uniformity (ALIGN-04..07) and quality gating (CHECK-07/08)."""
from __future__ import annotations

import numpy as np
import pytest

from chandralign.contracts import TransformModel
from chandralign.evaluate import quality
from chandralign.refine import uniformity

SHAPE = (384, 384)


# ---------------------------------------------------------------------------
# ALIGN-04..07: spatial uniformity
# ---------------------------------------------------------------------------
def test_grid_thinning_caps_per_cell_domination():
    """The point of thinning: stop one dense region dictating the RANSAC fit."""
    rng = np.random.default_rng(0)
    clustered = rng.uniform(0, 90, size=(400, 2))
    res = uniformity.enforce(clustered, rng.random(400), SHAPE, grid=8, top_k=6)
    assert res.occupancy.max() <= 6
    assert res.keep_mask.sum() < 400


def test_coverage_uses_image_extent_not_point_extent():
    """Guards a circularity bug: normalising by the points' own bounding box
    would spread a corner cluster across the whole grid and score it perfect."""
    rng = np.random.default_rng(0)
    corner = rng.uniform(0, 90, size=(300, 2))
    assert uniformity.coverage_of(corner, SHAPE, grid=8) < 0.2
    spread = rng.uniform(0, 384, size=(600, 2))
    assert uniformity.coverage_of(spread, SHAPE, grid=8) > 0.9


def test_thinning_alone_cannot_manufacture_coverage():
    """Selection is a subset operation, so coverage can never rise from it.

    This documents a real subtlety in the P2-T07 acceptance criterion: the
    promised jump from <0.15 to >0.6 coverage on a corner-clustered set is only
    achievable via the step-3 refill pass, which must actually FIND new matches
    in empty cells. Thinning cannot invent points where none were detected.
    """
    rng = np.random.default_rng(0)
    clustered = rng.uniform(0, 90, size=(400, 2))
    before = uniformity.coverage_of(clustered, SHAPE, grid=8)
    after = uniformity.enforce(clustered, rng.random(400), SHAPE, grid=8, top_k=6).coverage
    assert after <= before + 1e-9


def test_refill_raises_coverage_and_is_recorded_separately():
    """Coverage earned at a looser threshold must be reported as such."""
    rng = np.random.default_rng(0)
    clustered = rng.uniform(0, 90, size=(400, 2))
    res = uniformity.enforce(clustered, rng.random(400), SHAPE, grid=8, top_k=6)
    before = res.coverage
    res = uniformity.merge_refill(res.keep_mask, rng.uniform(100, 384, size=(150, 2)),
                                  res, SHAPE)
    assert res.coverage > before
    assert any("looser threshold" in n for n in res.notes)


def test_empty_cell_regions_are_inside_the_image():
    rng = np.random.default_rng(0)
    res = uniformity.enforce(rng.uniform(0, 90, size=(200, 2)), rng.random(200),
                             SHAPE, grid=8, top_k=4)
    for y0, y1, x0, x1 in uniformity.empty_cell_regions(res, SHAPE):
        assert 0 <= y0 < y1 <= SHAPE[0] and 0 <= x0 < x1 <= SHAPE[1]


def test_delaunay_gap_flags_a_hole():
    """Grid occupancy can look healthy while a band holds nothing."""
    rng = np.random.default_rng(0)
    left = rng.uniform([0, 0], [120, 384], size=(200, 2))
    right = rng.uniform([264, 0], [384, 384], size=(200, 2))
    gap = uniformity.max_delaunay_gap(np.vstack([left, right]), SHAPE)
    dense = uniformity.max_delaunay_gap(rng.uniform(0, 384, size=(400, 2)), SHAPE)
    assert gap > dense, f"holed set ({gap:.1f}) should score worse than uniform ({dense:.1f})" 


def test_adaptive_grid_scales_with_match_count():
    assert uniformity.adaptive_grid(SHAPE, 20) < uniformity.adaptive_grid(SHAPE, 2000)


# ---------------------------------------------------------------------------
# CHECK-07/08: confidence tiers and quality gating
# ---------------------------------------------------------------------------
def good_model():
    return TransformModel(kind="affine", matrix=np.eye(3), scale_estimated=1.0)


def test_strong_signals_earn_high_tier():
    v = quality.assess(inlier_count=500, inlier_ratio=0.9, spatial_coverage=0.9,
                       model=good_model())
    assert v.tier == "HIGH" and v.accepted


def test_a_single_weak_signal_limits_the_tier():
    """Worst-of, not average: a big inlier count must not carry a poor ratio.

    This is the exact shape of the false confidence found in the benchmark --
    hundreds of matches, 47 inliers, and a 13 px error.
    """
    v = quality.assess(inlier_count=500, inlier_ratio=0.08, spatial_coverage=0.9,
                       model=good_model())
    assert v.tier == "REJECTED"
    assert v.limiting_signal == "inlier_ratio"


def test_failed_control_gate_is_an_outright_rejection():
    """Rule H3/H4: a failed gate is invalid, not merely low confidence."""
    v = quality.assess(inlier_count=500, inlier_ratio=0.9, spatial_coverage=0.9,
                       model=good_model(), gates={"null_constant_grey": False})
    assert v.tier == "REJECTED" and v.limiting_signal == "control_gates"


def test_missing_gates_can_be_required():
    """A result with no gates run is unverified (rule H4)."""
    v = quality.assess(inlier_count=500, inlier_ratio=0.9, spatial_coverage=0.9,
                       model=good_model(), gates=None, require_gates=True)
    assert v.tier == "REJECTED"


def test_scale_failure_is_an_outright_rejection():
    v = quality.assess(inlier_count=500, inlier_ratio=0.9, spatial_coverage=0.9,
                       model=good_model(), scale_ok=False)
    assert v.tier == "REJECTED"
    assert quality.FM_SCALE_CONFUSION in v.failure_modes


def test_geometry_check_rejects_a_mirrored_transform():
    """A nadir orbital pair cannot produce a reflection."""
    mirror = np.diag([-1.0, 1.0, 1.0])
    ok, msg, _ = quality.geometry_validity(TransformModel(kind="affine", matrix=mirror))
    assert not ok and "mirror" in msg


def test_geometry_check_rejects_extreme_stretch():
    stretch = np.diag([10.0, 1.0, 1.0])
    ok, _, stats = quality.geometry_validity(TransformModel(kind="affine", matrix=stretch))
    assert not ok and stats["anisotropy"] == pytest.approx(10.0, rel=1e-6)


def test_geometry_check_abstains_without_a_matrix():
    ok, msg, _ = quality.geometry_validity(None)
    assert ok and "skipped" in msg


def test_unmeasured_signal_cannot_justify_high_confidence():
    """None means unmeasured, and unmeasured must never read as good (rule H1)."""
    v = quality.assess(inlier_count=500, inlier_ratio=None, spatial_coverage=0.9,
                       model=good_model())
    assert v.tier in ("LOW", "MEDIUM") and v.tier != "HIGH"


# ---------------------------------------------------------------------------
# CHECK-08 regression: the gate must keep rejecting the runs it was fitted to.
#
# These tuples are RECORDED SIGNAL VALUES from 60 runs of
# scripts/bench_matchers.py (4 matchers x 5 illumination cases x seeds 3/11/29),
# stored so the boundary cannot silently drift when thresholds are next touched.
# The RMSE column is ground truth against the known synthetic transform; it is
# never passed to assess(), it only says which side of the line a run belongs on.
#
# Only the LOAD-BEARING bad runs are listed -- those a single signal catches on
# its own. Runs vetoed redundantly by two or three signals prove nothing about
# where the boundary sits.
# ---------------------------------------------------------------------------

# (inliers, ratio, coverage, true_rmse_px, label)
BAD_ONLY_RATIO_CATCHES = [
    (43, 0.0837, 0.3281, 4.1752, "xfeat/sun_+90deg/s3"),
    (47, 0.0772, 0.4219, 13.0804, "xfeat/opposite_sun/s3"),
    (34, 0.0582, 0.3125, 15.3413, "xfeat/opposite_sun/s11"),
    (56, 0.0787, 0.5156, 10.9225, "xfeat/opposite_sun/s29"),
    (52, 0.1507, 0.3906, 3.7145, "eloftr/sun_+90deg/s3"),
    (105, 0.1977, 0.5156, 5.1131, "eloftr/opposite_sun/s3"),
    (82, 0.2087, 0.4062, 4.7220, "eloftr/opposite_sun/s11"),
    (41, 0.2000, 0.3438, 4.9426, "eloftr/sun_+90deg/s29"),
]
BAD_ONLY_COVERAGE_CATCHES = [
    (18, 0.5000, 0.1406, 5.4551, "aliked-lightglue/sun_+90deg/s29"),
    (11, 0.5238, 0.1406, 4.0177, "aliked-lightglue/opposite_sun/s29"),
]
# The single tightest correct run in the whole sweep: lowest inlier count,
# lowest ratio AND lowest coverage of all 33 good runs, all at once. The entire
# good-side margin rests on this one run, so it is pinned deliberately.
TIGHTEST_GOOD = (12, 0.4286, 0.1719, 0.924, "sift/low_sun_10deg/s3")


@pytest.mark.parametrize("inl,ratio,cov,rmse,label",
                         BAD_ONLY_RATIO_CATCHES + BAD_ONLY_COVERAGE_CATCHES)
def test_known_false_positives_stay_rejected(inl, ratio, cov, rmse, label):
    """Every one of these was once ACCEPTED while being 3.4-15.3 px wrong."""
    v = quality.assess(inlier_count=inl, inlier_ratio=ratio,
                       spatial_coverage=cov, model=good_model())
    assert v.tier == "REJECTED", (
        f"{label} is {rmse} px wrong but the gate returned {v.tier} "
        f"(limited by {v.limiting_signal})")


def test_the_tightest_correct_run_is_still_accepted():
    """Guards the other direction: rejecting everything is not a passing gate."""
    inl, ratio, cov, rmse, label = TIGHTEST_GOOD
    v = quality.assess(inlier_count=inl, inlier_ratio=ratio,
                       spatial_coverage=cov, model=good_model())
    assert v.accepted, f"{label} is correct to {rmse} px but was rejected on {v.limiting_signal}"


def test_inlier_ratio_alone_would_leak_the_clustered_failures():
    """Why the gate is a conjunction, part 1.

    These two runs have ratios of 0.50 and 0.52 -- better than the tightest
    CORRECT run in the sweep (0.4286). Any ratio threshold low enough to keep
    that correct run would accept both of these, and both are 4-5 px wrong.
    Coverage is the only signal that separates them.
    """
    good_ratio = TIGHTEST_GOOD[1]
    for inl, ratio, cov, rmse, label in BAD_ONLY_COVERAGE_CATCHES:
        assert ratio > good_ratio, f"{label} premise broken"
        assert cov < TIGHTEST_GOOD[2], f"{label} must be separable by coverage"


def test_spatial_coverage_alone_would_leak_the_illumination_failures():
    """Why the gate is a conjunction, part 2.

    These runs cover 0.28-0.52 of the frame -- better than the tightest CORRECT
    run (0.1719). A coverage-only gate accepts all of them. Their ratios
    (0.058-0.209) are what gives them away.
    """
    good_cov = TIGHTEST_GOOD[2]
    for inl, ratio, cov, rmse, label in BAD_ONLY_RATIO_CATCHES:
        assert cov > good_cov, f"{label} premise broken"
        assert ratio < TIGHTEST_GOOD[1], f"{label} must be separable by ratio"


def test_thresholds_are_declared_as_synthetic_in_config():
    """Rule H6: a fitted number must carry the provenance of its fit.

    If someone retunes these on real data they must also rewrite the comment
    block, so this asserts the comment still describes what the file contains.
    """
    from pathlib import Path
    text = Path(__file__).resolve().parents[1].joinpath("configs/default.yaml").read_text(encoding="utf-8")
    assert "SYNTHETIC" in text and "bench_matchers.py" in text


# ---------------------------------------------------------------------------
# Holes found by scripts/sabotage.py, now pinned.
#
# Both of these passed every existing test while the code was deliberately
# broken, which is the only way they were found. A green suite says the tests
# pass; it does not say they would fail if the code were wrong.
# ---------------------------------------------------------------------------
def test_assess_actually_applies_the_geometry_check():
    """The helper was tested; the WIRING into assess() was not.

    test_geometry_check_rejects_a_mirrored_transform calls geometry_validity()
    directly, so deleting the call inside assess() left every test green while
    mirrored transforms sailed through. A unit test of a function says nothing
    about whether anything calls it.
    """
    mirror = TransformModel(kind="affine", matrix=np.diag([-1.0, 1.0, 1.0]))
    v = quality.assess(inlier_count=500, inlier_ratio=0.95, spatial_coverage=0.95,
                       model=mirror)
    assert v.tier == "REJECTED", "a reflection reached the caller as an accepted transform"
    assert v.limiting_signal == "geometry"
    assert quality.FM_FALSE_CORRESPONDENCE in v.failure_modes


def test_assess_applies_the_anisotropy_limit_too():
    stretch = TransformModel(kind="affine", matrix=np.diag([10.0, 1.0, 1.0]))
    v = quality.assess(inlier_count=500, inlier_ratio=0.95, spatial_coverage=0.95,
                       model=stretch)
    assert v.tier == "REJECTED" and v.limiting_signal == "geometry"


def test_a_tier_rejection_names_a_failure_mode():
    """CHECK-07 requires REJECTED to arrive with failure_modes filled in.

    Nothing asserted it for the graded-signal path, so dropping the
    failure-mode append left the suite green and the failure log empty --
    a rejection no downstream report could explain.
    """
    v = quality.assess(inlier_count=4, inlier_ratio=0.02, spatial_coverage=0.02,
                       model=good_model())
    assert v.tier == "REJECTED"
    assert v.failure_modes, "a rejection with no failure mode cannot be reported or triaged"
    assert quality.FM_FALSE_CORRESPONDENCE in v.failure_modes


def test_an_ungated_rejection_also_names_a_failure_mode():
    """CHECK-07's contract holds on the no-gates path too: REJECTED always says why."""
    v = quality.assess(inlier_count=500, inlier_ratio=0.9, spatial_coverage=0.9, model=good_model(),
                       gates=None, require_gates=True)
    assert v.tier == "REJECTED" and quality.FM_UNVERIFIED_EVALUATION in v.failure_modes
