"""Acceptance tests for the Part 2 foundation.

Each test names the feature ID it closes. A feature is not done until its test
is green (PLAN.md section 0.1 rule 3).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from chandralign import config, synth
from chandralign.contracts import (GeometryLayers, ImagePlane, Metrics,
                                   RegistrationResult, SceneMeta, TransformModel)
from chandralign.estimate import models, robust
from chandralign.matching import classical, filters
from chandralign.matching import licence

SHAPE = (384, 384)
CENTRE = (SHAPE[1] / 2.0, SHAPE[0] / 2.0)


def transform_rmse(model, h_true, shape=SHAPE, n=24) -> float:
    """Error of the RECOVERED transform against the TRUE one, over a grid.

    Deliberately not reprojection error on the estimator's own inliers: that
    measures self-consistency and can look excellent while the transform is
    wrong. Comparing against ground truth is the only honest check, and the
    synthetic harness is what makes it possible.
    """
    ys, xs = np.mgrid[0:shape[0]:complex(n), 0:shape[1]:complex(n)]
    grid = np.stack([xs.ravel(), ys.ravel()], 1)
    return float(np.sqrt(((models.apply(model, grid)
                           - synth.transform_points(h_true, grid)) ** 2).sum(1).mean()))


# ---------------------------------------------------------------------------
# Contract conformance (P0-T02)
# ---------------------------------------------------------------------------
def test_contract_dataclasses_round_trip():
    meta = SceneMeta(product_id="X", instrument="OHRC", mission="CH2", gsd_m=0.25,
                     n_bands=1, wavelength_nm=(500.0, 800.0), array_shape=(4, 4),
                     dtype="float32", corner_latlon=[], sub_solar_azimuth_deg=None,
                     solar_incidence_deg=None, emission_deg=None, phase_deg=None,
                     acquisition_utc=None, label_path=Path("a"), raster_path=Path("b"))
    plane = ImagePlane(array=np.zeros((4, 4), np.float32),
                       valid_mask=np.ones((4, 4), bool),
                       shadow_mask=np.zeros((4, 4), bool),
                       gsd_m=0.25, meta=meta, geo=GeometryLayers())
    assert dataclasses.asdict(plane)["gsd_m"] == 0.25
    assert dataclasses.asdict(meta)["instrument"] == "OHRC"


def test_metrics_default_to_none_not_zero():
    """Rule H1: unmeasured is None, never a plausible-looking number."""
    m = Metrics()
    for field in dataclasses.fields(m):
        if field.name == "source":
            continue
        assert getattr(m, field.name) is None, f"{field.name} must default to None"


def test_metrics_source_can_express_synthetic():
    """Rule H5: synthetic results must not be presentable as real measurements."""
    assert Metrics(source="synthetic").source == "synthetic"


def test_registration_result_requires_gates_to_be_meaningful():
    """Rule H4: gates is where control-gate outcomes live; empty means unverified."""
    ms = synth_matchset()
    res = RegistrationResult(matches=ms, inlier_mask=np.ones(len(ms.src_pts), bool),
                             model=None, metrics=Metrics(source="synthetic"),
                             confidence_tier="REJECTED", gates={})
    assert res.gates == {}, "an empty gates dict must be representable and is not a pass"


def synth_matchset():
    src, ref, _ = synth.make_pair(out_shape=(64, 64), n_craters=10, seed=1)
    return classical.match(src, ref, detector="sift")


# ---------------------------------------------------------------------------
# Synthetic harness (PREC-06 foundation)
# ---------------------------------------------------------------------------
def test_ground_truth_homography_is_exact():
    """The harness is only useful if its 'known answer' is actually known."""
    import cv2
    src, ref, h = synth.make_pair(out_shape=SHAPE, rot_deg=12.0,
                                  shift=(0.37, -0.62), seed=3)
    warped = cv2.warpPerspective(ref.array, h, SHAPE[::-1],
                                 flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT_101)
    m = slice(48, -48)
    corr = np.corrcoef(warped[m, m].ravel(), src.array[m, m].ravel())[0, 1]
    assert corr > 0.99, f"ground truth not exact (corr={corr:.4f})"


def test_opposite_sun_inverts_contrast():
    """Sun-angle change must move shadows, not merely rescale brightness.

    Under a gamma/brightness fake this correlation stays strongly positive. A
    negative correlation is the evidence that the harness poses the real
    problem: slopes facing toward and away from the Sun swap roles.
    """
    src, ref, _ = synth.make_pair(out_shape=SHAPE, sun_ref=(135, 45),
                                  sun_src=(315, 45), seed=3)
    m = slice(48, -48)
    corr = np.corrcoef(src.array[m, m].ravel(), ref.array[m, m].ravel())[0, 1]
    assert corr < 0.0, f"opposite sun should invert contrast, got corr={corr:.3f}"


def test_terrain_is_physically_plausible():
    """Guards the scale-dependent-amplitude bug: terrain must stay lunar."""
    _, ref, _ = synth.make_pair(out_shape=SHAPE, seed=3)
    slope = ref.geo.slope_deg
    assert np.percentile(slope, 99) < 45.0, "terrain far too steep to be lunar"
    assert ref.geo.dem_elev_m is not None, "geometry filter needs a DEM"


# ---------------------------------------------------------------------------
# MATCH-01: classical baseline
# ---------------------------------------------------------------------------
def test_sift_ransac_recovers_known_homography_under_1px():
    """MATCH-01 acceptance: recover a known homography to < 1 px."""
    src, ref, h = synth.make_pair(out_shape=SHAPE, rot_deg=12.0,
                                  shift=(0.37, -0.62), seed=3)
    ms = classical.match(src, ref, detector="sift")
    res = robust.estimate(ms.src_pts, ms.ref_pts, expected_scale=1.0, centre=CENTRE)
    assert res.ok, f"estimation failed: {res.notes}"
    err = transform_rmse(res.model, h)
    assert err < 1.0, f"transform RMSE {err:.3f} px exceeds the 1 px acceptance bar"


def test_ratio_test_and_mutual_nn_reduce_outliers():
    """MATCH-14: both filters must actually remove wrong matches."""
    src, ref, _ = synth.make_pair(out_shape=SHAPE, rot_deg=8.0, seed=5)
    loose = classical.match(src, ref, detector="sift", ratio=0.99, mutual=False)
    tight = classical.match(src, ref, detector="sift", ratio=0.75, mutual=True)
    assert len(tight.src_pts) < len(loose.src_pts)


def test_adaptive_ratio_tightens_on_repetitive_terrain():
    """MATCH-15: the threshold must respond to the repetitiveness score."""
    base = filters.adaptive_ratio(None)
    assert filters.adaptive_ratio(1.0) < base
    assert filters.adaptive_ratio(0.0) == pytest.approx(base)


# ---------------------------------------------------------------------------
# ALIGN-01/02 and CHECK-05: estimation and the scale sanity check
# ---------------------------------------------------------------------------
def test_scale_check_rejects_physically_impossible_transform():
    """CHECK-05 / failure mode #13.

    The matches are internally consistent at scale 1.0, but we tell the
    estimator the instruments imply 5.0. A transform can be geometrically
    perfect and still be physically wrong, and only the instrument registry
    knows that.
    """
    src, ref, _ = synth.make_pair(out_shape=SHAPE, seed=3)
    ms = classical.match(src, ref, detector="sift")
    res = robust.estimate(ms.src_pts, ms.ref_pts, expected_scale=5.0, centre=CENTRE)
    assert not res.ok
    assert robust.FM_SCALE_CONFUSION in res.failure_modes
    assert res.inlier_count == 0, "a rejected estimate must not report inliers"


def test_scale_check_abstains_without_instrument_information():
    """No expected scale means no verdict -- the check must not invent one."""
    src, ref, _ = synth.make_pair(out_shape=SHAPE, seed=3)
    ms = classical.match(src, ref, detector="sift")
    res = robust.estimate(ms.src_pts, ms.ref_pts, expected_scale=None, centre=CENTRE)
    assert res.ok and robust.FM_SCALE_CONFUSION not in res.failure_modes


def test_too_few_correspondences_is_a_rejection_not_a_crash():
    res = robust.estimate(np.zeros((3, 2)), np.zeros((3, 2)))
    assert not res.ok and res.model is None


def test_estimated_scale_matches_a_known_scaling():
    h = synth.homography(scale=2.0, rot_deg=0.0, centre=CENTRE)
    assert models.estimated_scale(h, at=CENTRE) == pytest.approx(2.0, rel=1e-6)


def test_compose_chains_two_hops():
    """MATCH-10 relies on composition being exact, not approximate."""
    a = TransformModel(kind="affine", matrix=synth.homography(scale=2.0), scale_estimated=2.0)
    b = TransformModel(kind="affine", matrix=synth.homography(scale=3.0), scale_estimated=3.0)
    c = models.compose(a, b)
    pts = np.array([[10.0, 20.0], [100.0, 50.0]])
    assert np.allclose(models.apply(c, pts), models.apply(b, models.apply(a, pts)))
    assert c.scale_estimated == pytest.approx(6.0)


def test_residual_structure_detects_spatial_organisation():
    """Random residuals score low; region-clustered residuals score high."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 384, size=(400, 2))
    assert models.residual_structure(pts, rng.normal(1.0, 0.2, 400)) < 0.2
    assert models.residual_structure(pts, np.where(pts[:, 0] < 192, 0.2, 4.0)) > 0.5


# ---------------------------------------------------------------------------
# CHECK-10: licence gate
# ---------------------------------------------------------------------------
def test_licence_gate_blocks_non_commercial_models_in_ship_mode():
    for name in ("superglue", "r2d2", "superpoint-lightglue"):
        with pytest.raises(licence.LicenceRestrictedError):
            licence.assert_allowed(name, ship_mode=True)


def test_licence_gate_catches_compound_model_names():
    """The component trap: SuperPoint inside a compound name still taints it."""
    assert licence.is_restricted("minima-superpoint-lightglue")
    assert not licence.is_restricted("aliked-lightglue")


def test_licence_gate_allows_benchmarking_when_ship_mode_is_off():
    licence.assert_allowed("superglue", ship_mode=False)


def test_configured_shippable_matchers_are_all_clean():
    """A mistake in regimes.yaml must not be able to defeat the gate."""
    shippable = licence.shippable()
    assert shippable, "no shippable matchers configured"
    assert all(not licence.is_restricted(n) for n in shippable)


def test_default_matcher_is_shippable():
    assert not licence.is_restricted(config.load("regimes")["default_matcher"])


# ---------------------------------------------------------------------------
# Instrument registry (feeds CHECK-05 and MATCH-10)
# ---------------------------------------------------------------------------
def test_gsd_ratios_reflect_the_real_instrument_gaps():
    assert config.gsd_ratio("OHRC", "NAC") == pytest.approx(0.5)
    assert config.gsd_ratio("TMC2", "TC") == pytest.approx(0.5)
    assert config.gsd_ratio("OHRC", "IIRS") == pytest.approx(0.25 / 80.0)


def test_cross_modality_detection():
    assert config.is_cross_modal("OHRC", "IIRS")
    assert not config.is_cross_modal("OHRC", "NAC")


def test_pairing_status_distinguishes_solved_from_open():
    assert config.pairing_status("OHRC", "NAC")["status"] == "solved"
    assert config.pairing_status("TMC2", "TC")["status"] == "open"
    assert config.pairing_status("OHRC", "IIRS")["status"] == "known_hard"
