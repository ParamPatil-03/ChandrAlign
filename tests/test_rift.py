"""RIFT2-style matcher (MATCH-06). What it must do, and the conventions it depends on.

Every number asserted here was measured first (see matching/rift.py's RiftParams
comment); the tests pin behaviour so a later change cannot silently undo it.
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest

from chandralign import synth
from chandralign.estimate import models, robust
from chandralign.matching import classical, rift

warnings.filterwarnings("ignore")
SHAPE = (512, 512)


def rmse(ms, H) -> float | None:
    res = robust.estimate(ms.src_pts, ms.ref_pts)
    if res.model is None or not res.ok:
        return None
    g = np.linspace(0, SHAPE[0] - 1, 10)
    gx, gy = np.meshgrid(g, g)
    p = np.c_[gx.ravel(), gy.ravel()]
    return float(np.sqrt(np.mean(np.sum((models.apply(res.model, p) - synth.transform_points(H, p)) ** 2, axis=1))))


def pair(**kw):
    return synth.make_pair(out_shape=SHAPE, n_craters=60, seed=3, **kw)


# ---------------------------------------------------------------------------
# Conventions -- each one was a real bug or a real trap
# ---------------------------------------------------------------------------
def test_phasepack_outer_index_is_orientation():
    """phasepack's docstring contradicts itself on EO's nesting. A vertical-stripe
    pattern excites one orientation: it must show up on the OUTER index."""
    import phasepack
    x = np.tile(np.sin(np.arange(128) * 2 * np.pi / 8), (128, 1))
    *_, EO, _ = phasepack.phasecong(x, nscale=4, norient=6)
    assert len(EO) == 6 and len(EO[0]) == 4
    by_outer = [np.mean([np.abs(e).mean() for e in EO[o]]) for o in range(6)]
    assert int(np.argmax(by_outer)) == 0


def test_rotation_raises_the_mim_index_the_way_the_shift_assumes():
    """Turning the image +30 deg (OpenCV) raises the MIM index by +1 at the same
    physical point -- which is why the shift SUBTRACTS k (_INDEX_SHIFT_SIGN = -1)."""
    p = rift.RiftParams()
    src, _, _ = pair()
    img = rift._normalise(src.array)
    R = cv2.getRotationMatrix2D((255.5, 255.5), 30.0, 1.0)
    rot = cv2.warpAffine(img.astype(np.float32), R, SHAPE[::-1], flags=cv2.INTER_CUBIC)
    _, mim0, a0, _ = rift.phase_maps(img, p)
    _, mim1, _, _ = rift.phase_maps(rot.astype(np.float64), p)
    pts = np.random.default_rng(0).uniform(170, 342, (400, 2))
    q = (R @ np.c_[pts, np.ones(len(pts))].T).T
    i0, i1 = np.rint(pts).astype(int), np.rint(q).astype(int)
    strong = a0[i0[:, 1], i0[:, 0]] > np.percentile(a0, 60)
    d = (mim1[i1[:, 1], i1[:, 0]] - mim0[i0[:, 1], i0[:, 0]]) % 6
    assert np.bincount(d[strong], minlength=6).argmax() == 1
    # (No assertion on _INDEX_SHIFT_SIGN itself: that would only restate the
    # constant. test_published_variant_uses_the_measured_index_shift checks
    # that the sign actually works.)


# ---------------------------------------------------------------------------
# What it must do
# ---------------------------------------------------------------------------
def test_contrast_inversion_is_matched_exactly():
    """Inverting brightness leaves every filter amplitude, and so the MIM, unchanged."""
    src, _, _ = pair()
    inv = synth.ImagePlane(array=(1.0 - src.array).astype(np.float32), valid_mask=src.valid_mask.copy(),
                           shadow_mask=src.shadow_mask.copy(), gsd_m=src.gsd_m, meta=src.meta, geo=src.geo)
    ms = rift.match(src, inv)
    res = robust.estimate(ms.src_pts, ms.ref_pts)
    assert res.ok and np.allclose(res.model.matrix, np.eye(3), atol=1e-3)
    assert rmse(classical.match(src, inv, detector="sift"), np.eye(3)) is None      # SIFT cannot


@pytest.mark.parametrize("rot", [45.0, 75.0, 105.0])
def test_default_variant_holds_at_non_multiples_of_30_degrees(rot):
    """The published 6-bin MIM failed at all three; the continuous variant must not."""
    src, ref, H = pair(rot_deg=rot)
    e = rmse(rift.match(src, ref), H)
    assert e is not None and e < 0.3, f"rot {rot}: {e}"


def test_published_mim_variant_is_kept_and_shows_its_quantisation():
    src, ref, H = pair(rot_deg=75.0)
    assert rmse(rift.match(src, ref, params=rift.RiftParams(orientation="mim")), H) is None


@pytest.mark.parametrize("kw", [dict(sun_src=(315, 45)), dict(cross_modal=True),
                                dict(cross_modal=True, rot_deg=30.0)],
                         ids=["opposite_sun", "cross_modal", "cross_modal_rot30"])
def test_rift_registers_where_sift_cannot(kw):
    src, ref, H = pair(**kw)
    e = rmse(rift.match(src, ref), H)
    assert e is not None and e < 0.3
    assert rmse(classical.match(src, ref, detector="sift"), H) is None


def test_output_is_an_ordinary_matchset():
    """Same contract as every other matcher, so it flows through MAGSAC and the gates."""
    src, ref, _ = pair(rot_deg=10.0)
    ms = rift.match(src, ref)
    assert ms.method == "rift2" and ms.src_pts.shape == ms.ref_pts.shape
    assert len(ms.confidence) == len(ms.src_pts) and ms.device == "cpu"


def test_patch_must_divide_into_the_grid():
    with pytest.raises(ValueError):
        rift.features(np.random.rand(256, 256), rift.RiftParams(patch=100, grid=6))


def test_published_variant_uses_the_measured_index_shift():
    """Behavioural, not a restatement of the constant: with the measured sign the
    published MIM variant registers a 150 deg rotation (0.088 px); with the sign
    flipped it cannot. (A test that only asserted _INDEX_SHIFT_SIGN == -1 would
    pass whatever the sign did.)"""
    src, ref, H = pair(rot_deg=150.0)
    e = rmse(rift.match(src, ref, params=rift.RiftParams(orientation="mim")), H)
    assert e is not None and e < 0.3
