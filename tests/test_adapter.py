"""Acceptance tests for the vismatch adapter (P2-T01, MATCH-03/04/05).

The learned path is an optional extra, so these skip cleanly when vismatch is
absent. That is deliberate: the classical path must remain independently
testable on a machine with no torch and no network (risk R6).
"""
from __future__ import annotations

import numpy as np
import pytest

from chandralign import synth
from chandralign.contracts import MatchSet
from chandralign.matching import adapter, licence

vismatch = pytest.importorskip("vismatch", reason="learned path is an optional extra")

SHAPE = (256, 256)

# Kept small and cheap; all are licence-clean and CPU-capable.
THREE_MATCHERS = ["xfeat", "aliked-lightglue", "sift-lightglue"]


@pytest.fixture(scope="module")
def pair():
    return synth.make_pair(out_shape=SHAPE, n_craters=30, seed=3, rot_deg=8.0)


def test_image_conversion_matches_the_documented_format(pair):
    """vismatch documents (3, H, W) in [0, 1]; anything else silently misbehaves."""
    src, _, _ = pair
    img = adapter.to_vismatch_image(src)
    assert img.shape == (3, SHAPE[0], SHAPE[1])
    assert img.dtype == np.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


@pytest.mark.parametrize("model_name", THREE_MATCHERS)
def test_three_matchers_run_through_one_call_path(pair, model_name):
    """P2-T01 acceptance: three different matchers, identical adapter call."""
    src, ref, _ = pair
    try:
        ms = adapter.match(src, ref, model_name=model_name)
    except adapter.MatcherUnavailableError as exc:
        pytest.skip(f"{model_name} unavailable (likely offline): {exc}")
    assert isinstance(ms, MatchSet)
    assert ms.src_pts.shape[1] == 2 and ms.ref_pts.shape[1] == 2
    assert len(ms.src_pts) == len(ms.ref_pts) == len(ms.confidence)
    assert ms.method == model_name


def test_adapter_enforces_the_licence_gate(pair):
    """A restricted model must be refused before it is ever constructed."""
    src, ref, _ = pair
    with pytest.raises(licence.LicenceRestrictedError):
        adapter.match(src, ref, model_name="superpoint-lightglue", ship_mode=True)


def test_library_ransac_is_disabled(pair):
    """Our estimator owns the geometry, so the library's RANSAC must be off.

    If vismatch computed the homography, we would be shipping a transform that
    never passed our scale sanity check against the instrument registry.
    """
    matcher = adapter._load("xfeat", "cpu", 512)
    assert matcher.skip_ransac is True


def test_adapter_records_the_device_it_actually_used(pair):
    """Device is recorded from the run, never inferred from the machine."""
    src, ref, _ = pair
    try:
        ms = adapter.match(src, ref, model_name="xfeat", device="cpu")
    except adapter.MatcherUnavailableError as exc:
        pytest.skip(str(exc))
    assert ms.device == "cpu"
    assert ms.provenance["match_device"] == "cpu"
    assert "capability" in ms.provenance


def test_shadowed_endpoints_are_dropped(pair):
    """A cast shadow moves with the Sun, so matches on one are not surface matches."""
    src, ref, _ = pair
    forced_src = synth.ImagePlane(
        array=src.array, valid_mask=src.valid_mask,
        shadow_mask=np.ones_like(src.shadow_mask, bool),   # everything shadowed
        gsd_m=src.gsd_m, meta=src.meta, geo=src.geo)
    try:
        kept = adapter.match(src, ref, model_name="xfeat", drop_shadowed=True)
        dropped = adapter.match(forced_src, ref, model_name="xfeat", drop_shadowed=True)
    except adapter.MatcherUnavailableError as exc:
        pytest.skip(str(exc))
    assert len(dropped.src_pts) == 0
    assert dropped.n_dropped_shadowed > 0
    assert len(kept.src_pts) > 0
