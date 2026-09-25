"""CHECK-07: an impossible pair comes back REJECTED with failure_modes, end to end, and nothing crashes."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from chandralign import synth
from chandralign.pipeline import register

ROOT = Path(__file__).resolve().parents[1]


def _plane(a):
    a = np.asarray(a, np.float32)
    return synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool), shadow_mask=np.zeros(a.shape, bool),
                            gsd_m=5.0, meta=None, geo=None)


def _terrain(seed, n=320):
    rng = np.random.default_rng(seed)
    return cv2.normalize(cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 2.0), None, 0, 1, cv2.NORM_MINMAX)


def test_an_impossible_pair_is_rejected_with_a_named_failure_mode():
    r = register(_plane(_terrain(1)), _plane(_terrain(2)))          # two unrelated surfaces
    assert r.confidence_tier == "REJECTED"
    assert r.failure_modes, "a rejection must say why (CHECK-07)"
    assert r.gates, "and must carry its control-gate results (CHECK-08)"


def test_a_genuine_pair_is_not_rejected():
    a = _terrain(3)
    b = cv2.warpAffine(a, np.float32([[1, 0, 3.3], [0, 1, -2.1]]), a.shape[::-1], flags=cv2.INTER_CUBIC)
    r = register(_plane(a), _plane(b))
    assert r.confidence_tier in ("HIGH", "MEDIUM", "LOW"), (r.confidence_tier, r.failure_modes, r.notes)
    assert np.allclose(r.model.matrix[:2, 2], [3.3, -2.1], atol=0.3)


def test_the_process_exits_0_on_an_impossible_pair():
    code = ("import numpy as np, cv2\n"
            "from chandralign import synth\nfrom chandralign.pipeline import register\n"
            "def p(s):\n"
            "    a = cv2.normalize(cv2.GaussianBlur(np.random.default_rng(s).random((320, 320)).astype(np.float32), (0, 0), 2.0), None, 0, 1, cv2.NORM_MINMAX)\n"
            "    return synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool), shadow_mask=np.zeros(a.shape, bool), gsd_m=5.0, meta=None, geo=None)\n"
            "r = register(p(1), p(2))\nprint(r.confidence_tier, r.failure_modes)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().startswith("REJECTED")


def test_the_bundle_delivers_the_points_its_metrics_describe():
    """Part 3 handoff: export `delivered` (uniform, sub-pixel refined), not the pre-uniformity matches."""
    from chandralign.estimate import models
    from chandralign.pipeline import register_bundle
    from chandralign.refine import uniformity
    a = _terrain(4)
    b = cv2.warpAffine(a, np.float32([[1, 0, 2.6], [0, 1, 1.4]]), a.shape[::-1], flags=cv2.INTER_CUBIC)
    bun = register_bundle(_plane(a), _plane(b))
    r, d = bun.result, bun.delivered
    assert r.confidence_tier in ("HIGH", "MEDIUM", "LOW") and d.stage == "delivered"
    assert 0 < len(d.src_pts) <= int(r.inlier_mask.sum())              # thinned from the inliers
    assert np.isclose(uniformity.coverage_of(d.src_pts, a.shape, 8), r.metrics.spatial_coverage)
    # audit C-03: the affine's residual on the delivered points is only the FIT residual ...
    acc = r.provenance["accuracy"]
    res = np.hypot(*(models.apply(r.model, d.src_pts) - d.ref_pts).T)
    assert np.isclose(np.sqrt(np.mean(res ** 2)), acc["fit_residual_px"], atol=1e-6)
    # ... and rmse_px is the CHECK-POINT error of the geometry the product warps with, named
    assert r.metrics.rmse_px == acc["rmse_px_ref"] and r.metrics.rmse_px is not None
    assert acc["model"] == bun.geometry and acc["point_set"] and acc["n_check"] > 0
    assert r.metrics.rmse_m == pytest.approx(r.metrics.rmse_px * 5.0, rel=1e-3)     # gsd_m 5.0
    # audit I-10: the two metrics the report displays are filled
    assert r.metrics.max_delaunay_gap_px == bun.stages["uniformity"]["max_delaunay_gap_px"] is not None
    assert r.metrics.subpixel_recovery_err_px is not None and r.metrics.subpixel_recovery_err_px < 0.25
    assert bun.tps is not None and bun.src is not None
    assert r.metrics.runtime_s is not None and r.metrics.runtime_s > 0     # the whole call, timed
