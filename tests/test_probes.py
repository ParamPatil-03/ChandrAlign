"""Audit I-08: matcher-free probes measure a geometry's error in its own frame (evaluate/probes.py)."""
import cv2
import numpy as np

from chandralign.evaluate import probes as P

TH, K = np.deg2rad(8.0), 0.9
M = np.array([[K * np.cos(TH), -K * np.sin(TH), 40.0], [K * np.sin(TH), K * np.cos(TH), -20.0]])


def _pair():
    tex = cv2.GaussianBlur(np.random.default_rng(0).random((700, 700)).astype(np.float32), (0, 0), 2)
    src = tex[50:562, 50:562].copy()
    return src, cv2.warpAffine(src, M, (512, 512), flags=cv2.INTER_CUBIC)


def true(p):
    return p @ M[:, :2].T + M[:, 2]


def test_a_perfect_geometry_measures_near_zero_on_a_rotated_scaled_pair():
    src, ref = _pair()
    g = P.geometry_error_warped(src, ref, np.ones_like(src, bool), true, ref_px_per_src_px=K)
    assert g["n"] >= 50 and g["p50_px_src"] < 0.05 and g["p95_px_src"] < 0.1


def test_a_known_geometry_error_is_measured_in_source_px():
    src, ref = _pair()
    g = P.geometry_error_warped(src, ref, np.ones_like(src, bool), lambda p: true(p) + [0.6, 0.0],
                                ref_px_per_src_px=K)
    assert abs(g["p50_px_src"] - 0.6 / K) < 0.05


def test_the_axis_aligned_probe_was_biased_across_a_rotation():
    """Why the fix exists: the same perfect geometry read 0.6 px through axis-aligned templates."""
    src, ref = _pair()
    old = P.geometry_error(src, ref, np.ones_like(src, bool), true, K)
    new = P.geometry_error_warped(src, ref, np.ones_like(src, bool), true, ref_px_per_src_px=K)
    assert old["p50_px_src"] > 10 * new["p50_px_src"]


def test_a_rotated_pair_through_register_bundle_gets_a_measured_accuracy():
    from chandralign import synth
    from chandralign.pipeline import register_bundle
    src, ref, _ = synth.make_pair(out_shape=(512, 512), seed=3, rot_deg=8.0)
    b = register_bundle(src, ref, matcher="sift")
    pr = b.result.provenance["accuracy"]["probes"]
    assert pr["n"] >= 20, pr                          # was "unmeasured" on rotated pairs before the fix
    assert pr["p50_px_src"] < 0.5
