"""Sub-pixel shift estimation (PREC-01..04), on a REAL OHRC crop with EXACT shifts.

HOW THE KNOWN SHIFT IS MADE, AND WHY IT MATTERS
Shifting an image by interpolation bakes an interpolation model into the test,
and a method that shares that model looks better than it is; a Fourier shift
would flatter phase correlation, the method under test. Instead both images are
block-averaged 4x4 from the same real OHRC crop, the second starting a whole
number of NATIVE pixels later. That is an exact sub-pixel shift (in steps of
0.25 coarse px) formed the way a detector forms pixels, with no interpolation
assumed anywhere. scripts/verify_subpixel.py does the same at 100x on full
OHRC to reach (0.37, -0.62) exactly for PREC-06.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chandralign.refine import subpixel as sp

IMG = np.load(Path(__file__).resolve().parents[1] / "tests/fixtures/images/ohrc_crater_field.npy").astype(np.float64)
K, N, BASE = 4, 56, 12

SUBPIXEL_TRUTHS = [(0.25, -0.5), (0.75, -0.25), (-0.5, 0.75), (1.25, 0.5), (-1.75, -1.25)]


def pair(dx: float, dy: float):
    """(ref, mov) with a feature at ref p appearing at p + (dx, dy) in mov."""
    sx, sy = int(round(-dx * K)), int(round(-dy * K))
    assert abs(sx + dx * K) < 1e-9 and abs(sy + dy * K) < 1e-9, "shift must be a multiple of 1/K"
    blk = lambda ox, oy: IMG[oy:oy + K * N, ox:ox + K * N].reshape(N, K, N, K).mean(axis=(1, 3))
    return blk(BASE, BASE), blk(BASE + sx, BASE + sy)


def err(method: str, truth) -> float:
    e = sp.estimate(*pair(*truth), method)
    assert e.ok, f"{method} failed on {truth}: {e.notes}"
    return float(np.hypot(e.dx - truth[0], e.dy - truth[1]))


# ---------------------------------------------------------------------------
# The sign convention -- the classic silent bug with four different libraries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", sorted(sp.METHODS))
def test_every_method_uses_the_same_sign_convention(method):
    """skimage reports the shift that registers mov ONTO ref, ECC the warp from
    ref to mov, NCC a template offset. All must come back as the same d."""
    e = sp.estimate(*pair(2.0, -1.0), method)
    assert e.ok, e.notes
    assert e.dx == pytest.approx(2.0, abs=0.05) and e.dy == pytest.approx(-1.0, abs=0.05)


# ---------------------------------------------------------------------------
# PREC-01: NCC peak fit
# ---------------------------------------------------------------------------
def test_ncc_recovers_a_known_subpixel_shift_to_under_a_tenth_of_a_pixel():
    """PREC-01 done_when. Holds for the Gaussian peak fit on every shift tried."""
    worst = max(err("ncc_gaussian", t) for t in SUBPIXEL_TRUTHS)
    assert worst < 0.1, f"worst NCC-Gaussian error {worst:.3f} px"


def test_the_textbook_parabola_fit_shows_pixel_locking():
    """A parabola on a sharp NCC peak is pulled towards whole pixels. Measured
    here, not assumed: its mean error is about twice the Gaussian fit's. Pinned
    so nobody 'simplifies' the default back to the parabola without seeing it."""
    para = np.mean([err("ncc_parabola", t) for t in SUBPIXEL_TRUTHS])
    gauss = np.mean([err("ncc_gaussian", t) for t in SUBPIXEL_TRUTHS])
    assert para < 0.1
    assert gauss < para


# ---------------------------------------------------------------------------
# PREC-02: phase correlation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("truth", SUBPIXEL_TRUTHS)
def test_phase_correlation_leaves_less_shift_than_it_started_with(truth):
    """PREC-02 done_when: the leftover shift after correction is smaller than before."""
    assert err("phase", truth) < np.hypot(*truth)


def test_phase_correlation_is_subpixel_on_this_fixture():
    assert np.mean([err("phase", t) for t in SUBPIXEL_TRUTHS]) < 0.15


# ---------------------------------------------------------------------------
# PREC-03 / PREC-04
# ---------------------------------------------------------------------------
def test_corner_refinement_returns_real_subpixel_values():
    """Regression: cornerSubPix silently leaves most points unrefined, which once
    made every answer a whole number. Only corners refined in both patches count."""
    e = sp.estimate(*pair(0.75, -0.25), "corner")
    assert e.ok
    assert abs(e.dx - round(e.dx)) > 1e-3 or abs(e.dy - round(e.dy)) > 1e-3, "integer answer again"
    assert np.mean([err("corner", t) for t in SUBPIXEL_TRUTHS]) < 0.15


def test_ecc_is_the_most_accurate_method_on_this_fixture():
    """Measured, and recorded because it is not the usual textbook ranking."""
    means = {m: np.mean([err(m, t) for t in SUBPIXEL_TRUTHS])
             for m in ("ncc_parabola", "ncc_gaussian", "phase", "corner", "ecc")}
    assert means["ecc"] < 0.05
    assert min(means, key=means.get) == "ecc", means


# ---------------------------------------------------------------------------
# Failure is a result, not a crash or a made-up number (H1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["ncc_parabola", "phase"])
def test_a_flat_patch_is_reported_as_a_failure(method):
    flat = np.full((56, 56), 30.0)
    e = sp.estimate(flat, flat, method)
    assert not e.ok and np.isnan(e.dx)


def test_ecc_on_unrelated_noise_fails_gracefully_or_stays_finite():
    rng = np.random.default_rng(0)
    e = sp.estimate(rng.random((56, 56)), rng.random((56, 56)), "ecc")
    assert (not e.ok) or np.isfinite(e.dx)


def test_unknown_method_is_an_error():
    with pytest.raises(ValueError):
        sp.estimate(*pair(0.25, -0.5), "magic")


# ---------------------------------------------------------------------------
# Per-match refinement: what the pipeline actually calls
# ---------------------------------------------------------------------------
def test_refine_points_pulls_rounded_matches_to_their_true_position():
    ref, mov = pair(0.75, -0.25)
    rng = np.random.default_rng(1)
    src = rng.uniform(16, N - 16, size=(12, 2))
    truth = src + np.array([0.75, -0.25])
    start = np.round(truth)                              # a matcher's integer answer
    refined, moved = sp.refine_points(ref, mov, src, start, method="ecc", half=12)
    before = np.hypot(*(start - truth).T).mean()
    after = np.hypot(*(refined[moved] - truth[moved]).T).mean()
    assert moved.sum() >= 8
    assert after < before / 2, f"refinement {before:.3f} -> {after:.3f} px"


def test_refine_points_will_not_relocate_a_match():
    """A sub-pixel step corrects rounding; it must not move a match 3 px."""
    ref, mov = pair(0.0, 0.0)
    src = np.array([[28.0, 28.0]])
    refined, moved = sp.refine_points(ref, mov, src, src + 3.0, method="ecc", half=12, max_move=1.5)
    assert not moved[0] and np.allclose(refined, src + 3.0)


# ---------------------------------------------------------------------------
# Iterative refinement (the variants PREC-06 was met with)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base, it", [("ncc_gaussian", "ncc_gaussian_iter"), ("phase", "phase_iter")])
def test_iterating_removes_most_of_the_fit_bias(base, it):
    """Dev-fixture numbers, fixed before the OHRC run: NCC 0.041 -> 0.012 px
    mean, phase 0.097 -> 0.048. Pinned so iteration cannot silently stop helping."""
    before = np.mean([err(base, t) for t in SUBPIXEL_TRUTHS])
    after = np.mean([err(it, t) for t in SUBPIXEL_TRUTHS])
    assert after < 0.7 * before, f"{it}: {before:.3f} -> {after:.3f}"


def test_iterated_ncc_meets_a_twentieth_of_a_pixel_on_the_fixture():
    assert max(err("ncc_gaussian_iter", t) for t in SUBPIXEL_TRUTHS) < 0.05


def test_iteration_keeps_the_sign_convention():
    for m in ("ncc_gaussian_iter", "phase_iter"):
        e = sp.estimate(*pair(2.0, -1.0), m)
        assert e.dx == pytest.approx(2.0, abs=0.05) and e.dy == pytest.approx(-1.0, abs=0.05)


def test_the_default_method_is_one_that_met_prec06():
    """configs/default.yaml's subpixel.method must name a real, measured method: ncc_gaussian_iter
    and ecc met PREC-06 on translations; lsm met audit C-02 on a known non-rigid warp
    (docs/refinement_geometry_protocol.md), which a pure translation cannot test."""
    from chandralign import config
    assert config.get("subpixel.method") in ("ncc_gaussian_iter", "ecc", "lsm")
    for m in config.get("subpixel.fallbacks", []):
        assert m in sp.METHODS
    assert config.get("subpixel.method") in sp.METHODS
