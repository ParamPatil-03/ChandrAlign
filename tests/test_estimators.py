"""ALIGN-01: what the robust estimator must do, and what it provably cannot.

PLAN.md's accept bar for ALIGN-01 is "MAGSAC beats plain RANSAC on a synthetic
set with 70% outliers; a deliberately wrong-scale match set is rejected". The
second half is covered in test_part2_foundation.py. The first half was an
assumption until scripts/bench_estimators.py measured it, and the measurement
did not agree with it: at 70% scattered outliers every driver succeeds on every
seed, so that bar does not separate them at all.

These tests therefore pin what the benchmark actually established:
  1. the accuracy requirement itself (70% outliers -> the true transform)
  2. the real reason MAGSAC is the default: cost at extreme outlier rates
  3. the limit no robust driver crosses, so that a later reader does not go
     looking for a better RANSAC to fix it
  4. that estimate.max_iters is a cap and not a per-fit cost

Thresholds here are deliberately loose against the measured values -- they are
regression guards, not the measurement. The numbers live in
reports/estimator_benchmark.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench_estimators import CENTRE, grid_rmse, make_correspondences  # noqa: E402

from chandralign.estimate import robust  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
BAD_PX = 2.0


def _fit(outlier_fraction, kind, seed, method="magsac", **kw):
    src, ref, H, is_inlier = make_correspondences(outlier_fraction, kind, seed)
    res = robust.estimate(src, ref, kind="homography", method=method, centre=CENTRE, **kw)
    err = None if res.model is None else grid_rmse(res.model, H)
    return res, err, is_inlier


# ---------------------------------------------------------------------------
# 1. The accuracy requirement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_recovers_the_true_transform_at_70_percent_outliers(seed):
    """The ALIGN-01 bar. Measured median error here is ~0.21 px; 2 px is the
    pipeline's own accuracy bar, so that is what this asserts."""
    res, err, _ = _fit(0.7, "uniform", seed)
    assert res.ok, "a 30% inlier set with a real consensus must not be refused"
    assert err < BAD_PX, f"recovered transform is {err:.2f} px from truth"


@pytest.mark.parametrize("seed", SEEDS)
def test_inliers_it_reports_are_really_inliers(seed):
    """An estimate is only as trustworthy as its inlier set. Getting the right
    transform while calling half the outliers inliers would corrupt every later
    stage -- sub-pixel refinement and the coverage score both read this mask."""
    res, _, is_inlier = _fit(0.7, "uniform", seed)
    found = res.inlier_mask.astype(bool)
    precision = np.count_nonzero(found & is_inlier) / max(np.count_nonzero(found), 1)
    assert precision > 0.9, f"only {precision:.0%} of the reported inliers are real"


# ---------------------------------------------------------------------------
# 2. Why MAGSAC is the default: cost, not accuracy
# ---------------------------------------------------------------------------
def test_magsac_reaches_the_answer_far_cheaper_than_plain_ransac():
    """The measured justification for estimate.method: magsac.

    At 90% outliers both find the true transform, but plain RANSAC pays the
    whole iteration cap (3.27 s per fit measured) while MAGSAC terminates
    adaptively (0.021 s) -- about 150x. This asserts 5x, so the guard survives
    a loaded machine and still fails if MAGSAC ever loses the property that
    makes a large estimate.max_iters affordable.

    Both are timed in the same process, back to back, so a slow machine slows
    both and the RATIO is what is being measured.
    """
    def timed(method):
        t0 = time.perf_counter()
        results = [_fit(0.9, "uniform", s, method=method) for s in SEEDS]
        return time.perf_counter() - t0, results

    t_magsac, magsac = timed("magsac")
    t_ransac, ransac = timed("ransac")

    assert all(r.ok and e < BAD_PX for r, e, _ in magsac), "MAGSAC must still be correct"
    assert all(r.ok and e < BAD_PX for r, e, _ in ransac), "the comparison is only fair if both succeed"
    assert t_magsac * 5 < t_ransac, (
        f"MAGSAC took {t_magsac:.2f}s against plain RANSAC's {t_ransac:.2f}s; the "
        f"measured gap is ~150x, so anything under 5x means the adaptive "
        f"termination that justifies estimate.max_iters=100000 is gone")


# ---------------------------------------------------------------------------
# 3. The limit: outliers that agree with each other
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["magsac", "ransac", "usac_accurate"])
def test_a_false_consensus_majority_is_returned_and_accepted(method):
    """DOCUMENTED LIMITATION, asserted so it cannot be quietly forgotten.

    When false matches agree with each other -- repetitive crater fields,
    failure mode #12 -- they are a consensus too, and past 50% they are the
    LARGER one. Every driver then returns the outliers' transform and reports a
    clean fit. This is not a bug to be fixed by choosing a better RANSAC: a
    consensus method cannot prefer the minority by construction.

    If this test ever fails, the estimator has gained a genuinely new property
    and robust.py's docstring, the control gates and MATCH-06's cross-check all
    need revisiting -- do not simply delete the assertion.
    """
    res, err, _ = _fit(0.8, "structured", 0, method=method)
    assert res.ok, "the wrong answer looks clean -- that is precisely the danger"
    assert err > BAD_PX, "it is the decoy transform, not the true one"


def test_the_scale_check_does_not_rescue_a_scale_plausible_decoy():
    """And the scale check is not the answer here either.

    The benchmark's decoy is 1.02x against the truth's 1.08x, so it is only
    5.5% off -- well inside estimate.scale_tolerance (25%). The check is built
    to catch a transform that is physically impossible, and this one is not:
    it is merely wrong. Recording that here so nobody cites the scale check as
    the defence against failure mode #12.
    """
    src, ref, H, _ = make_correspondences(0.8, "structured", 0)
    res = robust.estimate(src, ref, kind="homography", centre=CENTRE, expected_scale=1.08)
    assert res.ok and res.scale_status == "consistent"
    assert grid_rmse(res.model, H) > BAD_PX


# ---------------------------------------------------------------------------
# 4. estimate.max_iters is a cap, not a cost
# ---------------------------------------------------------------------------
def test_raising_max_iters_does_not_slow_down_an_ordinary_match_set():
    """Why 100000 was safe to ship (configs/default.yaml).

    MAGSAC stops when it is confident, so at 30% outliers a 10x larger cap is
    never reached and costs nothing. Measured identical at 0.0011 s under both.
    """
    def timed(iters):
        t0 = time.perf_counter()
        for s in SEEDS:
            _fit(0.3, "uniform", s, max_iters=iters)
        return time.perf_counter() - t0

    timed(10_000)                                   # warm up, so neither run pays import costs
    small, large = timed(10_000), timed(100_000)
    assert large < small * 3 + 0.05, (
        f"a 10x larger cap cost {large / max(small, 1e-9):.1f}x the time; if the "
        f"cap is being spent rather than capped, estimate.max_iters=100000 is "
        f"no longer free")


def test_max_iters_argument_overrides_the_config():
    """The budget has to be controllable per call for the benchmark's iteration
    axis to mean anything -- starving it must actually starve it."""
    starved = [_fit(0.9, "uniform", s, max_iters=50)[0] for s in SEEDS]
    healthy = [_fit(0.9, "uniform", s, max_iters=100_000)[0] for s in SEEDS]
    assert sum(r.ok for r in starved) < sum(r.ok for r in healthy)
