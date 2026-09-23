"""Robust transform estimation with a physical scale check (ALIGN-01 / P2-T06).

MAGSAC++ comes free from OpenCV (USAC_MAGSAC), so the code here is not a
re-implementation of RANSAC. What we actually add is the two things no library
can do for us:

1. MODEL SELECTION by residual structure -- prefer the simplest model that
   explains the data, instead of always fitting a homography because it has the
   most freedom.

2. THE SCALE SANITY CHECK (failure mode #13). A matcher can return a confident,
   geometrically self-consistent transform that is physically impossible,
   because it knows nothing about the instruments. If the instrument registry
   says OHRC against LRO NAC must be near 0.5 reference pixels per source pixel
   and the estimate says 3.0, the estimate is wrong no matter how clean its
   inlier set looks. Nothing off the shelf can make that judgement: it needs
   configs/instruments.yaml.

WHY MAGSAC IS THE DEFAULT, AND WHAT THAT IS AND IS NOT WORTH
Measured, not assumed -- scripts/bench_estimators.py, 900 synthetic
correspondence sets, reports/estimator_benchmark.json:

  - ACCURACY IS A TIE, and PLAN.md's accept bar ("MAGSAC beats plain RANSAC at
    70% outliers") is NOT met as written. Against outliers that agree with
    nothing, all three drivers succeed on every seed out to 90% and land inside
    0.4 px; at 70% plain RANSAC's median error is marginally the better of the
    two (0.196 px against 0.213 px). 70% scattered outliers is simply not a
    hard problem, so that bar could not have separated them.
  - WHAT DOES SEPARATE THEM IS COST. At 90% outliers both reach 10/10, MAGSAC
    in 0.021 s against plain RANSAC's 3.27 s -- 150x less for the same answer,
    and there the better median too (0.387 px against 0.432 px). MAGSAC
    terminates adaptively, so it converts a raised iteration cap into
    reliability at almost no runtime; plain RANSAC pays the cap in full.
  - Threshold-insensitivity holds but is SMALL once the budget is adequate:
    across 1, 3 and 10 px MAGSAC is flat at 1.00, plain RANSAC dips to 0.90 at
    1 px and 90% outliers. Worth having, because estimate.reproj_threshold_px
    is one constant applied across instrument pairs whose true noise differs,
    but it is not on its own the reason for the default. (Measured before the
    max_iters fix that gap was 30 points, which would have overstated it -- an
    estimator benchmark run at a starved budget measures the budget.)
  - It buys NOTHING against outliers that agree with each other. See below.

So the default is MAGSAC for cost and predictability at high outlier rates,
not because it is more accurate. On easy match sets any of the three would do.

THE LIMIT, AND WHY IT IS DOCUMENTED HERE RATHER THAN FIXED HERE
When false matches are mutually consistent -- repetitive crater fields and mare
ridges produce exactly this, failure mode #12 -- they form their own consensus.
Past 50% every driver returns the outliers' transform and reports it as a clean
fit: 110 to 114 confident wrong answers in 300 runs each, with no meaningful
difference between them. That is not a defect in any of the three. A consensus
method is definitionally unable to prefer the minority, so no choice of robust
driver, threshold or iteration budget addresses it. The defences that do are
elsewhere: the scale check below, the control gates (CHECK-01..08) and the
independent cross-check (MATCH-06). Picking a better RANSAC is not one of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import config
from ..contracts import TransformModel
from . import models
from .scale import ExpectedScale, ScaleVerdict
from .scale import check as _check_expected

# Failure-mode IDs from PLAN.md section 15.
FM_FALSE_CORRESPONDENCE = 12
FM_SCALE_CONFUSION = 13

_METHODS = {
    "magsac": cv2.USAC_MAGSAC,
    "usac_accurate": cv2.USAC_ACCURATE,
    "ransac": cv2.RANSAC,
}


@dataclass
class EstimateResult:
    """Outcome of one estimation attempt. Internal to Part 2."""

    model: TransformModel | None
    inlier_mask: np.ndarray
    ok: bool
    notes: list[str] = field(default_factory=list)
    failure_modes: list[int] = field(default_factory=list)
    # consistent | unverified | inconsistent | abstained | skipped | degenerate.
    # `ok` alone cannot say whether the scale was CONFIRMED or merely not refuted.
    scale_status: str | None = None

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


def _method_flag(name: str) -> int:
    key = str(name or "magsac").lower()
    if key not in _METHODS:
        raise ValueError(f"unknown estimator {name!r}; choose from {sorted(_METHODS)}")
    return _METHODS[key]


def _fit_homography(src, ref, flag, thresh, iters, conf):
    matrix, mask = cv2.findHomography(src, ref, method=flag,
                                      ransacReprojThreshold=float(thresh),
                                      maxIters=int(iters), confidence=float(conf))
    if matrix is None:
        return None, np.zeros(len(src), bool)
    return np.asarray(matrix, np.float64), mask.ravel().astype(bool)


def _fit_affine(src, ref, flag, thresh, iters, conf):
    matrix, mask = cv2.estimateAffine2D(src, ref, method=flag,
                                        ransacReprojThreshold=float(thresh),
                                        maxIters=int(iters), confidence=float(conf))
    if matrix is None:
        return None, np.zeros(len(src), bool)
    full = np.vstack([np.asarray(matrix, np.float64), [0.0, 0.0, 1.0]])
    return full, mask.ravel().astype(bool)


def check_scale(model: TransformModel, expected_scale: float | ExpectedScale | None,
                centre: tuple[float, float] = (0.0, 0.0),
                tolerance: float | None = None) -> ScaleVerdict:
    """Compare the recovered scale against what the instruments imply.

    `expected_scale` may be:
      ExpectedScale  per-axis and provenance-aware (estimate/scale.py). Use this
                     for real products: it knows pixels need not be square and
                     that a label's GSD is not automatically a measurement.
      float          an isotropic ratio the CALLER vouches for -- e.g. a
                     synthetic pair whose true scale is known exactly. Taken on
                     trust, so a float from a label reintroduces the NAC problem.
      None           no information: the check abstains rather than invent one.

    Returns a ScaleVerdict, which unpacks as (ok, message).
    """
    if tolerance is None:
        tolerance = float(config.get("estimate.scale_tolerance", 0.25))
    if model.matrix is None:
        return ScaleVerdict(True, "skipped", "scale check skipped: non-matrix model")
    est = models.estimated_scale(model.matrix, at=centre)
    model.scale_estimated = None if np.isnan(est) else float(est)
    if expected_scale is None:
        return ScaleVerdict(True, "abstained", "scale check abstained: no expected scale available")

    if isinstance(expected_scale, ExpectedScale):
        model.scale_expected = float(expected_scale.area)
        return _check_expected(model.matrix, expected_scale, centre=centre, tolerance=tolerance)

    model.scale_expected = float(expected_scale)
    if np.isnan(est) or expected_scale == 0:
        return ScaleVerdict(False, "degenerate", "scale check failed: degenerate transform")
    rel = abs(est / float(expected_scale) - 1.0)
    if rel > tolerance:
        return ScaleVerdict(False, "inconsistent",
                            f"scale {est:.4g} disagrees with the {expected_scale:.4g} "
                            f"implied by the instrument GSDs "
                            f"({rel * 100:.0f}% off, tolerance {tolerance * 100:.0f}%)", est)
    return ScaleVerdict(True, "consistent",
                        f"scale {est:.4g} consistent with {expected_scale:.4g} "
                        f"({rel * 100:.1f}% off)", est)


def estimate(src_pts: np.ndarray, ref_pts: np.ndarray, *, kind: str = "auto",
             expected_scale: float | ExpectedScale | None = None,
             centre: tuple[float, float] | None = None,
             method: str | None = None,
             reproj_threshold: float | None = None,
             min_inliers: int | None = None,
             max_iters: int | None = None) -> EstimateResult:
    """Fit a source -> reference transform robustly, then sanity-check it."""
    src = np.asarray(src_pts, np.float64).reshape(-1, 2)
    ref = np.asarray(ref_pts, np.float64).reshape(-1, 2)
    notes: list[str] = []
    fms: list[int] = []

    if len(src) != len(ref):
        raise ValueError(f"point count mismatch: {len(src)} vs {len(ref)}")

    flag = _method_flag(method or config.get("estimate.method", "magsac"))
    thresh = float(reproj_threshold if reproj_threshold is not None
                   else config.get("estimate.reproj_threshold_px", 3.0))
    iters = int(max_iters if max_iters is not None
                else config.get("estimate.max_iters", 100000))
    conf = float(config.get("estimate.confidence", 0.9999))
    need = int(min_inliers if min_inliers is not None
               else config.get("estimate.min_inliers", 12))

    if len(src) < 4:
        return EstimateResult(None, np.zeros(len(src), bool), False,
                              [f"only {len(src)} correspondences; need at least 4"],
                              [FM_FALSE_CORRESPONDENCE])

    if centre is None:
        centre = tuple(np.mean(src, axis=0)) if len(src) else (0.0, 0.0)

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    if kind in ("auto", "affine"):
        m, mask = _fit_affine(src, ref, flag, thresh, iters, conf)
        if m is not None:
            candidates.append(("affine", m, mask))
    if kind in ("auto", "homography"):
        m, mask = _fit_homography(src, ref, flag, thresh, iters, conf)
        if m is not None:
            candidates.append(("homography", m, mask))
    if not candidates:
        return EstimateResult(None, np.zeros(len(src), bool), False,
                              ["no model could be fitted"], [FM_FALSE_CORRESPONDENCE])

    def score(item):
        name, matrix, mask = item
        if not mask.any():
            return (0, np.inf)
        model = TransformModel(kind=name, matrix=matrix)
        rms = float(np.sqrt((models.residuals(model, src[mask], ref[mask]) ** 2).mean()))
        return (int(mask.sum()), rms)

    scored = [(name, matrix, mask, *score((name, matrix, mask))) for name, matrix, mask in candidates]

    # Prefer the simpler model unless the richer one is clearly better. A
    # homography has 8 degrees of freedom against affine's 6, and those two
    # extra parameters will always fit the data at least as well -- which is
    # why "fits better" alone is not a reason to accept it.
    chosen = scored[0]
    if len(scored) > 1:
        aff = next((s for s in scored if s[0] == "affine"), None)
        hom = next((s for s in scored if s[0] == "homography"), None)
        if aff is None:
            chosen = hom
        elif hom is None:
            chosen = aff
        else:
            better_inliers = hom[3] > aff[3] * 1.10
            better_rms = hom[4] < aff[4] * 0.80
            chosen = hom if (better_inliers or better_rms) else aff
            notes.append(
                f"model selection: affine({aff[3]} inliers, {aff[4]:.3f}px) vs "
                f"homography({hom[3]}, {hom[4]:.3f}px) -> {chosen[0]}")

    name, matrix, mask = chosen[0], chosen[1], chosen[2]
    model = TransformModel(kind=name, matrix=matrix)

    verdict = check_scale(model, expected_scale, centre=centre)
    notes.append(verdict.message)
    if not verdict.ok:
        fms.append(FM_SCALE_CONFUSION)
        return EstimateResult(model, np.zeros(len(src), bool), False, notes, fms,
                              scale_status=verdict.status)

    n_in = int(mask.sum())
    if n_in < need:
        notes.append(f"only {n_in} inliers, below the {need} required: "
                     f"a RANSAC 'success' this thin is not trustworthy")
        fms.append(FM_FALSE_CORRESPONDENCE)
        return EstimateResult(model, mask, False, notes, fms, scale_status=verdict.status)

    struct = models.residual_structure(src[mask], models.residuals(model, src[mask], ref[mask]))
    if struct > 0.35:
        notes.append(f"residuals are spatially structured ({struct:.2f}); local "
                     f"relief may need TPS rather than a global {name}")

    return EstimateResult(model, mask, True, notes, fms, scale_status=verdict.status)
