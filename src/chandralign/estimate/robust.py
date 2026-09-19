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
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import config
from ..contracts import TransformModel
from . import models

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


def check_scale(model: TransformModel, expected_scale: float | None,
                centre: tuple[float, float] = (0.0, 0.0),
                tolerance: float | None = None) -> tuple[bool, str]:
    """Compare the recovered scale against what the instruments imply.

    Returns (ok, message). A None expected_scale means we have no instrument
    information, so the check abstains rather than inventing a verdict.
    """
    if tolerance is None:
        tolerance = float(config.get("estimate.scale_tolerance", 0.25))
    if model.matrix is None:
        return True, "scale check skipped: non-matrix model"
    est = models.estimated_scale(model.matrix, at=centre)
    model.scale_estimated = None if np.isnan(est) else float(est)
    if expected_scale is None:
        return True, "scale check abstained: no expected scale available"
    model.scale_expected = float(expected_scale)
    if np.isnan(est) or expected_scale == 0:
        return False, "scale check failed: degenerate transform"
    rel = abs(est / float(expected_scale) - 1.0)
    if rel > tolerance:
        return False, (f"scale {est:.4g} disagrees with the {expected_scale:.4g} "
                       f"implied by the instrument GSDs "
                       f"({rel * 100:.0f}% off, tolerance {tolerance * 100:.0f}%)")
    return True, f"scale {est:.4g} consistent with {expected_scale:.4g} ({rel * 100:.1f}% off)"


def estimate(src_pts: np.ndarray, ref_pts: np.ndarray, *, kind: str = "auto",
             expected_scale: float | None = None,
             centre: tuple[float, float] | None = None,
             method: str | None = None,
             reproj_threshold: float | None = None,
             min_inliers: int | None = None) -> EstimateResult:
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
    iters = int(config.get("estimate.max_iters", 10000))
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

    ok_scale, scale_msg = check_scale(model, expected_scale, centre=centre)
    notes.append(scale_msg)
    if not ok_scale:
        fms.append(FM_SCALE_CONFUSION)
        return EstimateResult(model, np.zeros(len(src), bool), False, notes, fms)

    n_in = int(mask.sum())
    if n_in < need:
        notes.append(f"only {n_in} inliers, below the {need} required: "
                     f"a RANSAC 'success' this thin is not trustworthy")
        fms.append(FM_FALSE_CORRESPONDENCE)
        return EstimateResult(model, mask, False, notes, fms)

    struct = models.residual_structure(src[mask], models.residuals(model, src[mask], ref[mask]))
    if struct > 0.35:
        notes.append(f"residuals are spatially structured ({struct:.2f}); local "
                     f"relief may need TPS rather than a global {name}")

    return EstimateResult(model, mask, True, notes, fms)
