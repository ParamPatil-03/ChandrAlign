"""Honest, reproducible registration metrics (OUT-04..08).

Only values computable from supplied inputs are populated. Missing inputs
produce ``None`` rather than a plausible-looking zero (honesty rule H1).
"""
from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from ..contracts import MatchSet, MetricSource, Metrics, TransformModel
from ..estimate.models import residuals
from ..refine.uniformity import coverage_of, max_delaunay_gap

_VALID_SOURCES = {"measured", "external", "synthetic"}


class RuntimeTimer:
    """Wall-clock timer for the complete registration operation.

    Put this around the whole pipeline, including preprocessing and tiling.
    Matcher-specific timings must remain separate and must not be passed as
    ``runtime_s`` to :func:`compute_metrics`.
    """

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock
        self._started_at: float | None = None
        self.runtime_s: float | None = None

    def __enter__(self) -> RuntimeTimer:
        return self.start()

    def start(self) -> RuntimeTimer:
        """Start (or restart) the timer and return it for convenient chaining."""
        self._started_at = self._clock()
        self.runtime_s = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds (idempotently)."""
        if self.runtime_s is not None:
            return self.runtime_s
        if self._started_at is None:
            raise RuntimeError("RuntimeTimer must be started with a context manager")
        self.runtime_s = max(0.0, float(self._clock() - self._started_at))
        return self.runtime_s

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def compute_metrics(
    matches: MatchSet,
    inlier_mask: np.ndarray | None = None,
    model: TransformModel | None = None,
    *,
    image_shape: tuple[int, int] | None = None,
    grid: int = 8,
    gsd_m: float | None = None,
    runtime_s: float | None = None,
    subpixel_recovery_err_px: float | None = None,
    keypoints_src: int | None = None,
    keypoints_ref: int | None = None,
    source: MetricSource = "measured",
) -> Metrics:
    """Compute metrics from candidate matches and their final inlier mask.

    ``image_shape`` is ``(height, width)`` in the source frame. Coverage and
    the Delaunay diagnostic use source-frame inlier coordinates because they
    describe how well the source image is spatially supported.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"unknown metric source: {source!r}")

    src_pts = np.asarray(matches.src_pts, dtype=np.float64)
    ref_pts = np.asarray(matches.ref_pts, dtype=np.float64)
    confidence = np.asarray(matches.confidence)
    if src_pts.ndim != 2 or src_pts.shape[1:] != (2,):
        raise ValueError("matches.src_pts must have shape (N, 2)")
    if ref_pts.shape != src_pts.shape:
        raise ValueError("matches.ref_pts must have the same (N, 2) shape as src_pts")
    if confidence.ndim != 1 or len(confidence) != len(src_pts):
        raise ValueError("matches.confidence must have shape (N,)")

    _validate_optional_nonnegative("gsd_m", gsd_m)
    _validate_optional_nonnegative("runtime_s", runtime_s)
    _validate_optional_nonnegative("subpixel_recovery_err_px", subpixel_recovery_err_px)
    _validate_optional_count("keypoints_src", keypoints_src)
    _validate_optional_count("keypoints_ref", keypoints_ref)

    result = Metrics(
        runtime_s=None if runtime_s is None else float(runtime_s),
        subpixel_recovery_err_px=(None if subpixel_recovery_err_px is None
                                  else float(subpixel_recovery_err_px)),
        keypoints_src=keypoints_src,
        keypoints_ref=keypoints_ref,
        source=source,
    )

    # Without a mask, none of the inlier-derived quantities was measured.
    if inlier_mask is None:
        return result

    mask = np.asarray(inlier_mask)
    if mask.ndim != 1 or len(mask) != len(src_pts):
        raise ValueError("inlier_mask must have shape (N,) matching the candidates")
    if not (np.issubdtype(mask.dtype, np.bool_) or np.all(np.isin(mask, (0, 1)))):
        raise ValueError("inlier_mask must contain only boolean or 0/1 values")
    mask = mask.astype(bool, copy=False)

    result.inlier_count = int(mask.sum())
    result.inlier_ratio = float(result.inlier_count / len(mask)) if len(mask) else 0.0

    inlier_src = src_pts[mask]
    inlier_ref = ref_pts[mask]
    if model is not None and result.inlier_count:
        errors_px = residuals(model, inlier_src, inlier_ref)
        result.rmse_px = float(np.sqrt(np.mean(np.square(errors_px))))
        if gsd_m is not None:
            result.rmse_m = float(result.rmse_px * gsd_m)

    if image_shape is not None:
        _validate_image_shape(image_shape)
        if (not isinstance(grid, (int, np.integer)) or isinstance(grid, (bool, np.bool_))
                or grid <= 0):
            raise ValueError("grid must be a positive integer")
        result.spatial_coverage = coverage_of(inlier_src, image_shape, int(grid))
        result.max_delaunay_gap_px = max_delaunay_gap(inlier_src, image_shape)

    return result


def _validate_optional_nonnegative(name: str, value: float | None) -> None:
    if value is None:
        return
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number")


def _validate_optional_count(name: str, value: int | None) -> None:
    if value is None:
        return
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or value < 0):
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_image_shape(shape: tuple[int, int]) -> None:
    if (not isinstance(shape, tuple) or len(shape) != 2 or
            any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
                or v <= 0 for v in shape)):
        raise ValueError("image_shape must be a (height, width) tuple of positive integers")


# Short form retained for callers that naturally import ``metrics.compute``.
compute = compute_metrics
