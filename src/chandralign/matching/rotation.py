"""Extreme-rotation handling (MATCH-12, failure mode #6): an image-level rotation search.

The LoFTR-family matchers we route to are not rotation-invariant (SE2-LoFTR, arXiv:2204.10144;
Steerers, arXiv:2312.02152); SIFT is, by its dominant-orientation step. Our registrations take
rotation from the label geometry, so this is for a pair without a usable prior. Rotating the
SOURCE image and matching again works for any matcher, needs no new weights or licences, and
costs nothing when the first match is already good (docs/rotation_protocol.md).
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import cv2
import numpy as np

from .. import config
from ..contracts import ImagePlane, MatchSet


def rotate_plane(plane: ImagePlane, angle_deg: float) -> tuple[ImagePlane, np.ndarray]:
    """The plane rotated by angle_deg about its centre on an expanded canvas (nothing cropped),
    and the 3x3 map from original px to rotated px."""
    h, w = np.asarray(plane.array).shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle_deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    # tolerance: cos(90 deg) is ~6e-17, and a bare ceil() would widen the canvas by one pixel,
    # putting every rotated pixel half a pixel off the grid
    W, H = int(np.ceil(w * c + h * s - 1e-6)), int(np.ceil(w * s + h * c - 1e-6))
    M[0, 2] += (W - w) / 2.0
    M[1, 2] += (H - h) / 2.0

    def warp(a, interp, fill):
        return cv2.warpAffine(np.asarray(a, np.float32), M, (W, H), flags=interp, borderValue=fill)

    rot = dataclasses.replace(
        plane, array=warp(plane.array, cv2.INTER_LINEAR, 0.0),
        valid_mask=warp(plane.valid_mask, cv2.INTER_NEAREST, 0.0) > 0.5,
        shadow_mask=warp(plane.shadow_mask, cv2.INTER_NEAREST, 0.0) > 0.5)
    return rot, np.vstack([M, [0.0, 0.0, 1.0]])


def rotation_search(src: ImagePlane, ref: ImagePlane, match_fn: Callable[[ImagePlane, ImagePlane], MatchSet],
                    *, step_deg: float | None = None, min_inliers: int | None = None) -> MatchSet:
    """match_fn(src, ref) at 0 deg; if the robust estimate is weak, at every `step_deg` rotation of
    the source, keeping the angle with the most inliers. Source points come back in the ORIGINAL
    source frame; `ms.rotation_search` records what happened."""
    from ..estimate import robust

    step = float(step_deg if step_deg is not None else config.get("matching.rotation_step_deg", 30))
    need = int(min_inliers if min_inliers is not None else config.get("matching.rotation_min_inliers", 40))

    def inliers(ms: MatchSet) -> int:
        if len(ms.src_pts) < 4:
            return 0
        try:
            return int(robust.estimate(ms.src_pts, ms.ref_pts).inlier_count)
        except Exception:                                   # a failed estimate is "no inliers"
            return 0

    ms0 = match_fn(src, ref)
    n0 = inliers(ms0)
    tried = [(0.0, n0)]
    if n0 >= need:
        ms0.rotation_search = {"applied": False, "angle_deg": 0.0, "inliers": n0}  # type: ignore[attr-defined]
        return ms0
    best = (n0, 0.0, ms0, np.eye(3))
    for k in range(1, int(round(360.0 / step))):
        a = k * step
        rp, M = rotate_plane(src, a)
        ms = match_fn(rp, ref)
        n = inliers(ms)
        tried.append((a, n))
        if n > best[0] or (n == best[0] and min(a, 360 - a) < min(best[1], 360 - best[1])):
            best = (n, a, ms, M)
    n, a, ms, M = best
    if a != 0.0 and len(ms.src_pts):
        back = np.linalg.inv(M)
        ms.src_pts = (np.c_[ms.src_pts, np.ones(len(ms.src_pts))] @ back.T)[:, :2]
    ms.rotation_search = {"applied": True, "angle_deg": a, "inliers": n,  # type: ignore[attr-defined]
                          "tried": [[float(x), int(y)] for x, y in tried]}
    return ms
