"""Matcher-free NCC probes: an accuracy signal independent of the matcher (audit I-08, I-11).

Moved here from scripts/parallax_probe_eval.py (unchanged: same constants, same acceptance rules), so the
product path can grade the geometry it delivers, not only the research scripts.

A 31 px source template every 48 px is searched +-32 px in the reference with normalised cross-correlation.
A probe is kept only when its peak is strong (>= MIN_PEAK), inside the search window, and clearly above any
other peak more than EXCL_R px away (>= MIN_MARGIN). Its displacement is then compared with a geometry's
prediction. The probes share no code or training with any matcher; they do see the same two images, so a
shading bias common to both is invisible to them (docs/AUDIT_2026-09-26.md section 0.5).
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

STEP, HALF, SEARCH = 48, 15, 32
MIN_PEAK, MIN_MARGIN, EXCL_R = 0.6, 0.1, 3


def _parabola(a: float, b: float, c: float) -> float:
    d = a - 2 * b + c
    return 0.0 if d >= 0 else 0.5 * (a - c) / d


def probes(si: np.ndarray, ri: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(x, y, dx, dy) per accepted probe: the reference sits at (x + dx, y + dy)."""
    si = np.asarray(si, np.float32)
    ri = np.asarray(ri, np.float32)
    ok = np.asarray(ok, bool)
    H, W = si.shape
    out = []
    yy, xx = np.mgrid[:2 * SEARCH + 1, :2 * SEARCH + 1]
    b = HALF + SEARCH
    for y in range(b, H - b, STEP):
        for x in range(b, W - b, STEP):
            if not ok[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1].all():
                continue
            t = si[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
            if t.std() < 1e-6:
                continue
            r = cv2.matchTemplate(ri[y - b:y + b + 1, x - b:x + b + 1], t, cv2.TM_CCOEFF_NORMED)
            iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
            pk = float(r[iy, ix])
            if pk < MIN_PEAK or iy in (0, r.shape[0] - 1) or ix in (0, r.shape[1] - 1):
                continue
            if pk - float(r[(yy - iy) ** 2 + (xx - ix) ** 2 > EXCL_R ** 2].max()) < MIN_MARGIN:
                continue
            dy = iy - SEARCH + _parabola(r[iy - 1, ix], pk, r[iy + 1, ix])
            dx = ix - SEARCH + _parabola(r[iy, ix - 1], pk, r[iy, ix + 1])
            out.append((x, y, dx, dy))
    return np.array(out, float).reshape(-1, 4)


def geometry_error(src_img, ref_img, src_ok, predict: Callable[[np.ndarray], np.ndarray],
                   ref_px_per_src_px: Optional[float] = None) -> dict:
    """p50 / p95 / max of |probe displacement - geometry displacement|, in reference px and source px.

    `predict(pts)` maps (N, 2) source px to reference px (the DELIVERED geometry). Probes are placed in the
    source frame and search the same position in the reference, so this suits images already in about the
    same frame (after a coarse lock or a resample); with a large offset or scale gap nothing is accepted."""
    pr = probes(src_img, ref_img, src_ok)
    if not len(pr):
        return {"n": 0}
    pts, meas = pr[:, :2], pr[:, 2:]
    e = np.hypot(*(meas - (np.asarray(predict(pts), float) - pts)).T)
    e = e[np.isfinite(e)]
    if not len(e):
        return {"n": 0}
    out = {"n": int(len(e)), "p50_px_ref": round(float(np.median(e)), 4),
           "p95_px_ref": round(float(np.percentile(e, 95)), 4), "max_px_ref": round(float(e.max()), 4)}
    if ref_px_per_src_px and np.isfinite(ref_px_per_src_px) and ref_px_per_src_px > 0:
        k = float(ref_px_per_src_px)
        out.update(p50_px_src=round(out["p50_px_ref"] / k, 4), p95_px_src=round(out["p95_px_ref"] / k, 4))
    return out


def geometry_error_warped(src_img, ref_img, src_ok, predict: Callable[[np.ndarray], np.ndarray],
                          ref_ok=None, step: int = 8,
                          ref_px_per_src_px: Optional[float] = None) -> dict:
    """Probe error of a geometry, measured in its OWN frame (audit I-08 fix).

    The reference is resampled into the source frame through the geometry (`predict`: source px -> reference
    px, evaluated every `step` px and interpolated), and the probes then compare the source with that resampled
    reference. A correct geometry leaves every probe at zero displacement, so each residual IS the geometry's
    error, in SOURCE px. Unlike geometry_error, this does not need the two images to share a frame: axis-aligned
    templates across a rotation or a scale gap were almost never accepted, so a raw rotated pair came out
    "unmeasured" (the synthetic C-04 replay: no result could be HIGH).
    """
    from scipy.ndimage import map_coordinates
    s = np.asarray(src_img, np.float32)
    r = np.asarray(ref_img, np.float32)
    H, W = s.shape
    ys, xs = np.mgrid[0:H + step:step, 0:W + step:step].astype(np.float64)
    g = np.c_[np.minimum(xs.ravel(), W - 1), np.minimum(ys.ravel(), H - 1)]
    q = np.asarray(predict(g), np.float64)
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
    at = [gy.ravel() / step, gx.ravel() / step]
    mx = map_coordinates(q[:, 0].reshape(ys.shape), at, order=1).reshape(H, W).astype(np.float32)
    my = map_coordinates(q[:, 1].reshape(ys.shape), at, order=1).reshape(H, W).astype(np.float32)
    r_in_s = cv2.remap(r, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rv = np.ones(r.shape, np.float32) if ref_ok is None else np.asarray(ref_ok, np.float32)
    inside = cv2.remap(rv, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    ok = np.asarray(src_ok, bool) & inside & np.isfinite(mx) & np.isfinite(my)
    pr = probes(s, r_in_s, ok)
    if not len(pr):
        return {"n": 0, "frame": "source (reference resampled through the geometry)"}
    e = np.hypot(pr[:, 2], pr[:, 3])
    out = {"n": int(len(e)), "frame": "source (reference resampled through the geometry)",
           "p50_px_src": round(float(np.median(e)), 4), "p95_px_src": round(float(np.percentile(e, 95)), 4),
           "max_px_src": round(float(e.max()), 4)}
    if ref_px_per_src_px and np.isfinite(ref_px_per_src_px) and ref_px_per_src_px > 0:
        k = float(ref_px_per_src_px)
        out.update(p50_px_ref=round(out["p50_px_src"] * k, 4), p95_px_ref=round(out["p95_px_src"] * k, 4))
    return out
