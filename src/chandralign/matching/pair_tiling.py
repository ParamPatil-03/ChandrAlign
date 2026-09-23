"""Match a PRE-ALIGNED pair tile by tile (xoftr speed work).

WHY
XoFTR matches at native resolution and builds a dense coarse-level similarity
matrix between every 1/8-resolution token of one image and every token of the
other. That matrix grows with the SQUARE of the pixel count: on the ~1040 px
TMC-2 -> SELENE TC windows it peaks near 8 GB on a 6 GB card, and the overflow
into shared system memory is what makes a call take 5 to 50 s instead of ~1 s.
Splitting the window keeps every call inside VRAM.

WHAT IT ASSUMES, AND REFUSES OTHERWISE
Source tile i is matched against the reference region it covers, enlarged by a
margin. That pairing is only meaningful once the coarse stage has put both
images in one frame with a residual shift smaller than the margin, so the two
planes must be the same size; anything else raises. Nothing here is a general
matcher: it is a memory strategy for the fine stage of an aligned pair.

HOW MATCHES MERGE
Source tiles partition the window (no overlap), so no correspondence can be
produced twice. Each tile's matches are shifted back to window coordinates and
concatenated; the robust estimator then runs ONCE on the union, so the reported
transform, inliers and RMSE are still one global fit, not an average of fits.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from ..contracts import MatchSet

MatchFn = Callable[[object, object], Optional[MatchSet]]


def grid(h: int, w: int, target: int = 640) -> list[tuple[int, int, int, int]]:
    """Equal tiles, ceil(side / target) per axis, that partition an h x w window.

    Returns (y0, y1, x0, x1) per tile. 640 is the size XoFTR's weights were
    trained at, so a tile is never larger than the model has seen.
    """
    ny, nx = max(1, math.ceil(h / target)), max(1, math.ceil(w / target))
    ys = np.linspace(0, h, ny + 1).round().astype(int)
    xs = np.linspace(0, w, nx + 1).round().astype(int)
    return [(int(ys[i]), int(ys[i + 1]), int(xs[j]), int(xs[j + 1]))
            for i in range(ny) for j in range(nx)]


def _crop(plane, y0: int, y1: int, x0: int, x1: int):
    from dataclasses import replace

    return replace(plane,
                   array=np.ascontiguousarray(np.asarray(plane.array)[y0:y1, x0:x1]),
                   valid_mask=np.ascontiguousarray(np.asarray(plane.valid_mask)[y0:y1, x0:x1]),
                   shadow_mask=np.ascontiguousarray(np.asarray(plane.shadow_mask)[y0:y1, x0:x1]),
                   tile_origin=(int(y0), int(x0)))


def match_tiled(src, ref, match_fn: MatchFn, *, target: int = 640, margin: int = 64) -> MatchSet:
    """Match `src` against `ref` tile by tile and return one merged MatchSet.

    `match_fn(src_tile, ref_tile)` is the real per-call matcher. Returned points
    are in whole-window pixel coordinates of each plane.
    """
    h, w = np.asarray(src.array).shape[:2]
    if np.asarray(ref.array).shape[:2] != (h, w):
        raise ValueError(
            f"tiled matching needs a pre-aligned pair of equal size; got source {(h, w)} "
            f"and reference {np.asarray(ref.array).shape[:2]}. Run the coarse stage first.")

    src_parts, ref_parts, conf_parts = [], [], []
    method = regime = stage = None
    tiles = grid(h, w, target)
    for y0, y1, x0, x1 in tiles:
        ry0, ry1 = max(0, y0 - margin), min(h, y1 + margin)
        rx0, rx1 = max(0, x0 - margin), min(w, x1 + margin)
        ms = match_fn(_crop(src, y0, y1, x0, x1), _crop(ref, ry0, ry1, rx0, rx1))
        if ms is None:
            continue
        method, regime, stage = ms.method, ms.regime, ms.stage
        sp = np.asarray(ms.src_pts, np.float64).reshape(-1, 2)
        rp = np.asarray(ms.ref_pts, np.float64).reshape(-1, 2)
        src_parts.append(sp + [x0, y0])
        ref_parts.append(rp + [rx0, ry0])
        conf_parts.append(np.asarray(ms.confidence, np.float32).ravel())

    empty = np.zeros((0, 2))
    out = MatchSet(
        src_pts=np.vstack(src_parts) if src_parts else empty,
        ref_pts=np.vstack(ref_parts) if ref_parts else empty.copy(),
        confidence=np.concatenate(conf_parts) if conf_parts else np.zeros(0, np.float32),
        method=method or "tiled", regime=regime or "same_modal_normal", stage=stage or "direct")
    out.tiles = len(tiles)                                   # type: ignore[attr-defined]
    out.tile_margin_px = int(margin)                         # type: ignore[attr-defined]
    out.tile_boxes = tiles                                   # type: ignore[attr-defined]
    return out
