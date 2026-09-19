"""CLAHE, percentile stretch, 0..1 normalisation. Owner: Member A (Part 1). Features: PREP-01.

Makes images taken under different sun angles comparable before matching. Works on
ImagePlane copies; nothing on disk is ever changed.

    plane = read_tile(meta, window)                      # io.tiling, raw min/max stretch
    ready = prepare_plane(plane)                         # percentile stretch + CLAHE
    ready.preprocess_chain                               # [..., "stretch_p1-99", "clahe_c2.0_t8x8"]

Steps:
1. Percentile stretch of the VALID pixels to 0..1 (default 1st..99th percentile), so a
   handful of saturated or dead pixels cannot set the scale.
2. CLAHE (contrast-limited adaptive histogram equalisation, OpenCV), which equalises
   contrast region by region: a dark crater floor and a bright rim each get their own
   stretch. The clip limit stops noise in flat regions being amplified without bound.
   Run in 16 bits so 16-bit sensors (TMC-2, NAC, SELENE) keep their precision.

Invalid pixels are filled with the median valid value before CLAHE -- so they cannot
distort the local histograms -- and set back to 0 afterwards; the valid mask is
carried through unchanged (as a new array, never shared with the shadow mask).
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from chandralign.contracts import ImagePlane

DEFAULT_CLIP = 2.0
DEFAULT_TILES = (8, 8)
DEFAULT_PERCENTILES = (1.0, 99.0)


def percentile_stretch(array: np.ndarray, valid: np.ndarray,
                       percentiles: tuple[float, float] = DEFAULT_PERCENTILES) -> tuple[np.ndarray, float, float]:
    """Map the valid pixels' [p_low, p_high] range to 0..1 (clipped). Invalid pixels -> 0.

    Returns (float32 image, low value, high value). A constant image gives all zeros.
    """
    low_p, high_p = percentiles
    if not 0.0 <= low_p < high_p <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    out = np.zeros(array.shape, dtype=np.float32)
    if not valid.any():
        return out, float("nan"), float("nan")
    values = array[valid].astype(np.float64)
    lo, hi = np.percentile(values, [low_p, high_p])
    if hi > lo:
        out[valid] = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return out, float(lo), float(hi)


def clahe(image01: np.ndarray, valid: np.ndarray, clip_limit: float = DEFAULT_CLIP,
          tiles: tuple[int, int] = DEFAULT_TILES) -> np.ndarray:
    """CLAHE on a 0..1 float image, computed in 16 bits. Invalid pixels come back as 0."""
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")
    if not valid.any():
        return np.zeros(image01.shape, dtype=np.float32)
    work = np.clip(image01, 0.0, 1.0).astype(np.float64)
    work[~valid] = np.median(work[valid])            # keep invalid pixels out of the histograms
    u16 = np.rint(work * 65535.0).astype(np.uint16)
    eq = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tiles[1]), int(tiles[0]))).apply(u16)
    out = eq.astype(np.float32) / 65535.0
    out[~valid] = 0.0
    return out


def local_contrast(image: np.ndarray, valid: np.ndarray, window: int = 15) -> float:
    """Mean local standard deviation over `window` x `window` neighbourhoods of valid pixels.

    The acceptance measure for PREP-01: CLAHE should raise it in a dark crop.
    """
    img = np.where(valid, image, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    k = (window, window)
    n = cv2.boxFilter(w, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    s1 = cv2.boxFilter(img, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    s2 = cv2.boxFilter(img * img, -1, k, normalize=False, borderType=cv2.BORDER_CONSTANT)
    full = (n >= window * window - 0.5) & valid       # only windows made entirely of valid pixels
    if not full.any():
        return float("nan")
    var = np.maximum(s2[full] / n[full] - (s1[full] / n[full]) ** 2, 0.0)
    return float(np.sqrt(var).mean())


def prepare_plane(plane: ImagePlane, clip_limit: float = DEFAULT_CLIP,
                  tiles: tuple[int, int] = DEFAULT_TILES,
                  percentiles: tuple[float, float] = DEFAULT_PERCENTILES) -> ImagePlane:
    """A new ImagePlane with percentile stretch + CLAHE applied; the input is not modified."""
    valid = plane.valid_mask.copy()
    stretched, _, _ = percentile_stretch(plane.array, valid, percentiles)
    equalised = clahe(stretched, valid, clip_limit, tiles)
    chain = [step for step in plane.preprocess_chain if step != "tile_minmax"]
    chain += [f"stretch_p{percentiles[0]:g}-{percentiles[1]:g}",
              f"clahe_c{clip_limit:g}_t{tiles[0]}x{tiles[1]}"]
    return replace(plane, array=equalised, valid_mask=valid,
                   shadow_mask=plane.shadow_mask.copy(), preprocess_chain=chain)
