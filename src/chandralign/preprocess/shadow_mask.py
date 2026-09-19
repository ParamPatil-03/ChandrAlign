"""Shadow detection and masking. Owner: Member A (Part 1). Features: PREP-04.

Shadows move with the sun, so a shadow edge in one image is never the same ground
feature in an image taken under another sun (failure mode #2). The shadow mask
tells the matcher which pixels to leave out.

    plane = with_shadow_mask(read_tile(meta, window))    # BEFORE prepare_plane (CLAHE)
    ready = prepare_plane(plane)                          # carries the shadow mask through
    ready.shadow_mask                                     # True = shadowed, exclude

Rule, derived from the image itself (so it adapts to every camera and sun angle):
    black level   = 1st percentile of the valid pixels (the sensor's dark floor --
                    the Moon has no air to scatter light into shadows)
    sunlit level  = median of the valid pixels
    shadow        = brightness < black + dark_fraction * (sunlit - black)
                    AND local variation (5 x 5 std) below max_local_std of that span
                    (the plan's region-level variance check: shadows are uniformly dark)
Pixels passing both tests are SEEDS; a 3 x 3 opening removes specks, a closing fills
pinholes, seeds smaller than min_area_px are dropped, and each remaining seed then grows
over every connected dark pixel (hysteresis), so shadow edges -- where the 5 x 5 window
straddles dark and bright -- are included. Dark patches with no seed stay out.

It must run on linear brightness (raw or min/max-stretched), not after CLAHE,
which reshapes brightness non-linearly; with_shadow_mask refuses a CLAHE'd plane.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np

from chandralign.contracts import ImagePlane

DEFAULT_DARK_FRACTION = 0.15
DEFAULT_MAX_LOCAL_STD = 0.10
DEFAULT_MIN_AREA_PX = 20


@dataclass(frozen=True)
class ShadowResult:
    mask: np.ndarray                 # bool, True = shadow
    black_level: float
    sunlit_level: float
    threshold: float
    fraction: float                  # shadowed share of the valid pixels
    notes: tuple[str, ...] = ()


def detect_shadows(image: np.ndarray, valid: np.ndarray,
                   dark_fraction: float = DEFAULT_DARK_FRACTION,
                   max_local_std: float = DEFAULT_MAX_LOCAL_STD,
                   min_area_px: int = DEFAULT_MIN_AREA_PX,
                   incidence_deg: Optional[float] = None) -> ShadowResult:
    """Shadow mask for a linear-brightness image. See the module docstring for the rule."""
    if not 0.0 < dark_fraction < 1.0:
        raise ValueError("dark_fraction must be within (0, 1)")
    mask = np.zeros(image.shape, dtype=bool)
    if not valid.any():
        return ShadowResult(mask, np.nan, np.nan, np.nan, 0.0, ("no valid pixels",))

    values = image[valid].astype(np.float64)
    black = float(np.percentile(values, 1.0))
    sunlit = float(np.median(values))
    notes = []
    if sunlit <= black:
        return ShadowResult(mask, black, sunlit, np.nan, 0.0,
                            ("no contrast between dark floor and median: no shadow decision possible",))
    span = sunlit - black
    threshold = black + dark_fraction * span

    img = image.astype(np.float64)
    fill = np.where(valid, img, sunlit)                           # invalid pixels never count as dark
    mean = cv2.blur(fill, (5, 5), borderType=cv2.BORDER_REFLECT)
    sq = cv2.blur(fill * fill, (5, 5), borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(np.maximum(sq - mean * mean, 0.0))

    dark = valid & (img < threshold)
    # Seeds: dark AND uniform. The uniformity test is unreliable right at a shadow's
    # edge (the 5x5 window straddles dark and bright), so it only picks seeds...
    seeds = dark & (local_std < max_local_std * span)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    good_seed = np.zeros(n, dtype=bool)
    good_seed[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    seed_mask = good_seed[labels]
    # ...then each seed grows over every CONNECTED dark pixel (hysteresis), so shadow
    # edges -- the features that move most with the sun -- end up inside the mask.
    # Dark patches that contain no uniform seed (dark textured material) stay out.
    n_dark, dark_labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    seeded = np.zeros(n_dark, dtype=bool)
    seeded[np.unique(dark_labels[seed_mask])] = True
    seeded[0] = False
    mask = seeded[dark_labels] & valid

    fraction = float(mask[valid].mean())
    if incidence_deg is not None and incidence_deg < 30.0 and fraction > 0.05:
        notes.append(f"sun {90 - incidence_deg:.0f} deg high but {fraction:.1%} flagged as shadow -- "
                     "check for dark material being mistaken for shadow")
    return ShadowResult(mask, black, sunlit, threshold, fraction, tuple(notes))


def with_shadow_mask(plane: ImagePlane, **kwargs) -> ImagePlane:
    """A new ImagePlane whose shadow_mask comes from detect_shadows(). Input unchanged."""
    if any(step.startswith("clahe") for step in plane.preprocess_chain):
        raise ValueError("detect shadows before CLAHE: CLAHE reshapes brightness non-linearly")
    incidence = plane.meta.solar_incidence_deg if plane.meta.label_fields_verified.get("solar_incidence_deg") else None
    result = detect_shadows(plane.array, plane.valid_mask, incidence_deg=incidence, **kwargs)
    return replace(plane, shadow_mask=result.mask, valid_mask=plane.valid_mask.copy(),
                   preprocess_chain=list(plane.preprocess_chain) + [f"shadow_f{kwargs.get('dark_fraction', DEFAULT_DARK_FRACTION):g}"])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks (1.0 when both are empty)."""
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)
