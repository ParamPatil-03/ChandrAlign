"""Flat- and repetitive-terrain scoring. Owner: Member A (Part 1). Features: PREP-07, PREP-08.

Two different ways terrain defeats a matcher, and one score for each:

    texture_score          how much real, matchable structure a tile holds.
                           Low -> featureless mare: too few keypoints, so Part 2
                           should fall back to phase-congruency descriptors and
                           widen its search (failure mode #1).

    repetitiveness_score   the share of features that have a LOOK-ALIKE elsewhere
                           in the same tile. High -> crater field: the matcher can
                           confidently match the wrong crater, so Part 2 should
                           tighten its ratio test (failure mode #11).

    plane = with_terrain_scores(plane)
    plane.texture_score, plane.repetitiveness_score

texture_score measures STRUCTURE, NOT NOISE. A noisy image is not a textured one:
the score is the amplitude of a band-pass (difference-of-Gaussians, sigma 1 and 4)
response with the noise contribution removed in quadrature, using the noise a white
signal would produce through that same filter. On a real dark, noise-dominated OHRC
region the plain pixel spread is the LARGEST in the strip while this score is the
SMALLEST -- which is the honest answer, because there is nothing there to match.

repetitiveness_score does NOT use autocorrelation. PLAN.md P1-T16 proposes
"autocorrelation peak structure", which detects PERIODIC patterns; measured on a
real OHRC crater field that scores 0.111 against 0.118 for ordinary terrain and
0.034 for pure noise -- no separation at all, because craters are scattered, not
laid out on a grid. What actually causes false matches is that craters LOOK ALIKE
wherever they sit, so the score compares descriptors instead: for every keypoint,
the distance to the most similar OTHER keypoint (at least `min_separation_px`
away, so a feature cannot twin with itself) against the typical distance in the
tile. Measured: synthetic periodic craters 0.98, real crater field 0.28, ordinary
terrain 0.24, SELENE TC 0.17, a smooth area 0.09, pure noise 0.00.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np

from chandralign.contracts import ImagePlane

STRUCTURE_SIGMA_SMALL = 1.0        # below this scale lives sensor noise
STRUCTURE_SIGMA_LARGE = 4.0        # above it, illumination gradients rather than terrain detail
TWIN_DISTANCE_FRACTION = 0.5       # "look-alike" = within half the tile's typical descriptor distance
MIN_SEPARATION_PX = 20.0           # a feature may not twin with itself or its immediate neighbours
MAX_KEYPOINTS = 400
MIN_KEYPOINTS = 20                 # fewer than this and repetitiveness is not measurable


@dataclass(frozen=True)
class TerrainScores:
    texture: float                  # structure amplitude, noise removed (0..1 image units)
    repetitiveness: Optional[float]  # share of features with a look-alike; None if too few features
    noise: float                    # pixel-to-pixel noise, same units
    raw_structure: float            # band-pass amplitude BEFORE removing noise
    keypoints: int


def pixel_noise(image: np.ndarray) -> float:
    """Pixel-to-pixel noise, robust to terrain.

    The diagonal double difference cancels any locally planar surface, and MAD
    ignores the craters and edges that survive it. For white noise of spread s the
    double difference also has spread s, so the estimate is unbiased. (The cube
    version used for IIRS bands is io-side, in preprocess.iirs_composite.band_noise.)
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError(f"expected a 2-D image of at least 2x2, got {a.shape}")
    d = (a[1:, 1:] - a[1:, :-1] - a[:-1, 1:] + a[:-1, :-1]) / 2.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def _noise_gain(sigma_small: float, sigma_large: float) -> float:
    """How much white noise survives the band-pass, as a factor on its spread.

    For a Gaussian of width s, a unit-variance white field comes through with
    variance 1/(4*pi*s^2), and the two blurs share covariance 1/(2*pi*(s1^2+s2^2)).
    """
    var = (1.0 / (4 * math.pi * sigma_small ** 2)
           + 1.0 / (4 * math.pi * sigma_large ** 2)
           - 2.0 / (2 * math.pi * (sigma_small ** 2 + sigma_large ** 2)))
    return math.sqrt(max(var, 0.0))


def band_pass(image: np.ndarray, valid: Optional[np.ndarray] = None,
              sigma_small: float = STRUCTURE_SIGMA_SMALL,
              sigma_large: float = STRUCTURE_SIGMA_LARGE) -> np.ndarray:
    """Difference of Gaussians: drops the illumination gradient and the pixel noise."""
    a = np.asarray(image, dtype=np.float64)
    if valid is not None and not valid.all():
        a = np.where(valid, a, np.median(a[valid]) if valid.any() else 0.0)
    return cv2.GaussianBlur(a, (0, 0), sigma_small) - cv2.GaussianBlur(a, (0, 0), sigma_large)


def texture_score(image: np.ndarray, valid: Optional[np.ndarray] = None,
                  sigma_small: float = STRUCTURE_SIGMA_SMALL,
                  sigma_large: float = STRUCTURE_SIGMA_LARGE) -> tuple[float, float, float]:
    """(texture, raw structure, noise). Texture is the structure left once noise is removed."""
    valid = np.ones(np.shape(image), bool) if valid is None else valid
    if not valid.any():
        return 0.0, 0.0, 0.0
    noise = pixel_noise(np.where(valid, image, np.median(np.asarray(image)[valid])))
    bp = band_pass(image, valid, sigma_small, sigma_large)
    raw = float(bp[valid].std())
    expected_from_noise = _noise_gain(sigma_small, sigma_large) * noise
    return float(math.sqrt(max(raw ** 2 - expected_from_noise ** 2, 0.0))), raw, noise


def _as_uint8(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(a[valid], [1, 99]) if valid.any() else (0.0, 1.0)
    return (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


def repetitiveness_score(image: np.ndarray, valid: Optional[np.ndarray] = None,
                         twin_fraction: float = TWIN_DISTANCE_FRACTION,
                         min_separation_px: float = MIN_SEPARATION_PX,
                         max_keypoints: int = MAX_KEYPOINTS) -> tuple[Optional[float], int]:
    """(share of features with a look-alike elsewhere, keypoint count).

    None when the tile holds too few features to say anything -- a featureless tile
    is a texture problem, not a repetitiveness one, and must not be reported as 0.
    """
    valid = np.ones(np.shape(image), bool) if valid is None else valid
    detector = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = detector.detectAndCompute(_as_uint8(image, valid), None)
    if descriptors is not None and not valid.all():                # drop features on no-data
        keep = [i for i, k in enumerate(keypoints)
                if valid[min(int(k.pt[1]), valid.shape[0] - 1), min(int(k.pt[0]), valid.shape[1] - 1)]]
        keypoints = [keypoints[i] for i in keep]
        descriptors = descriptors[keep] if keep else None
    if descriptors is None or len(descriptors) < MIN_KEYPOINTS:
        return None, 0 if descriptors is None else len(descriptors)

    d = descriptors.astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    distances = np.sqrt(np.maximum(2 - 2 * (d @ d.T), 0.0))
    np.fill_diagonal(distances, np.inf)
    pts = np.float32([k.pt for k in keypoints])
    close = np.hypot(pts[:, 0][:, None] - pts[:, 0][None, :],
                     pts[:, 1][:, None] - pts[:, 1][None, :]) < min_separation_px
    distances[close] = np.inf                                       # cannot twin with itself
    nearest = distances.min(axis=1)
    finite = distances[np.isfinite(distances)]
    if not finite.size or not np.isfinite(nearest).any():
        return None, len(keypoints)
    typical = float(np.median(finite))
    usable = np.isfinite(nearest)
    return float((nearest[usable] < twin_fraction * typical).mean()), len(keypoints)


def terrain_scores(image: np.ndarray, valid: Optional[np.ndarray] = None) -> TerrainScores:
    texture, raw, noise = texture_score(image, valid)
    repetitive, keypoints = repetitiveness_score(image, valid)
    return TerrainScores(texture, repetitive, noise, raw, keypoints)


def with_terrain_scores(plane: ImagePlane) -> ImagePlane:
    """A new ImagePlane with texture_score and repetitiveness_score filled in."""
    scores = terrain_scores(plane.array, plane.valid_mask)
    return replace(plane, texture_score=scores.texture, repetitiveness_score=scores.repetitiveness,
                   valid_mask=plane.valid_mask.copy(), shadow_mask=plane.shadow_mask.copy())
