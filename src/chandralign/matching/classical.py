"""Classical detectors through OpenCV (MATCH-01 / MATCH-02, P2-T02).

This is the documented baseline and the ablation floor: every real lunar
registration paper uses SIFT as its comparison point, so we must be able to
reproduce it honestly rather than only showing our best configuration.

It is deliberately dependency-light -- OpenCV only, no torch, no downloaded
weights, no licence restrictions. That makes it the path that still works when
the GPU is absent, the network is down at the venue (risk R6), or a checkpoint
fails to download.

Shadowed pixels are excluded from detection. A cast shadow is not a surface
feature: its position depends on the Sun, so keypoints found on a shadow edge
move between acquisitions and generate confident, wrong correspondences.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import compute, config
from ..contracts import ImagePlane, MatchSet
from . import filters

_BINARY = {"orb", "akaze", "brisk"}


def _to_u8(plane: ImagePlane) -> np.ndarray:
    arr = np.asarray(plane.array, np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-9:
        return np.zeros(arr.shape, np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _detection_mask(plane: ImagePlane) -> np.ndarray:
    """Where the detector is allowed to look: valid, and not in shadow."""
    mask = np.asarray(plane.valid_mask, bool)
    shadow = getattr(plane, "shadow_mask", None)
    if shadow is not None:
        mask = mask & ~np.asarray(shadow, bool)
    return mask.astype(np.uint8) * 255


def make_detector(name: str = "sift", max_keypoints: int | None = None):
    n = int(max_keypoints or config.get("matching.max_num_keypoints", 2048))
    key = name.lower()
    if key == "sift":
        return cv2.SIFT_create(nfeatures=n)
    if key == "akaze":
        return cv2.AKAZE_create()
    if key == "orb":
        return cv2.ORB_create(nfeatures=n)
    if key == "brisk":
        return cv2.BRISK_create()
    raise ValueError(f"unknown classical detector {name!r}")


def detect(plane: ImagePlane, detector: str = "sift", max_keypoints: int | None = None):
    det = make_detector(detector, max_keypoints)
    kps, desc = det.detectAndCompute(_to_u8(plane), _detection_mask(plane))
    return kps, desc


def match(src: ImagePlane, ref: ImagePlane, *, detector: str = "sift",
          ratio: float | None = None, mutual: bool | None = None,
          max_keypoints: int | None = None, regime: str = "same_modal_normal",
          stage: str = "direct") -> MatchSet:
    """Detect, describe and match two planes with a classical detector."""
    kp_src, desc_src = detect(src, detector, max_keypoints)
    kp_ref, desc_ref = detect(ref, detector, max_keypoints)

    if ratio is None:
        # Tighten automatically where the source looks self-similar.
        ratio = filters.adaptive_ratio(getattr(src, "repetitiveness_score", None))
    if mutual is None:
        mutual = bool(config.get("matching.mutual_nn", True))

    norm = cv2.NORM_HAMMING if detector.lower() in _BINARY else cv2.NORM_L2
    i_src, i_ref, dist = filters.match_descriptors(
        desc_src, desc_ref, ratio=float(ratio), mutual=mutual, norm=norm)

    src_pts = np.array([kp_src[int(i)].pt for i in i_src], np.float64).reshape(-1, 2)
    ref_pts = np.array([kp_ref[int(i)].pt for i in i_ref], np.float64).reshape(-1, 2)

    ms = MatchSet(src_pts=src_pts, ref_pts=ref_pts,
                  confidence=filters.confidence_from_distance(dist),
                  method=detector.lower(), regime=regime, stage=stage)
    # Carried for diagnostics (keypoint totals for Metrics, rotation check for
    # failure mode #6). Not part of the frozen contract, so read defensively.
    ms.n_keypoints_src = len(kp_src)      # type: ignore[attr-defined]
    ms.n_keypoints_ref = len(kp_ref)      # type: ignore[attr-defined]
    ms.orientation_peak = filters.orientation_consistency(  # type: ignore[attr-defined]
        kp_src, kp_ref, i_src, i_ref)
    # Record the device this actually ran on, rather than letting a reader infer
    # it from what the machine has available. OpenCV pip wheels carry no CUDA,
    # so this path is cpu regardless of any GPU present.
    ms.device = compute.classical_device()   # type: ignore[attr-defined]
    return ms
