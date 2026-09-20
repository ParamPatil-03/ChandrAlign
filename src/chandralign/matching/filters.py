"""Correspondence filtering: ratio test, mutual nearest neighbour, adaptive
thresholds (MATCH-14 / MATCH-15, P2-T02 and P2-T14).

The ratio test and mutual-NN check are standard. The part no library can do for
us is the ADAPTIVE threshold: Lowe's ratio assumes the second-best match is a
fair proxy for "a random wrong match", which stops being true on repetitive
terrain. A crater field is full of near-identical craters, so the true match and
a wrong one score almost the same and the ratio never fires -- failure mode #11.
The fix is to tighten the threshold exactly where the scene is self-similar,
which needs `repetitiveness_score` off the ImagePlane (Member A's PREP work).
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config


def adaptive_ratio(repetitiveness: float | None = None,
                   base: float | None = None,
                   repetitive: float | None = None) -> float:
    """Pick a Lowe ratio threshold for this scene.

    Interpolates between the normal and the tightened threshold using the
    repetitiveness score, instead of switching hard at an arbitrary cut-off.
    With no score available it returns the base threshold and does not guess.
    """
    if base is None:
        base = float(config.get("matching.ratio_test", 0.80))
    if repetitive is None:
        repetitive = float(config.get("matching.ratio_test_repetitive", 0.70))
    if repetitiveness is None:
        return base
    w = float(np.clip(repetitiveness, 0.0, 1.0))
    return float(base + w * (repetitive - base))


def ratio_test(knn_distances: np.ndarray, ratio: float) -> np.ndarray:
    """Lowe's ratio test over (N,2) sorted knn distances -> boolean keep mask."""
    d = np.asarray(knn_distances, np.float64)
    if d.ndim != 2 or d.shape[1] < 2:
        raise ValueError("ratio test needs the two nearest distances per query")
    second = np.where(d[:, 1] <= 0, 1e-12, d[:, 1])
    return (d[:, 0] / second) < float(ratio)


def match_descriptors(desc_src: np.ndarray, desc_ref: np.ndarray, *,
                      ratio: float = 0.8, mutual: bool = True,
                      norm: int = cv2.NORM_L2):
    """Match two descriptor sets. Returns (idx_src, idx_ref, distances).

    Mutual nearest neighbour is applied by matching in both directions and
    keeping only pairs that choose each other. It is cheap and removes a whole
    class of one-sided false matches before they reach RANSAC.
    """
    if desc_src is None or desc_ref is None or len(desc_src) < 2 or len(desc_ref) < 2:
        empty_i = np.zeros(0, int)
        return empty_i, empty_i.copy(), np.zeros(0, np.float64)

    bf = cv2.BFMatcher(norm)
    fwd = bf.knnMatch(desc_src, desc_ref, k=2)
    idx_src, idx_ref, dist, second = [], [], [], []
    for pair in fwd:
        if len(pair) < 2:
            continue
        a, b = pair[0], pair[1]
        idx_src.append(a.queryIdx)
        idx_ref.append(a.trainIdx)
        dist.append(a.distance)
        second.append(b.distance)
    if not idx_src:
        empty_i = np.zeros(0, int)
        return empty_i, empty_i.copy(), np.zeros(0, np.float64)

    idx_src = np.asarray(idx_src, int)
    idx_ref = np.asarray(idx_ref, int)
    dist = np.asarray(dist, np.float64)
    keep = ratio_test(np.stack([dist, np.asarray(second, np.float64)], 1), ratio)

    if mutual:
        back = bf.knnMatch(desc_ref, desc_src, k=1)
        best_back = {m[0].queryIdx: m[0].trainIdx for m in back if m}
        mutual_ok = np.array([best_back.get(int(r), -1) == int(s)
                              for s, r in zip(idx_src, idx_ref)], bool)
        keep &= mutual_ok

    return idx_src[keep], idx_ref[keep], dist[keep]


def confidence_from_distance(dist: np.ndarray) -> np.ndarray:
    """Turn descriptor distances into a 0..1 confidence for MatchSet.

    Monotonic and scale-free (normalised by the observed spread), so it is
    comparable across detectors with different distance ranges. It is a
    ranking signal, not a probability, and is not reported as one.
    """
    d = np.asarray(dist, np.float64)
    if d.size == 0:
        return np.zeros(0, np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-12:
        return np.ones(len(d), np.float32)
    return (1.0 - (d - lo) / (hi - lo)).astype(np.float32)


def orientation_consistency(kp_src, kp_ref, idx_src, idx_ref, bins: int = 36):
    """Peak fraction of the relative-orientation histogram, in [0, 1].

    A correct match set shares one dominant rotation, so the histogram of
    per-match orientation differences spikes. A flat histogram means the matches
    disagree about the rotation, which is the pre-RANSAC warning sign for
    failure mode #6 (extreme rotation) -- cheap to compute and worth checking
    before spending RANSAC iterations on a hopeless set.
    """
    if len(idx_src) == 0:
        return 0.0, 0.0
    diff = np.array([(kp_src[int(s)].angle - kp_ref[int(r)].angle) % 360.0
                     for s, r in zip(idx_src, idx_ref)], np.float64)
    hist, edges = np.histogram(diff, bins=bins, range=(0.0, 360.0))
    if hist.sum() == 0:
        return 0.0, 0.0
    peak = int(np.argmax(hist))
    centre = 0.5 * (edges[peak] + edges[peak + 1])
    return float(hist[peak] / hist.sum()), float(centre)
