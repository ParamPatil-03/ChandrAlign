"""RIFT2-style radiation-insensitive matching (MATCH-06), a benchmarked candidate.

WHAT IT IS
RIFT (Li, Hu & Zhang, IEEE TIP 2020) and RIFT2 (Li et al., 2023) match images
whose brightness is related NONLINEARLY -- different sensors, different lighting,
even inverted contrast -- by never describing brightness at all:

  1. phase congruency: structure is where the phases of many log-Gabor filters
     line up, which is independent of how bright the structure is
  2. keypoints: FAST on the phase-congruency maximum-moment (edge) map
  3. maximum index map (MIM): at each pixel, WHICH of the N_o filter
     orientations responds most strongly (summed over scales). Inverting the
     contrast leaves every amplitude, and so the MIM, unchanged
  4. descriptor: a grid x grid histogram of MIM indices over a patch (6x6x6 = 216)
  5. RIFT2's rotation step: rotating the image changes which orientation filter
     responds, so the patch is sampled on a grid rotated to the dominant
     orientation AND the MIM indices are cyclically shifted by the same angle

ONE MEASURED DEVIATION. With N_o = 6 the MIM is quantised to 30 degrees, and an
integer index shift cannot compensate a rotation of 2.5 steps: the published
descriptor failed outright at 45, 75 and 105 degrees. The default ("ori")
histograms phasepack's continuous structure orientation RELATIVE to the patch's
dominant orientation instead -- same radiation-insensitive principle, no
rounding -- and holds at every angle tested. The published form is kept as
orientation="mim"; RiftParams records the comparison.

WHAT IT IS NOT
The reference RIFT/RIFT2 code is MATLAB and has no clear redistribution licence,
so nothing is vendored: this is our own implementation of the published method's
structure, not a port. It is called "RIFT2-style" for that reason, and it is not
presumed to be the default. configs/regimes.yaml keeps it as a cross-modal
CANDIDATE, and scripts/bench_rift.py decides from measurements where, if
anywhere, it earns a place.

A PHASEPACK TRAP, PINNED BY A TEST
phasepack's docstring says both "the outer list corresponds to a spatial scale"
and "EO[o][s] is ... orientation o and scale s". Measured: len(EO) == norient
and a single-orientation stripe pattern lights up one OUTER index -- the outer
index is ORIENTATION. Building the MIM over the other axis would silently
produce descriptors of nothing.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from .. import compute
from ..contracts import ImagePlane, MatchSet
from . import filters

# Sign of the MIM index shift that compensates a patch rotation. MEASURED, not
# reasoned: turning an image by +a (OpenCV, anticlockwise on screen) raises the
# MIM index by a/30 at the same physical point (133/139 points for 30 deg,
# 136/139 for 60 deg), so the shift SUBTRACTS k = round(theta / 30).
# Pinned by tests/test_rift.py.
_INDEX_SHIFT_SIGN = -1


@dataclass(frozen=True)
class RiftParams:
    nscale: int = 4
    norient: int = 6
    patch: int = 96            # descriptor window, px (must be divisible by grid)
    grid: int = 6              # grid x grid cells, norient bins each
    max_keypoints: int = 1500
    fast_threshold: int = 5
    ratio: float = 0.9         # MIM histograms are less distinctive than SIFT; see bench
    both_orientations: bool = True
    # "mim": the published maximum-index map, 30-degree quantised (N_o = 6).
    # "ori": phase-congruency orientation relative to the dominant one, binned
    #        into the same N_o bins -- continuous, so no 30-degree rounding.
    # DEFAULT "ori", MEASURED (512 px synthetic, seed 3; RMSE px / inliers):
    #                     mim            ori           SIFT
    #   rot 45 / 75 / 105  FAIL x3        0.08-0.11      0.28-0.55
    #   rot 150            0.088          0.092          0.690
    #   opposite sun       0.170          0.101          FAIL
    #   cross-modal        0.000          0.077          FAIL
    #   cross-modal +30    0.153          0.098          FAIL
    # The published MIM only survives rotations near multiples of 30 deg: an
    # integer index shift cannot compensate 2.5 steps. It keeps one edge -- an
    # exact radiometric remap with no rotation -- so it stays selectable.
    orientation: str = "ori"


@dataclass
class RiftFeatures:
    points: np.ndarray         # (N, 2) x, y
    desc: np.ndarray           # (N, grid*grid*norient) float32
    n_keypoints: int


def _normalise(a: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(a, np.float64)
    v = a[valid] if valid is not None and valid.any() else a
    lo, hi = np.percentile(v, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def phase_maps(img: np.ndarray, p: RiftParams):
    """(M, MIM, A_max, ori): edge moment, maximum index map, strongest amplitude,
    and phasepack's continuous feature orientation (degrees, 0..180)."""
    import phasepack

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M, m, ori, ft, PC, EO, T = phasepack.phasecong(img, nscale=p.nscale, norient=p.norient)
    if len(EO) != p.norient:
        raise RuntimeError(f"phasepack EO has {len(EO)} outer entries, expected norient={p.norient}: "
                           "the orientation/scale nesting is not what this module assumes")
    amp = np.stack([np.sum([np.abs(EO[o][s]) for s in range(p.nscale)], axis=0)
                    for o in range(p.norient)])                       # (norient, H, W)
    return (np.asarray(M, np.float64), np.argmax(amp, axis=0).astype(np.int32), amp.max(axis=0),
            np.asarray(ori, np.float64))


def dominant_orientation(mim: np.ndarray, a_max: np.ndarray, p: RiftParams) -> np.ndarray:
    """Dominant AXIAL orientation (degrees, 0..180) at every pixel.

    Filter orientation o points at o * 180 / norient degrees. Orientations are
    axial -- 10 and 190 degrees are one edge -- so they are averaged as doubled
    angles, weighted by strength and smoothed over the descriptor window.
    """
    # From the MIM, as the published method does. Computing it from phasepack's
    # continuous `ori` instead was tried and MEASURED: over 14 cases (rotations
    # 45-150, cross-modal, opposite sun; 2 seeds) each source won 7 -- no
    # systematic difference. The sabotage harness caught that it was not load-
    # bearing, so it was removed rather than kept as an unjustified deviation.
    theta2 = np.deg2rad(2.0 * mim * (180.0 / p.norient))
    sigma = p.patch / 4.0
    c = cv2.GaussianBlur((a_max * np.cos(theta2)).astype(np.float32), (0, 0), sigma)
    s = cv2.GaussianBlur((a_max * np.sin(theta2)).astype(np.float32), (0, 0), sigma)
    return (np.rad2deg(np.arctan2(s, c)) / 2.0) % 180.0


def _describe(mim: np.ndarray, x: float, y: float, theta_deg: float, p: RiftParams,
              ori: np.ndarray | None = None) -> np.ndarray:
    J, g, No = p.patch, p.grid, p.norient
    u = np.arange(J) - (J - 1) / 2.0
    uu, vv = np.meshgrid(u, u)
    # MEASURED (image turned +a with OpenCV): the MIM index rises by a/30 and the
    # dominant orientation rises by ~a, while pixel offsets transform by Rot(-a).
    # So the sampling grid must turn by -theta for two grids to cover the same
    # ground. A first version turned it by +theta and no index sign could save it.
    t = np.deg2rad(-theta_deg)
    xs = np.rint(x + uu * np.cos(t) - vv * np.sin(t)).astype(int)
    ys = np.rint(y + uu * np.sin(t) + vv * np.cos(t)).astype(int)
    h, w = mim.shape
    np.clip(xs, 0, w - 1, out=xs)
    np.clip(ys, 0, h - 1, out=ys)
    if p.orientation == "ori" and ori is not None:
        # Continuous: orientation RELATIVE to the patch's own, then binned.
        rel = (ori[ys, xs] - theta_deg) % 180.0
        idx = np.minimum((rel / (180.0 / No)).astype(int), No - 1)
    else:
        idx = mim[ys, xs]
        k = int(np.rint(theta_deg / (180.0 / No)))
        idx = (idx + _INDEX_SHIFT_SIGN * k) % No
    onehot = (idx[..., None] == np.arange(No)).reshape(g, J // g, g, J // g, No)
    d = onehot.sum(axis=(1, 3)).astype(np.float32).ravel()
    n = float(np.linalg.norm(d))
    return d / n if n > 0 else d


def features(plane_or_img, p: RiftParams = RiftParams(), mask: np.ndarray | None = None) -> RiftFeatures:
    if p.patch % p.grid:
        raise ValueError(f"patch {p.patch} must be divisible by grid {p.grid}")
    if isinstance(plane_or_img, np.ndarray):
        img, valid = np.asarray(plane_or_img, np.float64), None
    else:
        img = np.asarray(plane_or_img.array, np.float64)
        valid = np.asarray(plane_or_img.valid_mask, bool)
        shadow = getattr(plane_or_img, "shadow_mask", None)
        mask = valid & ~np.asarray(shadow, bool) if shadow is not None else valid
    img = _normalise(img, valid)
    M, mim, a_max, ori = phase_maps(img, p)
    theta = dominant_orientation(mim, a_max, p)

    m8 = (_normalise(M) * 255).astype(np.uint8)
    border = int(np.ceil(p.patch / np.sqrt(2) / 2)) + 1
    det_mask = np.zeros(m8.shape, np.uint8)
    det_mask[border:-border, border:-border] = 255
    if mask is not None:
        det_mask &= (np.asarray(mask, bool).astype(np.uint8) * 255)
    fast = cv2.FastFeatureDetector_create(threshold=p.fast_threshold, nonmaxSuppression=True)
    kps = sorted(fast.detect(m8, det_mask), key=lambda k: -k.response)[: p.max_keypoints]

    pts, descs = [], []
    for kp in kps:
        x, y = kp.pt
        th = float(theta[int(round(y)), int(round(x))])
        for t in ((th, th + 180.0) if p.both_orientations else (th,)):
            pts.append((x, y))
            descs.append(_describe(mim, x, y, t, p, ori))
    if not descs:
        return RiftFeatures(np.zeros((0, 2)), np.zeros((0, p.grid * p.grid * p.norient), np.float32), 0)
    return RiftFeatures(np.asarray(pts, np.float64), np.stack(descs).astype(np.float32), len(kps))


def match(src: ImagePlane, ref: ImagePlane, *, params: RiftParams = RiftParams(),
          regime: str = "cross_modal", stage: str = "direct") -> MatchSet:
    """Detect, describe and match two planes; same MatchSet contract as the others,
    so the result goes through the same filters, MAGSAC and quality gate."""
    t0 = time.perf_counter()
    fs, fr = features(src, params), features(ref, params)
    i_src, i_ref, dist = filters.match_descriptors(fs.desc, fr.desc, ratio=params.ratio,
                                                   mutual=True, norm=cv2.NORM_L2)
    ms = MatchSet(src_pts=fs.points[i_src].reshape(-1, 2), ref_pts=fr.points[i_ref].reshape(-1, 2),
                  confidence=filters.confidence_from_distance(dist),
                  method="rift2", regime=regime, stage=stage)
    ms.n_keypoints_src = fs.n_keypoints            # type: ignore[attr-defined]
    ms.n_keypoints_ref = fr.n_keypoints            # type: ignore[attr-defined]
    ms.device = compute.classical_device()         # type: ignore[attr-defined]
    ms.runtime_s = round(time.perf_counter() - t0, 3)   # type: ignore[attr-defined]
    return ms
