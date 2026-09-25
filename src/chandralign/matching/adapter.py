"""Adapter from our contract to vismatch's matchers (MATCH-03/04/05, P2-T01).

We do not write matching algorithms. vismatch exposes 71 of them behind
get_matcher(name, device) under BSD-3, with weights fetched on first use. This
module is the thin seam between that library and our contract, and it is
deliberately the ONLY place a vismatch call can originate, because three things
have to be enforced on every single call:

1. THE LICENCE GATE. vismatch ships superglue, superpoint-* and r2d2 alongside
   the permissive models, one string apart. The gate runs here so no caller can
   reach a restricted model by passing a different name (CHECK-10).

2. skip_ransac = True. vismatch runs its own RANSAC and returns a homography.
   We disable it so OUR estimator does the geometry: we need the inlier mask,
   the affine-vs-homography model choice, and above all the scale sanity check
   against the instrument registry, none of which the library can do. Taking
   its homography instead would mean shipping a number no check of ours had
   ever seen.

3. THE DEVICE ACTUALLY USED, recorded on the result. A machine having a GPU is
   not evidence that a result came from one.

Shadowed endpoints are dropped after matching. The classical path can exclude
shadow at detection time via a mask; a learned matcher takes a whole image, so
the equivalent filter has to happen on the returned correspondences. The reason
is the same: a cast shadow's position is a function of the Sun, not the surface,
so a keypoint on a shadow edge is a confident wrong match waiting to happen.
"""
from __future__ import annotations

import functools
import time

import numpy as np

from .. import compute, config
from ..contracts import ImagePlane, MatchSet
from . import licence


class MatcherUnavailableError(RuntimeError):
    """Raised when a matcher cannot be constructed (missing weights, no network)."""


def to_vismatch_image(plane: ImagePlane) -> np.ndarray:
    """ImagePlane -> (3, H, W) float32 in [0, 1], the form vismatch documents."""
    arr = np.asarray(plane.array, np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in (1, 3, 4) else arr[0]
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    arr = (arr - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(arr)
    return np.repeat(np.clip(arr, 0.0, 1.0)[None, ...], 3, axis=0)


@functools.lru_cache(maxsize=8)
def _load(model_name: str, device: str, max_keypoints: int):
    """Construct and cache a matcher. Weight download happens on first use."""
    try:
        from vismatch import get_matcher
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MatcherUnavailableError(
            "vismatch is not installed; the learned path is an optional extra "
            "(pip install '.[learned]'). The classical path in matching/classical.py "
            "has no such dependency."
        ) from exc
    try:
        matcher = get_matcher(model_name, device=device, max_num_keypoints=max_keypoints)
    except Exception as exc:
        raise MatcherUnavailableError(
            f"could not construct {model_name!r} on {device!r}: {exc}. If this is a "
            f"first run it may need to download weights, which fails offline."
        ) from exc
    # Our estimator owns the geometry, not the library's internal RANSAC.
    matcher.skip_ransac = True
    return matcher


def available_models() -> list[str]:
    from vismatch import available_models as models
    return sorted(models)


def _drop_shadowed(pts_src, pts_ref, conf, src: ImagePlane, ref: ImagePlane):
    """Remove correspondences whose endpoints sit in a cast shadow."""
    def lit(plane, pts):
        shadow = getattr(plane, "shadow_mask", None)
        if shadow is None or len(pts) == 0:
            return np.ones(len(pts), bool)
        h, w = np.asarray(shadow).shape
        c = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        r = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        return ~np.asarray(shadow, bool)[r, c]

    keep = lit(src, pts_src) & lit(ref, pts_ref)
    return pts_src[keep], pts_ref[keep], (conf[keep] if conf is not None else None), int((~keep).sum())


DENSE_FAMILIES = ("roma", "dkm", "ufm")


def is_dense(model_name: str) -> bool:
    """Dense-warp matchers (RoMa family, DKM, UFM): they sample matches from a certainty map."""
    n = str(model_name).lower()
    return any(f in n for f in DENSE_FAMILIES)


def match(src: ImagePlane, ref: ImagePlane, *, model_name: str | None = None,
          device: str | None = None, max_keypoints: int | None = None,
          ship_mode: bool | None = None, drop_shadowed: bool = True,
          regime: str = "same_modal_normal", stage: str = "direct",
          precision: str = "fp32", tile_px: int | None = None,
          tile_margin_px: int = 64, rotation_search: bool | None = None) -> MatchSet:
    """Match two planes with a vismatch model, returning our MatchSet.

    precision  "fp32" (default, unchanged behaviour) or "fp16": the model runs
               under CUDA autocast. Ignored off CUDA, and recorded on the result.
    tile_px    None (default): one call on the whole pair. An int splits a
               PRE-ALIGNED, equal-size pair into tiles no larger than this; see
               matching/pair_tiling.py for what that assumes.
    rotation_search  MATCH-12: if the 0-deg match is weak, retry with the source rotated in
               30-deg steps (matching/rotation.py). None -> config matching.rotation_search
               (false: our registrations take rotation from the label geometry). For pairs
               WITHOUT a geometric prior; measured in docs/rotation_protocol.md.
    """
    if rotation_search is None:
        rotation_search = bool(config.get("matching.rotation_search", False))
    if rotation_search:
        from .rotation import rotation_search as _search
        return _search(src, ref, lambda a, b: match(
            a, b, model_name=model_name, device=device, max_keypoints=max_keypoints, ship_mode=ship_mode,
            drop_shadowed=drop_shadowed, regime=regime, stage=stage, precision=precision, tile_px=tile_px,
            tile_margin_px=tile_margin_px, rotation_search=False))
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be 'fp32' or 'fp16', not {precision!r}")
    if model_name is None:
        model_name = config.load("regimes").get("default_matcher", "eloftr")

    # Gate first: never construct a restricted model, even to fail later.
    licence.assert_allowed(model_name, ship_mode=ship_mode)

    if tile_px:
        from . import pair_tiling

        started = time.perf_counter()
        ms = pair_tiling.match_tiled(
            src, ref,
            lambda s, r: match(s, r, model_name=model_name, device=device,
                               max_keypoints=max_keypoints, ship_mode=ship_mode,
                               drop_shadowed=drop_shadowed, regime=regime, stage=stage,
                               precision=precision),
            target=int(tile_px), margin=int(tile_margin_px))
        ms.device = config.resolve_device(device)            # type: ignore[attr-defined]
        ms.runtime_s = float(time.perf_counter() - started)  # type: ignore[attr-defined]
        ms.precision = precision                             # type: ignore[attr-defined]
        ms.provenance = compute.provenance(ms.device, stage="match")  # type: ignore[attr-defined]
        return ms

    device = config.resolve_device(device)
    max_keypoints = int(max_keypoints or config.get("matching.max_num_keypoints", 2048))
    dense = is_dense(model_name)
    if dense:        # audit M-06: a dense model SAMPLES exactly this many matches; 2048 handicapped it
        max_keypoints = max(max_keypoints, int(config.get("matching.dense_max_num_keypoints", 5000)))
    matcher = _load(model_name, device, max_keypoints)

    started = time.perf_counter()
    if precision == "fp16" and str(device).startswith("cuda"):
        import torch

        with torch.autocast("cuda", dtype=torch.float16):
            out = matcher(to_vismatch_image(src), to_vismatch_image(ref))
    else:
        out = matcher(to_vismatch_image(src), to_vismatch_image(ref))
    elapsed = time.perf_counter() - started

    src_pts = np.asarray(out.get("matched_kpts0"), np.float64).reshape(-1, 2)
    ref_pts = np.asarray(out.get("matched_kpts1"), np.float64).reshape(-1, 2)
    conf = out.get("matched_confidences")
    conf = None if conf is None else np.asarray(conf, np.float32).ravel()

    n_low_certainty = 0
    if dense and conf is not None and len(conf) == len(src_pts):
        # G-03: a dense model returns N samples even on noise; drop those under RoMa's own sampling
        # threshold so a null test measures structure, not the sampler (docs/roma_benchmark_protocol.md)
        keep = conf >= float(config.get("matching.dense_min_certainty", 0.05))
        n_low_certainty = int((~keep).sum())
        src_pts, ref_pts, conf = src_pts[keep], ref_pts[keep], conf[keep]

    n_shadowed = 0
    if drop_shadowed:
        src_pts, ref_pts, conf, n_shadowed = _drop_shadowed(src_pts, ref_pts, conf, src, ref)

    if conf is None or len(conf) != len(src_pts):
        # Some matchers report no per-match confidence. Say so by using a flat
        # value rather than inventing a spread that would look informative.
        conf = np.ones(len(src_pts), np.float32)

    ms = MatchSet(src_pts=src_pts, ref_pts=ref_pts, confidence=conf,
                  method=model_name, regime=regime, stage=stage)
    ms.device = device                                   # type: ignore[attr-defined]
    ms.runtime_s = float(elapsed)                        # type: ignore[attr-defined]
    ms.precision = precision if str(device).startswith("cuda") else "fp32"  # type: ignore[attr-defined]
    ms.n_keypoints_src = int(len(out.get("all_kpts0", [])))   # type: ignore[attr-defined]
    ms.n_keypoints_ref = int(len(out.get("all_kpts1", [])))   # type: ignore[attr-defined]
    ms.n_dropped_shadowed = int(n_shadowed)              # type: ignore[attr-defined]
    ms.n_dropped_low_certainty = n_low_certainty         # type: ignore[attr-defined]
    ms.provenance = compute.provenance(device, stage="match")  # type: ignore[attr-defined]
    return ms
