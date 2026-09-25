"""The fine stage every registration runs: P2-T16's pipeline runner. Owner: Member B (Part 2).

WHY THIS EXISTS
Until this module each script hand-assembled its own partial chain (match ->
robust.estimate) and three built, tested modules were called by NO pipeline:
the terrain filter (ALIGN-03), uniform control points (ALIGN-04, the PS's
uniform-distribution mandate) and per-point sub-pixel refinement (PREC-01, the
PS's sub-pixel mandate). The audit of 2026-09-23 found that; this closes it.

THE ORDER (SIH26166_Deep_Technical_Research.md section 28, "recommended architecture")

    matches
      -> [terrain filter]        drop matches whose two ends are different ground
      -> robust estimate         ALIGN-01: THE MODEL, fitted on every inlier;
                                 its inlier count and ratio grade the result
      -> [uniform match points]  top-k per grid cell of the inliers (section 30)
      -> [sub-pixel refinement]  each delivered point, NCC on a local patch (section 31)

THE MODEL IS NEVER REFITTED ON THE THINNED POINTS. Measured on real TMC-2 -> TC
(docs/pipeline_stages_protocol.md): refitting on ~384 uniform points discarded
~97% of ~15,000 correct matches and made the perturbation error 3x worse (median
0.081 -> 0.261 px, worst 1.07 px); refining all 15,000 before refitting gained
nothing (0.082) at 5x the runtime. So uniformity and sub-pixel refinement shape
the DELIVERED match points -- the problem statement's "match points ... uniform
distribution ... sub-pixel accuracy" -- and the model keeps its full evidence.

Each bracketed stage is switched by `pipeline.<stage>` in configs/default.yaml
(PLAN P2-T18: "B exposes each stage as an independently toggleable config
flag"), so every stage can be measured on and off. With all three off the
result is exactly the old match -> estimate chain.

WHAT GRADES A RESULT. The confidence tiers were fitted on the FIRST robust
estimate's inlier count and ratio (configs/default.yaml, tiers). Thinning to
uniform control points changes those numbers by design, so they are always
reported from the first estimate, never from the thinned set.

The control gates (evaluate/control_gates.pipeline_from) call `fine_stage` too:
a gate must exercise the path that produced the result it certifies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import time

import numpy as np

from . import config
from .contracts import MatchSet, TransformModel
from .estimate import models, robust

STAGES = ("geometry_filter", "parallax", "dense_refine", "uniformity", "refill", "subpixel", "tps")


def stage_flags(overrides: Optional[dict[str, bool]] = None) -> dict[str, bool]:
    """Which optional stages run: configs/default.yaml `pipeline.*`, then overrides."""
    flags = {s: bool(config.get(f"pipeline.{s}", False)) for s in STAGES}
    for k, v in (overrides or {}).items():
        if k not in STAGES:
            raise ValueError(f"unknown pipeline stage {k!r}; stages are {STAGES}")
        flags[k] = bool(v)
    return flags


@dataclass
class FineResult:
    ok: bool
    model: Optional[TransformModel]           # the first robust estimate's model, on all inliers
    first: robust.EstimateResult              # first robust estimate: grades the result
    matches: MatchSet                         # after the terrain filter
    control_src: np.ndarray                   # the delivered match points, source frame
    control_ref: np.ndarray                   # ... and reference frame (sub-pixel if refined)
    coverage: float                           # share of grid cells holding a delivered point
    rmse_px: Optional[float]                  # the model's residual on the delivered points
    n_matches: int                            # before the terrain filter
    stages: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # ALIGN-02: a thin-plate spline through the delivered points, carried ALONGSIDE the
    # affine model (which drives the gates and scale check; a TPS has no matrix). Measured
    # to cut held-out residual by a median 40% where the affine leaves > 1 px
    # (docs/tps_protocol.md, reports/tps_heldout.json).
    tps: Optional[TransformModel] = None
    # ALIGN-08: affine + DEM parallax, jointly fitted by the parallax stage (apply it with
    # each point's DEM height). Where it exists it is the accurate geometry on relief;
    # `model` stays the first estimate, which the gates certify.
    parallax: Optional[models.ParallaxModel] = None

    @property
    def inlier_count(self) -> int:
        return self.first.inlier_count

    @property
    def inlier_ratio(self) -> float:
        n = len(self.matches.src_pts)
        return self.first.inlier_count / n if n else 0.0


def _rmse(model: Optional[TransformModel], src: np.ndarray, ref: np.ndarray) -> Optional[float]:
    if model is None or model.matrix is None or len(src) == 0:
        return None
    r = np.asarray(models.residuals(model, src, ref), float)
    return float(np.sqrt(np.mean(r ** 2)))


def _dense_refine(model: TransformModel, src_img: np.ndarray, ref_img: np.ndarray,
                  rounds: int = 2) -> tuple[TransformModel, dict]:
    """Refine the model's TRANSLATION on the whole overlap, robustly to illumination.

    The estimator of cascade.register_step_dense step 4, applied at the fine level: warp
    the source with the model, estimate the residual shift with NCC and phase on raw
    intensity AND NCC on every MIND channel (raw intensity fails under a lighting change;
    MIND does not), take the median of the estimates that succeed (at least 3, each
    within 1.5 px), and move the model by it. Two rounds. Chosen by measurement:
    docs/illumination_fix_protocol.md Q6.
    """
    import cv2
    from .preprocess.phase_congruency import mind
    from .refine import subpixel

    M = np.asarray(model.matrix, float).copy()
    h, w = np.asarray(ref_img).shape[:2]
    src = np.asarray(src_img, np.float32)
    ref = np.asarray(ref_img, np.float32)
    nz = lambda a: (a - a.min()) / max(float(a.max() - a.min()), 1e-9)   # noqa: E731
    moves = []
    for _ in range(rounds):
        warped = cv2.warpAffine(src, M[:2], (w, h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpAffine(np.ones(src.shape[:2], np.float32), M[:2], (w, h), flags=cv2.INTER_NEAREST) > 0.5
        ys, xs = np.where(valid)
        if len(ys) < 64 * 64:
            return model, {"applied": False, "reason": "overlap too small", "moves_px": moves}
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        while not valid[y0:y1, x0:x1].all() and y1 - y0 > 64 and x1 - x0 > 64:
            y0, y1, x0, x1 = y0 + 1, y1 - 1, x0 + 1, x1 - 1
        a, b = warped[y0:y1, x0:x1], ref[y0:y1, x0:x1]
        ests = [subpixel.estimate(a, b, m) for m in ("ncc_gaussian_iter", "phase_iter")]
        ma, mb = mind(nz(a)).astype(np.float32), mind(nz(b)).astype(np.float32)
        ests += [subpixel.estimate(ma[..., i], mb[..., i], "ncc_gaussian_iter") for i in range(ma.shape[-1])]
        ok = [e for e in ests if e.ok and np.all(np.isfinite(e.d)) and np.hypot(*e.d) <= 1.5]
        if len(ok) < 3:
            return (TransformModel(kind=model.kind, matrix=M, scale_estimated=model.scale_estimated),
                    {"applied": bool(moves), "reason": f"only {len(ok)} of {len(ests)} estimates succeeded",
                     "moves_px": moves})
        d = np.median(np.array([e.d for e in ok]), axis=0)
        step = np.eye(3); step[:2, 2] = d
        M = step @ M
        moves.append([round(float(d[0]), 4), round(float(d[1]), 4)])
    return (TransformModel(kind=model.kind, matrix=M, scale_estimated=model.scale_estimated),
            {"applied": True, "moves_px": moves, "overlap_px": [int(x1 - x0), int(y1 - y0)]})


def _parallax(ms: MatchSet, inl0: np.ndarray, dem, ground_model, rounds: int = 5, height_at: str = "ref"):
    """Inliers under ref = A.src + h.p, h = DEM height at the source point (ALIGN-08).

    Starts from the robust estimate's inliers, fits A (affine) and p (px per metre) by least
    squares, re-selects inliers from ALL matches at the reprojection threshold, and repeats
    until the set is stable. Matches with no DEM height keep their original status.
    """
    s, r = np.asarray(ms.src_pts, float), np.asarray(ms.ref_pts, float)
    at = r if height_at == "ref" else s         # both are in the one coarse-aligned ground frame
    lat, lon = ground_model.pixel_to_latlon(at[:, 1], at[:, 0])
    h = np.asarray(dem.sample(lat, lon), float)
    known = np.isfinite(h)
    base = inl0 & known
    if base.sum() < 10:
        return inl0, {"applied": False, "reason": "fewer than 10 inliers with a DEM height"}, None
    h0 = float(np.median(h[base]))
    X = np.c_[s, np.ones(len(s)), np.where(known, h - h0, 0.0)]
    thr = float(config.get("estimate.reproj_threshold_px", 3.0))
    inl, n_rounds = base, 0
    for n_rounds in range(1, rounds + 1):
        B = np.linalg.lstsq(X[inl], r[inl], rcond=None)[0]
        new = (known & (np.hypot(*(X @ B - r).T) <= thr)) | (~known & inl0)
        if np.array_equal(new, inl):
            break
        inl = new
    B = np.linalg.lstsq(X[inl & known], r[inl & known], rcond=None)[0]
    res = np.hypot(*(X @ B - r).T)[inl & known]
    return inl, {"applied": True, "p_px_per_m": [round(float(B[3, 0]), 5), round(float(B[3, 1]), 5)],
                 "h0_m": round(h0, 1), "rounds": n_rounds, "height_at": height_at,
                 "inliers_before": int(inl0.sum()), "inliers_after": int(inl.sum()),
                 "rms_px": round(float(np.sqrt(np.mean(res ** 2))), 4),
                 "affine": [[round(float(v), 6) for v in row] for row in B[:3].T]},         models.ParallaxModel(matrix=np.vstack([B[:3].T, [0.0, 0.0, 1.0]]),
                             p_px_per_m=(float(B[3, 0]), float(B[3, 1])), h0_m=h0, height_at=height_at)


def fine_stage(ms: MatchSet, src_img: np.ndarray, ref_img: np.ndarray, *,
               centre: tuple[float, float],
               expected_scale=None,
               ground_model=None, dem=None,
               flags: Optional[dict[str, bool]] = None,
               rematch=None, parallax_dem=None, parallax_height_at: Optional[str] = None) -> FineResult:
    """Run the fine stage on one matched pair.

    `src_img`/`ref_img` are the arrays the matches were found on (the refinement
    reads them). `ground_model` has pixel_to_latlon(rows, cols) valid for BOTH
    sets of points -- after a coarse lock both images share one frame, so one
    model serves -- and `dem` is a DemPatch covering them. Without either, the
    terrain filter is skipped and says so; it never pretends to have run.
    `parallax_dem`, if given, is the height model for the parallax stage only (a finer DEM
    can suit parallax but not the filter's slope thresholds); otherwise `dem` serves both.
    """
    from .estimate import geometry_filter
    from .refine import subpixel, uniformity

    flags = stage_flags(flags)
    stages: dict[str, Any] = {}
    n_matches = int(len(ms.src_pts))
    shape = np.asarray(src_img).shape[:2]

    # 1. terrain filter (ALIGN-03)
    if flags["geometry_filter"]:
        if dem is None or ground_model is None:
            stages["geometry_filter"] = {"applied": False,
                                         "reason": "no DEM" if dem is None else "no ground model"}
        else:
            filtered, rep = geometry_filter.filter_matches(ms, None, None, dem,
                                                           src_model=ground_model, ref_model=ground_model)
            # dataclasses.replace() drops what the matcher attached at run time
            # (device, tile boxes, precision); callers read those, so carry them over.
            for k, v in vars(ms).items():
                if not hasattr(filtered, k):
                    setattr(filtered, k, v)
            ms = filtered
            stages["geometry_filter"] = {"applied": rep.applied, "reason": rep.reason,
                                         "n_in": rep.n_in, "n_kept": rep.n_kept,
                                         "rejected_slope": rep.n_rejected_slope,
                                         "rejected_aspect": rep.n_rejected_aspect,
                                         "unknown_terrain": rep.n_unknown_terrain}
    else:
        stages["geometry_filter"] = {"applied": False, "reason": "off (pipeline.geometry_filter)"}

    # 2. robust estimate (ALIGN-01): grades the result
    first = robust.estimate(ms.src_pts, ms.ref_pts, expected_scale=expected_scale, centre=centre)
    if first.model is None or first.model.matrix is None:
        return FineResult(False, None, first, ms, np.zeros((0, 2)), np.zeros((0, 2)), 0.0, None,
                          n_matches, stages, ["no transform from the robust estimate"])
    inl = first.inlier_mask
    parallax_model = None

    # 2a. terrain parallax (ALIGN-08): an oblique view (TMC-2 fore/aft, 26 deg) moves each
    # point along-track in proportion to its height, which a 2-D affine cannot hold, so on
    # relief the right matches fail the affine threshold. Re-select the inliers under
    # ref = A.src + h.p; the delivered affine model is unchanged (docs/parallax_protocol.md).
    if flags["parallax"]:
        p_dem = parallax_dem if parallax_dem is not None else dem
        if p_dem is None or ground_model is None:
            stages["parallax"] = {"applied": False, "reason": "no DEM" if p_dem is None else "no ground model"}
        else:
            h_at = parallax_height_at or str(config.get("parallax.height_at", "ref"))
            inl, stages["parallax"], parallax_model = _parallax(ms, inl, p_dem, ground_model, height_at=h_at)
    else:
        stages["parallax"] = {"applied": False, "reason": "off (pipeline.parallax)"}
    cs, cr = ms.src_pts[inl], ms.ref_pts[inl]
    conf = np.asarray(ms.confidence, float)[inl] if len(ms.confidence) == len(inl) else np.ones(len(cs))

    # 2b. illumination-robust dense refinement of the MODEL's translation
    model = first.model
    if flags["dense_refine"]:
        model, stages["dense_refine"] = _dense_refine(model, src_img, ref_img)
    else:
        stages["dense_refine"] = {"applied": False, "reason": "off (pipeline.dense_refine)"}

    # 3. uniform control points (ALIGN-04) -- of the inliers, as section 30 prescribes
    grid = int(config.get("uniformity.grid", 8))
    if flags["uniformity"] and len(cs):
        u = uniformity.enforce(cs, conf, shape, grid=grid)
        cs, cr = cs[u.keep_mask], cr[u.keep_mask]
        coverage = float(u.coverage)
        stages["uniformity"] = {"applied": True, "grid": u.grid, "kept": int(u.keep_mask.sum()),
                                "of_inliers": int(len(u.keep_mask)), "empty_cells": len(u.empty_cells),
                                "max_delaunay_gap_px": u.max_delaunay_gap_px}
    else:
        coverage = uniformity.coverage_of(cs, shape, grid=grid) if len(cs) else 0.0
        stages["uniformity"] = {"applied": False, "reason": "off (pipeline.uniformity)"}

    # 3b. refill empty cells (ALIGN-05): the same matcher, on the empty cell only; kept
    # only where the model agrees (<= 2 px). The model is never refitted.
    n_primary = len(cs)
    if flags["refill"] and flags["uniformity"] and rematch is not None and len(cs) and u.empty_cells:
        k = int(config.get("uniformity.top_k_per_cell", 6))
        H, W = shape
        pad, add_s, add_r, add_c, filled = 24, [], [], [], 0
        for (y0, y1, x0, x1) in uniformity.empty_cell_regions(u, shape):
            ya, yb, xa, xb = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
            if yb - ya < 32 or xb - xa < 32:
                continue
            try:
                m2 = rematch(np.asarray(src_img)[ya:yb, xa:xb], np.asarray(ref_img)[ya:yb, xa:xb])
            except Exception:                                   # a failed local pass adds nothing
                continue
            if m2 is None or len(m2.src_pts) == 0:
                continue
            s2 = np.asarray(m2.src_pts, float) + [xa, ya]
            r2 = np.asarray(m2.ref_pts, float) + [xa, ya]
            c2 = np.asarray(m2.confidence, float) if len(m2.confidence) == len(s2) else np.ones(len(s2))
            inside = (s2[:, 0] >= x0) & (s2[:, 0] < x1) & (s2[:, 1] >= y0) & (s2[:, 1] < y1)
            res = np.hypot(*(models.apply(model, s2) - r2).T)
            ok = inside & (res <= 2.0)
            if not ok.any():
                continue
            order = np.argsort(-c2[ok])[:k]
            add_s.append(s2[ok][order]); add_r.append(r2[ok][order]); add_c.append(c2[ok][order]); filled += 1
        empty_before = len(u.empty_cells)
        if add_s:
            new_s = np.vstack(add_s)
            u = uniformity.merge_refill(u.keep_mask, new_s, u, shape)
            cs, cr = np.vstack([cs, new_s]), np.vstack([cr, np.vstack(add_r)])
            coverage = float(u.coverage)
        stages["refill"] = {"applied": True, "empty_cells_before": empty_before, "cells_filled": filled,
                            "refilled": int(len(cs) - n_primary), "empty_cells_after": len(u.empty_cells)}
    else:
        why = ("off (pipeline.refill)" if not flags["refill"] else "no rematch callable" if rematch is None
               else "no empty cells")
        stages["refill"] = {"applied": False, "reason": why}

    # 4. per-point sub-pixel refinement (PREC-01) of the DELIVERED points
    if flags["subpixel"] and len(cs):
        before = cr.copy()
        cr, moved = subpixel.refine_points(src_img, ref_img, cs, cr)
        shift = np.hypot(*(cr - before).T)
        rms_before, rms_after = _rmse(model, cs, before), _rmse(model, cs, cr)
        stages["subpixel"] = {"applied": True, "points": int(len(cs)), "moved": int(moved.sum()),
                              "median_move_px": round(float(np.median(shift[moved])), 4) if moved.any() else 0.0,
                              "residual_to_model_px": {"unrefined": rms_before, "refined": rms_after},
                              "method": str(config.get("subpixel.method", "ncc_gaussian_iter")),
                              "window_px": 2 * int(config.get("subpixel.refine_half_px", 16)) + 1}
    else:
        stages["subpixel"] = {"applied": False, "reason": "off (pipeline.subpixel)"}

    if stages["refill"].get("refilled"):
        stages["refill"]["residual_to_model_px"] = {"primary": _rmse(model, cs[:n_primary], cr[:n_primary]),
                                                    "refilled": _rmse(model, cs[n_primary:], cr[n_primary:])}

    # 5. TPS through the delivered points (ALIGN-02)
    tps = None
    if flags["tps"] and len(cs) >= 15:
        tps = models.fit_tps_cv(cs, cr)
        stages["tps"] = {"applied": True, "control_points": int(len(cs)),
                         "smoothing": tps.tps_params["smoothing"], "cv_rms_px": tps.tps_params.get("cv_rms_px")}
    else:
        stages["tps"] = {"applied": False, "reason": "off (pipeline.tps)" if not flags["tps"] else "too few points"}
    return FineResult(True, model, first, ms, cs, cr, coverage, _rmse(model, cs, cr),
                      n_matches, stages, [], tps, parallax_model)


@dataclass
class RegistrationBundle:
    """Everything Part 3 needs from one registration, WITHOUT changing the frozen contract.

    `result` is the RegistrationResult handoff (tier, gates, failure modes, metrics, affine model).
    Its `matches` / `inlier_mask` are the terrain-filtered correspondences and the robust estimate's
    inliers -- the evidence the tier was graded on. What to EXPORT as the match points is `delivered`:
    the final control points (uniform per ALIGN-04, sub-pixel refined per PREC-01), the same points
    `result.metrics.rmse_px` and `spatial_coverage` describe. To warp the registered product, use the
    best geometry available: `parallax` (TMC-2 on relief) if set, else `tps` if set, else `result.model`.
    Metric sources: inlier_count / inlier_ratio from the evidence (`result.matches` + `inlier_mask`);
    rmse_px / spatial_coverage from `delivered`; runtime_s over the whole call (matching, fine stage,
    all control gates).
    """
    result: Any                               # contracts.RegistrationResult
    delivered: MatchSet                       # final control points (export these); its `confidence` is all
                                              # ones -- the per-point score is not carried through selection
    tps: Optional[TransformModel] = None      # ALIGN-02, through the delivered points
    parallax: Optional[Any] = None            # ALIGN-08 models.ParallaxModel (needs DEM heights)
    stages: dict = field(default_factory=dict)
    src: Any = None                           # the ImagePlanes registered (with meta / geo for georeferencing)
    ref: Any = None
    # When the source was registered RESAMPLED onto the reference grid (the validated product
    # workflows), `src` is that frame and this 3x3 maps its px (x, y) to source PRODUCT px.
    src_to_product: Optional[np.ndarray] = None
    # Likewise for `ref` (a window, possibly block-averaged): its px (x, y) -> reference PRODUCT px.
    ref_to_product: Optional[np.ndarray] = None
    # Terrain heights for warping with `parallax`: heights_at((N, 2) ref-frame px) -> metres.
    heights_at: Optional[Any] = None


def _metric_source(*planes) -> str:
    """Rule H5: a number measured on synthetic imagery must never be labelled a measurement."""
    synthetic = any(getattr(getattr(p, "meta", None), "mission", None) == "SYNTH" for p in planes)
    return "synthetic" if synthetic else "measured"


def register(src, ref, *, matcher: str = "sift", device: Optional[str] = None, expected_scale=None,
             ground_model=None, dem=None, flags: Optional[dict[str, bool]] = None,
             match_kwargs: Optional[dict] = None, provenance: Optional[dict] = None):
    """RegistrationResult only (the frozen handoff); see `register_bundle` for the delivered points."""
    return register_bundle(src, ref, matcher=matcher, device=device, expected_scale=expected_scale,
                           ground_model=ground_model, dem=dem, flags=flags, match_kwargs=match_kwargs,
                           provenance=provenance).result


def register_bundle(src, ref, *, matcher: str = "sift", device: Optional[str] = None, expected_scale=None,
                    ground_model=None, dem=None, flags: Optional[dict[str, bool]] = None,
                    match_kwargs: Optional[dict] = None, provenance: Optional[dict] = None) -> RegistrationBundle:
    """One registration end to end -> RegistrationResult, THE Part 2 -> Part 3 handoff.

    Match, fine stage, all five control gates, quality tier. A bad or impossible pair is a RESULT,
    not an error: it comes back REJECTED with its failure_modes filled in (CHECK-07), and nothing
    raises -- a matcher that crashes is recorded in `notes` and treated as finding no matches.
    `src` / `ref` are ImagePlanes; `expected_scale` (estimate.scale.ExpectedScale) turns on the
    scale check (CHECK-05).
    """
    from .contracts import Metrics, RegistrationResult
    from .estimate import scale as scale_mod
    from .evaluate import control_gates, quality

    t_start = time.perf_counter()                            # metrics.runtime_s: the whole call
    match_kwargs = dict(match_kwargs or {})
    s_img, r_img = np.asarray(src.array, np.float32), np.asarray(ref.array, np.float32)
    notes: list[str] = []
    try:
        if matcher in ("sift", "akaze", "orb", "brisk"):
            from .matching import classical
            ms = classical.match(src, ref, detector=matcher)
        elif matcher == "rift2":
            from .matching import rift
            ms = rift.match(src, ref)
        else:
            from .matching import adapter
            ms = adapter.match(src, ref, model_name=matcher, device=device, **match_kwargs)
    except Exception as exc:                                  # a crash is a finding, never a success
        notes.append(f"matcher {matcher!r} failed: {type(exc).__name__}: {exc}"[:300])
        ms = MatchSet(src_pts=np.zeros((0, 2)), ref_pts=np.zeros((0, 2)), confidence=np.zeros(0, np.float32),
                      method=matcher, regime="same_modal_normal", stage="direct")
    h, w = s_img.shape[:2]
    fr = fine_stage(ms, s_img, r_img, centre=(w / 2.0, h / 2.0), expected_scale=expected_scale,
                    ground_model=ground_model, dem=dem, flags=flags)
    gates = control_gates.run_all(control_gates.pipeline_from(matcher, device=device, gsd_m=float(src.gsd_m or 1.0),
                                                              stages=flags, **match_kwargs),
                                  s_img, r_img, src, ref)
    scale_ok, scale_status = True, "not checked (no expected scale given)"
    if fr.ok and isinstance(expected_scale, scale_mod.ExpectedScale):
        v = scale_mod.check(np.asarray(fr.model.matrix, float), expected_scale, centre=(w / 2.0, h / 2.0))
        scale_ok, scale_status = v.ok, v.status
    q = quality.assess(inlier_count=fr.inlier_count if fr.ok else 0,
                       inlier_ratio=fr.inlier_ratio if fr.ok else 0.0,
                       spatial_coverage=fr.coverage if fr.ok else 0.0,
                       model=fr.model if fr.ok else None, scale_ok=scale_ok, scale_status=scale_status,
                       gates=gates.gates, require_gates=True)
    n = int(len(ms.src_pts))
    result = RegistrationResult(
        matches=fr.matches if fr.matches is not None else ms,
        inlier_mask=fr.first.inlier_mask if fr.first is not None else np.zeros(n, bool),
        model=fr.model if fr.ok else None,
        metrics=Metrics(rmse_px=fr.rmse_px if fr.ok else None, inlier_count=fr.inlier_count if fr.ok else 0,
                        inlier_ratio=fr.inlier_ratio if fr.ok else 0.0,
                        spatial_coverage=fr.coverage if fr.ok else 0.0,
                        runtime_s=round(time.perf_counter() - t_start, 3), source=_metric_source(src, ref)),
        confidence_tier=q.tier, gates=gates.gates, failure_modes=list(q.failure_modes),
        notes=notes + list(fr.notes) + list(q.notes),
        provenance={"matcher": matcher, "limiting_signal": q.limiting_signal, "scale_status": scale_status,
                    **(provenance or {})})
    control_gates.require_gates(result)
    cs = np.asarray(fr.control_src if fr.ok else np.zeros((0, 2)), float).reshape(-1, 2)
    cr = np.asarray(fr.control_ref if fr.ok else np.zeros((0, 2)), float).reshape(-1, 2)
    delivered = MatchSet(src_pts=cs, ref_pts=cr, confidence=np.ones(len(cs), np.float32),
                         method=ms.method, regime=ms.regime, stage="delivered")
    return RegistrationBundle(result=result, delivered=delivered, tps=fr.tps if fr.ok else None,
                              parallax=fr.parallax if fr.ok else None, stages=fr.stages, src=src, ref=ref)
