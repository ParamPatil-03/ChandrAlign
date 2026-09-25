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

STAGES = ("geometry_filter", "parallax", "dense_refine", "uniformity", "refill", "subpixel", "tps",
          "model_selection")


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
    rmse_px: Optional[float]                  # FIT RESIDUAL: the affine's residual on the delivered points
                                              # (not an accuracy figure; see `accuracy` -- audit C-03)
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
    # C-03 / I-01: which of model / parallax / tps the product should warp with, chosen by
    # check-point error (estimate/selection.py), and that geometry's check-point accuracy.
    geometry: str = "affine"
    accuracy: dict = field(default_factory=dict)
    geometry_model: Any = None                # the chosen model itself, fitted on the refined fit set (G-02)

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
               rematch=None, parallax_dem=None, parallax_height_at: Optional[str] = None,
               ref_ground_model=None) -> FineResult:
    """Run the fine stage on one matched pair.

    `src_img`/`ref_img` are the arrays the matches were found on (the refinement
    reads them). `ground_model` has pixel_to_latlon(rows, cols) valid for BOTH
    sets of points -- after a coarse lock both images share one frame, so one
    model serves -- and `dem` is a DemPatch covering them. Without either, the
    terrain filter is skipped and says so; it never pretends to have run.
    `parallax_dem`, if given, is the height model for the parallax stage only (a finer DEM
    can suit parallax but not the filter's slope thresholds); otherwise `dem` serves both.
    `ref_ground_model` (audit I-10): pixel_to_latlon for the REFERENCE points when the two images do
    NOT share one frame. Without it, `ground_model` serves both, which is only valid after a coarse
    lock; with it, reference points are never mapped through the source's model.
    """
    from .estimate import geometry_filter
    from .refine import subpixel, uniformity

    flags = stage_flags(flags)
    ref_gm = ref_ground_model if ref_ground_model is not None else ground_model
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
                                                           src_model=ground_model, ref_model=ref_gm)
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
    # I-02 (audit 2026-09-26): the estimator REFUSED (too few inliers, or a scale it could not
    # accept). Its refusal stands: tiers.low accepts from 8 inliers, estimate.min_inliers
    # requires 12, and a 10-inlier fit used to come back LOW (accepted).
    if not first.ok:
        return FineResult(False, None, first, ms, np.zeros((0, 2)), np.zeros((0, 2)), 0.0, None,
                          n_matches, stages, ["the robust estimate refused the fit: " + "; ".join(first.notes[-1:])])
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
            inl, stages["parallax"], parallax_model = _parallax(ms, inl, p_dem, ref_gm if h_at == "ref" else ground_model,
                                                                height_at=h_at)
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
        # C-02: the model's local Jacobian brings each source patch into the reference geometry
        cr, moved = subpixel.refine_points(src_img, ref_img, cs, cr, model=model)
        shift = np.hypot(*(cr - before).T)
        rms_before, rms_after = _rmse(model, cs, before), _rmse(model, cs, cr)
        stages["subpixel"] = {"applied": True, "points": int(len(cs)), "moved": int(moved.sum()),
                              "median_move_px": round(float(np.median(shift[moved])), 4) if moved.any() else 0.0,
                              "residual_to_model_px": {"unrefined": rms_before, "refined": rms_after},
                              "method": str(config.get("subpixel.method", "lsm")), "warped": True,
                              "window_px": 2 * int(config.get("subpixel.refine_half_px", 20)) + 1}
    else:
        stages["subpixel"] = {"applied": False, "reason": "off (pipeline.subpixel)"}

    if stages["refill"].get("refilled"):
        stages["refill"]["residual_to_model_px"] = {"primary": _rmse(model, cs[:n_primary], cr[:n_primary]),
                                                    "refilled": _rmse(model, cs[n_primary:], cr[n_primary:])}

    # 5. the delivered geometry and its check-point accuracy (audit C-03, I-01; G-02)
    tps, geometry, accuracy, geometry_model = None, "affine", {}, None
    if flags["model_selection"]:
        tps, geometry, accuracy, geometry_model = _select_geometry(ms, inl, model, src_img, ref_img, shape, flags,
                                                   parallax_model,
                                                   (ref_gm if parallax_model is not None and parallax_model.height_at == "ref"
                                                    else ground_model),
                                                   parallax_dem if parallax_dem is not None else dem, stages)
    # 5'. (model_selection off) TPS through the delivered points (ALIGN-02, the pre-2026-09-26 path)
    elif flags["tps"] and len(cs) >= 15:
        tps = models.fit_tps_cv(cs, cr)
        stages["tps"] = {"applied": True, "control_points": int(len(cs)),
                         "smoothing": tps.tps_params["smoothing"], "cv_rms_px": tps.tps_params.get("cv_rms_px")}
    else:
        stages["tps"] = {"applied": False, "reason": "off (pipeline.tps)" if not flags["tps"] else "too few points"}
    if not flags["model_selection"]:
        stages["model_selection"] = {"applied": False, "reason": "off (pipeline.model_selection)"}
        geometry = "parallax" if parallax_model is not None else "tps" if tps is not None else "affine"
    fit_residual = _rmse(model, cs, cr)
    accuracy = {"fit_residual_px": fit_residual, **accuracy}
    return FineResult(True, model, first, ms, cs, cr, coverage, fit_residual,
                      n_matches, stages, [], tps, parallax_model, geometry, accuracy, geometry_model)


def _select_geometry(ms, inl, model, src_img, ref_img, shape, flags, parallax_model, ground_model, p_dem, stages):
    """Stage 5: refine an even sample of ALL inliers, score affine / parallax / TPS on points each did
    not use, and choose the delivered geometry (estimate/selection.py). The 384 uniform points stay the
    exported match points; the geometry is fitted on the larger, refined, evenly spread fit set."""
    from .estimate import selection
    from .refine import subpixel

    s_in, r_in = np.asarray(ms.src_pts, float)[inl], np.asarray(ms.ref_pts, float)[inl]
    grid = int(config.get("uniformity.grid", 8))
    pick = selection.stratified_sample(s_in, shape, int(config.get("geometry.max_fit_points", 1000)), grid)
    fs, fr_ = s_in[pick], r_in[pick]
    refined = False
    if flags["subpixel"] and len(fs):
        fr_, _ = subpixel.refine_points(src_img, ref_img, fs, fr_, model=model)
        refined = True
    heights_at = None
    if parallax_model is not None and p_dem is not None and ground_model is not None:
        def heights_at(p):
            p = np.asarray(p, float).reshape(-1, 2)
            lat, lon = ground_model.pixel_to_latlon(p[:, 1], p[:, 0])
            return np.asarray(p_dem.sample(lat, lon), float)
    sel = selection.select(fs, fr_, shape, parallax=parallax_model, heights_at=heights_at,
                           with_tps=flags["tps"])
    tps = sel.tps
    stages["model_selection"] = {"applied": True, "refined_fit_points": refined, **sel.record()}
    stages["tps"] = ({"applied": True, "fit_points": int(len(fs)), "smoothing": tps.tps_params["smoothing"],
                      "cv_rms_px": tps.tps_params.get("cv_rms_px"), "robust": "huber_irls"} if tps is not None
                     else {"applied": False, "reason": "off (pipeline.tps)" if not flags["tps"] else "too few points"})
    point_set = (f"{len(fs)} first-estimate inliers, grid-stratified"
                 + (", sub-pixel refined" if refined else ""))
    accuracy = {"checkpoint_rmse_px_ref": sel.checkpoint_rmse_px, "model": sel.name, "n_check": int(len(fs)),
                "point_set": point_set, "scored_by": sel.candidates[sel.name]["scored_by"],
                "p50_px_ref": sel.candidates[sel.name].get("p50_px"),
                "p95_px_ref": sel.candidates[sel.name].get("p95_px"),
                # a LOWER bound (the fit's own variance, no point noise); the check-point RMSE is an upper bound
                "geometry_error_lower_px_ref": sel.candidates[sel.name].get("split_half_px")}
    return tps, sel.name, accuracy, sel.model


def delivered_geometry(fr: FineResult):
    """(name, model) the product should warp with: the fine stage's check-point choice (C-03, I-01)."""
    if fr.geometry_model is not None:
        return fr.geometry, fr.geometry_model
    if fr.geometry == "parallax" and fr.parallax is not None:
        return "parallax", fr.parallax
    if fr.geometry == "tps" and fr.tps is not None:
        return "tps", fr.tps
    return (fr.model.kind if fr.model is not None else "affine"), fr.model


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
    # C-03 / I-01: the geometry the fine stage CHOSE by check-point error ("affine" | "tps" | "parallax").
    # product.warp.best_geometry honours it; None (older bundles) falls back to the old priority.
    geometry: Optional[str] = None
    geometry_model: Any = None                # ... and that model, fitted on the refined fit set (G-02)


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
    from .evaluate import control_gates, quality, selftest

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
    # C-04: an independent method from a different family must agree (docs/crosscheck_protocol.md)
    crosscheck = {"applied": False, "reason": "off (gates.crosscheck)"}
    if not fr.ok:
        crosscheck = {"applied": False, "reason": "no transform to check"}
    elif bool(config.get("gates.crosscheck", True)):
        xgate, crosscheck = control_gates.independent_crosscheck(fr.model.matrix, src, ref, matcher,
                                                                 centre=(w / 2.0, h / 2.0),
                                                                 predict=_geometry_predictor(fr, ground_model, dem))
        if xgate is not None:
            gates.results.append(xgate)
    scale_ok, scale_status = True, "not checked (no expected scale given)"
    if not fr.ok and fr.first is not None and fr.first.scale_status in ("inconsistent", "degenerate"):
        scale_ok, scale_status = False, fr.first.scale_status     # refused on scale: failure mode 13, not 12
    if fr.ok and isinstance(expected_scale, scale_mod.ExpectedScale):
        v = scale_mod.check(np.asarray(fr.model.matrix, float), expected_scale, centre=(w / 2.0, h / 2.0))
        scale_ok, scale_status = v.ok, v.status
    probe_acc = _probe_accuracy(fr, s_img, r_img, src, ground_model, dem, (w / 2.0, h / 2.0)) if fr.ok else None
    q = quality.assess(inlier_count=fr.inlier_count if fr.ok else 0,
                       inlier_ratio=fr.inlier_ratio if fr.ok else 0.0,
                       spatial_coverage=fr.coverage if fr.ok else 0.0,
                       model=fr.model if fr.ok else None, scale_ok=scale_ok, scale_status=scale_status,
                       gates=gates.gates, require_gates=True, accuracy=probe_acc)
    if crosscheck.get("verdict") in ("inconclusive", "no checker"):
        cap = str(config.get("gates.crosscheck_inconclusive_cap", "MEDIUM"))
        if quality.TIER_ORDER.index(q.tier) < quality.TIER_ORDER.index(cap):
            q.notes.append(f"independent cross-check {crosscheck['verdict']}: tier capped {q.tier} -> {cap} "
                           f"(HIGH needs an independent method to agree)")
            q.tier, q.limiting_signal = cap, "independent_crosscheck"
    n = int(len(ms.src_pts))
    accuracy = _accuracy_record(fr, ref, (w / 2.0, h / 2.0)) if fr.ok else {}
    if probe_acc is not None:
        accuracy["probes"] = probe_acc
    result = RegistrationResult(
        matches=fr.matches if fr.matches is not None else ms,
        inlier_mask=fr.first.inlier_mask if fr.first is not None else np.zeros(n, bool),
        model=fr.model if fr.ok else None,
        metrics=Metrics(rmse_px=accuracy.get("rmse_px_ref"), rmse_m=accuracy.get("rmse_m"),
                        inlier_count=fr.inlier_count if fr.ok else 0,
                        inlier_ratio=fr.inlier_ratio if fr.ok else 0.0,
                        spatial_coverage=fr.coverage if fr.ok else 0.0,
                        max_delaunay_gap_px=(fr.stages.get("uniformity") or {}).get("max_delaunay_gap_px") if fr.ok else None,
                        subpixel_recovery_err_px=selftest.subpixel_recovery()["value_px"],
                        runtime_s=round(time.perf_counter() - t_start, 3), source="measured"),
        confidence_tier=q.tier, gates=gates.gates, failure_modes=list(q.failure_modes),
        notes=notes + list(fr.notes) + list(q.notes),
        provenance={"matcher": matcher, "limiting_signal": q.limiting_signal, "scale_status": scale_status,
                    "accuracy": accuracy, "crosscheck": crosscheck, "subpixel_selftest": selftest.subpixel_recovery(), **(provenance or {})})
    control_gates.require_gates(result)
    cs = np.asarray(fr.control_src if fr.ok else np.zeros((0, 2)), float).reshape(-1, 2)
    cr = np.asarray(fr.control_ref if fr.ok else np.zeros((0, 2)), float).reshape(-1, 2)
    delivered = MatchSet(src_pts=cs, ref_pts=cr, confidence=np.ones(len(cs), np.float32),
                         method=ms.method, regime=ms.regime, stage="delivered")
    return RegistrationBundle(result=result, delivered=delivered, tps=fr.tps if fr.ok else None,
                              parallax=fr.parallax if fr.ok else None, stages=fr.stages, src=src, ref=ref,
                              geometry=fr.geometry if fr.ok else None,
                              geometry_model=fr.geometry_model if fr.ok else None)


def _geometry_predictor(fr: FineResult, ground_model, dem):
    """src px -> ref px through the DELIVERED geometry, or None when it cannot be evaluated here
    (a parallax model with no heights)."""
    name, geo = delivered_geometry(fr)
    if name.startswith("parallax"):                           # parallax, parallax_tps: need DEM heights
        if ground_model is None or dem is None:
            return None
        return lambda p: geo.predict(p, lambda q: np.asarray(dem.sample(*ground_model.pixel_to_latlon(q[:, 1], q[:, 0])),
                                                              float))
    return lambda p: models.apply(geo, p)


def _probe_accuracy(fr: FineResult, s_img, r_img, src, ground_model, dem, centre) -> Optional[dict]:
    """I-08: matcher-free probes against the DELIVERED geometry, in source px (evaluate/probes.py)."""
    from .evaluate import probes
    predict = _geometry_predictor(fr, ground_model, dem)
    if predict is None:
        return {"n": 0, "reason": "parallax geometry but no heights to evaluate it"}
    k = models.estimated_scale(np.asarray(fr.model.matrix, float), at=centre)
    ok = np.asarray(getattr(src, "valid_mask", None) if getattr(src, "valid_mask", None) is not None
                    else np.ones(np.asarray(s_img).shape, bool), bool)
    try:
        return probes.geometry_error(s_img, r_img, ok, predict, k)
    except Exception as exc:                                  # a failed measurement is unmeasured, not a pass
        return {"n": 0, "reason": f"{type(exc).__name__}: {exc}"[:200]}


def _accuracy_record(fr: FineResult, ref, centre) -> dict:
    """What `metrics.rmse_px` means now (audit C-03): the CHECK-POINT RMS error of the geometry the
    product warps with, in reference px, tagged with its model and point set. The old number -- the
    affine's residual on its own pre-screened inliers -- is kept as `fit_residual_px` only.

    rmse_px_src divides by the model's local scale (ref px per src px) at the centre; it is in the
    source frame the fine stage saw (a caller that resampled the source must compose back).
    rmse_m multiplies by the reference pixel size, when that is known."""
    a = dict(fr.accuracy or {})
    ref_px = a.get("checkpoint_rmse_px_ref")
    out = {"rmse_px_ref": ref_px, "fit_residual_px": a.get("fit_residual_px"), "model": a.get("model", fr.geometry),
           "point_set": a.get("point_set"), "n_check": a.get("n_check"), "scored_by": a.get("scored_by"),
           "p50_px_ref": a.get("p50_px_ref"), "p95_px_ref": a.get("p95_px_ref"),
           "geometry_error_lower_px_ref": a.get("geometry_error_lower_px_ref"),
           "rmse_px_src": None, "rmse_m": None}
    if ref_px is not None and fr.model is not None and fr.model.matrix is not None:
        k = models.estimated_scale(np.asarray(fr.model.matrix, float), at=centre)
        if np.isfinite(k) and k > 0:
            out["rmse_px_src"] = round(float(ref_px) / k, 4)
        gsd = getattr(ref, "gsd_m", None)
        if gsd:
            out["rmse_m"] = round(float(ref_px) * float(gsd), 4)
    return out
