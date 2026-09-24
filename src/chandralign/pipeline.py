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

import numpy as np

from . import config
from .contracts import MatchSet, TransformModel
from .estimate import models, robust

STAGES = ("geometry_filter", "dense_refine", "uniformity", "subpixel")


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


def fine_stage(ms: MatchSet, src_img: np.ndarray, ref_img: np.ndarray, *,
               centre: tuple[float, float],
               expected_scale=None,
               ground_model=None, dem=None,
               flags: Optional[dict[str, bool]] = None) -> FineResult:
    """Run the fine stage on one matched pair.

    `src_img`/`ref_img` are the arrays the matches were found on (the refinement
    reads them). `ground_model` has pixel_to_latlon(rows, cols) valid for BOTH
    sets of points -- after a coarse lock both images share one frame, so one
    model serves -- and `dem` is a DemPatch covering them. Without either, the
    terrain filter is skipped and says so; it never pretends to have run.
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

    return FineResult(True, model, first, ms, cs, cr, coverage, _rmse(model, cs, cr),
                      n_matches, stages, [])
