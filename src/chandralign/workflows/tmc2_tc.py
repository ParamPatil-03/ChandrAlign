"""The adopted TMC-2 -> SELENE TC registration, one window at a time (the headline pairing).

Moved VERBATIM from scripts/register_tmc2_tc.py (audit 2026-09-26 C-01) so the product (CLI, API)
runs the same code the committed evidence came from; the script is now a thin wrapper. Its module
globals became `Options`. See the script's docstring for the method (prior from TMC-2 SYSTEM
corners -> MIND coarse lock -> fine stage on the coarse-aligned pair -> scale check, gates, tier).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .. import config, synth
from ..contracts import Metrics, RegistrationResult, TransformModel
from ..estimate import robust, scale
from ..estimate.models import ParallaxModel
from ..estimate.scale import PixelScale
from ..evaluate import control_gates, quality
from ..evaluate.source_px import jacobian_from_transform, to_source_px
from ..io import pds_raster
from ..matching import adapter
from ..matching.similarity import alignment_check
from ..pipeline import fine_stage, stage_flags
from ..preprocess.phase_congruency import mind
from ..preprocess.resample import warp_affine

TILES = {
    "N09": "TCO_MAP_02_N09E021N06E024SC",
    "N03": "TCO_MAP_02_N03E021N00E024SC",
    "N00": "TCO_MAP_02_N00E021S03E024SC",
}
M_PER_DEG = math.pi * 1_737_400.0 / 180.0


def norm(a: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    v = a[valid] if valid is not None and valid.any() else a
    lo, hi = np.percentile(v, [1, 99])
    out = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)
    if valid is not None:
        out[~valid] = 0.0
    return out


def block(a: np.ndarray, c: int) -> np.ndarray:
    h, w = (a.shape[0] // c) * c, (a.shape[1] // c) * c
    return a[:h, :w].reshape(h // c, c, w // c, c).mean(axis=(1, 3))


def affine_from(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Exact least-squares affine (3x3), no robust fitting: these are model points."""
    X = np.c_[src_pts, np.ones(len(src_pts))]
    sol, *_ = np.linalg.lstsq(X, dst_pts, rcond=None)
    A = np.eye(3)
    A[:2, :] = sol.T
    return A


def T(dx: float, dy: float) -> np.ndarray:
    m = np.eye(3)
    m[0, 2], m[1, 2] = dx, dy
    return m


def tc_pixel_scale(tc_meta, lat_deg: float) -> PixelScale:
    """TC's pixel on the ground at this latitude. Exact by construction: it is the
    map grid, not a measurement of anything, so it counts as verified."""
    y = float(tc_meta.gsd_m)
    x = y * math.cos(math.radians(lat_deg))
    return PixelScale(tc_meta.product_id, x, y, (x, x), (y, y), ("map_projection",), True,
                      ("map-projected: pixel size is the SIMPLE CYLINDRICAL grid at this latitude",))


def tmc_pixel_from(J: np.ndarray, g_tc: tuple[float, float]) -> tuple[float, float]:
    """(across, along) TMC-2 pixel in metres from J: TMC px -> TC px.

    J = diag(1/g_tc) R diag(h), so diag(g_tc) J = R diag(h): its COLUMN norms are h.
    """
    M = np.diag(g_tc) @ J
    return float(np.linalg.norm(M[:, 0])), float(np.linalg.norm(M[:, 1]))


class _OffsetModel:
    """TC's map model for pixels of a region that starts at TC px (x0, y0)."""

    def __init__(self, tcm, origin_xy):
        self.tcm, self.x0, self.y0 = tcm, float(origin_xy[0]), float(origin_xy[1])

    def pixel_to_latlon(self, rows, cols):
        return self.tcm.pixel_to_latlon(np.asarray(rows, float) + self.y0, np.asarray(cols, float) + self.x0)


def _dem_for(tcm, origin_xy, w: int, h: int, which: str | None = None, *, opts: Options):
    """Heights (`which`, default --dem) covering a TC region (with a margin), or None if no
    tile of that DEM is held there."""
    from chandralign.io.dem import DemError, dem_patch, find_tiles
    tiles = find_tiles(opts.root / DEM_DIRS[which or opts.dem])
    if not tiles:
        return None
    x0, y0 = origin_xy
    lat, lon = tcm.pixel_to_latlon(np.array([y0, y0 + h, y0, y0 + h], float),
                                   np.array([x0, x0, x0 + w, x0 + w], float))
    m = 0.02
    try:
        return dem_patch(tiles, (float(np.min(lat)) - m, float(np.max(lat)) + m,
                                 float(np.min(lon)) - m, float(np.max(lon)) + m))
    except DemError:
        if which is None:
            raise
        return None                    # an optional DEM (--parallax-dem) not held here


def _dem_grid(dem, tcm, o_f, w: int, h: int, step: int = 16) -> dict:
    """Height (m) every `step` px of a fine window, for the dump (empty if no DEM)."""
    if dem is None:
        return {}
    ys, xs = np.mgrid[0:h:step, 0:w:step].astype(float)
    lat, lon = _OffsetModel(tcm, o_f).pixel_to_latlon(ys.ravel(), xs.ravel())
    return {"dem_step": step, "dem_m": np.asarray(dem.sample(lat, lon), float).reshape(ys.shape)}


DEM_DIRS = {"sldem2015": "data/raw/dem/sldem2015", "tc_dtm": "data/raw/selene/tc_dtm"}
# ALIGN-08: parallax is ON for this pairing (TMC-2 fore/aft views are 26 deg oblique), adopted by
# docs/parallax_protocol.md amendment 3. Only here: the global default stays off.
PAIRING_STAGES = {"parallax": True}


@dataclass
class Options:
    """What scripts/register_tmc2_tc.py used to hold in module globals, set from its flags."""
    dem: str = "sldem2015"          # --dem: height model for the terrain filter (and parallax, by default)
    parallax_dem: Optional[str] = "tc_dtm"   # --parallax-dem: the adopted default (amendment 3)
    parallax_height_at: Optional[str] = None  # None -> configs/default.yaml parallax.height_at
    dump_dir: Optional[str] = None  # --dump-points: save each window's control points (tps_protocol)
    root: Path = field(default_factory=lambda: config.ROOT)   # where data/raw lives


def run_window(tmc, sysm, refm, tc, tcm, row_c: int, *, win: int, coarse: int,
               margin_km: float, matcher: str, device: str,
               prior_offset_m: tuple[float, float] = (0.0, 0.0),
               match_kwargs: dict | None = None, gate_reuse_base: bool = False,
               stages: dict | None = None, col_offset: int = 0, opts: Optional[Options] = None,
               capture: Optional[dict] = None) -> dict:
    """One TMC-2 window registered to one SELENE TC tile (the adopted TMC-2 -> TC path).

    Returns the evidence record scripts/register_tmc2_tc.py writes per window. Pass a dict as
    `capture` to also receive the registration itself (frames, fine result, gates, tier) --
    what `bundle_from_capture` turns into a RegistrationBundle for the product exporters.
    """
    opts = opts or Options()
    match_kwargs = dict(match_kwargs or {})
    stages = stage_flags(stages)
    col_c = tmc.array_shape[1] // 2 + int(col_offset)
    r0, c0 = row_c - win // 2, col_c - win // 2
    out: dict = {"tmc_row": row_c, "window_px": win, "dem": opts.dem}
    if col_offset:
        out["tmc_col"] = col_c

    # ---- stage 1: prior from SYSTEM corners only --------------------------------
    g = np.linspace(0, win - 1, 11)
    gx, gy = np.meshgrid(g, g)
    lat, lon = sysm.pixel_to_latlon(r0 + gy.ravel(), c0 + gx.ravel())
    tr, tcol = tcm.latlon_to_pixel(lat, lon)
    A_sys = affine_from(np.c_[gx.ravel(), gy.ravel()], np.c_[tcol, tr])     # window px -> TC px
    lat_c = float(np.mean(lat))
    # Optional known correction (east, north metres), applied to the SYSTEM prior so
    # a small search margin suffices. It is added back into the reported offset, so
    # system_offset_* is always the total correction from the system geolocation.
    prior_px = np.array([prior_offset_m[0] / (tc.gsd_m * math.cos(math.radians(lat_c))),
                         -prior_offset_m[1] / tc.gsd_m])
    A_sys = T(prior_px[0], prior_px[1]) @ A_sys
    out["lat"] = round(lat_c, 4)

    tmc_raw = pds_raster.read_raster(tmc, pds_raster.Window(r0, c0, win, win)).astype(np.float32)
    tmc_valid = np.isfinite(tmc_raw) & (tmc_raw > 0)
    if tmc_valid.mean() < 0.9:
        out["status"] = "skipped: TMC-2 window has too little valid data"
        return out
    tmc_n = norm(tmc_raw, tmc_valid)

    corners = np.array([[0, 0, 1], [win, 0, 1], [0, win, 1], [win, win, 1]], float) @ A_sys.T
    lo_xy, hi_xy = corners[:, :2].min(0), corners[:, :2].max(0)
    margin_px = margin_km * 1000.0 / tc.gsd_m
    TL, TS = tc.array_shape
    reg_lo = np.maximum(np.floor(lo_xy - margin_px), 0).astype(int)
    reg_hi = np.minimum(np.ceil(hi_xy + margin_px), [TS, TL]).astype(int)
    if np.any(reg_hi - reg_lo < (hi_xy - lo_xy) + 10):
        out["status"] = "skipped: window falls off the TC tile"
        return out

    # ---- stage 2: coarse MIND search --------------------------------------------
    tc_raw = pds_raster.read_raster(tc, pds_raster.Window(int(reg_lo[1]), int(reg_lo[0]),
                                                          int(reg_hi[1] - reg_lo[1]),
                                                          int(reg_hi[0] - reg_lo[0]))).astype(np.float32)
    tc_valid = tc_raw > 0
    region = norm(block(tc_raw, coarse), block(tc_valid.astype(np.float32), coarse) > 0.99)

    canvas_o = np.floor(lo_xy).astype(int)                              # TC px of template canvas
    size = np.ceil((hi_xy - canvas_o) / coarse).astype(int) + 1
    Wc = np.diag([1.0 / coarse, 1.0 / coarse, 1.0]) @ T(-canvas_o[0], -canvas_o[1]) @ A_sys
    tpl = warp_affine(tmc_n, Wc, (int(size[0]), int(size[1])))   # anti-aliased (cv2 ignores INTER_AREA)
    tmask = cv2.warpAffine(tmc_valid.astype(np.float32), Wc[:2], (int(size[0]), int(size[1])),
                           flags=cv2.INTER_NEAREST) > 0.5
    yy, xx = np.where(tmask)
    y0, y1, x0, x1 = yy.min(), yy.max(), xx.min(), xx.max()
    while not tmask[y0:y1, x0:x1].all():
        y0, y1, x0, x1 = y0 + 1, y1 - 1, x0 + 1, x1 - 1
    tpl = tpl[y0:y1, x0:x1]

    fr, ft = mind(region).astype(np.float32), mind(norm(tpl)).astype(np.float32)
    cmap = np.mean([cv2.matchTemplate(fr[..., i], ft[..., i], cv2.TM_CCOEFF_NORMED)
                    for i in range(fr.shape[-1])], axis=0)
    iy, ix = np.unravel_index(np.argmax(cmap), cmap.shape)
    z = (float(cmap[iy, ix]) - float(cmap.mean())) / float(cmap.std() + 1e-9)
    pred = (canvas_o - reg_lo) / coarse + np.array([x0, y0])        # where the prior says it sits
    d = (np.array([ix, iy]) - pred) * coarse                         # correction, TC px
    out.update(coarse_z=round(z, 1), coarse_peak=round(float(cmap[iy, ix]), 3),
               system_offset_px=[round(float((d + prior_px)[0]), 1), round(float((d + prior_px)[1]), 1)])
    dt = d + prior_px
    ground = np.array([dt[0] * tc.gsd_m * math.cos(math.radians(lat_c)), -dt[1] * tc.gsd_m])
    out["system_offset_m"] = {"east": round(float(ground[0])), "north": round(float(ground[1])),
                              "total": round(float(np.hypot(*ground)))}
    if z < 10:
        out["status"] = f"no unambiguous coarse match (z {z:.1f} < 10)"
        return out

    # ---- stage 3: fine registration on the coarse-aligned pair ------------------
    A1 = T(d[0], d[1]) @ A_sys                                        # window px -> TC px, corrected
    c1 = np.array([[0, 0, 1], [win, 0, 1], [0, win, 1], [win, win, 1]], float) @ A1.T
    o_f = np.floor(c1[:, :2].min(0)).astype(int) + 8
    e_f = np.ceil(c1[:, :2].max(0)).astype(int) - 8
    wF, hF = int(e_f[0] - o_f[0]), int(e_f[1] - o_f[1])
    Wf = T(-o_f[0], -o_f[1]) @ A1
    src_img = warp_affine(tmc_n, Wf, (wF, hF))                   # anti-aliased (cv2 ignores INTER_AREA)
    src_ok = cv2.warpAffine(tmc_valid.astype(np.float32), Wf[:2], (wF, hF), flags=cv2.INTER_NEAREST) > 0.5
    ref_raw = pds_raster.read_raster(tc, pds_raster.Window(int(o_f[1]), int(o_f[0]), hF, wF)).astype(np.float32)
    ref_ok = ref_raw > 0
    ref_img = norm(ref_raw, ref_ok)

    def plane(a, ok, gsd):
        return synth.ImagePlane(array=a, valid_mask=ok, shadow_mask=np.zeros(a.shape, bool),
                                gsd_m=gsd, meta=None, geo=None)

    t0 = time.perf_counter()
    src_plane, ref_plane = plane(src_img, src_ok, tc.gsd_m), plane(ref_img, ref_ok, tc.gsd_m)
    ms = adapter.match(src_plane, ref_plane,
                       model_name=matcher, device=device, **match_kwargs)
    out["match_seconds"] = round(time.perf_counter() - t0, 1)
    out["device"] = ms.device
    n = int(len(ms.src_pts))
    # The fine stage (pipeline.fine_stage): terrain filter, robust estimate, uniform
    # control points, per-point sub-pixel, final fit -- each switched by `stages`.
    # After the coarse lock both images are in TC's frame (offset o_f), so one
    # ground model serves both ends of every match.
    def rematch(a, b):                                   # ALIGN-05 refill: same matcher, one cell
        return adapter.match(plane(a, np.ones(a.shape, bool), tc.gsd_m), plane(b, np.ones(b.shape, bool), tc.gsd_m),
                             model_name=matcher, device=device,
                             **{k: v for k, v in match_kwargs.items() if k != "tile_px"})

    t_dem = _dem_for(tcm, o_f, wF, hF, opts=opts) if stages["geometry_filter"] else None
    p_dem, out["parallax_dem"] = t_dem, opts.dem
    if stages["parallax"] and opts.parallax_dem and opts.parallax_dem != opts.dem:
        own = _dem_for(tcm, o_f, wF, hF, which=opts.parallax_dem, opts=opts)
        if own is not None:            # where it is not held, the stage falls back to --dem
            p_dem, out["parallax_dem"] = own, opts.parallax_dem
    fr = fine_stage(ms, src_img, ref_img, centre=(wF / 2.0, hF / 2.0), flags=stages, rematch=rematch,
                    ground_model=_OffsetModel(tcm, o_f), dem=t_dem, parallax_dem=p_dem,
                    parallax_height_at=opts.parallax_height_at)
    res, ms = fr.first, fr.matches
    out.update(matcher=matcher, matches=n, inliers=res.inlier_count,
               inlier_ratio=round(res.inlier_count / n, 4) if n else 0.0,
               pipeline=fr.stages, control_points=int(len(fr.control_src)))
    if not fr.ok:
        out["status"] = "fine stage: no transform"
        return out
    if opts.dump_dir is not None:
        inl = fr.first.inlier_mask
        np.savez_compressed(Path(opts.dump_dir) / (f"window_{row_c}.npz" if not col_offset else f"window_{row_c}_c{col_c}.npz"),
                            control_src=fr.control_src, control_ref=fr.control_ref,
                            inlier_src=fr.matches.src_pts[inl], inlier_ref=fr.matches.ref_pts[inl],
                            model=np.asarray(fr.model.matrix, float), src_ok=src_ok, ref_ok=ref_ok,
                            src_img=src_img.astype(np.float16), ref_img=ref_img.astype(np.float16),
                            **_dem_grid(p_dem, tcm, o_f, wF, hF), gsd_m=float(tc.gsd_m),
                            tc_origin_px=np.asarray(o_f, float), tc_product=str(tc.product_id))

    R = np.asarray(fr.model.matrix, float)
    T_total = T(o_f[0], o_f[1]) @ R @ Wf                              # window px -> TC px
    coverage = fr.coverage

    # Tiled matching only: how the matches near an internal tile boundary behave
    # against the interior ones (verify protocol 3.B, reported, not gated).
    boxes = getattr(ms, "tile_boxes", None)
    if boxes and len(boxes) > 1 and res.inlier_count:
        margin = int(getattr(ms, "tile_margin_px", 64))
        ys = sorted({b[0] for b in boxes if b[0] > 0})
        xs = sorted({b[2] for b in boxes if b[2] > 0})
        p = ms.src_pts
        near = np.zeros(len(p), bool)
        for y in ys:
            near |= np.abs(p[:, 1] - y) < margin
        for x in xs:
            near |= np.abs(p[:, 0] - x) < margin
        r_all = np.asarray(robust.models.residuals(res.model, ms.src_pts, ms.ref_pts))
        inl = res.inlier_mask
        rms = lambda m: round(float(np.sqrt(np.mean(r_all[m] ** 2))), 3) if m.any() else None  # noqa: E731
        out["tile_edges"] = {
            "tiles": len(boxes), "margin_px": margin,
            "share_of_matches_near_edge": round(float(near.mean()), 4),
            "share_of_inliers_near_edge": round(float(near[inl].mean()), 4),
            "inlier_ratio_near_edge": round(float(inl[near].mean()), 4) if near.any() else None,
            "inlier_ratio_interior": round(float(inl[~near].mean()), 4) if (~near).any() else None,
            "inlier_rmse_near_edge_px": rms(inl & near),
            "inlier_rmse_interior_px": rms(inl & ~near)}

    # Scale check on the COMPOSED transform, against real per-axis expectations.
    exp = scale.expected_scale(scale.pixel_scale(tmc), tc_pixel_scale(tc, lat_c))
    verdict = scale.check(T_total, exp, centre=(win / 2.0, win / 2.0))
    # CHECK-01..04, 06 on THIS pair, with the SAME matcher that produced the result,
    # recorded with it and fed to the verdict: a failed gate rejects the window.
    t_g = time.perf_counter()
    # --gate-reuse-base: the main registration went through the same matcher and
    # estimator call on the same arrays as the perturbation gate's own baseline,
    # so it is handed over instead of being recomputed (G1, verify protocol 3.B).
    base = (control_gates.PipelineRun(bool(res.ok), n, int(res.inlier_count), R)
            if gate_reuse_base else None)
    gates = control_gates.run_all(control_gates.pipeline_from(matcher, device=device, gsd_m=tc.gsd_m,
                                                              stages=stages, **match_kwargs),
                                  src_img, ref_img, src_plane, ref_plane, base=base)
    out["gates"] = gates.gates
    out["gate_detail"] = gates.to_dict()
    out["gate_seconds"] = round(time.perf_counter() - t_g, 1)
    q = quality.assess(inlier_count=res.inlier_count,
                       inlier_ratio=res.inlier_count / n if n else 0.0,
                       spatial_coverage=coverage, model=fr.model,
                       scale_ok=verdict.ok, scale_status=verdict.status,
                       gates=gates.gates, require_gates=True)
    across, along = tmc_pixel_from(T_total[:2, :2], (tc.gsd_m * math.cos(math.radians(lat_c)), tc.gsd_m))
    rmse_px = fr.rmse_px                  # final model on the delivered control points
    out.update(coverage=round(coverage, 3), scale_status=verdict.status,
               tier=q.tier, limiting_signal=q.limiting_signal,
               inlier_rmse_px=round(rmse_px, 3) if rmse_px is not None else None,
               tmc_pixel_from_tc_m={"across": round(across, 3), "along": round(along, 3)},
               fine_residual_shift_px=[round(float(R[0, 2]), 2), round(float(R[1, 2]), 2)])

    # The Part 2 -> Part 3 handoff object, built for real and refused if ungated
    # (CHECK-08). Its transform is the COMPOSED TMC-2 window px -> TC px map.
    # ALIGN-08: the terrain-aware geometry, composed to the same frames as `model`
    # (TMC-2 window px -> TC px); p and h0 are unchanged by the composition.
    terrain = None
    if fr.parallax is not None:
        terrain = ParallaxModel(matrix=T(o_f[0], o_f[1]) @ np.asarray(fr.parallax.matrix, float) @ Wf,
                                p_px_per_m=fr.parallax.p_px_per_m, h0_m=fr.parallax.h0_m,
                                dem=out["parallax_dem"], height_at=fr.parallax.height_at)
    result = RegistrationResult(
        matches=ms, inlier_mask=res.inlier_mask,
        model=TransformModel(kind=fr.model.kind, matrix=T_total,
                             scale_estimated=fr.model.scale_estimated),
        metrics=Metrics(rmse_px=rmse_px,
                        rmse_m=None if rmse_px is None else rmse_px * tc.gsd_m,
                        inlier_count=res.inlier_count,
                        inlier_ratio=res.inlier_count / n if n else None,
                        spatial_coverage=coverage, runtime_s=out["match_seconds"],
                        source="measured"),
        confidence_tier=q.tier, gates=gates.gates, failure_modes=list(q.failure_modes),
        notes=list(res.notes) + list(q.notes),
        provenance={"matcher": matcher, "device": ms.device, "prior": "TMC-2 SYSTEM corners",
                    "reference": tc.product_id,
                    **({"terrain_model": terrain.as_dict()} if terrain is not None else {})})
    control_gates.require_gates(result)
    out["registration_result"] = {
        "confidence_tier": result.confidence_tier, "gates": result.gates,
        "failure_modes": result.failure_modes,
        "metrics": {k: v for k, v in asdict(result.metrics).items() if v is not None},
        "model": {"kind": result.model.kind, "matrix": [[round(float(v), 8) for v in row] for row in T_total]}}
    # The problem statement's unit is the SOURCE pixel; errors above are in fine-frame (TC) px.
    J_src = jacobian_from_transform(Wf)                              # TMC-2 window px -> frame px
    out["source_px"] = {
        "known_shift": to_source_px(out["gate_detail"].get("perturbation_sensitivity", {}).get("error_px"),
                                    J_src, "exact (Wf)"),
        "inlier_rmse_affine": to_source_px(rmse_px, J_src, "exact (Wf)"),
        "parallax_fit_rms": to_source_px((fr.stages.get("parallax") or {}).get("rms_px"), J_src, "exact (Wf)")}
    # MATCH-07: matcher-free check -- where NMI peaks around the delivered model (docs/mi_protocol.md)
    mi = alignment_check(src_img, ref_img, np.asarray(fr.model.matrix, float), src_ok=src_ok, ref_ok=ref_ok)
    if mi.get("peak_offset_px") is not None:
        mi["peak_offset_src_px"] = to_source_px(float(np.hypot(*mi["peak_offset_px"])), J_src, "exact (Wf)")
    out["mi_check"] = mi
    if terrain is not None:
        out["registration_result"]["terrain_model"] = {
            **terrain.as_dict(),
            "apply": "TC px = matrix . [x, y, 1] + (h - h0_m) * p_px_per_m; x, y are TMC-2 window px and h is "
                     "the DEM height (m) at the point's ground position (matrix . [x, y, 1] in TC px is close "
                     "enough to sample it: the DEM is smooth on that scale)"}

    # ---- stage 4: compare with ISRO's refined (SELENE-fitted) solution ----------
    centre = np.array([win / 2.0, win / 2.0, 1.0])
    ours = T_total @ centre
    la_r, lo_r = refm.pixel_to_latlon(np.array([r0 + win / 2.0]), np.array([c0 + win / 2.0]))
    rr, rc = tcm.latlon_to_pixel(la_r, lo_r)
    dx, dy = float(ours[0] - rc[0]), float(ours[1] - rr[0])
    out["vs_isro_refined_grid_m"] = round(float(np.hypot(dx * tc.gsd_m * math.cos(math.radians(lat_c)),
                                                    dy * tc.gsd_m)), 1)
    out["status"] = "registered"
    if capture is not None:
        capture.update(tmc_n=tmc_n, tmc_valid=tmc_valid, window_origin=(r0, c0), Wf=Wf, o_f=o_f,
                       src_img=src_img, src_ok=src_ok, ref_img=ref_img, ref_ok=ref_ok, fine=fr,
                       matches=ms, result=result, terrain=terrain, parallax_dem=p_dem, tcm=tcm)
    return out
