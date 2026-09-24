"""Register Chandrayaan-2 TMC-2 to SELENE TC: the first real CH-2 <-> reference result.

    .venv/Scripts/python scripts/register_tmc2_tc.py --tile N03 --windows 1
    .venv/Scripts/python scripts/register_tmc2_tc.py --tile all --windows 3

Needs data/raw/ch2/tmc2 and data/raw/selene/tc (see data/manifest.json).

THE ONE RULE (agreed with Member A): TMC-2's REFINED geolocation -- its refined
corners and its geometry grid -- was fitted by ISRO against SELENE. Using it
anywhere in this registration, or to judge it, would be circular. So:

  prior        TMC-2 SYSTEM corners (spacecraft data only) -> TC map projection
  registration pixels only, from that prior
  judged by    Part 2's own gates: MAGSAC inliers, per-axis scale check,
               quality gate. The refined geolocation is compared AFTERWARDS and
               reported as "agreement with ISRO's own SELENE fit" -- a
               reproduction check, explicitly not independent evidence.

STAGES
  1 prior    system-corner prediction of where each TMC-2 window lands in TC
  2 coarse   MIND template search over +-margin km (system geolocation of a
             CH-2 product can be km off: TMC-2's own refinement moved it 5.4 km)
  3 fine     learned feature matching on the coarse-aligned pair, MAGSAC, then
             the per-axis scale check and the quality gate on the COMPOSED
             TMC-2 -> TC transform
  4 report   system geolocation error (m), TMC-2 pixel size implied by TC, and
             distance from ISRO's refined solution

TC is SIMPLE CYLINDRICAL at 7.403 m/px (at the equator); its cross-track pixel
is 7.403 * cos(lat) on the ground, which the scale check accounts for.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dataclasses import asdict  # noqa: E402

from chandralign import synth  # noqa: E402
from chandralign.contracts import Metrics, RegistrationResult, TransformModel  # noqa: E402
from chandralign.estimate import robust, scale  # noqa: E402
from chandralign.estimate.scale import PixelScale  # noqa: E402
from chandralign.evaluate import control_gates, quality  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.pipeline import fine_stage, stage_flags  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.preprocess.resample import warp_affine  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.matching import adapter, routing  # noqa: E402
from chandralign.preprocess.phase_congruency import mind  # noqa: E402

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


def _dem_for(tcm, origin_xy, w: int, h: int):
    """SLDEM covering a TC region (with a margin), or None if no tile is held."""
    from chandralign.io.dem import dem_patch, find_tiles
    tiles = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")
    if not tiles:
        return None
    x0, y0 = origin_xy
    lat, lon = tcm.pixel_to_latlon(np.array([y0, y0 + h, y0, y0 + h], float),
                                   np.array([x0, x0, x0 + w, x0 + w], float))
    m = 0.02
    return dem_patch(tiles, (float(np.min(lat)) - m, float(np.max(lat)) + m,
                             float(np.min(lon)) - m, float(np.max(lon)) + m))


DUMP_DIR = None   # --dump-points: save each window's control points and inliers (docs/tps_protocol.md)


def run_window(tmc, sysm, refm, tc, tcm, row_c: int, *, win: int, coarse: int,
               margin_km: float, matcher: str, device: str,
               prior_offset_m: tuple[float, float] = (0.0, 0.0),
               match_kwargs: dict | None = None, gate_reuse_base: bool = False,
               stages: dict | None = None) -> dict:
    match_kwargs = dict(match_kwargs or {})
    stages = stage_flags(stages)
    col_c = tmc.array_shape[1] // 2
    r0, c0 = row_c - win // 2, col_c - win // 2
    out: dict = {"tmc_row": row_c, "window_px": win}

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
    fr = fine_stage(ms, src_img, ref_img, centre=(wF / 2.0, hF / 2.0), flags=stages,
                    ground_model=_OffsetModel(tcm, o_f),
                    dem=_dem_for(tcm, o_f, wF, hF) if stages["geometry_filter"] else None)
    res, ms = fr.first, fr.matches
    out.update(matcher=matcher, matches=n, inliers=res.inlier_count,
               inlier_ratio=round(res.inlier_count / n, 4) if n else 0.0,
               pipeline=fr.stages, control_points=int(len(fr.control_src)))
    if not fr.ok:
        out["status"] = "fine stage: no transform"
        return out
    if DUMP_DIR is not None:
        inl = fr.first.inlier_mask
        np.savez_compressed(Path(DUMP_DIR) / f"window_{row_c}.npz",
                            control_src=fr.control_src, control_ref=fr.control_ref,
                            inlier_src=fr.matches.src_pts[inl], inlier_ref=fr.matches.ref_pts[inl],
                            model=np.asarray(fr.model.matrix, float))

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
                    "reference": tc.product_id})
    control_gates.require_gates(result)
    out["registration_result"] = {
        "confidence_tier": result.confidence_tier, "gates": result.gates,
        "failure_modes": result.failure_modes,
        "metrics": {k: v for k, v in asdict(result.metrics).items() if v is not None},
        "model": {"kind": result.model.kind, "matrix": [[round(float(v), 8) for v in row] for row in T_total]}}

    # ---- stage 4: compare with ISRO's refined (SELENE-fitted) solution ----------
    centre = np.array([win / 2.0, win / 2.0, 1.0])
    ours = T_total @ centre
    la_r, lo_r = refm.pixel_to_latlon(np.array([r0 + win / 2.0]), np.array([c0 + win / 2.0]))
    rr, rc = tcm.latlon_to_pixel(la_r, lo_r)
    dx, dy = float(ours[0] - rc[0]), float(ours[1] - rr[0])
    out["vs_isro_refined_grid_m"] = round(float(np.hypot(dx * tc.gsd_m * math.cos(math.radians(lat_c)),
                                                    dy * tc.gsd_m)), 1)
    out["status"] = "registered"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="N03", choices=sorted(TILES) + ["all"])
    ap.add_argument("--windows", type=int, default=1, help="TMC-2 windows per tile, spread along-track")
    ap.add_argument("--win", type=int, default=1536, help="TMC-2 window size (px)")
    ap.add_argument("--coarse", type=int, default=2, help="block factor for the coarse search")
    ap.add_argument("--margin-km", type=float, default=7.0)
    ap.add_argument("--rows", type=int, nargs="+", default=None,
                    help="explicit TMC-2 window-centre rows; the tile is chosen by latitude")
    ap.add_argument("--prior-offset-m", type=float, nargs=2, default=(0.0, 0.0), metavar=("EAST", "NORTH"),
                    help="known correction applied to the system prior, so --margin-km can shrink")
    ap.add_argument("--matcher", default=None,
                    help="override the routed matcher; omit to let matching.routing choose")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--tile-px", type=int, default=None,
                    help="split the coarse-aligned pair into tiles no larger than this")
    ap.add_argument("--gate-reuse-base", action="store_true",
                    help="perturbation gate reuses the main registration as its baseline")
    ap.add_argument("--dump-points", default=None, help="folder: save control points and inliers per window")
    ap.add_argument("--stage", action="append", default=[], metavar="NAME=on|off",
                    help="override a pipeline stage (geometry_filter, uniformity, subpixel); "
                         "the default comes from configs/default.yaml pipeline.*")
    ap.add_argument("--out", default="reports/tmc2_tc_registration.json")
    args = ap.parse_args()
    global DUMP_DIR
    if args.dump_points:
        DUMP_DIR = args.dump_points
        Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)
    stages = stage_flags({k: v.lower() in ("on", "1", "true") for k, v in
                          (item.split("=", 1) for item in args.stage)})
    print(f"pipeline stages: {stages}", flush=True)
    args.matcher, routed_opts, matcher_choice = resolve_matcher(args.matcher)
    print(f"matcher: {args.matcher}  ({matcher_choice['chosen_by']})", flush=True)
    # Routed fine-stage options first, explicit CLI flags on top of them.
    mk = {**routed_opts, "precision": args.precision}
    if args.tile_px:
        mk["tile_px"] = args.tile_px

    tmc = parse_label(next((ROOT / "data/raw/ch2/tmc2").rglob("*_d_img_d18.xml")))
    sysm = projection.load_corner_model(tmc, corners="system")
    # Stage 4 ONLY: ISRO's dense geometry grid, which it fitted against SELENE. It
    # is the actual refined solution; the refined CORNERS are a 4-point bilinear
    # model stretched over ~800 km and too crude to compare against.
    refm = projection.load_grid_model(tmc)
    assert sysm.independent_of_references and not refm.independent_of_references

    rows_all = np.arange(0, tmc.array_shape[0], 250)
    lat_all, _ = sysm.pixel_to_latlon(rows_all, np.full(rows_all.shape, tmc.array_shape[1] / 2))

    results = []
    if args.rows:
        tcs = {k: parse_label(ROOT / "data/raw/selene/tc" / f"{v}.lbl") for k, v in TILES.items()}
        tcms = {k: projection.load_map_model(m) for k, m in tcs.items()}
        shift_deg = args.prior_offset_m[1] / M_PER_DEG
        for row_c in args.rows:
            la, _ = sysm.pixel_to_latlon(np.array([float(row_c)]), np.array([tmc.array_shape[1] / 2.0]))
            la = float(la[0]) + shift_deg
            key = next((k for k, m in tcms.items()
                        if min(m.pixel_to_latlon([0, tcs[k].array_shape[0] - 1], [0, 0])[0]) <= la
                        <= max(m.pixel_to_latlon([0, tcs[k].array_shape[0] - 1], [0, 0])[0])), None)
            if key is None:
                print(json.dumps({"tmc_row": row_c, "status": f"no TC tile held at lat {la:.3f}"}), flush=True)
                continue
            t = time.perf_counter()
            r = run_window(tmc, sysm, refm, tcs[key], tcms[key], int(row_c), win=args.win,
                           coarse=args.coarse, margin_km=args.margin_km, matcher=args.matcher,
                           device=args.device, prior_offset_m=tuple(args.prior_offset_m),
                           match_kwargs=mk, gate_reuse_base=args.gate_reuse_base, stages=stages)
            r["tile"] = key
            r["seconds"] = round(time.perf_counter() - t, 1)
            results.append(r)
            print(json.dumps(r), flush=True)
    for key in ([] if args.rows else (sorted(TILES) if args.tile == "all" else [args.tile])):
        tc = parse_label(ROOT / "data/raw/selene/tc" / f"{TILES[key]}.lbl")
        tcm = projection.load_map_model(tc)
        la0, la1 = sorted(tcm.pixel_to_latlon([0, tc.array_shape[0] - 1], [0, 0])[0])
        pad = (args.margin_km + 10.0) * 1000.0 / M_PER_DEG
        inside = rows_all[(lat_all > la0 + pad) & (lat_all < la1 - pad)]
        if len(inside) == 0:
            print(f"{key}: no TMC-2 rows safely inside the tile", flush=True)
            continue
        picks = np.linspace(inside.min(), inside.max(), args.windows + 2)[1:-1].astype(int)
        for row_c in picks:
            t = time.perf_counter()
            r = run_window(tmc, sysm, refm, tc, tcm, int(row_c), win=args.win, coarse=args.coarse,
                           margin_km=args.margin_km, matcher=args.matcher, device=args.device,
                           match_kwargs=mk, gate_reuse_base=args.gate_reuse_base, stages=stages)
            r["tile"] = key
            r["seconds"] = round(time.perf_counter() - t, 1)
            results.append(r)
            print(json.dumps(r), flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "measured",
        "pairing": "TMC-2 (ch2_tmc_nca_20250207T1102039417) <-> SELENE TC",
        "prior": "TMC-2 SYSTEM corners only; refined geolocation used for stage-4 comparison only",
        "match_options": {**mk, "gate_reuse_base": bool(args.gate_reuse_base)},
        "pipeline_stages": stages,
        "matcher_choice": matcher_choice,
        "run": run_record(),
        "rows": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def resolve_matcher(explicit: str | None, src: str = "TMC2", ref: str = "TC"):
    """(matcher, fine-stage options, provenance) for this run.

    With no --matcher, the choice comes from matching.routing.choose -- the same
    decision Part 3's CLI and API will use -- instead of a name hard-coded here.
    Until routing.py existed this script named eloftr directly and the regime
    selector had no caller outside the tests. An explicit --matcher still wins,
    and is recorded as an override so a result never claims routing chose it.
    """
    if explicit:
        return explicit, {}, {"route": "direct", "matcher": explicit,
                              "chosen_by": "--matcher override"}
    choice = routing.choose(src, ref)
    if choice.route != "direct" or not choice.model_name:
        raise SystemExit(f"routing sends {src} -> {ref} to route {choice.route!r}, not a direct "
                         f"match; this script only runs the direct TMC-2 -> TC path")
    return choice.model_name, dict(choice.fine_stage_options), choice.as_provenance()


if __name__ == "__main__":
    raise SystemExit(main())
