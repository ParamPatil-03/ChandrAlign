"""The adopted OHRC -> LRO NAC registration, one window at a time (the credibility pairing).

Moved VERBATIM from scripts/register_ohrc_nac.py (audit 2026-09-26 C-01) so the product (CLI, API)
runs the same code the committed evidence came from; the script is now a thin wrapper. Its WIN and
DUMP_DIR globals became the `win=` and `dump_dir=` arguments. Method and frozen settings:
docs/ohrc_nac_protocol.md and the script's docstring.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .. import config, synth
from ..estimate import scale
from ..estimate.scale import PixelScale
from ..evaluate import control_gates, quality
from ..evaluate.source_px import jacobian_from_transform, to_source_px
from ..geometry.nac import MOON_R_M, NULL_BELOW, Shifted, enu
from ..io import pds_raster
from ..io.dem import dem_patch
from ..matching import adapter, cascade, classical, routing
from ..matching.similarity import alignment_check
from ..pipeline import fine_stage
from ..preprocess.resample import warp_affine

# ---- frozen in docs/ohrc_nac_protocol.md -------------------------------------------
PRODUCTS = ("M102014464RC", "M106719774LC", "M175124932LC", "M1417360906LC", "M109080308LC")
WIN = 2048                    # OHRC native px per window side (~0.61 km); --win / win= overrides
N_WIN = 5
MARGIN_M = 4000.0             # position search, each side
COARSE_M = 2.0                # coarse lock resolution. Was 4.0; 2 m chosen by docs/illumination_fix_protocol.md Q1
MIN_Z = float(config.get("cascade.min_z", 10.0))
CONSISTENT_M = 150.0          # offset within this of the product's median
OHRC_PX = 0.30


def bridge_predictions(root=None) -> dict:
    """OHRC's predicted offset in each NAC product, metres east/north (MATCH-11 geodetic
    bridge; docs/illumination_fix_protocol.md Q5/Q8), COMPUTED from committed reports:
    OHRC vs SELENE TC (cascade_ohrc_tc) + that NAC vs TC measured through TMC-2 (PR #18)."""
    root = Path(root or config.ROOT)
    c = json.loads((root / "reports/cascade_ohrc_tc.json").read_text(encoding="utf-8"))
    o = np.median([[r["ohrc_system_offset_m"]["east"], r["ohrc_system_offset_m"]["north"]]
                   for r in c["rows"] if r.get("ohrc_system_offset_m")], axis=0)
    d = json.loads((root / "reports/tmc2_nac_registration.json").read_text(encoding="utf-8"))
    by = {}
    for w in d["windows"]:
        g = w.get("geolocation")
        if w.get("success") and g:
            by.setdefault(w["nac"], []).append(
                [g["system_offset_m"]["east"] - g["tc_measured_offset_m"]["east"],
                 g["system_offset_m"]["north"] - g["tc_measured_offset_m"]["north"]])
    return {p: tuple(float(v) for v in (o + np.median(x, axis=0))) for p, x in by.items()}

BRIDGE_MARGIN_M = 200.0
FINE_MIN_M = 0.5              # Q5 amendment 1: no fine-grid axis finer than this
FINE_MAX_PX = 1.5e6           # Q5 amendment 2: fine frame at most this many pixels (GPU memory)


def T(x, y):
    m = np.eye(3); m[:2, 2] = [x, y]; return m


def affine_fit(src, dst):
    sol, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
    A = np.eye(3); A[:2, :] = sol.T; return A


def norm(a, valid=None):
    v = a[valid] if valid is not None and valid.any() else a
    lo, hi = np.percentile(v, [1, 99])
    out = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)
    if valid is not None:
        out[~valid] = 0.0
    return out


def match(name, a, b, device, gsd):
    def plane(x):
        return synth.ImagePlane(array=x, valid_mask=np.ones(x.shape, bool), shadow_mask=np.zeros(x.shape, bool),
                                gsd_m=gsd, meta=None, geo=None)
    pa, pb = plane(a), plane(b)
    if name == "sift":
        return classical.match(pa, pb, detector="sift"), pa, pb
    return adapter.match(pa, pb, model_name=name, device=device), pa, pb


def run_window(ohrc, om, nacm, geo, nac, rc, cc, matchers, device, stages, dem_tiles,
               coarse_descriptor="mind", coarse_m=COARSE_M, coarse_only=False, bridge=None, *,
               win=WIN, dump_dir=None, capture=None):
    """One OHRC window registered to one LRO NAC product (the adopted OHRC -> NAC path).

    Returns the evidence record scripts/register_ohrc_nac.py writes per window. Pass a dict
    as `capture` to also receive each matcher's registration (capture[matcher] = frames, fine
    result, gates, tier) -- what the product layer turns into a RegistrationBundle.
    """
    WIN = win
    out = {"nac": nac.pid, "ohrc_row": int(rc), "ohrc_col": int(cc)}
    r0, c0 = int(rc) - WIN // 2, int(cc) - WIN // 2
    o_raw = pds_raster.read_raster(ohrc, pds_raster.Window(r0, c0, WIN, WIN)).astype(np.float32)
    o_ok = o_raw > 0
    if o_ok.mean() < 0.95:
        out["status"] = "skipped: OHRC window has invalid pixels"; return out
    ohrc_n = norm(o_raw, o_ok)

    # 1. system prior: OHRC window px -> NAC px
    g = np.linspace(0, WIN - 1, 9)
    gx, gy = np.meshgrid(g, g)
    lat, lon = om.pixel_to_latlon(r0 + gy.ravel(), c0 + gx.ravel())
    if bridge is not None:                  # Q5: place the window at its predicted true position
        k = math.pi / 180 * MOON_R_M
        lat = np.asarray(lat) + bridge[1] / k
        lon = np.asarray(lon) + bridge[0] / (k * np.cos(np.radians(lat)))
    nx, ny = geo.to_px(lat, lon)
    A_sys = affine_fit(np.c_[gx.ravel(), gy.ravel()], np.c_[nx, ny])
    lat_c, lon_c = (float(v) for v in om.pixel_to_latlon(np.array([r0 + WIN / 2]), np.array([c0 + WIN / 2])))
    out.update(lat=round(lat_c, 4), lon=round(lon_c, 4))

    # 2. coarse lock over +-MARGIN at ~COARSE_M (skipped in bridge mode)
    L, Sn = nacm.array_shape
    if bridge is not None:
        T_c = A_sys
        out["coarse"] = {"mode": "bridge", "offset_m": list(bridge)}
        out["system_offset_m"] = {"east": float(bridge[0]), "north": float(bridge[1])}
        return _fine(out, ohrc, ohrc_n, o_ok, T_c, nacm, geo, nac, lat_c, lon_c, matchers, device, stages,
                     dem_tiles, exp_margin_m=BRIDGE_MARGIN_M, win=WIN, dump_dir=dump_dir, capture=capture)
    cx, cy = (A_sys @ [WIN / 2, WIN / 2, 1])[:2]
    half = int(MARGIN_M / nac.px_h) + WIN
    l0, l1 = max(0, int(cy) - half), min(L, int(cy) + half)
    if l1 - l0 < WIN:
        out["status"] = "skipped: predicted position outside the NAC strip"; return out
    reg_raw = pds_raster.read_raster(nacm, pds_raster.Window(l0, 0, l1 - l0, Sn)).astype(np.float32)
    reg_ok = reg_raw > NULL_BELOW
    b = max(1, round(coarse_m / min(nac.px_w, nac.px_h)))
    f = max(1, round(coarse_m / OHRC_PX))
    reg_small = cascade.block_average(norm(reg_raw, reg_ok), b)
    D_ref = cascade.downsample_transform(b) @ T(0, -l0)                  # NAC px -> region-small px
    U_src = np.linalg.inv(cascade.downsample_transform(f))              # OHRC-small px -> OHRC px
    prior = D_ref @ A_sys @ U_src
    diag = {}
    step = cascade.register_step_dense(ohrc_n, reg_small, f, src="OHRC", ref=nac.pid,
                                       ref_pixel_m=coarse_m, prior=prior, diag=diag,
                                       descriptor=coarse_descriptor)
    out["coarse"] = {"z": diag.get("z"), "descriptor": coarse_descriptor, "resolution_m": coarse_m,
                     "block_ohrc": f, "block_nac": b, **({"failed": diag["failed"]} if "failed" in diag else {})}
    if step is None or (diag.get("z") or 0) < MIN_Z:
        out["status"] = "no coarse lock"; out["results"] = {}; return out
    T_c = np.linalg.inv(D_ref) @ np.asarray(step.model.matrix, float)    # OHRC window px -> NAC px

    # implied OHRC-system offset of the window centre, metres east/north
    px, py = (T_c @ [WIN / 2, WIN / 2, 1])[:2]
    la_n, lo_n = geo.pixel_to_latlon([py], [px])
    e = enu(float(la_n[0]), float(lo_n[0]), lat_c, lon_c)
    out["system_offset_m"] = {"east": round(float(e[0]), 1), "north": round(float(e[1]), 1)}
    if coarse_only:
        out["status"] = "locked"; out["results"] = {}; return out

    return _fine(out, ohrc, ohrc_n, o_ok, T_c, nacm, geo, nac, lat_c, lon_c, matchers, device, stages, dem_tiles,
                 win=WIN, dump_dir=dump_dir, capture=capture)


def _fine(out, ohrc, ohrc_n, o_ok, T_c, nacm, geo, nac, lat_c, lon_c, matchers, device, stages, dem_tiles,
          exp_margin_m=0.0, *, win=WIN, dump_dir=None, capture=None):
    WIN = win
    L, Sn = nacm.array_shape
    # 3. fine frame on the NAC grid
    cor = np.array([[0, 0, 1], [WIN, 0, 1], [0, WIN, 1], [WIN, WIN, 1]], float) @ T_c.T
    grow = np.array([exp_margin_m / nac.px_w, exp_margin_m / nac.px_h])
    o = (np.floor(cor[:, :2].min(0) - grow).astype(int) + 8)
    e_ = (np.ceil(cor[:, :2].max(0) + grow).astype(int) - 8)
    o[0], o[1] = max(o[0], 0), max(o[1], 0)
    e_[0], e_[1] = min(e_[0], Sn), min(e_[1], L)
    wF, hF = int(e_[0] - o[0]), int(e_[1] - o[1])
    if wF < 64 or hF < 64:
        out["status"] = "skipped: fine frame falls outside the NAC strip"; out["results"] = {}; return out
    # Fine grid: NAC native, block-averaged per axis so no axis is finer than FINE_MIN_M.
    bx, by = max(1, math.ceil(FINE_MIN_M / nac.px_w - 1e-9)), max(1, math.ceil(FINE_MIN_M / nac.px_h - 1e-9))
    while (wF // bx) * (hF // by) > FINE_MAX_PX:                     # coarsen the finer axis
        if nac.px_w * bx <= nac.px_h * by:
            bx += 1
        else:
            by += 1
    wF, hF = (wF // bx) * bx, (hF // by) * by
    D = np.array([[1 / bx, 0, 0.5 / bx - 0.5], [0, 1 / by, 0.5 / by - 0.5], [0, 0, 1]], float)
    Wf = D @ T(-o[0], -o[1]) @ T_c                                   # OHRC px -> fine-grid px
    wG, hG = wF // bx, hF // by
    src = warp_affine(ohrc_n, Wf, (wG, hG))
    ref_raw = pds_raster.read_raster(nacm, pds_raster.Window(int(o[1]), int(o[0]), hF, wF)).astype(np.float32)
    ref_ok = ref_raw > NULL_BELOW
    ref = norm(ref_raw, ref_ok)
    if bx > 1 or by > 1:
        ref = ref.reshape(hG, by, wG, bx).mean(axis=(1, 3)).astype(np.float32)
    src_okf = cv2.warpAffine(o_ok.astype(np.float32), Wf[:2], (wG, hG), flags=cv2.INTER_NEAREST) > 0.5
    ref_okf = ref_ok.reshape(hG, by, wG, bx).all(axis=(1, 3)) if (bx > 1 or by > 1) else ref_ok
    out["fine_frame_px"] = [wG, hG]
    out["fine_block"] = [bx, by]
    wF, hF = wG, hG

    dem = None
    if stages["geometry_filter"] and dem_tiles:
        m = 0.02
        dem = dem_patch(dem_tiles, (lat_c - m, lat_c + m, lon_c - m, lon_c + m))
    exp = scale.expected_scale(scale.pixel_scale(ohrc),
                               PixelScale(nac.pid, nac.px_w, nac.px_h, (nac.px_w, nac.px_w), (nac.px_h, nac.px_h),
                                          ("lroc_scaled_pixel",), False, ("LROC scaled pixel size",)))
    gsd = max(nac.px_w * bx, nac.px_h * by)
    results = {}

    def evaluate(name):
        if name in results:
            return results[name]
        t0 = time.perf_counter()
        r = {}
        try:
            ms, pa, pb = match(name, src, ref, device, gsd)
            n = int(len(ms.src_pts))
            if n < 4:
                results[name] = {"status": f"only {n} matches", "matches": n, "success": False}
                return results[name]
            fr = fine_stage(ms, src, ref, centre=(wF / 2, hF / 2), flags=stages,
                            ground_model=Shifted(geo, o[0], o[1], bx, by), dem=dem)
            r.update(matches=n, inliers=fr.inlier_count, inlier_ratio=round(fr.inlier_ratio, 4),
                     control_points=int(len(fr.control_src)), coverage=round(fr.coverage, 3),
                     inlier_rmse_px=None if fr.rmse_px is None else round(fr.rmse_px, 3), pipeline=fr.stages)
            if not fr.ok:
                results[name] = {**r, "status": "fine stage: no transform", "success": False}
                return results[name]
            if dump_dir is not None and name != "routed":
                gm = Shifted(geo, o[0], o[1], bx, by)
                ys, xs = np.mgrid[0:hF:8, 0:wF:8].astype(float)
                hts = (np.asarray(dem.sample(*gm.pixel_to_latlon(ys.ravel(), xs.ravel())), float).reshape(ys.shape)
                       if dem is not None else np.zeros(0))
                inl = fr.first.inlier_mask
                np.savez_compressed(Path(dump_dir) / f"{nac.pid}_{out.get('ohrc_row')}_{name}.npz",
                                    src_img=src, ref_img=ref, model=np.asarray(fr.model.matrix, float),
                                    src_ok=cv2.warpAffine(o_ok.astype(np.float32), Wf[:2], (wF, hF), flags=cv2.INTER_NEAREST) > 0.5,
                                    ref_ok=np.ones((hF, wF), bool), inlier_src=fr.matches.src_pts[inl],
                                    inlier_ref=fr.matches.ref_pts[inl], J=Wf[:2, :2], dem_m=hts, dem_step=8,
                                    frame_px_m=np.array([nac.px_w * bx, nac.px_h * by]),
                                    T_total=T(o[0], o[1]) @ np.linalg.inv(D) @ np.asarray(fr.model.matrix, float) @ Wf,
                                    ohrc_col=int(out.get("ohrc_col") or -1))
            T_total = T(o[0], o[1]) @ np.linalg.inv(D) @ np.asarray(fr.model.matrix, float) @ Wf
            fx, fy = (T_total @ [WIN / 2, WIN / 2, 1])[:2]
            la_f, lo_f = geo.pixel_to_latlon([fy], [fx])
            ef = enu(float(la_f[0]), float(lo_f[0]), lat_c, lon_c)
            r["implied_offset_m"] = {"east": round(float(ef[0]), 1), "north": round(float(ef[1]), 1)}
            verdict = scale.check(T_total, exp, centre=(WIN / 2, WIN / 2))
            gates = control_gates.run_all(control_gates.pipeline_from(name, device=device, gsd_m=gsd, stages=stages),
                                          src, ref, pa, pb)
            q = quality.assess(inlier_count=fr.inlier_count, inlier_ratio=fr.inlier_ratio,
                               spatial_coverage=fr.coverage, model=fr.model, scale_ok=verdict.ok,
                               scale_status=verdict.status, gates=gates.gates, require_gates=True)
            pert = gates.to_dict().get("perturbation_sensitivity", {})
            r.update(status="registered", tier=q.tier, gates=gates.gates, scale_status=verdict.status,
                     known_shift_error_px=pert.get("error_px"),
                     gates_pass=bool(gates.all_passed), tier_ok=q.tier in ("HIGH", "MEDIUM", "LOW"))
            J_src = jacobian_from_transform(Wf)                        # OHRC px -> fine-frame (NAC) px
            r["source_px"] = {"known_shift": to_source_px(pert.get("error_px"), J_src, "exact (Wf)"),
                              "inlier_rmse": to_source_px(fr.rmse_px, J_src, "exact (Wf)")}
            mi = alignment_check(src, ref, np.asarray(fr.model.matrix, float), src_ok=src_okf, ref_ok=ref_okf)
            if mi.get("peak_offset_px") is not None:
                mi["peak_offset_src_px"] = to_source_px(float(np.hypot(*mi["peak_offset_px"])), J_src, "exact (Wf)")
            r["mi_check"] = mi                                     # MATCH-07 (docs/mi_protocol.md)
            if capture is not None:
                capture[name] = dict(src=src, ref=ref, src_ok=src_okf, ref_ok=ref_okf, fine=fr, gates=gates,
                                     quality=q, scale=verdict, Wf=Wf, D=D, origin=(int(o[0]), int(o[1])),
                                     block=(bx, by), T_total=T_total, gsd=gsd, window_origin=(
                                         int(out["ohrc_row"]) - WIN // 2, int(out["ohrc_col"]) - WIN // 2),
                                     win=WIN, ohrc_n=ohrc_n, o_ok=o_ok, geo=geo, dem=dem, mi=mi)
        except Exception as exc:                                        # recorded, never hidden
            r.update(status=f"error: {type(exc).__name__}: {exc}"[:300], success=False)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = r
        return r

    ok = lambda r: r.get("status") == "registered" and r.get("gates_pass") and r.get("tier_ok")
    for name in matchers:
        if name != "routed":
            evaluate(name)
            continue
        # The shipped behaviour: routing's matcher, then its fallbacks ONLY if rejected.
        tried = []
        for cand in routing.choose("OHRC", "NAC").candidates():
            tried.append(cand)
            if ok(evaluate(cand)):
                break
        results["routed"] = {**results[tried[-1]], "used": tried[-1], "tried": tried}
    out["results"] = results
    out["status"] = "locked"
    return out


def place_windows(ohrc, om, nac, geo, nac_shape, win, *, bridge=None, base_win=None) -> dict:
    """{OHRC row: column} of centres whose `win`-px window lies inside the NAC footprint.

    Moved verbatim from scripts/register_ohrc_nac.py (its `place` closure, protocol amendments
    1 and 2). Candidate rows step 250 px from `base_win` (the run's window size; the script's
    Q10 enlargement places 2x windows on the same rows); `bridge` (east, north m) places by the
    geodetic-bridge prediction instead of the system position (Q5/Q8).
    """
    L, S = ohrc.array_shape
    Ln, Sn = nac_shape
    base = win if base_win is None else base_win
    rows_all = np.arange(base, L - base, 250)
    pad = 1.1 * (win / 2) * OHRC_PX / min(nac.px_w, nac.px_h)
    cols = np.arange(win // 2, S - win // 2 + 1, 256)
    cols = cols[np.argsort(np.abs(cols - S / 2.0), kind="stable")]
    out = {}
    for cc in cols:                                   # nearest-to-centre first
        la, lo = om.pixel_to_latlon(rows_all, np.full(rows_all.shape, float(cc)))
        if bridge is not None:                        # Q5/Q8: place by the PREDICTED position
            k = math.pi / 180 * MOON_R_M
            la = np.asarray(la) + bridge[1] / k
            lo = np.asarray(lo) + bridge[0] / (k * np.cos(np.radians(la)))
        x, y = geo.to_px(la, lo)
        ok = (x > pad) & (x < Sn - pad) & (y > pad) & (y < Ln - pad)
        for rc in rows_all[ok]:
            if win // 2 <= rc <= L - win // 2:
                out.setdefault(int(rc), int(cc))
    return out
