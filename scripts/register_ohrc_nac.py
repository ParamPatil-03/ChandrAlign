"""OHRC -> LRO NAC on real data: the credibility-floor pairing, and a real illumination test.

    python scripts/register_ohrc_nac.py                         # 5 NAC x 5 windows x 3 matchers
    python scripts/register_ohrc_nac.py --products M102014464RC --matchers eloftr

Protocol, frozen before this script existed: docs/ohrc_nac_protocol.md.

Per window: a system-level prior (OHRC's system grid, LROC's NAC corners) gives
orientation and scale; a MIND coarse lock (cascade.register_step_dense) at ~4 m finds
the position over +-4 km; OHRC is then warped (anti-aliased) onto the NAC grid and the
matcher under test runs, followed by pipeline.fine_stage, all five control gates and
quality.assess -- the same fine stage as scripts/register_tmc2_tc.py.
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
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from chandralign import config, synth  # noqa: E402
from chandralign.estimate import scale  # noqa: E402
from chandralign.estimate.scale import PixelScale  # noqa: E402
from chandralign.evaluate import control_gates, quality  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.dem import dem_patch, find_tiles  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.matching import adapter, cascade, classical  # noqa: E402
from chandralign.pipeline import fine_stage, stage_flags  # noqa: E402
from chandralign.preprocess.resample import warp_affine  # noqa: E402
from register_tmc2_nac import MOON_R_M, NULL_BELOW, Nac, enu  # noqa: E402

# ---- frozen in docs/ohrc_nac_protocol.md -------------------------------------------
PRODUCTS = ("M102014464RC", "M106719774LC", "M175124932LC", "M1417360906LC", "M109080308LC")
WIN = 2048                    # OHRC native px per window side (~0.61 km)
N_WIN = 5
MARGIN_M = 4000.0             # position search, each side
COARSE_M = 4.0                # coarse lock resolution
MIN_Z = float(config.get("cascade.min_z", 10.0))
CONSISTENT_M = 150.0          # offset within this of the product's median
OHRC_PX = 0.30


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


class NacGeo:
    """NAC line/sample <-> lat/lon from LROC's corners (Nac.latlon), inverted by an
    affine fit in a local metric frame; exact enough for a prior (position is searched)."""

    def __init__(self, nac: Nac, lines: int, samples: int):
        self.nac = nac
        g = np.array([(l, s) for l in np.linspace(0, lines - 1, 7) for s in np.linspace(0, samples - 1, 5)])
        ll = np.array([nac.latlon(l, s) for l, s in g])
        self.lat0, self.lon0 = ll.mean(0)
        e = np.array([enu(la, lo, self.lat0, self.lon0) for la, lo in ll])
        self.fwd = affine_fit(g[:, ::-1], e)                 # (samp, line) -> (east, north)
        self.inv = np.linalg.inv(self.fwd)

    def to_px(self, lat, lon):
        e = np.array([enu(a, b, self.lat0, self.lon0) for a, b in zip(np.atleast_1d(lat), np.atleast_1d(lon))])
        p = np.c_[e, np.ones(len(e))] @ self.inv.T
        return p[:, 0], p[:, 1]                              # samp (x), line (y)

    def pixel_to_latlon(self, rows, cols):
        rows, cols = np.asarray(rows, float).ravel(), np.asarray(cols, float).ravel()
        ll = np.array([self.nac.latlon(r, c) for r, c in zip(rows, cols)])
        return ll[:, 0], ll[:, 1]


class Shifted:
    """Ground model for fine-frame px: NAC px offset by the frame origin."""

    def __init__(self, geo, ox, oy):
        self.geo, self.ox, self.oy = geo, ox, oy

    def pixel_to_latlon(self, rows, cols):
        return self.geo.pixel_to_latlon(np.asarray(rows, float) + self.oy, np.asarray(cols, float) + self.ox)


def match(name, a, b, device, gsd):
    def plane(x):
        return synth.ImagePlane(array=x, valid_mask=np.ones(x.shape, bool), shadow_mask=np.zeros(x.shape, bool),
                                gsd_m=gsd, meta=None, geo=None)
    pa, pb = plane(a), plane(b)
    if name == "sift":
        return classical.match(pa, pb, detector="sift"), pa, pb
    return adapter.match(pa, pb, model_name=name, device=device), pa, pb


def run_window(ohrc, om, nacm, geo, nac, rc, cc, matchers, device, stages, dem_tiles):
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
    nx, ny = geo.to_px(lat, lon)
    A_sys = affine_fit(np.c_[gx.ravel(), gy.ravel()], np.c_[nx, ny])
    lat_c, lon_c = (float(v) for v in om.pixel_to_latlon(np.array([r0 + WIN / 2]), np.array([c0 + WIN / 2])))
    out.update(lat=round(lat_c, 4), lon=round(lon_c, 4))

    # 2. coarse lock over +-MARGIN at ~COARSE_M
    L, Sn = nacm.array_shape
    cx, cy = (A_sys @ [WIN / 2, WIN / 2, 1])[:2]
    half = int(MARGIN_M / nac.px_h) + WIN
    l0, l1 = max(0, int(cy) - half), min(L, int(cy) + half)
    if l1 - l0 < WIN:
        out["status"] = "skipped: predicted position outside the NAC strip"; return out
    reg_raw = pds_raster.read_raster(nacm, pds_raster.Window(l0, 0, l1 - l0, Sn)).astype(np.float32)
    reg_ok = reg_raw > NULL_BELOW
    b = max(1, round(COARSE_M / min(nac.px_w, nac.px_h)))
    f = max(1, round(COARSE_M / OHRC_PX))
    reg_small = cascade.block_average(norm(reg_raw, reg_ok), b)
    D_ref = cascade.downsample_transform(b) @ T(0, -l0)                  # NAC px -> region-small px
    U_src = np.linalg.inv(cascade.downsample_transform(f))              # OHRC-small px -> OHRC px
    prior = D_ref @ A_sys @ U_src
    diag = {}
    step = cascade.register_step_dense(ohrc_n, reg_small, f, src="OHRC", ref=nac.pid,
                                       ref_pixel_m=COARSE_M, prior=prior, diag=diag)
    out["coarse"] = {"z": diag.get("z"), "block_ohrc": f, "block_nac": b, **({"failed": diag["failed"]} if "failed" in diag else {})}
    if step is None or (diag.get("z") or 0) < MIN_Z:
        out["status"] = "no coarse lock"; out["results"] = {}; return out
    T_c = np.linalg.inv(D_ref) @ np.asarray(step.model.matrix, float)    # OHRC window px -> NAC px

    # implied OHRC-system offset of the window centre, metres east/north
    px, py = (T_c @ [WIN / 2, WIN / 2, 1])[:2]
    la_n, lo_n = geo.pixel_to_latlon([py], [px])
    e = enu(float(la_n[0]), float(lo_n[0]), lat_c, lon_c)
    out["system_offset_m"] = {"east": round(float(e[0]), 1), "north": round(float(e[1]), 1)}

    # 3. fine frame on the NAC grid
    cor = np.array([[0, 0, 1], [WIN, 0, 1], [0, WIN, 1], [WIN, WIN, 1]], float) @ T_c.T
    o = np.floor(cor[:, :2].min(0)).astype(int) + 8
    e_ = np.ceil(cor[:, :2].max(0)).astype(int) - 8
    o[0], o[1] = max(o[0], 0), max(o[1], 0)
    e_[0], e_[1] = min(e_[0], Sn), min(e_[1], L)
    wF, hF = int(e_[0] - o[0]), int(e_[1] - o[1])
    Wf = T(-o[0], -o[1]) @ T_c
    src = warp_affine(ohrc_n, Wf, (wF, hF))
    src_ok = cv2.warpAffine(o_ok.astype(np.float32), Wf[:2], (wF, hF), flags=cv2.INTER_NEAREST) > 0.5
    ref_raw = pds_raster.read_raster(nacm, pds_raster.Window(int(o[1]), int(o[0]), hF, wF)).astype(np.float32)
    ref_ok = ref_raw > NULL_BELOW
    ref = norm(ref_raw, ref_ok)
    out["fine_frame_px"] = [wF, hF]

    dem = None
    if stages["geometry_filter"] and dem_tiles:
        m = 0.02
        dem = dem_patch(dem_tiles, (lat_c - m, lat_c + m, lon_c - m, lon_c + m))
    exp = scale.expected_scale(scale.pixel_scale(ohrc),
                               PixelScale(nac.pid, nac.px_w, nac.px_h, (nac.px_w, nac.px_w), (nac.px_h, nac.px_h),
                                          ("lroc_scaled_pixel",), False, ("LROC scaled pixel size",)))
    gsd = max(nac.px_w, nac.px_h)
    results = {}
    for name in matchers:
        t0 = time.perf_counter()
        r = {}
        try:
            ms, pa, pb = match(name, src, ref, device, gsd)
            n = int(len(ms.src_pts))
            if n < 4:
                results[name] = {"status": f"only {n} matches", "matches": n, "success": False}; continue
            fr = fine_stage(ms, src, ref, centre=(wF / 2, hF / 2), flags=stages,
                            ground_model=Shifted(geo, o[0], o[1]), dem=dem)
            r.update(matches=n, inliers=fr.inlier_count, inlier_ratio=round(fr.inlier_ratio, 4),
                     control_points=int(len(fr.control_src)), coverage=round(fr.coverage, 3),
                     inlier_rmse_px=None if fr.rmse_px is None else round(fr.rmse_px, 3), pipeline=fr.stages)
            if not fr.ok:
                results[name] = {**r, "status": "fine stage: no transform", "success": False}; continue
            T_total = T(o[0], o[1]) @ np.asarray(fr.model.matrix, float) @ Wf
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
        except Exception as exc:                                        # recorded, never hidden
            r.update(status=f"error: {type(exc).__name__}: {exc}"[:300], success=False)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = r
    out["results"] = results
    out["status"] = "locked"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS))
    ap.add_argument("--matchers", nargs="+", default=["eloftr", "minima-loftr", "sift"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/ohrc_nac_registration.json")
    args = ap.parse_args()

    ohrc = parse_label(next((ROOT / "data/raw/ch2/ohrc").rglob("*_d_img_d18.xml")))
    om = projection.load_grid_model(ohrc)
    assert om.independent_of_references
    lroc = json.loads((ROOT / "data/pairs/tmc2_nac_lroc_meta.json").read_text(encoding="utf-8"))["products"]
    sun = json.loads((ROOT / "data/pairs/nac_sun_azimuth_computed.json").read_text(encoding="utf-8"))
    stages = stage_flags()
    dem_tiles = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")
    L, S = ohrc.array_shape
    rows_all = np.arange(WIN, L - WIN, 250)
    # Amendment 1: candidate columns every 256 px; per row, the one closest to the centre
    # whose window lies inside the NAC footprint.
    cols_all = np.arange(WIN // 2, S - WIN // 2 + 1, 256)
    cols_all = cols_all[np.argsort(np.abs(cols_all - S / 2.0), kind="stable")]

    windows = []
    for pid in args.products:
        nac = Nac(pid, lroc[pid])
        nacm = parse_label(next((ROOT / "data/raw/lro/nac").rglob(f"{pid}.XML")))
        geo = NacGeo(nac, *nacm.array_shape)
        # Amendment 2: the window's HALF-extent in NAC px, plus 10% (was the full extent,
        # which excluded every window on a strip narrower than two windows).
        pad = 1.1 * (WIN / 2) * OHRC_PX / min(nac.px_w, nac.px_h)
        Ln, Sn = nacm.array_shape
        best = {}
        for cc in cols_all:                                   # nearest-to-centre first
            la, lo = om.pixel_to_latlon(rows_all, np.full(rows_all.shape, float(cc)))
            x, y = geo.to_px(la, lo)
            ok = (x > pad) & (x < Sn - pad) & (y > pad) & (y < Ln - pad)
            for rc in rows_all[ok]:
                best.setdefault(int(rc), int(cc))
        inside = np.array(sorted(best))
        print(f"{pid}: {len(inside)} candidate OHRC rows inside the footprint", flush=True)
        if len(inside) == 0:
            continue
        picks = np.linspace(inside.min(), inside.max(), N_WIN).astype(int) if len(inside) >= N_WIN else inside
        picks = [int(inside[np.argmin(np.abs(inside - p))]) for p in picks]
        for rc in picks:
            t = time.perf_counter()
            w = run_window(ohrc, om, nacm, geo, nac, rc, best[rc], args.matchers, args.device, stages, dem_tiles)
            w["sun"] = sun.get(pid)
            w["seconds"] = round(time.perf_counter() - t, 1)
            windows.append(w)
            print(json.dumps({k: w[k] for k in ("nac", "ohrc_row", "status", "coarse") if k in w} |
                             {"ok": {m: (v.get("status"), v.get("tier"), v.get("known_shift_error_px"))
                                     for m, v in (w.get("results") or {}).items()}}), flush=True)

    # consistency (frozen): offset within CONSISTENT_M of the product's median locked offset
    for pid in args.products:
        ws = [w for w in windows if w["nac"] == pid and "system_offset_m" in w]
        if not ws:
            continue
        V = np.array([[w["system_offset_m"]["east"], w["system_offset_m"]["north"]] for w in ws])
        med = np.median(V, axis=0)
        for w, v in zip(ws, V):
            w["offset_from_product_median_m"] = round(float(np.hypot(*(v - med))), 1)
            w["consistent"] = bool(np.hypot(*(v - med)) <= CONSISTENT_M)

    summary = {}
    for pid in args.products:
        ws = [w for w in windows if w["nac"] == pid]
        for name in args.matchers:
            ok = 0
            for w in ws:
                r = (w.get("results") or {}).get(name, {})
                s = (r.get("status") == "registered" and r.get("gates_pass") and r.get("tier_ok")
                     and w.get("consistent", False))
                r["success"] = bool(s)
                ok += bool(s)
            rate = ok / len(ws) if ws else 0.0
            summary.setdefault(pid, {})[name] = {"success": ok, "windows": len(ws),
                                                 "verdict": "solved" if rate >= 0.9 else "degraded" if rate >= 0.6 else "unsolved"}
    print(json.dumps(summary, indent=1))
    out = ROOT / args.out
    out.write_text(json.dumps({"source": "measured", "protocol": "docs/ohrc_nac_protocol.md",
                               "ohrc": ohrc.product_id, "pipeline_stages": stages, "run": run_record(),
                               "summary": summary, "windows": windows}, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
