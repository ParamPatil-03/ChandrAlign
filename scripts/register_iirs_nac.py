"""IIRS -> LRO NAC on real data: the open, 160x scale-gap pairing.

    python scripts/register_iirs_nac.py

Protocol, frozen before this script existed: docs/iirs_nac_protocol.md.

A NAC strip (~5 km) is ~60 IIRS pixels across, so the NAC is brought to IIRS scale: the
coarse lock searches the NAC crop, block-averaged to the IIRS pixel size, over the IIRS
composite (MIND); the fine stage then matches the IIRS region (source) against the NAC
warped onto that region's grid (reference), followed by pipeline.fine_stage, all five
control gates and quality.assess. The prior corrects IIRS's ~12.8 km system error by our
measured IIRS -> WAC offset (the geodetic bridge); position is still searched.
"""
from __future__ import annotations

import argparse
import json
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

from chandralign import config  # noqa: E402
from chandralign.estimate import scale  # noqa: E402
from chandralign.estimate.scale import PixelScale  # noqa: E402
from chandralign.evaluate import control_gates, quality  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.dem import dem_patch, find_tiles  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.matching import cascade, routing  # noqa: E402
from chandralign.pipeline import fine_stage, stage_flags  # noqa: E402
from chandralign.preprocess.iirs_composite import iirs_composite_plane, product_band_selection  # noqa: E402
from chandralign.preprocess.resample import warp_affine  # noqa: E402
from register_ohrc_nac import T, affine_fit, match, norm  # noqa: E402
from register_tmc2_nac import MOON_R_M, NULL_BELOW, enu  # noqa: E402

# ---- frozen in docs/iirs_nac_protocol.md -------------------------------------------
PRODUCTS = ("M1415013176LC", "M172765160RC", "M1417360906LC")
N_WIN = 3
PAD_LINES = 60                      # IIRS lines searched either side of the NAC crop (~5 km)
CONSISTENT_M = 350.0
MIN_Z = float(config.get("cascade.min_z", 10.0))
WAC_WINDOW_LINES = 512              # docs/iirs_wac_protocol.md: offsets are at line0 + 256
K = np.pi / 180 * MOON_R_M          # metres per degree of latitude


def lroc_meta() -> dict:
    out = {}
    for f in ("tmc2_nac_lroc_meta.json", "iirs_nac_lroc_meta.json"):
        out.update(json.loads((ROOT / "data/pairs" / f).read_text(encoding="utf-8"))["products"])
    return out


class NacCorners:
    """NAC (line, sample) -> lat/lon, bilinear in LROC's corners (0.01 deg), with this
    product's own line/sample count (register_tmc2_nac.Nac assumes 52224 x 5064)."""

    def __init__(self, pid: str, meta: dict, lines: int, samples: int):
        self.pid, self.lines, self.samples = pid, lines, samples
        self.px_w, self.px_h = float(meta["scaled_pixel_width"]), float(meta["scaled_pixel_height"])
        self.c = {k: (float(meta[f"{k}_latitude"]), float(meta[f"{k}_longitude"]))
                  for k in ("upper_left", "upper_right", "lower_left", "lower_right")}

    def latlon(self, line, samp):
        u = np.asarray(samp, float) / (self.samples - 1)
        v = np.asarray(line, float) / (self.lines - 1)
        w = {"upper_left": (1 - u) * (1 - v), "upper_right": u * (1 - v),
             "lower_left": (1 - u) * v, "lower_right": u * v}
        return (sum(w[k] * self.c[k][0] for k in w), sum(w[k] * self.c[k][1] for k in w))


class Bridge:
    """IIRS -> WAC offset (east, north metres; true minus system), linear in IIRS line."""

    def __init__(self):
        rep = json.loads((ROOT / "reports/iirs_wac_mosaic.json").read_text(encoding="utf-8"))
        pts = [(w["iirs_line0"] + WAC_WINDOW_LINES / 2, w["results"]["xoftr"]["implied_offset_m"])
               for w in rep["windows"] if (w.get("results") or {}).get("xoftr", {}).get("success")]
        line = np.array([p[0] for p in pts], float)
        self.fe = np.polyfit(line, [p[1]["east"] for p in pts], 1)
        self.fn = np.polyfit(line, [p[1]["north"] for p in pts], 1)
        self.lines = (float(line.min()), float(line.max()))

    def at(self, line):
        return np.polyval(self.fe, line), np.polyval(self.fn, line)


class IirsGround:
    """IIRS pixel -> lat/lon: system corners plus the bridge offset at that line."""

    def __init__(self, im, bridge: Bridge, oy: float = 0.0, ox: float = 0.0):
        self.im, self.b, self.oy, self.ox = im, bridge, oy, ox

    def pixel_to_latlon(self, rows, cols):
        rows = np.asarray(rows, float).ravel() + self.oy
        cols = np.asarray(cols, float).ravel() + self.ox
        lat, lon = self.im.pixel_to_latlon(rows, cols)
        e, n = self.b.at(rows)
        lat = np.asarray(lat, float) + n / K
        return lat, np.asarray(lon, float) + e / (K * np.cos(np.radians(lat)))

    def latlon_to_pixel(self, lat, lon):
        """Corrected ground -> IIRS (row, col): undo the offset at the row, twice (it varies slowly)."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        rows, _ = self.im.latlon_to_pixel(lat, lon)
        for _ in range(2):
            e, n = self.b.at(np.asarray(rows, float))
            s_lat = lat - n / K
            rows, cols = self.im.latlon_to_pixel(s_lat, lon - e / (K * np.cos(np.radians(lat))))
        return np.asarray(rows, float), np.asarray(cols, float)


def nac_pixel_scale(nac: NacCorners) -> PixelScale:
    return PixelScale(nac.pid, nac.px_w, nac.px_h, (nac.px_w, nac.px_w), (nac.px_h, nac.px_h),
                      ("lroc_scaled_pixel",), False, ("LROC scaled pixel size",))


def dense_checks(nimg, ref, f, prior, pid, iirs_px, c_mid, locked):
    """Amendment 1: the control gates' logic applied to the dense step itself."""
    def run(r):
        dg = {}
        st = cascade.register_step_dense(nimg, r, f, src=pid, ref="IIRS", ref_pixel_m=iirs_px, prior=prior, diag=dg)
        return st, float(dg.get("z") or 0.0)
    shifted = cv2.warpAffine(ref, np.float32([[1, 0, 3], [0, 1, 4]]), ref.shape[::-1],
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    st_s, z_s = run(shifted)
    moved = None if st_s is None or z_s < MIN_Z else (np.asarray(st_s.model.matrix, float) @ c_mid)[:2] - locked
    ks_err = None if moved is None else float(np.hypot(moved[0] - 3.0, moved[1] - 4.0))
    st_c, z_c = run(np.full_like(ref, float(ref.mean())))
    st_n, z_n = run(np.random.default_rng(0).normal(0, 1, ref.shape).astype(np.float32))
    out = {"known_shift": {"z": round(z_s, 1), "moved_px": None if moved is None else [round(float(v), 3) for v in moved],
                           "error_px": None if ks_err is None else round(ks_err, 3), "pass": ks_err is not None and ks_err <= 0.5},
           "null_constant": {"z": round(z_c, 1), "pass": st_c is None or z_c < MIN_Z},
           "null_noise": {"z": round(z_n, 1), "pass": st_n is None or z_n < MIN_Z}}
    out["all_pass"] = all(v["pass"] for v in out.values())
    return out


def run_window(iirs, im, sel, bridge, nacm, nac, l0, l1, matchers, device, stages, dem_tiles, route_opts,
               dense: bool = False):
    out = {"nac": nac.pid, "nac_lines": [int(l0), int(l1)]}
    L, S = iirs.array_shape
    Sn = nacm.array_shape[1]
    ground = IirsGround(im, bridge)
    raw = pds_raster.read_raster(nacm, pds_raster.Window(int(l0), 0, int(l1 - l0), Sn)).astype(np.float32)
    ok = raw > NULL_BELOW
    nimg = norm(raw, ok)
    del raw
    # NAC crop -> IIRS lines it covers (corrected geolocation), +- PAD_LINES
    g_l, g_s = np.meshgrid(np.linspace(0, l1 - l0 - 1, 7), np.linspace(0, Sn - 1, 5), indexing="ij")
    lat, lon = nac.latlon(g_l.ravel() + l0, g_s.ravel())
    rr, cc = ground.latlon_to_pixel(lat, lon)
    r0 = int(max(0, np.floor(rr.min()) - PAD_LINES))
    r1 = int(min(L, np.ceil(rr.max()) + PAD_LINES))
    if r1 - r0 < 32:
        out["status"] = "skipped: NAC crop outside the IIRS strip"; return out
    plane = iirs_composite_plane(iirs, pds_raster.Window(r0, 0, r1 - r0, S), sel)
    ref, ref_ok = plane.array.astype(np.float32), plane.valid_mask
    out["iirs_lines"] = [r0, r1]
    iirs_px = float(iirs.gsd_m)
    A = affine_fit(np.c_[g_s.ravel(), g_l.ravel()], np.c_[cc, rr - r0])      # NAC crop px -> region px
    f = max(1, round(iirs_px / max(nac.px_w, nac.px_h)))
    prior = A @ np.linalg.inv(cascade.downsample_transform(f))
    diag = {}
    st = cascade.register_step_dense(nimg, ref, f, src=nac.pid, ref="IIRS", ref_pixel_m=iirs_px,
                                     prior=prior, diag=diag)
    z = float(diag.get("z") or 0.0)
    out["coarse"] = {"z": round(z, 1), "factor": f}
    if st is None or z < MIN_Z:
        out["status"] = "no coarse lock"; out["coarse"]["why"] = diag.get("failed"); out["results"] = {}
        return out
    T_c = np.asarray(st.model.matrix, float)                                  # NAC crop px -> region px
    c_mid = np.array([Sn / 2, (l1 - l0) / 2, 1.0])
    out["coarse"]["prior_error_iirs_px"] = round(float(np.hypot(*((T_c - A) @ c_mid)[:2])), 2)
    if dense:                                              # amendment 1: the dense lock is the answer
        locked = (T_c @ c_mid)[:2]
        la_n, lo_n = (float(v) for v in nac.latlon(l0 + (l1 - l0) / 2, Sn / 2))
        la_s, lo_s = (float(v[0]) for v in im.pixel_to_latlon([r0 + locked[1]], [locked[0]]))
        ef = enu(la_n, lo_n, la_s, lo_s)
        be, bn = bridge.at(r0 + locked[1])
        chk = dense_checks(nimg, ref, f, prior, nac.pid, iirs_px, c_mid, locked)
        vs = float(np.hypot(ef[0] - be, ef[1] - bn))
        out["dense"] = {"checks": chk, "precision_iirs_px": round(float(st.rmse_px), 4),
                        "implied_offset_m": {"east": round(float(ef[0]), 1), "north": round(float(ef[1]), 1)},
                        "bridge_offset_m": {"east": round(float(be), 1), "north": round(float(bn), 1)},
                        "vs_bridge_m": round(vs, 1), "success": bool(chk["all_pass"] and vs <= CONSISTENT_M)}
        out["status"] = "locked"; out["results"] = {}
        return out
    cor = np.array([[0, 0, 1], [Sn, 0, 1], [0, l1 - l0, 1], [Sn, l1 - l0, 1]], float) @ T_c.T
    o = np.maximum(np.floor(cor[:, :2].min(0)).astype(int) + 2, 0)
    e_ = np.minimum(np.ceil(cor[:, :2].max(0)).astype(int) - 2, [S, r1 - r0])
    wF, hF = int(e_[0] - o[0]), int(e_[1] - o[1])
    if wF < 32 or hF < 32:
        out["status"] = "skipped: fine frame too small"; out["results"] = {}; return out
    fsrc = ref[o[1]:o[1] + hF, o[0]:o[0] + wF].copy()                         # IIRS region (source)
    if ref_ok[o[1]:o[1] + hF, o[0]:o[0] + wF].mean() < 0.9:
        out["status"] = "skipped: IIRS fine frame has invalid pixels"; out["results"] = {}; return out
    Wf = T(-o[0], -o[1]) @ T_c                                                 # NAC crop px -> frame px
    Ws = Wf @ np.linalg.inv(cascade.downsample_transform(f))
    fref = warp_affine(cascade.block_average(nimg, f), Ws, (wF, hF))           # NAC at IIRS scale (reference)
    fok = cv2.warpAffine(cascade.block_average(ok.astype(np.float32), f), Ws[:2], (wF, hF),
                         flags=cv2.INTER_LINEAR, borderValue=0.0) > 0.99
    fref = norm(fref, fok)
    out["fine_frame_px"] = [wF, hF]
    out["nac_valid_in_frame"] = round(float(fok.mean()), 3)
    lat_c, lon_c = (float(v[0]) for v in ground.pixel_to_latlon([r0 + o[1] + hF / 2], [o[0] + wF / 2]))
    dem = None
    if stages["geometry_filter"] and dem_tiles:
        dem = dem_patch(dem_tiles, (lat_c - 0.3, lat_c + 0.3, lon_c - 0.3, lon_c + 0.3))
    exp = scale.expected_scale(scale.pixel_scale(iirs), nac_pixel_scale(nac))
    results = {}
    for name in matchers:
        t0 = time.perf_counter(); r = {}
        opts = route_opts if name == "xoftr" else {}
        try:
            if opts:
                from chandralign import synth
                from chandralign.matching import adapter
                mk = lambda a: synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool),  # noqa: E731
                                                shadow_mask=np.zeros(a.shape, bool), gsd_m=iirs_px, meta=None, geo=None)
                pa, pb = mk(fsrc), mk(fref); ms = adapter.match(pa, pb, model_name=name, device=device, **opts)
            else:
                ms, pa, pb = match(name, fsrc, fref, device, iirs_px)
            n = int(len(ms.src_pts))
            if n < 4:
                results[name] = {"status": f"only {n} matches", "success": False}; continue
            fr = fine_stage(ms, fsrc, fref, centre=(wF / 2, hF / 2), flags=stages,
                            ground_model=IirsGround(im, bridge, oy=r0 + o[1], ox=o[0]), dem=dem)
            r.update(matches=n, inliers=fr.inlier_count, inlier_ratio=round(fr.inlier_ratio, 4),
                     coverage=round(fr.coverage, 3), inlier_rmse_px=None if fr.rmse_px is None else round(fr.rmse_px, 3))
            if not fr.ok:
                results[name] = {**r, "status": "fine stage: no transform", "success": False}; continue
            # region px -> frame px -> (fine model) -> frame px -> NAC crop px
            Tt = np.linalg.inv(Wf) @ np.asarray(fr.model.matrix, float) @ T(-o[0], -o[1])
            q = np.array([o[0] + wF / 2, o[1] + hF / 2, 1.0])                  # frame centre, region px
            nx, ny = (Tt @ q)[:2]
            la_n, lo_n = (float(v) for v in nac.latlon(ny + l0, nx))
            la_s, lo_s = (float(v[0]) for v in im.pixel_to_latlon([r0 + q[1]], [q[0]]))
            ef = enu(la_n, lo_n, la_s, lo_s)                                   # NAC ground minus IIRS system
            be, bn = bridge.at(r0 + q[1])
            verdict = scale.check(Tt, exp, centre=(q[0], q[1]))
            gates = control_gates.run_all(control_gates.pipeline_from(name, device=device, gsd_m=iirs_px,
                                                                      stages=stages, **opts), fsrc, fref, pa, pb)
            qa = quality.assess(inlier_count=fr.inlier_count, inlier_ratio=fr.inlier_ratio, spatial_coverage=fr.coverage,
                                model=fr.model, scale_ok=verdict.ok, scale_status=verdict.status,
                                gates=gates.gates, require_gates=True)
            r.update(status="registered", tier=qa.tier, gates=gates.gates, scale_status=verdict.status,
                     known_shift_error_px=gates.to_dict().get("perturbation_sensitivity", {}).get("error_px"),
                     gates_pass=bool(gates.all_passed), tier_ok=qa.tier in ("HIGH", "MEDIUM", "LOW"),
                     implied_offset_m={"east": round(float(ef[0]), 1), "north": round(float(ef[1]), 1)},
                     bridge_offset_m={"east": round(float(be), 1), "north": round(float(bn), 1)},
                     vs_bridge_m=round(float(np.hypot(ef[0] - be, ef[1] - bn)), 1))
            r["success"] = bool(r["gates_pass"] and r["tier_ok"] and r["vs_bridge_m"] <= CONSISTENT_M)
        except Exception as exc:                                                # recorded, never hidden
            r.update(status=f"error: {type(exc).__name__}: {exc}"[:300], success=False)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = r
    out["results"] = results
    out["status"] = "locked"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS))
    ap.add_argument("--matchers", nargs="+", default=["xoftr", "minima-loftr", "sift"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/iirs_nac_registration.json")
    ap.add_argument("--dense", action="store_true",
                    help="amendment 1: the dense lock is the registration, with its own gates (no keypoint stage)")
    args = ap.parse_args()

    iirs = parse_label(next((ROOT / "data/raw/ch2/iirs").rglob("*_d_img_d18.xml")))
    im = projection.load_corner_model(iirs, corners="system")
    assert im.independent_of_references
    sel = product_band_selection(iirs)
    bridge = Bridge()
    ground = IirsGround(im, bridge)
    route = routing.choose("IIRS", "NAC")                 # cascade: the provenance of this pairing
    fine_choice = routing.choose("IIRS", "WAC")           # the 80-100 m regime the fine stage works in
    stages = stage_flags()
    dem_tiles = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")
    meta = lroc_meta()
    L, S = iirs.array_shape
    windows = []
    for pid in args.products:
        nacm = parse_label(next((ROOT / "data/raw/lro/nac").rglob(f"{pid}.XML")))
        Ln, Sn = nacm.array_shape
        nac = NacCorners(pid, meta[pid], Ln, Sn)
        lines = np.arange(0, Ln, 200)
        la, lo = nac.latlon(lines, np.full(lines.shape, (Sn - 1) / 2))
        rr, cc = ground.latlon_to_pixel(la, lo)
        inside = lines[(cc >= 0) & (cc <= S - 1) & (rr >= 0) & (rr <= L - 1)]
        print(f"{pid}: NAC lines {inside.min() if len(inside) else '-'}..{inside.max() if len(inside) else '-'} "
              f"inside the IIRS strip ({len(inside)} of {len(lines)} samples)", flush=True)
        if len(inside) == 0:
            continue
        edges = np.linspace(inside.min(), inside.max(), N_WIN + 1).astype(int)
        for l0, l1 in zip(edges[:-1], edges[1:]):
            t = time.perf_counter()
            w = run_window(iirs, im, sel, bridge, nacm, nac, l0, l1, args.matchers, args.device, stages, dem_tiles,
                           dict(fine_choice.fine_stage_options), dense=args.dense)
            w["seconds"] = round(time.perf_counter() - t, 1)
            windows.append(w)
            print(json.dumps({k: w.get(k) for k in ("nac", "nac_lines", "iirs_lines", "status", "coarse", "dense")} |
                             {"ok": {m: (v.get("status", "")[:20], v.get("tier"), v.get("known_shift_error_px"),
                                         v.get("vs_bridge_m"), v.get("success"))
                                     for m, v in (w.get("results") or {}).items()}}), flush=True)
    summary = {}
    names = ["dense"] if args.dense else args.matchers
    for pid in args.products:
        ws = [w for w in windows if w["nac"] == pid]
        for name in names:
            ok = sum(bool((w.get("dense") or {}).get("success") if name == "dense"
                          else (w.get("results") or {}).get(name, {}).get("success")) for w in ws)
            summary.setdefault(pid, {})[name] = {"success": ok, "windows": len(ws),
                                                 "verdict": "solved" if ws and ok == len(ws) else
                                                 "degraded" if ws and ok >= len(ws) - 1 and ok >= 2 else "unsolved"}
    overall = {name: sum(summary[p][name]["verdict"] == "solved" for p in summary) for name in names}
    print(json.dumps({"summary": summary, "nacs_solved": overall}, indent=1))
    (ROOT / args.out).write_text(json.dumps({"source": "measured", "protocol": "docs/iirs_nac_protocol.md",
                                             "iirs": iirs.product_id, "route": route.as_provenance(),
                                             "fine_route": fine_choice.as_provenance(),
                                             "bridge": {"from": "reports/iirs_wac_mosaic.json (xoftr)",
                                                        "east_fit": list(bridge.fe), "north_fit": list(bridge.fn),
                                                        "lines": bridge.lines},
                                             "pipeline_stages": stages, "run": run_record(),
                                             "summary": summary, "nacs_solved": overall, "windows": windows},
                                            indent=2, default=str), encoding="utf-8")
    print(f"wrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
