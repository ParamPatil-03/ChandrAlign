"""Known-warp harness: the true error of DELIVERED points and geometry on real CH-2 texture.

Rebuilt from the 2026-09-26 audit's scratch harness (docs/AUDIT_2026-09-26.md, C-02 / C-03 /
I-01, section 9) so the evidence is in the repository, not in a temporary folder.
Protocol: docs/refinement_geometry_protocol.md (frozen before any measurement here).

TRUTH CONSTRUCTION (identical to the audit's)
A real high-resolution crop H (TMC-2 native, block 2; OHRC native, block 4). The source is
block-average(H, b). The reference is block-average(remap(H, G, Lanczos-4), b), where G is a
KNOWN reference -> source map in coarse pixels: a sub-pixel shift (0.37, -0.62), a rotation,
a scale and (non-rigid cases) a smooth sinusoidal field of 1.2 px amplitude, 300-390 px
wavelength. Interpolation happens at the fine level and is then integrated over b x b, so no
estimator shares the interpolation kernel. Independent sensor-like noise is added to both
images at the crop's native noise ratio. The truth for any source point is G^-1, solved by
fixed-point iteration.

CONDITIONS
    small warp: 2.5 deg, scale 0.93      large warp: 10 deg, scale 0.60
    same      identical radiometry
    mild      gamma 0.6 and a 0.75-1.25 gain ramp (exposure / calibration difference)
    inverted  shading detail sign-flipped, low-pass kept (an opposite-sun proxy)

The scored 15 cases (the audit's): 3 scenes x {small same non-rigid, small same rigid, small
mild non-rigid, large same non-rigid, large mild non-rigid}. `inverted` is run and recorded,
not scored (in the audit nothing registered or refined on it).

ARMS (point refinement, C-02 / G-01), all on the SAME delivered points of one fine stage:
    unrefined     the matcher's positions
    legacy        the refinement shipped until 2026-09-26: axis-aligned patches of the raw
                  images, ncc_gaussian_iter, max_move 1.5 (inlined here so it stays measurable)
    library       subpixel.refine_points as it is NOW, given the model (the product path)

    .venv/Scripts/python scripts/known_warp_harness.py --set dev --out reports/known_warp_dev.json
    .venv/Scripts/python scripts/known_warp_harness.py --set heldout --out reports/known_warp_heldout.json
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
from scipy.spatial import Delaunay  # noqa: E402

from chandralign.contracts import ImagePlane  # noqa: E402
from chandralign.estimate import models  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.matching import classical  # noqa: E402
from chandralign.pipeline import fine_stage  # noqa: E402
from chandralign.refine import subpixel  # noqa: E402
from verify_subpixel import native_noise_ratio  # noqa: E402

N = 1024                                  # coarse image side
SHIFT = np.array([0.37, -0.62])
AMP, LAM = 1.2, 300.0

# name -> (instrument, row, col, block, gsd). Fixed by docs/refinement_geometry_protocol.md.
SETS = {
    "dev": {"scenes": [("tmc2_r60000", "tmc2", 60000, 1000, 2, 10.0),
                       ("tmc2_r110000", "tmc2", 110000, 1000, 2, 10.0),
                       ("ohrc_r30000", "ohrc", 30000, 4000, 4, 1.2)], "seed": 7},
    "heldout": {"scenes": [("tmc2_r30000", "tmc2", 30000, 1000, 2, 10.0),
                           ("tmc2_r90000", "tmc2", 90000, 1000, 2, 10.0),
                           ("ohrc_r50000", "ohrc", 50000, 4000, 4, 1.2)], "seed": 11},
    # protocol amendment 1: fixed after the held-out set was spent on the dev choice
    "fresh": {"scenes": [("tmc2_r140000", "tmc2", 140000, 1000, 2, 10.0),
                         ("tmc2_r10000", "tmc2", 10000, 1000, 2, 10.0),
                         ("ohrc_r10000", "ohrc", 10000, 4000, 4, 1.2)], "seed": 13},
}
WARPS = {"small": (2.5, 0.93), "large": (10.0, 0.60)}
# (warp, radiometry, non-rigid, scored)
CASES = [("small", "same", True, True), ("small", "same", False, True), ("small", "mild", True, True),
         ("large", "same", True, True), ("large", "mild", True, True), ("small", "inverted", True, False)]


class Warp:
    """The known map G: reference coarse px -> source coarse px, and its inverse."""

    def __init__(self, theta_deg: float, scale: float, nonrigid: bool):
        self.t, self.s, self.nonrigid = np.deg2rad(theta_deg), scale, nonrigid
        self.R = np.array([[np.cos(self.t), -np.sin(self.t)], [np.sin(self.t), np.cos(self.t)]])
        self.c = np.array([N / 2, N / 2])

    def G(self, r):
        r = np.asarray(r, float)
        s = (r - self.c - SHIFT) @ (self.R / self.s).T + self.c
        if self.nonrigid:
            s = s + AMP * np.c_[np.sin(2 * np.pi * r[:, 1] / LAM + 0.3),
                                np.cos(2 * np.pi * r[:, 0] / (1.3 * LAM) + 1.1)]
        return s

    def truth(self, s):
        """Reference position of source points: solve G(r) = s."""
        s = np.asarray(s, float).reshape(-1, 2)
        Ainv = np.linalg.inv(self.R / self.s)
        r = (s - self.c) @ Ainv.T + self.c + SHIFT
        for _ in range(60):
            r = r + (s - self.G(r)) @ Ainv.T
        return r


def _norm(a, v):
    lo, hi = np.percentile(a[v], [1, 99])
    out = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)
    out[~v] = 0
    return out


def build_pair(H, b, warp: Warp, cond: str, rng, ratio):
    src = H[: N * b, : N * b].reshape(N, b, N, b).mean(axis=(1, 3))
    yy, xx = np.mgrid[0: N * b, 0: N * b].astype(np.float64)
    rc = np.c_[(xx.ravel() + 0.5) / b - 0.5, (yy.ravel() + 0.5) / b - 0.5]
    sc = warp.G(rc)
    mx = (b * (sc[:, 0] + 0.5) - 0.5).reshape(N * b, N * b).astype(np.float32)
    my = (b * (sc[:, 1] + 0.5) - 0.5).reshape(N * b, N * b).astype(np.float32)
    Hw = cv2.remap(H.astype(np.float32), mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
                   borderValue=np.nan)
    valid_hr = np.isfinite(Hw)
    Hw = np.where(valid_hr, Hw, 0)
    if cond == "mild":
        lo, hi = np.percentile(H, [0.5, 99.5])
        Hw = np.clip((Hw - lo) / (hi - lo), 0, 1) ** 0.6 * np.linspace(0.75, 1.25, Hw.shape[1])[None, :]
    elif cond == "inverted":
        lp = cv2.GaussianBlur(Hw, (0, 0), 6.0 * b)
        Hw = lp - 0.7 * (Hw - lp)
    ref = Hw.reshape(N, b, N, b).mean(axis=(1, 3))
    vref = valid_hr.reshape(N, b, N, b).all(axis=(1, 3))
    src = src + rng.normal(0, ratio * src.std(), src.shape)
    ref = ref + rng.normal(0, ratio * ref[vref].std(), ref.shape)
    return _norm(src, np.ones(src.shape, bool)), _norm(ref, vref), vref


def plane(a, v, gsd):
    return ImagePlane(array=a, valid_mask=v, shadow_mask=np.zeros(a.shape, bool), gsd_m=gsd, meta=None)


def stats(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "p50": round(float(np.median(e)), 4), "p95": round(float(np.percentile(e, 95)), 4),
            "rms": round(float(np.sqrt(np.mean(e ** 2))), 4), "max": round(float(e.max()), 4)}


def legacy_refine(src_img, ref_img, src_pts, ref_pts, half=16, max_move=1.5):
    """The per-point refinement shipped until 2026-09-26 (refine/subpixel.py, pre-C-02), verbatim."""
    out = np.asarray(ref_pts, float).copy()
    moved = np.zeros(len(out), bool)
    size = 2 * half + 1
    s32, r32 = np.asarray(src_img, np.float32), np.asarray(ref_img, np.float32)
    for i, (s, r) in enumerate(zip(np.asarray(src_pts, float), out)):
        sp = cv2.getRectSubPix(s32, (size, size), (float(s[0]), float(s[1])))
        rp = cv2.getRectSubPix(r32, (size, size), (float(r[0]), float(r[1])))
        est = subpixel.estimate(sp, rp, "ncc_gaussian_iter")
        if est.ok and np.all(np.isfinite(est.d)) and np.hypot(est.dx, est.dy) <= max_move:
            out[i] = r + est.d
            moved[i] = True
    return out, moved


def load_scene(inst, row, col, b):
    lab = parse_label(next((ROOT / f"data/raw/ch2/{inst}").rglob("*_d_img_d18.xml")))
    return pds_raster.read_raster(lab, pds_raster.Window(row, col, N * b, N * b)).astype(np.float64)


def run_case(scene, H, b, gsd, warp_name, cond, nonrigid, seed, arms, geometry=False):
    theta, scale = WARPS[warp_name]
    warp = Warp(theta, scale, nonrigid)
    rng = np.random.default_rng(seed)
    src, ref, vref = build_pair(H, b, warp, cond, rng, native_noise_ratio(H))
    rec = {"scene": scene, "warp": warp_name, "cond": cond, "nonrigid": nonrigid, "seed": seed}
    t0 = time.perf_counter()
    ms = classical.match(plane(src, np.ones(src.shape, bool), gsd), plane(ref, vref, gsd), detector="sift")
    rec["matches"] = int(len(ms.src_pts))
    off = fine_stage(ms, src, ref, centre=(N / 2, N / 2), flags={"subpixel": False, "tps": False})
    if not off.ok or len(off.control_src) < 10:
        rec["status"] = "no registration"
        return rec
    cs, cr0 = off.control_src, off.control_ref
    tr = warp.truth(cs)
    rec.update(status="ok", inliers=off.inlier_count, delivered=int(len(cs)))
    pts = {"unrefined": stats(np.hypot(*(cr0 - tr).T))}
    for name, fn in arms.items():
        t1 = time.perf_counter()
        rr, mv = fn(src, ref, cs, cr0, off.model)
        pts[name] = {**stats(np.hypot(*(rr - tr).T)), "moved": int(mv.sum()),
                     "seconds": round(time.perf_counter() - t1, 2)}
    rec["points"] = pts
    if geometry:
        rec["geometry"] = geometry_eval(ms, src, ref, warp, off)
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def dense_grid(warp, hull_pts=None):
    gy, gx = np.mgrid[16:N - 16:16, 16:N - 16:16]
    gs = np.c_[gx.ravel(), gy.ravel()].astype(float)
    gt = warp.truth(gs)
    inside = (gt[:, 0] > 16) & (gt[:, 0] < N - 16) & (gt[:, 1] > 16) & (gt[:, 1] < N - 16)
    if hull_pts is not None:
        inside &= Delaunay(hull_pts).find_simplex(gs) >= 0
    return gs[inside], gt[inside]


def geometry_eval(ms, src, ref, warp, off):
    """C-03 / I-01 / G-02: the delivered geometry's TRUE dense-grid error vs what is REPORTED.

    Runs the shipped fine stage (current config) and scores the geometry the product would
    warp with, inside the convex hull of the inliers (where any model is supported by data).
    """
    from chandralign.pipeline import delivered_geometry
    on = fine_stage(ms, src, ref, centre=(N / 2, N / 2))
    inl_src = on.matches.src_pts[on.first.inlier_mask]
    gs, gt = dense_grid(warp, inl_src)
    out = {"n_grid": int(len(gs))}
    cand = {"affine": on.model}
    if on.tps is not None:
        cand["tps"] = on.tps
    for k, m in cand.items():
        out[f"true_{k}"] = stats(np.hypot(*(models.apply(m, gs) - gt).T))
    name, model = delivered_geometry(on)
    out["delivered"] = name
    out["true_delivered"] = stats(np.hypot(*(models.apply(model, gs) - gt).T))
    out["reported"] = {k: on.stages.get("accuracy", {}).get(k) for k in
                       ("checkpoint_rmse_px_ref", "fit_residual_px", "model", "n_check")}
    out["legacy_reported_rmse_px"] = on.rmse_px
    out["selection"] = on.stages.get("model_selection")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), default="dev")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these scene names")
    ap.add_argument("--geometry", action="store_true", help="also score the delivered geometry (C-03/I-01)")
    ap.add_argument("--no-legacy", action="store_true")
    args = ap.parse_args()

    arms = {"library": lambda s, r, cs, cr, m: subpixel.refine_points(s, r, cs, cr, model=m)}
    if not args.no_legacy:
        arms = {"legacy": lambda s, r, cs, cr, m: legacy_refine(s, r, cs, cr), **arms}
    spec = SETS[args.set]
    recs = []
    for scene, inst, row, col, b, gsd in spec["scenes"]:
        if args.only and scene not in args.only:
            continue
        H = load_scene(inst, row, col, b)
        for warp_name, cond, nr, scored in CASES:
            rec = run_case(scene, H, b, gsd, warp_name, cond, nr, spec["seed"], arms, geometry=args.geometry)
            rec["scored"] = scored
            recs.append(rec)
            p = rec.get("points", {})
            print(scene, warp_name, cond, "nonrigid" if nr else "rigid", rec.get("status"),
                  {k: (v.get("p50"), v.get("p95")) for k, v in p.items()},
                  rec.get("geometry", {}).get("true_delivered"), rec.get("geometry", {}).get("reported"), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"set": args.set, "seed": spec["seed"], "cases": recs}, indent=1))
    print(verdict(recs))


def verdict(recs):
    """C-02's bar: refined beats unrefined at p50 AND p95, and p95 <= 0.25 ref px, on every scored case."""
    scored = [r for r in recs if r.get("scored") and r.get("status") == "ok"]
    lines = []
    for arm in ("legacy", "library"):
        if not scored or arm not in scored[0]["points"]:
            continue
        beat = sum(r["points"][arm]["p50"] < r["points"]["unrefined"]["p50"]
                   and r["points"][arm]["p95"] < r["points"]["unrefined"]["p95"] for r in scored)
        tight = sum(r["points"][arm]["p95"] <= 0.25 for r in scored)
        lines.append(f"{arm}: beats unrefined at p50 and p95 in {beat}/{len(scored)}; p95 <= 0.25 in {tight}/{len(scored)}")
    missing = sum(1 for r in recs if r.get("scored") and r.get("status") != "ok")
    if missing:
        lines.append(f"{missing} scored case(s) did not register")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
