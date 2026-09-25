"""Usefulness checks U1-U4 (docs/usefulness_checks.md, criteria frozen before running).

    python scripts/usefulness_checks.py <dump dir of the adopted TMC-2 -> TC run>

Writes reports/usefulness_checks.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.refine import uniformity  # noqa: E402

SHIFTS = (0.0, 0.5, 1.0, 2.0, 5.0)
DIRS = [(np.cos(a), np.sin(a)) for a in np.arange(8) * np.pi / 4]
K_M = np.pi / 180 * 1737400.0


def nmi(a, b, ok, bins=64):
    x, y = a[ok], b[ok]
    h, _, _ = np.histogram2d(x, y, bins=bins)
    p = h / h.sum()
    px, py = p.sum(1), p.sum(0)
    ent = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))  # noqa: E731
    return float((ent(px) + ent(py)) / ent(p.ravel()))


def u1(z):
    si, ri = z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32)
    so, ro = z["src_ok"].astype(np.float32), z["ref_ok"]
    M = np.asarray(z["model"], float)
    H, W = ri.shape
    out = {}
    for s in SHIFTS:
        vals = []
        for dx, dy in (DIRS if s else [(0.0, 0.0)]):
            T = M.copy(); T[0, 2] += s * dx; T[1, 2] += s * dy
            a = cv2.warpAffine(si, T[:2], (W, H), flags=cv2.INTER_LINEAR)
            m = cv2.warpAffine(so, T[:2], (W, H), flags=cv2.INTER_NEAREST) > 0.5
            ok = m & ro
            ok[:8, :] = ok[-8:, :] = False; ok[:, :8] = ok[:, -8:] = False
            vals.append(nmi(a, ri, ok))
        out[s] = float(np.mean(vals))
    seq = [out[s] for s in SHIFTS]
    return {"nmi_by_shift": {str(s): round(v, 5) for s, v in out.items()},
            "peak_at_zero_and_monotone": bool(all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)))}


def fps(pts, k, seed_pt):
    chosen = [int(np.argmin(np.hypot(*(pts - seed_pt).T)))]
    d = np.hypot(*(pts - pts[chosen[0]]).T)
    for _ in range(k - 1):
        i = int(np.argmax(d)); chosen.append(i)
        d = np.minimum(d, np.hypot(*(pts - pts[i]).T))
    return pts[chosen]


def spread(pts, shape):
    nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
    return {"coverage": round(uniformity.coverage_of(pts, shape, 8), 3),
            "max_empty_circle_px": round(float(uniformity.max_delaunay_gap(pts, shape) or 0.0), 1),
            "nn_cv": round(float(nn.std() / nn.mean()), 3)}


def u2(z, clustered=False):
    pts = np.asarray(z["inlier_src"], float)
    shape = z["src_img"].shape
    if clustered:
        rng = np.random.default_rng(0)
        left = pts[:, 0] < shape[1] / 2
        drop = left & (rng.random(len(pts)) < 0.7)
        pts = pts[~drop]
    u = uniformity.enforce(pts, np.ones(len(pts)), shape, grid=8)
    g = pts[u.keep_mask]
    f = fps(pts, len(g), np.array([shape[1] / 2, shape[0] / 2]))
    return {"points": int(len(g)), "grid": spread(g, shape), "fps": spread(f, shape)}


def blobs(img):
    u8 = cv2.normalize(cv2.GaussianBlur(img, (0, 0), 1.5), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor, p.blobColor = True, 0
    p.filterByArea, p.minArea, p.maxArea = True, np.pi * 1.5 ** 2, np.pi * 20 ** 2
    p.filterByCircularity, p.minCircularity = True, 0.5
    p.filterByInertia = p.filterByConvexity = False
    return np.array([k.pt for k in cv2.SimpleBlobDetector_create(p).detect(u8)], float).reshape(-1, 2)


def u3(z):
    si, ri = z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32)
    M = np.asarray(z["model"], float)
    bs, br = blobs(si), blobs(ri)
    H, W = ri.shape

    def rep(Mx):
        if len(bs) == 0 or len(br) == 0:
            return 0.0
        m = (np.c_[bs, np.ones(len(bs))] @ Mx.T)[:, :2]
        inside = (m[:, 0] > 5) & (m[:, 0] < W - 5) & (m[:, 1] > 5) & (m[:, 1] < H - 5)
        if not inside.any():
            return 0.0
        d = cKDTree(br).query(m[inside])[0]
        return float((d <= 2.0).sum() / min(inside.sum(), len(br)))
    wrong = M.copy(); wrong[0, 2] += 10.0
    return {"blobs_src": int(len(bs)), "blobs_ref": int(len(br)),
            "repeatability_true": round(rep(M), 3), "repeatability_wrong_10px": round(rep(wrong), 3)}


def strip_polygon(label_glob, offset_en_m):
    meta = parse_label(next((ROOT / label_glob[0]).rglob(label_glob[1])))
    m = projection.load_corner_model(meta, corners="system")
    L, S = meta.array_shape[-2], meta.array_shape[-1]
    rows = np.linspace(0, L - 1, 200)
    la_l, lo_l = m.pixel_to_latlon(rows, np.zeros_like(rows))
    la_r, lo_r = m.pixel_to_latlon(rows[::-1], np.full_like(rows, S - 1))
    lat = np.r_[la_l, la_r] + offset_en_m[1] / K_M
    lon = np.r_[lo_l, lo_r] + offset_en_m[0] / (K_M * np.cos(np.radians(np.mean(lat))))
    return meta.product_id, Polygon(zip(lon, lat)).buffer(0)


def u4():
    wac = box(22.3013, -3.9986, 25.3004, 3.9986)          # data/raw/lro/wac_mosaic clip
    mi = box(23.0, 0.0, 24.0, 1.0)                           # MI_MAP_03_N01E023N00E024SC
    prods = {"OHRC": (("data/raw/ch2/ohrc", "*_d_img_d18.xml"), (-194.0, 2211.0)),
             "TMC-2": (("data/raw/ch2/tmc2", "*_d_img_d18.xml"), (600.0, -4700.0)),
             "IIRS": (("data/raw/ch2/iirs", "*_d_img_d18.xml"), (1316.0, 12819.0))}
    out = {}
    for name, (glob, off) in prods.items():
        pid, poly = strip_polygon(glob, off)
        out[name] = {"product": pid,
                     "WAC_clip": {"share_of_product": round(poly.intersection(wac).area / poly.area, 3)},
                     "MI_tile": {"share_of_tile": round(poly.intersection(mi).area / mi.area, 3),
                                 "share_of_product": round(poly.intersection(mi).area / poly.area, 3)}}
        out[name]["testable_now"] = {"WAC": out[name]["WAC_clip"]["share_of_product"] >= 0.2
                                     or poly.intersection(wac).area / wac.area >= 0.2,
                                     "MI": out[name]["MI_tile"]["share_of_tile"] >= 0.2
                                     or out[name]["MI_tile"]["share_of_product"] >= 0.2}
    return out


def main() -> int:
    d = Path(sys.argv[1])
    files = sorted(d.glob("window_*.npz"))
    r1, r2, r2c, r3 = {}, {}, {}, {}
    for f in files:
        z = np.load(f)
        r1[f.stem], r2[f.stem], r2c[f.stem], r3[f.stem] = u1(z), u2(z), u2(z, clustered=True), u3(z)
        print(f.stem, r1[f.stem]["peak_at_zero_and_monotone"], r2[f.stem]["grid"]["max_empty_circle_px"],
              r2[f.stem]["fps"]["max_empty_circle_px"], r3[f.stem]["repeatability_true"], r3[f.stem]["repeatability_wrong_10px"],
              flush=True)
    med = lambda rr, key, m: float(np.median([v[key][m] for v in rr.values()]))  # noqa: E731
    v = {
        "U1_mi_useful": sum(x["peak_at_zero_and_monotone"] for x in r1.values()) >= 14,
        "U2_fps_useful": bool(med(r2, "fps", "max_empty_circle_px") <= 0.8 * med(r2, "grid", "max_empty_circle_px")
                              or med(r2c, "fps", "max_empty_circle_px") <= 0.8 * med(r2c, "grid", "max_empty_circle_px")),
        "U3_craters_useful": sum(x["repeatability_true"] >= 0.30 and x["repeatability_wrong_10px"] <= 0.05
                                 for x in r3.values()) >= 12,
    }
    pairs = u4()
    out = {"source": "measured", "protocol": "docs/usefulness_checks.md", "run": run_record(), "verdict": v,
           "U1": r1, "U2": r2, "U2_clustered": r2c, "U3": r3, "U4": pairs,
           "U2_medians": {c: {m: {k: med(rr, k, m) for k in ("grid", "fps")}
                              for m in ("max_empty_circle_px", "nn_cv", "coverage")}
                          for c, rr in (("normal", r2), ("clustered", r2c))}}
    (ROOT / "reports/usefulness_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": v, "U2_medians": out["U2_medians"], "U4": pairs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
