"""MATCH-12: extreme rotation on real cross-sensor pairs (docs/rotation_protocol.md).

    python scripts/register_tmc2_tc.py --rows 16000 18750 21500 52000 54750 57500 --dump-points <dir> --out <json>
    python scripts/verify_rotation.py <dir>

Source = TMC-2 fine frame (900 px centre crop, rotated by theta, central 640 kept); reference =
SELENE TC fine frame (960 px centre crop). Truth = the rotation composed with the window's affine.
Writes reports/rotation_check.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from chandralign import synth  # noqa: E402
from chandralign.estimate import robust  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.matching import adapter, classical  # noqa: E402
from chandralign.matching.rotation import rotation_search  # noqa: E402

FLAT = (16000, 18750, 21500, 52000, 54750, 57500)
ANGLES = tuple(range(0, 360, 15))
SRC_CROP, SRC_OUT, REF_CROP = 900, 640, 960


def plane(a):
    a = np.asarray(a, np.float32)
    return synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool), shadow_mask=np.zeros(a.shape, bool),
                            gsd_m=7.4, meta=None, geo=None)


def matcher(name):
    if name == "sift":
        return lambda a, b: classical.match(a, b, detector="sift")
    return lambda a, b: adapter.match(a, b, model_name=name, device="cuda")


def case(z, theta):
    si, ri, A = z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32), np.asarray(z["model"], float)
    H, W = si.shape
    s0 = np.array([(W - SRC_CROP) // 2, (H - SRC_CROP) // 2])
    r0 = np.array([(ri.shape[1] - REF_CROP) // 2, (ri.shape[0] - REF_CROP) // 2])
    crop = si[s0[1]:s0[1] + SRC_CROP, s0[0]:s0[0] + SRC_CROP]
    c = (SRC_CROP - 1) / 2.0
    R = cv2.getRotationMatrix2D((c, c), theta, 1.0)                     # crop px -> rotated crop px
    rot = cv2.warpAffine(crop, R, (SRC_CROP, SRC_CROP), flags=cv2.INTER_LINEAR)
    k = (SRC_CROP - SRC_OUT) // 2
    src = rot[k:k + SRC_OUT, k:k + SRC_OUT]
    ref = ri[r0[1]:r0[1] + REF_CROP, r0[0]:r0[0] + REF_CROP]
    # truth: src px q -> rotated crop (q + k) -> crop (R^-1) -> frame (+ s0) -> ref frame (A) -> ref crop (- r0)
    Rinv = np.linalg.inv(np.vstack([R, [0, 0, 1]]))
    Tq = np.eye(3); Tq[:2, 2] = k
    Ts = np.eye(3); Ts[:2, 2] = s0
    Tr = np.eye(3); Tr[:2, 2] = -r0
    truth = Tr @ A @ Ts @ Rinv @ Tq
    return plane(src), plane(ref), truth


def score(ms, truth):
    if len(ms.src_pts) < 4:
        return {"ok": False, "inliers": 0, "why": "no matches"}
    est = robust.estimate(ms.src_pts, ms.ref_pts)
    if est.model is None or est.model.matrix is None:
        return {"ok": False, "inliers": 0, "why": "no transform"}
    M = np.asarray(est.model.matrix, float)
    pts = np.array([[SRC_OUT / 2, SRC_OUT / 2], [0, 0], [SRC_OUT, 0], [0, SRC_OUT], [SRC_OUT, SRC_OUT]], float)
    P = np.c_[pts, np.ones(5)]
    got = P @ M.T; want = P @ truth.T
    got = got[:, :2] / got[:, 2:3]; want = want[:, :2] / want[:, 2:3]
    e = np.hypot(*(got - want).T)
    n = int(est.inlier_count)
    return {"ok": bool(n >= 20 and e[0] <= 1.5 and e[1:].max() <= 3.0), "inliers": n,
            "err_centre_px": round(float(e[0]), 3), "err_corner_max_px": round(float(e[1:].max()), 3)}


def main() -> int:
    d = Path(sys.argv[1])
    rows = []
    for w in FLAT:
        z = np.load(d / f"window_{w}.npz")
        for theta in ANGLES:
            a, b, truth = case(z, theta)
            for name in ("eloftr", "minima-loftr", "sift"):
                fn = matcher(name)
                t0 = time.perf_counter(); plain = score(fn(a, b), truth); tp = time.perf_counter() - t0
                t0 = time.perf_counter(); ms = rotation_search(a, b, fn); ts = time.perf_counter() - t0
                srch = score(ms, truth)
                rs = getattr(ms, "rotation_search", {})
                rows.append({"window": w, "theta": theta, "matcher": name, "plain": plain, "search": srch,
                             "search_angle": rs.get("angle_deg"), "search_applied": rs.get("applied"),
                             "seconds_plain": round(tp, 2), "seconds_search": round(ts, 2)})
            print(w, theta, {r["matcher"]: (r["plain"]["ok"], r["search"]["ok"]) for r in rows[-3:]}, flush=True)
    summ = {}
    for name in ("eloftr", "minima-loftr", "sift"):
        rs = [r for r in rows if r["matcher"] == name]
        by = {t: sum(r["plain"]["ok"] for r in rs if r["theta"] == t) for t in ANGLES}
        summ[name] = {"plain_ok": sum(r["plain"]["ok"] for r in rs), "search_ok": sum(r["search"]["ok"] for r in rs),
                      "cases": len(rs), "plain_ok_by_theta": by,
                      "zero_deg_identical": all(r["search_applied"] is False and r["search"] == r["plain"]
                                                for r in rs if r["theta"] == 0 and r["plain"]["inliers"] >= 40),
                      "runtime_ratio_median": round(float(np.median([r["seconds_search"] / max(r["seconds_plain"], 1e-3)
                                                                     for r in rs])), 2)}
    e, s = summ["eloftr"], summ["sift"]
    verdict = {"a_eloftr_search_ge_95pct": e["search_ok"] >= 0.95 * e["cases"],
               "b_zero_deg_no_regression": all(v["zero_deg_identical"] for v in summ.values()),
               "c_sift_search_ge_plain": s["search_ok"] >= s["plain_ok"]}
    verdict["adopt"] = all(verdict.values())
    out = {"source": "measured", "protocol": "docs/rotation_protocol.md", "run": run_record(),
           "summary": summ, "verdict": verdict, "cases": rows}
    (ROOT / "reports/rotation_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"summary": {k: {kk: vv for kk, vv in v.items() if kk != "plain_ok_by_theta"} for k, v in summ.items()},
                      "plain_ok_by_theta": {k: v["plain_ok_by_theta"] for k, v in summ.items()}, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
