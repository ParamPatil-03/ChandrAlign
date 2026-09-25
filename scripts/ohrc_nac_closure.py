"""Closure test: is OHRC -> NAC A biased? (docs/ohrc_nac_closure_protocol.md, frozen before running)

    python scripts/register_ohrc_nac.py --products A B --matchers routed --auto-bridge --rows ... --dump-points <dir>
    python scripts/ohrc_nac_closure.py <dir> A B

T_A, T_B come from the dumps (OHRC px -> NAC px); T_AB (A px -> B px) is registered here directly.
Writes reports/ohrc_nac_closure.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from chandralign.estimate import robust  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.preprocess.resample import warp_affine  # noqa: E402
from register_ohrc_nac import T, match, norm  # noqa: E402
from register_tmc2_nac import NULL_BELOW  # noqa: E402

HALF = 800                                   # A crop half-size, A px (~0.64 km at 0.40 m)


def crop(meta, cx, cy, half):
    L, S = meta.array_shape
    r0, c0 = int(max(0, cy - half)), int(max(0, cx - half))
    r1, c1 = int(min(L, cy + half)), int(min(S, cx + half))
    raw = pds_raster.read_raster(meta, pds_raster.Window(r0, c0, r1 - r0, c1 - c0)).astype(np.float32)
    return norm(raw, raw > NULL_BELOW), np.array([c0, r0], float)


def main() -> int:
    d, A, B = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    meta = {p: parse_label(next((ROOT / "data/raw/lro/nac").rglob(f"{p}.XML"))) for p in (A, B)}
    lroc = {}
    for f in ("tmc2_nac_lroc_meta.json", "iirs_nac_lroc_meta.json"):
        lroc.update(json.loads((ROOT / "data/pairs" / f).read_text(encoding="utf-8"))["products"])
    bpx = (float(lroc[B]["scaled_pixel_width"]), float(lroc[B]["scaled_pixel_height"]))
    rows = []
    for fa in sorted(d.glob(f"{A}_*_eloftr.npz")):
        row = fa.stem.split("_")[1]
        fb = d / f"{B}_{row}_eloftr.npz"
        if not fb.exists():
            continue
        za, zb = np.load(fa), np.load(fb)
        TA, TB = np.asarray(za["T_total"], float), np.asarray(zb["T_total"], float)
        c = np.array([1024.0, 1024.0, 1.0])                            # OHRC window centre (2048 px windows)
        pa, pb = TA @ c, TB @ c
        P = TB @ np.linalg.inv(TA)                                     # prior A px -> B px (placement only)
        Acrop, a0 = crop(meta[A], pa[0], pa[1], HALF)
        halfB = int(np.ceil(HALF * np.linalg.norm(P[:2, :2], 2))) + 32
        Bcrop, b0 = crop(meta[B], pb[0], pb[1], halfB)
        Wab = T(-b0[0], -b0[1]) @ P @ T(a0[0], a0[1])                   # A-crop px -> B-crop px
        fsrc = warp_affine(Acrop, Wab, (Bcrop.shape[1], Bcrop.shape[0]))
        ms, _, _ = match("eloftr", fsrc, Bcrop, "cuda", max(bpx))
        est = robust.estimate(ms.src_pts, ms.ref_pts) if len(ms.src_pts) >= 4 else None
        if est is None or est.model is None or est.model.matrix is None:
            rows.append({"row": int(row), "status": "A->B: no transform"}); continue
        TAB = T(b0[0], b0[1]) @ np.asarray(est.model.matrix, float) @ Wab @ T(-a0[0], -a0[1])  # A px -> B px
        e_px = (pb - TAB @ pa)[:2]                                     # closure error, B px
        e_m = e_px * np.array(bpx)
        u = (TAB[:2, :2] @ np.array([1.0, 0.0])); u /= np.linalg.norm(u)   # A's sample (cross-track) axis in B
        u_m = u * np.array(bpx); u_m /= np.linalg.norm(u_m)
        cross = float(e_m @ u_m); along = float(np.cross(u_m, e_m))
        rows.append({"row": int(row), "status": "ok", "A_to_B_inliers": int(est.inlier_count),
                     "closure_m": round(float(np.hypot(*e_m)), 3), "cross_track_m": round(cross, 3),
                     "along_track_m": round(along, 3)})
        print(json.dumps(rows[-1]), flush=True)
    ok = [r for r in rows if r.get("status") == "ok"]
    cross = [r["cross_track_m"] for r in ok]
    med = float(np.median([r["closure_m"] for r in ok])) if ok else None
    same_sign_big = (sum(1 for x in cross if x >= 1.0) >= 2) or (sum(1 for x in cross if x <= -1.0) >= 2)
    reading = ("does not close cross-track: OHRC -> A taken as biased" if ok and same_sign_big and abs(np.median(cross)) >= 1.0
               else "closes: the MI peak on A taken as misled" if med is not None and med < 0.5 else "inconclusive")
    out = {"source": "measured", "protocol": "docs/ohrc_nac_closure_protocol.md", "run": run_record(),
           "A": A, "B": B, "windows": rows, "median_closure_m": med,
           "median_cross_track_m": None if not cross else round(float(np.median(cross)), 3), "reading": reading}
    (ROOT / "reports/ohrc_nac_closure.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("median_closure_m", "median_cross_track_m", "reading")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
