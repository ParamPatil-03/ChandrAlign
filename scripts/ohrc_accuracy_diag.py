"""Where does OHRC -> NAC's ~3.5 OHRC-px fit residual come from? (diagnostic, no decision)

    python scripts/register_ohrc_nac.py --products ... --matchers routed --auto-bridge --dump-points <dir>
    python scripts/ohrc_accuracy_diag.py <dir>

Per window: (1) point scatter = RMS of inliers about the affine; (2) MODEL accuracy = median
|NCC-probe offset - model| (matcher-free, scripts/parallax_probe_eval.py probes); (3) terrain
parallax = how much of the residual a height x p term explains (SLDEM at the source point).
All in fine-frame (NAC) px and in OHRC px (worst direction). Writes reports/ohrc_accuracy_diag.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

from chandralign.evaluate.source_px import factors  # noqa: E402
from parallax_probe_eval import probes  # noqa: E402


def main() -> int:
    rows = {}
    for f in sorted(Path(sys.argv[1]).glob("*.npz")):
        z = np.load(f)
        M = np.asarray(z["model"], float)
        fw = factors(z["J"])["worst"]                                   # OHRC px per frame px
        s, r = np.asarray(z["inlier_src"], float), np.asarray(z["inlier_ref"], float)
        res = r - (np.c_[s, np.ones(len(s))] @ M.T)[:, :2]
        scatter = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))
        par = None
        if z["dem_m"].size:
            h = map_coordinates(z["dem_m"], [s[:, 1] / 8, s[:, 0] / 8], order=1, mode="nearest")
            k = np.isfinite(h)
            h, rk, sk = h[k], res[k], s[k]
        if z["dem_m"].size and k.sum() >= 20:
            X = np.c_[np.ones(len(h)), sk, h - np.median(h)]                 # affine residual + h.p
            B = np.linalg.lstsq(X, rk, rcond=None)[0]
            left = rk - X @ B
            res_for_par = rk
            par = {"p_px_per_m": [round(float(v), 5) for v in B[3]],
                   "relief_p5_p95_m": [round(float(np.percentile(h, 5)), 1), round(float(np.percentile(h, 95)), 1)],
                   "points_with_height": int(k.sum()),
                   "variance_explained": round(float(1 - np.sum(left ** 2) / np.sum(res_for_par ** 2)), 3)}
        pr = probes(z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32), z["src_ok"])
        pe = None
        if len(pr):
            pred = (np.c_[pr[:, :2], np.ones(len(pr))] @ M.T)[:, :2] - pr[:, :2]
            e = np.hypot(*(pr[:, 2:] - pred).T)
            pe = {"probes": int(len(pr)), "median_frame_px": round(float(np.median(e)), 3),
                  "median_ohrc_px": round(float(np.median(e) * fw), 3)}
        rows[f.stem] = {"frame_px_m": [round(float(v), 3) for v in z["frame_px_m"]], "ohrc_px_per_frame_px": round(fw, 2),
                        "inliers": int(len(s)), "scatter_frame_px": round(scatter, 3), "scatter_ohrc_px": round(scatter * fw, 3),
                        "model_probe": pe, "parallax": par}
        print(f.stem, json.dumps(rows[f.stem]), flush=True)
    (ROOT / "reports/ohrc_accuracy_diag.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
