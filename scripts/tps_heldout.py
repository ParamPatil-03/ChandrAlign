"""ALIGN-02: does TPS describe real geometry better than the affine model, on HELD-OUT points?

    python scripts/register_tmc2_tc.py --tile all --windows 3 --dump-points <dir> --out <json>
    python scripts/tps_heldout.py <dir>

Protocol, frozen before this ran: docs/tps_protocol.md. TPS is fitted on each window's
delivered control points only (smoothing by 5-fold CV on those points); both models are
scored on every other inlier of the window, which neither was fitted to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.estimate import models  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402

GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def rms(a):
    return float(np.sqrt(np.mean(np.sum(np.asarray(a) ** 2, axis=1))))


def choose_smoothing(src, ref, seed=0):
    idx = np.random.default_rng(seed).permutation(len(src))
    folds = np.array_split(idx, 5)
    best = None
    for s in GRID:
        errs = []
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            m = models.fit_tps(src[tr], ref[tr], smoothing=s)
            errs.append(models.apply(m, src[te]) - ref[te])
        e = rms(np.concatenate(errs))
        if best is None or e < best[1]:
            best = (s, e)
    return best[0]


def main() -> int:
    d = Path(sys.argv[1])
    rows = []
    for f in sorted(d.glob("window_*.npz")):
        z = np.load(f)
        cs, cr, s_all, r_all = z["control_src"], z["control_ref"], z["inlier_src"], z["inlier_ref"]
        A = models.TransformModel(kind="affine", matrix=z["model"])
        # held-out: inliers that are not control points
        key = lambda a: {tuple(np.round(p, 3)) for p in a}  # noqa: E731
        cset = key(cs)
        mask = np.array([tuple(np.round(p, 3)) not in cset for p in s_all])
        hs, hr = s_all[mask], r_all[mask]
        sm = choose_smoothing(cs, cr)
        T = models.fit_tps(cs, cr, smoothing=sm)
        e_aff = rms(models.apply(A, hs) - hr)
        e_tps = rms(models.apply(T, hs) - hr)
        rows.append({"window": f.stem, "control_points": int(len(cs)), "heldout_points": int(len(hs)),
                     "smoothing": sm, "affine_rms_px": round(e_aff, 4), "tps_rms_px": round(e_tps, 4),
                     "improvement": round(1 - e_tps / e_aff, 4)})
        print(json.dumps(rows[-1]), flush=True)
    big = [r for r in rows if r["affine_rms_px"] > 1.0]
    med = float(np.median([r["improvement"] for r in big])) if big else None
    worst = min(r["improvement"] for r in rows)
    verdict = {"windows_affine_over_1px": len(big), "median_improvement_there": med,
               "worst_improvement_any": worst,
               "adopt": bool(big and med >= 0.20 and worst >= -0.05)}
    print(json.dumps(verdict))
    out = ROOT / "reports/tps_heldout.json"
    out.write_text(json.dumps({"source": "measured", "protocol": "docs/tps_protocol.md", "run": run_record(),
                               "verdict": verdict, "windows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
