"""Ablation table (docs/ablation_protocol.md): score each configuration's run + dumps.

    python scripts/ablation_eval.py <root>      # <root>/C0..C4/{run.json, dumps/}
Writes reports/ablation_tmc2_tc.json and prints the table.
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

from chandralign.evaluate.run_record import run_record  # noqa: E402
from parallax_probe_eval import probes  # noqa: E402

CONFIGS = {"C0": "SIFT, stages off", "C1": "+ eloftr", "C2": "+ terrain filter",
           "C3": "+ terrain parallax", "C4": "+ uniformity, sub-pixel, TPS (adopted)"}


def accuracy(z, stage):
    pr = probes(z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32), z["src_ok"])
    if not len(pr):
        return None
    pts, meas = pr[:, :2], pr[:, 2:]
    M = np.asarray(z["model"], float)
    pred = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :2]
    if stage.get("applied") and "dem_m" in z and z["dem_m"].size:       # affine + parallax, own convention
        B, p, h0, st = np.asarray(stage["affine"], float), np.asarray(stage["p_px_per_m"], float), float(stage["h0_m"]), float(z["dem_step"])
        hs = lambda q: map_coordinates(z["dem_m"], [q[:, 1] / st, q[:, 0] / st], order=1, mode="nearest")  # noqa: E731
        base = np.c_[pts, np.ones(len(pts))] @ B.T
        pred = base + (hs(pts) - h0)[:, None] * p
        if stage.get("height_at") == "ref":
            for _ in range(6):
                pred = base + (hs(pred) - h0)[:, None] * p
    return float(np.median(np.hypot(*(meas - (pred - pts)).T)))


def main() -> int:
    root = Path(sys.argv[1])
    table = {}
    for c, label in CONFIGS.items():
        run = json.loads((root / c / "run.json").read_text(encoding="utf-8"))
        rows = []
        for w in run["rows"]:
            f = root / c / "dumps" / f"window_{w['tmc_row']}.npz"
            fac = ((w.get("source_px") or {}).get("known_shift") or {}).get("factor_worst")
            acc = accuracy(np.load(f), (w.get("pipeline") or {}).get("parallax", {})) if f.exists() else None
            rows.append({"row": w["tmc_row"], "hilly": w.get("tile") == "N00", "registered": w.get("status") == "registered",
                         "gates_pass": bool(w.get("gates")) and all(w["gates"].values()), "tier": w.get("tier"),
                         "inliers": w.get("inliers"), "coverage": w.get("coverage"),
                         "known_shift_src_px": ((w.get("source_px") or {}).get("known_shift") or {}).get("worst"),
                         "probe_err_src_px": None if acc is None or fac is None else round(acc * fac, 3)})
        table[c] = {"config": label, "windows": rows}
        med = lambda k, sel: (lambda v: None if not v else round(float(np.median(v)), 3))(  # noqa: E731
            [r[k] for r in rows if r[k] is not None and sel(r)])
        print(f"{c} {label:40} gates {sum(r['gates_pass'] for r in rows)}/{len(rows)}  tiers {[r['tier'] for r in rows]}"
              f"  inliers {med('inliers', lambda r: True)}  coverage {med('coverage', lambda r: True)}"
              f"  known-shift {med('known_shift_src_px', lambda r: True)}  probe err hilly {med('probe_err_src_px', lambda r: r['hilly'])}"
              f" flat {med('probe_err_src_px', lambda r: not r['hilly'])} (src px)")
    (ROOT / "reports/ablation_tmc2_tc.json").write_text(json.dumps({"source": "measured", "protocol": "docs/ablation_protocol.md",
                                                                   "run": run_record(), "table": table}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
