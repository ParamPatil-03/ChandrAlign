"""ALIGN-08: does affine + DEM parallax predict the real per-cell offsets better than the affine?

    python scripts/register_tmc2_tc.py --tile all --windows 3 --stage parallax=on \\
        --dump-points <dir> --out <run.json>
    python scripts/parallax_eval.py <dir> <run.json> [<baseline.json>]

Protocol, frozen before this ran: docs/parallax_protocol.md. The measure is the matcher-free
phase-correlation offset of every 8x8 cell with full valid data; the DEM is an input to the
parallax model but not to this measure.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from chandralign.estimate import models  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402

G, PAD = 8, 33


def cell_offsets(z, affine, par):
    """(phase-corr offset, affine prediction, affine+parallax prediction) per usable cell."""
    si, ri, ok = z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32), z["src_ok"]
    dem, st = z["dem_m"], int(z["dem_step"])
    H, W = si.shape
    Bp = np.asarray(par["affine"], float)
    p, h0 = np.asarray(par["p_px_per_m"], float), float(par["h0_m"])
    rows = []
    for r in range(G):
        for c in range(G):
            y0, y1, x0, x1 = int(r * H / G), int((r + 1) * H / G), int(c * W / G), int((c + 1) * W / G)
            if ok[y0:y1, x0:x1].mean() < 0.99:
                continue
            ya, yb, xa, xb = max(0, y0 - PAD), min(H, y1 + PAD), max(0, x0 - PAD), min(W, x1 + PAD)
            a, b = si[ya:yb, xa:xb], ri[ya:yb, xa:xb]
            (sx, sy), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
            ctr = np.array([(xa + xb) / 2.0, (ya + yb) / 2.0])
            h = float(np.nanmean(dem[ya // st:yb // st + 1, xa // st:xb // st + 1]))
            aff = models.apply(affine, ctr[None])[0] - ctr
            prl = Bp @ np.r_[ctr, 1.0] + (h - h0) * p - ctr
            rows.append((np.array([sx, sy]), aff, prl))
    return rows


def rms(v):
    return float(np.sqrt(np.mean(np.sum(np.asarray(v) ** 2, axis=1))))


def main() -> int:
    d, run = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    base = json.loads(Path(sys.argv[3] if len(sys.argv) > 3 else ROOT / "reports/tmc2_tc_registration.json")
                      .read_text(encoding="utf-8"))
    key = lambda w: (w["tmc_row"], w.get("tmc_col"))  # noqa: E731
    before = {key(w): w for w in base["rows"]}
    out = []
    for w in run["rows"]:
        row, col, b = w["tmc_row"], w.get("tmc_col"), before.get(key(w), {})
        f = d / (f"window_{row}.npz" if col is None else f"window_{row}_c{col}.npz")
        par = (w.get("pipeline") or {}).get("parallax", {})
        rec = {"tile": w.get("tile"), "tmc_row": row, "tmc_col": col, "status": w.get("status"), "tier": w.get("tier"),
               "tier_before": b.get("tier"), "status_before": b.get("status"),
               "gates_pass": all(w.get("gates", {}).values()) if w.get("gates") else None,
               "gates_pass_before": all(b.get("gates", {}).values()) if b.get("gates") else None,
               "empty_cells": (w.get("pipeline") or {}).get("uniformity", {}).get("empty_cells"),
               "empty_cells_before": (b.get("pipeline") or {}).get("uniformity", {}).get("empty_cells"),
               "coverage": w.get("coverage"), "coverage_before": b.get("coverage"), "parallax": par}
        if f.exists() and par.get("applied"):
            z = np.load(f)
            cells = cell_offsets(z, models.TransformModel(kind="affine", matrix=z["model"]), par)
            rec["cells"] = len(cells)
            rec["rms_affine_px"] = round(rms([m - a for m, a, _ in cells]), 3)
            rec["rms_parallax_px"] = round(rms([m - q for m, _, q in cells]), 3)
            px, py = par["p_px_per_m"]
            pred = math.tan(math.radians(26.0)) / float(z["gsd_m"])
            rec["p_along_track"] = bool(abs(py) >= 5 * abs(px))
            rec["p_over_predicted"] = round(math.hypot(px, py) / pred, 3)
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "parallax"}), flush=True)

    n00 = [r for r in out if r["tile"] == "N00"]
    flat = [r for r in out if r["tile"] != "N00"]
    reg = lambda r: r["status"] == "registered"  # noqa: E731
    order = {"REJECTED": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, None: -1}
    a = all(reg(r) and r["empty_cells"] is not None and r["empty_cells"] <= 1 for r in n00)
    b = (all("rms_parallax_px" in r and r["rms_parallax_px"] <= 0.5 * r["rms_affine_px"] for r in n00)
         and all("rms_parallax_px" in r and r["rms_parallax_px"] <= r["rms_affine_px"] + 0.2 for r in flat))
    c = all((not reg_b or reg(r)) and order[r["tier"]] >= order[r["tier_before"]]
            and (not r["gates_pass_before"] or r["gates_pass"])
            for r in out for reg_b in [r["status_before"] == "registered"])
    dd = all(r.get("p_along_track") and 0.5 <= r.get("p_over_predicted", 0) <= 1.5 for r in n00)
    verdict = {"a_empty_cells": a, "b_offsets": b, "c_nothing_lost": c, "d_physical_p": dd,
               "adopt": bool(a and b and c and dd)}
    print(json.dumps(verdict))
    dst = ROOT / (sys.argv[4] if len(sys.argv) > 4 else "reports/tmc2_tc_parallax.json")
    dst.write_text(json.dumps({"source": "measured", "protocol": "docs/parallax_protocol.md", "run": run_record(),
                               "verdict": verdict, "windows": out}, indent=2), encoding="utf-8")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
