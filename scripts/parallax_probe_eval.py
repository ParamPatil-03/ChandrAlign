"""ALIGN-08 amendment 3: affine vs affine + parallax, scored on small NCC probes.

    python scripts/register_tmc2_tc.py --rows ... --stage parallax=on --parallax-dem tc_dtm \\
        --dump-points <dir> --out <run.json>
    python scripts/parallax_probe_eval.py <dir> <run.json> <baseline.json> <out.json>

Protocol, frozen before this ran: docs/parallax_protocol.md, amendment 3. A probe is a
31x31 source template found in the reference by normalised cross-correlation within +-32 px;
it uses neither the matcher nor any DEM, and whether a probe is accepted never depends on
a model. It is a proxy for the true local offset, not ground truth.
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
from scipy.ndimage import map_coordinates  # noqa: E402

from chandralign.estimate import models  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402

# The probe itself lives in the library now (audit I-08: the product path grades with it too).
from chandralign.evaluate.probes import EXCL_R, HALF, MIN_MARGIN, MIN_PEAK, SEARCH, STEP, probes  # noqa: E402,F401


def main() -> int:
    d = Path(sys.argv[1])
    run = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    base = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    key = lambda w: (w["tmc_row"], w.get("tmc_col"))  # noqa: E731
    before = {key(w): w for w in base["rows"]}
    out = []
    for w in run["rows"]:
        row, col, b = w["tmc_row"], w.get("tmc_col"), before.get(key(w), {})
        pipe, bpipe = w.get("pipeline") or {}, b.get("pipeline") or {}
        par = pipe.get("parallax", {})
        rec = {"tile": w.get("tile"), "tmc_row": row, "tmc_col": col, "hilly": w.get("tile") == "N00",
               "parallax_dem": w.get("parallax_dem"), "status": w.get("status"), "status_before": b.get("status"),
               "tier": w.get("tier"), "tier_before": b.get("tier"),
               "gates_pass": all(w["gates"].values()) if w.get("gates") else None,
               "gates_pass_before": all(b["gates"].values()) if b.get("gates") else None,
               "empty_cells": pipe.get("uniformity", {}).get("empty_cells"),
               "empty_cells_before": bpipe.get("uniformity", {}).get("empty_cells"), "parallax": par}
        f = d / (f"window_{row}.npz" if col is None else f"window_{row}_c{col}.npz")
        if f.exists() and par.get("applied"):
            z = np.load(f)
            pr = probes(z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32), z["src_ok"])
            pts, meas = pr[:, :2], pr[:, 2:]
            aff = models.apply(models.TransformModel(kind="affine", matrix=z["model"]), pts) - pts
            st = float(z["dem_step"])
            h = map_coordinates(z["dem_m"], [pts[:, 1] / st, pts[:, 0] / st], order=1, mode="nearest")
            B, p = np.asarray(par["affine"], float), np.asarray(par["p_px_per_m"], float)
            prl = np.c_[pts, np.ones(len(pts))] @ B.T + (h - float(par["h0_m"]))[:, None] * p - pts
            if par.get("height_at") == "ref":        # h at the reference point: iterate r = A.s + (h(r) - h0).p
                r = prl + pts
                for _ in range(6):
                    hr = map_coordinates(z["dem_m"], [r[:, 1] / st, r[:, 0] / st], order=1, mode="nearest")
                    r = np.c_[pts, np.ones(len(pts))] @ B.T + (hr - float(par["h0_m"]))[:, None] * p
                prl = r - pts
            e_aff, e_prl = np.hypot(*(meas - aff).T), np.hypot(*(meas - prl).T)
            rec.update(probes=int(len(pr)), median_err_affine_px=round(float(np.median(e_aff)), 3),
                       median_err_parallax_px=round(float(np.median(e_prl)), 3),
                       p90_err_affine_px=round(float(np.percentile(e_aff, 90)), 3),
                       p90_err_parallax_px=round(float(np.percentile(e_prl, 90)), 3))
            pred = math.tan(math.radians(26.0)) / float(z["gsd_m"])
            rec["p_along_track"] = bool(abs(p[1]) >= 5 * abs(p[0]))
            rec["p_over_predicted"] = round(float(np.hypot(*p)) / pred, 3)
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "parallax"}), flush=True)

    hilly, flat = [r for r in out if r["hilly"]], [r for r in out if not r["hilly"]]
    order = {"REJECTED": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, None: -1}
    a = all(r["empty_cells"] is not None and r["empty_cells"] <= 1 for r in hilly)
    b = (all("median_err_parallax_px" in r and r["median_err_parallax_px"] <= 1.0
             and r["median_err_parallax_px"] < r["median_err_affine_px"] for r in hilly)
         and all("median_err_parallax_px" in r
                 and r["median_err_parallax_px"] <= r["median_err_affine_px"] + 0.1 for r in flat))
    c = all((r["status_before"] != "registered" or r["status"] == "registered")
            and order[r["tier"]] >= order[r["tier_before"]]
            and (not r["gates_pass_before"] or r["gates_pass"]) for r in out)
    dd = all(r.get("p_along_track") and 0.5 <= r.get("p_over_predicted", 0) <= 1.5 for r in hilly)
    verdict = {"a_empty_cells": a, "b_probe_error": b, "c_nothing_lost": c, "d_physical_p": dd,
               "adopt": bool(a and b and c and dd)}
    print(json.dumps(verdict))
    dst = ROOT / sys.argv[4]
    dst.write_text(json.dumps({"source": "measured", "protocol": "docs/parallax_protocol.md (amendment 3)",
                               "run": run_record(), "verdict": verdict, "windows": out}, indent=2),
                   encoding="utf-8")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
