"""Score real TMC-2 -> SELENE TC runs against docs/verify_default_protocol.md section 3.

    .venv/Scripts/python scripts/score_real_check.py reports/tmc2_tc_<x>.json [...]

Consistency is the grid RMS (16x16 over the 1536 px window) between a run's
composed transform and the COMMITTED xoftr transform, in TC px. It is agreement
with an independent registration, not accuracy: these windows have no truth.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BAD_PX = 2.0
ELOFTR_VS_XOFTR = {1562: 0.477, 3125: 0.423, 4687: 0.697, 16000: 0.401, 18750: 0.408,
                   21500: 0.352, 52000: 0.388, 54750: 0.554, 57500: 0.393}   # frozen, protocol 3.B.2


def rows(path):
    return {r["tmc_row"]: r for r in json.loads(Path(path).read_text())["rows"]}


def grid_rms(a, b, win=1536):
    g = np.linspace(0, win, 16)
    gx, gy = np.meshgrid(g, g)
    P = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    Ta = np.array(a["registration_result"]["model"]["matrix"])
    Tb = np.array(b["registration_result"]["model"]["matrix"])
    return float(np.sqrt(np.mean(np.sum(((Ta @ P)[:2] - (Tb @ P)[:2]) ** 2, axis=0))))


def score(path, ref_path=ROOT / "reports/tmc2_tc_xoftr.json"):
    X, R = rows(ref_path), rows(path)
    out = []
    for k in ELOFTR_VS_XOFTR:
        r = R.get(k)
        if r is None or r.get("status") != "registered":
            out.append({"tmc_row": k, "registered": False, "status": None if r is None else r.get("status")})
            continue
        gp = r["gate_detail"].get("perturbation_sensitivity", {})
        out.append({"tmc_row": k, "registered": True, "tier": r["tier"],
                    "gates_failed": [g for g, v in r["gates"].items() if not v],
                    "perturbation_error_px": gp.get("error_px"),
                    "vs_xoftr_rms_px": round(grid_rms(r, X[k]), 3),
                    "eloftr_envelope_px": ELOFTR_VS_XOFTR[k],
                    "match_seconds": r.get("match_seconds"), "gate_seconds": r.get("gate_seconds"),
                    "seconds": r.get("seconds"), "inliers": r.get("inliers"),
                    "inlier_rmse_px": r.get("inlier_rmse_px")})
    reg = [o for o in out if o["registered"]]
    summary = {
        "registered": f"{len(reg)}/9",
        "rejected_windows": [o["tmc_row"] for o in reg if o["tier"] == "REJECTED"],
        "gate_failures": {o["tmc_row"]: o["gates_failed"] for o in reg if o["gates_failed"]},
        "max_vs_xoftr_rms_px": max((o["vs_xoftr_rms_px"] for o in reg), default=None),
        "windows_over_bad_px": [o["tmc_row"] for o in reg if o["vs_xoftr_rms_px"] > BAD_PX],
        "windows_outside_eloftr_envelope": [o["tmc_row"] for o in reg
                                            if o["vs_xoftr_rms_px"] > o["eloftr_envelope_px"]],
        "total_seconds": round(sum(o["seconds"] or 0 for o in reg), 1),
    }
    summary["passes_3A"] = (len(reg) == 9 and not summary["rejected_windows"]
                            and not summary["gate_failures"] and not summary["windows_over_bad_px"])
    return {"file": str(path), "summary": summary, "windows": out}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        s = score(p)
        print(p)
        for w in s["windows"]:
            print("  ", w)
        print("  SUMMARY", json.dumps(s["summary"]))
