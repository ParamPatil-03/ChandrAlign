"""Audit I-17 / G-08: how much search radius does ISRO's refined-label prior save? (leave-one-out)

    .venv/Scripts/python scripts/geoprior_refined_check.py --out reports/geoprior_refined_check.json

For every committed window of TMC-2 -> SELENE TC and IIRS -> LRO WAC, the MEASURED total correction
from system geolocation (our registration) is compared with three priors:
    system         zero correction -- what any product not in geoprior's tables gets today
    refined_label  geoprior.load(pid, use_measured=False): ISRO's refined-minus-system label corners
    measured       geoprior.load(pid): the table fitted on these same windows (in-sample; reference)
The residual |measured - prior| is the radius the coarse lock must search. LEAVE-ONE-OUT, not a new
product: only one product per instrument is held, so "not in the tables" is simulated by skipping them.
The refined label is SELENE-tuned: it is used ONLY to place the search, never to grade.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.geometry import geoprior  # noqa: E402


def windows():
    d = json.loads((ROOT / "reports/tmc2_tc_registration_height_ref.json").read_text(encoding="utf-8"))
    pid = d["pairing"].split()[0] if " " in str(d.get("pairing")) else None
    for w in d["rows"]:
        if w.get("status") == "registered" and isinstance(w.get("system_offset_m"), dict):
            yield "tmc2", w["tmc_row"], w["system_offset_m"]["east"], w["system_offset_m"]["north"]
    d = json.loads((ROOT / "reports/iirs_wac_mosaic.json").read_text(encoding="utf-8"))
    for w in d["windows"]:
        r = (w.get("results") or {}).get("xoftr", {})
        if r.get("success"):
            yield "iirs", w["iirs_line0"] + 256, r["implied_offset_m"]["east"], r["implied_offset_m"]["north"]


PIDS = {"tmc2": "ch2_tmc_nca_20250207T1102039417_d_img_d18", "iirs": "ch2_iir_nci_20240523T1600301891_d_img_d18"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    priors = {inst: {"system": geoprior.GeoPrior(pid, np.zeros(0), np.zeros(0), np.zeros(0), "system"),
                     "refined_label": geoprior.load(pid, use_measured=False), "measured": geoprior.load(pid)}
              for inst, pid in PIDS.items()}
    rows = []
    for inst, line, e, n in windows():
        rec = {"instrument": inst, "line": line, "measured_m": [round(e), round(n)]}
        for name, p in priors[inst].items():
            pe, pn = p.offset_at(line)
            rec[f"residual_{name}_m"] = round(float(np.hypot(e - pe, n - pn)))
        rows.append(rec)
    summary = {}
    for inst in PIDS:
        r = [x for x in rows if x["instrument"] == inst]
        summary[inst] = {k: {"median": float(np.median([x[f"residual_{k}_m"] for x in r])),
                             "max": float(max(x[f"residual_{k}_m"] for x in r))}
                         for k in ("system", "refined_label", "measured")}
        summary[inst]["windows"] = len(r)
        summary[inst]["refined_label_source"] = priors[inst]["refined_label"].source
    args.out.write_text(json.dumps({"source": "measured", "protocol": "audit I-17 (leave-one-out)",
                                    "summary": summary, "windows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
