"""Do the adopted Part 2 results reproduce from scratch on current code? (tolerances fixed before running)

    python scripts/verify_reproduction.py <dir with the rerun jsons>
Writes reports/reproduction_check.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def d_m(a, b):
    if not a or not b:
        return None
    return float(np.hypot(a["east"] - b["east"], a["north"] - b["north"]))


def tmc2_tc(new_dir):
    old = {(w["tmc_row"], w.get("tmc_col")): w for w in load(ROOT / "reports/tmc2_tc_registration_height_ref.json")["rows"]}
    rows, bad = [], []
    for part in ("c0", "cm", "cp"):
        for w in load(new_dir / f"tmc2_tc_{part}.json")["rows"]:
            k = (w["tmc_row"], w.get("tmc_col")); o = old.get(k)
            if o is None:
                continue
            po, pn = (o.get("pipeline") or {}).get("parallax", {}), (w.get("pipeline") or {}).get("parallax", {})
            pdiff = None
            if po.get("p_px_per_m") and pn.get("p_px_per_m"):
                pdiff = float(abs(np.hypot(*pn["p_px_per_m"]) / np.hypot(*po["p_px_per_m"]) - 1))
            r = {"window": list(k), "status": [o.get("status"), w.get("status")], "tier": [o.get("tier"), w.get("tier")],
                 "gates_same": o.get("gates") == w.get("gates"),
                 "offset_diff_m": None if not (o.get("system_offset_m") and w.get("system_offset_m")) else d_m(o["system_offset_m"], w["system_offset_m"]),
                 "parallax_rel_diff": pdiff}
            ok = (r["status"][0] == r["status"][1] and r["tier"][0] == r["tier"][1] and r["gates_same"]
                  and (r["offset_diff_m"] is None or r["offset_diff_m"] <= 15) and (pdiff is None or pdiff <= 0.05))
            r["reproduced"] = bool(ok); rows.append(r)
            if not ok:
                bad.append(r)
    return rows, bad


def per_window(old_path, new_path, key, pick, off_key, tol_m):
    old = {key(w): w for w in load(old_path)["windows"]}
    rows, bad = [], []
    for w in load(new_path)["windows"]:
        o = old.get(key(w))
        if o is None:
            continue
        ro, rn = pick(o), pick(w)
        r = {"window": str(key(w)), "success": [bool(ro.get("success")), bool(rn.get("success"))],
             "tier": [ro.get("tier"), rn.get("tier")], "offset_diff_m": d_m(ro.get(off_key), rn.get(off_key))}
        ok = r["success"][0] == r["success"][1] and (r["offset_diff_m"] is None or r["offset_diff_m"] <= tol_m)
        r["reproduced"] = bool(ok); rows.append(r)
        if not ok:
            bad.append(r)
    return rows, bad


def main() -> int:
    d = Path(sys.argv[1])
    out = {}
    out["tmc2_tc"] = tmc2_tc(d)
    out["ohrc_nac"] = per_window(ROOT / "reports/ohrc_nac_q8_auto_bridge.json", d / "ohrc_nac.json",
                                 lambda w: (w["nac"], w["ohrc_row"]), lambda w: (w.get("results") or {}).get("routed", {}),
                                 "implied_offset_m", 30)
    out["iirs_wac"] = per_window(ROOT / "reports/iirs_wac_mosaic.json", d / "iirs_wac.json",
                                 lambda w: w["iirs_line0"], lambda w: (w.get("results") or {}).get("xoftr", {}),
                                 "implied_offset_m", 50)
    out["iirs_nac"] = per_window(ROOT / "reports/iirs_nac_dense.json", d / "iirs_nac.json",
                                 lambda w: (w["nac"], tuple(w["nac_lines"])), lambda w: w.get("dense") or {},
                                 "implied_offset_m", 30)
    old_g, new_g = load(ROOT / "reports/tmc2_nac_registration.json")["groups"], load(d / "tmc2_nac.json")["groups"]
    g_rows = [{"group": g, "success": [old_g[g].get("success"), new_g.get(g, {}).get("success")],
               "reproduced": old_g[g].get("success") == new_g.get(g, {}).get("success")} for g in old_g]
    out["tmc2_nac"] = (g_rows, [r for r in g_rows if not r["reproduced"]])
    summary = {k: {"compared": len(v[0]), "not_reproduced": len(v[1])} for k, v in out.items()}
    (ROOT / "reports/reproduction_check.json").write_text(json.dumps(
        {"source": "measured", "run": run_record(), "summary": summary,
         "details": {k: {"rows": v[0], "not_reproduced": v[1]} for k, v in out.items()}}, indent=1, default=str),
        encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for k, v in out.items():
        for r in v[1]:
            print("NOT REPRODUCED", k, json.dumps(r, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
