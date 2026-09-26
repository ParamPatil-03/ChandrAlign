"""Do the adopted Part 2 results reproduce from scratch on current code?

    python scripts/verify_reproduction.py <dir with the rerun jsons> [--out FILE] [--tol-scale K]

The rerun directory holds tmc2_tc_{c0,cm,cp}.json, ohrc_nac.json, iirs_wac.json, iirs_nac.json and
tmc2_nac.json, as scripts/register_*.py write them (--out). Each is compared with the committed report.

STRICT BY CONSTRUCTION (audit 2026-09-26, I-15). The previous version iterated only the NEW windows
and skipped any window missing from the old report, so a dropped window passed silently. It compared
OHRC/IIRS on `success` and a 30-50 m offset only (~100 OHRC px), and reduced TMC-2 -> NAC to four
group booleans. Reruns are bit-identical on this machine, so now:
  - windows are keyed on the UNION of old and new; a window missing on either side is NOT reproduced
  - every pairing compares status/success, tier, inliers, known-shift error and the offset
  - tolerances are pixel-scale (TOL, multiplied by --tol-scale), not tens of metres
  - TMC-2 -> NAC is compared per window (lock, z, shift error, geodetic disagreement)
Writes to --out (default: <dir>/reproduction_check.json), never into reports/ unless asked to.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402

# Fixed before any comparison. Reruns are bit-identical; these allow float formatting only.
TOL = {"ks_err_px": 1e-3, "offset_m": 0.01, "parallax_rel": 1e-4, "inliers_rel": 0.0, "z": 1e-2,
       "shift_err_px": 1e-3, "geo_dis_m": 0.01}

OLD = {"tmc2_tc": "reports/tmc2_tc_registration_height_ref.json", "ohrc_nac": "reports/ohrc_nac_q8_auto_bridge.json",
       "iirs_wac": "reports/iirs_wac_mosaic.json", "iirs_nac": "reports/iirs_nac_dense.json",
       "tmc2_nac": "reports/tmc2_nac_registration.json"}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dig(w, *keys):
    return functools.reduce(lambda a, k: (a or {}).get(k) if isinstance(a, dict) else None, keys, w)


def d_m(a, b):
    if not a or not b:
        return None
    return float(np.hypot(a["east"] - b["east"], a["north"] - b["north"]))


def _num_close(a, b, tol, rel=False):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if rel:
        return abs(b - a) <= tol * max(abs(a), 1e-12)
    return abs(b - a) <= tol


def compare(old: dict, new: dict, fields: dict, tol_scale: float) -> tuple[list, list]:
    """fields: name -> (getter, kind, tol_key). kind: 'eq' exact, 'abs'/'rel' numeric, 'offset' (m)."""
    rows, bad = [], []
    for k in sorted(set(old) | set(new), key=str):
        o, n = old.get(k), new.get(k)
        if o is None or n is None:
            r = {"window": str(k), "missing_in": "new" if n is None else "old", "reproduced": False}
            rows.append(r); bad.append(r)
            continue
        r = {"window": str(k)}
        ok = True
        for name, (get, kind, tk) in fields.items():
            a, b = get(o), get(n)
            if kind == "offset":
                diff = d_m(a, b)
                r[name + "_diff_m"] = None if diff is None else round(diff, 4)
                same = (a is None) == (b is None) and (diff is None or diff <= TOL[tk] * tol_scale)
            else:
                r[name] = [a, b]
                same = (a == b) if kind == "eq" else _num_close(a, b, TOL[tk] * tol_scale, rel=kind == "rel")
            if not same:
                r.setdefault("changed", []).append(name)
                ok = False
        r["reproduced"] = ok
        rows.append(r)
        if not ok:
            bad.append(r)
    return rows, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rerun_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tol-scale", type=float, default=1.0,
                    help="multiply every tolerance (a deliberate code change is EXPLAINED, not tolerated)")
    args = ap.parse_args()
    d, s = args.rerun_dir, args.tol_scale
    out = {}

    old = {(w["tmc_row"], w.get("tmc_col")): w for w in load(ROOT / OLD["tmc2_tc"])["rows"]}
    new = {}
    for part in ("c0", "cm", "cp"):
        p = d / f"tmc2_tc_{part}.json"
        if p.exists():
            new.update({(w["tmc_row"], w.get("tmc_col")): w for w in load(p)["rows"]})
    out["tmc2_tc"] = compare(old, new, {
        "status": (lambda w: w.get("status"), "eq", None), "tier": (lambda w: w.get("tier"), "eq", None),
        "gates": (lambda w: w.get("gates"), "eq", None), "inliers": (lambda w: w.get("inliers"), "rel", "inliers_rel"),
        "ks_err": (lambda w: dig(w, "gate_detail", "perturbation_sensitivity", "error_px"), "abs", "ks_err_px"),
        "parallax_p": (lambda w: None if not dig(w, "pipeline", "parallax", "p_px_per_m")
                       else float(np.hypot(*dig(w, "pipeline", "parallax", "p_px_per_m"))), "rel", "parallax_rel"),
        "offset": (lambda w: w.get("system_offset_m"), "offset", "offset_m")}, s)

    def ks(w):
        return w.get("known_shift_error_px", dig(w, "checks", "known_shift", "error_px"))

    per = {"ohrc_nac": (lambda w: (w["nac"], w["ohrc_row"]), lambda w: dig(w, "results", "routed")),
           "iirs_wac": (lambda w: w["iirs_line0"], lambda w: dig(w, "results", "xoftr")),
           "iirs_nac": (lambda w: (w["nac"], tuple(w["nac_lines"])), lambda w: w.get("dense"))}
    for name, (key, pick) in per.items():
        p = d / f"{name}.json"
        o = {key(w): (pick(w) or {}) for w in load(ROOT / OLD[name])["windows"]}
        n = {key(w): (pick(w) or {}) for w in load(p)["windows"]} if p.exists() else {}
        out[name] = compare(o, n, {
            "success": (lambda w: bool(w.get("success")), "eq", None), "tier": (lambda w: w.get("tier"), "eq", None),
            "used": (lambda w: w.get("used"), "eq", None), "inliers": (lambda w: w.get("inliers"), "rel", "inliers_rel"),
            "ks_err": (ks, "abs", "ks_err_px"), "offset": (lambda w: w.get("implied_offset_m"), "offset", "offset_m")}, s)

    kf = lambda w: (w.get("nac"), w.get("fraction"))  # noqa: E731
    p = d / "tmc2_nac.json"
    o = {kf(w): w for w in load(ROOT / OLD["tmc2_nac"])["windows"]}
    n = {kf(w): w for w in load(p)["windows"]} if p.exists() else {}
    out["tmc2_nac"] = compare(o, n, {
        "success": (lambda w: w.get("success"), "eq", None), "locked": (lambda w: w.get("locked"), "eq", None),
        "z": (lambda w: w.get("z"), "abs", "z"), "shift_err": (lambda w: w.get("shift_err"), "abs", "shift_err_px"),
        "geo_dis_m": (lambda w: w.get("geo_dis_m"), "abs", "geo_dis_m")}, s)

    summary = {k: {"compared": len(v[0]), "not_reproduced": len(v[1]),
                   "missing": sum("missing_in" in r for r in v[1])} for k, v in out.items()}
    target = args.out or (d / "reproduction_check.json")
    target.write_text(json.dumps({"source": "measured", "run": run_record(), "tolerances": TOL, "tol_scale": s,
                                  "summary": summary,
                                  "details": {k: {"rows": v[0], "not_reproduced": v[1]} for k, v in out.items()}},
                                 indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for k, v in out.items():
        for r in v[1]:
            print("NOT REPRODUCED", k, json.dumps(r, default=str)[:300])
    print(f"written to {target}")
    return 1 if any(v[1] for v in out.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
