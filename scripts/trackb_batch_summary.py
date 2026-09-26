"""Summarise a Track B real-data batch against the committed baseline (audit C-02, C-03, C-04, I-07, I-08,
I-11, I-17). Reads only JSON outputs; writes one summary JSON.

    python scripts/trackb_batch_summary.py <batch out dir> --out reports/trackb_batch1_summary.json

Expects the file names of the Track B batch scripts (tmc2_tc_{c0,cm,cp}.json, ohrc_nac.json, iirs_wac.json,
tmc2_tc_c0_labelprior.json); a missing file is reported as missing, not skipped silently. Source-px figures:
TMC-2 fine frame x 1.501 (factor_worst, as the audit's probe percentiles); OHRC/IIRS as the scripts'
to_source_px already wrote them (probe_check.*_px_src.worst).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.evaluate.quality import TIER_ORDER, accuracy_tier  # noqa: E402

TMC_FACTOR = 1.501


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def worst(*tiers):
    return max((t for t in tiers if t), key=TIER_ORDER.index)


def c02(stages) -> bool | None:
    rm = ((stages or {}).get("subpixel") or {}).get("residual_to_model_px") or {}
    if rm.get("refined") is None or rm.get("unrefined") is None:
        return None
    return rm["refined"] < rm["unrefined"]


def tmc2_tc(d: Path, parts=("c0", "cm", "cp"), prefix="tmc2_tc") -> dict:
    rows = []
    for p in parts:
        j = load(d / f"{prefix}_{p}.json")
        rows += (j or {}).get("rows", [])
    if not rows:
        return {"missing": True}
    old = {(w["tmc_row"], w.get("tmc_col")): w
           for w in load(ROOT / "reports/tmc2_tc_registration_height_ref.json")["rows"]}
    out, pooled50, pooled95 = [], [], []
    for w in rows:
        k = (w["tmc_row"], w.get("tmc_col"))
        pc = (w.get("probe_check") or {}).get("delivered_frame_px") or {}
        acc = w.get("accuracy_fine_frame") or {}
        p50 = None if pc.get("p50") is None else round(pc["p50"] * TMC_FACTOR, 3)
        p95 = None if pc.get("p95") is None else round(pc["p95"] * TMC_FACTOR, 3)
        allowed = accuracy_tier({"n": (w.get("probe_check") or {}).get("n", 0), "p50_px_src": p50,
                                 "p95_px_src": p95})[0] if p50 is not None else "MEDIUM"
        xc = w.get("crosscheck") or {}
        rec = {"window": list(k), "status": w.get("status"), "tier_old": old.get(k, {}).get("tier"),
               "tier": w.get("tier"), "tier_with_accuracy": worst(w.get("tier"), allowed) if w.get("tier") else None,
               "geometry": acc.get("geometry"), "probe_p50_tmc_px": p50, "probe_p95_tmc_px": p95,
               "checkpoint_rmse_fine_px": acc.get("checkpoint_rmse_px_ref"),
               "ckpt_over_probe_p50": None if not (acc.get("checkpoint_rmse_px_ref") and pc.get("p50"))
               else round(acc["checkpoint_rmse_px_ref"] / pc["p50"], 2),
               "c02_refinement_lowered_residual": c02(w.get("pipeline")),
               "crosscheck": xc.get("verdict"), "crosscheck_gap_px": xc.get("gap_px"),
               "crosscheck_affine_gap_px": xc.get("affine_gap_px"),
               "coarse_z": w.get("coarse_z"), "seconds": w.get("seconds")}
        out.append(rec)
        if p50 is not None:
            pooled50.append(p50); pooled95.append(p95)
    n = len(out)
    return {"windows": out, "summary": {
        "windows": n, "registered": sum(r["status"] == "registered" for r in out),
        "tiers_changed": [r["window"] for r in out if r["tier"] != r["tier_old"]],
        "probe_p95_over_1_tmc_px": sum(1 for r in out if (r["probe_p95_tmc_px"] or 0) > 1.0),
        "probe_p50_tmc_px_range": [min(pooled50), max(pooled50)] if pooled50 else None,
        "probe_p95_tmc_px_range": [min(pooled95), max(pooled95)] if pooled95 else None,
        "c02_lowered": sum(bool(r["c02_refinement_lowered_residual"]) for r in out),
        "c03_within_1_5x": sum(1 for r in out if r["ckpt_over_probe_p50"] and 1 / 1.5 <= r["ckpt_over_probe_p50"] <= 1.5),
        "crosscheck": {v: sum(r["crosscheck"] == v for r in out) for v in ("agree", "flag", "inconclusive", None)},
        "tier_with_accuracy": {t: sum(r["tier_with_accuracy"] == t for r in out) for t in TIER_ORDER}}}


def method_windows(d: Path, name: str, key, old_path: str, old_pick) -> dict:
    j = load(d / f"{name}.json")
    if j is None:
        return {"missing": True}
    old = {key(w): old_pick(w) for w in load(ROOT / old_path)["windows"]}
    out = []
    for w in j["windows"]:
        for m, r in (w.get("results") or {}).items():
            pc = r.get("probe_check") or {}
            xc = r.get("crosscheck") or {}
            p50 = (pc.get("p50_px_src") or {}).get("worst")
            p95 = (pc.get("p95_px_src") or {}).get("worst")
            allowed = accuracy_tier({"n": pc.get("n", 0), "p50_px_src": p50, "p95_px_src": p95})[0]
            ow = old.get(key(w)) or {}
            out.append({"window": str(key(w)), "matcher": m, "used": r.get("used"), "status": r.get("status"),
                        "tier": r.get("tier"), "outcome": r.get("outcome"), "success": r.get("success"),
                        "old_success": (ow.get(m) or {}).get("success") if isinstance(ow, dict) else None,
                        "mi_flag": (r.get("mi_check") or {}).get("flag"),
                        "tier_with_accuracy": worst(r.get("tier"), allowed) if r.get("tier") else None,
                        "probe_n": pc.get("n"), "probe_p50_src_px": p50, "probe_p95_src_px": p95,
                        "known_shift_px": r.get("known_shift_error_px"),
                        "c02_refinement_lowered_residual": c02(r.get("pipeline")),
                        "crosscheck": xc.get("verdict"), "crosscheck_gap_px": xc.get("gap_px"),
                        "crosscheck_affine_gap_px": xc.get("affine_gap_px"),
                        "geometry": (r.get("accuracy_fine_frame") or {}).get("geometry"),
                        "seconds": r.get("seconds")})
    by_m = {}
    for m in sorted({r["matcher"] for r in out}):
        rs = [r for r in out if r["matcher"] == m]
        c = [r["c02_refinement_lowered_residual"] for r in rs if r["c02_refinement_lowered_residual"] is not None]
        p95 = [r["probe_p95_src_px"] for r in rs if r["probe_p95_src_px"] is not None]
        by_m[m] = {"windows": len(rs), "success": sum(bool(r["success"]) for r in rs),
                   "unconfirmed": sum(r["outcome"] == "unconfirmed" for r in rs),
                   "old_success": sum(bool(r["old_success"]) for r in rs),
                   "c02_lowered": f"{sum(c)}/{len(c)}",
                   "probe_windows": len(p95), "probe_p95_src_px_median": None if not p95 else round(float(np.median(p95)), 3),
                   "crosscheck": {v: sum(r["crosscheck"] == v for r in rs) for v in ("agree", "flag", "inconclusive", None)},
                   "tier_with_accuracy": {t: sum(r["tier_with_accuracy"] == t for r in rs) for t in TIER_ORDER}}
    return {"method_windows": out, "summary": by_m}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmc-prefix", default="tmc2_tc")
    args = ap.parse_args()
    d = args.batch_dir
    res = {"batch_dir": str(d), "tmc2_tc": tmc2_tc(d, prefix=args.tmc_prefix),
           "ohrc_nac": method_windows(d, "ohrc_nac", lambda w: (w["nac"], w["ohrc_row"]),
                                      "reports/ohrc_nac_q8_auto_bridge.json", lambda w: w.get("results") or {}),
           "iirs_wac": method_windows(d, "iirs_wac", lambda w: w["iirs_line0"],
                                      "reports/iirs_wac_mosaic.json", lambda w: w.get("results") or {})}
    lp = load(d / "tmc2_tc_c0_labelprior.json")
    if lp is not None:                                   # I-17: refined-label prior, 1.5 km search vs 7 km
        base = {w["tmc_row"]: w for w in (load(d / "tmc2_tc_c0.json") or {}).get("rows", [])}
        res["i17_label_prior"] = [{"row": w["tmc_row"], "status": w.get("status"), "tier": w.get("tier"),
                                   "coarse_z": w.get("coarse_z"), "base_coarse_z": base.get(w["tmc_row"], {}).get("coarse_z"),
                                   "offset_m": w.get("system_offset_m"),
                                   "base_offset_m": base.get(w["tmc_row"], {}).get("system_offset_m")}
                                  for w in lp.get("rows", [])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: (v.get("summary") if isinstance(v, dict) else v) for k, v in res.items()
                      if k != "batch_dir"}, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
