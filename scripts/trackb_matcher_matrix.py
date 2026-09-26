"""The real-data matcher x pairing matrix for G-03 / G-09: every matcher, every real pairing, one table.

    python scripts/trackb_matcher_matrix.py <scratch root holding batch*/out> --out reports/trackb_matcher_matrix.json

Reads the Track B batch outputs (docs/TRACKB_FINAL_REPORT.md lists which batch produced which cell). A cell is
missing, not zero, when its run has not been made. Per cell: windows, successes by that pairing's own rule
(OHRC: the I-07 rule; TMC-2 -> TC: status registered and tier not REJECTED; IIRS/MI: the script's `success`),
unconfirmed (OHRC, MI-flagged), tiers, probe p50/p95 medians in SOURCE px, cross-check verdicts, mean s/window,
and errors (runs that crashed on a window count as errors, never as failures of the matcher).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

TMC_FACTOR = 1.501
MATCHERS = ["eloftr", "minima-loftr", "xoftr", "roma", "minima-roma", "matchanything-roma", "ufm"]


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _med(v):
    v = [x for x in v if x is not None]
    return None if not v else round(float(np.median(v)), 3)


def tmc_cell(files):
    rows = []
    for f in files:
        d = _load(f)
        rows += (d or {}).get("rows", [])
    if not rows:
        return None
    p50 = [((w.get("probe_check") or {}).get("delivered_frame_px") or {}).get("p50") for w in rows]
    p95 = [((w.get("probe_check") or {}).get("delivered_frame_px") or {}).get("p95") for w in rows]
    tiers = collections.Counter(w.get("tier") for w in rows)
    return {"windows": len(rows),
            "success": sum(w.get("status") == "registered" and w.get("tier") not in (None, "REJECTED") for w in rows),
            "tiers": dict(tiers), "errors": sum(str(w.get("status", "")).startswith("error") for w in rows),
            "probe_p50_src_px": _med([x * TMC_FACTOR for x in p50 if x is not None]),
            "probe_p95_src_px": _med([x * TMC_FACTOR for x in p95 if x is not None]),
            "crosscheck": dict(collections.Counter((w.get("crosscheck") or {}).get("verdict") for w in rows)),
            "mean_s": _med([w.get("seconds") for w in rows])}


def window_cell(path, matcher, key="windows"):
    d = _load(path)
    if d is None:
        return None
    rs = [(w.get("results") or {}).get(matcher) for w in d.get(key, [])]
    rs = [r for r in rs if r]
    if not rs:
        return None
    pc = [r.get("probe_check") or {} for r in rs]
    return {"windows": len(rs), "success": sum(bool(r.get("success")) for r in rs),
            "unconfirmed": sum(r.get("outcome") == "unconfirmed" for r in rs),
            "tiers": dict(collections.Counter(r.get("tier") for r in rs)),
            "errors": sum(str(r.get("status", "")).startswith("error") for r in rs),
            "probe_p50_src_px": _med([(p.get("p50_px_src") or {}).get("worst") if isinstance(p.get("p50_px_src"), dict)
                                      else p.get("p50_px_src") for p in pc]),
            "probe_p95_src_px": _med([(p.get("p95_px_src") or {}).get("worst") if isinstance(p.get("p95_px_src"), dict)
                                      else p.get("p95_px_src") for p in pc]),
            "crosscheck": dict(collections.Counter((r.get("crosscheck") or {}).get("verdict") for r in rs)),
            "mean_s": _med([r.get("seconds") for r in rs])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    R = str(args.root)
    b = lambda n: f"{R}/{n}/out"  # noqa: E731
    tc = lambda *parts: [p for pat in parts for p in sorted(glob.glob(pat))]  # noqa: E731
    cells = {
        "tmc2_tc": {
            "eloftr": tmc_cell(tc(f"{b('batch2')}/c04_tmc2_tc_*.json")),
            "minima-loftr": tmc_cell(tc(f"{b('batch6')}/tc_minima-loftr_*.json")),
            "xoftr": tmc_cell(tc(f"{b('batch6')}/g09_xoftr_tmc2_tc_c0_r*.json", f"{b('batch5')}/g09_xoftr_tmc2_tc_c[mp].json")),
            "minima-roma": tmc_cell(tc(f"{b('batch2')}/g03_R2_tmc2_tc_c0.json", f"{b('batch6')}/tc_minima-roma_c[mp].json")),
            **{m: tmc_cell(tc(f"{b('batch6')}/tc_{m}_*.json")) for m in ("roma", "matchanything-roma", "ufm")}},
        "ohrc_nac": {
            "eloftr": window_cell(f"{b('batch2')}/g03_R1_ohrc.json", "eloftr"),
            "minima-loftr": window_cell(f"{b('batch')}/ohrc_nac.json", "minima-loftr"),
            "routed (eloftr -> minima-loftr)": window_cell(f"{b('batch')}/ohrc_nac.json", "routed"),
            "xoftr": window_cell(f"{b('batch5')}/g09_xoftr_ohrc.json", "xoftr"),
            "roma": window_cell(f"{b('batch2b')}/g03_R1_ohrc_roma_all.json", "roma"),
            "minima-roma": window_cell(f"{b('batch2')}/g03_R1_ohrc.json", "minima-roma"),
            **{m: window_cell(f"{b('batch6')}/ohrc_{m}.json", m) for m in ("matchanything-roma", "ufm")}},
        "iirs_wac": {
            "eloftr": window_cell(f"{b('batch6')}/iirs_eloftr.json", "eloftr"),
            "minima-loftr": window_cell(f"{b('batch')}/iirs_wac.json", "minima-loftr"),
            "xoftr": window_cell(f"{b('batch2')}/g03_R3_iirs_wac.json", "xoftr"),
            "minima-roma": window_cell(f"{b('batch2')}/g03_R3_iirs_wac.json", "minima-roma"),
            "matchanything-roma": window_cell(f"{b('batch2')}/g03_R3_iirs_wac.json", "matchanything-roma"),
            **{m: window_cell(f"{b('batch6')}/iirs_{m}.json", m) for m in ("roma", "ufm")}},
        "tmc2_mi": {
            **{m: window_cell(f"{b('batch2')}/g03_R4_tmc2_mi.json", m) for m in ("eloftr", "minima-loftr", "minima-roma")},
            **{m: window_cell(f"{b('batch6')}/mi_{m}.json", m) for m in ("xoftr", "roma", "matchanything-roma", "ufm")}},
    }
    args.out.write_text(json.dumps({"source": "measured", "cells": cells}, indent=1, default=str), encoding="utf-8")
    for pairing, row in cells.items():
        print(f"\n== {pairing}")
        for m, c in row.items():
            if c is None:
                print(f"   {m:34s} (not run yet)")
                continue
            print(f"   {m:34s} {c['success']:>2}/{c['windows']:<2} succ  unconf {c.get('unconfirmed', '-')!s:>2}  "
                  f"err {c['errors']}  p50/p95 src {c['probe_p50_src_px']}/{c['probe_p95_src_px']}  "
                  f"tiers {c['tiers']}  xc {c['crosscheck']}  s {c['mean_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
