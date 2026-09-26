"""OHRC -> LRO NAC on real data: the credibility-floor pairing, and a real illumination test.

    python scripts/register_ohrc_nac.py                         # 5 NAC x 5 windows x 3 matchers
    python scripts/register_ohrc_nac.py --products M102014464RC --matchers eloftr

Protocol, frozen before this script existed: docs/ohrc_nac_protocol.md.

Per window: a system-level prior (OHRC's system grid, LROC's NAC corners) gives
orientation and scale; a MIND coarse lock (cascade.register_step_dense) at ~4 m finds
the position over +-4 km; OHRC is then warped (anti-aliased) onto the NAC grid and the
matcher under test runs, followed by pipeline.fine_stage, all five control gates and
quality.assess -- the same fine stage as scripts/register_tmc2_tc.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io.dem import find_tiles  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.io.evidence import evidence_label  # noqa: E402
from chandralign.pipeline import stage_flags  # noqa: E402
from register_tmc2_nac import MOON_R_M, Nac  # noqa: E402

# The window registration moved VERBATIM to src/chandralign/workflows/ohrc_nac.py and the NAC
# geometry to src/chandralign/geometry/nac.py (audit 2026-09-26 C-01), so the CLI and API run the
# same code as this script. Names other scripts import from here are re-exported unchanged.
from chandralign.geometry.nac import NacGeo, Shifted  # noqa: E402,F401
from chandralign.workflows.ohrc_nac import (  # noqa: E402,F401
    BRIDGE_MARGIN_M, COARSE_M, CONSISTENT_M, FINE_MAX_PX, FINE_MIN_M, MARGIN_M, MIN_Z, N_WIN, OHRC_PX,
    PRODUCTS, T, WIN, _fine, affine_fit, bridge_predictions, match, norm, place_windows, run_window)

DUMP_DIR = None               # --dump-points: fine frames, model, inliers, DEM heights (accuracy study)


def main() -> int:
    global WIN, DUMP_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS))
    ap.add_argument("--matchers", nargs="+", default=["eloftr", "minima-loftr", "sift"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/ohrc_nac_registration.json")
    ap.add_argument("--coarse-descriptor", default="mind", choices=["mind", "phase_congruency"])
    ap.add_argument("--coarse-m", type=float, default=COARSE_M, help="coarse-lock resolution, metres")
    ap.add_argument("--coarse-only", action="store_true", help="stop after the coarse lock (protocol Q1)")
    ap.add_argument("--stage", action="append", default=[], metavar="NAME=on|off",
                    help="override a pipeline stage; defaults from configs/default.yaml pipeline.*")
    ap.add_argument("--win", type=int, default=WIN, help="OHRC window side, native px")
    ap.add_argument("--auto-enlarge", action="store_true",
                    help="protocol Q9: re-run a rejected window once at twice the size")
    ap.add_argument("--auto-bridge", action="store_true",
                    help="protocol Q8: use the bridge ONLY for windows whose coarse lock fails")
    ap.add_argument("--bridge", action="store_true",
                    help="protocol Q5: place windows at the geodetic-bridge prediction, no coarse search")
    ap.add_argument("--rows", type=int, nargs="+", default=None,
                    help="force these OHRC window rows (must lie inside the NAC footprint)")
    ap.add_argument("--fresh", action="store_true",
                    help="I-16: windows at the midpoints between the default picks (never overlapping them)")
    ap.add_argument("--dump-points", default=None, help="folder: fine frames + inliers + DEM per window/matcher")
    ap.add_argument("--benchmark-models", action="store_true",
                    help="allow unaudited / benchmark-only matchers (G-03); recorded as ship_mode false")
    args = ap.parse_args()
    if args.benchmark_models:
        from chandralign.matching import licence as _licence
        _licence.enable_benchmark_mode()
    if args.dump_points:
        DUMP_DIR = args.dump_points
        Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)

    ohrc = parse_label(evidence_label("OHRC", ROOT))
    om = projection.load_grid_model(ohrc)
    assert om.independent_of_references
    lroc = {}
    for f in ("tmc2_nac_lroc_meta.json", "iirs_nac_lroc_meta.json"):      # LROC corner metadata, all NACs held
        lroc.update(json.loads((ROOT / "data/pairs" / f).read_text(encoding="utf-8"))["products"])
    sun = json.loads((ROOT / "data/pairs/nac_sun_azimuth_computed.json").read_text(encoding="utf-8"))
    WIN = args.win
    BRIDGE = bridge_predictions(ROOT)
    stages = stage_flags({k: v.lower() in ("on", "1", "true") for k, v in (i.split("=", 1) for i in args.stage)})
    dem_tiles = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")
    # Window placement (amendments 1 and 2) is workflows.ohrc_nac.place_windows.

    windows = []
    for pid in args.products:
        nac = Nac(pid, lroc[pid])
        nacm = parse_label(next((ROOT / "data/raw/lro/nac").rglob(f"{pid}.XML")))
        geo = NacGeo(nac, *nacm.array_shape)
        place_bridge = BRIDGE[pid] if (args.bridge or args.auto_bridge) and pid in BRIDGE else None

        def place(win):
            """{OHRC row: column} of centres whose win-px window lies inside the NAC footprint."""
            return place_windows(ohrc, om, nac, geo, nacm.array_shape, win, bridge=place_bridge,
                                 base_win=int(args.win))

        best = place(WIN)
        big = None                                            # Q10: computed on first need
        inside = np.array(sorted(best))
        print(f"{pid}: {len(inside)} candidate OHRC rows inside the footprint", flush=True)
        if len(inside) == 0:
            continue
        picks = np.linspace(inside.min(), inside.max(), N_WIN).astype(int) if len(inside) >= N_WIN else inside
        picks = [int(inside[np.argmin(np.abs(inside - p))]) for p in picks]
        if args.fresh:                                        # I-16: midpoints between the default picks,
            mids = [(a + b) / 2 for a, b in zip(picks[:-1], picks[1:])]   # snapped to the grid, never
            picks = sorted({int(inside[np.argmin(np.abs(inside - m))]) for m in mids   # overlapping a default
                            if min(abs(inside[np.argmin(np.abs(inside - m))] - q) for q in picks) >= WIN})
        if args.rows:                                         # closure test: force OHRC rows
            picks = [int(r) for r in args.rows if int(r) in best]
        for rc in picks:
            t = time.perf_counter()
            w = run_window(ohrc, om, nacm, geo, nac, rc, best[rc], args.matchers, args.device, stages, dem_tiles,
                           coarse_descriptor=args.coarse_descriptor, coarse_m=args.coarse_m,
                           coarse_only=args.coarse_only,
                           bridge=BRIDGE.get(pid) if args.bridge else None, win=WIN, dump_dir=DUMP_DIR)
            if args.auto_bridge and w.get("status") == "no coarse lock" and pid in BRIDGE:
                failed = w.get("coarse")
                w = run_window(ohrc, om, nacm, geo, nac, rc, best[rc], args.matchers, args.device, stages,
                               dem_tiles, bridge=BRIDGE[pid], win=WIN, dump_dir=DUMP_DIR)
                w["coarse_lock_failed"] = failed
            if args.auto_enlarge:
                good = lambda r: r.get("status") == "registered" and r.get("gates_pass") and r.get("tier_ok")  # noqa: E731
                if not good((w.get("results") or {}).get("routed", {})):
                    base = WIN
                    big = place(2 * base) if big is None else big
                    w2 = {"status": "no 4096 px window fits the strip"}
                    if big:
                        rc2 = min(big, key=lambda r: abs(r - rc))       # Q10: nearest centre that fits
                        WIN = 2 * base
                        try:
                            w2 = run_window(ohrc, om, nacm, geo, nac, rc2, big[rc2], args.matchers, args.device,
                                            stages, dem_tiles,
                                            bridge=BRIDGE.get(pid) if w.get("coarse_lock_failed") else None,
                                            win=WIN, dump_dir=DUMP_DIR)
                        finally:
                            WIN = base
                    if good((w2.get("results") or {}).get("routed", {})):
                        w2["enlarged"] = {"from_px": base, "to_px": 2 * base, "small_result": w.get("results", {}).get("routed")}
                        w = w2
                    else:
                        w["enlarge_tried"] = {"to_px": 2 * base, "status": (w2.get("results") or {}).get("routed", {}).get("status") or w2.get("status")}
            w["sun"] = sun.get(pid)
            w["seconds"] = round(time.perf_counter() - t, 1)
            windows.append(w)
            print(json.dumps({k: w[k] for k in ("nac", "ohrc_row", "status", "coarse") if k in w} |
                             {"ok": {m: (v.get("status"), v.get("tier"), v.get("known_shift_error_px"))
                                     for m, v in (w.get("results") or {}).items()}}), flush=True)

    # consistency (frozen): offset within CONSISTENT_M of the product's median locked offset
    for pid in args.products:
        ws = [w for w in windows if w["nac"] == pid and "system_offset_m" in w]
        if not ws:
            continue
        V = np.array([[w["system_offset_m"]["east"], w["system_offset_m"]["north"]] for w in ws])
        med = np.median(V, axis=0)
        for w, v in zip(ws, V):
            w["offset_from_product_median_m"] = round(float(np.hypot(*(v - med))), 1)
            w["consistent"] = bool(np.hypot(*(v - med)) <= CONSISTENT_M)

    summary = {}
    for pid in args.products:
        ws = [w for w in windows if w["nac"] == pid]
        for name in args.matchers:
            ok = 0
            if args.bridge or args.auto_bridge:
                # Q5/Q8: consistency from each matcher's FINAL transform, over its gate-passing
                # windows; needs >= 3 of them, else no window counts.
                gp = [(w, (w.get("results") or {}).get(name, {})) for w in ws]
                gp = [(w, r) for w, r in gp if r.get("status") == "registered" and r.get("gates_pass")
                      and r.get("tier_ok") and r.get("implied_offset_m")]
                med = (np.median([[r["implied_offset_m"]["east"], r["implied_offset_m"]["north"]] for _, r in gp], axis=0)
                       if len(gp) >= 3 else None)
            unconfirmed = 0
            for w in ws:
                r = (w.get("results") or {}).get(name, {})
                if args.bridge or args.auto_bridge:
                    io = r.get("implied_offset_m")
                    cons = (med is not None and io is not None
                            and np.hypot(io["east"] - med[0], io["north"] - med[1]) <= CONSISTENT_M)
                    r["consistent"] = bool(cons)
                else:
                    cons = w.get("consistent", False)
                s = (r.get("status") == "registered" and r.get("gates_pass") and r.get("tier_ok") and cons)
                mi_flag = (r.get("mi_check") or {}).get("flag") is True
                r["success_before_mi"] = bool(s)
                r["outcome"] = "success" if s and not mi_flag else "unconfirmed" if s else "failed"
                r["success"] = bool(s and not mi_flag)
                ok += r["success"]
                unconfirmed += r["outcome"] == "unconfirmed"
            rate = ok / len(ws) if ws else 0.0
            summary.setdefault(pid, {})[name] = {"success": ok, "unconfirmed_mi_flagged": unconfirmed, "windows": len(ws),
                                                 "verdict": "solved" if rate >= 0.9 else "degraded" if rate >= 0.6 else "unsolved"}
    print(json.dumps(summary, indent=1))
    out = ROOT / args.out
    out.write_text(json.dumps({"source": "measured", "protocol": "docs/ohrc_nac_protocol.md",
                               "ohrc": ohrc.product_id, "pipeline_stages": stages, "run": run_record(),
                               "summary": summary, "windows": windows}, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
