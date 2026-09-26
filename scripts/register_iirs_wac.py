"""IIRS -> LRO WAC on real data: the second credibility-floor pairing, cross-modal.

    python scripts/register_iirs_wac.py

Protocol, frozen before this script existed: docs/iirs_wac_protocol.md.

IIRS enters as its PREP-06 band composite. WAC labels carry no geometry, so WAC pixel ->
ground comes from ODE's four-corner footprint, whose vertex order relative to the image
is unknown: all 8 orientations are tried in the MIND coarse lock (highest z kept, if clear
of the runner-up). Then the fine stage runs each matcher on the coarse-aligned pair,
followed by pipeline.fine_stage, all five control gates and quality.assess.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from shapely.geometry import Point, Polygon  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.dem import find_tiles  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.io.evidence import evidence_label  # noqa: E402
from chandralign.matching import routing  # noqa: E402
from chandralign.pipeline import stage_flags  # noqa: E402
from chandralign.preprocess.iirs_composite import product_band_selection  # noqa: E402
from chandralign.workflows.ohrc_nac import norm  # noqa: E402

# The window registration, WAC geometry and mosaic placement moved VERBATIM to
# src/chandralign/workflows/iirs_wac.py (audit 2026-09-26), so the CLI and API run the same code.
from chandralign.workflows.iirs_wac import (  # noqa: E402,F401
    CONSISTENT_M, LINES, MIN_Z, N_WIN, PRODUCTS, WAC_FILL, Z_MARGIN, MosaicGeo, Offset, WacGeo,
    load_mosaic, mosaic_windows, polygon_of, run_window)

DUMP_DIR = None     # --dump-points: fine frames of the xoftr result per window (docs/mi_protocol.md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS))
    ap.add_argument("--matchers", nargs="+", default=["xoftr", "minima-loftr", "rift2", "sift"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/iirs_wac_registration.json")
    ap.add_argument("--reference", choices=["cdr", "mosaic"], default="cdr",
                    help="cdr: raw WAC frames (run 1); mosaic: the map-projected WAC mosaic clip (amendment 1)")
    ap.add_argument("--dump-points", default=None, help="folder: save the xoftr fine frames per window")
    ap.add_argument("--benchmark-models", action="store_true",
                    help="allow unaudited / benchmark-only matchers (G-03); recorded as ship_mode false")
    args = ap.parse_args()
    if args.benchmark_models:
        from chandralign.matching import licence as _licence
        _licence.enable_benchmark_mode()
    global DUMP_DIR
    if args.dump_points:
        DUMP_DIR = args.dump_points
        Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)

    iirs = parse_label(evidence_label("IIRS", ROOT))
    im = projection.load_corner_model(iirs, corners="system")
    assert im.independent_of_references
    sel = product_band_selection(iirs)
    choice = routing.choose("IIRS", "WAC")
    stages = stage_flags()
    dem_tiles = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")
    L, S = iirs.array_shape
    rows = np.arange(0, L - LINES, 64)
    lat_all, lon_all = im.pixel_to_latlon(rows + LINES / 2, np.full(rows.shape, S / 2))
    windows = []
    if args.reference == "mosaic":
        mmeta, wimg, wok, geo = load_mosaic(ROOT)
        wac = type("Ref", (), {"product_id": "WAC_GLOBAL_MOSAIC_100M"})()
        args.products = ["WAC_GLOBAL_MOSAIC_100M"]
        picks = mosaic_windows(iirs, im, geo, wimg)
        print(f"mosaic: {len(picks)} IIRS windows picked inside the clip", flush=True)
        for r0 in picks:
            t = time.perf_counter()
            wdw = run_window(iirs, im, sel, wac, wimg, wok, [geo], r0, args.matchers, args.device, stages, dem_tiles,
                             dict(choice.fine_stage_options), dump_dir=DUMP_DIR)
            wdw["wac"] = "WAC_GLOBAL_MOSAIC_100M"; wdw["seconds"] = round(time.perf_counter() - t, 1)
            windows.append(wdw)
            print(json.dumps({k: wdw.get(k) for k in ("iirs_line0", "status", "coarse")} |
                             {"ok": {m: (v.get("status", "")[:20], v.get("tier"), v.get("known_shift_error_px"))
                                     for m, v in (wdw.get("results") or {}).items()}}), flush=True)
    for pid in ([] if args.reference == "mosaic" else args.products):
        xml = next((ROOT / "data/raw/lro/wac").rglob(f"{pid}.XML"))
        wac = parse_label(xml)
        rec = json.loads((xml.parent / "ode_metadata.json").read_text(encoding="utf-8"))["record"]
        poly = polygon_of(rec)
        raw = pds_raster.read_raster(wac).astype(np.float32)
        raw = raw if raw.ndim == 2 else raw[0]
        wok = np.isfinite(raw) & (raw > WAC_FILL)
        wimg = norm(np.where(wok, raw, 0), wok)
        h, w = wimg.shape
        geos = [WacGeo(poly, w, h, k, m) for k in range(4) for m in (False, True)]
        fp = Polygon([(lo, la) for lo, la in poly])
        margin = 0.3                                                  # deg, ~9 km: keep windows inside
        inside = rows[[fp.buffer(-margin).contains(Point(lo, la)) for la, lo in zip(lat_all, lon_all)]]
        print(f"{pid}: {len(inside)} candidate IIRS windows inside the footprint", flush=True)
        if len(inside) == 0:
            continue
        picks = np.linspace(inside.min(), inside.max(), N_WIN).astype(int)
        picks = [int(inside[np.argmin(np.abs(inside - p))]) for p in picks]
        for r0 in picks:
            t = time.perf_counter()
            wdw = run_window(iirs, im, sel, wac, wimg, wok, geos, r0, args.matchers, args.device, stages, dem_tiles,
                             dict(choice.fine_stage_options), dump_dir=DUMP_DIR)
            wdw["wac"] = pid; wdw["seconds"] = round(time.perf_counter() - t, 1)
            windows.append(wdw)
            print(json.dumps({k: wdw.get(k) for k in ("wac", "iirs_line0", "status", "coarse")} |
                             {"ok": {m: (v.get("status", "")[:20], v.get("tier"), v.get("known_shift_error_px"))
                                     for m, v in (wdw.get("results") or {}).items()}}), flush=True)
    summary = {}
    for pid in args.products:
        ws = [w for w in windows if w["wac"] == pid]
        for name in args.matchers:
            gp = [(w["results"][name]) for w in ws if (w.get("results") or {}).get(name, {}).get("gates_pass")
                  and w["results"][name].get("tier_ok")]
            med = (np.median([[r["implied_offset_m"]["east"], r["implied_offset_m"]["north"]] for r in gp], axis=0)
                   if len(gp) >= 3 else None)
            ok = 0
            for w in ws:
                r = (w.get("results") or {}).get(name, {})
                io = r.get("implied_offset_m")
                s = bool(r.get("status") == "registered" and r.get("gates_pass") and r.get("tier_ok") and med is not None
                         and io and np.hypot(io["east"] - med[0], io["north"] - med[1]) <= CONSISTENT_M)
                r["success"] = s; ok += s
            rate = ok / len(ws) if ws else 0.0
            summary.setdefault(pid, {})[name] = {"success": ok, "windows": len(ws),
                                                 "verdict": "solved" if rate >= 0.9 else "degraded" if rate >= 0.6 else "unsolved"}
    print(json.dumps(summary, indent=1))
    (ROOT / args.out).write_text(json.dumps({"source": "measured", "protocol": "docs/iirs_wac_protocol.md",
                                             "iirs": iirs.product_id, "route": choice.as_provenance(),
                                             "pipeline_stages": stages, "run": run_record(),
                                             "summary": summary, "windows": windows}, indent=2, default=str),
                                 encoding="utf-8")
    print(f"wrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
