"""Index the Chandrayaan-2 archive footprints and find regions all 3 cameras cover.

Reads the PRADAN shapefiles in data/raw/ch2/*/shapefiles/ (feature GEO-04), builds
one polygon per archived product, and reports the OHRC scenes that also have TMC-2
and IIRS coverage. Those regions are where we fetch LRO NAC/WAC and SELENE TC, so
we never download reference imagery for ground we have no source image of.

Usage:  .venv/Scripts/python scripts/ch2_footprints.py [--min-overlap 0.05]

Writes data/pairs/ch2_footprints.json (every product) and
       data/pairs/ch2_triple_coverage.json (OHRC scenes with TMC-2 + IIRS overlap).

Caveat: polygons are built in plain lat/lon degrees. Scenes crossing the +/-180 deg
meridian are flagged `antimeridian: true` and excluded from overlap scoring rather
than silently mis-measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import shapefile
from shapely.geometry import Polygon
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "pairs"

SOURCES = {
    "OHRC": "data/raw/ch2/ohrc/shapefiles/ohr_r1_r11_shape_ver4/ch2_ohr_raw.shp",
    "TMC2": "data/raw/ch2/tmc2/shapefiles/tmc2_s1_s14_v1_shape/ch2_tmc_raw.shp",
    "IIRS": "data/raw/ch2/iirs/shapefiles/iirs_s1_s12_v2_shape/ch2_iir_raw.shp",
}


def load(instrument: str, rel_path: str) -> list[dict]:
    reader = shapefile.Reader(str(ROOT / rel_path))
    names = [f[0] for f in reader.fields[1:]]
    products = []
    for shape_rec in reader.iterShapeRecords():
        rec = dict(zip(names, shape_rec.record))
        pts = list(shape_rec.shape.points)
        if len(pts) < 4:
            continue
        lons = [p[0] for p in pts]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        products.append(
            {
                "instrument": instrument,
                "product_id": rec["PRODUCT_ID"],
                "start_utc": rec["OBS_ST_TIM"],
                "end_utc": rec["OBS_ED_TIM"],
                "download": rec["DOWNLOAD"],
                "corners": [
                    [rec["UL_LAT"], rec["UL_LON"]],
                    [rec["UR_LAT"], rec["UR_LON"]],
                    [rec["BR_LAT"], rec["BR_LON"]],
                    [rec["BL_LAT"], rec["BL_LON"]],
                ],
                "bbox": [round(v, 4) for v in poly.bounds],
                "antimeridian": max(lons) - min(lons) > 180.0,
                "_poly": poly,
            }
        )
    return products


def overlaps(target: dict, candidates: list[dict], tree: STRtree, min_overlap: float) -> list[dict]:
    """Candidates intersecting `target`, scored by fraction of the target covered."""
    hits = []
    for idx in tree.query(target["_poly"]):
        cand = candidates[idx]
        if cand["antimeridian"]:
            continue
        inter = target["_poly"].intersection(cand["_poly"]).area
        if inter <= 0 or target["_poly"].area <= 0:
            continue
        frac = inter / target["_poly"].area
        if frac >= min_overlap:
            hits.append(
                {
                    "product_id": cand["product_id"],
                    "download": cand["download"],
                    "start_utc": cand["start_utc"],
                    "overlap_fraction": round(frac, 4),
                }
            )
    return sorted(hits, key=lambda h: -h["overlap_fraction"])


def strip(p: dict) -> dict:
    return {k: v for k, v in p.items() if k != "_poly"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-overlap", type=float, default=0.05,
                    help="minimum fraction of the OHRC scene a candidate must cover")
    ap.add_argument("--top", type=int, default=15, help="rows to print")
    args = ap.parse_args()

    catalog = {name: load(name, path) for name, path in SOURCES.items()}
    for name, products in catalog.items():
        skew = sum(p["antimeridian"] for p in products)
        print(f"{name:5} {len(products):5} products" + (f"  ({skew} cross the antimeridian)" if skew else ""))

    trees = {
        name: STRtree([p["_poly"] for p in products])
        for name, products in catalog.items()
        if name != "OHRC"
    }

    triples = []
    for ohrc in catalog["OHRC"]:
        if ohrc["antimeridian"]:
            continue
        tmc2 = overlaps(ohrc, catalog["TMC2"], trees["TMC2"], args.min_overlap)
        iirs = overlaps(ohrc, catalog["IIRS"], trees["IIRS"], args.min_overlap)
        if tmc2 and iirs:
            lat = (ohrc["bbox"][1] + ohrc["bbox"][3]) / 2
            lon = (ohrc["bbox"][0] + ohrc["bbox"][2]) / 2
            triples.append(
                {
                    "ohrc": strip(ohrc),
                    "centre_latlon": [round(lat, 4), round(lon, 4)],
                    "abs_latitude": round(abs(lat), 2),
                    "tmc2_count": len(tmc2),
                    "iirs_count": len(iirs),
                    "tmc2": tmc2[:5],
                    "iirs": iirs[:5],
                }
            )

    # Equatorial scenes first: LRO NAC coverage is densest there, and the Easy
    # benchmark tier is defined as near-equatorial (PLAN.md section 12.1).
    triples.sort(key=lambda t: (t["abs_latitude"], -t["tmc2_count"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ch2_footprints.json").write_text(
        json.dumps({k: [strip(p) for p in v] for k, v in catalog.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "ch2_triple_coverage.json").write_text(
        json.dumps({"min_overlap": args.min_overlap, "count": len(triples), "scenes": triples}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{len(triples)} OHRC scenes with both TMC-2 and IIRS overlap "
          f"(>= {args.min_overlap:.0%}), nearest the equator first:\n")
    print(f"{'OHRC product':46} {'lat':>8} {'lon':>9} {'TMC2':>5} {'IIRS':>5}")
    for t in triples[: args.top]:
        lat, lon = t["centre_latlon"]
        print(f"{t['ohrc']['product_id']:46} {lat:8.2f} {lon:9.2f} {t['tmc2_count']:5} {t['iirs_count']:5}")
    print(f"\nwrote {OUT_DIR.relative_to(ROOT)}/ch2_footprints.json and ch2_triple_coverage.json")


if __name__ == "__main__":
    main()
