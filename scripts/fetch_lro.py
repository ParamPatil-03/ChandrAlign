"""Find and download LRO NAC / WAC reference products covering a ground box.

Feature DATA-12 (ODE search) + DATA-06/DATA-07 (NAC/WAC reference support).
Uses NASA's PDS Orbital Data Explorer REST API, which needs no login.

Search only:
    .venv/Scripts/python scripts/fetch_lro.py --scene ch2_ohr_nrp_20240330T0035085365_d_img_d18
    .venv/Scripts/python scripts/fetch_lro.py --bbox -0.3 0.3 23.2 23.8 --product NAC

Download the top N results:
    .venv/Scripts/python scripts/fetch_lro.py --scene <id> --download 2

Products land in data/raw/lro/<nac|wac>/, each with its ODE catalogue record saved
beside it as ode_metadata.json: WAC labels carry no corners and NAC CDR labels no
sun angles, so without that record a product has no footprint and
scripts/build_pairs.py stops on it. Run scripts/make_manifest.py afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FOOTPRINTS = ROOT / "data" / "pairs" / "ch2_footprints.json"
ODE = "https://oderest.rsl.wustl.edu/live2/"

# ODE product types, as listed by query=iipy for target=moon.
# CDR = radiometrically calibrated, EDR = raw. We default to CDR because the
# reference image should already be calibrated; EDR stays selectable for the
# anti-stub test, which needs a raw product with a detached label.
PRODUCT_TYPES = {
    "NAC": ["CDRNAC4"],
    "NAC-EDR": ["EDRNAC4"],
    "WAC": ["CDRWAM4"],
    "WAC-EDR": ["EDRWAM4"],
}
IHID, IID = "LRO", "LROC"


def to_360(lon: float) -> float:
    """ODE expresses longitude in 0..360 east; our shapefiles use -180..180."""
    return lon % 360.0


def scene_bbox(scene_id: str, pad_deg: float) -> tuple[float, float, float, float]:
    """Ground box of a CH-2 product from the footprint index, padded by pad_deg."""
    if not FOOTPRINTS.exists():
        sys.exit("run scripts/ch2_footprints.py first to build the footprint index")
    catalog = json.loads(FOOTPRINTS.read_text(encoding="utf-8"))
    for products in catalog.values():
        for p in products:
            if p["product_id"] == scene_id:
                min_lon, min_lat, max_lon, max_lat = p["bbox"]
                return (min_lat - pad_deg, max_lat + pad_deg,
                        min_lon - pad_deg, max_lon + pad_deg)
    sys.exit(f"scene {scene_id} not found in the footprint index")


def query(bbox: tuple[float, float, float, float], ptype: str, limit: int) -> list[dict]:
    min_lat, max_lat, min_lon, max_lon = bbox
    results = []
    for ode_type in PRODUCT_TYPES[ptype]:
        params = {
            "query": "product", "target": "moon", "results": "fmp",
            "output": "JSON", "ihid": IHID, "iid": IID, "pt": ode_type, "limit": limit,
            "minlat": min_lat, "maxlat": max_lat,
            "westernlon": to_360(min_lon), "easternlon": to_360(max_lon),
        }
        r = requests.get(ODE, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json().get("ODEResults", {})
        if payload.get("Status", "").lower() != "success":
            print(f"  {ode_type}: {payload.get('Error', 'no results')}")
            continue
        products = payload.get("Products", {}).get("Product", [])
        if isinstance(products, dict):
            products = [products]
        for p in products:
            files = p.get("Product_files", {}).get("Product_file", [])
            if isinstance(files, dict):
                files = [files]
            results.append(
                {
                    "product_id": p.get("pdsid"),
                    "type": ode_type,
                    "start_utc": p.get("UTC_start_time"),
                    "incidence_deg": p.get("Incidence_angle"),
                    "emission_deg": p.get("Emission_angle"),
                    "phase_deg": p.get("Phase_angle"),
                    "bbox": [p.get("Minimum_latitude"), p.get("Maximum_latitude"),
                             p.get("Westernmost_longitude"), p.get("Easternmost_longitude")],
                    "files": [
                        {"name": f.get("FileName"), "url": f.get("URL"), "bytes": f.get("KBytes")}
                        for f in files
                        if str(f.get("FileName", "")).upper().endswith((".IMG", ".LBL", ".XML"))
                    ],
                }
            )
    return results


def _remote_size(url: str) -> Optional[int]:
    """Bytes the server will actually send, or None if it will not say.

    A HEAD costs nothing next to a 500 MB GET and is the only size either side
    can agree on; catalogue metadata is a description of the file, not the file.
    """
    try:
        h = requests.head(url, timeout=60, allow_redirects=True)
        h.raise_for_status()
        length = h.headers.get("Content-Length")
        return int(length) if length is not None else None
    except (requests.RequestException, ValueError):
        return None


def download(entry: dict, ptype: str) -> None:
    out_dir = ROOT / "data" / "raw" / "lro" / ptype.lower() / str(entry["product_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in entry["files"]:
        if not f["url"]:
            continue
        dest = out_dir / f["name"]

        # SIZE TRUTH: the HTTP Content-Length of the actual transfer, never the
        # ODE listing. ODE's "KBytes" field is kilobytes-of-1000, so reading it
        # as 1024 overstates every file by exactly 1.024 and makes complete
        # downloads look 2.4% short. Measured on two products:
        #   M1417360906LC.IMG  ODE 541,623,296  vs  Content-Length 528,929,736
        #   M1415013176LC.IMG  ODE 414,183,424  vs  Content-Length 404,476,872
        # Both ratios are 1.0240 to four decimals.
        want = _remote_size(f["url"])

        # Resume-safety: a file that merely EXISTS is not a file that finished.
        # An interrupted download leaves a short file behind, and skipping on
        # existence alone would hand that truncated product to the pipeline as
        # if it were complete -- silent data corruption, not merely a slow path.
        if dest.exists():
            have = dest.stat().st_size
            if want is None:
                print(f"    have {f['name']} (size unverified: no Content-Length)")
                continue
            if have != want:
                print(f"    redo {f['name']} (have {have/1e6:.1f} MB of {want/1e6:.1f} MB)")
                dest.unlink()
            else:
                print(f"    have {f['name']}")
                continue

        print(f"    get  {f['name']} ({(want or 0) / 1e6:.1f} MB)")
        # Download to a .part file and rename only on success, so an interrupted
        # run can never leave behind something that looks finished.
        part = dest.with_suffix(dest.suffix + ".part")
        with requests.get(f["url"], stream=True, timeout=600) as r:
            r.raise_for_status()
            declared = r.headers.get("Content-Length")
            declared = int(declared) if declared is not None else want
            with part.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        got = part.stat().st_size
        if declared is not None and got != declared:
            part.unlink()
            raise RuntimeError(f"{f['name']}: got {got} bytes, server declared {declared}")
        part.replace(dest)

    # The catalogue record is part of the product as far as this pipeline is
    # concerned: footprint_of() needs it whenever the label has no corners.
    from chandralign.io.ode_client import find_saved_record, save_product_record
    if find_saved_record(out_dir) is None:
        save_product_record(str(entry["product_id"]), out_dir)
        print(f"    saved {out_dir.name}/ode_metadata.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="CH-2 product_id to take the ground box from")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLAT", "MAXLAT", "MINLON", "MAXLON"))
    ap.add_argument("--product", choices=sorted(PRODUCT_TYPES), default="NAC")
    ap.add_argument("--pad", type=float, default=0.05, help="degrees of padding around the scene")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--download", type=int, default=0, help="download the top N results")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="download these product ids instead of the top N")
    args = ap.parse_args()

    if args.scene:
        bbox = scene_bbox(args.scene, args.pad)
    elif args.bbox:
        bbox = tuple(args.bbox)
    else:
        sys.exit("give --scene or --bbox")

    print(f"searching {args.product} over lat {bbox[0]:.3f}..{bbox[1]:.3f}, "
          f"lon {bbox[2]:.3f}..{bbox[3]:.3f}")
    results = query(bbox, args.product, args.limit)
    print(f"\n{len(results)} products\n")
    print(f"{'product':28} {'type':9} {'inc':>6} {'emi':>6} {'start':22} files")
    for e in results:
        inc = float(e["incidence_deg"]) if e["incidence_deg"] else float("nan")
        emi = float(e["emission_deg"]) if e["emission_deg"] else float("nan")
        print(f"{str(e['product_id']):28} {e['type']:9} {inc:6.1f} {emi:6.1f} "
              f"{str(e['start_utc'])[:22]:22} {len(e['files'])}")

    out = ROOT / "data" / "pairs" / f"lro_{args.product.lower()}_candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bbox": bbox, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")

    wanted = ([e for e in results if e["product_id"] in set(args.ids)] if args.ids
              else results[: args.download])
    for e in wanted:
        print(f"\n{e['product_id']}:")
        download(e, args.product)


if __name__ == "__main__":
    main()
