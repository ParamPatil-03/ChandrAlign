"""Download SELENE / Kaguya Terrain Camera tiles covering a ground box.

Feature DATA-08 (SELENE TC reference support). JAXA DARTS has no search API, so
we derive tile names from the 3 deg x 3 deg naming grid and fetch directly. No login.

TMC-2 <-> SELENE TC is the pairing with no published prior attempt, which makes
this the fuel for the project's headline claim (PLAN.md section 1.3).

    .venv/Scripts/python scripts/fetch_selene.py --scene ch2_ohr_nrp_20240330T0035085365_d_img_d18
    .venv/Scripts/python scripts/fetch_selene.py --bbox -0.5 0.5 23.4 23.7 --download

Tile names look like TCO_MAP_02_N03E021N00E024SC: product, version, then the
north-west corner and the south-east corner of a 3 deg tile. Products land in
data/raw/selene/tc/. Run scripts/make_manifest.py afterwards.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
FOOTPRINTS = ROOT / "data" / "pairs" / "ch2_footprints.json"
BASE = "https://data.darts.isas.jaxa.jp/pub/pds3"

# Simple-cylindrical products only, which is everything within +/-60 deg latitude.
# The poles use a different projection and a different naming grid.
BUNDLES = {
    "ortho": ("sln-l-tc-5-ortho-map-v2.0", "TCO_MAP_02"),
    "morning": ("sln-l-tc-5-morning-map-v4.0", "TCO_MAPm04"),
    "evening": ("sln-l-tc-5-evening-map-v4.0", "TCO_MAPe04"),
}
TILE_DEG = 3
MAX_ABS_LAT = 60


def fmt_lat(lat: int) -> str:
    return f"{'S' if lat < 0 else 'N'}{abs(lat):02d}"


def fmt_lon(lon: int) -> str:
    return f"E{lon % 360:03d}"


def tiles_for(bbox: tuple[float, float, float, float]) -> list[str]:
    """Tile stems covering (min_lat, max_lat, min_lon, max_lon)."""
    min_lat, max_lat, min_lon, max_lon = bbox
    if max(abs(min_lat), abs(max_lat)) > MAX_ABS_LAT:
        sys.exit(f"latitudes beyond +/-{MAX_ABS_LAT} deg use the polar grid, not handled here")

    stems = []
    south0 = math.floor(min_lat / TILE_DEG) * TILE_DEG
    west0 = math.floor(min_lon / TILE_DEG) * TILE_DEG
    for south in range(south0, math.ceil(max_lat / TILE_DEG) * TILE_DEG, TILE_DEG):
        for west in range(west0, math.ceil(max_lon / TILE_DEG) * TILE_DEG, TILE_DEG):
            north, east = south + TILE_DEG, west + TILE_DEG
            stems.append(f"{fmt_lat(north)}{fmt_lon(west)}{fmt_lat(south)}{fmt_lon(east)}SC")
    return stems


def scene_bbox(scene_id: str, pad_deg: float) -> tuple[float, float, float, float]:
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


def urls_for(stem: str, bundle: str) -> list[tuple[str, str]]:
    volume, prefix = BUNDLES[bundle]
    lon_band = int(stem.split("E")[1][:3])  # the western longitude in the stem
    vol_dir = f"lon{lon_band // TILE_DEG * TILE_DEG:03d}"
    name = f"{prefix}_{stem}"
    # .img and .lbl are separate files in PDS3; the label alone is useless.
    return [(f"{BASE}/{volume}/{vol_dir}/data/{name}.{ext}", f"{name}.{ext}")
            for ext in ("lbl", "img")]


def head(url: str) -> int | None:
    try:
        r = requests.head(url, timeout=60, allow_redirects=True)
        return int(r.headers.get("Content-Length", 0)) if r.ok else None
    except requests.RequestException:
        return None


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"    have {dest.name}")
        return
    with requests.get(url, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print(f"    got  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="CH-2 product_id to take the ground box from")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLAT", "MAXLAT", "MINLON", "MAXLON"))
    ap.add_argument("--bundle", choices=sorted(BUNDLES), default="ortho",
                    help="ortho = standard mosaic; morning/evening = fixed low-sun illumination")
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    if args.scene:
        bbox = scene_bbox(args.scene, args.pad)
    elif args.bbox:
        bbox = tuple(args.bbox)
    else:
        sys.exit("give --scene or --bbox")

    stems = tiles_for(bbox)
    print(f"{args.bundle}: lat {bbox[0]:.3f}..{bbox[1]:.3f}, lon {bbox[2]:.3f}..{bbox[3]:.3f}"
          f" -> {len(stems)} tile(s)\n")

    out_dir = ROOT / "data" / "raw" / "selene" / "tc"
    for stem in stems:
        print(stem)
        for url, name in urls_for(stem, args.bundle):
            size = head(url)
            if size is None:
                print(f"    MISSING {name}  ({url})")
                continue
            print(f"    {name:34} {size / 1e6:8.1f} MB")
            if args.download:
                fetch(url, out_dir / name)


if __name__ == "__main__":
    main()
