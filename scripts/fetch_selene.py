"""Download SELENE / Kaguya Terrain Camera or Multiband Imager tiles covering a ground box.

Features DATA-08 (SELENE TC) and DATA-09 (SELENE MI). JAXA DARTS has no search
API, so we derive tile names from the naming grid and fetch directly. No login.
TC maps are 3 deg x 3 deg tiles; MI maps are 1 deg x 1 deg.

TMC-2 <-> SELENE TC is the pairing with no published prior attempt, which makes
this the fuel for the project's headline claim (PLAN.md section 1.3).

    .venv/Scripts/python scripts/fetch_selene.py --scene ch2_ohr_nrp_20240330T0035085365_d_img_d18
    .venv/Scripts/python scripts/fetch_selene.py --bbox -0.5 0.5 23.4 23.7 --download
    .venv/Scripts/python scripts/fetch_selene.py --bbox 0.1 0.9 23.1 23.9 --bundle mi --download

Tile names look like TCO_MAP_02_N03E021N00E024SC: product, version, then the
north-west corner and the south-east corner of the tile. MI names follow the same
pattern (MI_MAP_03_N01E023N00E024SC). Products land in data/raw/selene/tc/ or
data/raw/selene/mi/. Files already recorded in data/manifest.json are checked
against their recorded SHA-256 after download. Run scripts/make_manifest.py
afterwards for anything new.
"""
from __future__ import annotations

import argparse
import hashlib
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
# bundle: (DARTS volume, file prefix, tile size deg, folder under data/raw/selene)
BUNDLES = {
    "ortho": ("sln-l-tc-5-ortho-map-v2.0", "TCO_MAP_02", 3, "tc"),
    "morning": ("sln-l-tc-5-morning-map-v4.0", "TCO_MAPm04", 3, "tc"),
    "evening": ("sln-l-tc-5-evening-map-v4.0", "TCO_MAPe04", 3, "tc"),
    # TC DTM: heights on the same 3 deg tiles as the ortho map (4096 px/deg, ~7.4 m);
    # finer than SLDEM2015 (59 m) for terrain parallax (docs/parallax_protocol.md).
    "dtm": ("sln-l-tc-5-dtm-map-v2.0", "DTM_MAP_02", 3, "tc_dtm"),
    # MI: 9-band multispectral map, 1 deg tiles, one DARTS folder per degree of longitude.
    "mi": ("sln-l-mi-5-map-v3.0", "MI_MAP_03", 1, "mi"),
}
TILE_DEG = 3            # the TC grid; kept for callers that import it
MAX_ABS_LAT = 60
MANIFEST = ROOT / "data" / "manifest.json"


def fmt_lat(lat: int) -> str:
    return f"{'S' if lat < 0 else 'N'}{abs(lat):02d}"


def fmt_lon(lon: int) -> str:
    return f"E{lon % 360:03d}"


def tiles_for(bbox: tuple[float, float, float, float], tile_deg: int = TILE_DEG) -> list[str]:
    """Tile stems covering (min_lat, max_lat, min_lon, max_lon)."""
    min_lat, max_lat, min_lon, max_lon = bbox
    if max(abs(min_lat), abs(max_lat)) > MAX_ABS_LAT:
        sys.exit(f"latitudes beyond +/-{MAX_ABS_LAT} deg use the polar grid, not handled here")

    t = tile_deg
    stems = []
    south0 = math.floor(min_lat / t) * t
    west0 = math.floor(min_lon / t) * t
    for south in range(south0, math.ceil(max_lat / t) * t, t):
        for west in range(west0, math.ceil(max_lon / t) * t, t):
            north, east = south + t, west + t
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
    volume, prefix, tile_deg, _ = BUNDLES[bundle]
    lon_band = int(stem.split("E")[1][:3])  # the western longitude in the stem
    vol_dir = f"lon{lon_band // tile_deg * tile_deg:03d}"
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
    """Download one file, retrying a dropped connection (fetch_dem.fetch).

    The earlier single-shot version gave up on a reset connection (a 302 MB TC DTM
    tile died at 181 MB, 2026-09-25). fetch_dem.fetch retries and only renames a
    complete file. DARTS ignores HTTP Range (checked 2026-09-25: 200, whole file), so
    there each retry starts from zero; a server that honours Range is resumed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    from fetch_dem import fetch as fetch_resumable
    fetch_resumable(url, dest)


def check_manifest(dest: Path) -> None:
    """Compare a file with the SHA-256 data/manifest.json recorded for it, if any.

    A mismatch is not deleted: DARTS may have reissued the file. It is reported,
    because a test written against the recorded file may no longer hold.
    """
    if not MANIFEST.exists():
        return
    key = dest.relative_to(ROOT / "data" / "raw").as_posix()
    entry = json.loads(MANIFEST.read_text(encoding="utf-8")).get("files", {}).get(key)
    if not entry:
        return
    h = hashlib.sha256()
    with dest.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    ok = h.hexdigest() == entry["sha256"]
    print(f"    {'sha256 matches the manifest' if ok else '*** sha256 DIFFERS from the manifest ***'}")


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

    volume, _, tile_deg, folder = BUNDLES[args.bundle]
    stems = tiles_for(bbox, tile_deg)
    print(f"{args.bundle}: lat {bbox[0]:.3f}..{bbox[1]:.3f}, lon {bbox[2]:.3f}..{bbox[3]:.3f}"
          f" -> {len(stems)} tile(s)\n")

    out_dir = ROOT / "data" / "raw" / "selene" / folder
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
                check_manifest(out_dir / name)


if __name__ == "__main__":
    main()
