"""Download lunar elevation-map (DEM) tiles covering a ground box.

Feature DATA-13. Source: NASA PDS Geosciences Node (no login), LRO LOLA RDR bundle.

    LOLA LDEM_1024   laser altimetry only, ~30 m/px, 15 x 30 deg tiles  -> independent of any camera
    SLDEM2015_512    LOLA merged with SELENE Terrain Camera stereo, ~59 m/px, 30 x 45 deg tiles
                     -> sharper, but partly derived from SELENE (one of our references)

    .venv/Scripts/python scripts/fetch_dem.py --scene ch2_ohr_ncp_20240330T0035085365_d_img_d18
    .venv/Scripts/python scripts/fetch_dem.py --bbox -0.5 0.4 23.4 23.7 --dem sldem2015 --download

Whole tiles are downloaded unmodified (.img + .lbl + .xml), so each can be checked
against the MD5 in its PDS4 label. Files land in data/raw/dem/<dem>/. Run
scripts/make_manifest.py afterwards.
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
BASE = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data"

DEMS = {
    # name: (url folder, file prefix, lat band deg, lon band deg, suffix)
    "lola": (f"{BASE}/lola_gdr/cylindrical/img", "ldem_1024", 15, 30, ""),
    "sldem2015": (f"{BASE}/sldem2015/tiles/float_img", "sldem2015_512", 30, 45, "_float"),
}


def _lat_band_name(lo: int, hi: int) -> str:
    """Tile latitude band as NASA names it: 00n_15n, 15s_00s, 30s_00s ..."""
    hemi = "s" if hi <= 0 else "n"
    return f"{abs(lo):02d}{hemi}_{abs(hi):02d}{hemi}"


def tiles_for(bbox: tuple[float, float, float, float], dem: str) -> list[str]:
    """Tile stems covering (min_lat, max_lat, min_lon, max_lon); longitudes in any convention."""
    _, prefix, dlat, dlon, suffix = DEMS[dem]
    min_lat, max_lat, min_lon, max_lon = bbox
    if min_lat < -60 or max_lat > 60:
        sys.exit("only +/-60 deg is supported (polar DEMs use a different product)")
    lon_lo, lon_hi = min_lon % 360.0, max_lon % 360.0
    if lon_hi < lon_lo:
        sys.exit("boxes crossing 0/360 deg longitude are not supported yet")
    stems = []
    for lat0 in range(math.floor(min_lat / dlat) * dlat, math.ceil(max_lat / dlat) * dlat, dlat):
        for lon0 in range(math.floor(lon_lo / dlon) * dlon, math.ceil(lon_hi / dlon) * dlon, dlon):
            stems.append(f"{prefix}_{_lat_band_name(lat0, lat0 + dlat)}_{lon0:03d}_{lon0 + dlon:03d}{suffix}")
    return stems


def scene_bbox(scene_id: str, pad: float) -> tuple[float, float, float, float]:
    if not FOOTPRINTS.exists():
        sys.exit("run scripts/ch2_footprints.py first to build the footprint index")
    for products in json.loads(FOOTPRINTS.read_text(encoding="utf-8")).values():
        for p in products:
            if p["product_id"].replace("_nrp_", "_ncp_").replace("_nra_", "_nca_").replace("_nri_", "_nci_") \
                    == scene_id or p["product_id"] == scene_id:
                min_lon, min_lat, max_lon, max_lat = p["bbox"]
                return (min_lat - pad, max_lat + pad, min_lon - pad, max_lon + pad)
    sys.exit(f"scene {scene_id} not found in the footprint index")


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"    have {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=3600) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    tmp.replace(dest)             # only a complete download gets the real name
    print(f"    got  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLAT", "MAXLAT", "MINLON", "MAXLON"))
    ap.add_argument("--dem", choices=sorted(DEMS), default="lola")
    ap.add_argument("--pad", type=float, default=0.1)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    bbox = scene_bbox(args.scene, args.pad) if args.scene else tuple(args.bbox) if args.bbox else None
    if bbox is None:
        sys.exit("give --scene or --bbox")

    folder = DEMS[args.dem][0]
    stems = tiles_for(bbox, args.dem)
    print(f"{args.dem}: lat {bbox[0]:.3f}..{bbox[1]:.3f}, lon {bbox[2]:.3f}..{bbox[3]:.3f} -> {len(stems)} tile(s)")
    out = ROOT / "data" / "raw" / "dem" / args.dem
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        print(stem)
        for ext in ("lbl", "xml", "img"):
            url = f"{folder}/{stem}.{ext}"
            if not args.download:
                size = requests.head(url, timeout=60, allow_redirects=True).headers.get("Content-Length")
                print(f"    {stem}.{ext:3} {int(size or 0) / 1e6:9.1f} MB")
            else:
                fetch(url, out / f"{stem}.{ext}")


if __name__ == "__main__":
    main()
