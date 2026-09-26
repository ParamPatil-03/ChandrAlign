"""Download lunar elevation-map (DEM) tiles covering a ground box.

Feature DATA-13. Source: NASA PDS Geosciences Node (no login), LRO LOLA RDR bundle.

    LOLA LDEM_1024   laser altimetry only, ~30 m/px, 15 x 30 deg tiles  -> independent of any camera
    SLDEM2015_512    LOLA merged with SELENE Terrain Camera stereo, ~59 m/px, 30 x 45 deg tiles
                     -> sharper, but partly derived from SELENE (one of our references)

    .venv/Scripts/python scripts/fetch_dem.py --scene ch2_ohr_ncp_20240330T0035085365_d_img_d18
    .venv/Scripts/python scripts/fetch_dem.py --bbox -0.5 0.4 23.4 23.7 --dem sldem2015 --download

Whole tiles are downloaded unmodified (.img + .lbl + .xml). The PDS4 labels carry
NO checksum, so completeness is checked against the server's Content-Length, and
an interrupted transfer RESUMES from where it stopped (HTTP Range) instead of
starting a 1.4 GB tile again. The PDS Geosciences server throttles each
connection (measured: ~40 KB/s per connection, scaling linearly to ~800 KB/s at
24), so large files are fetched as parallel byte ranges (--connections).
Files land in data/raw/dem/<dem>/. Run scripts/make_manifest.py afterwards.

--subtile (with --download) fetches only the part of each tile covering the box, as a sub-tile
on the same grid (chandralign.io.dem_subtile): a few MB instead of 1.4 GB per tile.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chandralign.io.dem_subtile import cut_subtile  # noqa: E402

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


def _remote_size(url: str) -> int | None:
    try:
        h = requests.head(url, timeout=60, allow_redirects=True)
        h.raise_for_status()
        n = h.headers.get("Content-Length")
        return int(n) if n is not None else None
    except (requests.RequestException, ValueError):
        return None


def _fetch_range(url: str, part: Path, start: int, end: int, attempts: int = 30) -> None:
    """Bytes start..end (inclusive) into `part`, resuming it after any drop."""
    want = end - start + 1
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            part.unlink(); have = 0
        try:
            with requests.get(url, stream=True, timeout=120,
                              headers={"Range": f"bytes={start + have}-{end}"}) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"server ignored the byte range (HTTP {r.status_code})")
                with part.open("ab") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        except requests.RequestException:
            time.sleep(min(60, 3 * attempt))
    have = part.stat().st_size if part.exists() else 0
    if have != want:
        raise RuntimeError(f"{part.name}: {have} of {want} bytes after {attempts} attempts")


def _range_reader(url: str, attempts: int = 8):
    """(first, last) byte -> bytes of `url`, retried; for dem_subtile.cut_subtile."""
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))

    def read(start: int, end: int) -> bytes:
        for attempt in range(1, attempts + 1):
            try:
                r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == end - start + 1:
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(min(30, 2 * attempt))
        raise RuntimeError(f"{url}: bytes {start}-{end} failed after {attempts} attempts")
    return read


def fetch_parallel(url: str, dest: Path, size: int, connections: int) -> None:
    """Fetch as `connections` byte ranges, then join. Each range resumes on its own,
    so a rerun after an interruption re-downloads nothing it already has."""
    step = -(-size // connections)
    ranges = [(k, k * step, min(size, (k + 1) * step) - 1) for k in range(connections) if k * step < size]
    parts = [dest.with_name(f"{dest.name}.part{k:02d}") for k, _, _ in ranges]
    print(f"    get  {dest.name} ({size / 1e6:.1f} MB) as {len(ranges)} ranges", flush=True)
    with ThreadPoolExecutor(len(ranges)) as pool:
        list(pool.map(lambda r: _fetch_range(url, parts[r[0]], r[1], r[2]), ranges))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as fh:
                while chunk := fh.read(1 << 24):
                    out.write(chunk)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: joined {tmp.stat().st_size} bytes, expected {size}")
    tmp.replace(dest)
    for part in parts:
        part.unlink()
    print(f"    got  {dest.name} ({size / 1e6:.1f} MB)")


def fetch(url: str, dest: Path, attempts: int = 20, connections: int = 1) -> None:
    """Download one file; resume after a dropped connection; never leave a short
    file under the real name.

    A file that merely EXISTS is not a file that finished (fetch_lro.py, 54bf21e),
    so an existing file is re-checked against Content-Length. A partial transfer
    is kept as .part and continued with an HTTP Range request.
    """
    want = _remote_size(url)
    if dest.exists():
        if want is None or dest.stat().st_size == want:
            print(f"    have {dest.name}" + ("" if want else " (size unverified: no Content-Length)"))
            return
        print(f"    redo {dest.name} (have {dest.stat().st_size / 1e6:.1f} of {want / 1e6:.1f} MB)")
        dest.unlink()
    if connections > 1 and want is not None and want > 64 * 2**20:
        dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)   # a single-stream leftover
        fetch_parallel(url, dest, want, connections)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if want is not None and have == want:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                r.raise_for_status()
                if have and r.status_code != 206:      # server ignored the Range: start over
                    have = 0
                with tmp.open("ab" if have else "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
            if want is None or tmp.stat().st_size == want:
                break
        except requests.RequestException as exc:
            got = tmp.stat().st_size if tmp.exists() else 0
            print(f"    attempt {attempt}: {type(exc).__name__} at {got / 1e6:.1f} MB; resuming",
                  flush=True)
            time.sleep(min(60, 5 * attempt))
    got = tmp.stat().st_size if tmp.exists() else 0
    if want is not None and got != want:
        raise RuntimeError(f"{dest.name}: have {got} of {want} bytes after {attempts} attempts "
                           f"(kept as {tmp.name}; rerun to resume)")
    tmp.replace(dest)             # only a complete download gets the real name
    print(f"    got  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLAT", "MAXLAT", "MINLON", "MAXLON"))
    ap.add_argument("--dem", choices=sorted(DEMS), default="lola")
    ap.add_argument("--pad", type=float, default=0.1)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--subtile", action="store_true", help="with --download: only the part covering the box")
    ap.add_argument("--connections", type=int, default=24,
                    help="parallel byte ranges for large files (the server throttles each connection)")
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
        if args.download and args.subtile:
            fetch(f"{folder}/{stem}.lbl", out / f"{stem}.lbl")
            sub = cut_subtile(out / f"{stem}.lbl", bbox, out, _range_reader(f"{folder}/{stem}.img"),
                              workers=args.connections)
            print(f"    cut  {sub.name}")
            continue
        for ext in ("lbl", "xml", "img"):
            url = f"{folder}/{stem}.{ext}"
            if not args.download:
                size = requests.head(url, timeout=60, allow_redirects=True).headers.get("Content-Length")
                print(f"    {stem}.{ext:3} {int(size or 0) / 1e6:9.1f} MB")
            else:
                fetch(url, out / f"{stem}.{ext}", connections=args.connections)


if __name__ == "__main__":
    main()
