"""Cut a sub-tile of a PDS3 DEM tile (SLDEM2015, LOLA) from byte ranges of its image.

A 1.4 GB SLDEM2015 tile comes off the PDS Geosciences server at ~0.3 MB/s, while a registration
window needs well under a megabyte of it. A sub-tile is the same product on the same grid: the
label keeps every keyword except the image size and the projection offsets, which are shifted by
the cut's first line and sample. io.dem reads it like any tile, stitches it with its neighbours,
and refuses (DemError) any box outside it -- nothing outside the cut is ever filled in.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from .dem import read_dem_tile

Reader = Callable[[int, int], bytes]          # (first byte, last byte) inclusive -> bytes


def _num(text: str, key: str) -> float:
    m = re.search(rf"^\s*{key}\s*=\s*([-+0-9.]+)", text, re.M)
    if m is None:
        raise ValueError(f"label has no {key}")
    return float(m.group(1))


def _set(text: str, key: str, value) -> str:
    """Replace one `KEY = value [<unit>]` line, keeping its indent and unit."""
    new, n = re.subn(rf"^(\s*{key}\s*=\s*)(\"[^\"]*\"|[-+0-9.]+)", lambda m: f"{m.group(1)}{value}", text,
                     count=1, flags=re.M)
    if n != 1:
        raise ValueError(f"label has no {key}")
    return new


def cut_subtile(label_path: str | Path, bbox: tuple[float, float, float, float], out_dir: str | Path,
                read: Reader, *, pad_px: int = 8, workers: int = 16) -> Path:
    """The part of the tile covering bbox (min_lat, max_lat, min_lon, max_lon) plus `pad_px`,
    clipped to the tile, written as <stem>_sub_<line>_<sample>_<lines>x<samples>.lbl/.img."""
    label_path = Path(label_path)
    tile = read_dem_tile(label_path)
    min_lat, max_lat, min_lon, max_lon = bbox
    r_top, c_left = tile.model.latlon_to_pixel(max_lat, min_lon)
    r_bot, c_right = tile.model.latlon_to_pixel(min_lat, max_lon)
    r0 = max(0, int(np.floor(float(r_top))) - pad_px)
    r1 = min(tile.lines, int(np.ceil(float(r_bot))) + pad_px + 1)
    c0 = max(0, int(np.floor(float(c_left))) - pad_px)
    c1 = min(tile.samples, int(np.ceil(float(c_right))) + pad_px + 1)
    if r0 >= r1 or c0 >= c1:
        raise ValueError(f"{label_path.name} does not overlap {bbox}")

    text = label_path.read_text(encoding="latin-1")
    record = int(_num(text, "RECORD_BYTES"))
    bps = record // tile.samples

    def row(r: int) -> bytes:
        b = read(r * record + c0 * bps, r * record + c1 * bps - 1)
        if len(b) != (c1 - c0) * bps:
            raise IOError(f"{label_path.name} line {r}: got {len(b)} bytes")
        return b
    with ThreadPoolExecutor(workers) as ex:
        data = b"".join(ex.map(row, range(r0, r1)))

    stem = f"{label_path.stem}_sub_{r0}_{c0}_{r1 - r0}x{c1 - c0}"
    img = f"{stem.upper()}.IMG"
    pid = re.search(r"^\s*PRODUCT_ID\s*=\s*\"?([^\"\s]+)", text, re.M).group(1)
    for key, value in [("PRODUCT_ID", f'"{pid}_SUB_{r0}_{c0}_{r1 - r0}X{c1 - c0}"'),
                       ("FILE_NAME", f'"{img}"'), (r"\^IMAGE", f'"{img}"'),
                       ("FILE_RECORDS", r1 - r0), ("RECORD_BYTES", (c1 - c0) * bps),
                       ("LINES", r1 - r0), ("LINE_SAMPLES", c1 - c0),
                       ("LINE_LAST_PIXEL", r1 - r0), ("SAMPLE_LAST_PIXEL", c1 - c0),
                       ("LINE_PROJECTION_OFFSET", repr(_num(text, "LINE_PROJECTION_OFFSET") - r0)),
                       ("SAMPLE_PROJECTION_OFFSET", repr(_num(text, "SAMPLE_PROJECTION_OFFSET") - c0))]:
        text = _set(text, key, value)
    note = (f"/* Sub-tile of {pid}: lines {r0 + 1}-{r1}, samples {c0 + 1}-{c1} of the original image, */\n"
            f"/* cut by chandralign.io.dem_subtile. Bounds keywords (MAXIMUM_LATITUDE ...) are the  */\n"
            f"/* original tile's; LINES/LINE_SAMPLES and the projection offsets describe this file. */\n")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.img").write_bytes(data)
    lbl = out_dir / f"{stem}.lbl"
    first, rest = text.split("\n", 1)                      # PDS_VERSION_ID must stay first
    lbl.write_text(f"{first}\n{note}{rest}", encoding="latin-1")
    return lbl
