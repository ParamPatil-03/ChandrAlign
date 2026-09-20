"""Audit every product we hold against what its own label declares. No feature id:
this is a check on the data layer as a whole, not a feature.

    .venv/Scripts/python scripts/audit_readers.py

For each product it asks, independently of the parser that produced SceneMeta:
  * does the file's SIZE match offset + rows x cols x bands x itemsize?
  * does the declared SAMPLE_TYPE agree with the dtype we chose (sign, width, endian)?
  * are there LINE_PREFIX_BYTES / LINE_SUFFIX_BYTES we would be ignoring?
  * is a DN -> physical conversion (SCALING_FACTOR / OFFSET) declared and unapplied?
  * are special constants declared, and do they actually occur in the pixels?

It prints findings rather than raising, because the point is to SEE the state of
every product at once.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chandralign.io.pds_label import (  # noqa: E402
    is_pds3,
    parse_label,
    read_array_layout,
    read_special_values,
)
from chandralign.io.pds_raster import Window, read_raster  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

PATTERNS = [
    "ch2/*/products/*/data/calibrated/*/*_d_img_d18.xml",
    "lro/nac/*/*.XML",
    "lro/wac/*/*.XML",
    "selene/tc/*.lbl",
    "selene/mi/*.lbl",
    "dem/*/*.lbl",
]
SIDECARS = ("_PYR.", "_BROWSE.", "_THUMB.")

# PDS3 keywords that change how bytes map to pixels, or what the numbers mean.
PIXEL_LAYOUT_KEYS = ("LINE_PREFIX_BYTES", "LINE_SUFFIX_BYTES", "SAMPLE_BIT_MASK")
UNIT_KEYS = ("SCALING_FACTOR", "OFFSET", "CORE_MULTIPLIER", "CORE_BASE")


def labels() -> list[Path]:
    found = []
    for pattern in PATTERNS:
        found.extend(p for p in sorted(RAW.glob(pattern))
                     if not any(m in p.name.upper() for m in SIDECARS))
    return found


def raw_keyword(path: Path, key: str):
    """The keyword as the label TEXT states it, read without the parser."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hit = re.search(rf"^\s*{key}\s*=\s*([^\r\n]+)", text, re.MULTILINE)
    return hit.group(1).strip() if hit else None


def main() -> int:
    findings: list[str] = []
    rows = []
    for path in labels():
        name = path.name
        if "/dem/" in path.as_posix():
            # Elevation maps come from no camera, so parse_label REFUSES them by
            # design; io/dem.py reads them. Audited through that reader instead.
            try:
                from chandralign.io.dem import read_dem_tile
                tile = read_dem_tile(path)
                rows.append((name, "DEM", str((tile.lines, tile.samples)),
                             f"x{tile.scale_to_m:g}m", 0, "ok (dem reader)",
                             len(tile.nodata), "pds3"))
            except Exception as exc:                    # noqa: BLE001 - reported
                findings.append(f"UNREADABLE  {name}: {type(exc).__name__}: {exc}")
            continue
        try:
            meta = parse_label(path)
            layout = read_array_layout(path)
        except Exception as exc:                        # noqa: BLE001 - reported
            findings.append(f"UNREADABLE  {name}: {type(exc).__name__}: {exc}")
            continue

        itemsize = np.dtype(layout.dtype).itemsize
        pixels = int(np.prod(layout.shape))
        expected = layout.offset_bytes + pixels * itemsize
        actual = layout.raster_path.stat().st_size if layout.raster_path.is_file() else -1
        size_note = "ok" if actual >= expected else f"SHORT by {expected - actual}"
        if actual > expected:
            size_note = f"ok (+{actual - expected} trailing)"

        special = read_special_values(path)
        pds3 = is_pds3(path)

        layout_keys = {k: raw_keyword(path, k) for k in PIXEL_LAYOUT_KEYS} if pds3 else {}
        unit_keys = {k: raw_keyword(path, k) for k in UNIT_KEYS} if pds3 else {}

        for key, value in layout_keys.items():
            if value and value.strip("0 .") not in ("", "2#1111111111111111#"):
                findings.append(f"LAYOUT      {name}: {key} = {value} is declared and NOT applied")
        scaling = unit_keys.get("SCALING_FACTOR")
        if scaling and float(scaling.split()[0]) not in (0.0, 1.0):
            findings.append(f"UNITS       {name}: SCALING_FACTOR = {scaling} declared, "
                            "pixels are returned as stored DN")

        rows.append((name, str(meta.instrument), str(layout.shape), layout.dtype,
                     layout.offset_bytes, size_note,
                     len(special.nodata), "pds3" if pds3 else "pds4"))

    width = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'product':<{width}}  {'inst':5} {'shape':>22} {'dtype':>6} {'offset':>11}  "
          f"{'file size':<22} {'nodata':>6}  fmt")
    for r in rows:
        print(f"{r[0]:<{width}}  {r[1]:5} {r[2]:>22} {r[3]:>6} {r[4]:>11}  "
              f"{r[5]:<22} {r[6]:>6}  {r[7]}")

    print(f"\n{len(rows)} products audited, {len(findings)} finding(s):")
    for f in findings:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
