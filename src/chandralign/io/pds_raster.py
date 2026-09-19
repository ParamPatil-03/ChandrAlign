"""Pixel reader with shape/dtype/offset taken from the label. Owner: Member A (Part 1). Features: DATA-01.

Reads any window of a product without loading the file: the pixel file is
memory-mapped with the exact layout the label declares (byte offset, dtype with
explicit byte order, axis order), and only the requested rows/columns/bands are
copied out. A 1000 x 1000 window of the 1.3 GB TMC-2 strip touches ~2 MB.

    meta = parse_pds4(label)
    tile = read_raster(meta, window=Window(row=5000, col=1000, height=1024, width=1024))
    band = read_raster(iirs_meta, window=w, bands=100)          # one IIRS band -> 2-D
    cube = read_raster(iirs_meta, window=w, bands=[10, 50, 100]) # -> (3, h, w)

Nothing here guesses a layout: every number comes from read_array_layout().
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from chandralign.contracts import SceneMeta
from chandralign.io.pds_label import ArrayLayout, PdsParseError, read_array_layout


class RasterIntegrityError(ValueError):
    """The pixel file on disk does not match what its label declares."""


@dataclass(frozen=True)
class Window:
    """A rectangle in full-product pixel coordinates (row = line, col = sample)."""
    row: int
    col: int
    height: int
    width: int

    def slices(self) -> tuple[slice, slice]:
        return slice(self.row, self.row + self.height), slice(self.col, self.col + self.width)


Bands = Union[None, int, Sequence[int]]


def _layout(meta_or_label: Union[SceneMeta, str, Path]) -> ArrayLayout:
    label = meta_or_label.label_path if isinstance(meta_or_label, SceneMeta) else meta_or_label
    return read_array_layout(label)


def _axis_positions(layout: ArrayLayout) -> tuple[Optional[int], int, int]:
    names = [n.lower() for n in layout.axis_names]
    try:
        line, sample = names.index("line"), names.index("sample")
    except ValueError as exc:
        raise PdsParseError(f"axes {layout.axis_names} lack Line/Sample") from exc
    band = names.index("band") if "band" in names else None
    return band, line, sample


def open_memmap(meta_or_label: Union[SceneMeta, str, Path]) -> tuple[np.memmap, ArrayLayout]:
    """Memory-map the whole pixel file with the label's layout. Nothing is read yet."""
    layout = _layout(meta_or_label)
    if not layout.raster_path.exists():
        raise FileNotFoundError(f"pixel file named by the label is missing: {layout.raster_path}")
    expected = layout.offset_bytes + int(np.prod(layout.shape)) * np.dtype(layout.dtype).itemsize
    actual = layout.raster_path.stat().st_size
    if actual < expected:
        raise RasterIntegrityError(
            f"{layout.raster_path.name}: {actual:,} bytes on disk, label layout needs {expected:,}"
        )
    mm = np.memmap(layout.raster_path, dtype=np.dtype(layout.dtype), mode="r",
                   offset=layout.offset_bytes, shape=layout.shape)
    return mm, layout


def read_raster(meta_or_label: Union[SceneMeta, str, Path],
                window: Optional[Window] = None,
                bands: Bands = None) -> np.ndarray:
    """Read pixels as an in-memory array in native dtype (no scaling, no conversion).

    Returns (height, width) for single-band products or when ``bands`` is an int,
    and (n_bands, height, width) otherwise. ``window=None`` reads the full extent --
    only sensible for small products; use tiling (DATA-11) for real strips.
    """
    mm, layout = open_memmap(meta_or_label)
    band_ax, line_ax, sample_ax = _axis_positions(layout)
    lines, samples = layout.shape[line_ax], layout.shape[sample_ax]

    w = window or Window(0, 0, lines, samples)
    if w.height <= 0 or w.width <= 0:
        raise ValueError(f"window must have positive size, got {w}")
    if w.row < 0 or w.col < 0 or w.row + w.height > lines or w.col + w.width > samples:
        raise ValueError(f"window {w} is outside the product ({lines} lines x {samples} samples)")
    rows, cols = w.slices()

    if band_ax is None:
        if bands not in (None, 0, [0], (0,)):
            raise ValueError(f"{layout.raster_path.name} has a single band; got bands={bands}")
        index = [slice(None)] * len(layout.shape)
        index[line_ax], index[sample_ax] = rows, cols
        return np.array(mm[tuple(index)])

    n_bands = layout.shape[band_ax]
    squeeze = isinstance(bands, (int, np.integer))
    band_list = list(range(n_bands)) if bands is None else ([int(bands)] if squeeze else [int(b) for b in bands])
    bad = [b for b in band_list if not 0 <= b < n_bands]
    if bad:
        raise ValueError(f"band index {bad} outside 0..{n_bands - 1}")

    index = [slice(None)] * len(layout.shape)
    index[line_ax], index[sample_ax] = rows, cols
    index[band_ax] = band_list
    out = np.array(mm[tuple(index)])
    # Normalise to (band, line, sample) whatever the on-disk interleave is.
    out = np.moveaxis(out, [band_ax, line_ax, sample_ax], [0, 1, 2])
    return out[0] if squeeze else out


# ----------------------------------------------------------------------------- integrity

def verify_raster(meta_or_label: Union[SceneMeta, str, Path], check_md5: bool = False) -> dict:
    """Check the pixel file against its label: exact size always, ISRO's MD5 on request.

    Raises RasterIntegrityError on any mismatch. MD5 reads the whole file (seconds per GB).
    """
    layout = _layout(meta_or_label)
    path = layout.raster_path
    actual = path.stat().st_size
    if layout.file_size_bytes is not None and actual != layout.file_size_bytes:
        raise RasterIntegrityError(f"{path.name}: {actual:,} bytes, label declares {layout.file_size_bytes:,}")
    report = {"file": path.name, "size_ok": True, "md5_checked": False}
    if check_md5:
        if not layout.md5:
            raise RasterIntegrityError(f"{path.name}: label declares no md5_checksum")
        h = hashlib.md5()
        with path.open("rb") as fh:
            while block := fh.read(1 << 24):
                h.update(block)
        if h.hexdigest() != layout.md5.lower():
            raise RasterIntegrityError(f"{path.name}: md5 {h.hexdigest()} != label {layout.md5}")
        report["md5_checked"] = True
    return report


_ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4"}
_ENVI_AXES = {"bsq": ("band", "line", "sample"), "bil": ("line", "band", "sample"),
              "bip": ("line", "sample", "band")}


def check_envi_header(meta_or_label: Union[SceneMeta, str, Path]) -> None:
    """Cross-check an ENVI .hdr sitting next to the pixel file against the PDS4 label.

    IIRS ships both. Two independent descriptions of one cube must agree; if they do
    not, raise rather than pick one.
    """
    layout = _layout(meta_or_label)
    hdr_path = layout.raster_path.with_suffix(".hdr")
    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)
    fields = {}
    for line in hdr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*([^=]+?)\s*=\s*(.+?)\s*$", line)
        if m:
            fields[m.group(1).lower()] = m.group(2).lower()

    interleave = fields.get("interleave")
    if interleave not in _ENVI_AXES:
        raise RasterIntegrityError(f"{hdr_path.name}: unknown interleave {interleave!r}")
    order = "<" if fields.get("byte order", "0") == "0" else ">"
    hdr_dtype = np.dtype(order + _ENVI_DTYPES[int(fields["data type"])]).str
    sizes = {"band": int(fields["bands"]), "line": int(fields["lines"]), "sample": int(fields["samples"])}
    hdr_shape = tuple(sizes[a] for a in _ENVI_AXES[interleave])
    hdr_axes = _ENVI_AXES[interleave]

    label_axes = tuple(a.lower() for a in layout.axis_names)
    problems = []
    if hdr_axes != label_axes:
        problems.append(f"axis order {hdr_axes} vs label {label_axes}")
    if hdr_shape != layout.shape:
        problems.append(f"shape {hdr_shape} vs label {layout.shape}")
    if hdr_dtype != layout.dtype:
        problems.append(f"dtype {hdr_dtype} vs label {layout.dtype}")
    if int(fields.get("header offset", 0)) != layout.offset_bytes:
        problems.append(f"offset {fields.get('header offset')} vs label {layout.offset_bytes}")
    if problems:
        raise RasterIntegrityError(f"{hdr_path.name} disagrees with the PDS4 label: " + "; ".join(problems))
