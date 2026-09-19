"""Overlapping tile generator for very large products. Owner: Member A (Part 1). Features: DATA-11.

Real products are far too large for any matcher in one piece (TMC-2 160k x 4k,
OHRC 80k x 12k, NAC 52k x 5k pixels), so everything downstream works on tiles.

    for plane in iter_tiles(meta, tile=1024, overlap=128):
        ...   # plane is a contracts.ImagePlane; plane.tile_origin = (row, col) in the product

Tiles are produced one at a time from windowed reads, so memory stays at a few
tiles no matter how large the product is. The files on disk are only ever read.

Coordinate convention (shared with Part 2's MatchSet): points are (x, y) =
(column, row), sub-pixel floats. tile_origin is (row, col), as the contract says.
Use tile_to_global / global_to_tile to move points between a tile and the product.

Normalisation here is deliberately minimal -- a per-tile min/max stretch of the
valid pixels to float32 0..1, recorded as "tile_minmax" in preprocess_chain. The
real radiometric preparation (CLAHE etc., PREP-01) replaces it later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from chandralign.contracts import ImagePlane, SceneMeta
from chandralign.io.pds_label import SpecialValues, read_special_values
from chandralign.io.pds_raster import Window, read_raster


@dataclass
class TilingStats:
    """What an iteration did -- filled in as tiles are produced."""
    planned: int = 0
    yielded: int = 0
    skipped_invalid: int = 0


def tile_grid(lines: int, samples: int, tile: int = 1024, overlap: int = 128) -> list[Window]:
    """Windows covering every pixel of a (lines x samples) product, row-major.

    Neighbouring tiles share `overlap` pixels. The last row/column of tiles is moved
    back so it ends exactly at the product edge (so it may overlap more), which keeps
    every tile full-size. A product smaller than `tile` gives one tile of its own size.
    """
    if tile <= 0:
        raise ValueError("tile must be positive")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be >= 0 and smaller than tile")
    if lines <= 0 or samples <= 0:
        raise ValueError("product must have positive size")

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        stride = tile - overlap
        out = list(range(0, size - tile + 1, stride))
        if out[-1] + tile < size:
            out.append(size - tile)
        return out

    h, w = min(tile, lines), min(tile, samples)
    return [Window(r, c, h, w) for r in starts(lines) for c in starts(samples)]


def valid_mask(array: np.ndarray, special: SpecialValues) -> np.ndarray:
    """True where a pixel is a real measurement according to the label."""
    mask = np.isfinite(array) if array.dtype.kind == "f" else np.ones(array.shape, dtype=bool)
    if special.nodata:
        mask &= ~np.isin(array, np.asarray(special.nodata, dtype=array.dtype))
    if special.valid_min is not None:
        mask &= array >= special.valid_min
    if special.valid_max is not None:
        mask &= array <= special.valid_max
    return mask


def _normalise(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(array.shape, dtype=np.float32)
    if not mask.any():
        return out
    values = array[mask].astype(np.float64)
    lo, hi = values.min(), values.max()
    if hi > lo:
        out[mask] = ((values - lo) / (hi - lo)).astype(np.float32)
    return out


def read_tile(meta: SceneMeta, window: Window, band: Optional[int] = None,
              special: Optional[SpecialValues] = None) -> ImagePlane:
    """One window of a product as an ImagePlane (float32 0..1, masks, origin)."""
    if meta.n_bands > 1 and band is None:
        raise ValueError(
            f"{meta.product_id} has {meta.n_bands} bands; pass band=<index>. "
            "Matching a whole hyperspectral cube is not supported -- use the IIRS "
            "composite (PREP-06) to make one plane."
        )
    special = special or read_special_values(meta.label_path)
    raw = read_raster(meta, window, bands=band if meta.n_bands > 1 else None)
    valid = valid_mask(raw, special)
    return ImagePlane(
        array=_normalise(raw, valid),
        valid_mask=valid,
        shadow_mask=np.zeros(raw.shape, dtype=bool),   # a distinct array; filled by PREP-04
        gsd_m=meta.gsd_m,
        meta=meta,
        tile_origin=(window.row, window.col),
        preprocess_chain=(["tile_minmax"] if meta.n_bands == 1 else [f"band_{band}", "tile_minmax"]),
    )


def iter_tiles(meta: SceneMeta, tile: int = 1024, overlap: int = 128,
               max_invalid_fraction: float = 0.5, band: Optional[int] = None,
               stats: Optional[TilingStats] = None) -> Iterator[ImagePlane]:
    """Yield ImagePlanes across the whole product, lazily.

    Tiles whose invalid fraction exceeds `max_invalid_fraction` are skipped and
    counted in `stats.skipped_invalid` -- never silently dropped.
    """
    if not 0.0 <= max_invalid_fraction <= 1.0:
        raise ValueError("max_invalid_fraction must be within 0..1")
    special = read_special_values(meta.label_path)
    windows = tile_grid(*meta.array_shape, tile=tile, overlap=overlap)
    if stats is not None:
        stats.planned = len(windows)
    for w in windows:
        plane = read_tile(meta, w, band=band, special=special)
        if 1.0 - plane.valid_mask.mean() > max_invalid_fraction:
            if stats is not None:
                stats.skipped_invalid += 1
            continue
        if stats is not None:
            stats.yielded += 1
        yield plane


def tile_to_global(points: np.ndarray, tile_origin: tuple[int, int]) -> np.ndarray:
    """(N, 2) (x, y) points in a tile -> the same points in full-product pixels."""
    points = np.asarray(points, dtype=np.float64)
    row, col = tile_origin
    return points + np.array([col, row], dtype=np.float64)


def global_to_tile(points: np.ndarray, tile_origin: tuple[int, int]) -> np.ndarray:
    """(N, 2) (x, y) points in full-product pixels -> the same points in a tile."""
    points = np.asarray(points, dtype=np.float64)
    row, col = tile_origin
    return points - np.array([col, row], dtype=np.float64)
