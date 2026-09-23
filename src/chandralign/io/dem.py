"""Elevation-map (DEM) loading. Owner: Member A (Part 1). Features: DATA-13.

Reads lunar DEM tiles through the same label-driven PDS3 reader as the images,
stitches the tiles covering a ground box into one regular grid, and returns
heights in METRES with the source recorded.

    patch = dem_patch(["data/raw/dem/lola/ldem_1024_00n_15n_000_030.lbl",
                       "data/raw/dem/lola/ldem_1024_15s_00s_000_030.lbl"],
                      bbox=(-0.5, 0.4, 23.4, 23.7))           # (min_lat, max_lat, min_lon, max_lon)
    patch.heights_m, patch.lat, patch.lon, patch.source, patch.independent_of_references
    patch.sample(lat, lon)                                    # bilinear, NaN outside the patch

Sources:
    lola        LOLA laser altimetry only -> independent of every camera image
    sldem2015   LOLA merged with SELENE Terrain Camera stereo -> NOT independent of
                SELENE, one of our references; do not use it to judge a SELENE registration

Height is DN x SCALING_FACTOR in the label's UNIT, converted to metres. The label's
OFFSET is the reference-sphere radius (1737.4 km) -- adding it would give a distance
from the Moon's centre, not a height, so it is deliberately NOT added.

Honest limit: LOLA tiles at 1024 px/deg (~30 m) are interpolated between laser
ground tracks spaced far wider than a pixel near the equator; most pixels are
filled in, not measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from chandralign.geometry.projection import MapModel
from chandralign.io.pds_label import (
    _pds3_find,
    _pvl_label,
    _pvl_value,
    is_pds3,
    read_array_layout,
    read_pds3_image_info,
    read_special_values,
)
from chandralign.io.pds_raster import Window, read_raster

UNIT_TO_M = {"METER": 1.0, "METERS": 1.0, "M": 1.0, "KILOMETER": 1000.0, "KILOMETERS": 1000.0, "KM": 1000.0}
DEM_KINDS = {
    # product-id prefix: (name, source description, independent of camera references?)
    "LDEM_": ("lola", "lola_laser_altimetry", True),
    "SLDEM2015": ("sldem2015", "sldem2015_lola_plus_selene_tc", False),
}


class DemError(ValueError):
    """A DEM tile is unreadable, unsupported, or the tiles do not fit together."""


@dataclass(frozen=True)
class DemTile:
    label_path: Path
    product_id: str
    kind: str                       # "lola" | "sldem2015"
    source: str
    independent_of_references: bool
    model: MapModel                 # pixel <-> lat/lon, from the label's map projection
    scale_to_m: float               # DN * scale_to_m = height in metres
    nodata: tuple[float, ...]
    lines: int
    samples: int

    def bounds(self) -> tuple[float, float, float, float]:
        """(min_lat, max_lat, min_lon, max_lon) of the pixel CENTRES."""
        lat_top, lon_left = self.model.pixel_to_latlon(0, 0)
        lat_bot, lon_right = self.model.pixel_to_latlon(self.lines - 1, self.samples - 1)
        return float(lat_bot), float(lat_top), float(lon_left), float(lon_right)


def read_dem_tile(label_path: str | Path) -> DemTile:
    label_path = Path(label_path)
    if not is_pds3(label_path):
        raise DemError(f"{label_path.name}: only PDS3 DEM labels are supported")
    label = _pvl_label(label_path)
    product_id = str(_pvl_value(label.get("PRODUCT_ID")) or label_path.stem).upper()
    kind = next((v for k, v in DEM_KINDS.items() if product_id.startswith(k)), None)
    if kind is None:
        raise DemError(f"{label_path.name}: unrecognised DEM product {product_id!r}")
    name, source, independent = kind

    info = read_pds3_image_info(label_path)
    unit = str(info.get("unit") or "").upper()
    if unit not in UNIT_TO_M:
        raise DemError(f"{label_path.name}: unknown height unit {unit!r}")
    scale = float(info.get("scaling_factor") or 1.0) * UNIT_TO_M[unit]

    proj = _pds3_find(label, "IMAGE_MAP_PROJECTION")
    if proj is None or str(_pvl_value(proj.get("MAP_PROJECTION_TYPE"))).upper() != "SIMPLE CYLINDRICAL":
        raise DemError(f"{label_path.name}: expected a SIMPLE CYLINDRICAL map projection")
    num = lambda k: float(_pvl_value(proj[k]))
    layout = read_array_layout(label_path)
    lines, samples = layout.shape[-2], layout.shape[-1]
    model = MapModel(num("MAP_RESOLUTION"), num("LINE_PROJECTION_OFFSET"), num("SAMPLE_PROJECTION_OFFSET"),
                     num("CENTER_LATITUDE"), num("CENTER_LONGITUDE"), lines, samples,
                     source=f"label_map_projection:{name}")
    return DemTile(label_path, product_id, name, source, independent, model, scale,
                   read_special_values(label_path).nodata, lines, samples)


@dataclass
class DemPatch:
    """Heights on a regular lat/lon grid of pixel centres, north-up, west-left."""
    heights_m: np.ndarray           # (n_lat, n_lon), NaN where the DEM has no data
    lat: np.ndarray                 # (n_lat,) descending
    lon: np.ndarray                 # (n_lon,) ascending
    res_px_per_deg: float
    source: str
    independent_of_references: bool
    tiles: tuple[str, ...]

    def sample(self, lat, lon) -> np.ndarray:
        """Bilinear heights at (lat, lon); NaN outside the patch (never extrapolated)."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        fr = (self.lat[0] - lat) * self.res_px_per_deg          # fractional row
        fc = (lon - self.lon[0]) * self.res_px_per_deg          # fractional column
        n_r, n_c = self.heights_m.shape
        inside = (fr >= 0) & (fr <= n_r - 1) & (fc >= 0) & (fc <= n_c - 1)
        r0 = np.clip(np.floor(fr).astype(int), 0, n_r - 2)
        c0 = np.clip(np.floor(fc).astype(int), 0, n_c - 2)
        t, s = fr - r0, fc - c0
        h = self.heights_m
        out = ((1 - t) * ((1 - s) * h[r0, c0] + s * h[r0, c0 + 1])
               + t * ((1 - s) * h[r0 + 1, c0] + s * h[r0 + 1, c0 + 1]))
        return np.where(inside, out, np.nan)


def _keys(values, res: float) -> np.ndarray:
    """Integer indices of pixel centres on a global half-pixel grid (robust to float noise)."""
    return np.rint(np.asarray(values, float) * res * 2).astype(np.int64)


def dem_patch(labels: Iterable[str | Path], bbox: tuple[float, float, float, float],
              margin_px: int = 2) -> DemPatch:
    """Heights (metres) for every DEM pixel centre inside `bbox`, stitched across tiles.

    All tiles must be the same product family and resolution, and share one grid.
    The patch is padded by `margin_px` so slope can be computed at its edges.
    Raises DemError if any part of the box is not covered by the given tiles.
    """
    tiles = [read_dem_tile(p) for p in labels]
    if not tiles:
        raise DemError("no DEM tiles given")
    kinds = {t.kind for t in tiles}
    res = {t.model.resolution_px_per_deg for t in tiles}
    if len(kinds) != 1 or len(res) != 1:
        raise DemError(f"tiles mix products or resolutions: {kinds}, {res}")
    res = res.pop()
    min_lat, max_lat, min_lon, max_lon = bbox
    pad = margin_px / res

    blocks = []            # (lat_keys, lon_keys, heights) per tile
    for tile in tiles:
        b_lat0, b_lat1, b_lon0, b_lon1 = tile.bounds()
        lat_lo, lat_hi = max(min_lat - pad, b_lat0), min(max_lat + pad, b_lat1)
        lon_lo, lon_hi = max(min_lon - pad, b_lon0), min(max_lon + pad, b_lon1)
        if lat_lo > lat_hi or lon_lo > lon_hi:
            continue
        r_top, c_left = tile.model.latlon_to_pixel(lat_hi, lon_lo)
        r_bot, c_right = tile.model.latlon_to_pixel(lat_lo, lon_hi)
        r0, r1 = int(np.floor(r_top)), int(np.ceil(r_bot))
        c0, c1 = int(np.floor(c_left)), int(np.ceil(c_right))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, tile.lines - 1), min(c1, tile.samples - 1)
        raw = read_raster(tile.label_path, Window(r0, c0, r1 - r0 + 1, c1 - c0 + 1)).astype(np.float64)
        heights = raw * tile.scale_to_m
        if tile.nodata:
            heights[np.isin(raw, tile.nodata)] = np.nan
        lats, _ = tile.model.pixel_to_latlon(np.arange(r0, r1 + 1), np.full(r1 - r0 + 1, c0))
        _, lons = tile.model.pixel_to_latlon(np.full(c1 - c0 + 1, r0), np.arange(c0, c1 + 1))
        blocks.append((_keys(lats, res), _keys(lons, res), heights))

    if not blocks:
        raise DemError(f"no tile covers the box {bbox}")
    lat_keys = np.unique(np.concatenate([b[0] for b in blocks]))[::-1]     # north first
    lon_keys = np.unique(np.concatenate([b[1] for b in blocks]))
    # neighbouring centres must be exactly one pixel apart (key step 2): no gaps, one grid
    if np.any(np.diff(lat_keys) != -2) or np.any(np.diff(lon_keys) != 2):
        raise DemError("tiles leave a gap inside the box or do not share one grid")
    grid = np.full((len(lat_keys), len(lon_keys)), np.nan)
    for la_k, lo_k, heights in blocks:
        rows = (lat_keys[0] - la_k) // 2
        cols = (lo_k - lon_keys[0]) // 2
        grid[np.ix_(rows, cols)] = heights
    lat = lat_keys / (2 * res)
    lon = lon_keys / (2 * res)
    if lat[0] < max_lat or lat[-1] > min_lat or lon[0] > min_lon or lon[-1] < max_lon:
        raise DemError(f"tiles do not cover the whole box {bbox} "
                       f"(covered lat {lat[-1]:.4f}..{lat[0]:.4f}, lon {lon[0]:.4f}..{lon[-1]:.4f})")
    t0 = tiles[0]
    return DemPatch(grid, lat, lon, res, t0.source, t0.independent_of_references,
                    tuple(t.product_id for t in tiles))


def find_tiles(dem_dir: str | Path) -> list[Path]:
    """Every USABLE DEM tile in a folder (e.g. data/raw/dem/lola): a label whose
    pixel file is also present.

    A label alone is not a tile. fetch_dem.py fetches the small label first and
    the 1.4 GB image last, so mid-download a folder holds labels without images;
    counting those made callers (and test skip-guards) believe a tile was there
    and then fail reading it.
    """
    folder = Path(dem_dir)
    return sorted(lbl for lbl in folder.glob("*.lbl")
                  if any(folder.glob(lbl.stem + ".img")) or any(folder.glob(lbl.stem + ".IMG")))
