"""Lunar CRS, explicit reprojection, and pixel <-> ground models. Owner: Member A (Part 1). Features: GEO-05.

Rule (failure mode #9): a raw or calibrated Chandrayaan-2 or LRO NAC product is in
SENSOR geometry -- its pixels are NOT on any map. Where a pixel lands on the Moon
comes from an explicit geolocation model, and the model says where it came from:

    model                    products        source                          independent of references?
    CornerModel              CH-2            label system-level corners      yes (coarse: straight edges)
    GridModel                CH-2            label geometry grid (g_grd)     only if the label says the
                                                                             refinement used "System"
    MapModel                 SELENE TC/MI    label IMAGE_MAP_PROJECTION      yes (map-projected product)
    (none yet)               LRO NAC CDR     labels carry no geometry        -- needs SPICE (later)

    model = geolocation_model(meta)                  # reference-independent default
    lat, lon = model.pixel_to_latlon(rows, cols)     # rows = lines, cols = samples, 0-based, float
    rows, cols = model.latlon_to_pixel(lat, lon)

    crs = scene_crs(lat0, lon0)                      # IAU 2015 Moon sphere, metres
    x, y = to_map(lat, lon, crs); lat, lon = from_map(x, y, crs)

Pixel convention: coordinates index the pixel array 0-based, exactly as the label's
own corner fields are reproduced (verified for SELENE, see tests). Whether a label's
corner coordinate refers to a pixel's centre or its outer edge is not stated by
either archive -- an unavoidable +/- 0.5 px ambiguity, documented, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pyproj
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import (
    EquidistantCylindricalConversion,
    PolarStereographicBConversion,
)

from chandralign.contracts import SceneMeta
from chandralign.io.pds_label import _load_xml, _pvl_label, _pvl_value, _text, is_pds3, read_corner_sets

MOON_CRS_CODE = "IAU_2015:30100"        # Moon (2015), sphere R = 1737.4 km, planetocentric, east-positive
POLAR_LATITUDE = 60.0                   # beyond this, a polar stereographic map instead of equirectangular


class GeolocationUnavailable(ValueError):
    """The product carries no information that places its pixels on the Moon."""


# ============================================================================= CRS

@lru_cache(maxsize=None)
def moon_geographic() -> pyproj.CRS:
    return pyproj.CRS(MOON_CRS_CODE)


@lru_cache(maxsize=512)
def scene_crs(lat0: float, lon0: float) -> ProjectedCRS:
    """A metric map for a scene centred at (lat0, lon0) on the IAU 2015 Moon sphere.

    |lat0| < 60: equirectangular with true scale at lat0 and origin at lon0.
    Otherwise: polar stereographic about the nearer pole.
    """
    lat0, lon0 = round(float(lat0), 6), round(float(lon0), 6)
    if abs(lat0) < POLAR_LATITUDE:
        conv = EquidistantCylindricalConversion(latitude_first_parallel=lat0, longitude_natural_origin=lon0)
        name = f"Moon 2015 equirectangular lat_ts={lat0} lon_0={lon0}"
    else:
        pole = 90.0 if lat0 > 0 else -90.0
        conv = PolarStereographicBConversion(latitude_standard_parallel=pole, longitude_origin=lon0)
        name = f"Moon 2015 polar stereographic {'north' if pole > 0 else 'south'} lon_0={lon0}"
    return ProjectedCRS(conversion=conv, geodetic_crs=moon_geographic(), name=name)


@lru_cache(maxsize=512)
def _transformers(crs: ProjectedCRS):
    geo = moon_geographic()
    return (pyproj.Transformer.from_crs(geo, crs, always_xy=True),
            pyproj.Transformer.from_crs(crs, geo, always_xy=True))


def to_map(lat, lon, crs: ProjectedCRS):
    """(lat, lon) degrees -> (x, y) metres in `crs`."""
    fwd, _ = _transformers(crs)
    return fwd.transform(np.asarray(lon, float), np.asarray(lat, float))


def from_map(x, y, crs: ProjectedCRS):
    """(x, y) metres in `crs` -> (lat, lon) degrees, longitudes in -180..180."""
    _, inv = _transformers(crs)
    lon, lat = inv.transform(np.asarray(x, float), np.asarray(y, float))
    return lat, (np.asarray(lon) + 180.0) % 360.0 - 180.0


def surface_distance_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance on the Moon sphere (metres)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * moon_geographic().ellipsoid.semi_major_metre * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================================= pixel models

def _unwrap(lon: np.ndarray, ref: float) -> np.ndarray:
    """Longitudes shifted by multiples of 360 to lie within 180 deg of `ref`."""
    return ref + (np.asarray(lon, float) - ref + 180.0) % 360.0 - 180.0


def _wrap(lon):
    return (np.asarray(lon, float) + 180.0) % 360.0 - 180.0


class _Model:
    source: str
    independent_of_references: bool
    lines: int
    samples: int

    def pixel_to_latlon(self, rows, cols):
        raise NotImplementedError

    def latlon_to_pixel(self, lat, lon, tol_px: float = 1e-6, max_iter: int = 30):
        """Invert by Newton's method on the forward model (numerical Jacobian)."""
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        r = np.full(lat.shape, (self.lines - 1) / 2.0)
        c = np.full(lat.shape, (self.samples - 1) / 2.0)
        h = 0.5
        for _ in range(max_iter):
            la, lo = self.pixel_to_latlon(r, c, clip=False)
            lo = _unwrap(lo, lon)
            f1, f2 = la - lat, lo - lon
            la_r, lo_r = self.pixel_to_latlon(r + h, c, clip=False)
            la_c, lo_c = self.pixel_to_latlon(r, c + h, clip=False)
            a, b = (la_r - la) / h, (la_c - la) / h
            cc, d = (_unwrap(lo_r, lo) - lo) / h, (_unwrap(lo_c, lo) - lo) / h
            det = a * d - b * cc
            dr = (d * f1 - b * f2) / det
            dc = (-cc * f1 + a * f2) / det
            r, c = r - dr, c - dc
            if np.all(np.abs(dr) < tol_px) and np.all(np.abs(dc) < tol_px):
                break
        return r, c


@dataclass
class CornerModel(_Model):
    """Bilinear interpolation between the label's four system-level corners.

    Reference-independent, but coarse: a pushbroom strip's edges are not straight in
    lat/lon. Measured against ISRO's grid with the SAME (refined) corners, the
    straight-edge error on our products is 0.5 m (OHRC, 25 km strip), up to ~1 km
    (TMC-2, ~810 km strip) and ~2.5 km (IIRS, ~1,040 km strip).
    """
    corners: list[tuple[float, float]]      # UL, UR, LR, LL as (lat, lon)
    lines: int
    samples: int
    source: str = "label_corners_system"
    independent_of_references: bool = True

    def pixel_to_latlon(self, rows, cols, clip: bool = True):
        rows, cols = np.asarray(rows, float), np.asarray(cols, float)
        u = cols / (self.samples - 1)
        v = rows / (self.lines - 1)
        (la0, lo0), (la1, lo1), (la2, lo2), (la3, lo3) = self.corners
        lo1, lo2, lo3 = (_unwrap(x, lo0) for x in (lo1, lo2, lo3))
        lat = (1 - v) * ((1 - u) * la0 + u * la1) + v * ((1 - u) * la3 + u * la2)
        lon = (1 - v) * ((1 - u) * lo0 + u * lo1) + v * ((1 - u) * lo3 + u * lo2)
        return lat, _wrap(lon)


@dataclass
class GridModel(_Model):
    """Bilinear interpolation in ISRO's per-product geometry grid (g_grd CSV).

    Far denser than the corners (every 100 px for OHRC/TMC-2, every 50 px for IIRS).
    The grid follows the label's REFINED geolocation. When the label's
    reference_data_used names a reference image (SELENE for our TMC-2 and IIRS), the
    grid was tuned against it and independent_of_references is False -- do not use it
    to evaluate a registration against that same reference. "System" (our OHRC) means
    spacecraft data only, and the grid is independent.
    """
    scans: np.ndarray        # grid line coordinates (ascending)
    pixels: np.ndarray       # grid sample coordinates (ascending)
    lat: np.ndarray          # (len(scans), len(pixels))
    lon: np.ndarray          # unwrapped around lon[0, 0]
    lines: int
    samples: int
    refined_against: Optional[str]
    padding_rows_dropped: int
    source: str = "label_grid_refined"
    independent_of_references: bool = False

    def pixel_to_latlon(self, rows, cols, clip: bool = True):
        rows, cols = np.asarray(rows, float), np.asarray(cols, float)
        if clip and (np.any(rows < 0) or np.any(rows > self.lines - 1)
                     or np.any(cols < 0) or np.any(cols > self.samples - 1)):
            raise ValueError("pixel outside the product; the grid is not extrapolated")
        i = np.clip(np.searchsorted(self.scans, rows, side="right") - 1, 0, len(self.scans) - 2)
        j = np.clip(np.searchsorted(self.pixels, cols, side="right") - 1, 0, len(self.pixels) - 2)
        t = (rows - self.scans[i]) / (self.scans[i + 1] - self.scans[i])
        s = (cols - self.pixels[j]) / (self.pixels[j + 1] - self.pixels[j])

        def interp(g):
            return ((1 - t) * ((1 - s) * g[i, j] + s * g[i, j + 1])
                    + t * ((1 - s) * g[i + 1, j] + s * g[i + 1, j + 1]))

        return interp(self.lat), _wrap(interp(self.lon))


@dataclass
class MapModel(_Model):
    """Map-projected product (SELENE TC/MI SIMPLE CYLINDRICAL): exact, from the label.

    lat = CENTER_LATITUDE  + (LINE_PROJECTION_OFFSET - row) / MAP_RESOLUTION
    lon = CENTER_LONGITUDE + (col - SAMPLE_PROJECTION_OFFSET) / MAP_RESOLUTION
    with 0-based row/col. This is the PDS3 standard relation (SAMPLE =
    SAMPLE_PROJECTION_OFFSET + MAP_RESOLUTION * (lon - CENTER_LONGITUDE) + 1, with
    1-based samples) rewritten for 0-based indices. It reproduces the label's own
    corner coordinates; dropping the standard formula's "+1" is off by one pixel (tested).
    """
    resolution_px_per_deg: float
    line_offset: float
    sample_offset: float
    center_lat: float
    center_lon: float
    lines: int
    samples: int
    source: str = "label_map_projection"
    independent_of_references: bool = True

    def pixel_to_latlon(self, rows, cols, clip: bool = True):
        rows, cols = np.asarray(rows, float), np.asarray(cols, float)
        lat = self.center_lat + (self.line_offset - rows) / self.resolution_px_per_deg
        lon = self.center_lon + (cols - self.sample_offset) / self.resolution_px_per_deg
        return lat, _wrap(lon)

    def latlon_to_pixel(self, lat, lon, tol_px: float = 0.0, max_iter: int = 0):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        rows = self.line_offset - (lat - self.center_lat) * self.resolution_px_per_deg
        lon = _unwrap(lon, self.center_lon + (self.samples / 2 - self.sample_offset) / self.resolution_px_per_deg)
        cols = self.sample_offset + (lon - self.center_lon) * self.resolution_px_per_deg
        return rows, cols


# ============================================================================= builders

def grid_path(meta: SceneMeta) -> Path:
    """ISRO geometry grid for a CH-2 product: <product>/geometry/calibrated/<date>/<id with _g_grd_>.csv."""
    stem = meta.product_id.replace("_d_img_", "_g_grd_")
    product_root = meta.label_path.parents[3]
    hits = sorted(product_root.glob(f"geometry/*/*/{stem}.csv"))
    if not hits:
        raise GeolocationUnavailable(f"{meta.product_id}: no geometry grid {stem}.csv under {product_root}")
    return hits[0]


def load_grid_model(meta: SceneMeta, path: Optional[Path] = None) -> GridModel:
    path = path or grid_path(meta)
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    # IIRS grids end with all-zero padding rows (lon=lat=pixel=scan=0). A genuine
    # point at pixel 0, scan 0 exactly on 0N 0E is not plausible for a real strip.
    data = raw[~np.all(raw == 0.0, axis=1)]
    dropped = int(len(raw) - len(data))
    lon, lat, pix, scan = data.T
    pixels, scans = np.unique(pix), np.unique(scan)
    if len(pixels) * len(scans) != len(data):
        raise GeolocationUnavailable(f"{path.name}: not a complete regular grid "
                                     f"({len(pixels)} x {len(scans)} != {len(data)} points)")
    order = np.lexsort((pix, scan))
    lat_g = lat[order].reshape(len(scans), len(pixels))
    lon_g = _unwrap(lon[order], lon[order][0]).reshape(len(scans), len(pixels))
    lines, samples = meta.array_shape
    if pixels[0] != 0 or scans[0] != 0 or pixels[-1] != samples - 1 or scans[-1] != lines - 1:
        raise GeolocationUnavailable(f"{path.name}: grid does not span the product "
                                     f"(pixels {pixels[0]}..{pixels[-1]}, scans {scans[0]}..{scans[-1]})")
    # The label names what the refined geolocation was adjusted against. "System"
    # (OHRC here) means spacecraft data only -- no reference image involved.
    refined_against = _text(_load_xml(meta.label_path), "reference_data_used")
    independent = (refined_against or "").strip().lower() in ("system", "")
    return GridModel(scans, pixels, lat_g, lon_g, lines, samples, refined_against, dropped,
                     source="label_grid_system" if independent else "label_grid_refined",
                     independent_of_references=independent)


def load_map_model(meta: SceneMeta) -> MapModel:
    label = _pvl_label(meta.label_path)
    proj = label.get("IMAGE_MAP_PROJECTION")
    if proj is None:
        raise GeolocationUnavailable(f"{meta.product_id}: no IMAGE_MAP_PROJECTION")
    kind = str(_pvl_value(proj.get("MAP_PROJECTION_TYPE")) or "").upper()
    if kind != "SIMPLE CYLINDRICAL":
        raise GeolocationUnavailable(f"{meta.product_id}: projection {kind!r} not supported yet")
    num = lambda k: float(_pvl_value(proj[k]))
    lines, samples = meta.array_shape
    return MapModel(num("MAP_RESOLUTION"), num("LINE_PROJECTION_OFFSET"), num("SAMPLE_PROJECTION_OFFSET"),
                    num("CENTER_LATITUDE"), num("CENTER_LONGITUDE"), lines, samples)


def load_corner_model(meta: SceneMeta, corners: str = "system") -> CornerModel:
    if corners == "system":
        if not (meta.label_fields_verified.get("corner_latlon") and len(meta.corner_latlon) == 4):
            raise GeolocationUnavailable(f"{meta.product_id}: no corner coordinates in the label")
        pts = meta.corner_latlon
    else:
        pts = read_corner_sets(meta.label_path)[corners]
    lines, samples = meta.array_shape
    return CornerModel(list(pts), lines, samples, source=f"label_corners_{corners}",
                       independent_of_references=(corners == "system"))


def geolocation_model(meta: SceneMeta, prefer: str = "independent"):
    """The pixel<->ground model for a product.

    prefer="independent" (default): never a model tuned against a reference image --
      map projection for SELENE, system corners for CH-2.
    prefer="precise": the densest model available, even if reference-tuned
      (CH-2 geometry grid). Check `model.independent_of_references` before evaluating.
    """
    if prefer not in ("independent", "precise"):
        raise ValueError("prefer must be 'independent' or 'precise'")
    if is_pds3(meta.label_path):
        return load_map_model(meta)
    if meta.mission == "CH2":
        return load_grid_model(meta) if prefer == "precise" else load_corner_model(meta)
    raise GeolocationUnavailable(
        f"{meta.product_id}: {meta.mission} {meta.instrument} labels carry no pixel geolocation; "
        "a SPICE camera model is needed (not built yet)"
    )


def is_map_projected(meta: SceneMeta) -> bool:
    """True only for products whose label states a map projection of the pixel array."""
    if not is_pds3(meta.label_path):
        return False
    return "IMAGE_MAP_PROJECTION" in _pvl_label(meta.label_path)


def scene_crs_for(meta: SceneMeta):
    """A metric map centred on the product (from its independent model's centre pixel)."""
    model = geolocation_model(meta)
    lat, lon = model.pixel_to_latlon((model.lines - 1) / 2, (model.samples - 1) / 2)
    return scene_crs(float(lat), float(lon))
