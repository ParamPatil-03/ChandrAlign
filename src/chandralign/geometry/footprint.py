"""Footprint overlap and pair pre-filter. Owner: Member A (Part 1). Features: GEO-04.

Before any matcher runs, check that two products actually show the same ground
(failure mode #10). Matching non-overlapping images wastes time and, worse,
produces confident-looking false matches.

    check = check_overlap(src_meta, ref_meta, min_overlap=0.10)
    if not check.ok:
        print(check.reason)               # rejected before any matcher is called

    run_if_overlapping(src_meta, ref_meta, matcher)   # calls matcher(src, ref) only if ok

Where outlines come from (recorded in Footprint.source):
    label          CH-2 corner coordinates, SELENE corner coordinates
    ode_catalogue  LRO NAC -- its labels hold no corners; ODE's footprint is used

Areas are measured in a sinusoidal equal-area projection centred on the pair, so
a km^2 is a km^2 at any latitude. That is accurate enough to decide "do these
overlap, and by how much"; exact map projection is GEO-05. Footprints are
straight-edged polygons through the corners, so a long pushbroom strip whose
edges curve slightly is approximated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from shapely import wkt
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from chandralign.contracts import SceneMeta
from chandralign.io.ode_client import find_saved_record

MOON_RADIUS_KM = 1737.4
KM_PER_DEG = MOON_RADIUS_KM * math.pi / 180.0
DEFAULT_MIN_OVERLAP = 0.10


class FootprintError(ValueError):
    """No usable outline for a product."""


class InsufficientOverlapError(RuntimeError):
    """A pair was rejected by the overlap pre-filter."""


@dataclass(frozen=True)
class Footprint:
    product_id: str
    polygon: BaseGeometry            # (lon, lat) degrees, longitudes in -180..180
    source: str                      # "label" | "ode_catalogue"
    crosses_antimeridian: bool


@dataclass(frozen=True)
class OverlapCheck:
    ok: bool
    reason: str
    src: Footprint
    ref: Footprint
    src_area_km2: float
    ref_area_km2: float
    overlap_km2: float
    fraction_of_src: float           # share of the source footprint covered by the reference
    fraction_of_ref: float
    fraction_of_smaller: float       # the number compared with min_overlap
    min_overlap: float


def _wrap(lon: float) -> float:
    """Longitude into -180..180."""
    return (lon + 180.0) % 360.0 - 180.0


def _normalise(poly: BaseGeometry) -> tuple[BaseGeometry, bool]:
    """Wrap longitudes to -180..180; detect a polygon that crosses the antimeridian."""
    wrapped = transform(lambda x, y, z=None: ([_wrap(v) for v in x], y), poly)
    lons = [p[0] for p in wrapped.exterior.coords]
    crosses = max(lons) - min(lons) > 180.0
    return wrapped, crosses


def _shift_east(poly: BaseGeometry) -> BaseGeometry:
    """Move negative longitudes to 180..360 (only for antimeridian-crossing pairs)."""
    return transform(lambda x, y, z=None: ([v + 360.0 if v < 0 else v for v in x], y), poly)


def footprint_of(meta: SceneMeta, catalogue: Optional[dict] = None) -> Footprint:
    """The product's ground outline, from its label corners or else a saved ODE record."""
    if meta.label_fields_verified.get("corner_latlon") and len(meta.corner_latlon) == 4:
        poly = Polygon([(lon, lat) for lat, lon in meta.corner_latlon])
        source = "label"
    else:
        record = catalogue if catalogue is not None else find_saved_record(meta.raster_path.parent)
        if record is None:
            raise FootprintError(
                f"{meta.product_id}: no corners in the label and no saved ODE record "
                "(fetch one with io.ode_client.save_product_record)"
            )
        if str(record.get("pdsid", "")).lower().split(".")[-1] != meta.product_id.lower():
            raise FootprintError(f"ODE record {record.get('pdsid')!r} does not belong to {meta.product_id!r}")
        text = record["record"].get("Footprint_geometry") or ""
        if not text.startswith(("POLYGON", "MULTIPOLYGON")):
            raise FootprintError(f"{meta.product_id}: ODE record has no footprint")
        poly = wkt.loads(text)
        source = "ode_catalogue"

    if poly.is_empty or not poly.is_valid:
        raise FootprintError(f"{meta.product_id}: corners do not form a valid polygon ({source})")
    if poly.geom_type != "Polygon":
        raise FootprintError(f"{meta.product_id}: expected one polygon, got {poly.geom_type}")
    poly, crosses = _normalise(poly)
    return Footprint(meta.product_id, poly, source, crosses)


# Edges are densified to this spacing (degrees) before projecting. Projecting only
# the corners would join them with straight lines in projected space, which for a
# 30-degree-long strip drifts from the true edge by as much as a narrow overlap is wide.
DENSIFY_DEG = 0.01


def _equal_area(geoms: list[BaseGeometry]) -> list[BaseGeometry]:
    """Sinusoidal equal-area projection (km), centred on the geometries' mean longitude."""
    lon0 = sum(g.centroid.x for g in geoms) / len(geoms)

    def project(x, y, z=None):
        xs = [(xi - lon0) * math.cos(math.radians(yi)) * KM_PER_DEG for xi, yi in zip(x, y)]
        ys = [yi * KM_PER_DEG for yi in y]
        return xs, ys

    return [transform(project, g.segmentize(DENSIFY_DEG)) for g in geoms]


def check_overlap(src: SceneMeta | Footprint, ref: SceneMeta | Footprint,
                  min_overlap: float = DEFAULT_MIN_OVERLAP) -> OverlapCheck:
    """Measure the overlap and decide whether the pair is worth matching.

    The pair passes when the overlap covers at least `min_overlap` of the SMALLER
    footprint: a 3 km OHRC strip fully inside a 700 km TMC-2 strip is a good pair,
    even though it covers a tiny share of the TMC-2 strip.
    """
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be within 0..1")
    a = src if isinstance(src, Footprint) else footprint_of(src)
    b = ref if isinstance(ref, Footprint) else footprint_of(ref)

    pa, pb = a.polygon, b.polygon
    if a.crosses_antimeridian or b.crosses_antimeridian:
        pa, pb = _shift_east(pa), _shift_east(pb)
    ea, eb = _equal_area([pa, pb])
    area_a, area_b = ea.area, eb.area
    inter = ea.intersection(eb).area
    frac_a = inter / area_a if area_a else 0.0
    frac_b = inter / area_b if area_b else 0.0
    frac_small = inter / min(area_a, area_b) if min(area_a, area_b) else 0.0

    if inter <= 0.0:
        ok, reason = False, f"no overlap: {a.product_id} and {b.product_id} show different ground"
    elif frac_small < min_overlap:
        ok, reason = False, (f"overlap {frac_small:.1%} of the smaller footprint "
                             f"({inter:.2f} km^2) is below the {min_overlap:.0%} minimum")
    else:
        ok, reason = True, f"overlap {frac_small:.1%} of the smaller footprint ({inter:.2f} km^2)"

    return OverlapCheck(ok, reason, a, b, area_a, area_b, inter, frac_a, frac_b, frac_small, min_overlap)


def run_if_overlapping(src: SceneMeta, ref: SceneMeta, fn: Callable, *args,
                       min_overlap: float = DEFAULT_MIN_OVERLAP, **kwargs):
    """Call fn(src, ref, ...) only if the pair passes the overlap pre-filter.

    Raises InsufficientOverlapError (carrying the OverlapCheck) otherwise, so a
    rejection is visible and explained rather than an empty result.
    """
    check = check_overlap(src, ref, min_overlap)
    if not check.ok:
        err = InsufficientOverlapError(check.reason)
        err.check = check
        raise err
    return fn(src, ref, *args, **kwargs)
