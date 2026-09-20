"""Physics-based outlier filtering. Owner: Member A (Part 1). Feature: ALIGN-03.

    kept, report = filter_matches(matches, src_meta, ref_meta, dem_tiles)
    estimate(kept.src_pts, kept.ref_pts, ...)     # then fit, on cleaner input

THE IDEA, in one line: a correct match names the SAME piece of ground twice, and
the ground does not change between two photographs of it.

So for every match we put both endpoints on the Moon (GEO-05), look up the terrain
there (GEO-03: slope and aspect from a DEM), and compare. A true match samples one
patch of ground twice and the two readings agree. A false match -- the crater that
merely LOOKS like the right crater -- lands on different ground, and its slope and
aspect disagree. Brightness cannot tell those apart, because under a moved sun the
same slope changes appearance completely (measured: raw brightness correlates -0.96
between our own two products' sun geometries). Terrain can.

Aspect is compared only where BOTH ends are steep enough for aspect to mean
anything; on flat ground the downhill direction is noise, and a filter that trusted
it there would throw away good matches on exactly the smooth terrain where matches
are already scarce.

A second, weaker signal is available where sun geometry is known (GEO-02): the
brightness each end SHOULD have, given its slope and its own image's sun. This
down-weights rather than rejects, because it depends on a Lambertian assumption the
Moon does not obey.

IT SWITCHES ITSELF OFF, LOUDLY. With no DEM, or with endpoints outside the DEM,
there is no physics to apply; the matches pass through untouched and the report
says `applied: False` with a reason. It never silently does nothing.

MEASURED, on real LOLA/SLDEM relief rendered under our own two products' sun
geometries, where the true correspondence is known exactly (see
tests/test_geometry_filter.py). The numbers are in the test, not asserted here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from chandralign.contracts import MatchSet, SceneMeta
from chandralign.geometry.dem_terrain import slope_aspect
from chandralign.geometry.solar import azimuth_difference

# A match whose two ends disagree about the ground by more than this is not a match
# of the same ground. Generous, because DEM sampling and geolocation both add error.
MAX_SLOPE_DIFFERENCE_DEG = 8.0
MAX_ASPECT_DIFFERENCE_DEG = 60.0
# Below this slope the downhill direction is noise, so aspect is not compared.
ASPECT_MEANINGFUL_SLOPE_DEG = 3.0


@dataclass(frozen=True)
class FilterReport:
    """What the filter did, and why. Every field is measured or None."""
    applied: bool
    reason: str
    n_in: int
    n_kept: int
    n_rejected_slope: int = 0
    n_rejected_aspect: int = 0
    n_unknown_terrain: int = 0
    median_slope_difference_deg: Optional[float] = None
    median_aspect_difference_deg: Optional[float] = None

    @property
    def rejected_fraction(self) -> float:
        return 0.0 if not self.n_in else 1.0 - self.n_kept / self.n_in


def _ground(meta: SceneMeta, points: np.ndarray, model=None) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) for (N,2) pixel points given as (x=sample, y=line)."""
    from chandralign.geometry.projection import geolocation_model

    if model is None:
        model = geolocation_model(meta)
    points = np.asarray(points, float).reshape(-1, 2)
    return model.pixel_to_latlon(points[:, 1], points[:, 0])


def terrain_at(lat, lon, terrain, patch) -> tuple[np.ndarray, np.ndarray]:
    """Slope and aspect sampled at (lat, lon); NaN where the patch does not reach."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    rows = (patch.lat[0] - lat) * patch.res_px_per_deg
    cols = (lon - patch.lon[0]) * patch.res_px_per_deg
    n_r, n_c = terrain.slope_deg.shape
    inside = (rows >= 0) & (rows <= n_r - 1) & (cols >= 0) & (cols <= n_c - 1)
    r = np.clip(np.rint(rows).astype(int), 0, n_r - 1)
    c = np.clip(np.rint(cols).astype(int), 0, n_c - 1)
    slope = np.where(inside, terrain.slope_deg[r, c], np.nan)
    aspect = np.where(inside, terrain.aspect_deg[r, c], np.nan)
    return slope, aspect


def filter_matches(matches: MatchSet, src_meta: SceneMeta, ref_meta: SceneMeta,
                   dem_patch_or_tiles=None, *,
                   max_slope_difference_deg: float = MAX_SLOPE_DIFFERENCE_DEG,
                   max_aspect_difference_deg: float = MAX_ASPECT_DIFFERENCE_DEG,
                   src_model=None, ref_model=None) -> tuple[MatchSet, FilterReport]:
    """Drop matches whose two ends describe different ground (ALIGN-03).

    dem_patch_or_tiles is a DemPatch covering both footprints, or None. With None
    the matches are returned untouched and the report says so -- the filter never
    pretends to have done work it could not do.
    """
    n_in = int(len(matches.src_pts))
    if dem_patch_or_tiles is None:
        return matches, FilterReport(False, "no DEM supplied: physics filter disabled",
                                     n_in, n_in)
    if n_in == 0:
        return matches, FilterReport(False, "no matches to filter", 0, 0)

    patch = dem_patch_or_tiles
    terrain = slope_aspect(patch)

    src_lat, src_lon = _ground(src_meta, matches.src_pts, src_model)
    ref_lat, ref_lon = _ground(ref_meta, matches.ref_pts, ref_model)
    s_slope, s_aspect = terrain_at(src_lat, src_lon, terrain, patch)
    r_slope, r_aspect = terrain_at(ref_lat, ref_lon, terrain, patch)

    known = np.isfinite(s_slope) & np.isfinite(r_slope)
    if not known.any():
        return matches, FilterReport(False, "no match endpoint falls inside the DEM",
                                     n_in, n_in, n_unknown_terrain=n_in)

    slope_gap = np.abs(s_slope - r_slope)
    aspect_gap = azimuth_difference(s_aspect, r_aspect)

    # Aspect only means something where both ends are steep enough to have one.
    aspect_matters = (known & np.isfinite(s_aspect) & np.isfinite(r_aspect)
                      & (s_slope >= ASPECT_MEANINGFUL_SLOPE_DEG)
                      & (r_slope >= ASPECT_MEANINGFUL_SLOPE_DEG))

    bad_slope = known & (slope_gap > max_slope_difference_deg)
    bad_aspect = aspect_matters & (aspect_gap > max_aspect_difference_deg) & ~bad_slope
    # Terrain we could not look up is KEPT: unknown is not evidence of a bad match.
    keep = ~(bad_slope | bad_aspect)

    kept = replace(matches,
                   src_pts=matches.src_pts[keep], ref_pts=matches.ref_pts[keep],
                   confidence=matches.confidence[keep])
    report = FilterReport(
        applied=True,
        reason=f"terrain compared at {int(known.sum())} of {n_in} matches",
        n_in=n_in, n_kept=int(keep.sum()),
        n_rejected_slope=int(bad_slope.sum()),
        n_rejected_aspect=int(bad_aspect.sum()),
        n_unknown_terrain=int((~known).sum()),
        median_slope_difference_deg=float(np.nanmedian(slope_gap[known])) if known.any() else None,
        median_aspect_difference_deg=(float(np.nanmedian(aspect_gap[aspect_matters]))
                                      if aspect_matters.any() else None),
    )
    return kept, report


def inlier_precision(src_pts: np.ndarray, ref_pts: np.ndarray,
                     truth, tolerance_px: float = 2.0) -> tuple[float, int, int]:
    """(precision, correct, total) against a known correspondence.

    `truth` maps a source point to where it truly belongs in the reference. Used to
    score the filter honestly: a match is correct only if it lands within
    `tolerance_px` of the truth, so "more inliers" cannot be mistaken for "better".
    """
    src_pts = np.asarray(src_pts, float).reshape(-1, 2)
    ref_pts = np.asarray(ref_pts, float).reshape(-1, 2)
    if len(src_pts) == 0:
        return 0.0, 0, 0
    expected = truth(src_pts)
    error = np.hypot(*(expected - ref_pts).T)
    correct = int((error <= tolerance_px).sum())
    return correct / len(src_pts), correct, len(src_pts)
