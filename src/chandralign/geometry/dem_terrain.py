"""Slope and aspect from a DEM. Owner: Member A (Part 1). Features: GEO-03.

Input to the physics-based outlier filter (ALIGN-03): a steep sun-facing slope
cannot be the same ground as a flat shadowed plain.

    terrain = slope_aspect(patch)                   # patch from io.dem.dem_patch
    terrain.slope_deg, terrain.aspect_deg           # same grid as patch.heights_m

Method: Horn (1981), the 3 x 3 finite-difference scheme gdaldem uses. It is
implemented here in numpy rather than by shelling out to gdaldem (PLAN.md P1-T09)
because the GDAL command-line tools are hard to install on Windows for every team
member; the result is checked against analytic planes to < 0.1 deg in the tests.

Conventions:
    slope   degrees from horizontal, 0..90
    aspect  compass direction the ground FACES (the steepest downhill direction),
            degrees clockwise from north: 0 = N, 90 = E, 180 = S, 270 = W.
            NaN where the ground is flat (no downhill direction exists).
Pixel spacing is computed per row from the patch's latitude on the Moon sphere,
so east-west distances shrink with cos(latitude). Border pixels and pixels next to
missing data are NaN -- never padded or guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOON_RADIUS_M = 1737400.0
FLAT_SLOPE_DEG = 1e-6        # below this, aspect is undefined


@dataclass
class Terrain:
    slope_deg: np.ndarray
    aspect_deg: np.ndarray
    source: str                     # the DEM's source, carried through
    independent_of_references: bool


def horn_gradients(z: np.ndarray, dx_m: np.ndarray | float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    """dz/d(east) and dz/d(north) in m/m by Horn's method.

    z is north-up (row 0 = north), west-left. dx_m may vary per row (shape (n_rows,)).
    Returns arrays the shape of z with NaN on the one-pixel border.
    """
    z = np.asarray(z, float)
    dx = np.broadcast_to(np.asarray(dx_m, float).reshape(-1, 1) if np.ndim(dx_m) else dx_m, z.shape)
    gx = np.full(z.shape, np.nan)
    gy = np.full(z.shape, np.nan)
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]      # north row
    d, f = z[1:-1, :-2], z[1:-1, 2:]                      # middle row (west, east)
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]          # south row
    gx[1:-1, 1:-1] = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx[1:-1, 1:-1])
    # rows increase southward, so "north minus south" is the northward gradient
    gy[1:-1, 1:-1] = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * dy_m)
    # Horn's kernel skips the centre pixel, so a pixel with no height of its own would
    # still get a slope from its neighbours. Where the DEM has no data, report none.
    missing = ~np.isfinite(z)
    gx[missing] = np.nan
    gy[missing] = np.nan
    return gx, gy


def slope_aspect_from_gradients(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # downhill direction is minus the gradient; azimuth = atan2(east, north)
    aspect = np.degrees(np.arctan2(-gx, -gy)) % 360.0
    aspect = np.where(slope < FLAT_SLOPE_DEG, np.nan, aspect)
    return slope, aspect


def slope_aspect(patch) -> Terrain:
    """Slope and aspect for a DemPatch (io.dem), on the patch's own grid."""
    res = patch.res_px_per_deg
    dy_m = MOON_RADIUS_M * np.pi / 180.0 / res
    dx_m = dy_m * np.cos(np.radians(patch.lat))
    gx, gy = horn_gradients(patch.heights_m, dx_m, dy_m)
    slope, aspect = slope_aspect_from_gradients(gx, gy)
    return Terrain(slope, aspect, patch.source, patch.independent_of_references)


def hillshade(terrain: Terrain, sun_azimuth_deg: float, sun_elevation_deg: float) -> np.ndarray:
    """Lambertian shading of a slope/aspect pair under a given sun, in 0..1.

    cos(incidence) for a surface of the given slope and aspect: 1 where the ground
    faces the sun squarely, 0 where it turns away. This is how a camera would see
    the terrain if the surface reflected equally in all directions -- enough to ask
    what a descriptor does when the SUN moves and the GROUND does not.

    It is a rendering aid for exactly that question, not a photometric model of the
    Moon: it ignores opposition surge, roughness and albedo variation, and it does
    not cast shadows (a slope turned away goes to zero, but it cannot be shaded by
    a ridge somewhere else).
    """
    slope = np.radians(terrain.slope_deg)
    aspect = np.radians(terrain.aspect_deg)
    sun_zenith = np.radians(90.0 - sun_elevation_deg)
    sun_azimuth = np.radians(sun_azimuth_deg)
    cos_incidence = (np.cos(sun_zenith) * np.cos(slope)
                     + np.sin(sun_zenith) * np.sin(slope) * np.cos(sun_azimuth - aspect))
    # Flat ground has no aspect (NaN); it simply faces straight up.
    flat = ~np.isfinite(aspect) & np.isfinite(slope)
    cos_incidence = np.where(flat, np.cos(sun_zenith), cos_incidence)
    return np.clip(cos_incidence, 0.0, 1.0)
