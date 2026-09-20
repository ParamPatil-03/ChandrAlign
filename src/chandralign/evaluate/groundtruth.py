"""Cross-validating a label against independent physics. Owner: Member A (Part 1). Feature: GEO-07.

No public ground truth exists for Chandrayaan-2 registration, and labels can be
wrong or out of date (failure mode #8). So we never trust a single source: every
quantity a label states is recomputed a second way, from something the label did
not use to state it, and the two are reported TOGETHER with their disagreement.

    report = cross_validate(meta)
    report.agrees                 # False if any check is outside its tolerance
    for d in report.checks:
        d.name, d.label, d.independent, d.ratio, d.agrees

Four independent routes are available from metadata alone:

    ground sampling distance   from the OPTICS: pixel pitch x altitude / focal length.
                               Nothing in that chain is the stated GSD.
    cross-track pixel size     from the CORNERS: scene width / number of samples.
    along-track pixel size     from the CORNERS: scene length / number of lines, and
                               separately from ORBITAL MECHANICS: the circular-orbit
                               ground speed at the stated altitude times the line
                               period. Those two know nothing about each other.
    corner agreement           ISRO ships two corner sets, `system` and `refined`.
                               Their separation on the ground is an uncertainty bound
                               that costs nothing to compute.

WHAT THIS FOUND ON OUR OWN THREE PRODUCTS, none of it assumed in advance:

  1. OHRC's line_exposure_duration is declared `unit="ms"` and is NOT milliseconds.
     Ground speed x 205.320 ms gives 312 m per line where the corners give 0.309 m --
     out by a factor of 1009. Read as MICROSECONDS it agrees to 0.9%. TMC-2 and IIRS
     are genuinely in ms (1.00x and 1.01x), so this is OHRC's label, not our reading.

  2. IIRS's stated GSD is exactly 2.0000x what its optics give (97.15 m vs 48.58 m).
     That is 2x2 detector binning, and the factor is exact to four decimals rather
     than approximate, which is what tells us it is binning and not an error.

  3. CH-2 pushbroom pixels are NOT square, because along-track spacing is set by how
     far the spacecraft flies between lines, not by the optics:

         OHRC    0.300 m across   x  0.309 m along
         TMC-2   4.41  m across   x  5.065 m along
         IIRS   97.15  m across   x 79.52  m along     (the registry's nominal 80 m
                                                        was the ALONG-track figure)

     IIRS is the one that bites: assuming square pixels there is a 22% scale error in
     one axis, which no matcher will recover from.

SCOPE. GEO-07 as planned also registers the image against a DEM re-projection and
reports that disagreement. That half needs CH-2 pixels matched to rendered relief,
which is Part 2's matcher and is not built yet, so it is NOT claimed here. What is
here is the metadata half, which is independent, runs today, and already found three
things. `dem_coverage()` checks that a scene's footprint is inside the elevation data
we hold, which is the precondition for the other half.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from chandralign.contracts import SceneMeta
from chandralign.geometry.projection import surface_distance_m
from chandralign.io.pds_label import read_corner_sets, read_viewing_geometry

# Selenocentric gravitational parameter, m^3/s^2 (IAU/NASA value) and mean radius.
MOON_GM = 4.9048695e12
MOON_RADIUS_M = 1737400.0

# How far apart two independent routes to the same number may be before we call it a
# disagreement. Generous, because these are different physical chains, not repeats.
DEFAULT_TOLERANCE = 0.05          # 5%


@dataclass(frozen=True)
class Discrepancy:
    """One quantity, worked out two ways."""
    name: str
    label: float                  # what the label states (or implies)
    independent: float            # what the other route gives
    tolerance: float
    note: str = ""
    binning: int = 1              # a declared integer factor the two sides differ by

    @property
    def ratio(self) -> float:
        """independent / label, with any declared binning factor taken out."""
        return (self.independent * self.binning / self.label) if self.label else float("inf")

    @property
    def raw_ratio(self) -> float:
        """The same, WITHOUT the binning allowance, so the raw disagreement stays visible."""
        return self.independent / self.label if self.label else float("inf")

    @property
    def relative_error(self) -> float:
        return abs(self.ratio - 1.0)

    @property
    def agrees(self) -> bool:
        return self.relative_error <= self.tolerance


@dataclass(frozen=True)
class CrossCheck:
    """Every independent check on one product, with the disagreements kept, not hidden."""
    product_id: str
    checks: tuple[Discrepancy, ...]
    corner_disagreement_m: Optional[float] = None
    corner_sets: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[Discrepancy, ...]:
        return tuple(d for d in self.checks if not d.agrees)

    @property
    def agrees(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        head = f"{self.product_id}: {len(self.checks) - len(self.failures)}/{len(self.checks)} agree"
        lines = [f"  {'OK ' if d.agrees else 'OUT'} {d.name}: label {d.label:.4g}, "
                 f"independent {d.independent:.4g} ({d.ratio:.4f}x){' -- ' + d.note if d.note else ''}"
                 for d in self.checks]
        return "\n".join([head, *lines])


def orbital_ground_speed(altitude_km: float) -> float:
    """How fast the sub-spacecraft point moves over the ground, m/s, for a circular orbit.

    sqrt(GM / r) is the orbital speed at radius r; the point directly beneath moves
    slower by R/r, because it is on a smaller circle. Depends only on the stated
    altitude, so it is independent of anything the label says about pixels.
    """
    r = MOON_RADIUS_M + float(altitude_km) * 1000.0
    return float(np.sqrt(MOON_GM / r) * MOON_RADIUS_M / r)


def gsd_from_optics(pixel_width_um: float, altitude_km: float, focal_length_mm: float) -> float:
    """Ground sampling distance implied by the camera itself, in metres.

    A detector pixel subtends pitch / focal_length radians; at the stated altitude
    that angle covers this much ground. Uses no georeferencing at all.
    """
    if focal_length_mm <= 0:
        raise ValueError(f"focal length must be positive, got {focal_length_mm}")
    return (pixel_width_um * 1e-6) * (altitude_km * 1000.0) / (focal_length_mm * 1e-3)


def footprint_scales(corners: Sequence[tuple[float, float]],
                     array_shape: tuple[int, int]) -> tuple[float, float]:
    """(cross-track, along-track) metres per pixel, from the corner coordinates.

    corners are clockwise from upper-left, as read_corner_sets returns them. The two
    figures differ for a pushbroom camera and are not interchangeable.
    """
    ul, ur, _lr, ll = corners[:4]
    n_lines, n_samples = array_shape
    across = float(surface_distance_m(*ul, *ur)) / n_samples
    along = float(surface_distance_m(*ul, *ll)) / n_lines
    return across, along


def corner_disagreement_m(sets: dict[str, list[tuple[float, float]]]) -> Optional[float]:
    """Largest ground separation between matching corners of two corner sets.

    ISRO ships `system` (no reference used) and `refined` (adjusted against the
    reference named in the label). Where both exist, how far apart they put the same
    corner is a free, honest uncertainty bound on the label's geolocation.
    """
    if "system" not in sets or "refined" not in sets:
        return None
    return max(float(surface_distance_m(*a, *b))
               for a, b in zip(sets["system"], sets["refined"]))


def cross_validate(meta: SceneMeta, tolerance: float = DEFAULT_TOLERANCE,
                   line_period_s: Optional[float] = None) -> CrossCheck:
    """Recompute what the label states, independently, and report the disagreements.

    Returns every check it could make; a check whose inputs the label does not carry
    is omitted rather than filled in. `line_period_s` overrides the label's
    line_exposure_duration, which is how the OHRC unit defect above is demonstrated.
    """
    viewing = read_viewing_geometry(meta.label_path)
    sets = read_corner_sets(meta.label_path)
    checks: list[Discrepancy] = []

    corners = sets.get("system") or sets.get("refined")
    across = along = None
    if corners:
        across, along = footprint_scales(corners, meta.array_shape)
        checks.append(Discrepancy(
            "cross_track_gsd_m", meta.gsd_m, across, tolerance,
            "scene width / number of samples"))

    if viewing.altitude_km and viewing.focal_length_mm and viewing.detector_pixel_width_um:
        optics = gsd_from_optics(viewing.detector_pixel_width_um,
                                 viewing.altitude_km, viewing.focal_length_mm)
        # `independent` stays the RAW optics figure. Where the label states an exact
        # integer multiple of it, that is detector binning, and it is recorded as a
        # factor rather than multiplied in silently -- so the two numbers that were
        # actually computed both remain visible in the report.
        factor = meta.gsd_m / optics
        binning = round(factor)
        if binning < 1 or abs(factor - binning) > 0.01:
            binning = 1
        note = "camera optics only"
        if binning > 1:
            note = (f"camera optics only; the label states exactly {binning}x this, which is "
                    f"{binning}x{binning} detector binning, not an error")
        checks.append(Discrepancy("optics_gsd_m", meta.gsd_m, optics, tolerance, note, binning))

    period = line_period_s
    if period is None and viewing.line_period_s is not None:
        period = viewing.line_period_s
    if period is not None and viewing.altitude_km and along is not None:
        speed = orbital_ground_speed(viewing.altitude_km)
        flown = speed * period
        note = "ground speed at the stated altitude x the line period"
        # A near-exact power of 1000 means the label's declared UNIT is wrong, not the
        # geometry. Say which unit does work rather than quietly rescaling.
        factor = flown / along if along else float("inf")
        for power, better in ((1000.0, "microseconds"), (0.001, "seconds")):
            if abs(factor / power - 1.0) <= 0.05:
                note = (f"the declared unit {viewing.line_exposure_unit!r} is out by "
                        f"{power:g}x; read as {better} it agrees to "
                        f"{abs(flown / power / along - 1.0) * 100:.1f}%")
                break
        checks.append(Discrepancy("along_track_gsd_m", along, flown, tolerance, note))

    return CrossCheck(meta.product_id, tuple(checks),
                      corner_disagreement_m(sets), tuple(sorted(sets)))


def dem_coverage(meta: SceneMeta, tiles) -> dict:
    """Is this scene's footprint inside the elevation data we hold?

    The precondition for the DEM half of GEO-07. Reports coverage and the relief
    found, or says plainly that the footprint is not covered -- it never pretends
    partial coverage is whole.
    """
    from chandralign.io.dem import dem_patch

    sets = read_corner_sets(meta.label_path)
    corners = sets.get("system") or sets.get("refined")
    if not corners:
        return {"covered": False, "reason": "the label carries no corner coordinates"}
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    bounds = (min(lats), max(lats), min(lons), max(lons))
    try:
        patch = dem_patch(tiles, bounds)
    except (ValueError, KeyError) as exc:
        return {"covered": False, "bounds": bounds, "reason": str(exc)}
    heights = patch.heights_m
    finite = np.isfinite(heights)
    return {
        "covered": bool(finite.all()),
        "bounds": bounds,
        "source": patch.source,
        "independent_of_references": patch.independent_of_references,
        "filled_fraction": float(finite.mean()),
        "relief_m": float(np.nanmax(heights) - np.nanmin(heights)) if finite.any() else None,
    }


# ----------------------------------------------------------------------------- GEO-07: the DEM half

# Below this correlation between the image and the rendered relief, the phase
# correlation peak is not a measurement of anything and no offset is reported.
MIN_SHADING_CORRELATION = 0.35


@dataclass(frozen=True)
class DemRegistration:
    """Where a DEM says an image sits, against where its label says it sits.

    `offset_m` is None whenever `trustworthy` is False. A number here would be worse
    than nothing: it would be read as an uncertainty bound on the label.
    """
    product_id: str
    correlation: float
    shift_px: Optional[tuple[float, float]]
    offset_m: Optional[float]
    dem_source: str
    dem_independent_of_references: bool
    slope_median_deg: float
    trustworthy: bool
    reason: str


def render_relief(patch, sun_azimuth_deg: float, sun_elevation_deg: float) -> np.ndarray:
    """What the DEM predicts this ground looks like under a given sun."""
    from chandralign.geometry.dem_terrain import hillshade, slope_aspect

    return hillshade(slope_aspect(patch), sun_azimuth_deg, sun_elevation_deg)


def resample_to_dem_grid(meta: SceneMeta, window, patch, model=None) -> np.ndarray:
    """An image window averaged onto the DEM's own grid; NaN where nothing landed.

    A CH-2 image is far finer than any lunar DEM we hold (TMC-2 4.41 m against
    SLDEM's ~59 m), so the image comes down to the DEM rather than the DEM being
    invented upwards.
    """
    from chandralign.geometry.projection import geolocation_model
    from chandralign.io.pds_raster import read_raster

    if model is None:
        model = geolocation_model(meta)
    image = read_raster(meta, window).astype(np.float64)
    rows = np.arange(window.row, window.row + window.height)
    cols = np.arange(window.col, window.col + window.width)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    lat, lon = model.pixel_to_latlon(rr, cc)

    fr = np.rint((patch.lat[0] - lat) * patch.res_px_per_deg).astype(int)
    fc = np.rint((lon - patch.lon[0]) * patch.res_px_per_deg).astype(int)
    n_r, n_c = patch.heights_m.shape
    inside = (fr >= 0) & (fr < n_r) & (fc >= 0) & (fc < n_c)
    total = np.zeros((n_r, n_c))
    count = np.zeros((n_r, n_c))
    np.add.at(total, (fr[inside], fc[inside]), image[inside])
    np.add.at(count, (fr[inside], fc[inside]), 1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _standardise(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = np.where(valid, a, np.nanmean(a[valid]))
    return (filled - filled.mean()) / (filled.std() + 1e-12)


def register_to_dem(meta: SceneMeta, window, patch, model=None,
                    min_correlation: float = MIN_SHADING_CORRELATION) -> DemRegistration:
    """Register an image against relief rendered from a DEM under its OWN sun (GEO-07).

    The second, independent route to where an image sits: the label says one thing,
    and the shape of the ground says another. Their disagreement is the uncertainty
    bound GEO-07 exists to produce.

    MEASURED ON OUR OWN DATA, and the answer is a refusal. The method itself is
    sound -- injecting a known shift into a rendered scene recovers it exactly (see
    tests) -- but a real TMC-2 window correlates with SLDEM-rendered relief at only
    +0.14 on the roughest ground in the strip and about 0.00 elsewhere. Lunar mare
    at ~2 degrees median slope has almost no shading contrast at 59 m, and what
    variation TMC-2 does see there is albedo, which a Lambertian hillshade knows
    nothing about. So there is no peak to trust, and none is reported.

    That is the honest result, not a placeholder: it says the DEM route needs
    either finer elevation data or genuinely rough ground, and it says so with the
    number that decides it.
    """
    from chandralign.geometry.dem_terrain import slope_aspect
    from chandralign.geometry.solar import scene_illumination
    from skimage.registration import phase_cross_correlation

    illumination = scene_illumination(meta)
    if illumination.incidence_deg is None or illumination.sub_solar_azimuth_deg is None:
        return DemRegistration(meta.product_id, float("nan"), None, None, patch.source,
                               patch.independent_of_references, float("nan"), False,
                               "no sun geometry: the relief cannot be rendered")

    relief = render_relief(patch, illumination.sub_solar_azimuth_deg,
                           90.0 - illumination.incidence_deg)
    observed = resample_to_dem_grid(meta, window, patch, model)
    valid = np.isfinite(observed) & np.isfinite(relief)
    slope_median = float(np.nanmedian(slope_aspect(patch).slope_deg))
    if valid.sum() < 100:
        return DemRegistration(meta.product_id, float("nan"), None, None, patch.source,
                               patch.independent_of_references, slope_median, False,
                               f"only {int(valid.sum())} pixels overlap the DEM")

    rows = valid.any(axis=1)
    cols = valid.any(axis=0)
    a = _standardise(observed[np.ix_(rows, cols)], valid[np.ix_(rows, cols)])
    b = _standardise(relief[np.ix_(rows, cols)], valid[np.ix_(rows, cols)])
    sub = valid[np.ix_(rows, cols)]
    correlation = float(np.corrcoef(a[sub], b[sub])[0, 1])

    if not np.isfinite(correlation) or abs(correlation) < min_correlation:
        return DemRegistration(
            meta.product_id, correlation, None, None, patch.source,
            patch.independent_of_references, slope_median, False,
            f"image and rendered relief correlate {correlation:+.3f}, below "
            f"{min_correlation}: there is no peak to trust, so no offset is reported")

    shift = phase_cross_correlation(b, a, upsample_factor=10, normalization=None)[0]
    metres_per_pixel = MOON_RADIUS_M * np.pi / 180.0 / patch.res_px_per_deg
    offset = float(np.hypot(*shift) * metres_per_pixel)
    return DemRegistration(meta.product_id, correlation, (float(shift[0]), float(shift[1])),
                           offset, patch.source, patch.independent_of_references,
                           slope_median, True,
                           f"registered at correlation {correlation:+.3f}")
