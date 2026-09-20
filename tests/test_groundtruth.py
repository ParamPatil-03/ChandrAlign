"""GEO-07: cross-validating labels against independent physics.

Each check recomputes something the label states, by a route the label did not use
to state it. The tests pin the numbers that came out of the real products, and the
last one injects a deliberate error to prove the checks would catch it.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chandralign.evaluate.groundtruth import (
    MOON_RADIUS_M,
    corner_disagreement_m,
    cross_validate,
    dem_coverage,
    footprint_scales,
    gsd_from_optics,
    orbital_ground_speed,
)
from chandralign.io.dem import find_tiles
from chandralign.io.pds_label import parse_label, read_corner_sets, read_viewing_geometry

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
PRODUCTS = {
    "OHRC": "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml",
    "TMC2": "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml",
    "IIRS": "ch2_iir_nci_20240523T1600301891_d_img_d18.xml",
}


def meta(name):
    return parse_label(LABELS / PRODUCTS[name])


def named(report, key):
    return next(d for d in report.checks if d.name == key)


# ----------------------------------------------------------------------------- the physics

def test_orbital_ground_speed_is_physical():
    """Lower orbit -> faster. Checked against the closed form at the surface."""
    assert orbital_ground_speed(100.0) == pytest.approx(1544.9, abs=1.0)
    assert orbital_ground_speed(50.0) > orbital_ground_speed(200.0)
    surface = np.sqrt(4.9048695e12 / MOON_RADIUS_M)
    assert orbital_ground_speed(0.0) == pytest.approx(surface, rel=1e-9)


def test_gsd_from_optics_is_the_similar_triangles_answer():
    # a 10 um pixel behind a 1 m lens at 100 km sees 1 m of ground
    assert gsd_from_optics(10.0, 100.0, 1000.0) == pytest.approx(1.0)
    assert gsd_from_optics(5.0, 100.0, 1000.0) == pytest.approx(0.5)    # smaller pixel
    assert gsd_from_optics(10.0, 50.0, 1000.0) == pytest.approx(0.5)    # lower orbit
    with pytest.raises(ValueError, match="positive"):
        gsd_from_optics(10.0, 100.0, 0.0)


def test_optics_reproduce_the_stated_gsd_on_the_two_unbinned_cameras():
    """Nothing in pitch x altitude / focal length is the stated GSD, so this is real."""
    for name, expected in (("OHRC", 0.30), ("TMC2", 4.41)):
        v = read_viewing_geometry(LABELS / PRODUCTS[name])
        optics = gsd_from_optics(v.detector_pixel_width_um, v.altitude_km, v.focal_length_mm)
        assert optics == pytest.approx(expected, rel=0.01), name


def test_iirs_is_binned_exactly_two_by_two():
    """97.15 / 48.58 is 2.0000, not 1.9 or 2.1 -- exactness is what proves it is binning."""
    v = read_viewing_geometry(LABELS / PRODUCTS["IIRS"])
    optics = gsd_from_optics(v.detector_pixel_width_um, v.altitude_km, v.focal_length_mm)
    assert meta("IIRS").gsd_m / optics == pytest.approx(2.0, abs=0.001)
    check = named(cross_validate(meta("IIRS")), "optics_gsd_m")
    assert "binning" in check.note
    # The RAW optics figure is what gets reported; the binning is a separate factor,
    # so the check cannot be made to agree by quietly copying the label's own number.
    assert check.independent == pytest.approx(48.577, abs=0.01)
    assert check.binning == 2
    assert check.raw_ratio == pytest.approx(0.5, abs=0.001)   # the unreconciled disagreement
    assert check.ratio == pytest.approx(1.0, abs=0.001)       # reconciled by the binning
    assert check.agrees


def test_binning_is_only_allowed_for_an_exact_integer_factor():
    """A 1.7x disagreement is an error, not binning, and must not be explained away."""
    m = meta("TMC2")
    bad = replace(m, gsd_m=m.gsd_m * 1.7)
    check = named(cross_validate(bad), "optics_gsd_m")
    assert check.binning == 1 and not check.agrees


# ----------------------------------------------------------------------------- the label defect

def test_ohrc_declares_the_wrong_unit_on_its_line_period():
    """Found, not assumed: ms is out by ~1000x and microseconds agrees to ~1%."""
    m = meta("OHRC")
    v = read_viewing_geometry(m.label_path)
    assert v.line_exposure_unit == "ms" and v.line_exposure == pytest.approx(205.320)
    check = named(cross_validate(m), "along_track_gsd_m")
    assert not check.agrees
    assert check.ratio == pytest.approx(1009.0, rel=0.01)
    assert "microseconds" in check.note
    # and read as microseconds it does agree
    fixed = named(cross_validate(m, line_period_s=205.320e-6), "along_track_gsd_m")
    assert fixed.agrees and fixed.ratio == pytest.approx(1.009, abs=0.01)


def test_the_other_two_cameras_really_are_in_milliseconds():
    """Which is what makes the OHRC result a defect in that label, not in our reading."""
    for name in ("TMC2", "IIRS"):
        check = named(cross_validate(meta(name)), "along_track_gsd_m")
        assert check.agrees, name
        assert check.ratio == pytest.approx(1.0, abs=0.02), name


def test_the_parser_does_not_silently_fix_the_unit():
    """line_period_s takes the declared unit at face value, so the defect stays visible."""
    v = read_viewing_geometry(LABELS / PRODUCTS["OHRC"])
    assert v.line_period_s == pytest.approx(0.20532)      # ms as declared, NOT 2.05e-7


# ----------------------------------------------------------------------------- pushbroom pixels

def test_pushbroom_pixels_are_not_square():
    """Along-track spacing comes from how far the spacecraft flies, not from the optics."""
    shapes = {}
    for name in PRODUCTS:
        m = meta(name)
        corners = read_corner_sets(m.label_path)["system"]
        shapes[name] = footprint_scales(corners, m.array_shape)
    across, along = shapes["IIRS"]
    assert across == pytest.approx(100.0, rel=0.02)
    assert along == pytest.approx(79.5, rel=0.02)
    assert abs(along / across - 1.0) > 0.15          # 20%+ out: square pixels are wrong here
    for name in ("OHRC", "TMC2"):
        a, b = shapes[name]
        assert a != pytest.approx(b, rel=1e-3), name


# ----------------------------------------------------------------------------- corner sets

def test_corner_disagreement_bounds_the_geolocation():
    """ISRO ships two corner sets; how far apart they are is a free uncertainty bound."""
    gaps = {n: cross_validate(meta(n)).corner_disagreement_m for n in PRODUCTS}
    assert gaps["OHRC"] == pytest.approx(0.0, abs=1.0)    # reference_data_used = System
    assert gaps["TMC2"] > 1000.0                          # kilometres apart
    assert gaps["IIRS"] > gaps["TMC2"]
    assert all(g is not None for g in gaps.values())


def test_corner_disagreement_is_none_when_there_is_only_one_set():
    assert corner_disagreement_m({"system": [(0.0, 0.0)] * 4}) is None
    assert corner_disagreement_m({}) is None


# ----------------------------------------------------------------------------- injected errors

def test_an_injected_gsd_error_is_caught():
    """GEO-07 acceptance: a deliberately wrong label must not pass."""
    good = cross_validate(meta("TMC2"))
    assert named(good, "optics_gsd_m").agrees

    broken = replace(meta("TMC2"), gsd_m=meta("TMC2").gsd_m * 1.5)
    report = cross_validate(broken)
    assert not report.agrees
    assert "optics_gsd_m" in {d.name for d in report.failures}


def test_an_injected_altitude_error_is_caught():
    """Altitude feeds optics and orbital speed, so corrupting it breaks two checks at once."""
    m = meta("TMC2")
    v = read_viewing_geometry(m.label_path)
    wrong = gsd_from_optics(v.detector_pixel_width_um, v.altitude_km * 2, v.focal_length_mm)
    assert abs(wrong / m.gsd_m - 1.0) > 0.05      # well outside tolerance


def test_a_small_error_still_passes_so_the_test_is_not_vacuous():
    m = meta("TMC2")
    nudged = replace(m, gsd_m=m.gsd_m * 1.01)     # 1%, inside the 5% tolerance
    assert named(cross_validate(nudged), "optics_gsd_m").agrees


def test_the_tolerance_is_honoured():
    m = meta("TMC2")
    broken = replace(m, gsd_m=m.gsd_m * 1.10)
    assert named(cross_validate(broken, tolerance=0.05), "optics_gsd_m").agrees is False
    assert named(cross_validate(broken, tolerance=0.20), "optics_gsd_m").agrees is True


# ----------------------------------------------------------------------------- DEM precondition

REAL_DEM = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")


@pytest.mark.skipif(len(REAL_DEM) < 1, reason="SLDEM tiles not downloaded")
def test_dem_coverage_reports_the_truth_about_our_elevation_data():
    """OHRC's 25 km footprint is covered; the 1,000 km strips are not."""
    ohrc = dem_coverage(meta("OHRC"), REAL_DEM)
    assert ohrc["covered"] is True
    assert ohrc["filled_fraction"] == 1.0
    assert ohrc["relief_m"] > 0
    assert ohrc["independent_of_references"] is False     # SLDEM uses SELENE stereo

    # All three footprints turn out to be covered: we hold two SLDEM tiles spanning
    # 30S-30N, 0-45E, and every CH-2 product we have sits inside that box. Recorded
    # because it was checked rather than assumed -- the DEM half of GEO-07 is not
    # blocked on data.
    iirs = dem_coverage(meta("IIRS"), REAL_DEM)           # spans 34 deg of latitude
    assert iirs["covered"] is True and iirs["filled_fraction"] == 1.0
    assert iirs["relief_m"] > 5000.0                      # 34 deg of latitude of terrain
    assert dem_coverage(meta("TMC2"), REAL_DEM)["covered"] is True
    assert ohrc["relief_m"] < iirs["relief_m"]            # 25 km of ground holds less


def test_dem_coverage_refuses_a_label_with_no_corners():
    m = meta("OHRC")
    out = dem_coverage(replace(m, label_path=LABELS / "M1417360906LC.XML"), REAL_DEM)
    assert out["covered"] is False and "corner" in out["reason"]


# ----------------------------------------------------------------------------- the report object

def test_the_report_keeps_disagreements_rather_than_hiding_them():
    report = cross_validate(meta("TMC2"))
    assert len(report.checks) == 3
    assert report.agrees == (len(report.failures) == 0)
    assert "TMC2".lower() in report.product_id.lower() or report.product_id
    text = report.summary()
    assert "agree" in text and "label" in text
    for d in report.checks:
        assert d.name in text


# ----------------------------------------------------------------------------- the DEM half

@pytest.mark.skipif(len(REAL_DEM) < 2, reason="SLDEM tiles not downloaded")
class TestDemRegistration:
    """GEO-07's second, independent route to where an image sits.

    The outcome on our own data is a REFUSAL, and that is the finding: the method
    works, the signal is not there. Both halves are pinned, because a refusal is
    only honest if the thing refusing has been shown to work.
    """

    def patch(self):
        from chandralign.io.dem import dem_patch
        return dem_patch(REAL_DEM, (11.68, 12.47, 22.54, 23.32))

    def test_the_method_recovers_a_shift_it_was_given(self):
        """The control. Without this, 'no offset found' would mean nothing."""
        from skimage.registration import phase_cross_correlation

        from chandralign.evaluate.groundtruth import render_relief

        relief = render_relief(self.patch(), 104.27, 44.0)
        filled = np.nan_to_num(relief, nan=float(np.nanmean(relief)))
        for dy, dx in ((7, -5), (20, 13)):
            moved = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
            shift = phase_cross_correlation(filled, moved, upsample_factor=10,
                                            normalization=None)[0]
            assert (int(shift[0]), int(shift[1])) == (-dy, -dx)

    def test_real_tmc2_does_not_correlate_with_rendered_relief(self):
        """The finding. Lunar mare at ~2 deg slope has almost no shading contrast at
        59 m, and what TMC-2 does see there is albedo, which a Lambertian hillshade
        knows nothing about."""
        from chandralign.evaluate.groundtruth import register_to_dem
        from chandralign.geometry.projection import geolocation_model
        from chandralign.io.dem import dem_patch
        from chandralign.io.pds_raster import Window

        tmc2 = next(iter(ROOT.glob(
            "data/raw/ch2/tmc2/products/*/data/calibrated/*/*_d_img_d18.xml")), None)
        if tmc2 is None:
            pytest.skip("TMC-2 product not downloaded")
        meta = parse_label(tmc2)
        model = geolocation_model(meta)
        row, height = 102400, 8192                      # the roughest ground in the strip
        lat, lon = model.pixel_to_latlon(np.array([row, row + height]),
                                         np.array([0, meta.array_shape[1] - 1]))
        patch = dem_patch(REAL_DEM, (float(lat.min()) - 0.03, float(lat.max()) + 0.03,
                                     float(lon.min()) - 0.06, float(lon.max()) + 0.06))
        result = register_to_dem(meta, Window(row, 0, height, meta.array_shape[1]),
                                 patch, model)
        assert abs(result.correlation) < 0.35
        assert result.trustworthy is False
        assert result.offset_m is None                  # NOT a number we could misread
        assert result.shift_px is None
        assert "no peak to trust" in result.reason
        assert result.slope_median_deg > 0
        assert result.dem_independent_of_references is False     # SLDEM uses SELENE

    def test_a_scene_with_no_sun_geometry_refuses_before_rendering(self):
        from chandralign.evaluate.groundtruth import register_to_dem
        from chandralign.io.pds_raster import Window

        m = replace(meta("OHRC"), solar_incidence_deg=None, sub_solar_azimuth_deg=None)
        out = register_to_dem(m, Window(0, 0, 8, 8), self.patch())
        assert out.trustworthy is False and out.offset_m is None
        assert "sun geometry" in out.reason

    def test_an_image_that_misses_the_dem_refuses(self):
        from chandralign.evaluate.groundtruth import register_to_dem
        from chandralign.geometry.projection import geolocation_model
        from chandralign.io.pds_raster import Window

        real = next(iter(ROOT.glob(
            "data/raw/ch2/ohrc/products/*/data/calibrated/*/*_d_img_d18.xml")), None)
        if real is None:
            pytest.skip("OHRC product not downloaded")
        m = parse_label(real)                           # sits near 0 deg, not 12 deg N
        out = register_to_dem(m, Window(0, 0, 256, 256), self.patch(),
                              geolocation_model(m))
        assert out.trustworthy is False and out.offset_m is None
        assert "overlap" in out.reason

    def test_render_relief_is_bounded_and_shaped_like_the_patch(self):
        from chandralign.evaluate.groundtruth import render_relief

        patch = self.patch()
        relief = render_relief(patch, 104.27, 44.0)
        assert relief.shape == patch.heights_m.shape
        finite = relief[np.isfinite(relief)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0
        assert finite.std() > 0
