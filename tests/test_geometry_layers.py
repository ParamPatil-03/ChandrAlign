"""GEO-02: per-pixel incidence, emission and phase.

The point of these tests is that the layers are DERIVED, not read, because no
Chandrayaan-2 label carries angle backplanes. So each piece is checked against
geometry that can be worked out by hand, and the whole thing is then checked
against the three real products we hold.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import GeometryLayers
from chandralign.geometry.solar import (
    MOON_RADIUS_KM,
    ROLL_TOLERANCE_DEG,
    angular_separation,
    azimuth_difference,
    bearing,
    emission_from_altitude,
    geometry_layers,
    phase_from_angles,
    sub_solar_point,
)
from chandralign.io.pds_label import parse_label, read_viewing_geometry

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
PRODUCTS = {
    "OHRC": "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml",
    "TMC2": "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml",
    "IIRS": "ch2_iir_nci_20240523T1600301891_d_img_d18.xml",
}


def meta(name):
    return parse_label(LABELS / PRODUCTS[name])


def grid(m, n=40):
    r, c = m.array_shape
    return np.linspace(0, r - 1, n).astype(int), np.linspace(0, c - 1, n).astype(int)


# ----------------------------------------------------------------------------- the primitives

def test_sub_solar_point_is_the_incidence_away_along_the_azimuth():
    """Sun 30 deg from vertical, due east: the sub-solar point is 30 deg of arc east."""
    lat, lon = sub_solar_point(0.0, 0.0, 30.0, 90.0)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert lon == pytest.approx(30.0)
    assert angular_separation(0.0, 0.0, lat, lon) == pytest.approx(30.0)
    assert bearing(0.0, 0.0, lat, lon) == pytest.approx(90.0)


def test_sub_solar_point_round_trips_from_anywhere():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lat, lon = rng.uniform(-80, 80), rng.uniform(-180, 180)
        inc, az = rng.uniform(1, 89), rng.uniform(0, 360)
        slat, slon = sub_solar_point(lat, lon, inc, az)
        assert angular_separation(lat, lon, slat, slon) == pytest.approx(inc, abs=1e-6)
        assert azimuth_difference(bearing(lat, lon, slat, slon), az) == pytest.approx(0, abs=1e-6)


def test_sun_overhead_means_zero_incidence_where_it_is_measured():
    lat, lon = sub_solar_point(12.0, 34.0, 0.0, 217.0)    # azimuth is meaningless at i = 0
    assert (float(lat), float(lon)) == pytest.approx((12.0, 34.0))


def test_emission_is_zero_at_nadir_and_grows_off_track():
    assert emission_from_altitude(0.0, 100.0) == 0.0
    e = emission_from_altitude([0.0, 0.1, 0.2, 0.5], 100.0)
    assert np.all(np.diff(e) > 0)


def test_emission_matches_a_hand_calculation():
    """A 5 km off-track pixel at 100 km altitude.

    The spacecraft sits at r = R + h and the zenith angle seen from the ground point
    is atan2(r sin psi, r cos psi - R), which gives 3.027 deg. The flat-ground answer
    atan(5/100) = 2.862 deg is a different number: the gap is real curvature, not
    slack in the test.
    """
    psi = np.degrees(5.0 / MOON_RADIUS_KM)
    assert emission_from_altitude(psi, 100.0) == pytest.approx(3.027, abs=0.001)
    assert np.degrees(np.arctan(5.0 / 100.0)) == pytest.approx(2.862, abs=0.001)


def test_phase_is_the_sun_to_camera_angle():
    assert phase_from_angles(40.0, 0.0, 123.0) == pytest.approx(40.0)    # nadir: phase = incidence
    assert phase_from_angles(0.0, 25.0, 77.0) == pytest.approx(25.0)     # sun overhead
    assert phase_from_angles(30.0, 30.0, 0.0) == pytest.approx(0.0, abs=1e-6)     # same direction
    assert phase_from_angles(30.0, 30.0, 180.0) == pytest.approx(60.0)            # opposite sides


def test_azimuth_difference_works_on_arrays_and_scalars():
    assert azimuth_difference(350.0, 10.0) == 20.0
    assert isinstance(azimuth_difference(350.0, 10.0), float)
    out = azimuth_difference(np.array([350.0, 10.0, 0.0]), np.array([10.0, 350.0, 180.0]))
    assert np.allclose(out, [20.0, 20.0, 180.0])


# ----------------------------------------------------------------------------- the real products

def test_every_real_product_gets_derived_layers():
    for name in PRODUCTS:
        m = meta(name)
        g = geometry_layers(m, *grid(m))
        assert isinstance(g, GeometryLayers)
        assert g.source == "derived"          # NOT "label": no CH-2 label carries backplanes
        assert g.incidence_deg is not None and np.all(np.isfinite(g.incidence_deg))
        assert 0.0 <= g.incidence_deg.min() and g.incidence_deg.max() <= 180.0


def test_the_scene_centre_reproduces_the_labelled_angle():
    """The anchor has to hold: at the centre, the derived value IS the label's."""
    for name in PRODUCTS:
        m = meta(name)
        g = geometry_layers(m, *grid(m, n=41))
        assert g.incidence_deg[20, 20] == pytest.approx(m.solar_incidence_deg, abs=0.15), name


def test_long_strips_vary_and_short_ones_do_not():
    """The measurement that justifies GEO-02 existing at all.

    A single scene-level incidence is a fair description of OHRC and a poor one of
    TMC-2 and IIRS, purely because of how much ground they cover.
    """
    spreads = {}
    for name in PRODUCTS:
        m = meta(name)
        g = geometry_layers(m, *grid(m))
        spreads[name] = float(g.incidence_deg.max() - g.incidence_deg.min())
    assert spreads["OHRC"] < 0.3                  # 25 km of ground: one number is honest
    assert spreads["TMC2"] > 5.0                  # 812 km: it is not
    assert spreads["IIRS"] > 5.0                  # 1042 km
    assert spreads["TMC2"] > 20 * spreads["OHRC"]


def test_emission_is_zero_on_the_nadir_column_and_largest_at_the_edge():
    for name in PRODUCTS:
        m = meta(name)
        g = geometry_layers(m, *grid(m, n=41))
        assert g.emission_deg is not None, name
        centre = g.emission_deg[:, 20]
        assert centre.max() < 0.5, name                       # the track runs down the middle
        assert g.emission_deg[:, 0].mean() > centre.mean(), name
        assert g.emission_deg[:, -1].mean() > centre.mean(), name


def test_phase_never_exceeds_incidence_plus_emission():
    """A triangle inequality on the sphere; a correct layer cannot violate it."""
    for name in PRODUCTS:
        m = meta(name)
        g = geometry_layers(m, *grid(m))
        assert np.all(g.phase_deg <= g.incidence_deg + g.emission_deg + 1e-6), name
        assert np.all(g.phase_deg >= np.abs(g.incidence_deg - g.emission_deg) - 1e-6), name


# ----------------------------------------------------------------------------- refusals

def test_emission_is_withheld_when_the_label_gives_no_altitude():
    m = meta("OHRC")
    blank = read_viewing_geometry(LABELS / "M1417360906LC.XML")   # PDS3: no ISRO fields
    assert blank.altitude_km is None
    g = geometry_layers(m, *grid(m), viewing=blank)
    assert g.incidence_deg is not None                      # the sun part still works
    assert g.emission_deg is None and g.phase_deg is None   # the camera part does not, and says so


def test_a_rolled_scene_withholds_emission_rather_than_biasing_it():
    m = meta("OHRC")
    rolled = replace(read_viewing_geometry(m.label_path), roll_deg=ROLL_TOLERANCE_DEG + 5.0)
    g = geometry_layers(m, *grid(m), viewing=rolled)
    assert g.emission_deg is None and g.phase_deg is None


def test_a_scene_with_no_sun_angle_reports_unknown():
    m = replace(meta("OHRC"), solar_incidence_deg=None, sub_solar_azimuth_deg=None)
    g = geometry_layers(m)
    assert g.source == "unknown" and g.incidence_deg is None
