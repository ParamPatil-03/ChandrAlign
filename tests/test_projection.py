"""GEO-05: lunar CRS, explicit reprojection, and pixel <-> ground models."""
import math
from pathlib import Path

import numpy as np
import pytest

from chandralign.geometry.projection import (
    CornerModel,
    GeolocationUnavailable,
    geolocation_model,
    is_map_projected,
    load_corner_model,
    load_grid_model,
    from_map,
    scene_crs,
    surface_distance_m,
    to_map,
)
from chandralign.io.pds_label import parse_label, read_corner_sets

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
OHRC = parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
TMC2 = parse_label(LABELS / "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml")
IIRS = parse_label(LABELS / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml")
IIRS_GRID = LABELS / "ch2_iir_nci_20240523T1600301891_g_grd_d18.csv"
NAC = parse_label(LABELS / "M1417360906LC.XML")
TC_N = parse_label(LABELS / "TCO_MAP_02_N03E021N00E024SC.lbl")
TC_S = parse_label(LABELS / "TCO_MAP_02_N00E021S03E024SC.lbl")
R = 1737400.0


# ----------------------------------------------------------------------------- CRS and reprojection

@pytest.mark.parametrize("meta", [OHRC, TMC2, IIRS, TC_N, TC_S], ids=lambda m: m.instrument)
def test_corners_reproject_and_back_within_1e6_deg(meta):
    """PLAN.md P1-T11 acceptance, on every product's real corners."""
    lats = np.array([c[0] for c in meta.corner_latlon])
    lons = np.array([c[1] for c in meta.corner_latlon])
    crs = scene_crs(lats.mean(), lons.mean())
    x, y = to_map(lats, lons, crs)
    lat2, lon2 = from_map(x, y, crs)
    assert np.max(np.abs(lat2 - lats)) < 1e-6 and np.max(np.abs(lon2 - lons)) < 1e-6


def test_polar_scene_uses_polar_stereographic_and_round_trips():
    crs = scene_crs(-85.0, 30.0)
    assert "polar stereographic south" in crs.name
    lat, lon = np.array([-80.0, -89.5, -75.2]), np.array([10.0, 150.0, -170.0])
    lat2, lon2 = from_map(*to_map(lat, lon, crs), crs)
    assert np.allclose(lat2, lat, atol=1e-9) and np.allclose(lon2, lon, atol=1e-9)
    assert "equirectangular" in scene_crs(10.0, 23.5).name


def test_projection_is_on_the_moon_sphere():
    crs = scene_crs(10.0, 23.5)
    x, _ = to_map(10.0, 24.5, crs)                     # 1 degree east at the true-scale latitude
    assert x == pytest.approx(R * math.pi / 180 * math.cos(math.radians(10)), rel=1e-9)
    assert surface_distance_m(0, 0, 0, 90) == pytest.approx(R * math.pi / 2, rel=1e-12)


# ----------------------------------------------------------------------------- honesty about sensor geometry

def test_only_map_products_are_map_projected():
    assert is_map_projected(TC_N)
    for meta in (OHRC, TMC2, IIRS, NAC):
        assert not is_map_projected(meta)              # failure mode #9: never assume a projection


def test_nac_has_no_pixel_geolocation_yet():
    with pytest.raises(GeolocationUnavailable, match="SPICE"):
        geolocation_model(NAC)


def test_default_models_are_reference_independent():
    for meta in (OHRC, TMC2, IIRS, TC_N):
        assert geolocation_model(meta).independent_of_references
    with pytest.raises(ValueError):
        geolocation_model(OHRC, prefer="best")


# ----------------------------------------------------------------------------- SELENE map model

@pytest.mark.parametrize("meta", [TC_N, TC_S], ids=["north", "south"])
def test_map_model_reproduces_the_label_corners(meta):
    model = geolocation_model(meta)
    L, S = meta.array_shape
    got = [model.pixel_to_latlon(r, c) for r, c in [(0, 0), (0, S - 1), (L - 1, S - 1), (L - 1, 0)]]
    for (lat, lon), (want_lat, want_lon) in zip(got, meta.corner_latlon):
        assert abs(lat - want_lat) < 1e-6 and abs(lon - want_lon) < 1e-6


def test_map_model_convention_is_zero_based():
    """Our 0-based form equals the PDS3 standard relation (which carries a +1 for 1-based
    indices). Dropping that +1 -- feeding a 1-based index into our formula -- misses the
    label corners by a full pixel."""
    model = geolocation_model(TC_N)
    lat_1based, _ = model.pixel_to_latlon(1, 1)
    assert abs(lat_1based - TC_N.corner_latlon[0][0]) == pytest.approx(1 / 4096)


def test_map_model_inverse_is_exact():
    model = geolocation_model(TC_N)
    rows, cols = np.array([0.0, 6143.5, 12287.0]), np.array([12287.0, 17.25, 0.0])
    r2, c2 = model.latlon_to_pixel(*model.pixel_to_latlon(rows, cols))
    assert np.allclose(r2, rows, atol=1e-7) and np.allclose(c2, cols, atol=1e-7)


# ----------------------------------------------------------------------------- CH-2 corner model

@pytest.mark.parametrize("meta", [OHRC, TMC2, IIRS], ids=lambda m: m.instrument)
def test_corner_model_hits_corners_and_inverts(meta):
    model = load_corner_model(meta)
    L, S = meta.array_shape
    for (r, c), want in zip([(0, 0), (0, S - 1), (L - 1, S - 1), (L - 1, 0)], meta.corner_latlon):
        assert model.pixel_to_latlon(r, c) == pytest.approx(want, abs=1e-12)
    rows = np.array([0.0, L / 3, L - 1.0, 12.5])
    cols = np.array([S - 1.0, S / 2, 0.0, 7.25])
    r2, c2 = model.latlon_to_pixel(*model.pixel_to_latlon(rows, cols))
    assert np.max(np.abs(r2 - rows)) < 1e-6 and np.max(np.abs(c2 - cols)) < 1e-6


def test_corner_model_handles_the_antimeridian():
    m = CornerModel([(1.0, 179.9), (1.0, -179.9), (0.0, -179.9), (0.0, 179.9)], lines=11, samples=11)
    lat, lon = m.pixel_to_latlon(5, 5)
    assert lat == pytest.approx(0.5) and abs(lon) == pytest.approx(180.0)


# ----------------------------------------------------------------------------- CH-2 grid model (IIRS fixture)

def test_iirs_grid_loads_drops_padding_and_is_flagged():
    grid = load_grid_model(IIRS, IIRS_GRID)
    assert grid.padding_rows_dropped == 6                 # all-zero rows at the end of ISRO's file
    assert grid.refined_against == "SELENE"
    assert grid.independent_of_references is False       # tuned against one of our references
    assert grid.source == "label_grid_refined"
    assert grid.lat.shape == (263, 6)


def test_iirs_grid_reproduces_its_own_nodes():
    grid = load_grid_model(IIRS, IIRS_GRID)
    raw = np.loadtxt(IIRS_GRID, delimiter=",", skiprows=1)
    raw = raw[~np.all(raw == 0.0, axis=1)]
    lat, lon = grid.pixel_to_latlon(raw[:, 3], raw[:, 2])
    assert np.max(np.abs(lat - raw[:, 1])) < 1e-9 and np.max(np.abs(lon - raw[:, 0])) < 1e-9


def test_iirs_grid_corners_are_the_refined_corners():
    grid = load_grid_model(IIRS, IIRS_GRID)
    L, S = IIRS.array_shape
    got = [grid.pixel_to_latlon(r, c) for r, c in [(0, 0), (0, S - 1), (L - 1, S - 1), (L - 1, 0)]]
    refined = read_corner_sets(IIRS.label_path)["refined"]
    system = read_corner_sets(IIRS.label_path)["system"]
    assert max(surface_distance_m(a[0], a[1], b[0], b[1]) for a, b in zip(got, refined)) < 1.0
    assert min(surface_distance_m(a[0], a[1], b[0], b[1]) for a, b in zip(got, system)) > 10_000


def test_iirs_grid_inverse_and_bounds():
    grid = load_grid_model(IIRS, IIRS_GRID)
    rows, cols = np.array([0.0, 6550.3, 13100.0]), np.array([249.0, 124.6, 0.0])
    r2, c2 = grid.latlon_to_pixel(*grid.pixel_to_latlon(rows, cols))
    assert np.max(np.abs(r2 - rows)) < 1e-6 and np.max(np.abs(c2 - cols)) < 1e-6
    with pytest.raises(ValueError, match="outside"):
        grid.pixel_to_latlon(13101.0, 0.0)


# ----------------------------------------------------------------------------- real products

def _real(pattern):
    hits = sorted((ROOT / "data" / "raw").glob(pattern))
    return parse_label(hits[0]) if hits else None


R_OHRC = _real("ch2/ohrc/products/*/data/calibrated/*/*_d_img_d18.xml")
R_TMC2 = _real("ch2/tmc2/products/*/data/calibrated/*/*_d_img_d18.xml")
needs_real = pytest.mark.skipif(R_OHRC is None or R_TMC2 is None, reason="real products not downloaded")


@needs_real
def test_real_ohrc_grid_is_system_only_and_agrees_with_corners():
    grid = geolocation_model(R_OHRC, prefer="precise")
    assert grid.refined_against == "System" and grid.independent_of_references
    corner = load_corner_model(R_OHRC)
    L, S = R_OHRC.array_shape
    rr, cc = np.meshgrid(np.linspace(0, L - 1, 40), np.linspace(0, S - 1, 15))
    d = surface_distance_m(*grid.pixel_to_latlon(rr, cc), *corner.pixel_to_latlon(rr, cc))
    assert d.max() < 1.0                                  # metres: a short, straight strip


@needs_real
def test_real_tmc2_independent_and_refined_geolocation_differ_by_km():
    """The measured size of ISRO's SELENE-based correction for this TMC-2 strip."""
    grid = geolocation_model(R_TMC2, prefer="precise")
    assert grid.refined_against == "SELENE" and not grid.independent_of_references
    corner = geolocation_model(R_TMC2)
    L, S = R_TMC2.array_shape
    rr, cc = np.meshgrid(np.linspace(0, L - 1, 60), np.linspace(0, S - 1, 20))
    d = surface_distance_m(*grid.pixel_to_latlon(rr, cc), *corner.pixel_to_latlon(rr, cc))
    assert 4_000 < np.median(d) < 6_500
