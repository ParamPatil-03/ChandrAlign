import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from chandralign.contracts import MatchSet, Metrics, SceneMeta, TransformModel
from chandralign.estimate import models
from chandralign.product.warp import best_geometry, export_bundle, warp_array


def _bundle(model, *, tps=None, parallax=None, shift=(3, -2)):
    src = np.zeros((40, 50), np.float32)
    src[15, 20] = 1
    ref = cv2.warpPerspective(src, model.matrix, (50, 40))
    points = np.array([[5, 5], [40, 5], [5, 30], [40, 30]], float)
    delivered = MatchSet(points, points + shift, np.ones(4, np.float32),
                         "test", "same_modal_normal", "delivered")
    result = SimpleNamespace(model=model, metrics=Metrics())
    plane = lambda array: SimpleNamespace(array=array, meta=SimpleNamespace(product_id="mock"))
    return SimpleNamespace(result=result, delivered=delivered, tps=tps, parallax=parallax,
                           src=plane(src), ref=plane(ref))


def test_affine_warp_places_source_on_reference_grid():
    matrix = np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], float)
    bundle = _bundle(TransformModel("affine", matrix=matrix))
    warped, kind = warp_array(bundle, interpolation=cv2.INTER_NEAREST)
    assert kind == "affine"
    assert np.array_equal(warped, bundle.ref.array)


def test_subpixel_warp_then_inverse_recovers_feature_position():
    yy, xx = np.mgrid[:80, :90]
    src = np.exp(-((xx - 40.2) ** 2 + (yy - 35.7) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
    matrix = np.array([[1, 0, 3.4], [0, 1, -2.25], [0, 0, 1]], float)
    ref = cv2.warpPerspective(src, matrix, (90, 80), flags=cv2.INTER_CUBIC)
    base = _bundle(TransformModel("affine", matrix=matrix))
    base.src.array, base.ref.array = src, ref
    warped, _ = warp_array(base)
    recovered = cv2.warpPerspective(warped, np.linalg.inv(matrix), (90, 80), flags=cv2.INTER_CUBIC)
    mass = recovered.sum()
    centre = np.array([(recovered * xx).sum() / mass, (recovered * yy).sum() / mass])
    assert np.linalg.norm(centre - [40.2, 35.7]) < 0.1


def test_geometry_priority_is_parallax_then_tps_then_affine():
    affine = TransformModel("affine", matrix=np.eye(3))
    tps = TransformModel("tps", tps_params={"interpolator": object()})
    parallax = SimpleNamespace()
    bundle = _bundle(affine, tps=tps, parallax=parallax)
    assert best_geometry(bundle) == ("parallax", parallax)
    bundle.parallax = None
    assert best_geometry(bundle) == ("tps", tps)
    bundle.tps = None
    assert best_geometry(bundle) == ("affine", affine)


def test_tps_warp_uses_delivered_points_to_build_inverse_map():
    affine = TransformModel("affine", matrix=np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], float))
    bundle = _bundle(affine)
    bundle.tps = models.fit_tps(bundle.delivered.src_pts, bundle.delivered.ref_pts, smoothing=0.0)
    warped, kind = warp_array(bundle, interpolation=cv2.INTER_NEAREST)
    assert kind == "tps"
    assert np.array_equal(warped, bundle.ref.array)


def test_parallax_warp_requires_dem_height_lookup():
    affine = TransformModel("affine", matrix=np.eye(3))
    bundle = _bundle(affine, parallax=SimpleNamespace())
    with pytest.raises(ValueError, match="heights_at"):
        warp_array(bundle)


def test_parallax_warp_runs_when_dem_height_lookup_is_supplied():
    from chandralign.estimate.models import ParallaxModel
    affine = TransformModel("affine", matrix=np.eye(3))
    parallax = ParallaxModel(np.eye(3), (0.0, 0.0), h0_m=100.0)
    bundle = _bundle(affine, parallax=parallax, shift=(0, 0))
    warped, kind = warp_array(bundle, heights_at=lambda pts: np.full(len(pts), 100.0),
                              interpolation=cv2.INTER_NEAREST)
    assert kind == "parallax"
    assert np.array_equal(warped, bundle.src.array)


class MapModel:
    resolution_px_per_deg = 10.0
    line_offset = 20.0
    sample_offset = 25.0
    center_lat = -80.0
    center_lon = 30.0

    def pixel_to_latlon(self, rows, cols):
        return (self.center_lat + (self.line_offset - np.asarray(rows)) / self.resolution_px_per_deg,
                self.center_lon + (np.asarray(cols) - self.sample_offset) / self.resolution_px_per_deg)


def test_geotiff_has_reference_geotransform_crs_tags_and_sidecar(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    matrix = np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], float)
    bundle = _bundle(TransformModel("affine", matrix=matrix))
    for role in ("src", "ref"):
        plane = getattr(bundle, role)
        plane.meta = SceneMeta(
            product_id=role.upper(), instrument="OHRC", mission="CH2", gsd_m=1.0,
            n_bands=1, wavelength_nm=None, array_shape=plane.array.shape, dtype="float32",
            corner_latlon=[], sub_solar_azimuth_deg=None, solar_incidence_deg=None,
            emission_deg=None, phase_deg=None, acquisition_utc=None,
            label_path=Path(tmp_path / f"{role}.xml"), raster_path=Path(tmp_path / f"{role}.img"))
    lunar = rasterio.crs.CRS.from_string("+proj=longlat +R=1737400 +no_defs")
    tif, sidecar = export_bundle(tmp_path / "registered.tif", bundle,
                                 ref_model=MapModel(), crs=lunar,
                                 provenance={"git_commit": "abc"})

    with rasterio.open(tif) as dataset:
        assert dataset.crs.to_dict()["R"] == 1737400
        assert dataset.transform.a == pytest.approx(0.1)
        assert dataset.transform.e == pytest.approx(-0.1)
        assert dataset.tags()["geometry"] == "affine"
        assert np.array_equal(dataset.read(1), bundle.ref.array)
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["source"]["product_id"] == "SRC"
    assert data["reference"]["product_id"] == "REF"
    assert data["geometry"]["matrix"] == matrix.tolist()
    assert data["provenance"]["git_commit"] == "abc"


# ---------------------------------------------------------------------------
# Audit 2026-09-26 C-07: a window away from the product origin must land where it is.
# Before: the GeoTIFF was placed at the PRODUCT's top-left (+5000/-6000 px = 57.8 km on
# a real TC window) and the GCP path was half a pixel off.
# ---------------------------------------------------------------------------
from chandralign.geometry.projection import MapModel as _RealMapModel  # noqa: E402

_ORIGIN = (5000, 6000)            # (row, col) of the window in the full product
_TC = dict(resolution_px_per_deg=4096.0, line_offset=12288.0, sample_offset=-86016.0,
           center_lat=0.0, center_lon=0.0, lines=12288, samples=12288)


def _windowed_bundle(tmp_path):
    matrix = np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], float)
    bundle = _bundle(TransformModel("affine", matrix=matrix))
    for role in ("src", "ref"):
        plane = getattr(bundle, role)
        plane.meta = SceneMeta(
            product_id=role.upper(), instrument="TC", mission="SELENE", gsd_m=7.4,
            n_bands=1, wavelength_nm=None, array_shape=plane.array.shape, dtype="float32",
            corner_latlon=[], sub_solar_azimuth_deg=None, solar_incidence_deg=None,
            emission_deg=None, phase_deg=None, acquisition_utc=None,
            label_path=Path(tmp_path / f"{role}.lbl"), raster_path=Path(tmp_path / f"{role}.img"))
    bundle.src.tile_origin = (0, 0)
    bundle.ref.tile_origin = _ORIGIN
    return bundle


def _expected(row, col):
    lat, lon = _RealMapModel(**_TC).pixel_to_latlon(np.array([row + _ORIGIN[0]]), np.array([col + _ORIGIN[1]]))
    return float(lat[0]), float(lon[0])


def test_geotiff_affine_places_a_window_where_it_is_in_the_product(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    bundle = _windowed_bundle(tmp_path)
    tif, _ = export_bundle(tmp_path / "w.tif", bundle, ref_model=_RealMapModel(**_TC),
                           crs=rasterio.crs.CRS.from_string("+proj=longlat +R=1737400 +no_defs"))
    with rasterio.open(tif) as ds:
        for row, col in [(0, 0), (39, 49), (20, 7)]:
            lon, lat = ds.transform * (col + 0.5, row + 0.5)       # centre of pixel (row, col)
            e_lat, e_lon = _expected(row, col)
            assert lat == pytest.approx(e_lat, abs=1e-9) and lon == pytest.approx(e_lon, abs=1e-9)


class _PointOnlyModel:
    """A ground model with no map projection (like a NAC or CH-2 product): GCP path."""
    def __init__(self):
        self._m = _RealMapModel(**_TC)

    def pixel_to_latlon(self, rows, cols, clip: bool = True):
        return self._m.pixel_to_latlon(rows, cols)


def test_geotiff_gcps_place_a_window_with_no_half_pixel_error(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_gcps
    bundle = _windowed_bundle(tmp_path)
    tif, _ = export_bundle(tmp_path / "g.tif", bundle, ref_model=_PointOnlyModel(),
                           crs=rasterio.crs.CRS.from_string("+proj=longlat +R=1737400 +no_defs"))
    with rasterio.open(tif) as ds:
        gcps, _ = ds.gcps
        t = from_gcps(gcps)
    res = 1 / _TC["resolution_px_per_deg"]
    for row, col in [(0, 0), (39, 49), (20, 7)]:
        lon, lat = t * (col + 0.5, row + 0.5)
        e_lat, e_lon = _expected(row, col)
        assert abs(lat - e_lat) < 0.01 * res and abs(lon - e_lon) < 0.01 * res


_TC_LABEL = Path(__file__).resolve().parents[1] / "data/raw/selene/tc/TCO_MAP_02_N03E021N00E024SC.lbl"


@pytest.mark.skipif(not _TC_LABEL.is_file(), reason="real SELENE TC product not downloaded")
def test_real_tc_window_geotiff_lands_on_the_labels_own_coordinates(tmp_path):
    """The audit's exact case: a real TC window at (5000, 6000) was placed 57.8 km away."""
    rasterio = pytest.importorskip("rasterio")
    from chandralign.geometry.projection import load_map_model
    from chandralign.io.pds_label import parse_label
    from chandralign.io.tiling import Window, read_tile
    meta = parse_label(_TC_LABEL)
    plane = read_tile(meta, Window(row=5000, col=6000, height=384, width=384))
    ident = TransformModel("affine", matrix=np.eye(3))
    delivered = MatchSet(np.zeros((0, 2)), np.zeros((0, 2)), np.ones(0, np.float32), "t", "same_modal_normal", "delivered")
    bundle = SimpleNamespace(result=SimpleNamespace(model=ident, metrics=Metrics()), delivered=delivered,
                             tps=None, parallax=None, src=plane, ref=plane)
    tif, _ = export_bundle(tmp_path / "tc.tif", bundle)
    truth = load_map_model(meta)
    with rasterio.open(tif) as ds:
        assert np.allclose(ds.read(1), plane.array)
        for row, col in [(0, 0), (383, 383), (192, 17)]:
            lon, lat = ds.transform * (col + 0.5, row + 0.5)
            e_lat, e_lon = truth.pixel_to_latlon(np.array([row + 5000]), np.array([col + 6000]))
            # < 0.01 TC pixel (1/4096 deg per pixel)
            assert abs(lat - e_lat[0]) < 0.01 / 4096 and abs(lon - e_lon[0]) < 0.01 / 4096
