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
