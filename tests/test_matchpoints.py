import json

import numpy as np
import pytest

from chandralign.contracts import MatchSet
from chandralign.product.matchpoints import FIELDS, export_bundle, read_csv, write_csv, write_geojson


class GroundModel:
    def pixel_to_latlon(self, rows, cols):
        return -80.0 + np.asarray(rows) / 1000, 20.0 + np.asarray(cols) / 1000


def sample():
    return MatchSet(
        np.array([[1.123456789012345, 2.5], [99.75, 50.125]], np.float64),
        np.array([[3.987654321098765, 4.25], [101.5, 52.875]], np.float64),
        np.array([0.12345679, 0.9876543], np.float32),
        "lightglue", "cross_modal", "TMC2->TC",
    )


def test_csv_round_trips_matchset_without_precision_loss(tmp_path):
    original = sample()
    mask = np.array([True, False])
    path = write_csv(tmp_path / "matches.csv", original, mask, src_model=GroundModel(),
                     ref_model=GroundModel(), image_shape=(100, 200), grid=2)
    restored, restored_mask = read_csv(path)

    assert np.array_equal(restored.src_pts, original.src_pts)
    assert np.array_equal(restored.ref_pts, original.ref_pts)
    assert np.array_equal(restored.confidence, original.confidence)
    assert (restored.method, restored.regime, restored.stage) == (
        original.method, original.regime, original.stage)
    assert np.array_equal(restored_mask, mask)
    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == list(FIELDS)


def test_geojson_has_lunar_crs_ground_geometry_and_properties(tmp_path):
    path = write_geojson(tmp_path / "matches.geojson", sample(), np.array([True, False]),
                         src_model=GroundModel(), ref_model=GroundModel(),
                         image_shape=(100, 200), grid=2)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["type"] == "FeatureCollection"
    assert data["crs"]["properties"]["name"] == "IAU_2015:30100"
    assert data["features"][0]["geometry"]["coordinates"] == pytest.approx(
        [20.0039876543211, -79.99575])
    assert data["features"][0]["properties"]["is_inlier"] is True
    assert data["features"][1]["properties"]["grid_cell"] == "1,0"


def test_geojson_refuses_pixel_coordinates_disguised_as_ground_coordinates(tmp_path):
    with pytest.raises(ValueError, match="ref_model"):
        write_geojson(tmp_path / "bad.geojson", sample(), np.ones(2, bool), ref_model=None)


def test_bundle_export_uses_delivered_not_evidence_matches(tmp_path):
    from types import SimpleNamespace

    delivered = sample()
    delivered.stage = "delivered"
    evidence = MatchSet(np.array([[999.0, 999.0]]), np.array([[999.0, 999.0]]),
                        np.ones(1, np.float32), "test", "cross_modal", "evidence")
    plane = SimpleNamespace(array=np.zeros((100, 200)), meta=None)
    bundle = SimpleNamespace(delivered=delivered,
                             result=SimpleNamespace(matches=evidence), src=plane, ref=plane)

    csv_path, geojson_path = export_bundle(tmp_path, bundle, src_model=GroundModel(),
                                           ref_model=GroundModel(), grid=2)
    restored, mask = read_csv(csv_path)

    assert np.array_equal(restored.src_pts, delivered.src_pts)
    assert np.array_equal(mask, np.ones(2, bool))
    assert len(json.loads(geojson_path.read_text(encoding="utf-8"))["features"]) == 2


def test_match_point_latlon_uses_the_window_origin(tmp_path):
    """Audit 2026-09-26 C-07: tile-local points were converted with the product model."""
    import csv
    from types import SimpleNamespace
    from chandralign.geometry.projection import MapModel
    tc = MapModel(resolution_px_per_deg=4096.0, line_offset=12288.0, sample_offset=-86016.0,
                  center_lat=0.0, center_lon=0.0, lines=12288, samples=12288)
    pts = np.array([[10.0, 20.0], [300.5, 40.25]])
    delivered = MatchSet(pts, pts + 1.0, np.ones(2, np.float32), "t", "same_modal_normal", "delivered")
    plane = lambda origin: SimpleNamespace(array=np.zeros((400, 400), np.float32), meta=None,
                                           tile_origin=origin)
    bundle = SimpleNamespace(delivered=delivered, result=SimpleNamespace(matches=delivered),
                             src=plane((0, 0)), ref=plane((5000, 6000)))
    csv_path, _ = export_bundle(tmp_path, bundle, src_model=tc, ref_model=tc, grid=2)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for row, (x, y) in zip(rows, pts + 1.0):
        lat, lon = tc.pixel_to_latlon(np.array([y + 5000]), np.array([x + 6000]))
        assert float(row["ref_lat"]) == pytest.approx(float(lat[0]), abs=1e-9)
        assert float(row["ref_lon"]) == pytest.approx(float(lon[0]), abs=1e-9)
