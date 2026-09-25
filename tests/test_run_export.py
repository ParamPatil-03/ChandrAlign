"""One run folder writer for the CLI and the API (audit 2026-09-26 I-12, I-14).

Before: the CLI and the API each carried their own copy of the export sequence. A REJECTED
result with no geometry crashed both (so its rejection reasons were lost), a REJECTED result
WITH a model shipped an unflagged `registered.tif`, a NAC reference crashed the exporters, and
the API never wrote the failure log.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from chandralign import synth
from chandralign.cli import _MockGroundModel
from chandralign.pipeline import register_bundle
from chandralign.product.run_export import write_run


def _pair(seed=7):
    return synth.make_pair(out_shape=(256, 256), shift=(3.4, -2.2), seed=seed, n_craters=35, shadows=False)[:2]


@pytest.fixture(scope="module")
def accepted():
    src, ref = _pair()
    bundle = register_bundle(src, ref, matcher="sift")
    assert bundle.result.confidence_tier != "REJECTED"
    return bundle


@pytest.fixture(scope="module")
def rejected_without_geometry():
    src, ref = _pair()
    ref.array = np.random.default_rng(0).random(ref.array.shape).astype(np.float32)
    bundle = register_bundle(src, ref, matcher="sift")
    assert bundle.result.confidence_tier == "REJECTED" and bundle.result.model is None
    return bundle


def _models(bundle):
    return dict(src_model=_MockGroundModel(bundle.src.gsd_m), ref_model=_MockGroundModel(bundle.ref.gsd_m))


def _result(out):
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def test_a_rejected_result_without_geometry_is_written_not_crashed(tmp_path, rejected_without_geometry):
    out = tmp_path / "run"
    write_run(out, rejected_without_geometry, manifest={"synthetic": True}, **_models(rejected_without_geometry))
    result = _result(out)
    assert result["confidence_tier"] == "REJECTED"
    assert result["failure_modes"]                         # the reasons survive (rule H3)
    assert not (out / "registered.tif").exists()
    assert "REJECTED" in result["exports_skipped"]["registered.tif"]
    log = json.loads((out / "failure-log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert log["confidence_tier"] == "REJECTED"


def test_a_rejected_result_with_a_model_issues_no_registered_product(tmp_path, accepted):
    bundle = replace(accepted, result=replace(accepted.result, confidence_tier="REJECTED", failure_modes=[12]))
    out = tmp_path / "run"
    write_run(out, bundle, manifest={"synthetic": True}, **_models(bundle))
    assert not (out / "registered.tif").exists()
    assert "REJECTED" in _result(out)["exports_skipped"]["registered.tif"]


def test_an_accepted_geotiff_carries_its_tier_and_gates(tmp_path, accepted):
    rasterio = pytest.importorskip("rasterio")
    out = tmp_path / "run"
    write_run(out, accepted, manifest={"synthetic": True}, **_models(accepted))
    with rasterio.open(out / "registered.tif") as ds:
        tags = ds.tags()
    assert tags["confidence_tier"] == accepted.result.confidence_tier
    assert json.loads(tags["failure_modes"]) == list(accepted.result.failure_modes)
    assert json.loads(tags["gates_passed"]) == {k: bool(v) for k, v in accepted.result.gates.items()}
    sidecar = json.loads((out / "registered.json").read_text(encoding="utf-8"))
    assert sidecar["confidence_tier"] == accepted.result.confidence_tier


def test_a_reference_with_no_ground_model_gets_pixel_only_exports(tmp_path, accepted):
    """A NAC reference has no map projection: the run must still be written, honestly."""
    from chandralign.geometry.projection import GeolocationUnavailable

    class NoGround:
        def pixel_to_latlon(self, rows, cols, clip=True):
            raise GeolocationUnavailable("M102014464RC: no IMAGE_MAP_PROJECTION")

    out = tmp_path / "run"
    write_run(out, accepted, manifest={"synthetic": True},
              src_model=_MockGroundModel(accepted.src.gsd_m), ref_model=NoGround())
    result = _result(out)
    assert (out / "matches.csv").exists()                   # pixel coordinates are still exact
    for name in ("matches.geojson", "registered.tif"):
        assert not (out / name).exists()
        assert "no ground model" in result["exports_skipped"][name]


def test_the_result_record_lists_what_was_written(tmp_path, accepted):
    out = tmp_path / "run"
    write_run(out, accepted, manifest={"synthetic": True}, **_models(accepted))
    written = set(_result(out)["exports"])
    assert {"result.json", "provenance.json", "failure-log.jsonl", "matches.csv"} <= written
    assert written <= {p.name for p in out.iterdir()}


def test_a_run_folder_carries_real_quicklooks_and_its_report(tmp_path, accepted):
    """Audit C-05 / M-12: the web UI's swipe compares REAL reference and registered imagery, and
    its report download is the run's own report."""
    import cv2
    out = tmp_path / "run"
    rec = write_run(out, accepted, manifest={"synthetic": True}, **_models(accepted))
    assert {"reference.png", "registered.png", "report.html"} <= set(rec["exports"])
    ref = cv2.imread(str(out / "reference.png"), cv2.IMREAD_GRAYSCALE)
    reg = cv2.imread(str(out / "registered.png"), cv2.IMREAD_GRAYSCALE)
    assert ref.shape == reg.shape == np.asarray(accepted.ref.array).shape
    assert "Reference / registered" in (out / "report.html").read_text(encoding="utf-8")


def test_a_rejected_run_without_geometry_has_no_registered_quicklook(tmp_path, rejected_without_geometry):
    out = tmp_path / "run"
    rec = write_run(out, rejected_without_geometry, manifest={"synthetic": True}, **_models(rejected_without_geometry))
    assert "registered.png" not in rec["exports"] and "registered.png" in rec["exports_skipped"]


def test_the_evidence_matches_are_exported_with_real_flags_and_confidence(tmp_path, accepted):
    """Audit M-11: only the delivered points were exported, all is_inlier=True, confidence 1.0."""
    from chandralign.product.matchpoints import read_csv
    out = tmp_path / "run"
    rec = write_run(out, accepted, manifest={"synthetic": True}, **_models(accepted))
    assert "evidence_matches.csv" in rec["exports"]
    ms, mask = read_csv(out / "evidence_matches.csv")
    r = accepted.result
    assert len(ms.src_pts) == len(r.matches.src_pts)
    assert np.array_equal(mask, r.inlier_mask) and (~mask).any()          # outliers are there, flagged
    assert np.allclose(ms.confidence, r.matches.confidence)
