"""The product entry point runs the validated path (audit 2026-09-26 C-01).

Before: `chandralign register` registered the top-left 1024 px tile of each product with SIFT,
so the TMC-2 -> SELENE TC headline pair came back REJECTED (4 inliers) while
scripts/register_tmc2_tc.py registers it HIGH, 0.46 px.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _one(pattern: str, base: Path):
    return next(iter(sorted(base.rglob(pattern))), None) if base.exists() else None


TMC2 = _one("*_d_img_d18.xml", RAW / "ch2" / "tmc2")
IIRS = _one("ch2_iir_nci_20240523T1600301891_d_img_d18.xml", RAW / "ch2" / "iirs")   # the evidence scene
TC = RAW / "selene" / "tc" / "TCO_MAP_02_N03E021N00E024SC.lbl"
MI = RAW / "selene" / "mi" / "MI_MAP_03_N01E023N00E024SC.lbl"
WAC = _one("M106698280MC.XML", RAW / "lro" / "wac")
needs_tmc2_tc = pytest.mark.skipif(TMC2 is None or not TC.is_file(), reason="real TMC-2 / TC not downloaded")


@pytest.mark.skipif(TMC2 is None or not MI.is_file(), reason="real TMC-2 / MI not downloaded")
def test_a_pairing_without_a_validated_workflow_is_refused_not_guessed():
    from chandralign.workflows.products import UnsupportedPairing, register_products
    with pytest.raises(UnsupportedPairing, match="TMC2 -> MI"):
        register_products(TMC2, MI)


@pytest.mark.skipif(IIRS is None or WAC is None, reason="real IIRS / WAC not downloaded")
def test_the_refusal_names_the_research_evidence_that_exists():
    from chandralign.workflows.products import UnsupportedPairing, register_products
    with pytest.raises(UnsupportedPairing, match="register_iirs_wac.py"):
        register_products(IIRS, WAC)


@pytest.fixture(scope="module")
def tmc2_tc_run():
    from chandralign.workflows.products import register_products
    return register_products(TMC2, TC, windows=1)


@pytest.mark.slow
@needs_tmc2_tc
def test_the_headline_pair_registers_on_the_validated_path(tmc2_tc_run):
    w = tmc2_tc_run.windows[0]
    assert tmc2_tc_run.matcher_choice["chosen_by"] == "matching.routing.choose"
    assert w.bundle is not None, w.failure
    r = w.bundle.result
    assert r.confidence_tier in ("HIGH", "MEDIUM"), (r.confidence_tier, r.failure_modes)
    assert r.metrics.inlier_count > 1000 and r.metrics.spatial_coverage > 0.9
    assert all(r.gates.values())
    assert w.window["scale_status"] == "consistent"          # CHECK-05 ran on the composed transform
    assert tmc2_tc_run.overlap["overlap_km2"] > 1000


@pytest.mark.slow
@needs_tmc2_tc
def test_its_exports_land_on_the_references_own_map(tmc2_tc_run, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from chandralign.geometry.projection import load_map_model
    from chandralign.io.pds_label import parse_label
    from chandralign.product.run_export import write_run
    w = tmc2_tc_run.windows[0]
    out = tmp_path / "w1"
    write_run(out, w.bundle, manifest={"test": True}, src_model=w.src_model, ref_model=w.ref_model,
              heights_at=w.bundle.heights_at)
    tcm = load_map_model(parse_label(TC))
    r0, c0 = w.bundle.ref.tile_origin
    with rasterio.open(out / "registered.tif") as ds:
        assert ds.tags()["confidence_tier"] == w.bundle.result.confidence_tier
        for row, col in [(0, 0), (ds.height - 1, ds.width - 1)]:
            lon, lat = ds.transform * (col + 0.5, row + 0.5)
            e_lat, e_lon = tcm.pixel_to_latlon(np.array([row + r0]), np.array([col + c0]))
            assert abs(lat - e_lat[0]) < 1e-6 and abs(lon - e_lon[0]) < 1e-6
    # every exported point: the recorded TMC-2 product -> TC product transform takes its source
    # product position to (about) its reference position -- the frames are consistent end to end
    T = np.array(w.bundle.result.provenance["product_transform"]["matrix"])
    rows = list(csv.DictReader(open(out / "matches.csv", encoding="utf-8")))
    assert len(rows) >= 100
    src = np.array([[float(x["src_product_x"]), float(x["src_product_y"])] for x in rows])
    ref = np.array([[float(x["ref_product_x"]), float(x["ref_product_y"])] for x in rows])
    assert np.allclose(ref, [[float(x["ref_x"]) + c0, float(x["ref_y"]) + r0] for x in rows])
    d = np.hypot(*((np.c_[src, np.ones(len(src))] @ T.T)[:, :2] - ref).T)
    # the recorded affine; the delivered geometry adds terrain parallax on top. Measured on
    # 2026-09-26: p50 0.30, p95 0.70 TC px. A frame mix-up is tens to thousands of px.
    assert np.median(d) < 1.0 and np.percentile(d, 95) < 2.0
    record = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert record["geometry_used"] in ("parallax", "tps", "affine")


@pytest.mark.slow
@needs_tmc2_tc
def test_the_cli_registers_the_headline_pair(tmp_path):
    from chandralign.cli import main
    out = tmp_path / "run"
    assert main(["register", "--src", str(TMC2), "--ref", str(TC), "--windows", "1", "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["pairing"] == "TMC2 -> TC" and summary["accepted"] == 1
    result = json.loads((out / "window_01" / "result.json").read_text(encoding="utf-8"))
    assert result["confidence_tier"] in ("HIGH", "MEDIUM")
    prov = json.loads((out / "window_01" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["matcher"] == "eloftr" and prov["run_arguments"]["windows"] == 1
    assert {"registered.tif", "matches.csv", "matches.geojson", "failure-log.jsonl"} <= set(result["exports"])


MOSAIC = RAW / "lro" / "wac_mosaic" / "wac_mosaic_100m_clip.json"


@pytest.mark.slow
@pytest.mark.skipif(IIRS is None or not MOSAIC.is_file(), reason="real IIRS / WAC mosaic not downloaded")
def test_iirs_registers_to_the_wac_mosaic_and_lands_on_its_map(tmp_path):
    """IIRS is named in the problem statement; its validated pairing (xoftr 5/5 HIGH) is now a product."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_gcps
    from chandralign.product.run_export import write_run
    from chandralign.workflows import iirs_wac
    from chandralign.workflows.products import register_products
    run = register_products(IIRS, MOSAIC, windows=1)
    w = run.windows[0]
    assert w.bundle is not None, w.failure
    assert w.bundle.result.confidence_tier in ("HIGH", "MEDIUM") and w.bundle.result.provenance["matcher"] == "xoftr"
    out = tmp_path / "w1"
    write_run(out, w.bundle, manifest={"test": True}, src_model=w.src_model, ref_model=w.ref_model)
    _, _, _, geo = iirs_wac.load_mosaic()
    ox, oy = w.bundle.ref_to_product[0, 2], w.bundle.ref_to_product[1, 2]
    with rasterio.open(out / "registered.tif") as ds:
        t = from_gcps(ds.gcps[0])
        for row, col in [(0, 0), (ds.height - 1, ds.width - 1)]:
            lon, lat = t * (col + 0.5, row + 0.5)
            e_lat, e_lon = geo.pixel_to_latlon(np.array([row + oy]), np.array([col + ox]))
            assert abs(lat - e_lat[0]) < 0.01 * geo.dpp and abs(lon - e_lon[0]) < 0.01 * geo.dpp
