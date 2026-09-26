"""Cutting the LROC WAC global-mosaic reference for any IIRS scene (IIRS generality, limit 2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

HELD = Path(__file__).resolve().parents[1] / "data/raw/lro/wac_mosaic/wac_mosaic_100m_clip.json"


@pytest.mark.network
@pytest.mark.skipif(not HELD.is_file(), reason="the committed evidence's mosaic clip is not held")
def test_the_fetcher_reproduces_the_evidence_clip_byte_for_byte(tmp_path):
    """Range-reading only the needed columns of each strip must give exactly the clip the
    committed IIRS -> WAC evidence used: every pixel, the grid, and the sha256."""
    from chandralign.io import wac_mosaic as W
    hm = json.loads(HELD.read_text(encoding="utf-8"))
    la1, lo0 = hm["pixel_centre_lat_of_row0"], hm["pixel_centre_lon_of_col0"]
    out = W.fetch_clip(la1, la1 - (hm["shape"][0] - 1) * W.DEG_PER_PX, lo0,
                       lo0 + (hm["shape"][1] - 1) * W.DEG_PER_PX, tmp_path / "clip.json")
    m = json.loads(out.read_text(encoding="utf-8"))
    assert (m["row0"], m["col0"], m["shape"], m["sha256"]) == (hm["row0"], hm["col0"], hm["shape"], hm["sha256"])
    assert np.array_equal(np.load(out.with_suffix(".npy")), np.load(HELD.with_suffix(".npy")))


def test_a_clip_json_is_recognised_as_a_reference(tmp_path):
    from chandralign.workflows.products import _is_mosaic_clip
    good = tmp_path / "c.json"; good.write_text(json.dumps({"deg_per_px": 0.0033}), encoding="utf-8")
    other = tmp_path / "o.json"; other.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert _is_mosaic_clip(good) and not _is_mosaic_clip(other) and not _is_mosaic_clip(tmp_path / "x.xml")


def _fake_mosaic(monkeypatch):
    """An offline mosaic: one byte per pixel, each pixel's value = its column mod 251."""
    from chandralign.io import wac_mosaic as W
    monkeypatch.setattr(W, "strip_layout", lambda s, url=None: (np.arange(W.HEIGHT, dtype=np.int64) * W.WIDTH, 1))
    monkeypatch.setattr(W, "_get", lambda s, url, a, b: bytes(((np.arange(a, b + 1) % W.WIDTH) % 251).astype(np.uint8)))
    return W


def test_a_scene_in_0_360_longitudes_gets_the_same_pixels_on_its_own_meridian(monkeypatch, tmp_path):
    """Chandrayaan-2 labels give 0..360 E; the mosaic is -180..180 E. A far-side scene at 206 E
    (= 154 W) used to give negative clip dimensions (fresh-scene run, 2026-09-26)."""
    W = _fake_mosaic(monkeypatch)
    east = json.loads(W.fetch_clip(-40.0, -40.01, 205.1, 206.7, tmp_path / "e.json").read_text(encoding="utf-8"))
    west = json.loads(W.fetch_clip(-40.0, -40.01, -154.9, -153.3, tmp_path / "w.json").read_text(encoding="utf-8"))
    assert east["col0"] == west["col0"] and east["shape"] == west["shape"]
    assert np.array_equal(np.load(tmp_path / "e.npy"), np.load(tmp_path / "w.npy"))
    # georeferenced in the mosaic's -180..180, as the IIRS corner model gives the scene
    assert east["pixel_centre_lon_of_col0"] == west["pixel_centre_lon_of_col0"]
    assert -154.9 <= east["pixel_centre_lon_of_col0"] < -154.9 + W.DEG_PER_PX


def test_a_box_across_the_mosaic_edge_is_refused(monkeypatch, tmp_path):
    W = _fake_mosaic(monkeypatch)
    with pytest.raises(NotImplementedError, match="180 deg edge"):
        W.fetch_clip(-40.0, -40.01, 179.5, 180.5, tmp_path / "x.json")
