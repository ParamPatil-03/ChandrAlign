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
