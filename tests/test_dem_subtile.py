"""A DEM sub-tile cut over byte ranges is the same DEM, only smaller.

The SLDEM2015 server gives ~0.3 MB/s; the fresh IIRS scenes need four 1.4 GB tiles but only a
few MB of them. A sub-tile keeps the product's grid (shifted projection offsets), so dem_patch
reads it unchanged, stitches it with its neighbours, and refuses any box it does not cover.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SLDEM = Path(__file__).resolve().parents[1] / "data" / "raw" / "dem" / "sldem2015"
NORTH = SLDEM / "sldem2015_512_00n_30n_000_045_float.lbl"
SOUTH = SLDEM / "sldem2015_512_30s_00s_000_045_float.lbl"
pytestmark = pytest.mark.skipif(not (NORTH.with_suffix(".img").is_file() and SOUTH.with_suffix(".img").is_file()),
                                reason="SLDEM2015 tiles not downloaded")


def _local(label: Path):
    img = label.with_suffix(".img")

    def read(start: int, end: int) -> bytes:
        with img.open("rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)
    return read


def test_a_subtile_gives_exactly_the_full_tiles_heights(tmp_path):
    from chandralign.io.dem import dem_patch
    from chandralign.io.dem_subtile import cut_subtile
    sub = cut_subtile(NORTH, (10.0, 11.0, 23.0, 24.0), tmp_path, _local(NORTH))
    box = (10.2, 10.8, 23.2, 23.8)
    a, b = dem_patch([sub], box), dem_patch([NORTH], box)
    assert np.array_equal(a.lat, b.lat) and np.array_equal(a.lon, b.lon)
    assert np.array_equal(a.heights_m, b.heights_m, equal_nan=True)


def test_subtiles_stitch_across_a_tile_edge(tmp_path):
    from chandralign.io.dem import dem_patch
    from chandralign.io.dem_subtile import cut_subtile
    subs = [cut_subtile(t, (-0.5, 0.5, 23.0, 24.0), tmp_path, _local(t)) for t in (NORTH, SOUTH)]
    box = (-0.3, 0.3, 23.2, 23.8)
    a, b = dem_patch(subs, box), dem_patch([NORTH, SOUTH], box)
    assert np.array_equal(a.heights_m, b.heights_m, equal_nan=True)


def test_a_box_outside_the_subtile_is_refused_not_filled(tmp_path):
    from chandralign.io.dem import DemError, dem_patch
    from chandralign.io.dem_subtile import cut_subtile
    sub = cut_subtile(NORTH, (10.0, 11.0, 23.0, 24.0), tmp_path, _local(NORTH))
    with pytest.raises(DemError, match="cover"):
        dem_patch([sub], (10.5, 11.5, 23.2, 23.8))
