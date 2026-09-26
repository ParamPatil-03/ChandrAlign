"""The IIRS -> WAC terrain filter where the DEM ends (fresh-scene run, 2026-09-26).

SLDEM2015 covers +/-60 deg only; an IIRS scene at 60-66 N crashed the window with DemError
instead of registering without the terrain filter. It now registers without it and says so.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SLDEM = Path(__file__).resolve().parents[1] / "data" / "raw" / "dem" / "sldem2015"
needs_sldem = pytest.mark.skipif(not (SLDEM / "sldem2015_512_00n_30n_000_045_float.lbl").is_file(),
                                 reason="SLDEM2015 tiles not downloaded")


@needs_sldem
def test_a_window_the_dem_does_not_cover_runs_without_it_and_records_why():
    from chandralign.io.dem import find_tiles
    from chandralign.workflows.iirs_wac import window_dem
    out: dict = {}
    assert window_dem(find_tiles(SLDEM), 62.0, -6.8, out) is None
    assert out["dem_status"].startswith("none: ") and "cover" in out["dem_status"]


@needs_sldem
def test_a_covered_window_still_gets_its_dem():
    from chandralign.io.dem import find_tiles
    from chandralign.workflows.iirs_wac import window_dem
    out: dict = {}
    dem = window_dem(find_tiles(SLDEM), 10.0, 23.5, out)
    assert dem is not None and dem.heights_m.size > 0 and out["dem_status"] == "sldem2015_lola_plus_selene_tc"
