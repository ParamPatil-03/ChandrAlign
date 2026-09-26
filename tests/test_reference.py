"""DATA-06/07/08/09: loading LRO NAC/WAC and SELENE TC/MI references.

One loader for all four cameras, because the pipeline must not branch on which
reference it was given. Every number here was measured on the real products we
hold, including three defects that only real data exposed.
"""
from pathlib import Path

import numpy as np
import pytest

from chandralign.io.pds_label import parse_label, read_special_values
from chandralign.io.pds_raster import Window
from chandralign.io.reference import (
    DARK_REFLECTANCE,
    NotAReferenceError,
    find_references,
    load_all,
    load_reference,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "tests" / "fixtures" / "labels"

# This module asserts on all four reference cameras, so it needs all four on
# disk. Guarding on `lro/` alone was too coarse: a checkout holding only NAC
# (the products anyone can fetch without a login) passed the guard and then
# failed four tests on the missing WAC/TC/MI. A partial dataset should skip,
# not go red -- otherwise the first thing a new machine sees is a broken suite.
_REFERENCE_GLOBS = {
    "NAC": "lro/nac/*/*.XML",
    "WAC": "lro/wac/*/*.XML",
    "TC": "selene/tc/*.lbl",
    "MI": "selene/mi/*.lbl",
}
_MISSING = sorted(name for name, pattern in _REFERENCE_GLOBS.items()
                  if not any(RAW.glob(pattern)))
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"reference products not downloaded: {', '.join(_MISSING)} (data/raw is gitignored)")


@pytest.fixture(scope="module")
def refs():
    return {r.meta.product_id: r for r in load_all(RAW)}


def by_instrument(refs, name):
    return [r for r in refs.values() if r.instrument == name]


# ----------------------------------------------------------------------------- all four cameras

def test_all_four_reference_cameras_load(refs):
    found = {r.instrument for r in refs.values()}
    assert found == {"NAC", "WAC", "TC", "MI"}


def test_every_reference_has_a_usable_scene_meta(refs):
    for r in refs.values():
        assert r.meta.array_shape[0] > 0 and r.meta.array_shape[1] > 0
        assert r.meta.n_bands >= 1
        assert r.meta.gsd_m > 0
        assert r.meta.raster_path.is_file()


def test_a_chandrayaan_product_is_refused(refs):
    ch2 = next(RAW.glob("ch2/*/products/*/data/calibrated/*/*_d_img_d18.xml"))
    with pytest.raises(NotAReferenceError, match="not one of"):
        load_reference(ch2)


def test_sidecars_are_not_offered_as_references():
    names = [p.name for p in find_references(RAW)]
    assert names, "no reference labels found"
    assert not any("_PYR" in n.upper() for n in names)


# ----------------------------------------------------------------------------- DATA-07: WAC

def test_wac_loads_and_pairs_with_iirs(refs):
    """DATA-07 acceptance: a WAC product loads, and an IIRS <-> WAC pair can be built."""
    from chandralign.geometry.footprint import check_overlap

    wacs = by_instrument(refs, "WAC")
    assert wacs, "no WAC products"
    iirs = parse_label(next(RAW.glob("ch2/iirs/products/ch2_iir_nci_20240523T1600301891*/data/calibrated/*/*_d_img_d18.xml")))  # pinned: 4 more IIRS products were added to data/raw on 2026-09-26; every committed IIRS result uses this one
    overlaps = [check_overlap(iirs, w.meta) for w in wacs]
    assert any(o.ok for o in overlaps), [o.reason for o in overlaps]
    best = max(overlaps, key=lambda o: o.overlap_km2)
    assert best.overlap_km2 > 1000.0


def test_wac_has_no_corners_so_its_outline_comes_from_ode(refs):
    """Recorded because it is a real difference between the references, not a bug."""
    for w in by_instrument(refs, "WAC"):
        assert w.meta.label_fields_verified["corner_latlon"] is False
        assert w.footprint is not None and w.footprint.source == "ode_catalogue"


def test_wac_special_constants_are_hex_bit_patterns(refs):
    """The defect that made this worth doing: 0xFF7FFFFB is NOT the integer 4287102971.

    Reinterpreted as the array's float32 it is -3.4028227e+38, which is exactly the
    fill value sitting in the pixels. A reader that does not reinterpret the bits
    masks nothing at all.
    """
    wac = by_instrument(refs, "WAC")[0]
    special = read_special_values(wac.meta.label_path)
    assert special.declared
    assert special.nodata, "no no-data constants parsed from the WAC label"
    assert min(special.nodata) < -3.4e38
    assert np.array([0xFF7FFFFB], dtype=np.uint32).view(np.float32)[0] in [
        pytest.approx(v, rel=1e-6) for v in special.nodata]


def test_the_wac_fill_value_is_actually_masked(refs):
    wac = next(r for r in by_instrument(refs, "WAC") if r.meta.product_id == "M107908070MC")
    values, mask = wac.raw(Window(100, 300, 200, 400))
    assert (~mask).any(), "expected some fill pixels in this window"
    assert np.all(values[mask] > -1e30)          # nothing extreme survives the mask


# ----------------------------------------------------------------------------- night side

def test_two_of_our_wac_products_are_night_side(refs):
    """A black reference is worse than none: it matches nothing and explains nothing."""
    lit = {r.meta.product_id: r.sunlit() for r in by_instrument(refs, "WAC")}
    dark = {k: v for k, v in lit.items() if not v["lit"]}
    bright = {k: v for k, v in lit.items() if v["lit"]}
    assert len(dark) == 2 and len(bright) == 2, lit
    for v in dark.values():
        assert v["mean_raw"] < DARK_REFLECTANCE
        assert v["incidence_deg"] > 90.0          # the sun is below the horizon
    for v in bright.values():
        assert v["mean_raw"] > 0.02
        assert v["incidence_deg"] < 45.0


def test_brightness_is_measured_on_raw_values_not_the_stretched_plane(refs):
    """The bug this guards: a min-max stretch made the night-side products look BRIGHTER.

    read() normalises each window to 0..1, which is right for matching and wrong for
    asking how bright something is.
    """
    dark = next(r for r in by_instrument(refs, "WAC") if not r.sunlit()["lit"])
    window = Window(100, 300, 200, 400)
    raw_values, mask = dark.raw(window)
    stretched = dark.read(window)
    assert float(raw_values[mask].mean()) < DARK_REFLECTANCE
    assert float(stretched.array[stretched.valid_mask].mean()) > 0.2   # the stretch lies


# ----------------------------------------------------------------------------- DATA-09: MI

def test_mi_loads_with_its_band_information(refs):
    """DATA-09 acceptance: an MI product loads into SceneMeta WITH its band information."""
    mi = by_instrument(refs, "MI")
    assert mi, "no MI product"
    mi = mi[0]
    assert mi.meta.n_bands == 9
    assert mi.band_names == ("MV1", "MV2", "MV3", "MV4", "MV5", "MN1", "MN2", "MN3", "MN4")
    assert mi.band_wavelengths_nm == (414.0, 749.0, 901.0, 950.0, 1001.0,
                                      1000.0, 1049.0, 1248.0, 1548.0)
    assert len(mi.band_names) == len(mi.band_wavelengths_nm) == mi.meta.n_bands


def test_mi_pixels_read_past_the_elevation_backplane(refs):
    """The MI file holds a 16 MB float32 altitude plane BEFORE the image."""
    mi = by_instrument(refs, "MI")[0]
    from chandralign.io.pds_label import read_array_layout
    layout = read_array_layout(mi.meta.label_path)
    assert layout.offset_bytes == 2048 * 2048 * 4       # exactly the backplane's size
    assert layout.shape == (9, 2048, 2048)
    values, mask = mi.raw(Window(900, 900, 128, 128), band=0)
    assert mask.all()
    assert values.min() > 0 and values.max() > values.min()


def test_mi_resolution_is_read_from_the_label_not_the_nominal(refs):
    """14.806 m/pixel on this tile, where the registry's nominal figure is 20 m."""
    mi = by_instrument(refs, "MI")[0]
    assert mi.meta.gsd_m == pytest.approx(14.806, abs=0.001)


def test_mi_corners_come_from_its_projection_bounds(refs):
    """MI states no explicit corners, only the projection's own bounds -- still label values."""
    mi = by_instrument(refs, "MI")[0]
    assert mi.meta.label_fields_verified["corner_latlon"] is True
    assert mi.footprint is not None and mi.footprint.source == "label"
    west, south, east, north = mi.footprint.polygon.bounds
    assert (round(south), round(north), round(west), round(east)) == (0, 1, 23, 24)


# ----------------------------------------------------------------------------- reference independence

def test_selene_references_are_flagged_as_not_independent(refs):
    """Never judge a SELENE registration with SELENE's own cameras."""
    for r in refs.values():
        assert r.independent_of_selene == (r.meta.mission != "SELENE")
    assert all(not r.independent_of_selene for r in by_instrument(refs, "TC"))
    assert all(not r.independent_of_selene for r in by_instrument(refs, "MI"))
    assert all(r.independent_of_selene for r in by_instrument(refs, "NAC"))
