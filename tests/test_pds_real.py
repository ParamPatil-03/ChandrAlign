"""The anti-stub gate (DATA-01, PLAN.md section 2.2, failure mode #15).

A parser that loads pixels but does not genuinely read the label FAILS here. Every
expected value below was copied by hand from the real ISRO labels in
tests/fixtures/labels/, so a parser returning defaults cannot pass.
"""
from pathlib import Path

import pytest

from chandralign.io import pds_label
from chandralign.io.pds_label import PdsParseError, parse_pds4, read_array_layout, read_corner_sets

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
OHRC = LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml"
TMC2 = LABELS / "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml"
IIRS = LABELS / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml"

# Hand-copied from the labels. (lines, samples), dtype, gsd, sun azimuth, incidence, start time.
EXPECTED = {
    OHRC: dict(instrument="OHRC", shape=(79796, 12000), dtype="|u1", n_bands=1, gsd=0.30,
               azimuth=269.823288, incidence=82.733560, utc="2024-03-30T00:35:08.5365Z",
               upper_left=(0.372032, 23.472939)),
    TMC2: dict(instrument="TMC2", shape=(160269, 4000), dtype="<u2", n_bands=1, gsd=4.41,
               azimuth=104.269696, incidence=45.993599, utc="2025-02-07T11:02:03.9417Z",
               upper_left=(-1.629410, 23.703472)),
    IIRS: dict(instrument="IIRS", shape=(13101, 250), dtype="<f4", n_bands=256, gsd=97.15,
               azimuth=285.611693, incidence=32.682976, utc="2024-05-23T16:00:30.1891Z",
               upper_left=(8.195483, 23.771330)),
}


@pytest.mark.parametrize("label", EXPECTED, ids=lambda p: p.name[:7])
def test_parser_reads_real_label_fields(label):
    meta = parse_pds4(label)
    exp = EXPECTED[label]

    assert meta.instrument == exp["instrument"]
    assert meta.mission == "CH2"
    assert meta.array_shape == exp["shape"]
    assert meta.dtype == exp["dtype"]
    assert meta.n_bands == exp["n_bands"]
    assert meta.gsd_m == pytest.approx(exp["gsd"])
    assert meta.sub_solar_azimuth_deg == pytest.approx(exp["azimuth"])
    assert meta.solar_incidence_deg == pytest.approx(exp["incidence"])
    assert meta.acquisition_utc == exp["utc"]
    assert len(meta.corner_latlon) == 4
    assert meta.corner_latlon[0] == pytest.approx(exp["upper_left"])
    assert meta.array_shape[0] > 10_000          # a real strip, not a thumbnail


@pytest.mark.parametrize("label", EXPECTED, ids=lambda p: p.name[:7])
def test_verified_fields_are_honest(label):
    meta = parse_pds4(label)
    verified = meta.label_fields_verified
    assert sum(verified.values()) >= 6, verified
    for field in ("array_shape", "dtype", "gsd_m", "corner_latlon",
                  "solar_incidence_deg", "sub_solar_azimuth_deg", "acquisition_utc"):
        assert verified[field] is True, field
    # Not in any CH-2 label: must be reported as NOT from the label, and left empty.
    for field in ("emission_deg", "phase_deg", "wavelength_nm"):
        assert verified[field] is False, field
    assert meta.emission_deg is None and meta.phase_deg is None


def test_plan_section_2_2_example():
    """The exact assertion PLAN.md section 2.2 gives for OHRC."""
    meta = parse_pds4(OHRC)
    assert meta.label_fields_verified["gsd_m"] is True
    assert meta.label_fields_verified["corner_latlon"] is True
    assert meta.gsd_m == pytest.approx(0.25, rel=0.2)
    assert meta.array_shape[0] > 10_000


def test_sun_angles_are_self_consistent():
    """Incidence is measured from overhead, so incidence + elevation = 90 in every label."""
    from lxml import etree
    for label in EXPECTED:
        tree = etree.parse(str(label))
        elevation = float(tree.xpath("string(//*[local-name()='sun_elevation'])"))
        assert parse_pds4(label).solar_incidence_deg + elevation == pytest.approx(90.0, abs=1e-5)


def test_array_layout_matches_label():
    layout = read_array_layout(IIRS)
    assert layout.axis_names == ("BAND", "LINE", "SAMPLE")
    assert layout.shape == (256, 13101, 250)
    assert layout.offset_bytes == 0
    assert layout.file_size_bytes == 3_353_856_000
    assert layout.md5 == "54395aed1efbce4d0a4a56a45e109714"
    # the declared size is exactly what the axes and pixel type imply
    import numpy as np
    for label in EXPECTED:
        L = read_array_layout(label)
        assert L.file_size_bytes == L.offset_bytes + np.prod(L.shape) * np.dtype(L.dtype).itemsize


def test_lxml_fallback_gives_identical_layout(monkeypatch):
    """If pds4_tools rejects a label (risk R12), the lxml path must give the same answer."""
    primary = read_array_layout(TMC2)
    assert primary.parsed_with == "pds4_tools"

    import pds4_tools

    def refuse(*args, **kwargs):
        raise RuntimeError("simulated pds4_tools rejection")

    monkeypatch.setattr(pds4_tools, "read", refuse)
    pds_label.clear_layout_cache()          # otherwise the cached pds4_tools result is returned
    fallback = read_array_layout(TMC2)
    pds_label.clear_layout_cache()
    assert fallback.parsed_with == "lxml"
    for attr in ("raster_path", "offset_bytes", "dtype", "axis_names", "shape", "file_size_bytes", "md5"):
        assert getattr(fallback, attr) == getattr(primary, attr), attr


def test_default_corners_are_independent_of_references():
    """Refined corners were adjusted against SELENE; the default must not use them."""
    sets = read_corner_sets(TMC2)
    assert set(sets) == {"system", "refined"}
    assert sets["system"] != sets["refined"]
    assert parse_pds4(TMC2).corner_latlon == sets["system"]
    assert parse_pds4(TMC2, corners="refined").corner_latlon == sets["refined"]


def test_truncated_label_raises(tmp_path):
    text = OHRC.read_text(encoding="utf-8")
    broken = tmp_path / OHRC.name
    broken.write_text(text[: len(text) // 2], encoding="utf-8")
    with pytest.raises(PdsParseError):
        parse_pds4(broken)


def test_label_without_array_description_raises(tmp_path):
    from lxml import etree
    tree = etree.parse(str(OHRC))
    for arr in tree.xpath("//*[local-name()='Array_2D_Image']"):
        arr.getparent().remove(arr)
    broken = tmp_path / OHRC.name
    tree.write(str(broken))
    with pytest.raises(PdsParseError):
        parse_pds4(broken)


def test_reader_never_uses_generic_image_loaders():
    """Failure mode #15: no silent fallback to cv2.imread / PIL / generic TIFF loading."""
    source = Path(pds_label.__file__).read_text(encoding="utf-8")
    for forbidden in ("cv2", "imread", "PIL", "imageio", "tifffile"):
        assert forbidden not in source, forbidden


REAL_PRODUCTS = sorted((ROOT / "data" / "raw" / "ch2").glob("*/products/*/data/calibrated/*/*_d_img_d18.xml"))


@pytest.mark.skipif(not REAL_PRODUCTS, reason="real products not downloaded (data/raw is gitignored)")
@pytest.mark.parametrize("label", REAL_PRODUCTS, ids=lambda p: p.name[:7])
def test_real_pixel_file_matches_label(label):
    """On machines holding the data: the pixel file the label names exists at the declared size."""
    meta = parse_pds4(label)
    layout = read_array_layout(label)
    assert meta.raster_path.exists()
    assert meta.raster_path.stat().st_size == layout.file_size_bytes


def test_layout_cache_notices_an_edited_label(tmp_path):
    """The cache is keyed on the label's size and modification time: an edit is re-read."""
    import os, shutil
    copy = tmp_path / OHRC.name
    shutil.copy(OHRC, copy)
    first = read_array_layout(copy)
    assert read_array_layout(copy) is first                       # served from the cache
    text = copy.read_text(encoding="utf-8").replace("<elements>12000</elements>", "<elements>11999</elements>")
    copy.write_text(text, encoding="utf-8")
    os.utime(copy, ns=(first_ns := copy.stat().st_mtime_ns + 10**9, first_ns))
    assert read_array_layout(copy).shape == (79796, 11999)

