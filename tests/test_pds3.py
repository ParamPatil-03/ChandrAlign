"""DATA-02 / DATA-08: PDS3 (SELENE) label reading, plus windowed pixels through the same reader.

Real SELENE TC labels are committed as fixtures; expected values are hand-copied
from them. Synthetic products cover every ^IMAGE pointer form, byte order, and
multi-band interleave. The real tiles are checked against JAXA's own statistics.
"""
from pathlib import Path

import numpy as np
import pytest

from chandralign.io.pds_label import (
    PdsParseError,
    is_pds3,
    parse_label,
    parse_pds3,
    read_array_layout,
    read_pds3_image_info,
)
from chandralign.io.pds_raster import Window, read_raster

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "labels"
TC_NORTH = FIXTURES / "TCO_MAP_02_N03E021N00E024SC.lbl"
TC_SOUTH = FIXTURES / "TCO_MAP_02_N00E021S03E024SC.lbl"


# ----------------------------------------------------------------------------- real labels

def test_real_selene_label_fields():
    meta = parse_pds3(TC_NORTH)
    assert meta.product_id == "TCO_MAP_02_N03E021N00E024SC"
    assert meta.instrument == "TC" and meta.mission == "SELENE"
    assert meta.array_shape == (12288, 12288)
    assert meta.dtype == ">u2"                                  # MSB_UNSIGNED_INTEGER, 16 bit
    assert meta.gsd_m == pytest.approx(7.4031617246699)         # MAP_SCALE, km -> m
    assert meta.corner_latlon == [(3.0, 21.0), (3.0, 23.999756), (0.000244, 23.999756), (0.000244, 21.0)]
    assert meta.raster_path.name == "TCO_MAP_02_N03E021N00E024SC.img"


def test_real_selene_verification_is_honest():
    verified = parse_pds3(TC_NORTH).label_fields_verified
    assert sum(verified.values()) >= 6
    for f in ("product_id", "instrument", "mission", "array_shape", "dtype", "gsd_m", "corner_latlon"):
        assert verified[f] is True, f
    # A mosaic: START_TIME = UNK and no single sun position. Must NOT be claimed.
    for f in ("acquisition_utc", "sub_solar_azimuth_deg", "solar_incidence_deg", "emission_deg", "phase_deg"):
        assert verified[f] is False, f
    meta = parse_pds3(TC_NORTH)
    assert meta.acquisition_utc is None and meta.solar_incidence_deg is None


def test_real_image_pointer_and_value_info():
    layout = read_array_layout(TC_NORTH)
    assert layout.offset_bytes == 0 and layout.parsed_with == "pvl"   # ("file", 1 <BYTES>) -> byte 0
    info = read_pds3_image_info(TC_NORTH)
    assert info["dummy"] == 0 and info["valid_minimum"] == 2 and info["valid_maximum"] == 32766
    assert info["scaling_factor"] == pytest.approx(0.0183)
    assert info["minimum"] == 124 and info["maximum"] == 651


def test_format_detection_is_by_content():
    assert is_pds3(TC_NORTH)
    assert not is_pds3(FIXTURES / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
    assert parse_label(TC_NORTH).instrument == "TC"
    assert parse_label(FIXTURES / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml").instrument == "OHRC"


# ----------------------------------------------------------------------------- synthetic products

def odl(lines, samples, sample_type="MSB_UNSIGNED_INTEGER", bits=16, pointer='("tile.img", 1 <BYTES>)',
        bands=1, storage="BAND_SEQUENTIAL", product="TCO_MAP_02_TEST", instrument="TC",
        record_bytes=None, extra=""):
    rb = f"RECORD_BYTES = {record_bytes}\n" if record_bytes else ""
    return (
        f'PDS_VERSION_ID = "PDS3"\n{rb}^IMAGE = {pointer}\nMISSION_NAME = "SELENE"\n'
        f'PRODUCT_ID = "{product}"\nINSTRUMENT_ID = "{instrument}"\nSTART_TIME = UNK\n{extra}'
        f"OBJECT = IMAGE_MAP_PROJECTION\n  MAP_SCALE = 0.01 <km/pixel>\nEND_OBJECT = IMAGE_MAP_PROJECTION\n"
        f"OBJECT = IMAGE\n  BANDS = {bands}\n  BAND_STORAGE_TYPE = {storage}\n  LINES = {lines}\n"
        f"  LINE_SAMPLES = {samples}\n  SAMPLE_TYPE = {sample_type}\n  SAMPLE_BITS = {bits}\n"
        f"END_OBJECT = IMAGE\nEND\n"
    )


def write(tmp_path, label_text, data: np.ndarray, name="tile"):
    (tmp_path / f"{name}.lbl").write_text(label_text, encoding="ascii")
    data.tofile(tmp_path / f"{name}.img")
    return tmp_path / f"{name}.lbl"


@pytest.mark.parametrize("sample_type, bits, dtype", [
    ("MSB_UNSIGNED_INTEGER", 16, ">u2"),
    ("LSB_UNSIGNED_INTEGER", 16, "<u2"),
    ("PC_REAL", 32, "<f4"),
    ("IEEE_REAL", 32, ">f4"),
    ("UNSIGNED_INTEGER", 8, "|u1"),
])
def test_byte_order_and_type(tmp_path, sample_type, bits, dtype):
    data = (np.arange(60 * 40) % 200).reshape(60, 40).astype(dtype)
    label = write(tmp_path, odl(60, 40, sample_type, bits), data)
    assert parse_pds3(label).dtype == np.dtype(dtype).str
    assert np.array_equal(read_raster(label, Window(5, 7, 20, 11)), data[5:25, 7:18])


def test_byte_offset_pointer(tmp_path):
    data = np.arange(30 * 20, dtype=">u2").reshape(30, 20)
    (tmp_path / "tile.img").write_bytes(b"\x00" * 100 + data.tobytes())
    (tmp_path / "tile.lbl").write_text(odl(30, 20, pointer='("tile.img", 101 <BYTES>)'), encoding="ascii")
    assert read_array_layout(tmp_path / "tile.lbl").offset_bytes == 100
    assert np.array_equal(read_raster(tmp_path / "tile.lbl"), data)


def test_record_pointer_with_attached_label(tmp_path):
    """Label and pixels in one file; ^IMAGE counts records. The label parser must not read the pixels."""
    record_bytes = 512
    data = np.arange(50 * 40, dtype=">u2").reshape(50, 40)
    text = odl(50, 40, pointer="3", record_bytes=record_bytes).encode("ascii")
    assert len(text) <= 2 * record_bytes
    product = tmp_path / "TCO_MAP_02_ATTACHED.img"
    product.write_bytes(text.ljust(2 * record_bytes, b" ") + data.tobytes())
    meta = parse_pds3(product)
    assert read_array_layout(product).offset_bytes == 2 * record_bytes
    assert meta.raster_path == product
    assert np.array_equal(read_raster(product, Window(10, 10, 5, 5)), data[10:15, 10:15])


def test_record_pointer_without_record_bytes_raises(tmp_path):
    label = write(tmp_path, odl(10, 10, pointer='("tile.img", 2)'), np.zeros((10, 10), ">u2"))
    with pytest.raises(PdsParseError, match="RECORD_BYTES"):
        read_array_layout(label)


@pytest.mark.parametrize("storage, axes", [
    ("BAND_SEQUENTIAL", "bls"), ("LINE_INTERLEAVED", "lbs"), ("SAMPLE_INTERLEAVED", "lsb"),
])
def test_multiband_interleaves(tmp_path, storage, axes):
    cube = np.random.default_rng(3).random((4, 12, 9)).astype(">f4")      # (band, line, sample)
    on_disk = np.transpose(cube, ["bls".index(a) for a in axes])
    label = write(tmp_path, odl(12, 9, "IEEE_REAL", 32, bands=4, storage=storage,
                                product="MI_MAP_TEST", instrument="MI"), np.ascontiguousarray(on_disk))
    meta = parse_pds3(label)
    assert meta.n_bands == 4 and meta.instrument == "MI"
    got = read_raster(label, Window(2, 1, 6, 5), bands=[3, 1])
    assert np.array_equal(got, cube[[3, 1], 2:8, 1:6])


def test_instrument_mismatch_raises(tmp_path):
    label = write(tmp_path, odl(10, 10, instrument="MI"), np.zeros((10, 10), ">u2"))
    with pytest.raises(PdsParseError, match="INSTRUMENT_ID"):
        parse_pds3(label)


@pytest.mark.parametrize("breakage", ["truncate", "no_image_object", "no_pointer"])
def test_broken_labels_raise(tmp_path, breakage):
    text = odl(10, 10)
    if breakage == "truncate":
        text = text[: len(text) // 2]
    elif breakage == "no_image_object":
        text = text.split("OBJECT = IMAGE\n")[0] + "END\n"
    else:
        text = "\n".join(l for l in text.splitlines() if not l.startswith("^IMAGE")) + "\n"
    label = write(tmp_path, text, np.zeros((10, 10), ">u2"))
    with pytest.raises(PdsParseError):
        read_array_layout(label)


# ----------------------------------------------------------------------------- real products

REAL = sorted((ROOT / "data" / "raw" / "selene" / "tc").glob("*.lbl"))
needs_real = pytest.mark.skipif(len(REAL) < 2, reason="SELENE tiles not downloaded (data/raw is gitignored)")


@needs_real
@pytest.mark.parametrize("label", REAL, ids=lambda p: p.stem[-16:])
def test_real_pixels_reproduce_jaxa_statistics(label):
    """Our pixels reproduce the MINIMUM/MAXIMUM/AVERAGE/STDEV that JAXA wrote in the label.
    A wrong offset or byte order cannot pass this."""
    info = read_pds3_image_info(label)
    px = read_raster(label).astype(np.float64)
    valid = px[px != info["dummy"]]
    assert valid.min() == info["minimum"]
    assert valid.max() == info["maximum"]
    assert valid.mean() == pytest.approx(info["average"], abs=5e-7)
    assert valid.std() == pytest.approx(info["stdev"], abs=5e-7)


@needs_real
def test_real_window_matches_direct_bytes():
    label = REAL[0]
    layout = read_array_layout(label)
    tile = read_raster(label, Window(6000, 9000, 32, 32))
    samples = layout.shape[1]
    with layout.raster_path.open("rb") as fh:
        for r, c in [(0, 0), (31, 31), (10, 22)]:
            fh.seek(layout.offset_bytes + ((6000 + r) * samples + 9000 + c) * 2)
            assert tile[r, c] == np.frombuffer(fh.read(2), ">u2")[0]
