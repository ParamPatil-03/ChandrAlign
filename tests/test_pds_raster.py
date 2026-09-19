"""DATA-01 pixel reading: windowed reads with the label's exact layout.

Two groups:
* synthetic products -- a real ISRO label with its array shrunk and a small pixel file
  whose every value is known. Runs anywhere, covers byte order and band order.
* the real multi-GB products -- runs where data/raw holds them; skipped elsewhere.
"""
import time
from pathlib import Path

import numpy as np
import pytest
from lxml import etree

from chandralign.io.pds_label import parse_pds4, read_array_layout
from chandralign.io.pds_raster import (
    RasterIntegrityError,
    Window,
    check_envi_header,
    read_raster,
    verify_raster,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "labels"
OHRC_LABEL = FIXTURES / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml"
IIRS_LABEL = FIXTURES / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml"

NUMPY_TO_PDS4 = {"u1": "UnsignedByte", "<u2": "UnsignedLSB2", ">u2": "UnsignedMSB2",
                 "<f4": "IEEE754LSBSingle", ">f4": "IEEE754MSBSingle"}


def make_product(tmp_path, template: Path, array: np.ndarray, pds4_type: str, name: str) -> Path:
    """Write `array` to disk and a copy of a real label rewritten to describe it."""
    tree = etree.parse(str(template))
    ns = lambda tag: f"//*[local-name()='{tag}']"
    tree.xpath(ns("file_name"))[0].text = f"{name}.img"
    tree.xpath(ns("file_size"))[0].text = str(array.nbytes)
    tree.xpath(ns("data_type"))[0].text = pds4_type
    for axis, n in zip(tree.xpath(ns("Axis_Array")), array.shape):
        axis.xpath("*[local-name()='elements']")[0].text = str(n)
    label = tmp_path / f"{name}.xml"
    tree.write(str(label))
    array.tofile(tmp_path / f"{name}.img")
    return label


def ohrc_like(tmp_path, dtype: str) -> tuple[Path, np.ndarray]:
    data = (np.arange(300 * 200) % 251).reshape(300, 200).astype(dtype)
    name = "ch2_ohr_ncp_20240330T0035085365_d_img_d18"
    return make_product(tmp_path, OHRC_LABEL, data, NUMPY_TO_PDS4[dtype], name), data


@pytest.mark.parametrize("dtype", ["u1", "<u2", ">u2", ">f4"])
def test_window_matches_known_pixels(tmp_path, dtype):
    label, data = ohrc_like(tmp_path, dtype)
    w = Window(row=17, col=33, height=40, width=25)
    out = read_raster(parse_pds4(label), window=w)
    assert out.shape == (40, 25)
    assert np.array_equal(out, data[17:57, 33:58])
    assert out.dtype == np.dtype(dtype)   # byte order honoured, no silent conversion


def test_full_read_without_window(tmp_path):
    label, data = ohrc_like(tmp_path, "u1")
    assert np.array_equal(read_raster(label), data)


def test_band_selection_on_bsq_cube(tmp_path):
    cube = np.random.default_rng(1).random((6, 20, 15)).astype("<f4")   # (band, line, sample)
    name = "ch2_iir_nci_20240523T1600301891_d_img_d18"
    label = make_product(tmp_path, IIRS_LABEL, cube, "IEEE754LSBSingle", name)
    w = Window(2, 3, 10, 8)
    assert np.array_equal(read_raster(label, w, bands=4), cube[4, 2:12, 3:11])
    picked = read_raster(label, w, bands=[5, 0, 2])
    assert picked.shape == (3, 10, 8)
    assert np.array_equal(picked, cube[[5, 0, 2], 2:12, 3:11])
    assert read_raster(label, w).shape == (6, 10, 8)


@pytest.mark.parametrize("window", [Window(-1, 0, 5, 5), Window(0, 0, 301, 5),
                                    Window(290, 0, 20, 5), Window(0, 0, 0, 5)])
def test_out_of_bounds_windows_raise(tmp_path, window):
    label, _ = ohrc_like(tmp_path, "u1")
    with pytest.raises(ValueError):
        read_raster(label, window)


def test_bad_band_requests_raise(tmp_path):
    label, _ = ohrc_like(tmp_path, "u1")
    with pytest.raises(ValueError):
        read_raster(label, Window(0, 0, 5, 5), bands=3)   # single-band product


def test_short_pixel_file_is_rejected(tmp_path):
    label, data = ohrc_like(tmp_path, "u1")
    img = tmp_path / f"{label.stem}.img"
    img.write_bytes(img.read_bytes()[:-100])
    with pytest.raises(RasterIntegrityError):
        read_raster(label, Window(0, 0, 5, 5))
    with pytest.raises(RasterIntegrityError):
        verify_raster(label)


def test_md5_mismatch_is_detected(tmp_path):
    label, _ = ohrc_like(tmp_path, "u1")   # label still carries ISRO's md5 for the real file
    with pytest.raises(RasterIntegrityError, match="md5"):
        verify_raster(label, check_md5=True)


def test_missing_pixel_file(tmp_path):
    label, _ = ohrc_like(tmp_path, "u1")
    (tmp_path / f"{label.stem}.img").unlink()
    with pytest.raises(FileNotFoundError):
        read_raster(label, Window(0, 0, 5, 5))


# ----------------------------------------------------------------------------- real products

REAL = {p.name[:7]: p for p in sorted(
    (ROOT / "data" / "raw" / "ch2").glob("*/products/*/data/calibrated/*/*_d_img_d18.xml"))}
needs_real = pytest.mark.skipif(len(REAL) < 3, reason="real products not downloaded (data/raw is gitignored)")


def _direct_pixel(label: Path, band: int, line: int, sample: int):
    """One pixel read with plain seek/read and hand-computed offset -- independent of read_raster."""
    L = read_array_layout(label)
    names = [a.lower() for a in L.axis_names]
    idx = {"band": band, "line": line, "sample": sample}
    flat = 0
    for axis, size in zip(names, L.shape):
        flat = flat * size + idx[axis]
    item = np.dtype(L.dtype).itemsize
    with L.raster_path.open("rb") as fh:
        fh.seek(L.offset_bytes + flat * item)
        return np.frombuffer(fh.read(item), dtype=L.dtype)[0]


@needs_real
@pytest.mark.parametrize("key", ["ch2_ohr", "ch2_tmc", "ch2_iir"])
def test_real_window_matches_direct_byte_reads(key):
    label = REAL[key]
    meta = parse_pds4(label)
    lines, samples = meta.array_shape
    w = Window(row=lines // 2, col=samples // 3, height=64, width=64)
    band = 100 if meta.n_bands > 1 else 0
    tile = read_raster(meta, w, bands=band if meta.n_bands > 1 else None)
    for r, c in [(0, 0), (31, 17), (63, 63), (5, 60)]:
        assert tile[r, c] == _direct_pixel(label, band, w.row + r, w.col + c)


@needs_real
@pytest.mark.parametrize("key", ["ch2_ohr", "ch2_tmc", "ch2_iir"])
def test_real_small_window_is_fast(key):
    """PLAN.md P1-T03: a small window from a multi-GB product in < 100 ms."""
    meta = parse_pds4(REAL[key])
    lines, samples = meta.array_shape
    w = Window(lines - 40, samples - 40, 16, 16)   # far end of the file: no free ride from caching the start
    read_raster(meta, Window(0, 0, 1, 1), bands=0 if meta.n_bands > 1 else None)  # warm the label parse
    t = time.perf_counter()
    read_raster(meta, w, bands=0 if meta.n_bands > 1 else None)
    assert time.perf_counter() - t < 0.1


@needs_real
def test_real_iirs_envi_header_agrees_with_label():
    check_envi_header(REAL["ch2_iir"])


@needs_real
@pytest.mark.parametrize("key", ["ch2_ohr", "ch2_tmc", "ch2_iir"])
def test_real_sizes_match_label(key):
    assert verify_raster(REAL[key])["size_ok"]


@needs_real
@pytest.mark.slow
@pytest.mark.parametrize("key", ["ch2_ohr", "ch2_tmc", "ch2_iir"])
def test_real_pixels_match_isro_md5(key):
    """Our copy is byte-identical to what ISRO published. Reads the whole file."""
    assert verify_raster(REAL[key], check_md5=True)["md5_checked"]
