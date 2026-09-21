"""DATA-11 tiling, and PLAN.md GATE A: real ImagePlanes from a real OHRC and a real NAC product."""
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from lxml import etree

from chandralign.contracts import ImagePlane
from chandralign.io.pds_label import parse_label, read_special_values
from chandralign.io.pds_raster import Window, read_raster
from chandralign.io.tiling import (
    TilingStats,
    global_to_tile,
    iter_tiles,
    read_tile,
    tile_grid,
    tile_to_global,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "labels"


# ----------------------------------------------------------------------------- the grid

@pytest.mark.parametrize("lines, samples, tile, overlap", [
    (5000, 4000, 1024, 128), (1024, 1024, 1024, 128), (300, 200, 1024, 128),
    (2049, 1025, 512, 64), (160269, 4000, 1024, 128),
])
def test_grid_covers_every_pixel_with_full_tiles(lines, samples, tile, overlap):
    grid = tile_grid(lines, samples, tile, overlap)
    rows = sorted({w.row for w in grid})
    cols = sorted({w.col for w in grid})
    h, w = min(tile, lines), min(tile, samples)
    assert all(win.height == h and win.width == w for win in grid)
    assert rows[0] == 0 and rows[-1] + h == lines          # last tile flush with the edge
    assert cols[0] == 0 and cols[-1] + w == samples
    for a, b in zip(rows, rows[1:]):
        assert b - a <= h - overlap                          # at least `overlap` shared: no gaps
    for a, b in zip(cols, cols[1:]):
        assert b - a <= w - overlap
    assert len(grid) == len(rows) * len(cols)


def test_grid_counts_for_real_strip_sizes():
    # TMC-2 strip: 160269 x 4000 with 1024/128 -> stride 896
    assert len(tile_grid(160269, 4000)) == 179 * 5


@pytest.mark.parametrize("kwargs", [dict(tile=0), dict(tile=512, overlap=512), dict(overlap=-1)])
def test_grid_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        tile_grid(1000, 1000, **kwargs)


def test_points_round_trip_between_tile_and_product():
    rng = np.random.default_rng(7)
    pts = rng.random((500, 2)) * 1024                      # sub-pixel (x, y) in a tile
    for origin in [(0, 0), (896, 1792), (159245, 2976)]:
        back = global_to_tile(tile_to_global(pts, origin), origin)
        assert np.max(np.abs(back - pts)) < 1e-9           # float64: exact to well below 1e-9 px
    # convention: (x, y) = (col, row); origin is (row, col)
    assert np.array_equal(tile_to_global([[3.5, 7.25]], (100, 20)), [[23.5, 107.25]])


# ----------------------------------------------------------------------------- special values

def test_special_values_from_real_labels():
    nac = read_special_values(FIXTURES / "M1417360906LC.XML")
    assert nac.nodata == (-32768.0, -32767.0, -32766.0, -32765.0, -32764.0)
    assert nac.valid_min == -32752.0
    # the PDS3 header embedded in the same .IMG states the same codes
    assert read_special_values(FIXTURES / "M1417360906LC_attached_header.lbl") == nac
    selene = read_special_values(FIXTURES / "TCO_MAP_02_N03E021N00E024SC.lbl")
    assert 0.0 in selene.nodata and (selene.valid_min, selene.valid_max) == (2.0, 32766.0)
    ohrc = read_special_values(FIXTURES / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
    assert ohrc.declared is False and ohrc.nodata == ()


# ----------------------------------------------------------------------------- synthetic products

def nac_like(tmp_path, data: np.ndarray) -> Path:
    """A real NAC label rewritten to describe `data` (int16 LE), keeping its 5064-byte header offset."""
    tree = etree.parse(str(FIXTURES / "M1417360906LC.XML"))
    q = lambda t: tree.xpath(f"//*[local-name()='{t}']")
    for el in q("file_name"):
        el.text = "M1417360906LC.IMG"
    q("file_size")[0].text = str(5064 + data.nbytes)
    q("md5_checksum")[0].text = "0" * 32
    for axis, n in zip(q("Axis_Array"), data.shape):
        axis.xpath("*[local-name()='elements']")[0].text = str(n)
    tree.write(str(tmp_path / "M1417360906LC.XML"))
    (tmp_path / "M1417360906LC.IMG").write_bytes(b" " * 5064 + data.astype("<i2").tobytes())
    return tmp_path / "M1417360906LC.XML"


def test_tiles_mask_nodata_and_skip_empty_tiles(tmp_path):
    data = (np.arange(600 * 400) % 3000).reshape(600, 400).astype("<i2")
    data[:, :150] = -32768                                  # a missing strip on the left
    data[500, 300] = -32765                                 # one saturated pixel
    meta = parse_label(nac_like(tmp_path, data))

    stats = TilingStats()
    planes = list(iter_tiles(meta, tile=200, overlap=0, max_invalid_fraction=0.5, stats=stats))
    assert stats.planned == 3 * 2
    assert stats.skipped_invalid == 3                      # left-hand tiles are 75% missing
    assert stats.yielded == len(planes) == 3

    for p in planes:
        assert isinstance(p, ImagePlane)
        assert p.array.dtype == np.float32 and p.array.min() >= 0.0 and p.array.max() <= 1.0
        assert p.valid_mask is not p.shadow_mask            # failure mode #19
        assert p.preprocess_chain == ["tile_minmax"] and p.gsd_m == meta.gsd_m
        r, c = p.tile_origin
        raw = data[r:r + 200, c:c + 200]
        expected = ~np.isin(raw, [-32768, -32767, -32766, -32765, -32764]) & (raw >= -32752)
        assert np.array_equal(p.valid_mask, expected)
        assert np.all(p.array[~p.valid_mask] == 0.0)


def test_tile_pixels_are_the_product_pixels(tmp_path):
    data = np.random.default_rng(5).integers(0, 4000, (300, 250)).astype("<i2")
    meta = parse_label(nac_like(tmp_path, data))
    plane = read_tile(meta, Window(100, 50, 64, 64))
    raw = data[100:164, 50:114].astype(np.float64)
    assert np.allclose(plane.array, (raw - raw.min()) / (raw.max() - raw.min()), atol=1e-6)
    # a point in tile coordinates lands on the matching product pixel
    (x, y), = tile_to_global([[10, 20]], plane.tile_origin)
    assert data[int(y), int(x)] == data[100 + 20, 50 + 10]


def test_multiband_needs_a_band():
    meta = parse_label(FIXTURES / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml")
    with pytest.raises(ValueError, match="bands"):
        read_tile(meta, Window(0, 0, 8, 8))


# ----------------------------------------------------------------------------- real products: GATE A

def _find(pattern: str):
    hits = sorted((ROOT / "data" / "raw").glob(pattern))
    return hits[0] if hits else None


OHRC = _find("ch2/ohrc/products/*/data/calibrated/*/*_d_img_d18.xml")
TMC2 = _find("ch2/tmc2/products/*/data/calibrated/*/*_d_img_d18.xml")
IIRS = _find("ch2/iirs/products/*/data/calibrated/*/*_d_img_d18.xml")
NAC = _find("lro/nac/nac.m1417360906lc/M1417360906LC.XML")
needs_real = pytest.mark.skipif(not any([OHRC, TMC2, IIRS, NAC]), reason="real products not downloaded")


def require(label, name: str):
    """Skip unless THIS product is present; holding the others does not help."""
    if label is None:
        pytest.skip(f"{name} product not downloaded")
    return label


@needs_real
@pytest.mark.parametrize("label", [OHRC, NAC], ids=["OHRC", "NAC"])
def test_gate_a_real_imageplane(label):
    """PLAN.md GATE A: a real ImagePlane from a real OHRC and a real NAC product."""
    meta = parse_label(require(label, "OHRC/NAC"))
    lines, samples = meta.array_shape
    w = Window(lines // 2, samples // 2, 1024, 1024)
    plane = read_tile(meta, w)
    assert isinstance(plane, ImagePlane) and plane.meta is meta
    assert plane.array.shape == (1024, 1024) and plane.array.dtype == np.float32
    assert 0.0 <= plane.array.min() and plane.array.max() <= 1.0
    assert plane.valid_mask.mean() > 0.99                   # mid-strip: essentially all real data
    assert plane.array.std() > 0.01                         # real terrain, not a flat fill
    raw = read_raster(meta, w)
    real = raw[plane.valid_mask]
    assert (plane.array[plane.valid_mask & (raw == real.max())] == 1.0).all()
    assert (plane.array[plane.valid_mask & (raw == real.min())] == 0.0).all()


@needs_real
def test_real_iirs_band_tile():
    meta = parse_label(require(IIRS, "IIRS"))
    plane = read_tile(meta, Window(6000, 0, 256, 250), band=100)
    assert plane.array.shape == (256, 250)
    assert plane.preprocess_chain == ["band_100", "tile_minmax"]


@needs_real
def test_tiling_a_full_strip_keeps_memory_bounded():
    """Stepping along the 1.3 GB TMC-2 strip holds only a few tiles in memory at once."""
    meta = parse_label(require(TMC2, "TMC-2"))
    stats = TilingStats()
    tracemalloc.start()
    n = 0
    for plane in iter_tiles(meta, stats=stats):
        n += 1
        if n == 40:
            break
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert stats.planned == 179 * 5
    assert n == 40
    assert peak < 64 * 2**20, f"peak {peak / 2**20:.1f} MiB"
