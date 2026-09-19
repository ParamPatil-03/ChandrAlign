"""PREP-01: percentile stretch + CLAHE."""
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import ImagePlane
from chandralign.io.pds_label import parse_label
from chandralign.preprocess.radiometric import clahe, local_contrast, percentile_stretch, prepare_plane

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
META = parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")


def terrain(n=256, amplitude=0.05, offset=0.05, seed=0):
    """Dark, low-contrast synthetic terrain: smooth bumps plus a little noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    z = np.sin(12 * x) * np.cos(9 * y) + 0.5 * np.sin(31 * x + 17 * y) + 0.05 * rng.standard_normal((n, n))
    z = (z - z.min()) / (z.max() - z.min())
    return (offset + amplitude * z).astype(np.float32)


def plane(array, valid=None, chain=("tile_minmax",)):
    valid = np.ones(array.shape, bool) if valid is None else valid
    return ImagePlane(array=array, valid_mask=valid, shadow_mask=np.zeros(array.shape, bool),
                      gsd_m=0.3, meta=META, preprocess_chain=list(chain))


# ----------------------------------------------------------------------------- stretch

def test_percentile_stretch_matches_numpy():
    img = terrain()
    out, lo, hi = percentile_stretch(img, np.ones(img.shape, bool), (2, 98))
    want_lo, want_hi = np.percentile(img.astype(np.float64), [2, 98])
    assert (lo, hi) == pytest.approx((want_lo, want_hi))
    assert out.dtype == np.float32 and out.min() == 0.0 and out.max() == 1.0
    assert np.allclose(out, np.clip((img - want_lo) / (want_hi - want_lo), 0, 1), atol=1e-6)


def test_a_few_hot_pixels_do_not_set_the_scale():
    img = terrain()
    hot = img.copy()
    hot[:3, :3] = 50.0                                         # 9 saturated pixels
    a, _, _ = percentile_stretch(img, np.ones(img.shape, bool))
    b, _, _ = percentile_stretch(hot, np.ones(img.shape, bool))
    assert np.abs(a[10:, 10:] - b[10:, 10:]).max() < 0.02


def test_stretch_edge_cases():
    valid = np.ones((10, 10), bool)
    flat, _, _ = percentile_stretch(np.full((10, 10), 7.0), valid)
    assert np.all(flat == 0.0) and not np.isnan(flat).any()   # constant image: zeros, not NaN
    empty, lo, _ = percentile_stretch(np.ones((10, 10)), np.zeros((10, 10), bool))
    assert np.all(empty == 0.0) and np.isnan(lo)
    with pytest.raises(ValueError):
        percentile_stretch(np.ones((4, 4)), np.ones((4, 4), bool), (99, 1))


# ----------------------------------------------------------------------------- CLAHE

def test_clahe_raises_local_contrast_of_dark_terrain():
    img = terrain()
    valid = np.ones(img.shape, bool)
    before = local_contrast(img, valid)
    after = local_contrast(clahe(img, valid), valid)
    assert after > 2 * before


def test_clahe_output_range_type_and_determinism():
    img = terrain()
    valid = np.ones(img.shape, bool)
    a, b = clahe(img, valid), clahe(img, valid)
    assert a.dtype == np.float32 and a.min() >= 0.0 and a.max() <= 1.0
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        clahe(img, valid, clip_limit=0)


def test_invalid_pixels_do_not_influence_the_result():
    img = terrain()
    valid = np.ones(img.shape, bool)
    valid[:, :64] = False                                     # left quarter is no-data
    zeros, ones = img.copy(), img.copy()
    zeros[~valid], ones[~valid] = 0.0, 1.0
    a, b = clahe(zeros, valid), clahe(ones, valid)
    assert np.array_equal(a, b)                              # whatever sits in the gap, output is identical
    assert np.all(a[~valid] == 0.0)


def test_local_contrast_measure():
    valid = np.ones((64, 64), bool)
    assert local_contrast(np.full((64, 64), 0.4), valid) == pytest.approx(0.0, abs=1e-6)   # float rounding only
    checker = (np.indices((64, 64)).sum(0) % 2).astype(float)
    assert local_contrast(checker, valid) == pytest.approx(0.5, abs=0.01)


# ----------------------------------------------------------------------------- the ImagePlane step

def test_prepare_plane_returns_a_new_plane_and_leaves_the_input_alone():
    img = terrain()
    p = plane(img.copy(), chain=("band_100", "tile_minmax"))
    before = p.array.copy()
    q = prepare_plane(p)
    assert q is not p and np.array_equal(p.array, before)      # input untouched
    assert q.preprocess_chain == ["band_100", "stretch_p1-99", "clahe_c2_t8x8"]
    assert q.valid_mask is not p.valid_mask and q.valid_mask is not q.shadow_mask
    assert np.array_equal(q.valid_mask, p.valid_mask)
    assert q.array.dtype == np.float32 and 0.0 <= q.array.min() and q.array.max() <= 1.0
    assert q.meta is p.meta and q.tile_origin == p.tile_origin


# ----------------------------------------------------------------------------- real products

def _real(pattern):
    hits = sorted((ROOT / "data" / "raw").glob(pattern))
    return parse_label(hits[0]) if hits else None


REAL = {
    "OHRC": _real("ch2/ohrc/products/*/data/calibrated/*/*_d_img_d18.xml"),
    "TMC2": _real("ch2/tmc2/products/*/data/calibrated/*/*_d_img_d18.xml"),
    "NAC_2009": _real("lro/nac/nac.m102000149rc/M102000149RC.XML"),
    "NAC_2022": _real("lro/nac/nac.m1417360906lc/M1417360906LC.XML"),
    "TC": _real("selene/tc/TCO_MAP_02_N03E021N00E024SC.lbl"),
}
needs_real = pytest.mark.skipif(not all(REAL.values()), reason="real products not downloaded")


def _mid_tile(meta):
    from chandralign.io.pds_raster import Window
    from chandralign.io.tiling import read_tile
    L, S = meta.array_shape
    return read_tile(meta, Window(L // 2, max(0, S // 2 - 512), 1024, 1024))


@needs_real
def test_real_shadowed_ohrc_crop_gains_local_contrast():
    """PLAN.md P1-T12 acceptance on the darkest 256 x 256 region of a mid-strip OHRC tile (sun 7.3 deg up)."""
    tile = _mid_tile(REAL["OHRC"])
    means = [(tile.array[r:r + 256, c:c + 256].mean(), r, c) for r in range(0, 769, 128) for c in range(0, 769, 128)]
    _, r, c = min(means)
    crop = plane(tile.array[r:r + 256, c:c + 256].copy())
    before = local_contrast(crop.array, crop.valid_mask)
    after = local_contrast(prepare_plane(crop).array, crop.valid_mask)
    assert after > 1.5 * before


@needs_real
def test_real_products_become_more_alike():
    """Across cameras and sun angles, the spread of local contrast narrows after CLAHE."""
    before, after = [], []
    for meta in REAL.values():
        t = _mid_tile(meta)
        before.append(local_contrast(t.array, t.valid_mask))
        after.append(local_contrast(prepare_plane(t).array, t.valid_mask))
    assert min(a / b for a, b in zip(after, before)) > 1.5
    assert max(after) / min(after) < max(before) / min(before)
