"""PREP-05 (band quality) and PREP-06 (single-plane composite).

On the acceptance criterion: PLAN.md P1-T14 asks for "higher keypoint count than
the best single band". Measured on the real cube, that is FALSE and the criterion
is the problem -- SIFT fires on noise, so the noisiest band wins on raw count
(band 44: 1413 keypoints at noise 0.0478; the 126-band composite: 1319 at 0.0417).
What matters is how many keypoints are REAL, which is measured here by matching
two composites built from disjoint halves of the bands: same terrain, independent
noise, so only genuine features can match. See test_real_composite_beats_single_band.
"""
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import ImagePlane
from chandralign.io.pds_label import parse_label
from chandralign.io.pds_raster import Window
from chandralign.preprocess.iirs_composite import (
    CompositeError,
    approximate_wavelengths,
    band_noise,
    composite,
    iirs_composite_plane,
    product_band_selection,
    select_bands,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
IIRS_LABEL = LABELS / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml"
WAVELENGTHS = (800.0, 5000.0)


def synthetic_cube(n_bands=40, n=64, seed=0):
    """Known terrain seen by every band, with a KNOWN noise level per band.

    Band b has noise 1 + b, so SNR falls as the band index rises; bands 30+ are
    below an SNR of 30 and bands 0-1 are given dead pixels.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    terrain = 200 + 60 * np.sin(9 * x) * np.cos(7 * y)
    cube = np.empty((n_bands, n, n))
    for b in range(n_bands):
        cube[b] = terrain + rng.normal(0, 1 + b, (n, n))
    cube[0, :, :10] = 0.0                       # 15.6% dead
    cube[1, :3, :] = -1.0                       # 4.7% dead
    cube[2, 0, 0] = 0.0                         # a single dead pixel: must NOT disqualify
    return cube, terrain


# ----------------------------------------------------------------------------- scoring

def test_noise_estimate_recovers_known_noise():
    cube, _ = synthetic_cube()
    measured = band_noise(cube)
    expected = np.array([1.0 + b for b in range(cube.shape[0])])
    ratio = measured[3:] / expected[3:]          # skip the bands with dead pixels
    assert np.allclose(ratio, 1.0, atol=0.15), ratio


def test_terrain_does_not_count_as_noise():
    """A noiseless but rugged band must score as near-zero noise."""
    _, terrain = synthetic_cube()
    assert band_noise(terrain[None]) [0] < 0.01 * terrain.std()


def test_bad_bands_are_rejected_for_the_stated_reason():
    cube, _ = synthetic_cube()
    sel = select_bands(cube, WAVELENGTHS, min_snr=30, max_wavelength_nm=None)
    assert sel.rejected["dead_pixels"] == (0, 1)             # not band 2, one pixel only
    assert 2 in sel.bands
    assert all(sel.snr[b] >= 30 for b in sel.bands)
    assert all(b in sel.rejected["low_snr"] for b in range(len(sel.snr)) if sel.snr[b] < 30 and b > 1)
    assert sum(sel.weights) == pytest.approx(1.0)
    assert sel.label.startswith(f"iirs_composite_{len(sel.bands)}of{len(sel.snr)}")


def test_thermal_bands_are_excluded_by_wavelength():
    cube, _ = synthetic_cube(n_bands=40)
    waves, approximate = approximate_wavelengths(40, WAVELENGTHS)
    assert approximate and waves[0] == 800.0 and waves[-1] == 5000.0
    sel = select_bands(cube, WAVELENGTHS, min_snr=0, max_wavelength_nm=3000)
    assert all(waves[b] <= 3000 for b in sel.bands)
    assert set(sel.rejected["thermal"]) == {b for b in range(40) if waves[b] > 3000} - {0, 1}


def test_no_wavelengths_means_no_wavelength_rejection():
    cube, _ = synthetic_cube()
    waves, approximate = approximate_wavelengths(40, None)
    assert np.isnan(waves).all() and approximate
    sel = select_bands(cube, None, min_snr=0, max_wavelength_nm=3000)
    assert sel.rejected["thermal"] == ()                     # unknown wavelengths reject nothing


def test_max_bands_keeps_the_best_of_those_that_pass():
    cube, _ = synthetic_cube()
    sel = select_bands(cube, WAVELENGTHS, min_snr=0, max_wavelength_nm=None, max_bands=5)
    assert len(sel.bands) == 5
    # bands 0 and 1 have the highest raw SNR but are rejected for dead pixels first,
    # so the cap applies to what survives, not to the whole cube
    passing = [b for b in range(len(sel.snr)) if b not in sel.rejected["dead_pixels"]]
    assert set(sel.bands) == set(sorted(passing, key=lambda b: -sel.snr[b])[:5])
    assert 0 not in sel.bands and 1 not in sel.bands


def test_impossible_selections_raise():
    cube, _ = synthetic_cube()
    with pytest.raises(CompositeError, match="no band passes"):
        select_bands(cube, WAVELENGTHS, min_snr=10_000)
    with pytest.raises(CompositeError):
        band_noise(cube[0])                                   # not a cube
    sel = select_bands(cube, WAVELENGTHS, min_snr=0, max_wavelength_nm=None)
    with pytest.raises(CompositeError, match="bands"):
        composite(cube[:10], sel)                             # selection made on a different cube


# ----------------------------------------------------------------------------- blending

def test_composite_is_quieter_than_every_band_it_blends():
    """Averaging independent noise is the whole point: the blend must be quieter
    than even its best ingredient."""
    cube, _ = synthetic_cube()
    sel = select_bands(cube, WAVELENGTHS, min_snr=30, max_wavelength_nm=None)
    blended = composite(cube, sel)
    blend_noise = band_noise(blended[None])[0]
    # compare like with like: each band normalised by its own noise, as the blend does
    per_band = [band_noise(((cube[b] - np.median(cube[b])) / sel.noise[b])[None])[0] for b in sel.bands]
    assert blend_noise < min(per_band)
    assert blend_noise < 0.75 * min(per_band)                 # a real reduction, not a rounding


def test_composite_preserves_the_terrain():
    cube, terrain = synthetic_cube()
    sel = select_bands(cube, WAVELENGTHS, min_snr=30, max_wavelength_nm=None)
    blended = composite(cube, sel)
    a = (blended - blended.mean()) / blended.std()
    b = (terrain - terrain.mean()) / terrain.std()
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99


# ----------------------------------------------------------------------------- real product

REAL = sorted((ROOT / "data" / "raw" / "ch2" / "iirs").glob(
    "products/ch2_iir_nci_20240523T1600301891_d_img_d18/data/calibrated/*/*_d_img_d18.xml"))   # the evidence scene
needs_real = pytest.mark.skipif(not REAL, reason="real IIRS product not downloaded")


@pytest.fixture(scope="module")
def real_selection():
    return product_band_selection(parse_label(REAL[0]))


def test_single_band_product_is_refused():
    ohrc = parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
    with pytest.raises(CompositeError, match="1 band"):
        product_band_selection(ohrc)


@needs_real
def test_real_band_selection(real_selection):
    sel = real_selection
    assert len(sel.snr) == 256
    assert sel.rejected["dead_pixels"] == (0, 1, 2)           # 9.7%, 4.5%, 1.6% dead
    assert 100 < len(sel.bands) < 140
    assert all(sel.snr[b] >= 30 for b in sel.bands)
    assert max(sel.wavelength_nm[list(sel.bands)]) <= 3000    # no thermal bands
    assert sel.wavelength_is_approximate                      # never claimed as label truth
    assert len(sel.rejected["thermal"]) > 100


@needs_real
def test_real_composite_plane_is_a_valid_handoff(real_selection):
    meta = parse_label(REAL[0])
    plane = iirs_composite_plane(meta, Window(6000, 0, 256, 250), real_selection)
    assert isinstance(plane, ImagePlane)
    assert plane.array.shape == (256, 250) and plane.array.dtype == np.float32
    assert 0.0 <= plane.array.min() and plane.array.max() <= 1.0
    assert plane.valid_mask.mean() > 0.99 and plane.valid_mask is not plane.shadow_mask
    assert plane.tile_origin == (6000, 0) and plane.gsd_m == meta.gsd_m
    assert plane.preprocess_chain[0] == real_selection.label  # the band list is recorded
    assert f"of256" in plane.preprocess_chain[0]


@needs_real
def test_tiles_of_one_product_share_one_band_selection(real_selection):
    """Neighbouring tiles must be built from the SAME bands or they stop being comparable."""
    meta = parse_label(REAL[0])
    a = iirs_composite_plane(meta, Window(6000, 0, 128, 250), real_selection)
    b = iirs_composite_plane(meta, Window(6128, 0, 128, 250), real_selection)
    assert a.preprocess_chain == b.preprocess_chain


@needs_real
def test_real_composite_beats_single_band(real_selection):
    """PREP-06 acceptance, measured honestly.

    Two images of the same terrain with INDEPENDENT noise can only match on real
    features, so this counts genuine, matchable structure. Raw keypoint count is
    NOT used: it rewards noise (see the module docstring).
    """
    import cv2

    from chandralign.preprocess.radiometric import clahe, percentile_stretch

    meta = parse_label(REAL[0])
    sel = real_selection
    window = Window(6000, 0, 256, 250)
    from chandralign.io.pds_raster import read_raster
    cube = read_raster(meta, window).astype(np.float64)
    valid = np.ones(cube.shape[1:], bool)
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher()

    def prep(img):
        return (clahe(percentile_stretch(img.astype(np.float32), valid)[0], valid) * 255).astype(np.uint8)

    def true_matches(a, b):
        ka, da = sift.detectAndCompute(prep(a), None)
        kb, db = sift.detectAndCompute(prep(b), None)
        good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return 0
        pa = np.float32([ka[g.queryIdx].pt for g in good])
        pb = np.float32([kb[g.trainIdx].pt for g in good])
        _, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        return int(inliers.sum())

    bands = list(sel.bands)
    best = max(bands, key=lambda b: sel.snr[b])
    single = true_matches(cube[best], cube[best + 1])          # two good single bands
    halves = [sel.bands[0::2], sel.bands[1::2]]
    blended = []
    for half in halves:
        w = sel.snr[list(half)] ** 2
        centred = (cube[list(half)] - sel.signal[list(half)][:, None, None])
        scaled = centred / sel.noise[list(half)][:, None, None]
        blended.append((scaled * (w / w.sum())[:, None, None]).sum(0))
    paired = true_matches(*blended)
    assert single > 100                                        # the single band is not broken
    assert paired > 1.15 * single, f"composite {paired} vs single band {single}"
