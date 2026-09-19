"""PREP-04: shadow detection.

The plan's acceptance (IoU > 0.7 against a HAND-DRAWN mask of a crater-interior crop)
needs a mask traced by a person; until tests/fixtures/images/ohrc_shadow_crop_mask.png
exists, that test is skipped rather than graded against the detector's own output.
"""
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import ImagePlane
from chandralign.io.pds_label import parse_label
from chandralign.preprocess.radiometric import prepare_plane
from chandralign.preprocess.shadow_mask import detect_shadows, mask_iou, with_shadow_mask

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
IMAGES = ROOT / "tests" / "fixtures" / "images"
META = parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")   # sun 7.3 deg up
CROP = IMAGES / "ohrc_shadow_crop.npy"            # real OHRC DN, rows 49152.., cols 7168.. of the strip
HAND_MASK = IMAGES / "ohrc_shadow_crop_mask.png"  # to be traced by a person


def scene(n=256, seed=0):
    """Sunlit textured terrain (DN ~20-60) with a uniformly dark disk at the sensor floor (DN 3-5)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    img = 35 + 12 * np.sin(x / 7.0) * np.cos(y / 11.0) + rng.normal(0, 3, (n, n))
    disk = (x - 90) ** 2 + (y - 120) ** 2 < 50 ** 2
    img[disk] = rng.integers(3, 6, disk.sum())
    return img, disk


def plane(array, chain=("tile_minmax",)):
    return ImagePlane(array=array.astype(np.float32), valid_mask=np.ones(array.shape, bool),
                      shadow_mask=np.zeros(array.shape, bool), gsd_m=0.3, meta=META,
                      preprocess_chain=list(chain))


# ----------------------------------------------------------------------------- synthetic

def test_dark_uniform_region_is_found():
    img, disk = scene()
    r = detect_shadows(img, np.ones(img.shape, bool))
    assert mask_iou(r.mask, disk) > 0.95
    assert r.black_level < 6 and 25 < r.sunlit_level < 45


def test_dark_but_textured_ground_is_not_shadow():
    img, _ = scene()
    rng = np.random.default_rng(1)
    img[:60, 180:] = rng.uniform(2, 60, (60, 76))         # dark material with strong texture
    r = detect_shadows(img, np.ones(img.shape, bool))
    assert r.mask[:60, 180:].mean() < 0.05                 # the local-variation check rejects it


def test_specks_are_removed_and_invalid_pixels_never_count():
    img, disk = scene()
    img[200:203, 200:203] = 3                              # 9-pixel dark speck
    valid = np.ones(img.shape, bool)
    valid[110:130, 80:100] = False                         # no-data hole inside the shadow
    r = detect_shadows(img, valid)
    assert not r.mask[200:203, 200:203].any()
    assert not r.mask[~valid].any()


def test_no_contrast_means_no_decision():
    r = detect_shadows(np.full((50, 50), 10.0), np.ones((50, 50), bool))
    assert not r.mask.any() and "no contrast" in r.notes[0]
    with pytest.raises(ValueError):
        detect_shadows(np.ones((5, 5)), np.ones((5, 5), bool), dark_fraction=1.5)


def test_high_sun_with_many_shadows_is_flagged():
    img, _ = scene()
    r = detect_shadows(img, np.ones(img.shape, bool), incidence_deg=10.0)   # sun 80 deg up
    assert any("mistaken for shadow" in n for n in r.notes)


def test_mask_iou():
    a = np.zeros((10, 10), bool); a[:5] = True
    b = np.zeros((10, 10), bool); b[:5, :5] = True
    assert mask_iou(a, b) == pytest.approx(0.5)
    assert mask_iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 1.0


# ----------------------------------------------------------------------------- ImagePlane step

def test_with_shadow_mask_then_clahe_keeps_the_mask():
    img, disk = scene()
    p = plane(img)
    s = with_shadow_mask(p)
    assert s is not p and not p.shadow_mask.any()          # input untouched
    assert s.shadow_mask is not s.valid_mask               # failure mode #19
    assert s.preprocess_chain == ["tile_minmax", "shadow_f0.15"]
    ready = prepare_plane(s)
    assert np.array_equal(ready.shadow_mask, s.shadow_mask) and ready.shadow_mask is not s.shadow_mask


def test_refuses_to_run_after_clahe():
    img, _ = scene()
    with pytest.raises(ValueError, match="before CLAHE"):
        with_shadow_mask(plane(img, chain=("stretch_p1-99", "clahe_c2_t8x8")))


# ----------------------------------------------------------------------------- real OHRC crater crop

def test_real_crater_interior_is_shadow_and_sunlit_ground_is_not():
    crop = np.load(CROP).astype(float)
    r = detect_shadows(crop, np.ones(crop.shape, bool), incidence_deg=META.solar_incidence_deg)
    assert (r.black_level, r.sunlit_level) == (3.0, 24.0)
    assert r.mask[150:300, 20:120].all()                   # deep inside the large crater's shadow
    assert not r.mask[200:300, 300:400].any()             # sunlit plain to its east
    assert 0.30 < r.fraction < 0.38
    assert r.notes == ()                                   # sun 7 deg up: many shadows are expected


@pytest.mark.skipif(not HAND_MASK.exists(), reason="hand-drawn reference mask not provided yet")
def test_real_crop_matches_hand_drawn_mask():
    """PLAN.md P1-T15 acceptance: IoU > 0.7 against a mask traced by a person (pure red, 255,0,0)."""
    import cv2
    drawn = cv2.imread(str(HAND_MASK), cv2.IMREAD_COLOR)
    truth = (drawn[:, :, 2] > 200) & (drawn[:, :, 1] < 60) & (drawn[:, :, 0] < 60)
    crop = np.load(CROP).astype(float)
    r = detect_shadows(crop, np.ones(crop.shape, bool))
    assert truth.any(), "no pure-red pixels found in the hand-drawn mask"
    assert mask_iou(r.mask, truth) > 0.7
