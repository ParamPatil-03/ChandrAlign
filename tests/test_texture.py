"""PREP-07 (flat terrain) and PREP-08 (repetitive terrain).

Real OHRC crops are committed as fixtures: a crater field, a featureless dark area
and a noise-dominated dark area, so both acceptance checks run anywhere.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import ImagePlane
from chandralign.io.pds_label import parse_label
from chandralign.preprocess.texture import (
    _noise_gain,
    band_pass,
    pixel_noise,
    repetitiveness_score,
    terrain_scores,
    texture_score,
    with_terrain_scores,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "tests" / "fixtures" / "images"
META = parse_label(ROOT / "tests" / "fixtures" / "labels" / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")


def crop(name):
    return np.load(IMAGES / f"ohrc_{name}.npy").astype(np.float64) / 255.0


def terrain(n=256, amplitude=0.05, seed=0):
    y, x = np.mgrid[0:n, 0:n] / n
    z = np.sin(11 * x) * np.cos(7 * y) + 0.4 * np.sin(29 * x + 13 * y)
    return 0.5 + amplitude * (z - z.mean()) / z.std()


def varied_craters(n=256, count=30, seed=0):
    """Craters of DIFFERENT sizes and depths: plenty of features, but no look-alikes."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 0.6)
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(count):
        cy, cx = rng.integers(0, n, 2)
        img -= rng.uniform(0.1, 0.45) * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(4, 22) ** 2)))
    return img


def craters(n=256, spacing=60, radius=9.0, jitter=0, seed=0):
    """Identical dark bowls; jitter>0 moves them off the grid without changing their look."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 0.6)
    yy, xx = np.mgrid[0:n, 0:n]
    for cy in range(spacing // 2, n, spacing):
        for cx in range(spacing // 2, n, spacing):
            dy, dx = (rng.integers(-jitter, jitter + 1, 2) if jitter else (0, 0))
            img -= 0.3 * np.exp(-(((xx - cx - dx) ** 2 + (yy - cy - dy) ** 2) / (2 * radius ** 2)))
    return img


def plane(array):
    return ImagePlane(array=array.astype(np.float32), valid_mask=np.ones(array.shape, bool),
                      shadow_mask=np.zeros(array.shape, bool), gsd_m=0.3, meta=META,
                      preprocess_chain=["tile_minmax"])


# ----------------------------------------------------------------------------- texture: structure, not noise

def test_pure_noise_has_no_texture():
    noise = np.random.default_rng(0).normal(0.5, 0.05, (256, 256))
    texture, raw, level = texture_score(noise)
    assert raw > 0.005                        # the band-pass does respond to noise...
    assert texture < 0.15 * raw               # ...and the score removes ~89% of it
    assert level == pytest.approx(0.05, rel=0.15)


def test_texture_survives_added_noise():
    """The same terrain under more and more noise keeps roughly the same texture score."""
    base = terrain()
    rng = np.random.default_rng(1)
    scores, raws = [], []
    for sigma in (0.0, 0.01, 0.03, 0.06):
        texture, raw, _ = texture_score(base + rng.normal(0, sigma, base.shape) if sigma else base)
        scores.append(texture)
        raws.append(raw)
    assert raws[-1] > 2 * raws[0]                                  # raw structure inflates with noise
    assert max(scores) / min(scores) < 1.3                          # the corrected score does not


def test_noise_gain_matches_measurement():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1.0, (512, 512))
    assert band_pass(noise).std() == pytest.approx(_noise_gain(1.0, 4.0), rel=0.05)
    assert _noise_gain(1.0, 4.0) == pytest.approx(0.2566, abs=0.001)


def test_flat_image_has_no_texture_and_no_features():
    flat = np.full((256, 256), 0.4)
    texture, raw, level = texture_score(flat)
    assert texture == 0.0 and raw == 0.0 and level == 0.0
    score, keypoints = repetitiveness_score(flat)
    assert score is None and keypoints == 0          # unmeasurable, NOT reported as zero
    assert pixel_noise(flat) == 0.0


def test_texture_rejects_non_images():
    with pytest.raises(ValueError):
        pixel_noise(np.zeros((5,)))


# ----------------------------------------------------------------------------- repetitiveness

def test_look_alikes_are_caught_whether_or_not_they_sit_on_a_grid():
    """It is the LOOK that repeats, not the spacing.

    Gridded craters are periodic and an autocorrelation test would find them.
    Jittered ones are NOT periodic -- autocorrelation would miss them -- but they
    still look alike, and this score still flags them. Craters of varied sizes hold
    just as many features and correctly score near zero.
    """
    gridded = repetitiveness_score(craters())[0]
    scattered = repetitiveness_score(craters(jitter=12, seed=3))[0]
    varied = repetitiveness_score(varied_craters())[0]
    assert gridded > 0.8
    assert scattered > 0.4                            # non-periodic, still flagged
    assert varied < 0.1                               # same crater count, no look-alikes
    assert scattered > 4 * max(varied, 0.02)


def test_terrain_without_look_alikes_scores_low():
    assert repetitiveness_score(varied_craters(seed=7))[0] < 0.1
    noise = np.random.default_rng(4).normal(0.5, 0.05, (256, 256))
    assert repetitiveness_score(noise)[0] == pytest.approx(0.0, abs=0.02)


def test_too_few_features_is_reported_as_unknown():
    """A smooth swell holds almost no features: repetitiveness is unmeasurable, not zero."""
    score, keypoints = repetitiveness_score(terrain())
    assert score is None and keypoints < 20


def test_a_feature_cannot_be_its_own_twin():
    """With a large separation nothing can twin; the score must fall, not stay put."""
    img = craters()
    close = repetitiveness_score(img, min_separation_px=5)[0]
    far = repetitiveness_score(img, min_separation_px=400)[0]     # wider than the tile
    assert close > 0.8
    assert far is None or far == 0.0


def test_no_data_regions_contribute_nothing():
    img = terrain()
    valid = np.ones(img.shape, bool)
    valid[:, :128] = False
    img[:, :128] = 0.0                                # a hard edge on the no-data boundary
    texture_masked, _, _ = texture_score(img, valid)
    texture_clean, _, _ = texture_score(terrain()[:, 128:], None)
    assert texture_masked == pytest.approx(texture_clean, rel=0.35)
    score, keypoints = repetitiveness_score(img, valid)
    assert keypoints > 0


# ----------------------------------------------------------------------------- the ImagePlane step

def test_with_terrain_scores_fills_the_contract_fields():
    p = plane(craters())
    q = with_terrain_scores(p)
    assert p.texture_score is None and p.repetitiveness_score is None    # input untouched
    assert q.texture_score > 0 and q.repetitiveness_score > 0.8
    assert q.valid_mask is not p.valid_mask and q.valid_mask is not q.shadow_mask
    assert np.array_equal(q.array, p.array) and q.preprocess_chain == p.preprocess_chain


# ----------------------------------------------------------------------------- real OHRC crops

def test_real_crater_field_scores_high_repetitiveness():
    """PLAN.md P1-T16 acceptance (second half), on a real crater field."""
    field = terrain_scores(crop("crater_field"))
    smooth = terrain_scores(crop("smooth"))
    noisy = terrain_scores(crop("noisy_dark"))
    assert field.repetitiveness > 0.08
    assert field.repetitiveness > 3 * max(smooth.repetitiveness, noisy.repetitiveness, 0.01)


def test_real_featureless_crop_scores_low_texture():
    """PLAN.md P1-T16 acceptance (first half): a featureless area must score low."""
    field = terrain_scores(crop("crater_field"))
    smooth = terrain_scores(crop("smooth"))
    assert smooth.texture < 0.001
    assert field.texture > 0.01
    assert field.texture > 10 * max(smooth.texture, 1e-4)


def test_real_noise_dominated_crop_is_not_mistaken_for_texture():
    """The honest case: its band-pass response is NOT zero, but almost all of it is noise."""
    noisy = terrain_scores(crop("noisy_dark"))
    assert noisy.raw_structure > 0.0
    assert noisy.texture < 0.5 * noisy.raw_structure + 0.002
    assert noisy.texture < terrain_scores(crop("crater_field")).texture / 5
    assert noisy.repetitiveness == pytest.approx(0.0, abs=0.02)


def test_scores_are_repeatable():
    a, b = terrain_scores(crop("crater_field")), terrain_scores(crop("crater_field"))
    assert (a.texture, a.repetitiveness, a.keypoints) == (b.texture, b.repetitiveness, b.keypoints)
