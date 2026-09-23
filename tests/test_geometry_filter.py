"""ALIGN-03: physics-based outlier filtering.

The experiment uses REAL lunar relief (our LOLA/SLDEM tiles) rendered under one of
our own products' sun geometries, and matches a 384 px tile back into the 3,072 px
area it was cut from. The true correspondence is therefore known exactly, so a
match is correct or it is not -- no eyeballing, and "more inliers" cannot be
mistaken for "better".

The headline number is in test_ransac_alone_confidently_returns_a_wrong_transform.
"""
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import ImagePlane
from chandralign.estimate import robust
from chandralign.estimate.geometry_filter import (
    ASPECT_MEANINGFUL_SLOPE_DEG,
    FilterReport,
    filter_matches,
    inlier_precision,
    terrain_at,
)
from chandralign.geometry.dem_terrain import hillshade, slope_aspect
from chandralign.io.dem import dem_patch, find_tiles
from chandralign.io.pds_label import parse_label
from chandralign.matching import classical

ROOT = Path(__file__).resolve().parents[1]
META = parse_label(ROOT / "tests" / "fixtures" / "labels"
                   / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
REAL_DEM = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")

OHRC_SUN = (269.8, 7.3)
TILE_ROW, TILE_COL, TILE_SIZE = 1200, 1500, 384
HILLSHADE_CROP = 3          # hillshade() output is trimmed by 3 px on every side

pytestmark = pytest.mark.skipif(len(REAL_DEM) < 2, reason="SLDEM tiles not downloaded")


class Grid:
    """pixel -> lat/lon on the DEM's own grid, offset to where the tile was cut from."""

    def __init__(self, patch, row0, col0):
        self.patch, self.row0, self.col0 = patch, row0, col0

    def pixel_to_latlon(self, rows, cols):
        rows = np.asarray(rows, float) + self.row0 + HILLSHADE_CROP
        cols = np.asarray(cols, float) + self.col0 + HILLSHADE_CROP
        return (self.patch.lat[0] - rows / self.patch.res_px_per_deg,
                self.patch.lon[0] + cols / self.patch.res_px_per_deg)


def plane(array):
    return ImagePlane(array=array.astype(np.float32), valid_mask=np.ones(array.shape, bool),
                      shadow_mask=np.zeros(array.shape, bool), gsd_m=59.0, meta=META,
                      preprocess_chain=["tile_minmax"])


@pytest.fixture(scope="module")
def scene():
    """A real 6-degree patch of the Moon, its terrain, and a tile cut from it."""
    patch = dem_patch(REAL_DEM, (-3.0, 3.0, 21.0, 27.0))
    terrain = slope_aspect(patch)
    big = hillshade(terrain, *OHRC_SUN)[HILLSHADE_CROP:-HILLSHADE_CROP,
                                        HILLSHADE_CROP:-HILLSHADE_CROP]
    tile = big[TILE_ROW:TILE_ROW + TILE_SIZE, TILE_COL:TILE_COL + TILE_SIZE]
    return patch, terrain, big, tile


@pytest.fixture(scope="module")
def truth():
    return lambda pts: pts + np.array([TILE_COL, TILE_ROW], float)


def run(scene, ratio, mutual=False):
    patch, _terrain, big, tile = scene
    matches = classical.match(plane(tile), plane(big), detector="sift",
                              ratio=ratio, mutual=mutual)
    kept, report = filter_matches(matches, META, META, patch,
                                  src_model=Grid(patch, TILE_ROW, TILE_COL),
                                  ref_model=Grid(patch, 0, 0))
    return matches, kept, report


# ----------------------------------------------------------------------------- the headline

def test_the_estimator_is_no_longer_fooled_on_this_pool(scene, truth):
    """ALIGN-03's original headline, CORRECTED 2026-09-23.

    Measured 2026-09-20 (commit 163935c): at 3.7% pool precision the estimator
    found 13 look-alike craters, none correct, and reported ok=True; the terrain
    filter turned that into 24 inliers, all correct. Re-run on the SAME 646-match
    pool with the estimator as it has been since c82b25b: 24 inliers, all correct,
    with or without the filter -- and 0 of 20 random seeds are fooled, for RANSAC
    and for MAGSAC alike. The estimator improved; the confident wrong answer the
    filter was shown to prevent no longer happens here. This test pins the current
    behaviour so a regression that brings it back is caught. What the filter still
    does on this pool is the next test: it removes most wrong matches and loses none.
    """
    matches, kept, _ = run(scene, ratio=0.95)
    for pool in (matches, kept):
        res = robust.estimate(pool.src_pts, pool.ref_pts, kind="affine")
        mask = np.asarray(res.inlier_mask, bool)
        precision, correct, n = inlier_precision(pool.src_pts[mask], pool.ref_pts[mask], truth)
        assert res.ok and correct >= 20 and precision == 1.0, (len(pool.src_pts), correct, n)


def test_the_filter_raises_precision_without_losing_true_matches(scene, truth):
    """Measured at three match-pool qualities. Every true match survives, every time."""
    results = {}
    for ratio in (0.95, 0.90, 0.85):
        matches, kept, report = run(scene, ratio=ratio)
        p_before, correct_before, n_before = inlier_precision(
            matches.src_pts, matches.ref_pts, truth)
        p_after, correct_after, n_after = inlier_precision(kept.src_pts, kept.ref_pts, truth)
        results[ratio] = (p_before, p_after, correct_before, correct_after, n_before, n_after)

        assert report.applied is True
        assert p_after > p_before, f"ratio {ratio}: precision did not improve"
        assert correct_after == correct_before, (
            f"ratio {ratio}: lost {correct_before - correct_after} TRUE matches")
        assert n_after < n_before                      # it did remove something

    # The worst pool benefits most, which is the point: it is a rescue, not a polish.
    assert results[0.95][1] / max(results[0.95][0], 1e-9) > 2.0
    assert results[0.85][1] > 0.5


def test_it_rejects_on_slope_far_more_often_than_on_aspect(scene):
    """Recorded because it says which signal is doing the work."""
    _matches, _kept, report = run(scene, ratio=0.85)
    assert report.n_rejected_slope > report.n_rejected_aspect
    assert report.n_rejected_slope + report.n_rejected_aspect == report.n_in - report.n_kept
    assert report.n_unknown_terrain == 0               # the whole tile is inside the DEM


# ----------------------------------------------------------------------------- switching off

def test_with_no_dem_it_passes_everything_through_and_says_so(scene):
    matches, _kept, _ = run(scene, ratio=0.85)
    out, report = filter_matches(matches, META, META, None)
    assert out is matches
    assert report.applied is False
    assert "no DEM" in report.reason
    assert report.n_kept == report.n_in
    assert report.rejected_fraction == 0.0


def test_an_empty_match_set_is_handled(scene):
    patch, *_ = scene
    empty = classical.match(plane(np.zeros((64, 64), np.float32)),
                            plane(np.zeros((64, 64), np.float32)), detector="sift")
    out, report = filter_matches(empty, META, META, patch)
    assert len(out.src_pts) == 0
    assert report.applied is False and report.n_in == 0


def test_endpoints_outside_the_dem_are_not_judged(scene):
    """Unknown terrain is not evidence of a bad match, so those matches are KEPT."""
    patch, *_ = scene
    matches, _kept, _ = run(scene, ratio=0.85)
    far = Grid(patch, 0, 0)
    far.row0 = 10 ** 6                                 # push every point off the patch
    out, report = filter_matches(matches, META, META, patch,
                                 src_model=far, ref_model=far)
    assert report.applied is False
    assert "outside" in report.reason or "no match endpoint" in report.reason
    assert report.n_kept == report.n_in


# ----------------------------------------------------------------------------- the pieces

def test_terrain_at_returns_nan_outside_the_patch(scene):
    patch, terrain, *_ = scene
    inside_lat, inside_lon = patch.lat[100], patch.lon[100]
    slope, _aspect = terrain_at([inside_lat], [inside_lon], terrain, patch)
    assert np.isfinite(slope[0])
    slope, aspect = terrain_at([80.0], [200.0], terrain, patch)     # nowhere near it
    assert np.isnan(slope[0]) and np.isnan(aspect[0])


def test_inlier_precision_counts_only_matches_that_land_on_the_truth():
    src = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    ref = np.array([[110.0, 10.0], [120.0, 20.0], [999.0, 999.0]])
    truth = lambda pts: pts + np.array([100.0, 0.0])
    precision, correct, total = inlier_precision(src, ref, truth)
    assert (correct, total) == (2, 3)
    assert precision == pytest.approx(2 / 3)
    assert inlier_precision(np.empty((0, 2)), np.empty((0, 2)), truth) == (0.0, 0, 0)


def test_the_report_is_honest_about_doing_nothing():
    report = FilterReport(False, "no DEM supplied: physics filter disabled", 10, 10)
    assert report.applied is False and report.rejected_fraction == 0.0
    assert FilterReport(True, "x", 0, 0).rejected_fraction == 0.0


def test_aspect_is_ignored_on_flat_ground(scene):
    """A filter that trusted aspect on flat ground would discard good matches where
    they are already scarce: the downhill direction of a level plain is noise."""
    assert ASPECT_MEANINGFUL_SLOPE_DEG > 0
    patch, terrain, *_ = scene
    flat = np.nanpercentile(terrain.slope_deg, 5)
    assert flat < ASPECT_MEANINGFUL_SLOPE_DEG          # real lunar plains are this flat
