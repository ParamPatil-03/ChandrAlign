"""Matcher-level tiling of a PRE-ALIGNED pair (xoftr speed work).

Not DATA-11: that splits a product into tiles for reading. This splits one
coarse-aligned source/reference pair so a dense matcher whose memory grows with
the square of the pixel count never sees the whole window at once. No GPU and no
weights: every test drives the tiler with a fake matcher whose answer is known.
"""
import numpy as np
import pytest

from chandralign import synth
from chandralign.contracts import MatchSet
from chandralign.evaluate import control_gates
from chandralign.matching import pair_tiling


def plane(h, w, fill=0.5):
    a = np.full((h, w), fill, np.float32)
    return synth.ImagePlane(array=a, valid_mask=np.ones((h, w), bool),
                            shadow_mask=np.zeros((h, w), bool), gsd_m=1.0, meta=None, geo=None)


def test_grid_partitions_the_window_exactly():
    tiles = pair_tiling.grid(1040, 1100, target=640)
    assert len(tiles) == 4                                  # ceil(1040/640) x ceil(1100/640)
    cover = np.zeros((1040, 1100), int)
    for y0, y1, x0, x1 in tiles:
        cover[y0:y1, x0:x1] += 1
    assert (cover == 1).all(), "source tiles must partition the window: no gaps, no duplicates"


def test_small_window_is_one_tile():
    assert pair_tiling.grid(500, 600, target=640) == [(0, 500, 0, 600)]


def test_matches_are_returned_in_window_coordinates():
    """A fake matcher reports every tile's centre as a match to the same spot in
    the reference crop. After merging, each must land at the true window position,
    which fails if either the source offset or the margin-shifted reference offset
    is dropped."""
    src, ref = plane(1040, 1040), plane(1040, 1040)
    seen = []

    def fake(s, r):
        h, w = s.array.shape
        c = np.array([[w / 2.0, h / 2.0]])
        # the reference crop carries the margin, so the same ground point sits
        # displaced by the crop's own offset relative to the source tile
        dx, dy = s.tile_origin[1] - r.tile_origin[1], s.tile_origin[0] - r.tile_origin[0]
        seen.append((s.array.shape, r.array.shape, s.tile_origin, r.tile_origin))
        return MatchSet(src_pts=c, ref_pts=c + [dx, dy], confidence=np.ones(1, np.float32),
                        method="fake", regime="same_modal_normal", stage="direct")

    ms = pair_tiling.match_tiled(src, ref, fake, target=640, margin=64)
    assert len(ms.src_pts) == 4
    np.testing.assert_allclose(ms.src_pts, ms.ref_pts)       # identity pair -> identity matches
    expected = sorted([(260.0, 260.0), (780.0, 260.0), (260.0, 780.0), (780.0, 780.0)])
    assert sorted(map(tuple, ms.src_pts)) == expected
    # reference crops carry the margin, clamped at the image border
    for s_shape, r_shape, s_org, r_org in seen:
        assert r_shape[0] == min(1040, s_org[0] + s_shape[0] + 64) - max(0, s_org[0] - 64)


def test_tile_count_and_margin_are_recorded():
    src, ref = plane(1040, 1040), plane(1040, 1040)

    def empty(s, r):
        z = np.zeros((0, 2))
        return MatchSet(src_pts=z, ref_pts=z, confidence=np.zeros(0, np.float32),
                        method="fake", regime="same_modal_normal", stage="direct")

    ms = pair_tiling.match_tiled(src, ref, empty, target=640, margin=64)
    assert ms.tiles == 4 and ms.tile_margin_px == 64
    assert ms.src_pts.shape == (0, 2)


def test_refuses_a_pair_that_is_not_the_same_size():
    """Tiling pairs source tile i with reference region i. That is only meaningful
    for a pair already brought into one frame by the coarse stage."""
    with pytest.raises(ValueError, match="pre-aligned"):
        pair_tiling.match_tiled(plane(1040, 1040), plane(520, 520), lambda s, r: None)


def test_perturbation_gate_can_reuse_the_main_registration():
    """G1: the gate's baseline IS the registration being certified. Passing it in
    must skip exactly that one pipeline call, and nothing else about the verdict."""
    calls = []
    shift = np.array([[1, 0, 3.0], [0, 1, 4.0], [0, 0, 1]])

    def pipeline(s, r):
        calls.append(1)
        return control_gates.PipelineRun(True, 100, 100, np.eye(3) - (shift - np.eye(3)))

    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    base = control_gates.PipelineRun(True, 100, 100, np.eye(3))
    g = control_gates.perturbation_gate(pipeline, img, img, base=base)
    assert len(calls) == 1, "only the moved image may be re-matched"
    assert g.passed, g.reason
    g2 = control_gates.perturbation_gate(pipeline, img, img)
    assert len(calls) == 3, "without a base, the gate still runs its own baseline"
