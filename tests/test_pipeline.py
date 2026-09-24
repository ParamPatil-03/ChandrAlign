"""pipeline.fine_stage: the fine stage every registration and every control gate runs.

What these check is the WIRING, the defect class the 2026-09-23 audit found:
modules that pass their own tests but that no pipeline calls.
  1. all stages off == the old match -> robust.estimate chain, exactly
  2. each stage, switched on, really runs and does its job
  3. the control gates run the same fine stage, with the same switches
"""
import numpy as np
import pytest

from chandralign import config
from chandralign.contracts import MatchSet
from chandralign.estimate import models, robust
from chandralign.evaluate import control_gates
from chandralign.io.dem import DemPatch
from chandralign.pipeline import STAGES, fine_stage, stage_flags
from chandralign.refine import uniformity

OFF = {s: False for s in STAGES}
A = np.array([[0.98, -0.05, 12.3], [0.04, 1.01, -7.9], [0, 0, 1.0]])   # the true transform


def matchset(src, ref, conf=None):
    return MatchSet(np.asarray(src, float), np.asarray(ref, float),
                    np.ones(len(src), np.float32) if conf is None else np.asarray(conf, np.float32),
                    "test", "same_modal_normal", "test")


def clustered_pairs(seed=0, n_in=600, n_out=150, shape=(400, 400)):
    """Inliers crowded into one corner plus a thin spread elsewhere, and outliers."""
    rng = np.random.default_rng(seed)
    h, w = shape
    dense = rng.uniform([0, 0], [w * 0.3, h * 0.3], (int(n_in * 0.85), 2))
    sparse = rng.uniform([0, 0], [w, h], (n_in - len(dense), 2))
    src_in = np.vstack([dense, sparse])
    ref_in = src_in @ A[:2, :2].T + A[:2, 2] + rng.normal(0, 0.3, src_in.shape)
    src_out = rng.uniform([0, 0], [w, h], (n_out, 2))
    ref_out = rng.uniform([0, 0], [w, h], (n_out, 2))
    return np.vstack([src_in, src_out]), np.vstack([ref_in, ref_out]), rng.random(n_in + n_out)


def blank(shape=(400, 400)):
    return np.zeros(shape, np.float32)


# ---------------------------------------------------------------------------
# 1. all stages off is the old chain, exactly
# ---------------------------------------------------------------------------
def test_all_stages_off_is_the_old_chain():
    src, ref, conf = clustered_pairs()
    centre = (200.0, 200.0)
    fr = fine_stage(matchset(src, ref, conf), blank(), blank(), centre=centre, flags=OFF)
    old = robust.estimate(src, ref, expected_scale=None, centre=centre)
    assert np.array_equal(fr.first.inlier_mask, old.inlier_mask)
    assert np.allclose(fr.model.matrix, old.model.matrix)
    inl = old.inlier_mask
    assert fr.coverage == uniformity.coverage_of(src[inl], (400, 400), grid=8)
    r = models.residuals(old.model, src[inl], ref[inl])
    assert fr.rmse_px == pytest.approx(float(np.sqrt(np.mean(np.asarray(r) ** 2))))
    assert len(fr.control_src) == old.inlier_count
    assert all(not v["applied"] for v in fr.stages.values())


def test_the_config_default_is_what_runs_when_nothing_is_overridden():
    assert stage_flags() == {s: bool(config.get(f"pipeline.{s}", False)) for s in STAGES}


def test_an_unknown_stage_name_is_refused():
    with pytest.raises(ValueError):
        stage_flags({"subpixle": True})


# ---------------------------------------------------------------------------
# 2. each stage does its job when switched on
# ---------------------------------------------------------------------------
def test_uniformity_thins_the_crowded_corner_and_keeps_coverage():
    src, ref, conf = clustered_pairs()
    off = fine_stage(matchset(src, ref, conf), blank(), blank(), centre=(200, 200), flags=OFF)
    on = fine_stage(matchset(src, ref, conf), blank(), blank(), centre=(200, 200),
                    flags={**OFF, "uniformity": True})
    k = int(config.get("uniformity.top_k_per_cell", 6))
    cy, cx = uniformity._cell_index(on.control_src, (400, 400), 8)
    assert np.bincount(cy * 8 + cx).max() <= k, "a cell holds more than top-k control points"
    assert len(on.control_src) < 0.5 * len(off.control_src)
    assert on.coverage == pytest.approx(off.coverage)      # thinning never empties a cell
    # the tier signals still come from the FIRST estimate, not the thinned set
    assert on.inlier_count == off.inlier_count
    m = np.asarray(on.model.matrix)
    assert np.allclose(m[:2, :2], A[:2, :2], atol=0.01) and np.allclose(m[:2, 2], A[:2, 2], atol=0.1)


def test_delivery_stages_never_change_the_model():
    """Measured on real data: refitting on the thinned points made the known-shift
    error 3x worse (docs/pipeline_stages_protocol.md). The model is the first
    estimate on every inlier, whatever the delivery stages do."""
    src, ref, conf = clustered_pairs()
    off = fine_stage(matchset(src, ref, conf), blank(), blank(), centre=(200, 200), flags=OFF)
    for flags in ({"uniformity": True}, {"subpixel": True}, {"uniformity": True, "subpixel": True}):
        on = fine_stage(matchset(src, ref, conf), blank(), blank(), centre=(200, 200), flags={**OFF, **flags})
        assert np.array_equal(on.model.matrix, off.model.matrix), flags


def _textured_pair(shift=(0.37, -0.62), blk=50, n=160, seed=3):
    """Two block-averaged views of one random texture, `shift` coarse px apart,
    formed exactly (integer native offset), as scripts/verify_subpixel.py does."""
    rng = np.random.default_rng(seed)
    import cv2
    big = cv2.GaussianBlur(rng.random((n * blk + 2 * blk, n * blk + 2 * blk)).astype(np.float32), (0, 0), blk / 2)
    sx, sy = int(round(-shift[0] * blk)), int(round(-shift[1] * blk))
    cut = lambda ox, oy: big[blk + oy:blk + oy + n * blk, blk + ox:blk + ox + n * blk] \
        .reshape(n, blk, n, blk).mean(axis=(1, 3))
    return cut(0, 0), cut(sx, sy)


def test_subpixel_moves_integer_matches_onto_the_true_position():
    shift = np.array([0.37, -0.62])
    a, b = _textured_pair(tuple(shift))
    g = np.arange(30, 130, 10, dtype=float)
    src = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    ref = src + np.round(shift)                    # what an integer-pixel matcher reports
    off = fine_stage(matchset(src, ref), a, b, centre=(80, 80), flags=OFF)
    on = fine_stage(matchset(src, ref), a, b, centre=(80, 80), flags={**OFF, "subpixel": True})
    assert on.stages["subpixel"]["moved"] > 0.9 * len(src)
    # The DELIVERED points: each (ref - src) should be the true shift.
    err_off = np.abs(np.median(off.control_ref - off.control_src, axis=0) - shift).max()
    err_on = np.abs(np.median(on.control_ref - on.control_src, axis=0) - shift).max()
    assert err_off > 0.3                           # integer matches: off by the rounding
    assert err_on < 0.05                           # refined: sub-pixel
    assert np.percentile(np.hypot(*(on.control_ref - on.control_src - shift).T), 90) < 0.1


class _Identity:
    """Ground model: pixel (row, col) -> lat/lon on a 0.001 deg grid near the equator."""
    def pixel_to_latlon(self, rows, cols):
        return 0.1 - np.asarray(rows, float) * 0.001, 23.4 + np.asarray(cols, float) * 0.001


def _ridge_dem():
    """Flat ground on the west half, a steep slope on the east half."""
    lat = np.linspace(0.1, -0.35, 451)
    lon = np.linspace(23.4, 23.85, 451)
    h = np.where(lon[None, :] > 23.6, (lon[None, :] - 23.6) * 30_000.0 * 0.5, 0.0) + 0 * lat[:, None]
    return DemPatch(h, lat, lon, 1000.0, "test", True, ())


def test_geometry_filter_drops_matches_that_join_different_terrain():
    rng = np.random.default_rng(1)
    src = rng.uniform([20, 20], [180, 380], (200, 2))            # all on the flat west half
    ref = src + [0.4, -0.3]
    ref[:20] = rng.uniform([260, 20], [380, 380], (20, 2))        # 20 land on the steep east half
    ms = matchset(src, ref)
    ms.device = "cuda"                 # the matcher attaches this at run time; callers read it
    fr = fine_stage(ms, blank(), blank(), centre=(200, 200),
                    flags={**OFF, "geometry_filter": True}, ground_model=_Identity(), dem=_ridge_dem())
    assert fr.matches.device == "cuda", "the filter dropped a run-time attribute (crashed a real run)"
    st = fr.stages["geometry_filter"]
    assert st["applied"] and st["n_in"] == 200
    assert st["rejected_slope"] == 20 and st["n_kept"] == 180


def test_geometry_filter_without_a_dem_says_so_and_changes_nothing():
    src, ref, conf = clustered_pairs()
    fr = fine_stage(matchset(src, ref, conf), blank(), blank(), centre=(200, 200),
                    flags={**OFF, "geometry_filter": True})
    assert fr.stages["geometry_filter"] == {"applied": False, "reason": "no DEM"}
    assert fr.n_matches == len(fr.matches.src_pts) == len(src)


# ---------------------------------------------------------------------------
# 3. the gates run the same fine stage, with the same switches
# ---------------------------------------------------------------------------
def test_gate_pipeline_passes_its_stage_switches_to_fine_stage(monkeypatch):
    import chandralign.pipeline as pl
    seen = []
    real = pl.fine_stage

    def spy(ms, src_img, ref_img, **kw):
        seen.append(kw.get("flags"))
        return real(ms, src_img, ref_img, **kw)

    monkeypatch.setattr(pl, "fine_stage", spy)
    a, b = _textured_pair((0.0, 0.0), blk=8, n=200, seed=5)
    run = control_gates.pipeline_from("sift", stages={**OFF, "uniformity": True})
    run(a, b)
    assert seen == [{**OFF, "uniformity": True}], "the gate did not run the registration's fine stage"


def test_dense_refine_moves_an_off_model_onto_a_known_shift():
    """The model starts 0.37 / 0.62 px wrong; dense refinement must find the true shift."""
    shift = np.array([0.37, -0.62])
    a, b = _textured_pair(tuple(shift))
    g = np.arange(30, 130, 10, dtype=float)
    src = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    ref = src + np.round(shift)                      # integer matches: the model is off
    off = fine_stage(matchset(src, ref), a, b, centre=(80, 80), flags=OFF)
    on = fine_stage(matchset(src, ref), a, b, centre=(80, 80), flags={**OFF, "dense_refine": True})
    assert on.stages["dense_refine"]["applied"]
    assert np.abs(np.asarray(off.model.matrix)[:2, 2] - shift).max() > 0.3
    assert np.abs(np.asarray(on.model.matrix)[:2, 2] - shift).max() < 0.05
    assert on.inlier_count == off.inlier_count          # grading signals unchanged


def test_tps_beats_affine_on_held_out_points_of_a_relief_distorted_pair():
    """ALIGN-02 done_when: on a synthetic pair bent by smooth relief, TPS fitted on the
    delivered points leaves less error than the affine model on points it never saw --
    and switching TPS on never changes the affine model the gates rely on."""
    rng = np.random.default_rng(7)
    def truth(p):                                          # affine + smooth bending
        q = p @ A[:2, :2].T + A[:2, 2]
        return q + np.c_[3 * np.sin(p[:, 1] / 60.0), 2.5 * np.cos(p[:, 0] / 70.0)]
    src = rng.uniform(0, 400, (3000, 2))
    ref = truth(src) + rng.normal(0, 0.2, src.shape)
    off = fine_stage(matchset(src, ref), blank(), blank(), centre=(200, 200), flags={**OFF, "uniformity": True})
    on = fine_stage(matchset(src, ref), blank(), blank(), centre=(200, 200), flags={**OFF, "uniformity": True, "tps": True})
    assert np.array_equal(on.model.matrix, off.model.matrix)
    assert on.tps is not None and on.stages["tps"]["applied"]
    test = rng.uniform(20, 380, (2000, 2))
    e_aff = np.sqrt(np.mean(np.sum((models.apply(on.model, test) - truth(test)) ** 2, axis=1)))
    e_tps = np.sqrt(np.mean(np.sum((models.apply(on.tps, test) - truth(test)) ** 2, axis=1)))
    assert e_tps < 0.5 * e_aff, (e_tps, e_aff)


def test_refill_fills_an_empty_cell_only_with_points_the_model_agrees_with():
    """ALIGN-05: an empty cell is re-searched locally; a match that disagrees with the
    model (> 2 px) is refused; the model never changes."""
    rng = np.random.default_rng(3)
    src = rng.uniform([0, 0], [400, 400], (2000, 2))
    src = src[~((src[:, 0] < 50) & (src[:, 1] < 50))]          # cell (0, 0) left empty
    ref = src @ A[:2, :2].T + A[:2, 2]
    def rematch(a, b):                                        # crop of cell (0, 0), padded
        good = np.array([[20.0, 20.0], [30.0, 35.0]])
        bad = np.array([[40.0, 10.0]])
        s = np.vstack([good, bad])
        r = s @ A[:2, :2].T + A[:2, 2]
        r[-1] += 9.0                                          # disagrees with the model
        return matchset(s, r)
    flags = {**OFF, "uniformity": True, "refill": True}
    off = fine_stage(matchset(src, ref), blank(), blank(), centre=(200, 200), flags={**OFF, "uniformity": True})
    on = fine_stage(matchset(src, ref), blank(), blank(), centre=(200, 200), flags=flags, rematch=rematch)
    assert np.array_equal(on.model.matrix, off.model.matrix)
    assert on.stages["refill"]["refilled"] == 2 and on.coverage > off.coverage
    assert not any(np.allclose(p, [40.0, 10.0]) for p in on.control_src)


def test_parallax_keeps_the_hill_an_affine_throws_away_and_still_refuses_outliers():
    """ALIGN-08: an oblique view shifts a 300 m hill ~20 px along-track. The affine alone
    drops the hill's matches; the parallax stage keeps them, recovers p, refuses the
    outliers, and leaves the delivered model unchanged."""
    rng = np.random.default_rng(5)
    lat, lon = np.linspace(0.1, -0.35, 451), np.linspace(23.4, 23.85, 451)
    rr, cc = np.meshgrid(np.arange(451), np.arange(451), indexing="ij")
    hill = 300.0 * np.exp(-((rr - 300) ** 2 + (cc - 300) ** 2) / (2 * 50.0 ** 2))
    dem = DemPatch(hill, lat, lon, 1000.0, "test", True, ())
    p = np.array([0.004, -0.07])                                   # px per metre, along-track
    src = rng.uniform([0, 0], [400, 400], (1500, 2))
    h = hill[src[:, 1].round().astype(int), src[:, 0].round().astype(int)]
    ref = src @ A[:2, :2].T + A[:2, 2] + h[:, None] * p + rng.normal(0, 0.3, src.shape)
    bad = rng.random(len(src)) < 0.15
    ref[bad] += rng.uniform(-40, 40, (bad.sum(), 2))
    on_hill = (h > 60) & ~bad
    kw = dict(centre=(200, 200), ground_model=_Identity(), dem=dem)
    off = fine_stage(matchset(src, ref), blank(), blank(), flags=OFF, **kw)
    on = fine_stage(matchset(src, ref), blank(), blank(), flags={**OFF, "parallax": True}, **kw)
    got = np.zeros(len(src), bool)
    for pt in on.control_src:
        got |= np.all(src == pt, axis=1)
    assert off.first.inlier_mask[on_hill].mean() < 0.5                         # the affine loses the hill
    assert got[on_hill].mean() > 0.95 and got[bad].mean() < 0.05
    st = on.stages["parallax"]
    assert st["applied"] and np.allclose(st["p_px_per_m"], p, atol=0.005)
    assert np.array_equal(on.model.matrix, off.model.matrix)


def test_parallax_without_a_dem_says_so():
    fr = fine_stage(matchset(*clustered_pairs()[:2]), blank(), blank(), centre=(200, 200),
                    flags={**OFF, "parallax": True})
    assert fr.stages["parallax"] == {"applied": False, "reason": "no DEM"}


def test_parallax_uses_its_own_dem_when_given_one():
    """A finer parallax DEM must not reach the terrain filter, and vice versa."""
    src, ref = clustered_pairs()[:2]
    fr = fine_stage(matchset(src, ref), blank(), blank(), centre=(200, 200),
                    flags={**OFF, "parallax": True}, ground_model=_Identity(), parallax_dem=_ridge_dem())
    assert fr.stages["parallax"]["applied"]
    assert fr.stages["geometry_filter"] == {"applied": False, "reason": "off (pipeline.geometry_filter)"}
