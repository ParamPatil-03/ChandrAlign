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


def _hill_case(seed=5):
    """A world whose heights are taken at the SOURCE point (the stage's earlier convention); the
    ground-point world is test_parallax_height_at_ref_fits_a_ground_height_world_and_inverts_directly."""
    rng = np.random.default_rng(seed)
    lat, lon = np.linspace(0.1, -0.35, 451), np.linspace(23.4, 23.85, 451)
    rr, cc = np.meshgrid(np.arange(451), np.arange(451), indexing="ij")
    hill = 300.0 * np.exp(-((rr - 300) ** 2 + (cc - 300) ** 2) / (2 * 50.0 ** 2))
    dem = DemPatch(hill, lat, lon, 1000.0, "test", True, ())
    p = np.array([0.004, -0.07])
    src = rng.uniform([0, 0], [400, 400], (1500, 2))
    h = dem.sample(*_Identity().pixel_to_latlon(src[:, 1], src[:, 0]))
    ref = src @ A[:2, :2].T + A[:2, 2] + h[:, None] * p + rng.normal(0, 0.3, src.shape)
    return dem, p, src, h, ref


def test_the_parallax_model_is_delivered_and_predicts_the_hill():
    """ALIGN-08 as an output: FineResult.parallax, applied with heights, lands on the hill's
    true positions; the affine alone misses them by the parallax."""
    dem, p, src, h, ref = _hill_case()
    fr = fine_stage(matchset(src, ref), blank(), blank(), centre=(200, 200), flags={**OFF, "parallax": True},
                    ground_model=_Identity(), dem=dem, parallax_height_at="src")   # _hill_case is a src-height world
    m = fr.parallax
    assert m is not None
    hill = h > 100
    heights_at = lambda q: dem.sample(*_Identity().pixel_to_latlon(q[:, 1], q[:, 0]))  # noqa: E731
    err_p = np.hypot(*(m.predict(src[hill], heights_at) - ref[hill]).T)
    err_a = np.hypot(*(models.apply(fr.model, src[hill]) - ref[hill]).T)
    assert np.median(err_p) < 0.6 and np.median(err_a) > 5.0
    with pytest.raises(ValueError):
        m.apply(src[:3], None)                                    # heights are not optional


def test_parallax_source_map_inverts_the_model():
    dem, p, src, h, ref = _hill_case()
    fr = fine_stage(matchset(src, ref), blank(), blank(), centre=(200, 200), flags={**OFF, "parallax": True},
                    ground_model=_Identity(), dem=dem, parallax_height_at="src")   # _hill_case is a src-height world
    heights_at = lambda q: dem.sample(*_Identity().pixel_to_latlon(q[:, 1], q[:, 0]))  # noqa: E731
    mx, my = models.parallax_source_map(fr.parallax, heights_at, (400, 400), step=4)
    ys, xs = np.mgrid[20:380:37, 20:380:37]
    s = np.c_[mx[ys, xs].ravel(), my[ys, xs].ravel()]
    back = fr.parallax.predict(s, heights_at)                    # in the model's own height convention
    assert np.abs(back - np.c_[xs.ravel(), ys.ravel()]).max() < 0.1


def test_parallax_height_at_ref_fits_a_ground_height_world_and_inverts_directly():
    """docs/parallax_height_protocol.md: with h taken at the REFERENCE (ground) point, the stage
    recovers p, predict() lands on the true positions, and the source map is the direct RPC form."""
    rng = np.random.default_rng(7)
    lat, lon = np.linspace(0.1, -0.35, 451), np.linspace(23.4, 23.85, 451)
    rr, cc = np.meshgrid(np.arange(451), np.arange(451), indexing="ij")
    hill = 300.0 * np.exp(-((rr - 300) ** 2 + (cc - 300) ** 2) / (2 * 50.0 ** 2))
    dem = DemPatch(hill, lat, lon, 1000.0, "test", True, ())
    p = np.array([0.004, -0.07])
    heights_at = lambda q: dem.sample(*_Identity().pixel_to_latlon(q[:, 1], q[:, 0]))  # noqa: E731
    ref = rng.uniform([20, 20], [380, 380], (1500, 2))                 # ground points (TC is ortho)
    Ai = np.linalg.inv(A)
    src = (np.c_[ref - (heights_at(ref) - 0.0)[:, None] * p, np.ones(len(ref))] @ Ai.T)[:, :2]
    ref_obs = ref + rng.normal(0, 0.2, ref.shape)
    fr = fine_stage(matchset(src, ref_obs), blank(), blank(), centre=(200, 200), flags={**OFF, "parallax": True},
                    ground_model=_Identity(), dem=dem, parallax_height_at="ref")
    m = fr.parallax
    assert m.height_at == "ref" and np.allclose(m.p_px_per_m, p, atol=0.003)
    assert np.median(np.hypot(*(m.predict(src, heights_at) - ref).T)) < 0.3
    mx, my = models.parallax_source_map(m, heights_at, (400, 400), step=4)
    ys, xs = np.mgrid[40:360:41, 40:360:41]
    s = np.c_[mx[ys, xs].ravel(), my[ys, xs].ravel()]
    assert np.abs(m.predict(s, heights_at) - np.c_[xs.ravel(), ys.ravel()]).max() < 0.1


def test_a_registration_of_synthetic_planes_is_labelled_synthetic():
    """Audit 2026-09-26 C-08: register_bundle hard-coded source="measured" (rule H5)."""
    from chandralign import synth
    from chandralign.pipeline import register_bundle
    src, ref, _ = synth.make_pair(out_shape=(256, 256), shift=(3.4, -2.2), seed=7, n_craters=35, shadows=False)
    assert register_bundle(src, ref, matcher="sift").result.metrics.source == "synthetic"
# ---------------------------------------------------------------------------
# I-02 (audit 2026-09-26): the estimator's refusal must not be overridden
# ---------------------------------------------------------------------------
def _thin_pairs(seed=5, n_true=10, n_rand=14):
    """10 true matches + 14 random ones: an estimate below estimate.min_inliers (12)."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(64, n_true, replace=False)            # one per 50 px cell: coverage 10/64 >= LOW's 0.15
    s_true = np.c_[cells % 8, cells // 8] * 50.0 + rng.uniform(10, 40, (n_true, 2))
    r_true = s_true @ A[:2, :2].T + A[:2, 2]
    s_rand, r_rand = rng.uniform(20, 380, (n_rand, 2)), rng.uniform(20, 380, (n_rand, 2))
    return np.vstack([s_true, s_rand]), np.vstack([r_true, r_rand])


def test_a_thin_estimate_the_estimator_refused_is_not_accepted_by_the_fine_stage():
    src, ref = _thin_pairs()
    first = robust.estimate(src, ref, centre=(200.0, 200.0))
    assert first.model is not None and not first.ok, "the case must be one the estimator refuses"
    fr = fine_stage(matchset(src, ref), blank(), blank(), centre=(200.0, 200.0), flags=OFF)
    assert not fr.ok
    assert any("not trustworthy" in n for n in fr.notes + first.notes)


def test_a_thin_estimate_the_estimator_refused_is_rejected_end_to_end(monkeypatch):
    from chandralign import synth
    from chandralign.matching import classical
    from chandralign.pipeline import register
    src, ref = _thin_pairs()
    monkeypatch.setattr(classical, "match", lambda s, r, detector="sift": matchset(src, ref))
    # every control gate passes, so the thin estimate is the ONLY thing that can reject it
    names = ("null_constant_grey", "null_random_noise", "perturbation_sensitivity", "identity", "masks_independent")
    monkeypatch.setattr(control_gates, "run_all", lambda *a, **k: control_gates.GateReport(
        [control_gates.GateResult(n, True, "stubbed") for n in names]))
    img = np.random.default_rng(0).random((400, 400)).astype(np.float32)
    p = lambda a: synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool),       # noqa: E731
                                   shadow_mask=np.zeros(a.shape, bool), gsd_m=5.0, meta=None, geo=None)
    r = register(p(img), p(img.copy()), matcher="sift")
    assert r.confidence_tier == "REJECTED", (r.confidence_tier, r.notes)
    assert r.failure_modes


# ---------------------------------------------------------------------------
# C-03 / I-01 / G-02 (audit 2026-09-26): the delivered geometry and its reported accuracy
# ---------------------------------------------------------------------------
def _field_pairs(nonrigid, seed=2, n=900, shape=(512, 512), noise=0.05):
    """Matches of a known map (affine, plus a smooth ripple if non-rigid), with point noise."""
    rng = np.random.default_rng(seed)
    src = rng.uniform(8, shape[0] - 8, (n, 2))

    def truth(s):
        r = s @ A[:2, :2].T + A[:2, 2]
        if nonrigid:
            r = r + 1.2 * np.c_[np.sin(2 * np.pi * s[:, 1] / 300.0), np.cos(2 * np.pi * s[:, 0] / 390.0)]
        return r
    return src, truth(src) + rng.normal(0, noise, (n, 2)), truth


def test_a_rigid_pair_is_delivered_as_an_affine_and_a_rippled_one_as_a_tps():
    for nonrigid, want in ((False, "affine"), (True, "tps")):
        src, ref, truth = _field_pairs(nonrigid)
        fr = fine_stage(matchset(src, ref), blank((512, 512)), blank((512, 512)), centre=(256, 256),
                        flags={**OFF, "model_selection": True, "tps": True})
        assert fr.geometry == want, (nonrigid, fr.stages["model_selection"])
        from chandralign.pipeline import delivered_geometry
        name, m = delivered_geometry(fr)
        g = np.stack(np.meshgrid(np.arange(40, 470, 20.0), np.arange(40, 470, 20.0)), -1).reshape(-1, 2)
        err = np.hypot(*(models.apply(m, g) - truth(g)).T)
        assert np.sqrt(np.mean(err ** 2)) < 0.05, (want, np.sqrt(np.mean(err ** 2)))


def test_the_reported_rmse_describes_the_delivered_geometry_not_the_affine_fit():
    """The old figure (the affine's residual on its own inliers) read ~1 px on a rippled pair whose
    delivered TPS is ~0.05 px wrong. rmse must now be a check-point bound on the DELIVERED model."""
    src, ref, truth = _field_pairs(True)
    fr = fine_stage(matchset(src, ref), blank((512, 512)), blank((512, 512)), centre=(256, 256),
                    flags={**OFF, "model_selection": True, "tps": True})
    a = fr.accuracy
    assert a["model"] == "tps" and a["fit_residual_px"] > 0.5            # the old number, kept as such
    assert a["checkpoint_rmse_px_ref"] < 0.2                              # the new one
    assert a["geometry_error_lower_px_ref"] <= a["checkpoint_rmse_px_ref"]


def test_the_gates_skip_model_selection_but_keep_every_other_switch(monkeypatch):
    seen = {}
    import chandralign.pipeline as pl
    real = pl.fine_stage

    def spy(*a, **k):
        seen.update(k.get("flags") or {})
        return real(*a, **k)
    monkeypatch.setattr(pl, "fine_stage", spy)
    src, ref, _ = _field_pairs(False, n=200)
    img = np.random.default_rng(0).random((256, 256)).astype(np.float32)
    control_gates.pipeline_from("sift", stages={"uniformity": False})(img, img)
    assert seen.get("model_selection") is False and seen.get("uniformity") is False


def test_refinement_improves_points_on_a_rotated_and_scaled_pair():
    """Audit C-02: axis-aligned patches cannot refine across rotation + scale (the shipped
    refinement made points WORSE there); patches warped through the model's Jacobian can."""
    import cv2
    rng = np.random.default_rng(4)
    tex = cv2.GaussianBlur(rng.random((1200, 1200)).astype(np.float32), (0, 0), 2.5)
    th, k = np.deg2rad(10.0), 0.8
    M = np.array([[k * np.cos(th), -k * np.sin(th), 60.0], [k * np.sin(th), k * np.cos(th), -40.0]])
    M[:, 2] += [0.37, -0.62]
    src = tex[100:700, 100:700].copy()
    ref = cv2.warpAffine(src, M, (600, 600), flags=cv2.INTER_LANCZOS4)
    g = np.arange(120, 480, 30, dtype=float)
    s = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    true = s @ M[:, :2].T + M[:, 2]
    keep = (true > 40).all(1) & (true < 560).all(1)
    s, true = s[keep], true[keep]
    noisy = true + rng.uniform(-0.4, 0.4, true.shape)            # a matcher's imprecision
    off = fine_stage(matchset(s, noisy), src, ref, centre=(300, 300), flags=OFF)
    on = fine_stage(matchset(s, noisy), src, ref, centre=(300, 300), flags={**OFF, "subpixel": True})
    e_off = np.hypot(*(off.control_ref - true).T)
    e_on = np.hypot(*(on.control_ref - true).T)
    assert np.median(e_on) < 0.5 * np.median(e_off) and np.percentile(e_on, 95) < np.percentile(e_off, 95)


def test_the_delivered_affine_is_not_dragged_by_inliers_a_3px_threshold_let_through():
    """G-02: the geometry is fitted with Huber IRLS. 10% of 'inliers' 2.5 px off in one direction
    (under the 3 px RANSAC threshold) would shift a plain least-squares affine by ~0.25 px."""
    from chandralign.estimate import selection
    rng = np.random.default_rng(9)
    src = rng.uniform(0, 500, (400, 2))
    ref = src @ A[:2, :2].T + A[:2, 2] + rng.normal(0, 0.05, src.shape)
    ref[:40] += [2.5, 0.0]
    fit = selection.fit_affine_robust(src, ref)
    g = rng.uniform(0, 500, (200, 2))
    err = np.hypot(*(models.apply(fit, g) - (g @ A[:2, :2].T + A[:2, 2])).T)
    assert np.sqrt(np.mean(err ** 2)) < 0.05


class _Shifted(_Identity):
    """A reference frame 250 columns to the east of the source frame: the same ground is 250 px right."""
    def pixel_to_latlon(self, rows, cols):
        return super().pixel_to_latlon(rows, np.asarray(cols, float) - 250.0)


def test_reference_points_use_the_reference_ground_model_when_the_frames_differ():
    """Audit I-10: with one ground model for both, reference points were mapped through the SOURCE
    model. Here the reference frame is 250 px east of the source: through the source model every
    reference point lands on the steep east half and is thrown away as 'different terrain'."""
    rng = np.random.default_rng(1)
    src = rng.uniform([20, 20], [180, 380], (200, 2))            # flat west half
    ref = src + [250.4, -0.3]                                     # the same ground, in the shifted frame
    ms = matchset(src, ref)
    one = fine_stage(ms, blank(), blank(), centre=(200, 200), flags={**OFF, "geometry_filter": True},
                     ground_model=_Identity(), dem=_ridge_dem())
    two = fine_stage(ms, blank(), blank(), centre=(200, 200), flags={**OFF, "geometry_filter": True},
                     ground_model=_Identity(), ref_ground_model=_Shifted(), dem=_ridge_dem())
    assert one.stages["geometry_filter"]["n_kept"] < 50               # the old, wrong mapping
    assert two.stages["geometry_filter"]["n_kept"] == 200             # same ground, all kept


def test_parallax_plus_residual_tps_is_delivered_where_the_dem_explains_only_part_of_the_geometry():
    """G-06: relief the DEM predicts (a hill, parallax) PLUS a ripple it cannot (DEM error, attitude drift).
    Neither parent is right; the composite should be chosen and beat both on the truth."""
    rng = np.random.default_rng(7)
    lat, lon = np.linspace(0.1, -0.35, 451), np.linspace(23.4, 23.85, 451)
    rr, cc = np.meshgrid(np.arange(451), np.arange(451), indexing="ij")
    dem = DemPatch(300.0 * np.exp(-((rr - 250) ** 2 + (cc - 250) ** 2) / (2 * 60.0 ** 2)), lat, lon, 1000.0,
                   "test", True, ())
    p = np.array([0.004, -0.07])
    hts = lambda q: dem.sample(*_Identity().pixel_to_latlon(q[:, 1], q[:, 0]))      # noqa: E731
    ripple = lambda s: 0.8 * np.c_[np.sin(s[:, 1] / 45.0), np.cos(s[:, 0] / 55.0)]   # noqa: E731

    def truth(s):
        return s @ A[:2, :2].T + A[:2, 2] + hts(s)[:, None] * p + ripple(s)
    src = rng.uniform(10, 440, (2500, 2))
    ref = truth(src) + rng.normal(0, 0.05, src.shape)
    # the composite is OFF by default since G-06 (docs/tmc2_tail_protocol.md); its code path is still pinned here
    # ... and at the 1000-point fit set it was designed under: at the adopted 3000 points a plain TPS fits this
    # world within 1% of the composite (G-06 step 2), so the composite no longer wins its 5% margin here
    geo_cfg = config.load("default")["geometry"]
    saved = dict(geo_cfg)
    geo_cfg.update(parallax_tps=True, max_fit_points=1000)
    try:
        fr = fine_stage(matchset(src, ref), blank((451, 451)), blank((451, 451)), centre=(225, 225),
                        flags={**OFF, "parallax": True, "model_selection": True, "tps": True},
                        ground_model=_Identity(), dem=dem, parallax_height_at="src")
    finally:
        geo_cfg.clear(); geo_cfg.update(saved)
    sel = fr.stages["model_selection"]
    assert fr.geometry == "parallax_tps", sel["notes"]
    from chandralign.pipeline import delivered_geometry
    g = rng.uniform(40, 410, (400, 2))
    err = lambda pred: float(np.sqrt(np.mean(np.sum((pred - truth(g)) ** 2, axis=1))))   # noqa: E731
    _, m = delivered_geometry(fr)
    e_pt = err(m.predict(g, hts))
    assert e_pt < 0.1, e_pt
    assert e_pt < 0.5 * sel["candidates"]["parallax"]["checkpoint_rmse_px"]


def test_the_parallax_tps_source_map_inverts_the_model():
    from chandralign.estimate import models as m_
    from chandralign.estimate.selection import fit_tps_robust
    rng = np.random.default_rng(3)
    dem, p, src, h, ref = _hill_case()
    par = m_.ParallaxModel(matrix=A.copy(), p_px_per_m=tuple(p), h0_m=0.0, height_at="src")
    s = rng.uniform(0, 400, (400, 2))
    res = fit_tps_robust(s, 0.6 * np.c_[np.sin(s[:, 1] / 50.0), np.cos(s[:, 0] / 60.0)], 1.0)
    model = m_.ParallaxTPSModel(par, res, "src")
    hts = lambda q: dem.sample(*_Identity().pixel_to_latlon(q[:, 1], q[:, 0]))      # noqa: E731
    mx, my = m_.parallax_tps_source_map(model, hts, (400, 400), step=4)
    r = np.c_[rng.uniform(60, 340, 200), rng.uniform(60, 340, 200)]
    s_back = np.c_[mx[r[:, 1].astype(int), r[:, 0].astype(int)], my[r[:, 1].astype(int), r[:, 0].astype(int)]]
    r_int = np.floor(r)
    assert np.percentile(np.hypot(*(model.predict(s_back, hts) - r_int).T), 95) < 0.1
