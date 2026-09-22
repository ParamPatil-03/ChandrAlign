"""Scale-bridging cascade (MATCH-10).

done_when: "A direct OHRC <-> IIRS request is intercepted and routed via TMC-2;
the chained transform on synthetic data matches the analytic result; per-step
and total error are both reported." Each clause has a test below.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chandralign import synth
from chandralign.contracts import TransformModel
from chandralign.estimate import models
from chandralign.io.pds_label import parse_label
from chandralign.matching import cascade as cc

LABELS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labels"
OHRC = cc.node_from(parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml"), "OHRC")
TMC2 = cc.node_from(parse_label(LABELS / "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml"), "TMC2")
IIRS = cc.node_from(parse_label(LABELS / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml"), "IIRS")


# ---------------------------------------------------------------------------
# done_when, clause 1: OHRC <-> IIRS is intercepted and routed via TMC-2
# ---------------------------------------------------------------------------
def test_ohrc_to_iirs_is_routed_via_tmc2():
    p = cc.plan(OHRC, IIRS, via=[TMC2])
    assert p.route == "cascade"
    assert p.path == ["OHRC", "TMC2", "IIRS"]


def test_the_reason_is_footprint_not_ratio():
    """OHRC is 3.7 km wide: 727 TMC-2 pixels, 37 IIRS pixels. The ratio of both
    steps is ~16x; only the footprint separates the feasible one."""
    to_tmc2, to_iirs = cc.step_between(OHRC, TMC2), cc.step_between(OHRC, IIRS)
    assert to_tmc2.feasible and not to_iirs.feasible
    assert to_iirs.footprint_px < 50 and to_tmc2.footprint_px > 500
    assert "37" in to_iirs.reason


def test_without_an_intermediate_the_request_is_impossible_not_forced_direct():
    """No path is better than a doomed direct attempt -- two teams got 0 inliers."""
    p = cc.plan(OHRC, IIRS, via=[])
    assert p.route == "impossible" and not p.steps


def test_a_feasible_direct_step_is_preferred_to_a_detour():
    p = cc.plan(OHRC, TMC2, via=[IIRS])
    assert p.route == "direct" and p.path == ["OHRC", "TMC2"]


def test_the_threshold_is_the_measured_one_and_it_decides():
    assert cc.min_footprint_px() == 112          # scripts/measure_cascade_footprint.py
    assert cc.plan(OHRC, IIRS, via=[], threshold=30).route == "direct"


def test_step_feasibility_does_not_depend_on_direction():
    a, b = cc.step_between(OHRC, TMC2), cc.step_between(TMC2, OHRC)
    assert a.feasible == b.feasible and a.footprint_px == b.footprint_px


# ---------------------------------------------------------------------------
# done_when, clause 3: per-step and total error, propagated correctly
# ---------------------------------------------------------------------------
def _affine(scale: float, tx: float = 0.0, ty: float = 0.0) -> TransformModel:
    m = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]], float)
    return TransformModel(kind="affine", matrix=m)


def test_errors_are_carried_through_later_steps_and_added_in_quadrature():
    """0.5 px in B, then a 4x reduction to C: that step is 0.125 C px. Plus the
    second step's own 0.3 C px, independent: sqrt(0.125^2 + 0.3^2)."""
    s1 = cc.StepResult("A", "B", _affine(0.25, 3, 4), rmse_px=0.5, ref_pixel_m=4.0)
    s2 = cc.StepResult("B", "C", _affine(0.25, 1, 2), rmse_px=0.3, ref_pixel_m=16.0)
    ch = cc.chain([s1, s2])
    assert [s["step"] for s in ch.per_step] == ["A -> B", "B -> C"]
    assert ch.per_step[0]["rmse_px_final"] == pytest.approx(0.125)
    assert ch.total_rmse_px == pytest.approx(math.hypot(0.125, 0.3), abs=1e-4)
    assert ch.total_rmse_m == pytest.approx(math.hypot(0.125, 0.3) * 16.0, abs=1e-3)


def test_the_chain_composes_first_step_first():
    s1 = cc.StepResult("A", "B", _affine(0.25, 3, 4), 0.1, 4.0)
    s2 = cc.StepResult("B", "C", _affine(0.5, 1, 2), 0.1, 16.0)
    p = np.array([[10.0, 20.0]])
    expect = models.apply(s2.model, models.apply(s1.model, p))
    assert np.allclose(models.apply(cc.chain([s1, s2]).model, p), expect)


def test_a_failed_step_ends_the_chain():
    ok = cc.StepResult("A", "B", _affine(0.25), 0.1, 4.0)
    with pytest.raises(ValueError, match="never bridged"):
        cc.chain([ok, None])


def test_steps_that_do_not_connect_are_refused():
    s1 = cc.StepResult("A", "B", _affine(0.25), 0.1, 4.0)
    s2 = cc.StepResult("X", "C", _affine(0.25), 0.1, 16.0)
    with pytest.raises(ValueError, match="do not connect"):
        cc.chain([s1, s2])


# ---------------------------------------------------------------------------
# done_when, clause 2: the chain on synthetic images matches the analytic result
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world():
    """Three images cut from one rendered world by block averaging at integer
    offsets, so every transform between them is EXACT and known by hand.

      C  the whole 3072 m world at 16 m, under a DIFFERENT sun (192 px)
      B  a 2048 m crop at 4 m    -> spans 128 C px   (feasible)
      A  a  512 m crop at 1 m    -> spans 128 B px   (feasible), 32 C px (NOT)

    B's corner (row 600) is deliberately NOT on C's 16 m grid, so the B -> C
    step has a half-pixel fractional truth -- the hardest case for sub-pixel
    estimation, not a convenient one.
    """
    dem = synth.lunar_dem((3072, 3072), 1.0, 700, 7)
    w1 = synth.render(dem, 1.0, 135.0, 40.0, seed=1)[0]
    w2 = synth.render(dem, 1.0, 150.0, 36.0, seed=2)[0]
    BX, BY = 400, 600
    AX, AY = BX + 900, BY + 700
    return {
        "A": w1[AY:AY + 512, AX:AX + 512],
        "B": cc.block_average(w1[BY:BY + 2048, BX:BX + 2048], 4),
        "C": cc.block_average(w2, 16),
        "A_origin": (AX, AY),
    }


def test_the_synthetic_layout_forces_a_cascade():
    p = cc.plan(cc.Node("A", 1, 1, 512, 512), cc.Node("C", 16, 16, 3072, 3072),
                via=[cc.Node("B", 4, 4, 2048, 2048)])
    assert p.route == "cascade" and p.path == ["A", "B", "C"]


def test_the_chained_transform_matches_the_analytic_one(world):
    s1 = cc.register_step_dense(world["A"], world["B"], 4, src="A", ref="B", ref_pixel_m=4.0)
    s2 = cc.register_step_dense(world["B"], world["C"], 4, src="B", ref="C", ref_pixel_m=16.0)
    assert s1 is not None and s2 is not None
    ch = cc.chain([s1, s2], at=(256, 256))

    g = np.linspace(0, 511, 12)
    gx, gy = np.meshgrid(g, g)
    pts = np.c_[gx.ravel(), gy.ravel()]
    # A px -> world (1 m, origin A_origin) -> C px (16 m blocks from the world origin)
    truth = (pts + np.array(world["A_origin"]) + 0.5) / 16.0 - 0.5
    err = np.hypot(*(models.apply(ch.model, pts) - truth).T)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    assert rmse < 0.3, f"chained A->C is {rmse:.3f} C px from the analytic answer"

    # per-step and total error are REPORTED, finite, and not an under-statement
    assert len(ch.per_step) == 2
    assert all(np.isfinite(s["rmse_px_final"]) for s in ch.per_step)
    assert np.isfinite(ch.total_rmse_px) and ch.total_rmse_px >= rmse * 0.5


def test_the_dense_step_refuses_an_ambiguous_match(world):
    """Noise has no structure to find: the step must return None, not a guess."""
    rng = np.random.default_rng(0)
    noise = rng.random((512, 512)).astype(np.float32)
    assert cc.register_step_dense(noise, world["B"], 4, src="N", ref="B", ref_pixel_m=4.0) is None


def test_the_ambiguity_check_itself_can_refuse_a_step(world, monkeypatch):
    """Found by scripts/sabotage.py: the noise test above passed for the wrong
    reason -- with the z check deleted, noise was still rejected by the later
    sub-pixel stage. Here only the z check can say no: a GOOD match, with the
    bar raised out of reach, must be refused if (and only if) the check is wired in."""
    real_get = cc.config.get
    monkeypatch.setattr(cc.config, "get",
                        lambda key, default=None, **kw: 1e9 if key == "cascade.min_z" else real_get(key, default, **kw))
    assert cc.register_step_dense(world["A"], world["B"], 4, src="A", ref="B", ref_pixel_m=4.0) is None
