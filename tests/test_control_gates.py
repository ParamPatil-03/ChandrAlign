"""Control gates (CHECK-01..04, 06, 08): each must PASS a good pipeline and FAIL a broken one.

A gate that passes a working pipeline has proved nothing -- a gate that always
says "pass" does that too. So every gate is run against deliberately broken
pipelines here, each broken in the specific way that gate exists to catch.
The good pipeline is the real SIFT + MAGSAC path (CPU, no optional extras).
"""
from __future__ import annotations

import numpy as np
import pytest

from chandralign import synth
from chandralign.contracts import TransformModel
from chandralign.evaluate import control_gates as cg
from chandralign.evaluate import quality


@pytest.fixture(scope="module")
def pair():
    src, ref, _ = synth.make_pair(out_shape=(256, 256), n_craters=30, seed=3, rot_deg=8.0)
    return src, ref


@pytest.fixture(scope="module")
def good():
    return cg.pipeline_from("sift")


# ---------------------------------------------------------------------------
# Broken pipelines, each wrong in one specific way
# ---------------------------------------------------------------------------
def constant_answer(src, ref):
    """Returns the same confident 'registration' whatever it is shown."""
    return cg.PipelineRun(True, 200, 200, np.eye(3), "always identity")


def wrapping(good, fn):
    def run(src, ref):
        r = good(src, ref)
        if r.matrix is not None:
            r = cg.PipelineRun(r.ok, r.matches, r.inliers, fn(np.asarray(r.matrix, float)), r.note)
        return r
    return run


def ignores_y(good):
    """Loses the y component of the translation -- would pass an x-only shift test."""
    def fn(m):
        m = m.copy(); m[1, 2] = 0.0; return m
    return wrapping(good, fn)


def ten_times_translation(good):
    """Right direction, wrong magnitude: moves 50 px for a 5 px shift."""
    def fn(m):
        m = m.copy(); m[:2, 2] *= 10.0; return m
    return wrapping(good, fn)


def half_pixel_off(good):
    """A systematic half-pixel bias -- exactly what the identity floor must catch."""
    def fn(m):
        m = m.copy(); m[0, 2] += 0.5; return m
    return wrapping(good, fn)


def crashes(src, ref):
    raise RuntimeError("simulated matcher crash")


# ---------------------------------------------------------------------------
# The good pipeline passes everything
# ---------------------------------------------------------------------------
def test_the_real_pipeline_passes_every_gate(pair, good):
    src, ref = pair
    rep = cg.run_all(good, src.array, ref.array, src, ref)
    assert rep.all_passed, [(r.name, r.reason) for r in rep.failed]
    assert set(rep.gates) == {"null_constant_grey", "null_random_noise",
                              "perturbation_sensitivity", "identity", "masks_independent"}


def test_perturbation_recovers_the_known_move_closely(pair, good):
    src, ref = pair
    r = cg.perturbation_gate(good, src.array, ref.array, shift=(3, 4))
    assert r.passed and r.detail["error_px"] < 0.3


# ---------------------------------------------------------------------------
# CHECK-01 / 02: null tests catch a pipeline that invents structure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gate", [cg.blank_gate, cg.noise_gate])
def test_null_gates_fail_a_pipeline_that_always_finds_something(pair, gate):
    src, ref = pair
    r = gate(constant_answer, src.array, ref.array)
    assert not r.passed and "invents structure" in r.reason


def test_noise_has_the_references_brightness_statistics(pair, good):
    """So brightness cannot give the noise away -- only missing structure can."""
    src, ref = pair
    seen = {}
    def spy(s, r):
        seen["s"] = s
        return good(s, r)
    cg.noise_gate(spy, src.array, ref.array)
    assert seen["s"].mean() == pytest.approx(ref.array.mean(), abs=0.05)


# ---------------------------------------------------------------------------
# CHECK-03: done_when -- "a deliberately broken pipeline fails it"
# ---------------------------------------------------------------------------
def test_perturbation_fails_a_pipeline_that_returns_the_same_answer(pair):
    """The pipeline that passes the identity gate AND ignores its input: only
    CHECK-03 notices, which is why it exists."""
    src, ref = pair
    assert cg.identity_gate(constant_answer, ref.array).passed          # fooled
    r = cg.perturbation_gate(constant_answer, src.array, ref.array)
    assert not r.passed and r.detail["recovered"] == [0.0, 0.0]         # not fooled


def test_perturbation_fails_a_pipeline_that_ignores_one_axis(pair, good):
    """Why the shift is (3, 4) and not (5, 0): an x-only test passes this pipeline."""
    src, ref = pair
    assert cg.perturbation_gate(ignores_y(good), src.array, ref.array, shift=(5, 0)).passed
    assert not cg.perturbation_gate(ignores_y(good), src.array, ref.array, shift=(3, 4)).passed


def test_perturbation_fails_a_pipeline_that_moves_fifty_for_five(pair, good):
    src, ref = pair
    assert not cg.perturbation_gate(ten_times_translation(good), src.array, ref.array).passed


def test_perturbation_is_not_passed_when_it_cannot_be_evaluated(pair):
    src, ref = pair
    r = cg.perturbation_gate(crashes, src.array, ref.array)
    assert not r.passed and "cannot evaluate" in r.reason


# ---------------------------------------------------------------------------
# CHECK-04: identity floor
# ---------------------------------------------------------------------------
def test_identity_fails_a_half_pixel_bias(pair, good):
    src, ref = pair
    assert cg.identity_gate(good, ref.array).passed
    r = cg.identity_gate(half_pixel_off(good), ref.array)
    assert not r.passed and "RMSE" in r.reason


# ---------------------------------------------------------------------------
# CHECK-06: shared masks
# ---------------------------------------------------------------------------
def _plane(src, valid, shadow):
    return synth.ImagePlane(array=src.array, valid_mask=valid, shadow_mask=shadow,
                            gsd_m=src.gsd_m, meta=src.meta, geo=src.geo)


def test_the_same_mask_object_fails(pair):
    src, ref = pair
    shared = _plane(ref, src.valid_mask, ref.shadow_mask)
    assert not cg.shared_mask_gate(src, shared).passed


def test_a_view_of_the_same_buffer_fails_too(pair):
    """`is` would miss this: a[:] is a different object over the SAME memory."""
    src, ref = pair
    view = _plane(ref, src.valid_mask[:], ref.shadow_mask)
    assert view.valid_mask is not src.valid_mask
    assert not cg.shared_mask_gate(src, view).passed


def test_separate_copies_pass(pair):
    src, ref = pair
    copy = _plane(ref, src.valid_mask.copy(), ref.shadow_mask.copy())
    assert cg.shared_mask_gate(src, copy).passed


def test_a_skipped_mask_check_is_recorded_as_not_passed(pair, good):
    """Leaving the planes out must not make the gate disappear from the record."""
    src, ref = pair
    rep = cg.run_all(good, src.array, ref.array)
    assert rep.gates["masks_independent"] is False
    assert not rep.all_passed


# ---------------------------------------------------------------------------
# Wiring: gates reach the quality verdict, and CHECK-08's guard
# ---------------------------------------------------------------------------
def test_a_failed_gate_rejects_an_otherwise_perfect_result(pair):
    src, ref = pair
    rep = cg.run_all(constant_answer, src.array, ref.array, src, ref)
    v = quality.assess(inlier_count=900, inlier_ratio=0.95, spatial_coverage=0.95,
                       model=TransformModel("affine", np.eye(3)), gates=rep.gates)
    assert v.tier == "REJECTED" and v.limiting_signal == "control_gates"


def test_a_crashing_pipeline_cannot_pass_the_gates(pair):
    src, ref = pair
    assert not cg.run_all(crashes, src.array, ref.array, src, ref).all_passed


class _Result:
    def __init__(self, gates):
        self.gates = gates


def test_require_gates_refuses_a_result_nobody_checked():
    """CHECK-08: an ungated result must raise, not render."""
    with pytest.raises(cg.UngatedResultError):
        cg.require_gates(_Result({}))
    with pytest.raises(cg.UngatedResultError):
        cg.require_gates(_Result(None))


def test_require_gates_lets_a_failed_but_checked_result_through():
    """A failed gate is a finding (shown as REJECTED); only an ABSENT check is refused."""
    cg.require_gates(_Result({"identity": False}))


# ---------------------------------------------------------------------------
# Independent cross-check (the role the RIFT2 benchmark gave RIFT2)
# ---------------------------------------------------------------------------
def _shift(dx, dy):
    m = np.eye(3); m[0, 2], m[1, 2] = dx, dy; return m


def test_crosscheck_flags_a_consistently_wrong_primary():
    g = cg.crosscheck_gate(_shift(4.0, 0.0), np.eye(3), True, (512, 512))
    assert g is not None and not g.passed and g.detail["gap_px"] == pytest.approx(4.0)


def test_crosscheck_passes_when_the_methods_agree():
    g = cg.crosscheck_gate(_shift(0.3, -0.2), np.eye(3), True, (512, 512))
    assert g is not None and g.passed


@pytest.mark.parametrize("checker, accepted", [(None, False), (np.eye(3), False)])
def test_an_unconfident_checker_abstains_rather_than_pass_or_fail(checker, accepted):
    """Neither a pass (no check happened) nor a fail (29 correct results would
    have been rejected at +30 deg lighting, where RIFT2 itself fails)."""
    assert cg.crosscheck_gate(_shift(4.0, 0.0), checker, accepted, (512, 512)) is None


def test_crosscheck_threshold_comes_from_config():
    from chandralign import config
    assert config.get("gates.crosscheck_flag_px") == 2.0
    assert cg.crosscheck_gate(_shift(1.5, 0.0), np.eye(3), True, (512, 512), flag_px=1.0).passed is False
