"""Audit C-04: the independent cross-check runs inside register_bundle (docs/crosscheck_protocol.md).

The five control gates pass a CONSISTENTLY wrong answer by design -- shift the input and the wrong
answer shifts too. A checker from a different family (RIFT2, phase congruency) fails differently.
These tests drive the three outcomes end to end: agree (pass), flag (REJECTED), and no confident
checker (inconclusive: the tier is capped, never passed or failed on the checker's silence).
"""
import numpy as np
import pytest

from chandralign import config, synth
from chandralign.contracts import MatchSet
from chandralign.evaluate import control_gates
from chandralign.matching import classical, rift
from chandralign.pipeline import register_bundle

GATES = ("null_constant_grey", "null_random_noise", "perturbation_sensitivity", "identity", "masks_independent")


@pytest.fixture(scope="module")
def pair():
    return synth.make_pair(out_shape=(512, 512), seed=3, rot_deg=8.0)


def _stub_gates(monkeypatch):
    """Every control gate passes: the cross-check is then the only thing that can object."""
    monkeypatch.setattr(control_gates, "run_all", lambda *a, **k: control_gates.GateReport(
        [control_gates.GateResult(n, True, "stubbed") for n in GATES]))


def _consistent_matches(H, offset, n=600, seed=0):
    """Many mutually consistent matches of the transform H moved by `offset` px: a wrong answer
    that every self-consistency check accepts."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(20, 492, (n, 2))
    r = synth.transform_points(H, s) + np.asarray(offset, float)
    return MatchSet(s, r, np.ones(n, np.float32), "stub", "same_modal_normal", "direct")


def test_a_correct_result_is_confirmed_by_the_independent_method(pair):
    src, ref, H = pair
    b = register_bundle(src, ref, matcher="sift")
    xc = b.result.provenance["crosscheck"]
    assert xc["verdict"] == "agree", xc
    assert b.result.gates.get("independent_crosscheck") is True
    assert b.result.confidence_tier != "REJECTED"


def test_a_consistently_wrong_result_the_gates_pass_is_rejected(pair, monkeypatch):
    src, ref, H = pair
    _stub_gates(monkeypatch)
    monkeypatch.setattr(classical, "match", lambda s, r, detector="sift": _consistent_matches(H, (6.0, 0.0)))
    b = register_bundle(src, ref, matcher="sift")
    assert b.result.provenance["crosscheck"]["verdict"] == "flag"
    assert b.result.gates["independent_crosscheck"] is False
    assert b.result.confidence_tier == "REJECTED" and 12 in b.result.failure_modes


def test_a_checker_with_no_answer_caps_the_tier_and_neither_passes_nor_fails(pair, monkeypatch):
    src, ref, H = pair
    _stub_gates(monkeypatch)
    monkeypatch.setattr(classical, "match", lambda s, r, detector="sift": _consistent_matches(H, (0.0, 0.0)))
    empty = MatchSet(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, np.float32), "rift2", "x", "direct")
    monkeypatch.setattr(rift, "match", lambda *a, **k: empty)
    # the I-08 accuracy signal would also allow HIGH here, so only the cross-check cap can hold it at MEDIUM
    import chandralign.pipeline as pl
    monkeypatch.setattr(pl, "_probe_accuracy", lambda *a, **k: {"n": 300, "p50_px_src": 0.1, "p95_px_src": 0.3})
    b = register_bundle(src, ref, matcher="sift")
    assert b.result.provenance["crosscheck"]["verdict"] == "inconclusive"
    assert "independent_crosscheck" not in b.result.gates
    cap = config.get("gates.crosscheck_inconclusive_cap", "MEDIUM")
    assert b.result.confidence_tier == cap, (b.result.confidence_tier, b.result.notes)


def test_the_switch_turns_it_off(pair, monkeypatch):
    src, ref, H = pair
    monkeypatch.setitem(config.load("default")["gates"], "crosscheck", False)
    b = register_bundle(src, ref, matcher="sift")
    assert "independent_crosscheck" not in b.result.gates
    assert b.result.provenance["crosscheck"]["applied"] is False


def test_on_relief_the_checker_is_compared_with_the_delivered_geometry_not_an_affine(monkeypatch, pair):
    """Amendment 1: on hilly TMC-2 windows no affine is the geometry, and affine-vs-affine flagged all four
    correct windows. The checker's matches must be judged against the geometry actually delivered."""
    src, ref, _ = pair
    rng = np.random.default_rng(1)
    s = rng.uniform(20, 492, (800, 2))
    curve = lambda p: p + np.c_[3.0 * np.sin(p[:, 1] / 80.0), 2.0 * np.cos(p[:, 0] / 90.0)]   # noqa: E731
    checker = MatchSet(s, curve(s), np.ones(len(s), np.float32), "rift2", "x", "direct")
    monkeypatch.setattr(rift, "match", lambda *a, **k: checker)
    monkeypatch.setitem(config.load("default")["estimate"], "reproj_threshold_px", 6.0)   # the curve is one affine's inliers
    primary_affine = np.array([[1, 0, 5.0], [0, 1, 0], [0, 0, 1]])                       # 5 px from any fit to the curve
    g_aff, rec_aff = control_gates.independent_crosscheck(primary_affine, src, ref, "eloftr")
    g_geo, rec_geo = control_gates.independent_crosscheck(primary_affine, src, ref, "eloftr", predict=curve)
    assert g_aff is not None and not g_aff.passed                  # the old statistic: a false alarm
    assert g_geo is not None and g_geo.passed and rec_geo["gap_px"] < 0.5
    assert rec_geo["affine_gap_px"] > 2.0                          # still recorded
