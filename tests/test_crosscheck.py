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
