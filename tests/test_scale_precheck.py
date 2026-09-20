"""GEO-06: the scale pre-check that runs before any matcher is built."""
import itertools
from dataclasses import replace
from pathlib import Path

import pytest

from chandralign.io.instruments import (
    cascade_threshold,
    get_spec,
    load_registry,
    scale_gap,
    scale_precheck,
)
from chandralign.io.pds_label import parse_label

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
PRODUCTS = {
    "OHRC": "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml",
    "TMC2": "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml",
    "IIRS": "ch2_iir_nci_20240523T1600301891_d_img_d18.xml",
}


def meta(name):
    return parse_label(LABELS / PRODUCTS[name])


# ----------------------------------------------------------------------------- the acceptance

def test_ohrc_to_iirs_is_flagged_before_any_matcher_runs():
    """GEO-06 acceptance, on the two real labels."""
    check = scale_precheck(meta("OHRC"), meta("IIRS"))
    assert check.extreme is True
    assert check.route == "cascade"
    assert check.ratio == pytest.approx(97.15 / 0.30, rel=1e-6)
    assert check.source == "label"
    assert "cascade" in check.reason


def test_a_close_pair_is_allowed_through():
    check = scale_precheck("OHRC", "NAC")          # 0.25 vs 0.5 m
    assert check.extreme is False and check.route == "direct"
    assert check.ratio == pytest.approx(2.0)


def test_the_check_is_symmetric():
    a = scale_precheck(meta("OHRC"), meta("IIRS"))
    b = scale_precheck(meta("IIRS"), meta("OHRC"))
    assert a.ratio == pytest.approx(b.ratio) and a.extreme == b.extreme


def test_the_threshold_comes_from_member_bs_config_not_a_constant():
    """If the ablation retunes the threshold, the pre-check must follow it."""
    assert cascade_threshold() == 4.0            # configs/regimes.yaml, extreme_scale rule
    lenient = scale_precheck(meta("OHRC"), meta("TMC2"), threshold=1000.0)
    strict = scale_precheck(meta("OHRC"), meta("TMC2"), threshold=1.1)
    assert lenient.extreme is False and strict.extreme is True


# ----------------------------------------------------------------------------- label vs nominal

def test_label_gsds_are_preferred_and_differ_from_nominal():
    """The registry's nominal figures are design values; our real products are not them."""
    for name, nominal in (("OHRC", 0.25), ("TMC2", 5.0), ("IIRS", 80.0)):
        m = meta(name)
        assert get_spec(name).gsd_m == nominal
        assert m.gsd_m != nominal                              # the label disagrees
        assert scale_precheck(m, "NAC").gsd_a_m == m.gsd_m     # and the label wins
        assert scale_precheck(m, "NAC").source == "mixed"      # one label, one nominal


def test_no_registry_pairing_changes_its_decision_when_label_gsds_are_used():
    """Measured, and recorded because it is the honest result rather than the hoped-for one.

    All three nominal GSDs are 12-21% out, but every one of the 21 pairings sits far
    enough from the threshold that the yes/no answer is unchanged. The label GSD still
    matters: TMC-2 <-> MI is nominally 4.00 against a 4.00 threshold -- a knife-edge --
    and the label value moves it to 4.54, off the boundary.
    """
    real = {n: meta(n).gsd_m for n in PRODUCTS}
    names = list(load_registry())
    for a, b in itertools.combinations(names, 2):
        nominal = scale_precheck(a, b)
        ga, gb = real.get(a, get_spec(a).gsd_m), real.get(b, get_spec(b).gsd_m)
        ratio = max(ga, gb) / min(ga, gb)
        assert (ratio >= nominal.threshold) == nominal.extreme, (a, b, ratio, nominal.ratio)

    assert scale_precheck("TMC2", "MI").ratio == pytest.approx(4.0)     # exactly on the edge
    assert scale_precheck(meta("TMC2"), "MI").ratio == pytest.approx(4.54, abs=0.01)


def test_it_agrees_with_scale_gap_when_both_sides_are_nominal():
    for a, b in itertools.combinations(load_registry(), 2):
        assert scale_precheck(a, b).ratio == pytest.approx(scale_gap(a, b))


# ----------------------------------------------------------------------------- refusals

def test_a_nonsense_gsd_is_refused_rather_than_ratioed():
    broken = replace(meta("OHRC"), gsd_m=0.0)
    with pytest.raises(ValueError, match="positive"):
        scale_precheck(broken, meta("IIRS"))


def test_an_unknown_instrument_is_refused():
    from chandralign.io.instruments import UnknownInstrumentError
    with pytest.raises(UnknownInstrumentError):
        scale_precheck("OHRC", "HUBBLE")


def test_no_pixels_are_read(monkeypatch):
    """The whole point: the decision is made on metadata, before anything is opened."""
    import chandralign.io.pds_raster as pds_raster

    def explode(*args, **kwargs):
        raise AssertionError("scale_precheck must not read pixels")

    monkeypatch.setattr(pds_raster, "read_raster", explode)
    assert scale_precheck(meta("OHRC"), meta("IIRS")).extreme is True
