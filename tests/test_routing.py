"""Integration tests: does the pipeline actually FOLLOW the routing decision? (MATCH-09)

Until matching/routing.py, `regime.select` had no caller outside tests and
`default_matcher` was a config value no pipeline read -- the one real
registration script named its matcher directly. 500 unit tests passed the whole
time, because unit tests of the selector cannot notice that nothing calls it.

So these tests check the WIRING, not the selector's internals:
  1. each defined regime resolves to the intended matcher
  2. the conditions routing must NOT act on (unvalidated illumination bands)
     do not change the matcher
  3. the real entry point goes through routing, and cannot silently bypass it
  4. the quality-tier thresholds are ordered, so a tier cannot be granted below
     the measured accept/reject line
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from chandralign import config
from chandralign.evaluate import quality
from chandralign.io.pds_label import parse_label
from chandralign.matching import licence, regime, routing

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
TMC2 = parse_label(LABELS / "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml")


def with_sun(meta, azimuth_deg, incidence_deg=30.0):
    return dataclasses.replace(meta, sub_solar_azimuth_deg=azimuth_deg,
                               solar_incidence_deg=incidence_deg)


# ---------------------------------------------------------------------------
# 1. every defined regime resolves to the intended matcher
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("src,ref", [("TMC2", "TC"), ("OHRC", "NAC")])
def test_same_modality_pair_gets_the_default_matcher(src, ref):
    c = routing.choose(src, ref)
    assert c.route == "direct"
    assert c.model_name == config.load("regimes")["default_matcher"] == "eloftr"
    assert c.fine_stage_options == {}, "the default path is not tiled"


def test_cross_modal_pair_gets_tiled_xoftr():
    """IIRS (hyperspectral) <-> WAC (multiband): cross-modal, and only ~1.25x
    apart in scale, so it is a DIRECT cross-modal pair rather than a cascade."""
    c = routing.choose("IIRS", "WAC")
    assert c.route == "direct" and c.regime == "cross_modal"
    assert c.model_name == "xoftr"
    assert c.fine_stage_options == {"tile_px": 640}


@pytest.mark.parametrize("src,ref", [("OHRC", "IIRS"), ("TMC2", "NAC")])
def test_extreme_scale_gap_goes_to_the_cascade_with_no_direct_matcher(src, ref):
    """A matcher name here would invite a caller to match directly across a gap
    measured to be infeasible (OHRC -> IIRS: a 37 px footprint)."""
    c = routing.choose(src, ref)
    assert c.route == "cascade"
    assert c.model_name is None


# ---------------------------------------------------------------------------
# 2. what routing must NOT act on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d_azimuth", [0.0, 90.0, 180.0])
def test_illumination_never_switches_the_matcher(d_azimuth):
    """The illumination bands are validated on real data for INCIDENCE only, not
    for sun azimuth. Routing on them would act on an unvalidated claim, so a
    same-modality pair gets the default whatever the sun does -- including the
    90 deg band the synthetic sweep calls hardest. The expectation flag carries
    the uncertainty instead."""
    c = routing.choose(with_sun(TMC2, 0.0), with_sun(TMC2, d_azimuth))
    assert c.model_name == "eloftr"
    assert c.model_name != "xoftr"


def test_unknown_sun_geometry_does_not_select_xoftr():
    """Instrument names carry no sun at all: expectation 'unknown', matcher default."""
    c = routing.choose("TMC2", "TC")
    assert c.expectation == "unknown"
    assert c.model_name == "eloftr"


def test_routing_never_upgrades_the_expectation():
    for src, ref in [("TMC2", "TC"), ("IIRS", "WAC"), ("OHRC", "IIRS")]:
        assert routing.choose(src, ref).expectation == regime.select(src, ref).expectation


def test_routed_matchers_are_licence_clean():
    cfg = config.load("regimes")
    for name in (cfg["default_matcher"], cfg["cross_modal_matcher"]):
        assert not licence.is_restricted(name), name


# ---------------------------------------------------------------------------
# 3. the real entry point goes THROUGH routing
# ---------------------------------------------------------------------------
@pytest.fixture
def script():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import register_tmc2_tc
        yield register_tmc2_tc
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def test_entry_point_asks_routing_when_no_matcher_is_given(script, monkeypatch):
    calls = []
    real = routing.choose

    def spy(src, ref):
        calls.append((src, ref))
        return real(src, ref)

    monkeypatch.setattr(script.routing, "choose", spy)
    name, opts, prov = script.resolve_matcher(None)
    assert calls == [("TMC2", "TC")], "the entry point did not consult routing"
    assert name == "eloftr" and prov["chosen_by"] == "matching.routing.choose"


def test_entry_point_obeys_what_routing_returns(script, monkeypatch):
    """Not merely that routing is called: that its ANSWER is used. A resolver
    that called routing and then ran a hard-coded name would pass the spy test."""
    fake = routing.MatcherChoice(route="direct", model_name="sentinel-matcher",
                                 fine_stage_options={"tile_px": 123})
    monkeypatch.setattr(script.routing, "choose", lambda s, r: fake)
    name, opts, _ = script.resolve_matcher(None)
    assert name == "sentinel-matcher" and opts == {"tile_px": 123}


def test_explicit_override_is_recorded_as_an_override(script):
    name, _, prov = script.resolve_matcher("aliked-lightglue")
    assert name == "aliked-lightglue"
    assert prov["chosen_by"] == "--matcher override", \
        "a result must never claim routing chose a matcher the user forced"


def test_main_resolves_the_matcher_before_doing_any_work(script, monkeypatch):
    """main() must route through resolve_matcher. Proven by making it raise:
    if main bypassed it, main would go on to load data instead."""
    class Resolved(Exception):
        pass

    def boom(explicit):
        raise Resolved(explicit)

    monkeypatch.setattr(script, "resolve_matcher", boom)
    monkeypatch.setattr(sys, "argv", ["register_tmc2_tc.py"])
    with pytest.raises(Resolved) as e:
        script.main()
    assert e.value.args == (None,), "--matcher must default to None so routing decides"


# ---------------------------------------------------------------------------
# 4. quality tiers are ordered
# ---------------------------------------------------------------------------
def test_tier_thresholds_are_ordered_high_medium_low():
    """medium's inlier-ratio bar was 0.25 against low's 0.325. Because tiers are
    checked high -> medium -> low, a ratio of 0.30 was graded MEDIUM although it
    sits under the measured accept/reject line."""
    tiers = config.get("tiers")
    for key in ("min_inliers", "min_inlier_ratio", "min_coverage"):
        h, m, lo = tiers["high"][key], tiers["medium"][key], tiers["low"][key]
        assert h >= m >= lo, f"{key}: high {h} >= medium {m} >= low {lo} violated"


def test_a_ratio_below_the_measured_line_is_not_graded_medium():
    t = config.get("tiers")
    below = t["low"]["min_inlier_ratio"] - 0.02
    assert quality._tier_for(below, t, "min_inlier_ratio") == "REJECTED"


def test_same_modality_pairs_carry_the_measured_fallback():
    """docs/illumination_fix_protocol.md Q2: minima-loftr, tried only when the default is rejected."""
    c = routing.choose("TMC2", "TC")
    assert c.fallbacks == tuple(config.load("regimes")["fallback_matchers"])
    assert c.candidates()[0] == c.model_name == "eloftr"
    assert c.model_name not in c.fallbacks
    for m in c.fallbacks:
        assert not licence.is_restricted(m), m


def test_cascade_and_cross_modal_routes_have_no_same_modality_fallback():
    assert routing.choose("OHRC", "IIRS").fallbacks == ()
    assert routing.choose("IIRS", "WAC").fallbacks == ()
