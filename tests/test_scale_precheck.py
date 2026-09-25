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
    from chandralign.estimate.scale import pixel_scale

    check = scale_precheck(meta("OHRC"), meta("IIRS"))
    assert check.extreme is True
    assert check.route == "cascade"
    # The ratio uses each pixel's AREA (geometric mean of both axes), from the
    # best-supported source. IIRS is 99.995 x 79.515 m, so its effective size is
    # 89.17 m, not the label's 97.15 -- the ratio is 290, not the 324 the labels give.
    po, pi = pixel_scale(meta("OHRC")), pixel_scale(meta("IIRS"))
    expected = ((pi.across_m * pi.along_m) / (po.across_m * po.along_m)) ** 0.5
    assert check.ratio == pytest.approx(expected, rel=1e-9)
    assert check.ratio == pytest.approx(290.15, abs=0.1)
    assert check.source == "corners"
    assert check.verified is True
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

def test_the_best_supported_size_is_used_not_the_label():
    """Neither the registry nominal nor the label: the best-supported size, per product.

    Every CH-2 label states the design GSD scaled to altitude, and TMC-2's is 11.6%
    below its measured size, so routing on labels would route on the wrong number.
    """
    from chandralign.estimate.scale import pixel_scale

    for name, nominal in (("OHRC", 0.25), ("TMC2", 5.0), ("IIRS", 80.0)):
        m = meta(name)
        ps = pixel_scale(m)
        check = scale_precheck(m, "NAC")
        assert get_spec(name).gsd_m == nominal
        assert check.gsd_a_m == pytest.approx((ps.across_m * ps.along_m) ** 0.5)
        assert check.source == "mixed"                         # best source vs nominal
    tmc2 = scale_precheck(meta("TMC2"), "NAC")
    assert tmc2.gsd_a_m == pytest.approx((4.920 * 5.037) ** 0.5)   # the MEASURED size
    assert tmc2.gsd_a_m != meta("TMC2").gsd_m                        # not the label


def test_no_registry_pairing_changes_its_decision_when_label_gsds_are_used():
    """Measured, and recorded because it is the honest result rather than the hoped-for one.

    Across all 21 registry pairings, using each product's best-supported size
    instead of the nominal one changes NO routing decision.

    CORRECTION of an earlier claim: I said the label GSD moved TMC-2 <-> MI off
    the 4.00 knife-edge to 4.54. That 4.54 came from TMC-2's label, which is 11.6%
    wrong. With TMC-2's measured size the ratio is 4.02 -- still on the edge. What
    actually takes the pair off it is MI's OWN label: 14.806 m on the tile we hold,
    not the 20 m nominal, which gives 2.97 and routes DIRECT.
    """
    real = {n: meta(n) for n in PRODUCTS}
    for a, b in itertools.combinations(list(load_registry()), 2):
        nominal = scale_precheck(a, b)
        best = scale_precheck(real.get(a, a), real.get(b, b))
        assert best.extreme == nominal.extreme, (a, b, nominal.ratio, best.ratio)

    assert scale_precheck("TMC2", "MI").ratio == pytest.approx(4.0)       # nominal: on the edge
    assert scale_precheck(meta("TMC2"), "MI").ratio == pytest.approx(4.02, abs=0.01)
    mi = parse_label(ROOT / "tests" / "fixtures" / "labels" / "MI_MAP_03_N01E023N00E024SC.lbl")
    real_pair = scale_precheck(meta("TMC2"), mi)
    assert real_pair.ratio == pytest.approx(2.97, abs=0.01)
    assert real_pair.route == "direct"


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


_NAC = Path(__file__).resolve().parents[1] / "data/raw/lro/nac"


def _nac(pid):
    hits = [p for p in _NAC.glob(f"nac.{pid.lower()}/{pid}.XML")]
    return hits[0] if hits else None


@pytest.mark.skipif(_nac("M102014464RC") is None or _nac("M109080308LC") is None,
                    reason="real LRO NAC products not downloaded")
def test_nac_pixel_size_comes_from_its_footprint_not_the_nominal_label():
    """Audit 2026-09-26 I-04: every NAC label says 0.5 m, so this pair came back 1.00x.

    scripts/audit_gsd.py measures its real ratio at about 2.15x from the footprints; a 2x
    error near the 4x cascade threshold flips routing, and CHECK-05 would certify it.
    """
    from chandralign.io.instruments import scale_precheck
    from chandralign.io.pds_label import parse_label
    chk = scale_precheck(parse_label(_nac("M102014464RC")), parse_label(_nac("M109080308LC")))
    assert chk.ratio == pytest.approx(2.15, rel=0.10)
    assert "footprint" in chk.source
