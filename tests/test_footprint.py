"""GEO-04: footprint overlap and the pre-filter that runs before any matcher."""
import dataclasses
from pathlib import Path

import pytest
from shapely.geometry import Polygon, box

from chandralign.geometry.footprint import (
    Footprint,
    FootprintError,
    InsufficientOverlapError,
    check_overlap,
    footprint_of,
    run_if_overlapping,
)
from chandralign.io.ode_client import load_product_record
from chandralign.io.pds_label import parse_label, parse_pds4

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
ODE = ROOT / "tests" / "fixtures" / "ode"

OHRC = parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
TMC2 = parse_label(LABELS / "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml")
IIRS_LABEL = LABELS / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml"
IIRS = parse_label(IIRS_LABEL)
TC_N = parse_label(LABELS / "TCO_MAP_02_N03E021N00E024SC.lbl")
NAC_2022 = parse_label(LABELS / "M1417360906LC.XML")


def fp(poly, name="x") -> Footprint:
    return Footprint(name, poly, "label", False)


# ----------------------------------------------------------------------------- geometry of the check

def test_half_overlap():
    c = check_overlap(fp(box(0, 0, 1, 1), "a"), fp(box(0.5, 0, 1.5, 1), "b"))
    assert c.fraction_of_smaller == pytest.approx(0.5, rel=1e-3)
    assert c.ok


def test_disjoint_pair_is_rejected_with_reason():
    c = check_overlap(fp(box(0, 0, 1, 1), "a"), fp(box(5, 5, 6, 6), "b"))
    assert not c.ok and c.overlap_km2 == 0.0
    assert "no overlap" in c.reason


def test_small_inside_large_counts_against_the_smaller():
    c = check_overlap(fp(box(0.2, 0.2, 0.3, 0.3), "small"), fp(box(0, 0, 5, 5), "large"))
    assert c.fraction_of_smaller == pytest.approx(1.0)
    assert c.fraction_of_ref < 0.01 and c.ok


def test_threshold_boundary():
    a, b = fp(box(0, 0, 1, 1), "a"), fp(box(0.95, 0, 1.95, 1), "b")   # 5% overlap
    assert not check_overlap(a, b, min_overlap=0.10).ok
    assert check_overlap(a, b, min_overlap=0.04).ok
    with pytest.raises(ValueError):
        check_overlap(a, b, min_overlap=1.5)


def test_areas_are_equal_area_not_degrees():
    """A 1x1 degree box at 60 deg latitude covers about half the ground of one at the equator."""
    eq = check_overlap(fp(box(0, -0.5, 1, 0.5)), fp(box(0, -0.5, 1, 0.5)))
    hi = check_overlap(fp(box(0, 59.5, 1, 60.5)), fp(box(0, 59.5, 1, 60.5)))
    assert hi.src_area_km2 / eq.src_area_km2 == pytest.approx(0.5, rel=0.01)
    assert eq.src_area_km2 == pytest.approx((1737.4 * 3.14159265 / 180) ** 2, rel=1e-3)


def test_long_slanted_strip_edges_follow_the_ground():
    """Regression: projecting only corners made OHRC vs IIRS 11.8% instead of 26.6%."""
    c = check_overlap(OHRC, IIRS)
    po, pi = footprint_of(OHRC).polygon, footprint_of(IIRS).polygon
    degree_space = po.intersection(pi).area / po.area       # near-exact at the equator
    assert c.fraction_of_src == pytest.approx(degree_space, abs=0.005)
    assert c.fraction_of_src == pytest.approx(0.266, abs=0.005)


def test_antimeridian_crossing():
    east = Polygon([(179.5, 0), (180, 0), (180, 1), (179.5, 1)])
    across = Polygon([(179.75, 0), (-179.75, 0), (-179.75, 1), (179.75, 1)])
    a = Footprint("a", east, "label", False)
    b = Footprint("b", across, "label", True)
    c = check_overlap(a, b)
    assert c.fraction_of_smaller == pytest.approx(0.5, rel=0.01)


# ----------------------------------------------------------------------------- where outlines come from

def test_ch2_and_selene_outlines_come_from_labels():
    for meta in (OHRC, TMC2, IIRS, TC_N):
        assert footprint_of(meta).source == "label"


def test_nac_outline_comes_from_the_catalogue():
    rec = load_product_record(ODE / "nac.m1417360906lc.json")
    f = footprint_of(NAC_2022, catalogue=rec)
    assert f.source == "ode_catalogue"
    assert f.polygon.bounds == pytest.approx((23.44, -1.43, 23.67, -0.01))


def test_nac_without_catalogue_cannot_be_placed():
    with pytest.raises(FootprintError, match="no corners"):
        footprint_of(NAC_2022)


def test_catalogue_for_another_product_is_refused():
    rec = load_product_record(ODE / "nac.m102000149rc.json")
    with pytest.raises(FootprintError, match="does not belong"):
        footprint_of(NAC_2022, catalogue=rec)


# ----------------------------------------------------------------------------- real pairs

@pytest.mark.parametrize("a, b, expected", [
    (OHRC, TMC2, 1.000),      # OHRC strip entirely inside the TMC-2 strip
    (OHRC, TC_N, 0.455),
    (TMC2, TC_N, 0.218),
])
def test_real_pair_overlaps(a, b, expected):
    c = check_overlap(a, b)
    assert c.ok and c.fraction_of_smaller == pytest.approx(expected, abs=0.005)


def test_real_nac_pairs():
    nac = footprint_of(NAC_2022, catalogue=load_product_record(ODE / "nac.m1417360906lc.json"))
    assert check_overlap(OHRC, nac).fraction_of_smaller == pytest.approx(0.208, abs=0.005)
    rejected = check_overlap(nac, TC_N)
    assert not rejected.ok and "no overlap" in rejected.reason


def test_iirs_overlap_depends_on_which_corners_are_trusted():
    """ISRO's refined IIRS corners move it ~13 km: the OHRC overlap vanishes. Both are reported."""
    refined = parse_pds4(IIRS_LABEL, corners="refined")
    assert check_overlap(OHRC, IIRS).fraction_of_src == pytest.approx(0.266, abs=0.005)
    assert check_overlap(OHRC, refined).overlap_km2 == 0.0


# ----------------------------------------------------------------------------- the pre-filter gate

def test_rejected_pair_never_reaches_the_matcher():
    far_away = dataclasses.replace(
        OHRC, product_id="ch2_ohr_ncp_FAKE_FAR_AWAY",
        corner_latlon=[(lat + 40.0, lon + 90.0) for lat, lon in OHRC.corner_latlon],
    )
    calls = []
    with pytest.raises(InsufficientOverlapError) as err:
        run_if_overlapping(OHRC, far_away, lambda s, r: calls.append((s, r)))
    assert calls == []                                       # zero matcher calls
    assert err.value.check.overlap_km2 == 0.0


def test_accepted_pair_reaches_the_matcher_once():
    calls = []
    out = run_if_overlapping(OHRC, TMC2, lambda s, r, tag: calls.append(tag) or "matched", "t1")
    assert out == "matched" and calls == ["t1"]
