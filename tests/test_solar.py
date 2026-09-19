"""GEO-01: scene sun geometry with a recorded source for every value, and illumination_delta."""
import json
from pathlib import Path

import pytest

from chandralign.geometry.solar import (
    SceneIllumination,
    azimuth_difference,
    illumination_delta,
    scene_illumination,
)
from chandralign.io.ode_client import catalogue_number, load_product_record
from chandralign.io.pds_label import parse_label

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
ODE = ROOT / "tests" / "fixtures" / "ode"

OHRC = parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
TMC2 = parse_label(LABELS / "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml")
NAC_HIGH_SUN = parse_label(LABELS / "M1417360906LC.XML")
NAC_LOW_SUN = parse_label(LABELS / "M102000149RC.XML")
SELENE = parse_label(LABELS / "TCO_MAP_02_N03E021N00E024SC.lbl")
ODE_HIGH = load_product_record(ODE / "nac.m1417360906lc.json")
ODE_LOW = load_product_record(ODE / "nac.m102000149rc.json")


def test_ch2_sun_comes_from_the_label():
    ill = scene_illumination(OHRC)
    assert ill.incidence_deg == pytest.approx(82.733560)
    assert ill.sub_solar_azimuth_deg == pytest.approx(269.823288)
    assert ill.sun_elevation_deg == pytest.approx(7.266440)
    assert ill.sources["incidence_deg"] == ill.sources["sub_solar_azimuth_deg"] == "label"
    # not in ISRO labels, and no catalogue: stays unknown
    assert ill.emission_deg is None and ill.sources["emission_deg"] is None
    assert ill.phase_deg is None


def test_nac_angles_come_from_the_catalogue_and_say_so():
    ill = scene_illumination(NAC_HIGH_SUN, catalogue=ODE_HIGH)
    assert (ill.incidence_deg, ill.emission_deg, ill.phase_deg) == (7.71, 1.72, 9.4)
    assert set(v for v in ill.sources.values() if v) == {"ode_catalogue"}
    assert ill.sub_solar_azimuth_deg is None          # not in the ODE product record
    assert ill.sources["sub_solar_azimuth_deg"] is None


def test_nac_without_catalogue_is_unknown_not_guessed():
    ill = scene_illumination(NAC_HIGH_SUN, use_saved_catalogue=False)
    assert not ill.known
    assert all(getattr(ill, a) is None for a in ("incidence_deg", "emission_deg", "phase_deg"))
    assert any("no catalogue record" in n for n in ill.notes)


def test_selene_mosaic_has_no_single_sun():
    ill = scene_illumination(SELENE)
    assert not ill.known
    assert any("mosaic" in n for n in ill.notes)


def test_label_beats_catalogue_and_disagreement_is_reported():
    fake = {"pdsid": OHRC.product_id, "record": {"Incidence_angle": "60.0", "Emission_angle": "3.0",
                                                  "Phase_angle": "62.0"}}
    ill = scene_illumination(OHRC, catalogue=fake)
    assert ill.incidence_deg == pytest.approx(82.73356)      # label kept
    assert ill.sources["incidence_deg"] == "label"
    assert ill.emission_deg == 3.0 and ill.sources["emission_deg"] == "ode_catalogue"   # gap filled
    assert any("label 82.73 vs catalogue 60.00" in n for n in ill.notes)


def test_catalogue_for_another_product_is_refused():
    with pytest.raises(ValueError, match="does not belong"):
        scene_illumination(NAC_LOW_SUN, catalogue=ODE_HIGH)


@pytest.mark.parametrize("a, b, expected", [(350, 10, 20), (0, 180, 180), (269.8, 104.3, 165.5), (90, 90, 0)])
def test_azimuth_difference_wraps_around(a, b, expected):
    assert azimuth_difference(a, b) == pytest.approx(expected)
    assert azimuth_difference(b, a) == pytest.approx(expected)


def test_delta_between_two_ch2_scenes_is_complete():
    d = illumination_delta(OHRC, TMC2)
    assert d["d_incidence_deg"] == pytest.approx(82.733560 - 45.993599)
    assert d["d_azimuth_deg"] == pytest.approx(azimuth_difference(269.823288, 104.269696))
    assert d["max_incidence_deg"] == pytest.approx(82.733560)
    assert d["complete"] is True
    assert d["d_phase_deg"] is None                   # neither label has phase


def test_delta_with_nac_reports_the_gaps():
    """The real Easy vs stress pairs over the same ground: same OHRC, two NAC sun angles."""
    easy = illumination_delta(OHRC, scene_illumination(NAC_LOW_SUN, catalogue=ODE_LOW))
    stress = illumination_delta(OHRC, scene_illumination(NAC_HIGH_SUN, catalogue=ODE_HIGH))
    assert easy["d_incidence_deg"] == pytest.approx(82.73356 - 79.75)
    assert stress["d_incidence_deg"] == pytest.approx(82.73356 - 7.71)
    for d in (easy, stress):
        assert d["d_azimuth_deg"] is None and d["complete"] is False
        assert d["sources"]["ref"]["incidence_deg"] == "ode_catalogue"
        assert d["sources"]["src"]["incidence_deg"] == "label"


def test_delta_with_unknown_side_is_none_not_zero():
    d = illumination_delta(TMC2, SELENE)
    assert d["d_incidence_deg"] is None and d["max_incidence_deg"] is None


def test_saved_ode_records_are_complete():
    for rec in (ODE_HIGH, ODE_LOW):
        assert rec["source"] == "ode_catalogue" and rec["query_url"].startswith("https://oderest")
        r = rec["record"]
        for key in ("Incidence_angle", "Emission_angle", "Phase_angle"):
            assert catalogue_number(r, key) is not None
        assert r["Footprint_geometry"].startswith("POLYGON")
