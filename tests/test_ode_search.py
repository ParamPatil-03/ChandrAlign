"""DATA-12: reference product search by location, and DATA-10's WAC/MI verification.

The area search used to live in scripts/fetch_lro.py, where nothing could test it.
The response PARSING is the part that breaks -- ODE collapses a one-element list to
a bare object -- so it is split from the network call and tested against saved real
payloads. The one test that does reach ODE is marked `network`.
"""
import json
from pathlib import Path

import pytest

from chandralign.io.instruments import UnknownInstrumentError, detect_instrument, get_spec
from chandralign.io.ode_client import (
    PRODUCT_TYPES,
    OdeError,
    parse_products,
    search_area,
    to_360,
)

ROOT = Path(__file__).resolve().parents[1]
ODE_FIXTURES = ROOT / "tests" / "fixtures" / "ode"


def payload(name):
    return json.loads((ODE_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["ODEResults"]


# ----------------------------------------------------------------------------- parsing real payloads

def test_a_real_nac_payload_parses():
    hits = parse_products(payload("nac_area_search"), "CDRNAC4")
    assert len(hits) == 3
    ids = [h["product_id"] for h in hits]
    assert "nac.m1417360906lc" in ids            # a product we actually hold
    first = hits[0]
    assert first["type"] == "CDRNAC4"
    assert first["start_utc"].startswith("2022-")
    assert first["incidence_deg"] is not None
    assert all(f["name"].upper().endswith((".IMG", ".LBL", ".XML")) for f in first["files"])
    assert any(f["name"].upper().endswith(".IMG") for f in first["files"])
    assert len(first["bbox"]) == 4


def test_a_real_wac_payload_parses():
    hits = parse_products(payload("wac_area_search"), "CDRWAM4")
    assert [h["product_id"] for h in hits] == [
        "wac.m107908070mc", "wac.m171594275mc", "wac.m106698280mc"]


def test_a_single_product_is_not_mistaken_for_a_dict_of_fields():
    """ODE returns a bare object, not a one-element list, when exactly one matches.

    A reader that iterates it directly gets the product's KEYS and silently produces
    nonsense, so this is the failure worth pinning.
    """
    one = payload("nac_area_search")
    products = one["Products"]["Product"]
    one["Products"]["Product"] = products[0]              # collapse it, as ODE does
    hits = parse_products(one, "CDRNAC4")
    assert len(hits) == 1
    assert hits[0]["product_id"] == "nac.m1417360906lc"


def test_no_products_is_an_empty_list_not_an_error():
    assert parse_products({"Products": {}}, "CDRNAC4") == []
    assert parse_products({}, "CDRNAC4") == []


def test_longitudes_are_converted_to_odes_convention():
    assert to_360(23.5) == 23.5
    assert to_360(-10.0) == 350.0
    assert to_360(0.0) == 0.0


# ----------------------------------------------------------------------------- refusals

def test_an_unknown_product_type_is_refused_before_any_request():
    with pytest.raises(OdeError, match="unknown product"):
        search_area((-1.0, 1.0, 23.0, 24.0), "HUBBLE")


def test_an_inverted_box_is_refused():
    with pytest.raises(OdeError, match="inverted"):
        search_area((1.0, -1.0, 23.0, 24.0), "NAC")


def test_the_known_product_types_cover_what_the_fetchers_use():
    assert set(PRODUCT_TYPES) == {"NAC", "NAC-EDR", "WAC", "WAC-EDR"}


# ----------------------------------------------------------------------------- DATA-10: all 7 cameras

def test_all_seven_cameras_resolve_from_real_product_ids():
    """DATA-10 acceptance. Every id here is one a real archive actually issued."""
    real = {
        "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml": "OHRC",
        "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml": "TMC2",
        "ch2_iir_nci_20240523T1600301891_d_img_d18.xml": "IIRS",
        "M1417360906LC.IMG": "NAC",                        # held, and from ODE
        "M102000149RC.IMG": "NAC",
        "M107908070MC.IMG": "WAC",                         # from ODE, our own test box
        "M171594275MC.IMG": "WAC",
        "TCO_MAP_02_N03E021N00E024SC.lbl": "TC",           # held
        "MI_MAP_03_N01E023N00E024SC.lbl": "MI",            # from JAXA DARTS, held
    }
    for product, expected in real.items():
        assert detect_instrument(product) == expected, product
    assert len(set(real.values())) == 7                    # all seven, not a subset


def test_a_wac_id_is_not_taken_for_a_nac_one():
    """They differ only in the final letters: NAC is LC/RC, WAC is MC."""
    assert detect_instrument("M107908070MC.IMG") == "WAC"
    assert detect_instrument("M1417360906LC.IMG") == "NAC"
    assert detect_instrument("M1417360906RC.IMG") == "NAC"


def test_something_that_is_not_a_product_is_refused():
    for junk in ("notes.txt", "M1417360906.IMG", "sldem2015_512_00n_30n_000_045_float.lbl"):
        with pytest.raises(UnknownInstrumentError):
            detect_instrument(junk)


# ----------------------------------------------------------------------------- the live call

@pytest.mark.network
def test_ode_really_answers_for_a_known_box():
    """DATA-12 acceptance: at least one real NAC product id for a known box."""
    hits = search_area((-0.3, 0.3, 23.2, 23.8), "NAC", limit=5)
    assert hits, "ODE returned nothing for a box we know NAC covers"
    assert any(h["product_id"].startswith("nac.") for h in hits)
    assert all(detect_instrument(h["files"][0]["name"]) == "NAC" for h in hits if h["files"])
