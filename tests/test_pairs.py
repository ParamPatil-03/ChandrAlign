"""DATA-08: genuinely overlapping source/reference pairs, with the overlap recorded.

The acceptance is specifically about TMC-2 <-> SELENE TC, the ~2:1 pairing with no
published prior art. These tests read data/pairs/registered_pairs.json, which
scripts/build_pairs.py writes from the real labels we hold, and re-derive the
overlap from the labels so the file cannot drift away from the data.
"""
import json
from pathlib import Path

import pytest

from chandralign.geometry.footprint import check_overlap
from chandralign.io.pds_label import parse_label

ROOT = Path(__file__).resolve().parents[1]
PAIRS_FILE = ROOT / "data" / "pairs" / "registered_pairs.json"

# registered_pairs.json is COMMITTED; the products it names are not. So only the
# tests that re-read LABELS are gated on the labels. The rest read the committed
# file alone and run everywhere. (A module-wide gate used to skip all 12 whenever
# ANY named label was missing -- one absent WAC product hid the TMC-2 <-> TC
# headline, whose own labels were present.)
def _labels_present(pairing: str | None = None) -> bool:
    if not PAIRS_FILE.exists():
        return False
    try:
        doc = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for pair in doc.get("pairs", []):
        if pairing is not None and pair.get("pairing") != pairing:
            continue
        for side in ("source", "reference"):
            label = pair.get(side, {}).get("label")
            if label and not (ROOT / label).is_file():
                return False
    return True


pytestmark = pytest.mark.skipif(not PAIRS_FILE.exists(), reason="run scripts/build_pairs.py")
needs_all_labels = pytest.mark.skipif(
    not _labels_present(), reason="some products the file names are not downloaded")
needs_headline_labels = pytest.mark.skipif(
    not _labels_present("TMC2<->TC"), reason="the TMC-2 / SELENE TC products are not downloaded")


@pytest.fixture(scope="module")
def doc():
    return json.loads(PAIRS_FILE.read_text(encoding="utf-8"))


def headline(doc):
    return [p for p in doc["pairs"] if p["pairing"] == "TMC2<->TC"]


# ----------------------------------------------------------------------------- the acceptance

def test_at_least_three_tmc2_to_selene_tc_pairs(doc):
    """DATA-08 acceptance, stated exactly: at least 3 genuinely overlapping pairs."""
    pairs = headline(doc)
    assert len(pairs) >= 3, f"only {len(pairs)} TMC-2 <-> TC pairs"
    assert doc["headline_pair_count"] == len(pairs)


def test_each_headline_pair_shares_real_ground(doc):
    """Not a token sliver: every one shares over 100 km2 and over 10% of the smaller."""
    for p in headline(doc):
        assert p["overlap_km2"] > 100.0, p
        assert p["fraction_of_smaller"] > 0.10, p


def test_the_headline_pairs_are_three_different_pieces_of_ground(doc):
    """Three overlaps with the SAME reference tile would be one pair counted thrice."""
    references = {p["reference"]["product_id"] for p in headline(doc)}
    assert len(references) >= 3, references


def test_the_scale_ratio_is_the_two_to_one_case_we_claim(doc):
    """TMC-2 4.41 m against TC 7.403 m is 1.68:1 -- the regime the project is about.

    These are LABEL pixel sizes, which is what the file records. TMC-2's label
    value is wrong: measured from pixels it is 4.920 m (configs/measured_scales.yaml),
    so the physical ratio is ~1.50:1. Still the ~2:1 regime; not the same number.
    """
    for p in headline(doc):
        assert p["scale_ratio"] == pytest.approx(1.679, abs=0.01), p
        assert p["source"]["gsd_m"] == pytest.approx(4.41)
        assert p["reference"]["gsd_m"] == pytest.approx(7.403, abs=0.001)


def test_the_headline_pairs_are_listed_first(doc):
    """So that reading the file top-down shows the contribution, not the easy pairs."""
    assert doc["pairs"][0]["pairing"] == "TMC2<->TC"


# ----------------------------------------------------------------------------- the file matches the data

def _rederive(doc, p):
    src = parse_label(ROOT / p["source"]["label"])
    ref = parse_label(ROOT / p["reference"]["label"])
    again = check_overlap(src, ref, min_overlap=doc["min_overlap"])
    assert again.ok, p
    # the file stores km2 rounded to 2 dp, so allow half a unit in the last place
    assert again.overlap_km2 == pytest.approx(p["overlap_km2"], abs=0.005), p
    assert again.fraction_of_smaller == pytest.approx(p["fraction_of_smaller"], abs=1e-4), p


@needs_headline_labels
def test_the_headline_overlaps_are_reproducible_from_the_labels(doc):
    """DATA-08's own pairs, re-derived whenever THEIR labels are present."""
    for p in headline(doc):
        _rederive(doc, p)


@needs_all_labels
def test_every_recorded_overlap_is_reproducible_from_the_labels(doc):
    """The recorded numbers are re-derived here, so the file cannot go stale silently."""
    for p in doc["pairs"]:
        _rederive(doc, p)


@needs_all_labels
def test_every_label_named_in_the_file_exists(doc):
    for p in doc["pairs"]:
        assert (ROOT / p["source"]["label"]).is_file(), p["source"]["label"]
        assert (ROOT / p["reference"]["label"]).is_file(), p["reference"]["label"]


def test_no_pair_falls_below_the_stated_threshold(doc):
    for p in doc["pairs"]:
        assert p["fraction_of_smaller"] >= doc["min_overlap"], p


def test_a_source_is_never_paired_with_itself(doc):
    for p in doc["pairs"]:
        assert p["source"]["product_id"] != p["reference"]["product_id"]
        assert p["source"]["mission"] == "CH2"
        assert p["reference"]["mission"] in ("SELENE", "LRO")


# ----------------------------------------------------------------------------- the wider inventory

def test_the_other_pairings_are_recorded_too(doc):
    """The headline is TMC-2 <-> TC, but the file is the whole inventory."""
    pairings = {p["pairing"] for p in doc["pairs"]}
    assert "TMC2<->TC" in pairings
    assert len(pairings) >= 8                     # all 3 CH-2 cameras x all 4 references
    assert doc["pair_count"] == len(doc["pairs"])


def test_pairs_against_an_unlit_reference_are_flagged_not_hidden(doc):
    """Two WAC products are night-side. A pair with one is geometrically real and
    practically useless, so it stays in the file with reference_lit False."""
    unlit = [p for p in doc["pairs"] if p["reference_lit"] is False]
    assert unlit, "expected the night-side WAC pairs to be present and flagged"
    for p in unlit:
        assert p["reference_incidence_deg"] > 90.0      # the sun is below the horizon
    assert doc["unlit_reference_pair_count"] == len(unlit)
    assert doc["usable_pair_count"] == doc["pair_count"] - len(unlit)


def test_no_headline_pair_uses_an_unlit_reference(doc):
    """The claim we make is about usable pairs, so this must hold for TMC-2 <-> TC."""
    assert all(p["reference_lit"] is not False for p in headline(doc))
