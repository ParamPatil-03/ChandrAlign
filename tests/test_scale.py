"""Per-axis, provenance-aware scale checking (CHECK-05, estimate/scale.py).

Pixel sizes come from the REAL product labels in tests/fixtures/labels, because
the defects this module exists for are properties of real products: IIRS
pixels are 26% anisotropic, TMC-2's label disagrees with its own corners by 11%,
and every LRO NAC label says 0.5 m whatever altitude it was taken from.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import TransformModel
from chandralign.estimate import robust, scale
from chandralign.estimate.scale import PixelScale
from chandralign.evaluate import quality
from chandralign.io.pds_label import parse_label

LABELS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "labels"
OHRC = parse_label(LABELS / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
TMC2 = parse_label(LABELS / "ch2_tmc_nca_20250207T1102039417_d_img_d18.xml")
IIRS = parse_label(LABELS / "ch2_iir_nci_20240523T1600301891_d_img_d18.xml")
NAC = parse_label(LABELS / "M102000149RC.XML")

# A square, verified reference pixel, so a test can isolate the SOURCE side.
SQUARE_REF = PixelScale("square_ref", 100.0, 100.0, (100.0, 100.0), (100.0, 100.0),
                        ("controlled",), True)


def transform(src: tuple[float, float], ref: tuple[float, float],
              rot_deg: float = 0.0, shift=(12.0, -7.0)) -> np.ndarray:
    """The physically correct src->ref pixel map: pixels -> metres -> rotate -> ref pixels.

    Points are (x, y) = (sample, line), so x steps across-track, y along-track.
    """
    th = math.radians(rot_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    a = np.diag([1.0 / ref[0], 1.0 / ref[1]]) @ rot @ np.diag([src[0], src[1]])
    m = np.eye(3)
    m[:2, :2], m[:2, 2] = a, shift
    return m


# ---------------------------------------------------------------------------
# Per-product pixel size, from real labels
# ---------------------------------------------------------------------------
def test_iirs_pixels_are_not_square():
    p = scale.pixel_scale(IIRS)
    assert p.across_m == pytest.approx(99.995, rel=1e-3)
    assert p.along_m == pytest.approx(79.515, rel=1e-3)
    assert p.anisotropy > 1.2, "a 26% anisotropic pixel read as square"


def test_a_label_confirmed_by_its_own_corners_is_verified():
    """OHRC: label 0.300 m, corners 0.305 m -- two sources within 5%."""
    for meta in (OHRC, IIRS):
        assert scale.pixel_scale(meta).verified, meta.instrument


def test_tmc2s_label_disagrees_with_its_own_corners():
    """Not only NAC: TMC-2 states 4.41 m across while its corners say 4.88."""
    p = scale.pixel_scale(TMC2)
    assert not p.verified
    assert p.across_range[0] == pytest.approx(4.41, rel=1e-3)
    assert p.across_range[1] == pytest.approx(4.881, rel=1e-2)
    assert any("disagree" in n for n in p.notes)


def test_a_nac_label_alone_is_never_verification():
    """One source, however official, is not two sources agreeing."""
    p = scale.pixel_scale(NAC)
    assert p.across_m == 0.5 and not p.verified
    assert any("one cross-track source" in n for n in p.notes)
    assert any("assumed square" in n for n in p.notes)


def test_a_footprint_exposes_what_the_nac_label_hides():
    """M102000149's ODE footprint covers 62.2 km along-track, not 26.1 km."""
    p = scale.pixel_scale(NAC, ground_extent_m=(6_000.0, 62_200.0))
    assert p.along_m == pytest.approx(62_200.0 / NAC.array_shape[0], rel=1e-6)
    assert p.along_m > 1.1                           # not 0.5
    assert not p.verified                            # label and footprint disagree


def test_extent_from_bbox_uses_lunar_not_terrestrial_degrees():
    across, along = scale.extent_from_bbox((0.0, 1.0, 23.0, 23.0))
    assert along == pytest.approx(30_323.0, rel=1e-3)
    assert across == 0.0


# ---------------------------------------------------------------------------
# Anisotropy: what the old single-number check could not see
# ---------------------------------------------------------------------------
def test_a_correct_anisotropic_transform_is_consistent():
    p = scale.pixel_scale(IIRS)
    exp = scale.expected_scale(p, SQUARE_REF)
    v = scale.check(transform((p.across_m, p.along_m), (100.0, 100.0), rot_deg=23.0), exp)
    assert v.status == "consistent", v.message


def test_right_area_wrong_axes_is_rejected():
    """An IIRS transform that scales both axes equally has the right AREA.

    It is still wrong: it treats a 100 x 79.5 m pixel as square. sqrt|det J|
    cannot see that, which is why the old check passed it.
    """
    p = scale.pixel_scale(IIRS)
    g = math.sqrt(p.across_m * p.along_m)            # same area, square pixel
    wrong = transform((g, g), (100.0, 100.0), rot_deg=23.0)
    exp = scale.expected_scale(p, SQUARE_REF)

    legacy_ok, _ = robust.check_scale(TransformModel("affine", wrong), exp.area)
    assert legacy_ok, "premise: the old area-only check accepts it"

    v = scale.check(wrong, exp)
    assert v.status == "inconsistent" and "axis ratio" in v.message


@pytest.mark.parametrize("rot", [0.0, 37.0, 90.0, 181.0])
def test_the_check_does_not_depend_on_rotation(rot):
    p = scale.pixel_scale(IIRS)
    exp = scale.expected_scale(p, SQUARE_REF)
    verdict = scale.check(transform((p.across_m, p.along_m), (100.0, 100.0), rot), exp)
    assert verdict.status == "consistent", verdict.message


def test_real_label_pair_both_verified_gives_a_consistent_verdict():
    """OHRC -> IIRS from real labels: both verified, per-axis from corners."""
    po, pi = scale.pixel_scale(OHRC), scale.pixel_scale(IIRS)
    exp = scale.expected_scale(OHRC, IIRS)
    assert exp.verified
    m = transform((po.across_m, po.along_m), (pi.across_m, pi.along_m), rot_deg=4.0)
    assert scale.check(m, exp).status == "consistent"


# ---------------------------------------------------------------------------
# Provenance: the NAC case, both directions of the old failure
# ---------------------------------------------------------------------------
NAC_TRUE_RATIO = 2.15      # M102014464RC against M109080308LC, scripts/audit_gsd.py


def nac_pair_expectation(with_footprints: bool):
    if not with_footprints:
        return scale.expected_scale(scale.pixel_scale(NAC), scale.pixel_scale(NAC))
    near = PixelScale("nac_low_orbit", 0.5, 0.5, (0.5, 0.55), (0.5, 0.55),
                      ("label", "footprint"), False)
    far = PixelScale("nac_high_orbit", 0.5, 0.5, (0.5, 1.18), (0.5, 1.18),
                     ("label", "footprint"), False)
    return scale.expected_scale(far, near)


def test_old_check_certified_a_wrong_nac_transform():
    """The regression this module exists for, stated as the old behaviour.

    Labels give 0.5 / 0.5 = 1.0. A WRONG transform at scale 1.0, when the truth
    is 2.15, was reported "consistent with the instrument GSDs" ...
    """
    wrong = transform((0.5, 0.5), (0.5, 0.5))
    ok, msg = robust.check_scale(TransformModel("affine", wrong), 1.0)
    assert ok and "consistent" in msg
    # ... and now it is never called consistent.
    assert scale.check(wrong, nac_pair_expectation(False)).status == "unverified"
    assert scale.check(wrong, nac_pair_expectation(True)).status == "unverified"


def test_old_check_rejected_the_correct_nac_transform():
    """... and the CORRECT 2.15x transform was rejected as a scale error."""
    right = transform((NAC_TRUE_RATIO, NAC_TRUE_RATIO), (1.0, 1.0))
    ok, _ = robust.check_scale(TransformModel("affine", right), 1.0)
    assert not ok
    v = scale.check(right, nac_pair_expectation(True))
    assert v.ok and v.status == "unverified"


def test_a_label_alone_cannot_reject_a_correct_transform():
    """Found by scripts/sabotage.py: nothing pinned the label-only case.

    With no footprint, a NAC pair's only source is the labels' 1.0, and the
    correct 2.15x transform lies outside it. Trusting that label strictly would
    reject the right answer on the strength of a number already known to be off
    by up to 2.33x -- so an unverified source gets slack, not a veto.
    """
    right = transform((NAC_TRUE_RATIO, NAC_TRUE_RATIO), (1.0, 1.0))
    v = scale.check(right, nac_pair_expectation(False))
    assert v.ok, v.message
    assert v.status == "unverified"


def test_an_unverified_expectation_still_catches_gross_errors():
    """Unverified is not unlimited: past the 3x slack it still rejects (FM #13)."""
    absurd = transform((10.0, 10.0), (1.0, 1.0))
    v = scale.check(absurd, nac_pair_expectation(True))
    assert v.status == "inconsistent" and not v.ok


def test_estimate_reports_the_scale_status():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 500, (200, 2))
    m = transform((1.0, 1.0), (1.0, 1.0), rot_deg=5.0)
    ref = (np.c_[src, np.ones(len(src))] @ m.T)[:, :2]
    res = robust.estimate(src, ref, expected_scale=nac_pair_expectation(False), centre=(250, 250))
    assert res.ok and res.scale_status == "unverified"


def test_verdict_still_unpacks_as_ok_and_message():
    ok, msg = robust.check_scale(TransformModel("affine", np.eye(3)), 1.0)
    assert ok is True and isinstance(msg, str)


# ---------------------------------------------------------------------------
# The quality gate must not read "not refuted" as "confirmed"
# ---------------------------------------------------------------------------
def good_model():
    return TransformModel(kind="affine", matrix=np.eye(3), scale_estimated=1.0)


def test_an_unverified_scale_caps_the_tier_at_low():
    strong = dict(inlier_count=900, inlier_ratio=0.95, spatial_coverage=0.95, model=good_model())
    assert quality.assess(**strong, scale_status="consistent").tier == "HIGH"
    v = quality.assess(**strong, scale_status="unverified")
    assert v.tier == "LOW" and v.accepted
    assert v.limiting_signal == "scale"


def test_an_inconsistent_scale_status_rejects_even_if_scale_ok_was_left_true():
    v = quality.assess(inlier_count=900, inlier_ratio=0.95, spatial_coverage=0.95,
                       model=good_model(), scale_status="inconsistent")
    assert v.tier == "REJECTED" and quality.FM_SCALE_CONFUSION in v.failure_modes
