"""What scale a correct transform should have, per axis, and how sure we are (CHECK-05).

WHY THE OLD CHECK WAS NOT ENOUGH
The first scale check compared one measured number, sqrt|det J|, against one
expected number, src_gsd / ref_gsd. Real products broke both halves of that.

1. PIXELS ARE NOT SQUARE. A pushbroom camera's along-track spacing is set by
   how far the spacecraft flies per line, not by the optics, so it need not
   match the cross-track spacing. From the corners of the labels we hold:

       OHRC    0.305 m across  x   0.309 m along
       TMC-2   4.881 m across  x   5.065 m along
       IIRS   99.995 m across  x  79.515 m along     (26% anisotropic)

   A single-number check cannot tell a correct IIRS transform from one that
   scales both axes equally, and both have almost the same area scale.

2. A LABEL'S GSD IS NOT ALWAYS A MEASUREMENT. Every LRO NAC label we hold says
   0.5 m; their footprints imply 0.55 to 1.27 m (scripts/audit_gsd.py), because
   LRO flew an eccentric orbit. From labels alone a NAC<->NAC ratio is 1.0 by
   construction, and for one of our pairs the truth is about 2.15. The old check
   then failed in BOTH directions: it rejected the correct 2.15x transform, and
   accepted a wrong 1.0x one as "consistent with the instrument GSDs".

   Nor is it only NAC. TMC-2's own corners give 4.88 m across where its label
   states 4.41 -- an 11% disagreement inside one Chandrayaan-2 product.

WHAT THIS MODULE DOES INSTEAD
- Every product gets BOTH axes, and the RANGE spanned by every source available
  for it (label, corners, footprint), not one favoured value.
- A product is `verified` only when two independent sources agree within
  Member A's cross-validation tolerance (5%, evaluate/groundtruth.py). One
  source, however official, is not verification.
- A pair's expectation is an interval on area scale plus bounds on anisotropy.
  Area scale is sqrt(gx*gy / hx*hy), exact under any rotation. Anisotropy is
  bounded without knowing the rotation: for J = diag(1/h) R diag(g), the
  singular-value ratio lies between max(ks,kr)/min(ks,kr) and ks*kr.
- An unverified expectation can never produce "consistent". It produces
  "unverified", which the quality gate treats as an unmeasured signal (rule H1).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import config
from ..contracts import SceneMeta

# Member A's cross-validation tolerance: two sources within this agree.
AGREEMENT_TOLERANCE = 0.05

MOON_RADIUS_M = 1_737_400.0


@dataclass(frozen=True)
class PixelScale:
    """Ground size of one pixel, both axes, with every source's value kept."""

    product: str
    across_m: float                          # best estimate, sample direction
    along_m: float                           # best estimate, line direction
    across_range: tuple[float, float]        # (min, max) over all sources
    along_range: tuple[float, float]
    sources: tuple[str, ...]
    verified: bool
    notes: tuple[str, ...] = ()

    @property
    def anisotropy(self) -> float:
        return max(self.across_m, self.along_m) / min(self.across_m, self.along_m)


@dataclass(frozen=True)
class ExpectedScale:
    """What a correct src -> ref pixel transform's scale should be."""

    area: float                               # best sqrt(gx*gy / hx*hy)
    area_range: tuple[float, float]           # over every source combination
    anisotropy_range: tuple[float, float]     # rotation-free bounds on s1/s2
    verified: bool
    src: PixelScale
    ref: PixelScale

    def describe(self) -> str:
        lo, hi = self.area_range
        span = (f"{self.area:.4g}" if hi / lo < 1.0 + 1e-6
                else f"{self.area:.4g} (sources span {lo:.4g}-{hi:.4g})")
        return (f"expected area scale {span}, anisotropy "
                f"{self.anisotropy_range[0]:.3f}-{self.anisotropy_range[1]:.3f}, "
                f"{'verified' if self.verified else 'UNVERIFIED'}")


@dataclass(frozen=True)
class ScaleVerdict:
    """Outcome of one scale check.

    Unpacks as (ok, message) so existing `ok, msg = check_scale(...)` callers
    keep working; `status` carries the distinction a bool cannot.
    """

    ok: bool
    status: str        # consistent | unverified | inconsistent | abstained | skipped | degenerate
    message: str
    area: Optional[float] = None
    anisotropy: Optional[float] = None

    def __iter__(self):
        yield self.ok
        yield self.message


# ---------------------------------------------------------------------------
# Per-product pixel size
# ---------------------------------------------------------------------------
def extent_from_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """(across_m, along_m) ground extent of a strip from a lat/lon bounding box.

    bbox is (min_lat, max_lat, min_lon, max_lon), as ODE returns it. Assumes a
    roughly north-south strip, which LRO's polar orbit gives. A bounding box of
    a tilted strip overstates its width, so this is a crude second source -- good
    enough to show a label is 2x wrong, not to replace geolocation.
    """
    lat0, lat1, lon0, lon1 = (float(v) for v in bbox)
    m_per_deg = math.pi * MOON_RADIUS_M / 180.0
    along = abs(lat1 - lat0) * m_per_deg
    across = abs(lon1 - lon0) * m_per_deg * math.cos(math.radians((lat0 + lat1) / 2.0))
    return across, along


def _corner_scales(meta: SceneMeta) -> Optional[tuple[float, float]]:
    from ..evaluate.groundtruth import footprint_scales
    from ..io.pds_label import read_corner_sets

    try:
        sets = read_corner_sets(meta.label_path)
    except Exception:
        return None
    corners = sets.get("system") or sets.get("refined")
    if not corners or len(corners) < 4:
        return None
    return footprint_scales(corners, meta.array_shape)


def _agree(values: list[float], tolerance: float) -> bool:
    return len(values) >= 2 and max(values) / min(values) - 1.0 <= tolerance


def pixel_scale(scene: SceneMeta | str, *,
                ground_extent_m: Optional[tuple[float, float]] = None,
                tolerance: float = AGREEMENT_TOLERANCE) -> PixelScale:
    """Both axes of one product's pixel, from every source it offers.

    Sources, in order of preference for the BEST value:
      corners    the label's own corner coordinates -- both axes, per product
      footprint  ground_extent_m / array shape, if the caller has a footprint
      label      the label's single gsd_m, read as cross-track
      nominal    the instrument registry, when only a name is given
    Preference only picks the point estimate. The RANGE keeps every source, so
    a disagreement is carried forward instead of resolved by fiat.
    """
    if isinstance(scene, str):
        from ..io.instruments import get_spec
        g = float(get_spec(scene).gsd_m)
        return PixelScale(scene, g, g, (g, g), (g, g), ("nominal",), False,
                          ("registry design value only; not a property of any product",))

    meta = scene
    n_lines, n_samples = meta.array_shape
    across: dict[str, float] = {"label": float(meta.gsd_m)}
    along: dict[str, float] = {}
    notes: list[str] = []

    corners = _corner_scales(meta)
    if corners is not None:
        across["corners"], along["corners"] = corners
    if ground_extent_m is not None:
        ex, el = ground_extent_m
        across["footprint"] = float(ex) / n_samples
        along["footprint"] = float(el) / n_lines

    if not along:
        along = {"label (assumed square)": across["label"]}
        notes.append("no along-track source: pixel assumed square, which a "
                     "pushbroom camera does not guarantee")

    pick = lambda d: next(d[k] for k in ("corners", "footprint", "label",
                                          "label (assumed square)") if k in d)
    a_vals, l_vals = list(across.values()), list(along.values())

    # Verified = the cross-track size is confirmed by an independent source.
    # The along-track size comes from corners or a footprint, each already a
    # measurement of extent rather than a restated design value.
    verified = _agree(a_vals, tolerance) and "label (assumed square)" not in along
    if len(a_vals) < 2:
        notes.append("only one cross-track source; a single source is not verification")
    elif not _agree(a_vals, tolerance):
        worst = max(a_vals) / min(a_vals) - 1.0
        notes.append(
            "cross-track sources disagree by "
            f"{worst * 100:.0f}% ({', '.join(f'{k} {v:.4g} m' for k, v in across.items())}), "
            f"past the {tolerance * 100:.0f}% agreement tolerance")

    return PixelScale(
        product=meta.product_id,
        across_m=pick(across), along_m=pick(along),
        across_range=(min(a_vals), max(a_vals)),
        along_range=(min(l_vals), max(l_vals)),
        sources=tuple(sorted(set(across) | set(along))),
        verified=verified, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Per-pair expectation
# ---------------------------------------------------------------------------
def _anisotropy_bounds(ks: float, kr: float) -> tuple[float, float]:
    """Bounds on the singular-value ratio of diag(1/h) R diag(g), any rotation R."""
    return max(ks, kr) / min(ks, kr), ks * kr


def expected_scale(src: SceneMeta | str | PixelScale, ref: SceneMeta | str | PixelScale, *,
                   src_extent_m: Optional[tuple[float, float]] = None,
                   ref_extent_m: Optional[tuple[float, float]] = None) -> ExpectedScale:
    """Expected scale of a transform mapping src pixel coordinates to ref pixels.

    A src pixel spans (gx, gy) metres, which is (gx/hx, gy/hy) ref pixels, so the
    area scale is sqrt(gx*gy / hx*hy) whatever the rotation between them.
    """
    ps = src if isinstance(src, PixelScale) else pixel_scale(src, ground_extent_m=src_extent_m)
    pr = ref if isinstance(ref, PixelScale) else pixel_scale(ref, ground_extent_m=ref_extent_m)

    area = math.sqrt(ps.across_m * ps.along_m / (pr.across_m * pr.along_m))
    lo = math.sqrt(ps.across_range[0] * ps.along_range[0]
                   / (pr.across_range[1] * pr.along_range[1]))
    hi = math.sqrt(ps.across_range[1] * ps.along_range[1]
                   / (pr.across_range[0] * pr.along_range[0]))

    # Anisotropy bounds over every combination of source endpoints.
    def kappas(p: PixelScale):
        return [max(a, l) / min(a, l) for a, l in itertools.product(p.across_range, p.along_range)]
    bounds = [_anisotropy_bounds(ks, kr) for ks in kappas(ps) for kr in kappas(pr)]
    k_lo = min(b[0] for b in bounds)
    k_hi = max(b[1] for b in bounds)

    return ExpectedScale(area=area, area_range=(lo, hi), anisotropy_range=(k_lo, k_hi),
                         verified=ps.verified and pr.verified, src=ps, ref=pr)


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------
def check(matrix: np.ndarray, expected: ExpectedScale,
          centre: tuple[float, float] = (0.0, 0.0),
          tolerance: Optional[float] = None,
          anisotropy_tolerance: Optional[float] = None) -> ScaleVerdict:
    """Judge a fitted transform against a per-axis, provenance-aware expectation.

    Outcomes:
      inconsistent  outside what ANY source allows -> reject (failure mode #13)
      consistent    inside the range, and the expectation is verified
      unverified    inside the range, but the expectation rests on sources that
                    do not agree; never reported as consistent

    An UNVERIFIED expectation can still reject, but only a gross mismatch --
    beyond `estimate.unverified_scale_slack` (3x by default) outside the range.
    The slack is set from measurement: NAC labels were off by up to 2.33x, so a
    tighter bound would reject correct transforms on the strength of a label we
    already know to be wrong.
    """
    from . import models

    if tolerance is None:
        tolerance = float(config.get("estimate.scale_tolerance", 0.25))
    if anisotropy_tolerance is None:
        anisotropy_tolerance = float(config.get("estimate.anisotropy_tolerance", 0.10))
    slack = float(config.get("estimate.unverified_scale_slack", 3.0))

    s1, s2 = models.local_axis_scales(matrix, at=centre)
    if not (np.isfinite(s1) and np.isfinite(s2)) or s2 <= 0:
        return ScaleVerdict(False, "degenerate", "scale check failed: degenerate transform")
    area, aniso = math.sqrt(s1 * s2), s1 / s2

    lo, hi = expected.area_range
    k_lo, k_hi = expected.anisotropy_range
    if expected.verified:
        a_lo, a_hi = lo / (1.0 + tolerance), hi * (1.0 + tolerance)
    else:
        a_lo, a_hi = lo / slack, hi * slack
    k_lo_t, k_hi_t = k_lo / (1.0 + anisotropy_tolerance), k_hi * (1.0 + anisotropy_tolerance)

    if not (a_lo <= area <= a_hi):
        why = "every source" if not expected.verified else "the verified expectation"
        return ScaleVerdict(
            False, "inconsistent",
            f"area scale {area:.4g} is outside what {why} allows "
            f"({a_lo:.4g}-{a_hi:.4g}); {expected.describe()}", area, aniso)

    # Anisotropy is only meaningful when the axis sizes themselves are trusted.
    if expected.verified and not (k_lo_t <= aniso <= k_hi_t):
        return ScaleVerdict(
            False, "inconsistent",
            f"axis ratio {aniso:.3f} is outside {k_lo_t:.3f}-{k_hi_t:.3f}: the area "
            f"is right but the axes are scaled wrongly -- e.g. a non-square pushbroom "
            f"pixel treated as square; {expected.describe()}", area, aniso)

    if expected.verified:
        return ScaleVerdict(True, "consistent",
                            f"area scale {area:.4g}, axis ratio {aniso:.3f}: consistent "
                            f"with {expected.describe()}", area, aniso)

    reasons = [n for p in (expected.src, expected.ref) for n in p.notes]
    return ScaleVerdict(
        True, "unverified",
        f"area scale {area:.4g} cannot be confirmed: {expected.describe()}. "
        + "; ".join(reasons), area, aniso)
