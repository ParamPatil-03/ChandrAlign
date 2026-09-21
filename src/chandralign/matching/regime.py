"""Automatic matcher routing (MATCH-09, P2-T10).

WHAT THIS DECIDES, AND WHY IT IS NOT A RANKING
Given two scenes and nothing but their metadata, choose three things before a
single pixel is read:

    route           direct match, or the staged cascade (MATCH-10)
    representation  raw pixels, phase congruency, or MIND
    matcher         which model to run

These are separate decisions because they fail for separate reasons. The scale
gap is a property of the instruments; the illumination difference is a property
of the two acquisitions; the matcher is a property of what we have benchmarked.
Collapsing them into one ordered list of "best matchers" loses that.

THE MEASUREMENT THAT SHAPES THIS
Correlation between two descriptions of the same lunar relief lit from two
directions, 5 seeds, elevation fixed (full table in configs/regimes.yaml):

    d_az      raw      phase congruency    MIND
     30     +0.867         +0.705         +0.634      <- raw wins
     90     +0.010         +0.465         +0.038      <- PC wins, 12x MIND
    180     -0.991         +0.860         +0.975      <- MIND wins

Two results here are easy to get backwards:

  - Preprocessing is not free. Below about 60 degrees of sun azimuth difference
    the plain intensity relationship is the STRONGEST signal available, and
    both descriptors throw it away. Always preprocessing is worse than never.

  - The difficulty is not monotonic. 90 degrees is the worst case, not 180,
    because at 90 the lit and shadowed facets swap while at 180 the scene is
    closer to a straight contrast inversion -- which MIND is built to ignore.

So the selector interpolates a band structure rather than sorting a list.

WHAT IT REFUSES TO PRETEND
Above 60 degrees of azimuth difference, every matcher we have benchmarked
(sift, xfeat, aliked-lightglue, eloftr) failed on every seed, best case 3.4 px
against a 2 px bar. The selector still returns its best effort there, but marks
the expectation `unsolved` so that no caller can read the result as supported.
That flag is advisory only: it is never allowed to stand in for the quality
gate, which re-decides the question from the actual match evidence.

HONEST STATUS OF THE NUMBERS
The bands are measured on SYNTHETIC hillshaded relief. Fitting a rule on
synthetic data and then reporting it on synthetic data is the same in-sample
trap as tuning a threshold on the runs used to score it, so `Decision.evidence`
says so on every decision until the bands are re-derived on the real LRO NAC
illumination ladder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .. import config
from ..contracts import SceneMeta

# What the selector can conclude about a regime, worst last.
Expectation = str          # "solved" | "degraded" | "unsolved" | "unknown"
Route = str                # "direct" | "cascade"
Representation = str       # "raw" | "phase_congruency" | "mind"


@dataclass(frozen=True)
class SceneConditions:
    """The measurable differences between two scenes, with the gaps left visible."""

    src_instrument: str
    ref_instrument: str
    scale_ratio: float
    scale_source: str                       # "label" | "nominal" | "mixed"
    cross_modality: bool
    d_azimuth_deg: Optional[float] = None   # None means UNKNOWN, never zero
    d_incidence_deg: Optional[float] = None
    max_incidence_deg: Optional[float] = None
    illumination_known: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    """A routing decision, with the reasoning attached to it rather than logged."""

    regime: str
    route: Route
    representation: Representation
    matcher: Optional[str]
    expectation: Expectation
    conditions: Optional[SceneConditions] = None
    candidates: tuple[str, ...] = ()
    reason: str = ""
    evidence: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def supported(self) -> bool:
        """Whether any tested configuration has met the accuracy bar in this regime.

        False does NOT mean do not run: it means do not trust the output on the
        strength of the route alone. The quality gate decides that separately.
        """
        return self.expectation == "solved"


_SYNTHETIC_EVIDENCE = (
    "representation bands measured on synthetic hillshaded relief "
    "(configs/regimes.yaml representation_bands, 5 seeds); "
    "NOT yet validated on real imagery"
)


def _instrument_of(scene: SceneMeta | str) -> str:
    return scene if isinstance(scene, str) else str(scene.instrument)


def conditions(src: SceneMeta | str, ref: SceneMeta | str) -> SceneConditions:
    """Measure the differences that routing depends on, reading no pixels.

    Scale comes from Part 1's `scale_precheck`, which reads the threshold out of
    our own configs/regimes.yaml. Calling it rather than recomputing the ratio
    here is deliberate: two implementations of the same rule drift apart, and
    this one has to agree with the pre-check the pipeline already runs.
    """
    from ..io.instruments import scale_precheck

    notes: list[str] = []
    check = scale_precheck(src, ref)

    src_name, ref_name = _instrument_of(src), _instrument_of(ref)
    cross_modal = config.is_cross_modal(src_name, ref_name)

    d_az = d_inc = max_inc = None
    known = False
    if isinstance(src, SceneMeta) and isinstance(ref, SceneMeta):
        from ..geometry.solar import illumination_delta

        delta = illumination_delta(src, ref)
        d_az = delta.get("d_azimuth_deg")
        d_inc = delta.get("d_incidence_deg")
        max_inc = delta.get("max_incidence_deg")
        known = bool(delta.get("complete"))
        if not known:
            notes.append(
                "sun geometry incomplete; illumination routing falls back to the "
                "default and the decision is marked unknown rather than assuming "
                "the suns match")
    else:
        notes.append("instrument names only, no scene metadata: illumination unknown")

    return SceneConditions(
        src_instrument=src_name, ref_instrument=ref_name,
        scale_ratio=float(check.ratio), scale_source=str(check.source),
        cross_modality=bool(cross_modal),
        d_azimuth_deg=d_az, d_incidence_deg=d_inc, max_incidence_deg=max_inc,
        illumination_known=known, notes=tuple(notes))


def representation_for(d_azimuth_deg: Optional[float]) -> tuple[Representation, str]:
    """Which description of the image to match on, from the measured bands.

    Returns ("raw", reason) when the difference is unknown. That is the
    conservative choice: raw is what every matcher is trained and benchmarked
    on, so an unknown regime gets the best-understood path rather than a
    speculative one.
    """
    if d_azimuth_deg is None:
        return "raw", "sun azimuth difference unknown; defaulting to raw pixels"
    bands = config.load("regimes").get("representation_bands") or []
    for band in bands:
        if float(d_azimuth_deg) <= float(band["max_d_azimuth_deg"]):
            return str(band["representation"]), str(band.get("reason", ""))
    return "raw", "no band matched; defaulting to raw pixels"


def _expectation_for(cond: SceneConditions) -> tuple[Expectation, str]:
    """How much confidence the ROUTE alone justifies, before any matching runs."""
    if not cond.illumination_known:
        return "unknown", "sun geometry incomplete, so the regime cannot be identified"

    unsolved = config.load("regimes").get("unsolved_illumination") or {}
    limit = unsolved.get("min_d_azimuth_deg")
    if limit is not None and cond.d_azimuth_deg is not None \
            and float(cond.d_azimuth_deg) >= float(limit):
        return "unsolved", (
            f"{cond.d_azimuth_deg:.0f} deg of sun azimuth difference is past "
            f"{float(limit):.0f} deg, where every benchmarked matcher failed on "
            f"every seed (best 3.4 px against a 2 px bar)")

    if cond.d_azimuth_deg is not None and float(cond.d_azimuth_deg) >= 30.0:
        return "degraded", (
            f"{cond.d_azimuth_deg:.0f} deg of sun azimuth difference is inside the "
            f"tested range but accuracy falls off across it")
    return "solved", "illumination difference is small and well covered by the benchmark"


def select(src: SceneMeta | str, ref: SceneMeta | str) -> Decision:
    """Choose route, representation and matcher for a pair. Reads no pixels.

    The order of the checks is the order of the constraints' severity: a scale
    gap that no matcher can cross makes the illumination question irrelevant,
    so it is settled first.
    """
    cond = conditions(src, ref)
    cfg = config.load("regimes")
    default_matcher = str(cfg.get("default_matcher", "aliked-lightglue"))
    notes = list(cond.notes)

    # 1. Scale first. Nothing about illumination rescues a gap this wide.
    from ..io.instruments import scale_precheck

    check = scale_precheck(src, ref)
    if check.extreme:
        return Decision(
            regime="extreme_scale", route="cascade", representation="raw",
            matcher=None, expectation="unknown", conditions=cond,
            reason=check.reason,
            evidence="scale threshold from configs/regimes.yaml via io.instruments.scale_precheck",
            notes=tuple(notes + [
                "the cascade (MATCH-10) chooses the matcher for each step, so no "
                "single matcher is named here"]))

    # 2. Cross-modality is a radiometric relationship, not a geometric one, so
    #    it is decided on its own candidate list rather than by azimuth band.
    if cond.cross_modality:
        candidates = tuple((cfg.get("cross_modal_candidates") or {}).keys())
        return Decision(
            regime="cross_modal", route="direct", representation="mind",
            matcher=None, expectation="unknown", conditions=cond,
            candidates=candidates,
            reason=(f"{cond.src_instrument} and {cond.ref_instrument} are different "
                    f"modalities, a nonlinear and sometimes contrast-reversing "
                    f"relationship. MIND is the representation measured to survive a "
                    f"response change (+0.946 against phase congruency's +0.388 on a "
                    f"gamma shift)."),
            evidence=_SYNTHETIC_EVIDENCE,
            notes=tuple(notes + [
                "candidates are benchmarked, not ranked: none has been measured on "
                "this pairing yet"]))

    # 3. Same modality: the sun decides how to describe the image.
    representation, why = representation_for(cond.d_azimuth_deg)
    expectation, expectation_why = _expectation_for(cond)

    if cond.d_azimuth_deg is None:
        regime = "same_modal_unknown_illumination"
    elif expectation == "unsolved":
        regime = "same_modal_opposed_sun"
    elif float(cond.d_azimuth_deg) >= 30.0:
        regime = "same_modal_shifted_sun"
    else:
        regime = "same_modal_normal"

    if (cond.max_incidence_deg is not None
            and float(cond.max_incidence_deg) >= 70.0):
        notes.append(
            f"low sun: {cond.max_incidence_deg:.0f} deg incidence. The synthetic "
            f"sweep found low sun alone did NOT degrade matching (0.118 px at 80 "
            f"deg incidence with a small azimuth difference), so this is recorded, "
            f"not routed on. Real polar imagery may still differ.")

    return Decision(
        regime=regime, route="direct", representation=representation,
        matcher=default_matcher, expectation=expectation, conditions=cond,
        candidates=tuple(cfg.get("shippable_matchers") or ()),
        reason=f"{why}. {expectation_why}.",
        evidence=_SYNTHETIC_EVIDENCE,
        notes=tuple(notes))


def explain(decision: Decision) -> str:
    """A human-readable account of one decision, for reports and the failure log."""
    cond = decision.conditions
    lines = [
        f"regime          {decision.regime}",
        f"route           {decision.route}",
        f"representation  {decision.representation}",
        f"matcher         {decision.matcher or '(chosen per cascade step)'}",
        f"expectation     {decision.expectation}"
        + ("" if decision.supported else "   <- not a supported configuration"),
    ]
    if cond is not None:
        az = "unknown" if cond.d_azimuth_deg is None else f"{cond.d_azimuth_deg:.1f} deg"
        lines += [
            f"scale ratio     {cond.scale_ratio:.2f}x (from {cond.scale_source})",
            f"d_azimuth       {az}",
            f"cross-modality  {cond.cross_modality}",
        ]
    lines.append(f"reason          {decision.reason}")
    lines.append(f"evidence        {decision.evidence}")
    for note in decision.notes:
        lines.append(f"note            {note}")
    return "\n".join(lines)
