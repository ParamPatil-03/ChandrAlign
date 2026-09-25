"""Confidence tiers and quality gating (CHECK-07 / CHECK-08, P2-T16).

WHY THIS EXISTS, IN ONE MEASUREMENT
The synthetic matcher benchmark produced four registrations that the estimator
accepted and that were 3.7 to 13.1 px wrong. They passed because the estimator
asked only two questions -- are there at least 12 inliers, and is the scale
physically plausible -- and both answers were yes. Their scale was right; their
rotation and translation were not.

Worse, those four would look EXCELLENT by reprojection error on their own
inliers, because a set of mutually consistent wrong matches is still mutually
consistent. Internal consistency cannot detect a globally wrong correspondence
structure. Only ground truth can, and at run time on real data there is none.

So the gate does not try to answer "is this registration correct?", which it
cannot. It answers a question it can actually support:

    does this registration satisfy enough INDEPENDENT quality conditions
    to be trusted?

Independence is the point. Inlier count, inlier ratio, scale consistency,
spatial coverage and geometric validity fail in different ways, so a wrong
transform has to defeat all of them at once to get through. The four failures
above all shared a low inlier ratio (<= 0.20) while every correct result sat at
>= 0.43 -- but five samples is not a threshold, which is why the numbers live in
configs/default.yaml and are validated across seeds rather than hard-coded here.

A tier is claimed only when EVERY signal reaches it (worst-of, not average), so
one strong signal cannot carry a weak one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..contracts import Tier, TransformModel

# Failure-mode IDs from PLAN.md section 15.
FM_FALSE_CORRESPONDENCE = 12
FM_SCALE_CONFUSION = 13
FM_UNVERIFIED_EVALUATION = 19     # a result nobody checked presented as a finding (rule H4)

TIER_ORDER: list[Tier] = ["HIGH", "MEDIUM", "LOW", "REJECTED"]


@dataclass
class QualityVerdict:
    tier: Tier
    signals: dict[str, object] = field(default_factory=dict)
    limiting_signal: str | None = None
    failure_modes: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.tier != "REJECTED"


def geometry_validity(model: TransformModel | None,
                      max_anisotropy: float = 4.0) -> tuple[bool, str, dict]:
    """Cheap structural checks on the transform itself.

    Independent of how many points voted for it: a transform can have a
    healthy inlier set and still be a reflection or a wild shear, neither of
    which a nadir-viewing orbital pair can produce.
    """
    if model is None or model.matrix is None:
        return True, "geometry check skipped (no matrix model)", {}
    a = np.asarray(model.matrix, np.float64)[:2, :2]
    det = float(np.linalg.det(a))
    try:
        sv = np.linalg.svd(a, compute_uv=False)
        anisotropy = float(sv[0] / max(sv[-1], 1e-12))
    except np.linalg.LinAlgError:
        return False, "geometry check failed: SVD did not converge", {"det": det}
    stats = {"det": det, "anisotropy": anisotropy}
    if det <= 0:
        return False, f"transform mirrors the image (det={det:.3g})", stats
    if anisotropy > max_anisotropy:
        return False, (f"extreme anisotropic stretch (axis ratio {anisotropy:.2f} "
                       f"> {max_anisotropy})"), stats
    return True, f"geometry plausible (det={det:.3g}, anisotropy={anisotropy:.2f})", stats


def _tier_for(value: float | None, thresholds: dict[str, float], key: str) -> Tier:
    """Highest tier whose threshold this one signal reaches."""
    if value is None:
        return "LOW"        # unmeasured cannot justify confidence, but is not fatal
    for tier in ("high", "medium", "low"):
        need = thresholds.get(tier, {}).get(key)
        if need is None or value >= need:
            return tier.upper()  # type: ignore[return-value]
    return "REJECTED"


def accuracy_tier(acc: dict) -> tuple[Tier, str]:
    """The tier an independent accuracy measurement allows (audit I-08, docs/tier_accuracy_protocol.md).

    Probe p50/p95 in SOURCE px against PS-derived limits. Unmeasured (too few probes) allows MEDIUM at most:
    HIGH needs a measured accuracy. A poor measurement caps at LOW and never rejects on its own (probes can
    be biased by illumination)."""
    cfg = config.get("tiers.accuracy", {}) or {}
    n, p50, p95 = acc.get("n") or 0, acc.get("p50_px_src"), acc.get("p95_px_src")
    if n < int(cfg.get("min_probes", 20)) or p50 is None or p95 is None:
        return "MEDIUM", f"accuracy unmeasured ({n} probes): at most MEDIUM"
    for tier in ("high", "medium"):
        lim = cfg.get(tier, {})
        if p50 <= float(lim.get("p50_px_src", 0)) and p95 <= float(lim.get("p95_px_src", 0)):
            return tier.upper(), f"probe accuracy p50 {p50:.2f} / p95 {p95:.2f} src px -> {tier.upper()}"
    return "LOW", f"probe accuracy p50 {p50:.2f} / p95 {p95:.2f} src px is not sub-pixel -> at most LOW"


def assess(*, inlier_count: int | None = None, inlier_ratio: float | None = None,
           spatial_coverage: float | None = None,
           model: TransformModel | None = None,
           scale_ok: bool = True,
           scale_status: str | None = None,
           gates: dict[str, bool] | None = None,
           require_gates: bool = False,
           accuracy: dict | None = None) -> QualityVerdict:
    """Combine independent quality signals into a confidence tier.

    `scale_status` is EstimateResult.scale_status. It matters because a scale
    check can pass without CONFIRMING anything: "unverified" means the transform
    was not refuted by instrument sources that disagree with each other, and
    "abstained" means there were none. Neither may lift a result above LOW --
    the same rule that applies to any unmeasured signal (H1).
    """
    thresholds = config.get("tiers", {}) or {}
    notes: list[str] = []
    fms: list[int] = []
    signals: dict[str, object] = {
        "inlier_count": inlier_count,
        "inlier_ratio": None if inlier_ratio is None else round(float(inlier_ratio), 4),
        "spatial_coverage": None if spatial_coverage is None else round(float(spatial_coverage), 4),
        "scale_ok": bool(scale_ok),
        "scale_status": scale_status,
    }

    # Hard rejections first: these are not "low confidence", they are invalid.
    if gates:
        failed_gates = [name for name, passed in gates.items() if not passed]
        signals["failed_gates"] = failed_gates
        if failed_gates:
            notes.append(f"control gates failed: {', '.join(failed_gates)}")
            return QualityVerdict("REJECTED", signals, "control_gates",
                                  [FM_FALSE_CORRESPONDENCE], notes)
    elif require_gates:
        notes.append("no control gates were run; a result without gates is unverified (rule H4)")
        return QualityVerdict("REJECTED", signals, "control_gates", [FM_UNVERIFIED_EVALUATION], notes)

    if not scale_ok or scale_status in ("inconsistent", "degenerate"):
        notes.append("scale disagrees with the instrument GSD ratio")
        return QualityVerdict("REJECTED", signals, "scale_ok", [FM_SCALE_CONFUSION], notes)

    geom_ok, geom_msg, geom_stats = geometry_validity(model)
    signals.update(geom_stats)
    notes.append(geom_msg)
    if not geom_ok:
        return QualityVerdict("REJECTED", signals, "geometry", [FM_FALSE_CORRESPONDENCE], notes)

    # Graded signals. Every one must reach a tier for that tier to be claimed,
    # so a large inlier count cannot paper over a poor ratio.
    per_signal = {
        "inlier_count": _tier_for(None if inlier_count is None else float(inlier_count),
                                  thresholds, "min_inliers"),
        "inlier_ratio": _tier_for(inlier_ratio, thresholds, "min_inlier_ratio"),
        "spatial_coverage": _tier_for(spatial_coverage, thresholds, "min_coverage"),
    }
    if accuracy is not None:                      # I-08 (docs/tier_accuracy_protocol.md)
        per_signal["accuracy"], why = accuracy_tier(accuracy)
        signals["accuracy"] = {k: accuracy.get(k) for k in ("n", "p50_px_src", "p95_px_src")}
        notes.append(why)
    if scale_status in ("unverified", "abstained"):
        per_signal["scale"] = "LOW"
        notes.append(f"scale {scale_status}: the instruments could not confirm it, "
                     f"so the result is capped at LOW")
    signals["per_signal_tier"] = per_signal
    worst = max(per_signal.values(), key=TIER_ORDER.index)
    limiting = max(per_signal, key=lambda k: TIER_ORDER.index(per_signal[k]))

    if worst == "REJECTED":
        notes.append(f"{limiting} below the minimum acceptable level")
        fms.append(FM_FALSE_CORRESPONDENCE)
    else:
        notes.append(f"tier {worst}, limited by {limiting}")
    return QualityVerdict(worst, signals, limiting, fms, notes)
