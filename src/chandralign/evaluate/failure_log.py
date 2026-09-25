"""Canonical failure-mode register and append-only run logger (CHECK-09).

The integer IDs are the frozen twenty-mode taxonomy in PLAN.md section 15.
Operational problems found during the project audit are deliberately kept in a
separate string-keyed register: adding a new integer would change the Part 2 ->
Part 3 contract and make old results ambiguous.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Literal

DetectionKind = Literal["automatic", "manual"]


@dataclass(frozen=True)
class FailureMode:
    id: int
    name: str
    detection: str
    mitigation: str
    detection_kind: DetectionKind
    hooks: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationalReason:
    code: str
    description: str
    canonical_mode: int | None = None


@dataclass(frozen=True)
class FailureObservation:
    """One reason observed in a completed registration."""

    code: str
    description: str
    canonical_mode: int | None
    evidence: str


# Hooks name the code which performs (or records) the check. Modes that cannot
# be inferred from one run are explicitly manual rather than falsely automated.
FAILURE_MODES: dict[int, FailureMode] = {
    1: FailureMode(1, "Featureless terrain", "Low matchable-texture score.",
                   "Route to phase/self-similarity features.", "automatic",
                   ("preprocess.texture.terrain_scores",)),
    2: FailureMode(2, "Severe shadows", "Near-zero intensity and low local variation, optionally checked against illumination.",
                   "Mask shadows and exclude or down-weight them.", "automatic",
                   ("preprocess.shadow_mask.detect_shadows",)),
    3: FailureMode(3, "Huge scale ratio", "Instrument GSD preflight finds a ratio outside direct-matcher support.",
                   "Use the coarse-to-fine cascade.", "automatic",
                   ("io.instruments.scale_precheck", "matching.cascade.plan")),
    4: FailureMode(4, "Hyperspectral noise", "Per-band SNR and dead-pixel checks reject weak IIRS bands.",
                   "Select only supported bands.", "automatic",
                   ("preprocess.iirs_composite.select_bands",)),
    5: FailureMode(5, "Spectral mismatch", "Known sensor-band mismatch plus composite validation evidence.",
                   "Build a pan-equivalent IIRS composite.", "automatic",
                   ("preprocess.iirs_composite.composite",)),
    6: FailureMode(6, "Extreme rotation", "Orientation consistency or image-level rotation search detects a large angle.",
                   "Use rotation search or a rotation-invariant descriptor.", "automatic",
                   ("matching.filters.orientation_consistency", "matching.rotation.rotation_search")),
    7: FailureMode(7, "Large displacement", "Direct matches collapse before the coarse geolocation lock.",
                   "Use the DEM/metadata-guided coarse search.", "automatic",
                   ("matching.cascade.register_step_dense",)),
    8: FailureMode(8, "Incorrect or stale metadata", "Independent metadata and DEM routes disagree.",
                   "Report uncertainty and never trust one source.", "automatic",
                   ("evaluate.groundtruth.cross_validate",)),
    9: FailureMode(9, "Wrong or missing projection", "Output geolocation/projection validation fails.",
                   "Apply an explicit lunar reprojection.", "automatic",
                   ("geometry.projection.geolocation_model",)),
    10: FailureMode(10, "Insufficient overlap", "Ground-footprint intersection is below the configured minimum.",
                    "Preselect overlapping products before matching.", "automatic",
                    ("geometry.footprint.check_overlap",)),
    11: FailureMode(11, "Repetitive terrain", "Many descriptors have near-equal alternatives in the same tile.",
                    "Tighten ratio and geometry checks.", "automatic",
                    ("preprocess.texture.repetitiveness_score", "matching.filters.adaptive_ratio")),
    12: FailureMode(12, "False correspondences survive robust fitting", "Low evidence signals, implausible geometry, or a failed control gate rejects the result.",
                    "Reject or lower confidence; expose the failing evidence.", "automatic",
                    ("evaluate.quality.assess", "estimate.robust.estimate")),
    13: FailureMode(13, "Scale confusion", "Estimated transform scale disagrees with supported instrument GSD.",
                    "Constrain or reject the estimate.", "automatic",
                    ("estimate.scale.check", "evaluate.quality.assess")),
    14: FailureMode(14, "Licence contamination", "The dependency and model licence audit finds a non-permissive component.",
                    "Block it from ship mode and CI.", "automatic",
                    ("matching.licence.assert_allowed", "scripts/check_licences.py")),
    15: FailureMode(15, "Stub PDS parser", "Real-label field verification or anti-stub tests fail.",
                    "Use the genuine PDS parser and verify label provenance.", "automatic",
                    ("io.pds_raster.verify_raster", "tests/test_pds_real.py")),
    16: FailureMode(16, "Classical matcher polar collapse", "Solar-elevation regime metadata identifies a validated hard regime.",
                    "Route to the learned matcher and disclose the regime.", "automatic",
                    ("matching.regime.conditions", "matching.regime.select")),
    17: FailureMode(17, "Deep matcher extreme-sun collapse", "Solar-elevation regime metadata identifies the benchmarked extreme-lighting limit.",
                    "Avoid unsupported vanilla LoFTR conditions and disclose uncertainty.", "automatic",
                    ("matching.regime.conditions", "matching.regime.select")),
    18: FailureMode(18, "ISIS or ASP deployment fragility", "Detected manually in the planetary-toolchain build/deployment check; no ISIS/ASP runtime is shipped.",
                    "Use the containerised toolchain or the native fallback.", "manual"),
    19: FailureMode(19, "Fabricated or inflated evaluation", "Control gates, null tests, perturbations, or evidence-source checks fail.",
                    "Reject the claim and retain all gate evidence.", "automatic",
                    ("evaluate.control_gates.run_all", "evaluate.quality.assess")),
    20: FailureMode(20, "Fine-tuning overclaim", "Detected manually by comparing claims with training logs and model provenance.",
                    "State pretrained/no-fine-tuning unless training evidence exists.", "manual"),
}


# Additional failure reasons found by auditing current code and project evidence.
# They help operators, but are intentionally not new PLAN.md integer IDs.
OPERATIONAL_REASONS: dict[str, OperationalReason] = {
    "matcher_exception": OperationalReason(
        "matcher_exception", "The selected matcher raised an exception; the pipeline returned a rejected result.", 12),
    "optional_geometry_skipped": OperationalReason(
        "optional_geometry_skipped", "A DEM-dependent geometry/parallax stage was skipped because its inputs were unavailable."),
    "unclassified_rejection": OperationalReason(
        "unclassified_rejection", "The run was rejected without a canonical failure-mode ID.", 19),
    "missing_control_gates": OperationalReason(
        "missing_control_gates", "The result contains no control-gate evidence.", 19),
    "incomplete_metrics": OperationalReason(
        "incomplete_metrics", "One or more core registration metrics were not measured."),
    "synthetic_real_gap": OperationalReason(
        "synthetic_real_gap", "A claim is supported by synthetic evidence but has not generalized to real imagery.", 19),
    "resource_or_dependency_failure": OperationalReason(
        "resource_or_dependency_failure", "A runtime dependency, accelerator, memory, or optional output backend is unavailable."),
}


def validate_registry() -> None:
    """Raise if the documented taxonomy has drifted or contains a fake hook."""

    if set(FAILURE_MODES) != set(range(1, 21)):
        raise ValueError("failure-mode register must contain exactly IDs 1 through 20")
    for mode in FAILURE_MODES.values():
        if mode.detection_kind == "automatic" and not mode.hooks:
            raise ValueError(f"automatic failure mode {mode.id} has no detection hook")
        if mode.detection_kind == "manual" and "manual" not in mode.detection.lower():
            raise ValueError(f"manual failure mode {mode.id} lacks an explicit manual-only note")


def modes(mode_ids: Iterable[int]) -> list[FailureMode]:
    """Resolve IDs in input order, rejecting contract drift instead of hiding it."""

    resolved: list[FailureMode] = []
    for mode_id in dict.fromkeys(int(value) for value in mode_ids):
        try:
            resolved.append(FAILURE_MODES[mode_id])
        except KeyError as exc:
            raise ValueError(f"unknown failure-mode ID {mode_id}; expected 1..20") from exc
    return resolved


def inspect(result_or_bundle: Any) -> list[FailureObservation]:
    """Return canonical and operational reasons visible in a result/bundle."""

    result = getattr(result_or_bundle, "result", result_or_bundle)
    observations = [
        FailureObservation(f"FM-{mode.id:02d}", mode.name, mode.id, "RegistrationResult.failure_modes")
        for mode in modes(getattr(result, "failure_modes", ()))
    ]
    seen = {item.code for item in observations}

    def add(code: str, evidence: str) -> None:
        if code in seen:
            return
        reason = OPERATIONAL_REASONS[code]
        observations.append(FailureObservation(code, reason.description, reason.canonical_mode, evidence))
        seen.add(code)

    notes = [str(note) for note in (getattr(result, "notes", None) or ())]
    failed_note = next((note for note in notes
                        if "matcher " in note.lower() and " failed:" in note.lower()), None)
    if failed_note:
        add("matcher_exception", failed_note)

    gates = getattr(result, "gates", None)
    if not gates:
        add("missing_control_gates", "RegistrationResult.gates is empty")

    if getattr(result, "confidence_tier", None) == "REJECTED" and not getattr(result, "failure_modes", None):
        add("unclassified_rejection", "confidence_tier=REJECTED and failure_modes is empty")

    metric_values = getattr(result, "metrics", None)
    if metric_values is not None:
        missing = [name for name in ("inlier_count", "inlier_ratio", "spatial_coverage", "runtime_s")
                   if getattr(metric_values, name, None) is None]
        if missing:
            add("incomplete_metrics", "unmeasured: " + ", ".join(missing))

    for stage, detail in (getattr(result_or_bundle, "stages", None) or {}).items():
        if not isinstance(detail, dict) or detail.get("applied") is not False:
            continue
        reason = str(detail.get("reason", ""))
        if "no dem" in reason.lower() or "no ground model" in reason.lower():
            add("optional_geometry_skipped", f"{stage}: {reason}")

    return observations


def log_run(path: str | Path, result_or_bundle: Any, *, created_utc: str | None = None) -> dict[str, Any]:
    """Append one stable JSON-lines record and return the written dictionary."""

    result = getattr(result_or_bundle, "result", result_or_bundle)
    record = {
        "schema": "chandralign.failure-log.v1",
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "confidence_tier": getattr(result, "confidence_tier", None),
        "failure_modes": [asdict(mode) for mode in modes(getattr(result, "failure_modes", ()))],
        "observations": [asdict(item) for item in inspect(result_or_bundle)],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


validate_registry()
