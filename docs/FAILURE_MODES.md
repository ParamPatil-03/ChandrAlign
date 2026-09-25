# Failure modes and runtime logging

This document is the human-readable view of the frozen taxonomy in `PLAN.md`
section 15. The executable source of truth is
`src/chandralign/evaluate/failure_log.py`; `RegistrationResult.failure_modes`
contains only these integer IDs. Unknown IDs are rejected instead of being
silently displayed as valid failures.

| ID | Failure mode | Detection hook | Mitigation |
|---:|---|---|---|
| 1 | Featureless terrain | `preprocess.texture.terrain_scores` | Phase/self-similarity route |
| 2 | Severe shadows | `preprocess.shadow_mask.detect_shadows` | Mask or down-weight shadows |
| 3 | Huge scale ratio | `io.instruments.scale_precheck`, `matching.cascade.plan` | Coarse-to-fine cascade |
| 4 | Hyperspectral noise | `preprocess.iirs_composite.select_bands` | Reject weak IIRS bands |
| 5 | Spectral mismatch | `preprocess.iirs_composite.composite` | Pan-equivalent composite |
| 6 | Extreme rotation | `matching.filters.orientation_consistency`, `matching.rotation.rotation_search` | Rotation search/invariant descriptor |
| 7 | Large displacement | `matching.cascade.register_step_dense` | Metadata/DEM-guided coarse lock |
| 8 | Incorrect or stale metadata | `evaluate.groundtruth.cross_validate` | Independent checks and uncertainty |
| 9 | Wrong or missing projection | `geometry.projection.geolocation_model` | Explicit lunar reprojection |
| 10 | Insufficient overlap | `geometry.footprint.check_overlap` | Reject before matching |
| 11 | Repetitive terrain | `preprocess.texture.repetitiveness_score`, `matching.filters.adaptive_ratio` | Stricter ratio and geometry checks |
| 12 | False correspondences survive robust fitting | `estimate.robust.estimate`, `evaluate.quality.assess` | Reject/lower confidence and expose evidence |
| 13 | Scale confusion | `estimate.scale.check`, `evaluate.quality.assess` | Constrain or reject scale |
| 14 | Licence contamination | `matching.licence.assert_allowed`, `scripts/check_licences.py` | Block shipment and CI |
| 15 | Stub PDS parser | `io.pds_raster.verify_raster`, `tests/test_pds_real.py` | Genuine parser plus field provenance |
| 16 | Classical matcher polar collapse | `matching.regime.conditions`, `matching.regime.select` | Learned route and regime disclosure |
| 17 | Deep matcher extreme-sun collapse | `matching.regime.conditions`, `matching.regime.select` | Avoid unsupported conditions; disclose uncertainty |
| 18 | ISIS/ASP deployment fragility | **Manual only:** planetary build/deployment check | Container or native fallback |
| 19 | Fabricated or inflated evaluation | `evaluate.control_gates.run_all`, `evaluate.quality.assess` | Reject claim and retain gate evidence |
| 20 | Fine-tuning overclaim | **Manual only:** compare claim, provenance, and training logs | Say pretrained/no fine-tuning unless proved |

## Additional audited operational reasons

The code and verification evidence contain useful run failures that are not new
scientific taxonomy entries. They use stable string codes so the frozen 1–20
contract is not altered:

| Code | Meaning | Canonical relationship |
|---|---|---|
| `matcher_exception` | Matcher raised; pipeline returned a result rather than crashing | Supports #12 |
| `optional_geometry_skipped` | DEM or ground model absent, so geometry/parallax was skipped | Context only; not necessarily rejection |
| `unclassified_rejection` | `REJECTED` arrived without an ID | Evidence-integrity defect related to #19 |
| `missing_control_gates` | No gate evidence is attached | #19 |
| `incomplete_metrics` | A core metric is genuinely unmeasured | Context; it must remain `None` |
| `synthetic_real_gap` | Synthetic success has not generalized to real imagery | Claim risk related to #19 |
| `resource_or_dependency_failure` | Accelerator, memory, dependency, or optional output backend is unavailable | Operational context |

The last two are audit/register entries rather than claims that they can always
be inferred from a `RegistrationResult`. They require benchmark/provenance or
environment evidence respectively.

## Log format

`failure_log.log_run(path, result_or_bundle)` appends one JSON object per line.
Each record contains a schema name, UTC time, confidence tier, full definitions
for triggered canonical modes, and observed operational reasons with evidence.
JSON Lines is append-only, streamable, and keeps separate runs independently
parseable if a later run is interrupted.

The logger never changes the pipeline decision. Member B's quality path raises
canonical IDs; Member C resolves, validates, explains, and records them.
