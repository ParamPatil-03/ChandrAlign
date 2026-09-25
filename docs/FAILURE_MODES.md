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

## Additional anticipated failure conditions

The following conditions were found in the broader Part 3 audit. They are
documented here so they cannot be forgotten during later integration and
full-resolution testing. They are **not additional integer IDs**, and they are
**not all automatically detected yet**. The status column distinguishes an
existing partial safeguard from work that still needs a dedicated detector.

| Proposed operational code | Condition and possible effect | Current safeguard/status | Canonical relationship |
|---|---|---|---|
| `radiometric_saturation` | Clipped bright or dark regions become flat and produce no trustworthy structure. | Shadow and texture scoring provide partial coverage; a saturation-fraction check is still needed. | #1, #2 |
| `input_integrity_failure` | A corrupt/truncated raster or label, wrong dimensions, datatype, byte order, or missing lines can yield invalid pixels or metadata. | PDS parsing and `io.pds_raster.verify_raster` cover part of this; complete raster-length and decoded-shape checks are still needed. | #15 |
| `band_selection_mismatch` | A wrong IIRS band index/order can create a misleading composite or compare incompatible wavelengths. | Band SNR/selection and provenance provide partial coverage; verify selected indices against label wavelengths. | #4, #5 |
| `pushbroom_jitter` | Spacecraft jitter or line-timing distortion produces row-dependent geometry that one affine/homography cannot model. | TPS can absorb some local residual, but no dedicated row-residual/jitter detector exists yet. | #12 |
| `dem_quality_insufficient` | A present but coarse, stale, void-filled, or locally incorrect DEM can make parallax correction worse. | Missing DEM is logged; DEM resolution, void fraction, and correction improvement still need explicit checks. | #8, #12 |
| `terrain_occlusion` | Crater walls or relief visible in one acquisition may be hidden in the other, creating physically impossible correspondences. | Geometry filtering provides partial coverage; an explicit visibility/occlusion test is not yet implemented. | #12 |
| `control_points_clustered` | Low residuals in one small region may coexist with poor registration elsewhere. | Spatial coverage and maximum Delaunay gap already expose much of this risk; report both values and reject inadequate coverage. | #12, #19 |
| `warp_extrapolation` | TPS or another nonlinear warp can behave unrealistically outside the convex hull of delivered control points, especially near image edges. | No dedicated boundary-distortion/Jacobian check yet; avoid claiming accuracy outside supported control-point coverage. | #12, #19 |
| `coordinate_convention_error` | Row/column, x/y, zero/one-based, or pixel-centre/corner confusion can create a consistent offset or transposition. | Round-trip and export tests cover known paths; every new import/export integration needs coordinate-contract tests. | #9, #19 |
| `projection_boundary_case` | Polar geometry, longitude wrap at +/-180 degrees, lunar-radius choice, or axis-order mistakes can shift otherwise plausible output. | Explicit lunar CRS handling provides partial coverage; add polar and antimeridian round-trip fixtures. | #9 |
| `nodata_mask_error` | Incorrect nodata metadata can expose black borders, fill values, or invalid pixels to the matcher as features. | Separate valid/shadow masks help; decoded nodata values must also be verified against label metadata. | #1, #12, #19 |
| `correlated_matcher_consensus` | Several matchers can agree because they share training data, descriptors, preprocessing, or the same systematic bias; agreement is not ground truth. | Independent control gates and perturbation tests reduce the risk; matcher votes must never replace external validation. | #19 |
| `ground_truth_error` | Incorrect reference coordinates can penalise a correct result or make an incorrect result appear accurate. | Metadata/DEM cross-validation provides partial coverage; retain uncertainty and the provenance of every reference measurement. | #8, #19 |
| `partial_output_failure` | Registration succeeds but GeoTIFF, CSV/GeoJSON, visualization, or report export fails, leaving an incomplete deliverable. | Individual exporters have tests; the final run manifest/report must verify that every required artifact was produced and readable. | Operational only |
| `configuration_drift` | Evaluation, CLI, UI, or demo uses different thresholds, matcher settings, or stage switches, making results irreproducible. | Provenance records configuration; add a run-manifest/config fingerprint comparison across all entry points. | #19 |

Until a dedicated detector is implemented, these entries are risk-register
items rather than automatically emitted observations. When implemented, they
must use string codes (as above) or map to an existing canonical ID; the frozen
`RegistrationResult.failure_modes` integer contract remains 1–20.

## Log format

`failure_log.log_run(path, result_or_bundle)` appends one JSON object per line.
Each record contains a schema name, UTC time, confidence tier, full definitions
for triggered canonical modes, and observed operational reasons with evidence.
JSON Lines is append-only, streamable, and keeps separate runs independently
parseable if a later run is interrupted.

The logger never changes the pipeline decision. Member B's quality path raises
canonical IDs; Member C resolves, validates, explains, and records them.
