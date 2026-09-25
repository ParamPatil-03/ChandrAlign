# Protocol: geometry-aware sub-pixel refinement and the delivered geometry (C-02, C-03, I-01; G-01, G-02). Frozen before measuring.

Work order: `docs/AUDIT_2026-09-26.md`, Track B. This file fixes the data, the method-selection
rules and the acceptance bars BEFORE any new method is measured. The acceptance bars are the
audit's own, so they come from outside this experiment's data. Amendments, if any, go at the end
with their date and reason, and they never replace the original text.

## Harness

`scripts/known_warp_harness.py`: the audit's known-warp construction, which is now committed. A real
CH-2 crop is resampled through a KNOWN map G (a shift of (0.37, -0.62), plus rotation, scale and an optional
non-rigid field), block-averaged, and given independent sensor-like noise. Truth for any source
point = G^-1. Errors are in reference px unless stated.

Two disjoint sets:

| set | crops | noise seed | used for |
|---|---|---|---|
| **dev** | TMC-2 rows 60000 and 110000 (col 1000, block 2), OHRC row 30000 (col 4000, block 4) | 7 | method development and every choice |
| **held-out** | TMC-2 rows 30000 and 90000 (col 1000, block 2), OHRC row 50000 (col 4000, block 4) | 11 | confirmation ONLY; no choice may be made on it |

The held-out crops were chosen by a validity rule (0 invalid pixels, inside the product) before any
result existed. The TMC-2 product is `ch2_tmc_nca_20250207T1102039417` and the OHRC product is
`ch2_ohr_ncp_20240330T0035085365` (the first `*_d_img_d18.xml` of each in `data/raw`, as the audit used).

Scored cases per set (15), as the audit scored them:
3 crops x {small warp (2.5 deg, x0.93) same non-rigid, small same rigid, small mild non-rigid,
large warp (10 deg, x0.60) same non-rigid, large mild non-rigid}. The inverted-shading proxy is run
and recorded, but not scored.

## C-02 / G-01: per-point refinement

**Baseline, from the audit, dev set:** the shipped refinement is worse than no refinement in 11/15
cases, and its p95 reaches 1.29-1.39 on large warps.

**Change under test:** in `refine/subpixel.refine_points`, resample the source patch through the model's
local Jacobian into the reference patch's geometry, estimate the residual with least-squares matching
(affine + radiometric invariance) and warped NCC as the fallback, apply a move only if the match score
increases, and cap the move lower.

**Acceptance (audit C-02), per set:**
1. The refined points beat the unrefined points at **both p50 and p95** in **15/15** scored cases.
2. p95 <= **0.25 ref px** in every scored (non-inverted) case.
3. The same bars hold on the **held-out** set. Meeting them on dev alone is reported as "dev only", not
   as accepted.

**Real data (audit C-02, second bar):** on the real OHRC -> NAC and TMC-2 -> TC windows, the refinement
lowers the residual-to-model (the fine stage's `stages.subpixel.residual_to_model_px`, refined vs
unrefined, against the model the stage reports) in >= 90% of method-windows. The baseline on the
committed reports is 22/56 (OHRC -> NAC) and 7/9 (TMC-2 -> TC). Caveat, stated now and not after the
fact: on terrain with parallax or non-rigid distortion, a point that moves closer to the truth can move
AWAY from an affine. So this bar is necessary evidence, not sufficient evidence, and the harness bars
above are the accuracy test.

## C-03: the reported RMSE

**Change under test:** hold out ~20% of the inliers, stratified by grid cell, before the delivered
geometry is fitted. Report the **check-point RMSE of the geometry the product warps with** (`rmse_px_ref`,
plus `rmse_px_src` and `rmse_m` where the scale is known), tagged with the point set and the model. The
old figure is kept only as `fit_residual_px`. `rmse_px` is None below `estimate.min_inliers`.

**Acceptance (audit C-03):**
1. On the harness, the reported check-point RMSE is within **25%** of the true dense-grid RMS error of
   the delivered geometry (inside the inliers' convex hull), in every scored case of each set.
2. On TMC-2 -> TC, it agrees with the NCC-probe p50 to within a factor of **1.5**.

Caveat, stated now: a check point carries its own matching noise, so check-point RMSE is roughly
sqrt(geometry error^2 + point noise^2). Where the geometry error is much smaller than the point noise
(rigid affine cases, ~0.03 px), bar 1 may not be met by ANY check-point estimator. If so, it is reported
as not met, with the measured ratio. Neither the bar nor the statistic is changed afterwards.

## I-01 / G-02: which geometry is delivered

**Change under test:** candidates are affine (the robust estimate), and TPS fitted on ALL refined
inliers with robust (Huber) IRLS reweighting and smoothing chosen by cross-validation. The delivered
model is chosen per registration by **spatial-block cross-validation** on the refined inliers. The
parallax model, where one exists, is judged by the same CV. The 384 uniform points remain the EXPORTED
match points.

**Acceptance (audit I-01):**
1. On the harness, the selected model's true dense-grid RMS is never worse than the best single
   candidate's by more than **10%**, in every scored case of each set.
2. TPS on all LSM-refined inliers reaches **0.02-0.09 px** (median true error, non-rigid cases).

## Recording

Each run writes `reports/known_warp_{dev,heldout}.json` with the per-case p50/p95/RMS/max and the
verdict lines. The PR states the command and the numbers.
