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

## Amendment 1 (2026-09-26): C-02 move cap 0.75 -> 1.5 px, re-scored on a FRESH set

**What happened.** The method chosen on dev (commit `ec51334`: LSM + warped NCC, 41 px, Lanczos-scored
gate, `max_move_px` 0.75) was run once on the held-out set (`reports/known_warp_heldout.json`). It met
the bars in **14/15** cases. The failure was TMC-2 r90000, small warp, mild radiometry: p50 0.197 -> 0.066,
but p95 **0.790 = the unrefined p95**, so it did not beat unrefined at p95. By this protocol C-02 is
therefore **not accepted on the held-out set**, and that result stays recorded.

**Why.** A diagnostic on that case showed that 25 of 357 delivered points carry 0.6-1.9 px of matcher
error. The 0.75 px cap forbids moving 20 of them, so they keep their error and form the p95. With a 1.5 px
cap the same gate and LSM correct 24 of the 25 to <= 0.12 px. The 0.75 value came from the audit's advice
("~0.75 px"), not from measurement. On dev the 1.5 px cap had already measured better (worst p95 0.145
vs 0.172, `E1_lz` vs `E1_lz_075`). The score gate, not the cap, is what protects points; in the
inverted-shading case it refused every move.

**Amendment.** `subpixel.max_move_px` = 1.5 (the same as the pre-existing axis-aligned cap). Nothing else
changes. Because this was decided after seeing the held-out result, the held-out set can no longer
confirm it. A **fresh** set is fixed now, by the same validity rule (0 invalid pixels, inside the
product), from candidate lists written before any run: TMC-2 rows [140000, 10000, 125000] (take the first
two valid) and OHRC rows [70000, 10000] at col 4000 (take the first valid; row 70000 has one invalid
pixel, so it is excluded).

| set | crops | noise seed |
|---|---|---|
| **fresh** | TMC-2 rows 140000 and 10000 (col 1000, block 2), OHRC row 10000 (col 4000, block 4) | 13 |

The C-02 bars 1-2 must hold on the fresh set for C-02 to be accepted. The held-out set is re-run with the
amended setting and reported alongside, but it is labelled as post-hoc.

## Result: C-02 harness bars (2026-09-26)

`PYTHONPATH=src .venv/Scripts/python scripts/known_warp_harness.py --set <set> --out reports/known_warp_<set>.json`.
Delivered-point true error in ref px (p50 / p95). Unrefined = the matcher's positions; legacy = the
refinement shipped before; library = the refinement as it is now.

| set | setting | beats unrefined at p50 AND p95 | p95 <= 0.25 | worst p95 | legacy: beats / worst p95 |
|---|---|---|---|---|---|
| dev (choice) | cap 0.75 | 15/15 | 15/15 | 0.172 | 4/15, 1.385 |
| held-out | cap 0.75 | **14/15** | 14/15 | 0.790 | 3/15, 1.472 |
| **fresh (confirmation)** | **cap 1.5** | **15/15** | **15/15** | **0.220** | 4/15, 1.356 |
| held-out, post-hoc | cap 1.5 | 15/15 | 15/15 | 0.154 | (spent; labelled post-hoc) |

**C-02 harness bars 1-2: met on the fresh set, so accepted.** Across the fresh set, library p50 is
0.033-0.067 and p95 0.065-0.220, against unrefined p50 0.088-0.245 and p95 0.289-0.731. The
inverted-shading proxy is unchanged by refinement wherever it registered (0.917/2.623 on held-out): the
gate refuses moves it cannot justify.

The real-data bar (>= 90% of method-windows lower their residual-to-model) is measured together with C-03
on the committed real windows; see below.

## Result: C-03 / I-01 / G-02 harness bars (2026-09-26)

Settings chosen on dev (commit `5ce8d20`): Huber-IRLS fits of affine / parallax / TPS on <= 1000
grid-stratified refined inliers, **stratified** 5-fold CV (spatial-block read 2.2-5.3x the truth on dev,
stratified 1.2-3.6x), a 5% gain required of a richer model. Confirmed on the fresh set and on the held-out
set; no geometry choice was made on either. Files: `reports/known_warp_{dev,fresh,heldout}_geometry.json`
(`--geometry --no-legacy`). "True" = the RMS error of the delivered geometry on a 16 px grid inside the
inliers' convex hull.

| bar | dev | fresh | held-out | verdict |
|---|---|---|---|---|
| I-01: delivered geometry within 10% of the best candidate | 15/15 | 15/15 | 15/15 | **met** |
| G-02: TPS on all refined inliers, p50 within 0.02-0.09 px (non-rigid cases) | 0.031-0.066 | 0.028-0.065 | 0.030-0.061 | **met** |
| C-03 bar 1: reported check-point RMSE within 25% of the true RMS | 2/15 | 4/15 | 4/15 | **not met** |

What changed for a user, fresh set:
- Delivered geometry, rigid cases: the affine is chosen, with a true RMS of **0.003-0.007 px**. The previously
  shipped TPS (through the 384 points) had 0.040-0.101 px.
- Delivered geometry, non-rigid cases: TPS, with a true RMS of 0.039-0.096 px (previously 0.039-0.169).
- Reported "RMSE", previously 0.715-1.133 px against a true 0.04-0.10 px (7-30x). It is now 1.05-2.6x the
  truth on non-rigid cases. On rigid cases it reads 0.057-0.127 px against a true 0.003-0.007 px: that is
  the check points' matching noise, which no check-point statistic can get below (the caveat recorded
  before measuring).

**C-03 bar 1 is not met, and the bar is not changed.** The accuracy record therefore carries two labelled
figures:
- `rmse_px_ref`: the check-point RMSE (the audit's definition). It was at or above the truth in 44 of 45
  cases (0.86x once).
- `geometry_error_lower_px_ref`: split-half agreement of the chosen model. It was at or below the truth in
  41 of 45 cases (up to 1.48x, all on rigid cases with a true error under 0.005 px).

These are bounds, not the error itself.

## Result: the real-data bars (2026-09-26, Track B batches 1, 2 and 4)

**C-02 real bar** (refinement lowers the residual-to-model in >= 90% of method-windows):
- TMC-2 -> TC: **14/15 (93%): met.**
- OHRC -> NAC, routed: **14/24 (58%): not met.** eloftr windows 18/20 (90%); minima-loftr windows 5/14. The misses
  are the cross-illumination NACs (M175, M109, M1417), where intensity refinement has the least to hold on to.
  As caveated before measuring, residual-to-affine is not accuracy: the independent probes are.
- **Independent accuracy, TMC-2 -> TC** (matcher-free probes vs the delivered geometry, TMC-2 px): audit baseline
  pooled p50 0.46 / p95 1.32 (11/15 windows p95 > 1). After C-02 + G-02 + G-06 step 2: p50 0.135-0.267, p95
  0.369-1.333 per window, 1/15 windows p95 > 1 (`reports/trackb_batch4_g06_step2_summary.json`).

**C-03 bar 2** (check-point RMSE within a factor 1.5 of the probe p50 on TMC-2 -> TC): **not met**, 2/15 (batch 1)
and 1/15 (batch 2). The ratio is 1.3-2.6: an RMS over a heavy-tailed error distribution against a median. The
bar compares two different statistics; it is recorded as not met rather than redefined.
