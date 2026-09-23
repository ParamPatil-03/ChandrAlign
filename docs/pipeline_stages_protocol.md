# Protocol: turning on the three fine-stage steps (and the anti-aliasing fix)

Frozen and committed **before** any of the runs below. The decision rules are
not revised after seeing results; if one proves unworkable, that is reported as
such and a new protocol is written.

## Question

`pipeline.fine_stage` now runs three steps that were built and unit-tested but
called by no pipeline until 2026-09-23:

| stage | feature | why it exists |
|---|---|---|
| `uniformity` | ALIGN-04 | the problem statement mandates uniformly distributed match points |
| `subpixel` | PREC-01 | the problem statement mandates sub-pixel accuracy |
| `geometry_filter` | ALIGN-03 | research doc section 29: reject matches inconsistent with terrain |

Separately, `preprocess/resample.warp_affine` replaces `cv2.warpAffine`, which
silently ignored `INTER_AREA` (no anti-aliasing when TMC-2 is shrunk onto TC).

For each: does switching it on keep or improve real registration? Which ones
become the default in `configs/default.yaml`?

## Data and runs

Real TMC-2 `ch2_tmc_nca_20250207T1102039417` -> SELENE TC, the committed nine
windows (3 per tile N00 / N03 / N09, 1536 px), matcher `eloftr` (the routed
default), whole window, fp32, all control gates on. Command:
`python scripts/register_tmc2_tc.py --tile all --windows 3 [--stage NAME=on|off ...]`

| run | warp | uniformity | subpixel | geometry_filter |
|---|---|---|---|---|
| A | old (bilinear, aliasing) | off | off | off | ← `reports/tmc2_tc_registration.json`, reproduced exactly 2026-09-23 |
| B | anti-aliased | off | off | off |
| U | anti-aliased | **on** | off | off |
| S | anti-aliased | off | **on** | off |
| US | anti-aliased | **on** | **on** | off |
| USG | anti-aliased | **on** | **on** | **on** (needs SLDEM) |

## What is measured

- **Truth:** the perturbation gate's error: the source moved by a KNOWN (3, 4) px,
  the recovered shift compared with it. This is the only accuracy truth on real
  data, and it runs the same fine stage (the gate calls `pipeline.fine_stage`
  with the same switches).
- **Proxy, named as a proxy:** `vs_isro_refined_grid_m`, the distance to ISRO's
  own refined geolocation. ISRO fitted that grid against SELENE, so it is a
  comparison with another solution, not ground truth.
- Registration status, tier and every gate, per window.
- Delivered control points: count, coverage, max Delaunay gap.
- Internal precision: RMSE of the final model on the delivered points. This is
  NOT accuracy and is not used to decide anything.
- Runtime per window.

## Decision rules (frozen)

"Loses a window" means: a window that registers in the comparison run fails to
register, drops a tier, or fails a gate.

1. **Anti-aliasing (B vs A).** A signal-processing correctness fix, kept unless
   it harms: kept if it loses no window and the median perturbation error rises
   by no more than 0.10 px. Otherwise reverted and reported.
2. **uniformity (U vs B).** Mandated by the problem statement, so its job is the
   delivered distribution, not accuracy. Default ON if it loses no window and the
   median proxy distance moves by less than 1 TC pixel (7.4 m).
3. **subpixel (S vs B).** Default ON if it loses no window and the median
   perturbation error is no worse than B's + 0.05 px. If it is worse, OFF, and
   the 33 px window (never measured at that size) is the first suspect.
4. **Both (US).** If rules 2 and 3 both say ON, US must also lose no window
   against B; if it does, the stage whose single run lost least stays ON alone.
5. **geometry_filter (USG vs US).** Default ON only if it rejects at least one
   match somewhere AND loses no window. If it rejects nothing on all nine windows
   (expected: after a coarse lock both ends of a match sit in one frame, and the
   first robust estimate already keeps ~99% of matches), it stays OFF and the
   result is recorded as "no measurable effect on this pairing", not as a success.

## What this cannot show

Nine windows of one TMC-2 strip against one reference product, one matcher.
A null or small effect here says nothing about harder pairings (NAC, cross-modal),
where the filter and the refinement could matter more.

---

## Result of runs A-US, and why v1's rules are NOT applied (2026-09-23)

| run | perturbation error median / max (px, truth) | ISRO-grid proxy median (m) | s / window |
|---|---|---|---|
| A old warp | 0.082 / 0.464 | 312.4 | 10.5 |
| B anti-aliased | 0.081 / 0.208 | 312.4 | 10.9 |
| U + uniformity | **0.261 / 1.069** | 312.6 | 10.9 |
| S + sub-pixel (all inliers) | 0.082 / 0.154 | 312.4 | 50.1 |
| US both | **0.206 / 0.830** | 312.4 | 11.7 |

All nine windows register, at the same tiers, with every gate passing, in every run.

- Rule 1 (anti-aliasing): **kept**. Median unchanged, worst window 0.464 -> 0.208 px.
- Rule 2 (uniformity) says ON: it loses no window and moves the proxy by 0.2 m.
  **But rule 2 checked only the proxy, and the truth got 3x worse**, with one window
  past 1 px -- no longer sub-pixel. That is a defect in the rule as written, not a
  property the rule was meant to allow. Cause: the final fit on ~384 thinned points
  discards ~97% of ~15,000 correct matches. Coverage was already 1.0, so there was
  no clustering bias for thinning to remove; there was only precision to lose.
- Rule 3 (sub-pixel) says ON: no loss, +0.001 px. No gain either, at 5x the runtime.
- Rule 4 would switch both on: US loses no window, but carries the same 2.5x
  regression of the truth metric.

Applying v1 literally would ship a measured accuracy regression, so it is not
applied. As section "Question" of v1 allows, a v2 is written below and frozen
before its run.

## v2 (frozen before run V2)

**Design change.** The MODEL is always the first robust estimate on all (filtered)
inliers -- the evidence is not thrown away. Uniformity and sub-pixel refinement act
on the DELIVERED match points only: the problem statement's "match points ... with
uniform distribution" and "sub-pixel accuracy" are properties of what is delivered,
and the model keeps its full precision.

**Run V2:** anti-aliased warp, `uniformity=on`, `subpixel=on`, `geometry_filter=off`,
same nine windows, same command otherwise.

**Rules.**
1. V2 loses no window against B, and its perturbation error per window equals B's
   within the run-to-run noise seen between A and B on unchanged windows
   (|difference| <= 0.02 px at the median). By construction the model is B's; a
   larger difference means the construction is wrong.
2. Delivered points: coverage equal to B's; no grid cell holds more than top-k;
   at least 90% of delivered points refined; RMS residual of the refined delivered
   points to the model lower than of the same points unrefined.
3. If 1 and 2 hold: `uniformity: true`, `subpixel: true` become the defaults.
   `geometry_filter` is still decided by v1 rule 5 (run USG, on the v2 design).

## Result of run V2 (2026-09-23)

| row | B pert (px) | V2 pert | tier | coverage | delivered | refined | residual to model, unrefined -> refined (px) |
|---|---|---|---|---|---|---|---|
| 1562 | 0.064 | 0.064 | H | 0.984 | 378 | 378 | 1.866 -> **1.887** |
| 3125 | 0.119 | 0.119 | M | 0.906 | 337 | 334 | 1.847 -> **1.904** |
| 4687 | 0.208 | 0.208 | H | 0.859 | 320 | 318 | 1.742 -> **1.748** |
| 16000 | 0.041 | 0.041 | H | 1.000 | 384 | 377 | 1.106 -> 1.042 |
| 18750 | 0.079 | 0.079 | H | 1.000 | 384 | 377 | 0.699 -> 0.462 |
| 21500 | 0.177 | 0.177 | H | 1.000 | 384 | 380 | 0.981 -> 0.813 |
| 52000 | 0.081 | 0.081 | H | 1.000 | 384 | 380 | 0.632 -> 0.420 |
| 54750 | 0.120 | 0.120 | H | 1.000 | 384 | 383 | 1.331 -> 1.196 |
| 57500 | 0.022 | 0.022 | H | 1.000 | 384 | 379 | 0.875 -> 0.695 |

- Rule 1: **holds exactly** -- every window's perturbation error is B's to the
  last digit, so the model is untouched. Runtime 10.9 -> 16.2 s per window.
- Rule 2: coverage equal, top-k respected, >= 90% refined -- but the refined
  residual is HIGHER on the three N00 windows. **Rule 2 fails, so under v2 neither
  stage becomes a default.**

Recorded, not acted on: "residual to the affine model" is not a truth metric --
v1 itself says internal precision "is not used to decide anything", and v2 should
not have used it. The N00 windows have the largest residuals (~1.8 px), which is
what ground the affine model does not describe would look like; moving a point
to its TRUE position would then raise its residual. That is a hypothesis. v3
measures the delivered points against a known truth instead.

## v3 (frozen before its run)

**Question:** does per-point refinement move delivered match points closer to
their true positions on real imagery, with the real matcher?

**Truth:** `scripts/verify_point_refinement.py`. Real TMC-2 windows (native
3072 px, nine along-track positions spread over the strip) are block-averaged 4x
twice, the second copy starting an integer number of native pixels later, so the
pair differs by EXACTLY (0.25, -0.75) coarse px -- no interpolation, as
scripts/verify_subpixel.py does for PREC-06. Independent sensor-realistic noise is
added to each copy (the same noise model as verify_subpixel.py). `eloftr` matches
the pair; `fine_stage` runs with `uniformity=on` and `subpixel` off, then on. For
every delivered point, error = |(ref - src) - (0.25, -0.75)|.

**Rule.** `subpixel` becomes a default if, on at least 8 of the 9 windows, the
refined points' median error is lower than the unrefined points' median error
AND the refined 90th percentile is not higher than the unrefined one.
`uniformity` becomes a default if, on every window, it keeps coverage equal to
all inliers' coverage and no cell exceeds top-k (it cannot change the model: v2
rule 1). The two are decided separately.

**Limits:** same image, same sun, same sensor: this measures how precisely a
point is LOCATED, not cross-sensor accuracy -- the same limit PREC-06 states.
