# Ablation (research doc section 33) on real TMC-2 -> SELENE TC (fixed before running)

Descriptive (no adopt/reject decision): what each stage adds, on 4 windows fixed now -- hilly N00
rows 3125 and 4687, flat rows 16000 and 54750. Configurations (cumulative):
- C0 baseline: SIFT + robust estimate (MAGSAC), every optional stage off
- C1: + deep features (eloftr, the routed matcher), stages off
- C2: + terrain filter (ALIGN-03, SLDEM)
- C3: + terrain parallax (ALIGN-08, TC DTM, height at the ground point)
- C4: + uniformity, sub-pixel, TPS (= the adopted configuration)
Always on in all configs (not toggleable): the MIND coarse lock (illumination-robust, scale-bridging).
Metrics per window: registered, gates pass, tier, inliers, coverage, known-shift error (source px),
and ACCURACY = median NCC-probe error of the delivered geometry (affine, or affine+parallax in
C3-C4) in source px (scripts/parallax_probe_eval.py probes; matcher-free proxy).
Predictions: C1 >> C0 on inliers and coverage; C3 cuts hilly-window error; C4 lifts coverage.

## Result (2026-09-25) -- `reports/ablation_tmc2_tc.json`
Medians over the 4 windows; errors in TMC-2 (source) px; accuracy = matcher-free NCC-probe error.

| Config | gates | tiers | inliers | coverage | known shift | accuracy hilly | accuracy flat |
|---|---|---|---|---|---|---|---|
| C0 SIFT (vismatch `sift-nn`), stages off | 4/4 | 4 HIGH | 457 | 0.88 | 0.47 | 4.44 | 1.05 |
| C1 + eloftr | 4/4 | 3 HIGH, 1 MED | **10,884** | 0.95 | **0.18** | 4.08 | 1.00 |
| C2 + terrain filter | 4/4 | same | 10,756 | 0.95 | 0.18 | 4.33 | 0.97 |
| C3 + terrain parallax | 4/4 | same | 10,756 | **1.00** | 0.18 | **0.69** | **0.40** |
| C4 + uniformity, sub-pixel, TPS | 4/4 | same | 10,756 | 1.00 | 0.18 | 0.69 | 0.40 |

Reading:
- Deep features (C1): 24x the inliers, better coverage, 2.6x better precision (known shift) -- but NOT
  better accuracy: on this near-matched pair SIFT already locks, and both share the affine's limit.
- The terrain filter (C2) changes little here (flat-ish windows, SLDEM).
- **Terrain parallax (C3) is the accuracy step: hilly 4.1 -> 0.69 src px, flat 1.0 -> 0.40.**
- C4's stages do not change the MODEL (they shape the delivered points; by design). Their measured
  value is elsewhere: TPS -40% held-out residual (docs/tps_protocol.md), sub-pixel point refinement.
- Prediction "C1 >> C0" held for inliers/coverage/precision, not accuracy; "C3 cuts hilly error" held;
  "C4 lifts coverage" did not (already 1.0 after C3).
- The MIND coarse lock and scale cascade are always on (not ablated); C0's success depends on them.
