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
