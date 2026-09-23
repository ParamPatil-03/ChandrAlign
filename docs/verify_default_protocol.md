# Verification protocol: default matcher re-check, and xoftr speed

**Status: FROZEN. Committed before any measurement it governs was run.**
`git log` on this file must precede every report it names in section 4.
Branch `part2/verify-default`. Written by an independent verifier of the
`part2/default-matcher` work; PLAN.md is not a source of criteria here.

---

## 1. What is re-analysis and what is new measurement

**Re-analysis (no new data, no new thresholds).** The 720 rows of
`reports/default_matcher_selection.json`, the four `reports/tmc2_tc_*.json`
files and `reports/runtime_budget.json` are re-derived from their raw rows. I
have already SEEN the synthetic rows while re-counting them, so any re-scoring
of them is post hoc and is reported as such. It can refute, it cannot promote.

**New measurement (governed by the criteria below):**

- A. the real TMC-2 -> SELENE TC check (old protocol section 6) for the two
  candidates that re-scoring puts ahead of `eloftr` and that have never met real
  data: `sift-lightglue`, `disk-lightglue`;
- B. `xoftr` speed variants on the same 9 real windows, plus an fp16 synthetic
  check.

## 2. The scoping deviation, stated before the real data exists

`docs/default_matcher_protocol.md` made `cross_modal` a CORE regime, citing
"OHRC-to-IIRS". Read from the code, not the docs:

- `matching/regime.py::select` returns `matcher=None` plus the
  `cross_modal_candidates` list for any cross-modal pair. `default_matcher` is
  returned only on the same-modality branch.
- `select` checks scale first. OHRC->IIRS (320x), IIRS->TC (~11x) and IIRS->NAC
  (~160x) all route to the cascade before cross-modality is examined, and the
  cascade's steps default to `sift` / dense MIND, never `default_matcher`.
- The research document (section 1.2) records that the PS dataset names no
  Chandrayaan-to-Chandrayaan pairing.

So the default is re-scored on the same-modality regimes only: CORE =
`same_lighting, lighting_+30, resolution_2x, low_texture`; STRETCH =
`lighting_+90, lighting_opposite`. Every other rule of the old protocol
(eligibility 4.2-4.4, priority 5) is applied unchanged. Licence: all nine
pass per `reports/licence_audit.json`.

## 3. Criteria

### A. Real-data check for a default candidate (old protocol section 6, made concrete)

Run on the 9 committed windows (rows 1562 3125 4687 16000 18750 21500 52000
54750 57500, `--win 1536`, same script, same data). A candidate PASSES only if:

1. 9/9 windows reach `status: registered`;
2. no window is tiered `REJECTED` and every control gate passes;
3. **consistency**: on every window, the RMS over a 16x16 grid of the window of
   the difference between its composed transform and the committed `xoftr`
   transform is at most **2.0 TC px**. Source: `BAD_PX = 2.0`, the repository's
   accuracy bar before this work. This is agreement with an independent
   registration, not accuracy against truth; none exists for these windows.

The same three tests are applied to the committed `aliked-lightglue` and
`eloftr` results. For those two the outcome is already known to me, so their
verdicts are labelled post hoc.

### B. xoftr speed

**Target (from the brief):** the 9-window TMC-2 -> SELENE path under **180 s**.
Measured as the sum of per-window `seconds` from ONE fresh process, cold model
load included, exactly as the 948.3 s baseline was measured. Warm-only time is
also reported. Note, not a criterion: research doc section 38 gives the
TMC-2 segment of the demo 40 s, which no live 9-window run meets; a live demo
would show precomputed windows or fewer of them.

**Variants**, each in its own process, never two on the GPU at once:

| id | matcher call | gates |
|---|---|---|
| V0 | fp32, whole window (control: re-run of the committed baseline) | all, as now |
| V1 | fp16 autocast, whole window | all |
| V2 | fp32, tiled | all |
| V3 | fp16, tiled | all |
| G1 | best of V1-V3 | perturbation gate reuses the main registration as its baseline |

Tiling, fixed here: the source window is split into an n x n grid of equal
tiles, n = ceil(side / 640) (640 = the size XoFTR's weights were trained at);
each source tile is matched against the reference region it covers, enlarged by
a **64 px** margin on every side (the largest residual shift after the coarse
stage in the committed runs is 22 px). Matches are shifted back to window
coordinates and concatenated; the estimator runs once on the union. Tiling is
valid only for a PRE-ALIGNED pair and is not applied anywhere else.

A variant is **acceptable** only if ALL hold:

1. 9/9 registered, every gate passes, and no window's tier is worse than V0's;
2. **accuracy**: on each window, the grid RMS difference between its composed
   transform and the COMMITTED `xoftr` transform is no larger than the
   committed `eloftr`-vs-`xoftr` difference on that window: 0.477, 0.423,
   0.697, 0.401, 0.408, 0.352, 0.388, 0.554, 0.393 TC px (row order above).
   Rationale: a speed change must move the answer less than swapping to the
   matcher it is being compared against. V0's own difference from the
   committed run is reported as the run-to-run noise floor;
3. **fp16 only, synthetic**: on seeds 101-110, `xoftr` under fp16 keeps at least
   9/10 gate-accepted successes on each of `same_lighting, lighting_+30,
   resolution_2x, low_texture, cross_modal`, with 0 false confidences over
   those 50 runs (old protocol 4.2/4.3, unchanged). Tiling is not tested
   synthetically: those pairs are not pre-aligned.

**Gate scheduling** beyond G1 (null and identity gates once per reference tile
instead of per window) weakens verification. It is timed and reported with the
loss stated, and it is **not** allowed to count toward meeting the target.

Tile-edge behaviour is reported descriptively: the share of inliers within
64 px of an internal tile boundary, and their residual against interior inliers.

## 4. Reports this governs

`reports/verify_default_rescore.json` (re-analysis, `source: synthetic`),
`reports/tmc2_tc_sift_lightglue.json`, `reports/tmc2_tc_disk_lightglue.json`,
`reports/xoftr_speed_*.json` (`source: measured`),
`reports/xoftr_fp16_synthetic.json` (`source: synthetic`).

## 5. Standing commitment

No threshold above changes once any governed report exists. If a measurement
contradicts what I expect, the measurement is reported.
