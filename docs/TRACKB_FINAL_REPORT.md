# Track B final report: accuracy fixes for the 2026-09-26 audit

Branch **`track-b/accuracy`** (worktree `chandrayana2-trackB`), based on `8f87746`, the same base as Track A.
Work order: `docs/AUDIT_2026-09-26.md`, section 2 (Track B: C-02, C-03, C-04, the I-items tagged [B], G-01...G-10).
**Not pushed. Not merged.** Track A merges (see section 9).

---

## 1. Summary

| | |
|---|---|
| Items in Track B's scope | 21 (C-02, C-03, C-04, I-01, I-02, I-05, I-06, I-07, I-08, I-09, I-10, I-11, I-15, I-16, I-17, G-02...G-10) |
| **Done, with evidence** | 16 |
| Done, but a bar is honestly NOT met (recorded, not redefined) | C-03 (bars 1 and 2), C-02 real bar on OHRC, G-06 (14/15), G-03 decision "no RoMa model recommended" |
| Implemented, measured, **not adopted** (switchable, default off) | G-05 part 1 (MIND refinement fallback), G-07 (within-cell spread), G-06 step 1 (parallax + residual TPS) |
| **Blocked on data or scope** (plans in `docs/trackb_blocked_items.md`) | G-04 (independent truth), G-05 part 2 (rendered references), G-10 (bundle adjustment); fresh PRODUCTS for I-16 |
| Tests | full suite green (the only skip is `test_api.py`, which Track A's C-06 fixes); 83 test files |
| Sabotage (`scripts/sabotage.py --target all`) | **90/90 caught**: the audit's 70 plus 20 new, one per Track B fix |
| Adopted real-data results | **reproduce bit-for-bit** under strict checks (TMC-2->TC 15/15, IIRS->WAC 5/5, IIRS->NAC 9/9, TMC-2->NAC 45/45). OHRC->NAC differs on exactly the 8 MI-flagged windows (intended, I-07) |
| Commits | 39 on top of `8f87746` |

**The headline numbers:**

| what | before (audit) | after (Track B) |
|---|---|---|
| Sub-pixel refinement of delivered points, known warp, worst p95 | 1.39 px (worse than NO refinement in 11/15 cases) | **0.22 px**, better than unrefined in 15/15 (fresh set) |
| TMC-2 -> TC accuracy (matcher-free probes, TMC-2 px) | p50 0.36-0.76, p95 0.77-1.87; **11/15 windows p95 > 1** | p50 **0.135-0.267**, p95 **0.37-1.33**; **1/15 windows p95 > 1** |
| Delivered geometry on rigid pairs (true error) | 0.04-0.10 px (TPS always shipped) | **0.001-0.007 px** (affine chosen) |
| Reported "RMSE" vs truth | 0.7-1.1 px against a true 0.04-0.1 (7-30x off) | a labelled check-point bound, 1.05-2.6x on non-rigid pairs |
| Consistently wrong answers passing every gate (synthetic, 12 cases) | 12/12 accepted | **10/12 rejected**, 2 remaining are 2.2-2.5 px and not HIGH; **0/64 false alarms** |
| Cross-check false alarms on real hilly TMC-2 windows | (gate not wired) -> first version: 7/15 | **0/15** after amendment 1 |
| OHRC -> NAC "success" | 20/25, 8 of them MI-flagged | **12 success + 8 unconfirmed** (honest) |
| OHRC opposed-sun NAC M175124932LC | 0 solved (5 MI-flagged) | `xoftr`: **5/5 solved, MI clean; confirmed 4/4 on fresh windows** (G-09, section 6) |

---

## 2. How the work was done (and how to check it)

- **Frozen protocols.** Every fix that needed a measurement got a protocol committed BEFORE it was measured
  (`docs/*_protocol.md`, listed per item below). Amendments are dated, give their reason, and were re-scored
  on data not used for the change.
- **Dev / held-out / fresh sets.** Choices were made on the dev set only. When a held-out result failed (C-02,
  14/15), that failure is kept in the record, and the amended method was scored on a third, fresh set fixed
  by rule before running.
- **Real data from frozen snapshots.** Every real-data run used `git archive` of a named commit, never the
  live tree (audit rule 0.4.2). The snapshots were `db3a350` (batch 1), `74171b3` (batches 2, 2b, 5, 6),
  `13afb81` (batch 3), and `74171b3` + 2 config values (batch 4).
- **Reproducible numbers.** `scripts/trackb_batch_summary.py` and `scripts/trackb_matcher_matrix.py` regenerate
  every table here from the run outputs (`reports/trackb_*.json`).

```bash
cd chandrayana2-trackB
PYTHONPATH=src MPLBACKEND=Agg ../chandrayana2/.venv/Scripts/python -m pytest -q          # green
PYTHONPATH=src ../chandrayana2/.venv/Scripts/python scripts/sabotage.py --target all        # 90/90
PYTHONPATH=src ../chandrayana2/.venv/Scripts/python scripts/known_warp_harness.py --set fresh --out /tmp/f.json
```

---

## 3. Item by item

Status key: **DONE** = fix in, bar met. **DONE / bar not met** = fix in and useful, but the audit's bar is not
reached, and it says so. **NOT ADOPTED** = implemented and measured, but kept off by the frozen rule.
**BLOCKED** = needs data or time this project does not have.

### Critical

| item | status | what changed | evidence | protocol |
|---|---|---|---|---|
| **C-02** sub-pixel refinement made points worse | **DONE** (harness); real bar: TMC-2 met, OHRC not | `refine/subpixel.refine_points`: the source patch is resampled through the model's local Jacobian; LSM (ECC affine, NCC-seeded, divergence refused) plus warped NCC; a move is applied only if the Lanczos-scored NCC rises; cap 1.5 px | Known warp, fresh set: 15/15 better than unrefined at p50 and p95, worst p95 **0.220** (bar 0.25); legacy 4/15, p95 1.36. Held-out 14/15 at the first cap (0.75 px): recorded, amended, re-scored fresh. Real: residual lowered in TMC-2 14/15 (93%, met), OHRC 14/24 (58%, **not met**; the cross-illumination NACs) | `refinement_geometry_protocol.md` |
| **C-03** reported RMSE was not accuracy | **DONE / bar not met** | `metrics.rmse_px` = check-point RMSE of the DELIVERED geometry (stratified 5-fold CV); `rmse_m`; `provenance["accuracy"]` names the model and point set, keeps `fit_residual_px` (the old number), adds a split-half lower bound | Harness: bound 1.05-2.6x the truth (non-rigid), was 7-30x. Bar 1 (within 25%): 4/15 (fresh), **not met**: check points carry their own ~0.07 px matching noise, the floor predicted before measuring. Bar 2 (within 1.5x of the probe p50): 1-2/15, **not met** (an RMS against a median over heavy tails) | same |
| **C-04** the cross-check gate was never called | **DONE** | `register_bundle` runs RIFT2 (a different family) via `control_gates.independent_crosscheck`: flag -> gate fails (REJECTED, mode 12); no confident checker -> "inconclusive", tier capped at MEDIUM. **Amendment 1:** the checker's matches are scored against the DELIVERED geometry (affine vs affine false-alarmed on relief) | Synthetic end to end: **8/12** caught (bar >= 8), 10/12 rejected, **0/64 false alarms**, unchanged after the amendment. Real: TMC-2 7 flags -> **0** after amendment 1; OHRC 0; IIRS 0; fresh 0. Real true positives: it flagged 6 bad `minima-loftr` TMC-2 windows (probe p50 2.4 px). The periodic-aliasing trap was **not built** (follow-up) | `crosscheck_protocol.md` |

### Important

| item | status | what changed | evidence |
|---|---|---|---|
| **I-01** TPS always overrode the affine | **DONE** | `estimate/selection.py`: affine / parallax / TPS each fitted (Huber IRLS) on up to 3000 refined, grid-stratified inliers, scored on the same CV folds; a richer model must win by 5%. `RegistrationBundle.geometry` / `.geometry_model`; `best_geometry` honours them | Harness: the delivered geometry is within 10% of the best candidate **15/15** on dev, held-out and fresh |
| **G-02** fit on all refined inliers | **DONE** | as above | TPS p50 0.028-0.065 px (bar 0.02-0.09); rigid affine 0.001-0.007 px |
| **I-02** estimator refusal ignored | **DONE** | `fine_stage` returns ok=False when `robust.estimate` refused; a scale refusal stays mode 13 | the audit's exact case (10 true + 14 random) is now REJECTED; test + sabotage |
| **I-05 / G-03** RoMa family never benchmarked | **DONE**: no RoMa model recommended | fairness fixes first (M-06: 5000 samples for dense models; certainty floor), explicit `--benchmark-models` (recorded as `ship_mode: false`) | Section 5. `roma`: the most precise matcher on OHRC (known shift 0.003-0.07 px) but it **hallucinates on the null gates** (RoMa's sampler saturates certainty to 1.0 even on pure noise), so it is REJECTED everywhere. `minima-roma`: 24/24 synthetic, but on real data worse than current matchers. `ufm` / `matchanything-roma`: 3 / 5 synthetic false confidences. `gim-dkm`: does not fit a 6 GB GPU. `romav2`: benchmark only (DINOv3 licence) |
| **I-06** licence gate was a denylist | **DONE** (Track A did it too; reconcile at merge) | allowlist = `licence_audit.json` pass rows + `shippable_matchers` + our own methods; master, duster, gim-lightglue, omniglue, romav2 get named reasons | test: every `vismatch.available_models` entry off the allowlist is refused in ship mode |
| **I-07** OHRC success ignored the MI check | **DONE** | MI flag is a routing fallback trigger; outcomes are success / unconfirmed / failed. Rule R stays **off** (its own frozen protocol rejected it; the audit's suggestion would overturn a negative result) | 12 success / 8 unconfirmed, exactly the audit's estimate; strict diff changes exactly those 8 windows |
| **I-08** tiers carried no accuracy | **DONE** | matcher-free probes (`evaluate/probes.py`, moved from scripts) of the delivered geometry join the tier; PS-derived limits (HIGH p50 <= 0.5 and p95 <= 1.5 src px); unmeasured caps at MEDIUM; never rejects alone. **Fix found by measurement:** the probes now measure in the geometry's own frame (a perfect geometry read 0.64 px across an 8 deg rotation; it reads 0.01 px now) | TMC-2 12 HIGH / 3 MEDIUM (unchanged); IIRS p95 ~0.23 IIRS px | `tier_accuracy_protocol.md` |
| **I-09** fallbacks obeyed by one script only | **DONE** | `routing.run_candidates` (the one loop); `register_to_map --matchers` and a `routed` mode; cross-modality decided on the matched band (`matched_as`); map pairings routed on the real tile scale (TMC-2 -> MI is 3x, not the nominal 4.00) | tests + 2 sabotages. `representation` left advisory (wiring it needs real evidence) |
| **I-10** metrics never filled; wrong ground model | **DONE** (fine-stage part) | `fine_stage(ref_ground_model=...)`; `max_delaunay_gap_px`, `subpixel_recovery_err_px` filled. The coarse lock and the new `register_bundle` parameters are Track A's C-01 | test: the reference frame 250 px east is kept (it was thrown away) |
| **I-11** accuracy quoted at the median only | **DONE** | p50 / p95 in SOURCE px everywhere in Track B's records | section 7 |
| **I-15** reproduction check could pass a regression | **DONE** | keyed on the union of windows; tier / inliers / known shift / offset at bit-level tolerances; `--out` | the audit's clean run of main reproduces 99/99; a dropped window, a changed tier or a 1 m offset now fail |
| **I-16** rules amended after seeing results | **DONE as far as the data allows** | rule-based fresh WINDOWS (`--fresh`) | TMC-2 fresh 5/5 HIGH, p95 **0.45-0.68** TMC-2 px; OHRC fresh 33% vs 37% committed (consistent). Fresh PRODUCTS: blocked |
| **I-17 / G-08** ISRO refined geolocation unused | **DONE** (leave-one-out) | `geoprior.load()` falls back to the label's refined-minus-system corners (search prior only, never evaluation) | search residual TMC-2 **4683 -> 571 m**, IIRS 12900 -> 1198 m (88% / 91%). Lock test: all 10 windows locked at the same place with a **1.5 km** search instead of 7 km; 9/10 same tier, 1 marginal gate result (1.58 vs 1.5 px) on a hilly window |

### Upgrades

| item | status | result |
|---|---|---|
| **G-01** | **DONE** | = C-02 |
| **G-04** independent truth | **BLOCKED** (data) | plan: LROC NAC DTM / ortho tie points, crater centroids |
| **G-05** illumination-invariant refinement | part 1 **NOT ADOPTED**; part 2 **BLOCKED** | MIND fallback: never worse (0/45), opposite-sun p50 0.79 -> 0.34, but p95 unchanged -> bar not met. Rendered references need a NAC DTM |
| **G-06** TMC-2 tail | **step 2 ADOPTED / bar not met (14/15)** | windows p95 > 1 TMC-2 px: **8 -> 1**; p50 better on every window. Step 1 (parallax + residual TPS) broke its guard: off. Cost: the TMC-2 fine stage is ~2.3x slower |
| **G-07** uniformity inside cells | **NOT ADOPTED** (switchable) | NN-CV 0.865 -> 0.42, but a small real p95 cost (+~0.005 px); frozen rule said no |
| **G-09** default matcher with the cross-check live | **DONE** (confirmed on fresh windows) | keep `eloftr` for TMC-2 -> TC; **OHRC -> NAC routing `eloftr` -> `xoftr` -> `minima-loftr`**: fresh windows 9 success + 3 unconfirmed vs 5 + 7 today (section 6); ready to adopt at merge |
| **G-10** sensor model + bundle adjustment | **BLOCKED** (scope: weeks) | stepwise plan in `docs/trackb_blocked_items.md` |

---

## 4. What changed where (for code review and the merge)

| file | change |
|---|---|
| `src/chandralign/refine/subpixel.py` | C-02 geometry-aware LSM refinement; G-05 MIND fallback (off) |
| `src/chandralign/estimate/selection.py` (new) | I-01 / G-02 / C-03 model selection and check-point accuracy; G-06 composite (off) |
| `src/chandralign/estimate/models.py` | `ParallaxTPSModel`, its source map |
| `src/chandralign/pipeline.py` | I-02; stage `model_selection`; `ref_ground_model`; C-03 metrics; C-04 wiring; I-08 probes; `delivered_geometry`. **`register_bundle` signature unchanged** |
| `src/chandralign/evaluate/control_gates.py` | C-04 `independent_crosscheck` (amended statistic); gates skip `model_selection` |
| `src/chandralign/evaluate/probes.py` (new), `quality.py`, `selftest.py` (new) | I-08, I-10 |
| `src/chandralign/matching/routing.py`, `config.py`, `configs/instruments.yaml` | I-09 |
| `src/chandralign/matching/licence.py`, `adapter.py` | I-06 allowlist, benchmark mode; M-06 dense-model sampling |
| `src/chandralign/geometry/geoprior.py` | I-17 |
| `src/chandralign/refine/uniformity.py` | G-07 (off) |
| `src/chandralign/product/warp.py` | `best_geometry` honours the chosen geometry; TPS inverse; parallax(-TPS) source maps |
| `configs/default.yaml` | `subpixel:`, `geometry:`, `tiers.accuracy`, `gates.crosscheck*`, `matching.dense_*`, `uniformity.within_cell` |
| `scripts/register_{tmc2_tc,ohrc_nac,iirs_wac,iirs_nac,to_map}.py` | I-07 rule; recording of the cross-check, accuracy and probes; `--benchmark-models`, `--fresh`, `--matchers`; IIRS product pinned |
| `scripts/known_warp_harness.py`, `crosscheck_replay.py`, `geoprior_refined_check.py`, `trackb_batch_summary.py`, `trackb_matcher_matrix.py`, `verify_reproduction.py` | the measurement tools |
| `docs/*_protocol.md` (8 new, 2 amended) | the frozen protocols and their results |

---

## 5. G-03: the RoMa family, synthetic

`bench_rift.py` regimes (8 regimes x 3 seeds, exact truth), one model per process (a first run cached every
model on the 6 GB GPU and ran out of memory; those runs were discarded and re-run):

| model | success | false confidences | note |
|---|---|---|---|
| `roma` | 24/24 | 0 | fails the null gates in the product (below) |
| `minima-roma` | 24/24 | 0 | |
| `romav2` | 24/24 | 0 | not shippable (DINOv3 licence) |
| `ufm` | 21/24 | **3** | fails decision rule 1 |
| `matchanything-roma` | 17/23 | **5** | fails rule 1 |
| `gim-dkm` | - | - | needs a 5.2 GB allocation: does not fit a 6 GB GPU even alone |

**Why `roma` is REJECTED on real data despite its precision.** It registers OHRC with a known-shift error of
0.003-0.07 px, but a dense warp always predicts something, and RoMa's sampler returns certainty 1.0 for any
sample above 0.05. Measured on a flat grey source, 43% of its matches carry certainty 1.0; on pure noise, 93%.
The null gates (grey, noise) therefore see invented structure, and correctly reject it. **Follow-up:** use
RoMa's raw, pre-sampling certainty map as an abstain signal, or use RoMa only as a second stage inside a
lock made by a matcher that passes the null gates.

---

## 6. G-09 and the real-data matcher matrix

Same code for every matcher (snapshot `74171b3`), same windows. Success = the pairing's own rule. Probe figures
are medians over windows of the matcher-free probe p50 / p95, in SOURCE px. `reports/trackb_matcher_matrix.json`.

**TMC-2 -> SELENE TC (15 windows)**

| matcher | success | tiers | probe p50 / p95 (TMC-2 px) | s / window |
|---|---|---|---|---|
| **`eloftr`** (default) | **15/15** | 12 H / 3 M | 0.267 / 1.022 | 48 |
| `xoftr` | 14/15 | 11 H / 3 M / 1 R | 0.261 / 1.046 | 132 |
| `matchanything-roma` | 13/15 | 11 H / 2 M / 2 R | 0.285 / 1.016 | 80 |
| `minima-roma` | 11/15 | 7 H / 4 M / 4 R | 0.294 / 1.075 | 68 |
| `minima-loftr` | 7/15 | 6 H / 1 M / 8 R | **2.394 / 6.145** (6 cross-check flags: real catches) | 56 |
| `roma` | 0/15 | 15 R (null gates) | 0.273 / 0.988 (accurate, but hallucinates on nulls) | 86 |
| `ufm` | 0/15 | 15 R | 0.763 / 5.795 | 41 |

**OHRC -> LRO NAC (25 windows, I-07 rule)**

| matcher | success | unconfirmed (MI) | probe p50 / p95 (OHRC px) |
|---|---|---|---|
| **`xoftr`** | **15** (M102 5, M106 5, **M175 5**) | 0 | 1.06 / 1.62 |
| routed today (`eloftr` -> `minima-loftr`) | 12 | 8 | - |
| `eloftr` alone | 10 | 0 | 0.81 / 2.38 |
| `minima-roma` | 5 (M109 5/5) | 5 | 0.84 / 1.67 |
| `matchanything-roma` | 12 | 3 | 0.85 / 1.84 (but 5 synthetic false confidences: excluded by rule 1) |
| `roma` | 0 (null gates; see section 5) | 0 | 0.88 / 1.88 |
| `ufm` | 0 (22 rejected) | 0 | 1.25 / 5.60 |
| `minima-loftr` | not scored alone: it ran only as the routed fallback; its successes are inside the routed 12 | | |

**IIRS -> LRO WAC (5 windows):** `eloftr`, `minima-loftr`, `xoftr`, `matchanything-roma`: 5/5 HIGH, probe p95
0.195-0.245 IIRS px. `roma`, `minima-roma`, `ufm`: 0/5 REJECTED (their probe p95 is similar, 0.21-0.25, so they
are rejected by the gates, not because they are inaccurate).

**TMC-2 -> SELENE MI (5 windows):** 0/5 for every matcher: genuinely unsolved.

**G-09 recommendation.**
1. **TMC-2 -> TC: keep `eloftr` as the default.** With the cross-check live its false confidence is gone
   (synthetic, 0/24), it is the most reliable on real windows, and it is 2.7x faster than `xoftr` per window.
2. **OHRC -> NAC: route `eloftr` -> `xoftr` -> `minima-loftr`.** On the committed windows this gives about 17
   success + 3 unconfirmed instead of 12 + 8, and solves the opposed-sun NAC for the first time. Because it was
   chosen on those same windows, it was **confirmed on the rule-based FRESH windows** (I-16's `--fresh` windows,
   batch 7b, one window per process, scored with the script's own consistency and MI rules;
   `reports/trackb_g09_fresh_confirmation.json`):

   | fresh NAC (4 windows each) | `xoftr` success / unconfirmed | routed today |
   |---|---|---|
   | M102014464RC | 4 / 0 | 4 / 0 |
   | **M175124932LC (opposed sun)** | **4 / 0** (known shift 0.006-0.048 px, MI clean) | 0 / 4 |
   | M109080308LC (overhead sun) | 0 / 0 (gates fail) | 1 / 3 |
   | M1417360906LC (75 deg) | 0 / 0 | 0 / 0 |
   | **total** | **8 / 0** | 5 / 7 |

   **Confirmed:** `xoftr` solves the opposed-sun NAC on ground no choice was tuned on (5/5 committed, 4/4 fresh).
   It is weak on M109, where `minima-loftr` is better, so the order `eloftr` -> `xoftr` -> `minima-loftr` keeps
   both strengths. On the fresh windows that gives **9 success + 3 unconfirmed** against today's 5 + 7.
3. **Ready to adopt, and left to the merge:** it is a routing change on the product path
   (`configs/regimes.yaml`; OHRC -> NAC needs `xoftr` placed before `minima-loftr` in its fallbacks, while
   TMC-2 -> TC keeps `eloftr` -> `minima-loftr` because `minima-loftr` is poor there: 7/15). It is not applied
   on this branch, so that Track A decides it together with its moved routing code. Note: the first batch-7 run
   was invalid (a foreign process filled the GPU, and one CUDA out-of-memory error poisoned 13/16 windows),
   and a re-run lost its files to a line-ending bug. Only the clean third run (16/16 result files) is used.

---

## 7. Honest accuracy per pairing (source px, p50 / p95, matcher-free probes unless stated)

| pairing | result | accuracy | sub-pixel in source px? |
|---|---|---|---|
| **TMC-2 -> SELENE TC** | 15/15 (12 HIGH, 3 MEDIUM); fresh 5/5 HIGH | per window p50 **0.135-0.267**, p95 **0.37-1.33** TMC-2 px (G-06 step 2); 14/15 windows p95 < 1 | **yes at p50; p95 on 14/15** |
| **OHRC -> LRO NAC** | 12 success + 8 unconfirmed of 25 (xoftr: 15 success) | eloftr p50 0.81 / p95 2.38; xoftr 1.06 / 1.62 OHRC px | **no** (p50 about 1 px) |
| **IIRS -> LRO WAC** | 5/5 HIGH | p50 ~0.11 / p95 ~0.23 IIRS px | **yes** (narrow: one scene, 5 windows) |
| IIRS -> NAC, TMC-2 -> NAC | reproduce the committed results exactly | unchanged from the audit's section 7 | not demonstrated |
| TMC-2 -> MI | 0/5 for all 7 matchers | - | unsolved |

---

## 8. What went wrong along the way (recorded, because it matters for trusting the rest)

- **C-02 held-out failure.** The dev-chosen 0.75 px cap failed one held-out case. It is recorded, the cap was
  amended with a stated reason, and it was re-scored on a fresh set. The held-out result stays in the record.
- **C-04 false alarms on real relief.** The first cross-check version rejected 7 correct hilly windows. This
  was caught by the frozen real-window measurement, amended, and re-measured on both synthetic and real data.
- **GPU out-of-memory contamination.** The first RoMa benchmark cached every model on the 6 GB GPU; 3 synthetic
  models and 5 OHRC `minima-loftr` windows errored. They were discarded, re-run one model per process, and
  the contaminated cells are never used.
- **A merge I did not intend.** A `git merge` of Track A into this branch that the user had declined still ran
  (05:52). It left a conflicted merge in this worktree until it was aborted with the user's agreement
  (branch restored exactly to `ae0af31`). A 40 GB accidental data copy was also deleted (the original is
  intact), and a mistyped `git checkout` detached the worktree for about a minute. None of these affected a
  result or a commit.
- **New data under our feet.** Four IIRS products appeared in the shared data folder at 09:55 on 2026-09-26.
  Code that took "the first IIRS product" would have switched products silently. Scripts, tests and the running
  batch are pinned to the product every committed IIRS result uses (`ch2_iir_nci_20240523T1600301891`).

---

## 9. For the merge (Track A merges; see also `docs/TRACKB_INTERFACE_NOTES.md`)

- A trial merge of `track-a/product-wiring` (tip at audit time) gives **7 conflicting files**: `pipeline.py`,
  `matching/licence.py`, `tests/test_pipeline.py`, `tests/test_licence_gate.py`, and the three research scripts.
- **The scripts:** Track A moved their bodies into `src/chandralign/workflows/{tmc2_tc,ohrc_nac,iirs_wac}.py`.
  Track B's script changes must be re-applied there: the I-07 success/unconfirmed rule and the MI fallback
  (OHRC), the per-window `crosscheck` / `accuracy_fine_frame` / `probe_check` recording, `routing.run_candidates`,
  and the flags `--benchmark-models` and `--fresh` in the wrappers.
- **`licence.py`:** both tracks built an allowlist. Keep one, and keep Track B's named reasons (master, duster,
  gim-lightglue, omniglue, romav2) and `enable_benchmark_mode()` (used by `--benchmark-models`).
- **`pipeline.py`:** Track B changed `fine_stage` internals and added fields and helpers; Track A added
  `register_bundle` parameters. Both sets are needed.
- After the merge: `pytest -q`, `sabotage.py --target all` (expect 90 + Track A's own), and
  `verify_reproduction.py` on a fresh run.

---

## 10. Recommended next steps (ranked)

1. Adopt the OHRC routing `eloftr -> xoftr -> minima-loftr` at the merge (confirmed on fresh windows; section 6).
2. RoMa as a second stage inside a null-gate-passing lock, or its raw certainty as an abstain signal. It is the
   most precise matcher measured on OHRC.
3. G-04: independent check points (LROC NAC DTM / ortho), the only way to turn every figure here from
   "consistent" into "true".
4. The periodic-aliasing trap for C-04; a second checker family (MIND dense step) so that +30 deg lighting and
   2x scale are not capped at MEDIUM.
5. G-06's dense residual field for the last TMC-2 window (4687 c3232, p95 1.33).
