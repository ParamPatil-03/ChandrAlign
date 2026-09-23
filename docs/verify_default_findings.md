# Independent verification: default matcher, and making xoftr fast enough

Branch `part2/verify-default`, off `part2/default-matcher`. Criteria frozen in
`docs/verify_default_protocol.md` (commit `1d9a03f`) before any measurement
below. **Nothing here changes a production setting.** `configs/regimes.yaml`
still ships `default_matcher: aliked-lightglue`.

---

## Verdict on "switch the default to `eloftr`": CONFIRM WITH CHANGES

`eloftr` is the right same-modality default **of the matchers measured on real
data**, but the stated reasons are not the ones that carry it:

- "15 vs 1 false confidences" is **true**, but it is synthetic. Every one of
  aliked's same-modality false confidences sits at 90 or 180 degrees of sun
  AZIMUTH difference. The real LRO NAC ladder spans only 11.2 degrees of azimuth,
  so real data has neither confirmed nor refuted them.
- Re-scored on the regimes the default actually serves, the frozen protocol
  does **not** pick `eloftr`. It picks `sift-lightglue`, then `disk-lightglue`,
  then `xoftr`, then `eloftr`. Both leaders **fail the real-data check**, and
  that check, not the synthetic ranking, is what makes `eloftr` the answer.
- "Better than aliked" is only weakly established. It means no control-gate
  rejection, a tighter perturbation tail and ~10x the inliers. It does not mean
  higher accuracy: see section 4.

**Part 2 changes the picture for the TMC-2 -> SELENE path specifically.** xoftr
ranks ABOVE eloftr on the re-score (0 vs 1 false confidences, 0.104 vs
0.123 px), passes the real check, and runtime was the only reason to prefer
eloftr. Tiled xoftr with gate base reuse runs the 9 windows in **80.3 s** against
eloftr's 107.5 s, although eloftr was measured without either optimisation. Tiling
is valid only after the coarse stage has aligned the pair, so this is a
**fine-stage matcher for pre-aligned pairs, not a general default**. It also fails
one frozen criterion by a threshold-edge margin (section 6). If the team accepts
that, the evidence favours `xoftr` (tiled, fp32) for the coarse-aligned fine stage
and `eloftr` as the general same-modality default.

Changes the recommendation needs before it ships:

1. State the real-data check as the reason, and attach its limits (9 windows,
   3 TC tiles, no ground truth).
2. `eloftr` is Apache-2.0 **only at the pinned version** (upstream relicensed
   2026-09-15). The pin `vismatch>=1.3,<=1.3.2` is now load-bearing for the
   *default*, not for an optional matcher. `aliked-lightglue` has no such exposure.
3. `eloftr` is not memory-flat: 4.9 GB peak at 1024 px and OOM at 1536 px on this
   6 GB card (`reports/runtime_budget.json`). The ~1040 px real windows fit; a
   larger window, or a 4 GB card, does not. `aliked-lightglue` is flat at 1.8 GB.
   Tiling (section 6) removes this limit for eloftr too, but that is unmeasured.

---

## 1. False confidences, recounted from the 720 raw rows

Recount (`scripts/verify_default_rescore.py`, `reports/verify_default_rescore.json`):
**aliked-lightglue 15, eloftr 1, xoftr 0. Confirmed.**

The definition is: estimator accepted, quality tier not REJECTED, error > 2 px.
(My first recount forgot the tier condition and got eloftr 28 / xoftr 9. The row
field `accepted` is the ESTIMATOR's verdict, not the quality gate's. Recorded here
because it is an easy trap for the next reader too.)

**A defect in the selector, not in the counts:** its `success` field ignores
the quality tier while `false_confidence` uses it. 7 of 720 runs count as
successes although the quality gate rejected them. None involve xoftr, eloftr or
aliked-lightglue, so no eligibility changes, but the two fields are not measured
on the same terms.

## 2. What `default_matcher` actually serves: claim (a) is TRUE, and understated

From `matching/regime.py::select`:

- a cross-modal pair gets `matcher=None` plus the `cross_modal_candidates` list;
  `default_matcher` is returned only on the same-modality branch;
- scale is checked **before** modality. OHRC->IIRS (320x), IIRS->TC (~11x) and
  IIRS->NAC (~160x) all go to the cascade, whose steps default to `sift` / dense
  MIND and never read `default_matcher`. The protocol's stated reason for making
  `cross_modal` CORE ("OHRC-to-IIRS") names a pair the default can never see.
  The PS dataset names no Chandrayaan-to-Chandrayaan pairing in any case
  (research doc 1.2);
- `select()` has **no caller** outside `tests/test_regime.py`. Every real run in
  `scripts/` passes an explicit `model_name`. Today `default_matcher` reaches
  production only through `adapter.match(model_name=None)`, which nothing calls.
  The routing is designed but not wired in.

### Re-scored on same-modality regimes only (every other rule unchanged)

| candidate | core ok | false conf. (of 60) | core px | eligible | rank |
|---|---|---|---|---|---|
| sift-lightglue | 100% | 0 | 0.045 | yes | **1** |
| disk-lightglue | 98% | 0 | 0.075 | yes | 2 |
| xoftr | 100% | 0 | 0.104 | yes | 3 |
| eloftr | 100% | 1 (1.7%) | 0.123 | yes | 4 |
| aliked-lightglue | 100% | **11** (18.3%) | 0.407 | no | |
| minima-loftr | 75% | 1 | 0.162 | no | |
| xfeat | 78% | 0 | 0.292 | no | |
| matchanything-eloftr | 75% | 15 | 0.131 | no | |
| sift-nn | 55% | 0 | 0.101 | no | |

Post hoc: I saw these rows before writing the protocol. This table can show that
the recommendation does not follow from the protocol. It cannot promote anything.

## 3. Real TMC-2 -> SELENE TC check, all six matchers (protocol 3.A)

Pass = 9/9 registered, no REJECTED tier, and composed transform within 2.0 TC px
(grid RMS) of the committed xoftr transform on every window.

| matcher | rejected windows | max vs xoftr | 9-window total | passes |
|---|---|---|---|---|
| xoftr | none | (reference) | 948.3 s | yes |
| **eloftr** | none | 0.70 px | 107.5 s | **yes** (post hoc) |
| aliked-lightglue | 4687 (perturbation 1.76 px) | 2.02 px | 100.7 s | no (post hoc) |
| sift-lightglue | 4687 (perturbation 2.24 px) | 2.96 px | 48.3 s | **no, new** |
| disk-lightglue | 3125 (perturbation 4.72 px) | **8.63 px** | 45.5 s | **no, new** |
| minima-loftr | 5 of 9 | 8.59 px | 280.8 s | no (post hoc) |

**The synthetic winner fails on real data**, on the same window as the incumbent.
That makes five synthetic or label-derived claims in this project that did not
survive real data.

**A real-data false confidence, found here for the first time.** On window
4687, `disk-lightglue` is tiered **HIGH with every control gate passing**, yet
its transform is 6.2 to 9.7 px from ALL five other matchers, from both the
sparse and the dense families. Its implied along-track pixel is 5.163 m where
the other five give 5.077 to 5.097 m. The perturbation gate cannot catch this:
it tests sensitivity to a shift, not correctness, and a consistently wrong
answer moves correctly. This is failure mode #19 on real data, not a citation.

## 4. Is `eloftr` better than `aliked-lightglue`, or only different?

Mostly different. Pairwise grid RMS on the three tile-N00 windows shows **two
clusters**:

| window | within sparse (aliked, sift-lg, disk-lg) | within dense (eloftr, xoftr) | between |
|---|---|---|---|
| 1562 | 0.5-1.1 px | 0.48 px | 1.2-2.4 px |
| 3125 | 0.8-2.7 px | 0.42 px | 1.1-1.5 px |

Every matcher's own affine inlier RMSE on these windows is 1.0-1.7 px, so an
affine model does not fit them better than that. A 1-2 px gap between two affine
fits built from differently distributed matches is inside the model's own error.
Agreement with xoftr therefore **cannot** show that eloftr is more accurate than
aliked, because eloftr and xoftr are both LoFTR descendants and share failure
modes. Commit history calls them "two matchers sharing no architecture"; that is
not right.

What does separate them on real data:

- the perturbation gate (a KNOWN 5 px shift, the one real accuracy measure
  available): aliked median 0.015 px but 0.63 and **1.76 px** on N00; eloftr
  max 0.46 px. aliked tracks a known move more tightly on typical windows and
  worse on hard ones;
- one rejection in nine, on the window where EVERY matcher is at its worst
  (eloftr 0.46 and xoftr 0.93 px there are their own maxima too);
- ~10x the inliers (7,000-15,000 against 800-1,400), which the PS's
  inlier-count and uniform-distribution requirements reward directly.

That is a defensible preference with a small sample (9 windows, 3 tiles, one
hard window), not a demonstrated accuracy win.

## 5. How much weight the synthetic evidence deserves

- **It still refutes.** The synthetic data correctly shows that no protocol
  reading makes `eloftr` the default without real data.
- **It does not promote.** Its top pick failed on real data, the fifth time a
  synthetic or label-derived claim here has been contradicted. It is not
  uniformly wrong: it predicted xoftr and eloftr handle the 2x resolution gap
  and aliked has the lowest runtime, and real data agrees. It is unreliable at
  ranking candidates that are close.
- **The aliked false confidences are neither weakened nor confirmed by finding
  (b).** The NAC evidence that "synthetic overstates difficulty" concerns
  INCIDENCE at near-constant azimuth (11.2 degree span). aliked's 11
  same-modality false confidences are all at 90 and 180 degrees of AZIMUTH, an
  axis real data has not reached (it is structurally rare for equatorial
  targets; `a98b4cc`). Treat them as an untested hypothesis, not as evidence
  against aliked, and not as refuted either.

---

## 6. xoftr under 180 s: achievable, measured, 9-12x faster

All runs: the same 9 windows, each variant in its own fresh process, one GPU
job at a time, cold model load included (as the 948.3 s baseline was).
`reports/xoftr_speed_summary.json` and `reports/xoftr_speed_*.json`.

| variant | 9-window total | max drift from committed xoftr | frozen criteria |
|---|---|---|---|
| V0 fp32, whole (control) | 887.2 s | **0.000 px** | control |
| V1 fp16, whole | 140.3 s | 1.52 px | FAIL: 3125, 4687 drift past the eloftr envelope |
| **V2 fp32, tiled** | **95.2 s** | 0.53 px | FAIL, one item: 4687 tier HIGH -> MEDIUM |
| V3 fp16, tiled | 69.7 s | 1.77 px | FAIL: 4687 drift past the envelope |
| **G1 = V2 + gate base reuse** | **80.3 s** (repeat: 70.9 s) | 0.53 px | FAIL, same one item as V2 |

**The control reproduced the committed run exactly** (0.000 px on every
window, 887 s against 948 s). xoftr in fp32 is deterministic here and the
environment has not drifted, so every difference below is caused by the variant.

### What works: tiling in fp32, plus reusing the main registration in the gate

- **Tiling** (2x2 tiles of ~520 px, 64 px reference margin): every window stays
  inside the frozen accuracy envelope (max 0.53 px from the committed transform,
  median 0.08 px), every gate passes, and the perturbation error on the hardest
  window IMPROVES (0.934 -> 0.495 px).
- **G1** hands the main registration to the perturbation gate as its baseline
  instead of recomputing it. The main call and the gate's baseline call are the
  same matcher and estimator on the same arrays, and G1's transforms and gate
  results are bit-identical to V2's on all 9 windows. It is **lossless**; it saves
  one of six matcher calls per window (gates 46.4 -> 34.8 s).

**It does not strictly pass my frozen criteria.** On window 4687 the inlier
ratio moves from 0.5032 to 0.4985 and crosses the 0.50 HIGH bar
(`configs/default.yaml`), so the tier drops to MEDIUM. Count, coverage and RMSE
are unchanged and the transform moves 0.53 px, inside that window's 0.70 px
envelope. I froze "no tier worse than V0" before measuring and am not relaxing it
after the fact: **whether a 0.005 ratio shift on one window blocks a 12x speedup
is a team decision.** My read: V0's HIGH on that window was itself 0.003 above a
bar, so it was never robust.

### What does not work: fp16

fp16 halves memory and is the fastest (69.7 s tiled), but it **changes the
answer** on the two hardest windows: 0.85 and 1.52 px (whole) and 1.77 px (tiled)
on 4687. That is more than eloftr differs from xoftr there, so fp16 buys speed by
spending the accuracy that is the only reason to use xoftr. Easy windows move
under 0.1 px, so an fp16 check on easy data would have missed this. Its synthetic
check was not run: it cannot rescue a variant that already fails on real data.

### Why the extrapolated 14.5x roughly held, and why the model behind it is wrong

Match-only median, windows 2-9: whole 21.6 s, tiled 1.25 s, **measured 17.3x**
against the predicted 14.5x. But whole-window calls on near-identical ~1040 px
windows range from **3.9 to 38.5 s**. A power law in pixel count cannot produce
a 10x spread at fixed size. The mechanism is memory: xoftr at ~1040 px needs about
8.3 GB on a 6 GB card and spills into shared system memory, with the observed
5.9 GB of 6.1 GB in use during V0. Tiling and fp16 both remove the spill, which is
why both are fast. The k = 2.82 exponent in `reports/runtime_budget.json` was
fitted across that memory cliff (768 px fits at 4.7 GB, 1024 px does not). It is
a property of this card, not of xoftr, and will not transfer to a larger GPU.

### Tile edges (descriptive, `reports/xoftr_speed_G1_repeat_edges.json`)

- ~19% of matches lie within 64 px of an internal tile boundary, slightly under
  the ~23% area share. Edges are mildly under-sampled because context is cut.
- Inlier RMSE near edges vs interior differs by at most 0.25 px: higher on 4
  windows, lower on 5. There is no systematic edge penalty.
- Inlier ratio near edges is 1-6 points lower on 7 of 9 windows.
- Merging: source tiles partition the window, so no match is duplicated. The
  estimator runs once on the union, so RMSE, inliers and tier are one global fit.

### Gate scheduling

Per window the gates cost 5 matcher calls against 1 for the registration: blank,
noise, identity (**reference-only**), and the perturbation pair (**pair-specific**).

| schedule | calls / window | status | verification lost |
|---|---|---|---|
| as committed | 6 | measured | none |
| G1: reuse main as perturbation base | 5 | **measured, 80.3 s** | **none** (bit-identical) |
| G2: blank / noise / identity once per TC tile | 2 + 3 per tile | **derived only**, ~63 s | CHECK-01/02/04 no longer see the crop each window used; a window whose own crop makes the matcher invent structure (seam, repetitive texture) escapes them |

G2 is not recommended: G1 is already under budget without weakening any gate.

### Budget caveats

- 180 s is the brief's figure. Research doc section 38 gives the TMC-2 segment of
  the demo **40 s**. G1 is 53 s warm (cold load excluded), so a live 9-window
  run still does not fit that slot. Show precomputed windows, or fewer windows.
- Timing noise between identical runs is about +-10 s, mostly the cold load.
- `eloftr` (the proposed default) has the same memory shape at larger windows
  (OOM at 1536 px). Tiling applies to it unchanged but was not measured here.

---

## 7. Errors found in the previous agent's work

1. **The recommendation does not follow from its own protocol.** Re-scored on
   the regimes the default serves, the protocol ranks sift-lightglue, then
   disk-lightglue, then xoftr, then eloftr. "eloftr" was reached without testing
   the two candidates ranked above it on real data. Both fail that test.
2. **"All 15 false confidences are outside aliked's lane"** (`routing_proposal.md`
   point 4) contradicts the headline table that charges all 15 against it. 11 are
   in same-modality regimes that `select()` DOES send to the default, marked
   `unsolved`.
3. **"Two matchers sharing no architecture"** (eloftr, xoftr). Both are LoFTR
   descendants; their agreement is not independent confirmation.
4. **Failure mode #1 again: "synthetic overstates difficulty" was generalised
   from one axis.** The NAC evidence covers incidence at 11.2 degrees of azimuth
   span; the aliked false confidences are on the azimuth axis.
5. **The runtime model was taken at face value.** k = 2.82 was fitted across a
   VRAM-spill cliff (failure mode #5's shape: a contaminated timing read as
   scaling).
6. **Not an error of that agent, found on the way:** the selector's `success`
   ignores the quality tier (7 runs), and `configs/default.yaml` makes the LOW
   tier's inlier-ratio bar (0.325) stricter than MEDIUM's (0.25). Neither is
   changed here.

## 8. What I did not do

- No production setting changed. `adapter.match` gains `precision` and `tile_px`,
  and `control_gates` gains an optional `base`; every default reproduces the old
  behaviour (V0 = committed, 0.000 px).
- Tiling is NOT wired into any default path. It is valid only after the coarse
  stage, and the adapter refuses unequal-size pairs.
- Tests: 506 passed, 49 skipped (6 new). Sabotage: 50/50 caught (6 new, covering
  tile offsets, margin, grid, the pre-alignment refusal and gate base reuse).
