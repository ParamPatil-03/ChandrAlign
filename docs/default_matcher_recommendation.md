# Default matcher: recommendation and evidence

Produced by applying `docs/default_matcher_protocol.md` (frozen in d5b4d43,
before any of this was written or run). **This is a recommendation, not a
change.** Protocol section 8 leaves the decision to the team;
`configs/regimes.yaml` still ships `aliked-lightglue`.

---

## Recommendation

**Replace `aliked-lightglue` with `xoftr` as `default_matcher`**, and keep the
routing table's existing honesty about where no matcher works.

Also, independently of that decision: **`eloftr` must stop being used**, in the
shippable list and in `scripts/register_tmc2_tc.py`'s default, on licence
grounds alone (`reports/licence_audit.json`).

---

## How the incumbent came to be the default

PLAN.md P2-T03's bar was *"beats SIFT on the synthetic illumination-shifted
pair"*. SIFT scores zero successful seeds at every sun-azimuth difference of 30
degrees or more, so that comparison was decided before it began. The incumbent
was never measured against an alternative that could also pass it.

That is why this was re-run under a protocol fixed in advance, on fresh seeds,
rather than by reading an existing table.

---

## 1. Licence gate (protocol 4.1) — two candidates eliminated before measuring

| model | licence | |
|---|---|---|
| `eloftr` | Project Registration License v1.0 | **was in `shippable_matchers`** |
| `matchanything-eloftr` | Project Registration License v1.0 | **was a cross-modal candidate** |

Not OSI-approved. Free of charge only once the project is **registered with the
authors beforehand**, commercial or not — a precondition that travels to anyone
we ship to. Verified by reading each repository's own LICENSE file.

The trap: zju3dv license their models individually. LoFTR is Apache-2.0;
EfficientLoFTR, same group, is not. A licence cannot be inferred from a sibling
model, an author, or an earlier version of the same repository.

## 2. Synthetic measurement (protocol 3–5) — 560 runs, seeds 101–110

| candidate | core ok | core px | false conf. | stretch | sec | |
|---|---|---|---|---|---|---|
| **xoftr** | **100%** | 0.098 | **0** | 2/3 | 0.53 | **ELIGIBLE** |
| aliked-lightglue | 80% | 0.407 | **15** | 1/3 | 0.36 | incumbent |
| minima-loftr | 80% | 0.181 | 1 | **3/3** | 0.47 | |
| sift-lightglue | 80% | 0.045 | 0 | 1/3 | 0.74 | |
| disk-lightglue | 78% | 0.075 | 0 | 1/3 | 0.67 | |
| xfeat | 62% | 0.292 | 0 | 1/3 | 0.07 | |
| sift-nn | 44% | 0.101 | 0 | 1/3 | 0.10 | |

**Candidates fail by whole regimes, not marginally.** Every eliminated candidate
scored 0/10 on some core regime, never 7/10:

- `cross_modal` — **0/10 for five of seven candidates**, the incumbent among
  them. They cannot do the problem statement's own hardest named pairing at all.
- `resolution_2x` — **0/10 for `minima-loftr`**, which is otherwise the best of
  the field at extreme illumination. It cannot do TMC-2 to SELENE TC, a route we
  ship.

So the operative question is not "is 90% the right threshold" but "must the
default handle all five core regimes".

**The incumbent's problem is not accuracy, it is confident error.**
`aliked-lightglue` produced **15 false confidences in 80 runs — 18.8% against a
2% limit** — in `cross_modal` (4), `lighting_+90` (5) and `lighting_opposite`
(6), errors from 2.0 px to **16.1 px**, every one accepted by the quality gate.
Roughly one run in five returns a confidently wrong transform.

## 3. Real-data check (protocol 6) — TMC-2 to SELENE TC, the same 9 windows

Behaviour and consistency, **not** a second accuracy table: these windows have
no exact ground truth, and TMC-2 is scored on SYSTEM corners only.

| | eloftr (frozen) | xoftr | |
|---|---|---|---|
| windows registered | 9/9 | 9/9 | |
| confidence tiers | 8 HIGH, 1 MEDIUM | 8 HIGH, 1 MEDIUM | identical |
| median inlier RMSE | 1.061 px | 1.126 px | +0.065 px |
| median vs ISRO refined grid | 312.4 m | 315.0 m | agree to 2.6 m |
| TMC-2 pixel implied by TC | 4.988 x 5.057 m | 4.987 x 5.057 m | **agree to 1 mm** |
| control gates | all pass | all pass | |
| **median match time** | **1.6 s** | **24.7 s** | **15x slower** |

The pixel-size agreement is the strongest line here: two matchers sharing no
architecture recover the same physical pixel to a millimetre, which is
independent support for the ~4.99 x 5.06 m measurement itself, not just for
either matcher.

---

## The one thing that argues against this recommendation

**Runtime at real window sizes.** The protocol's runtime gate was "under 5 s per
512 px pair", and `xoftr` passed it at 0.53 s. On real 1536 px windows it takes
**24.7 s against eloftr's 1.6 s**. That is not a protocol violation — it is
outside what the gate measured, which is a limitation of the gate.

It needs a decision rather than being waved through:

- it is 15x, not a few percent;
- the research document's section 38 three-minute demo budget was the source of
  the 5 s figure, and that budget was never checked at real scale;
- a full TMC-2 strip is many more windows than nine.

Tiling and batching are the obvious mitigations and neither has been tried.
**This should be measured at real scale before the switch ships**, and if the
demo budget cannot absorb it, the honest options are a faster eligible
candidate or a routed default — not quietly accepting the cost.

## Other honest limits

- `xoftr` fails `lighting_+90` on every seed. It is the best candidate, not a
  universal one. `configs/regimes.yaml`'s `unsolved_illumination` band must keep
  saying that no matcher solves that regime.
- **`minima-loftr` beats `xoftr` at extreme illumination** (7/10 at +90 where
  `xoftr` scores 0/10, and 3/3 stretch regimes against 2/3). Ineligible as a
  *default* is not useless; it is a real candidate for an opposed-sun route.
- All of section 2 is synthetic, from one terrain generator.
- The 90% core threshold is load-bearing at the margin: at 80% or lower,
  `sift-lightglue` would have qualified and won on false confidence then
  accuracy (0.045 px). The threshold was fixed in advance and is not being
  revisited, but the result is sensitive to it rather than robust to it.
- Section 2's "success" means the estimator accepted and the transform is within
  2 px (protocol 4.2). Some runs counted as successes were tiered REJECTED by
  the quality gate, which is a stricter and separate judgement.

## Separately confirmed: the `unsolved` band needs re-deriving

`configs/regimes.yaml` declares every pair above 60 degrees of sun azimuth
difference `unsolved`, citing a benchmark of sift, xfeat, aliked-lightglue and
eloftr — none of which can do it. Both `xoftr` and `minima-loftr` solve
`lighting_opposite` (180 degrees) on 10 seeds out of 10. The band was fitted on
a matcher set that excluded the methods that work, which is the same pattern as
P2-T03's bar. It was left untouched here so as not to contaminate this
experiment, and is tracked as separate work.

## If the switch is approved

Rerun only results the old default directly produced. Verified by inspection:

| result | depends on the default? | |
|---|---|---|
| `reports/tmc2_tc_registration.json` | no — explicit `--matcher eloftr` | **rerun anyway, on licence** — done, `reports/tmc2_tc_xoftr.json` |
| `reports/tmc2_tc_gap_rows.json` | no — explicit `eloftr` | **rerun, on licence** |
| `reports/cascade_ohrc_tc.json` | **no** — the cascade matches with MIND | unaffected |
| `reports/rift_benchmark.json` | no — explicit model list | unaffected (benchmark of matchers) |
| `reports/estimator_benchmark.json` | no — correspondence sets, no matcher | unaffected |
| `reports/subpixel_verification.json` | no — explicit methods | unaffected |
