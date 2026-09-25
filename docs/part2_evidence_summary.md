# Part 2 evidence summary — what is established, and what is not

Branch `part2/default-matcher`, 19 commits. **Nothing here is applied.**
`configs/regimes.yaml` still ships `default_matcher: aliked-lightglue` and every
production threshold is unchanged. This document is the evidence and the
decisions it asks for.

500 tests passed, 49 skipped, 44/44 sabotages caught when this was written. The sabotage count has grown since: run `scripts/sabotage.py --target all` for the current figure (70/70 on 2026-09-26).

---

## 1. How this work started

`configs/regimes.yaml` shipped `aliked-lightglue` as the default matcher,
accepted against PLAN.md P2-T03's bar: *"beats SIFT on the synthetic
illumination-shifted pair"*. SIFT scores **zero successful seeds** at every sun
difference of 30 degrees or more, so that bar is passed automatically and
separates nothing. The default had never been compared against an alternative
that could also pass it.

That turned out to be one instance of a pattern. PLAN.md's accept criteria are
pass/fail bars written before anything was built, and several cannot fail:

| criterion | why it cannot discriminate |
|---|---|
| P2-T03 "beats SIFT" | SIFT scores 0 in those regimes |
| ALIGN-01 "MAGSAC beats RANSAC at 70% outliers" | all estimators succeed there |
| TMC-2 GSD "agrees with the label" | the label is a pre-launch prediction, off by 12% |
| P2-T13 "MI higher when aligned" | MI is *defined* to peak at alignment |

**PLAN.md is no longer used as the source of acceptance criteria.** Every
experiment below was run against a criterion frozen and committed *before* the
measurement existed, with thresholds taken from sources outside the experiment's
own data. `git log` on each protocol file shows the ordering.

## 2. What is established

### The default matcher fails its own eligibility rules

720 runs, 9 candidates, 8 regimes, 10 fresh seeds
(`docs/default_matcher_protocol.md`, `reports/default_matcher_selection.json`).

| candidate | core ok | core px | false conf. | |
|---|---|---|---|---|
| **xoftr** | **100%** | 0.098 | **0** | **only eligible candidate** |
| aliked-lightglue | 80% | 0.407 | **15** | incumbent |
| minima-loftr | 80% | 0.181 | 1 | |
| sift-lightglue | 80% | 0.045 | 0 | |
| disk-lightglue | 78% | 0.075 | 0 | |
| eloftr | 80% | 0.123 | 1 | |
| xfeat | 62% | 0.292 | 0 | |
| matchanything-eloftr | 60% | 0.131 | **23** | |
| sift-nn | 44% | 0.101 | 0 | floor, expected to fail |

Candidates fail by **whole regimes**, not marginally: seven of nine score
**0/10 on cross-modal**, the problem statement's own hardest named pairing.

**The incumbent's problem is confident error, not accuracy.** 15 false
confidences in 80 runs — 18.8% against a 2% limit — with errors to 16.1 px,
every one accepted by the quality gate. Roughly one run in five returns a
confidently wrong transform. Two further measurements agree independently: 22
of 30 confident wrong answers in the illumination sweep, and a control-gate
failure on real TMC-2 data.

### Illumination difference does not break matching on real terrain

100 real tests, LRO NAC, incidence gaps 25.8-54.5 degrees at near-constant
azimuth (`docs/nac_illumination_protocol.md`).

| | xoftr | minima-loftr | eloftr | aliked-lg | sift-nn |
|---|---|---|---|---|---|
| overall | 19/20 | 19/20 | 19/20 | 19/20 | 13/20 |
| median perturbation error | 0.02 px | 0.06 px | 0.04 px | 0.01 px | 0.03 px |

**This contradicts the synthetic model**, which predicts `eloftr` and
`aliked-lightglue` collapse to 0/10 past 60 degrees. On real terrain all four
learned matchers are indistinguishable and effectively perfect; only SIFT
degrades, as the literature says it should. The single learned-matcher failure
is the *same window* for all four.

### The TMC-2 pixel, four ways

| matcher | implied TMC-2 pixel |
|---|---|
| eloftr | 4.988 x 5.057 m |
| xoftr | 4.987 x 5.057 m |
| aliked-lightglue | 4.986 x 5.058 m |
| minima-loftr | 4.989 x 5.058 m |

Four matchers sharing no architecture agree to about **2 mm**, against a label
claiming 4.41 m. This is no longer evidence about matchers; it is independent
confirmation of the measurement.

### MAGSAC's advantage is cost, not accuracy

900 runs (`reports/estimator_benchmark.json`). Accuracy is a tie; at 90%
outliers both reach 10/10, MAGSAC in 0.021 s against plain RANSAC's 3.27 s.
The benchmark also found `estimate.max_iters` was set too low (10000 sat on the
edge of solvable; raised to 100000, free because MAGSAC terminates adaptively).

**A limit worth quoting:** when false matches agree with *each other*, past 50%
every driver returns the outliers' transform and reports a clean fit — 110-114
confident wrong answers in 300 runs each. A consensus method cannot prefer the
minority. No robust driver fixes this, and the scale check does not either.

### Licences attach to an artefact at a version, not to a project

`zju3dv` relicensed EfficientLoFTR and MatchAnything from Apache-2.0 to the
Project Registration License on 2026-09-15 — not OSI-approved, registration
required before organisational use. **What we install predates it**
(vismatch 1.3.2, released 2026-08-17; weights last modified 2026-02-10), and
Apache-2.0 section 2 is irrevocable. The control is a **version pin**, not a ban:
`vismatch>=1.3,<=1.3.2`, enforced by a test that fails on upgrade.

## 3. What is NOT established

### The solar-azimuth axis is unvalidated

`configs/regimes.yaml` indexes its illumination bands on **azimuth
difference**. Every number in them comes from one synthetic terrain generator.
The attempt to validate them on real data **failed and produced no result**
(`a98b4cc`):

- ~90 degrees, the angle the synthetic model calls worst, is **structurally
  unavailable**: an equatorial target imaged from a polar orbit is lit from the
  east or the west, never from the side. Four photometric sites scanned; three
  sit within a degree of the equator.
- The one 126.7 degree pair that exists **does not overlap on the ground** — a
  selection error of mine, having assumed photometric-site observations overlap
  by construction.
- The coarse-alignment stage fails on the Apollo 11 products (z = 4.7-5.8
  against a 6.0 bar) **for same-sun pairs exactly as badly as for opposed-sun
  pairs**, so it carries no information about azimuth.

The pre-registered prediction stands **unscored**. Six verified products
(3.2 GB) and the frozen protocol remain for whoever picks this up.

**So the illumination bands on branch `part2/unsolved-band` are
synthetic-only and must be labelled that way wherever they are quoted.** They
are a large improvement on the band they replace — which was wrong in both
directions — but they have not met real data.

### Other open items

- The 60-90 degree **incidence** band is untested: both available pairs carry a
  1.56x and 2.29x scale gap, which confounds illumination with resolution.
- `minima-loftr` fails CHECK-03 on 5 of 9 real TMC-2 windows while its accuracy
  metrics look normal — only the control gate sees it. It scores 19/20 on NAC,
  so this is specific to that pairing and **unexplained**.
- All regime boundaries are synthetic, from one terrain generator, at one site.
- No result has been compared against published numbers. The research
  document's Easy/Medium credibility floors have never been run.

## 4. Decisions this asks for

1. **Change the default matcher?** `xoftr` is the only eligible candidate but
   is 5.3x over the 180 s demo budget as a universal default
   (`reports/runtime_budget.json`). `docs/routing_proposal.md` argues the real
   answer is a routing table rather than a single default — which is what the
   research document's section 28 predicted from the literature.
2. **Gates per window or per product?** The control gates re-run the matcher
   about five times per window and are 54-71% of end-to-end runtime. That one
   choice moves the budget more than the entire xoftr-vs-eloftr difference.
3. **Merge `part2/unsolved-band`?** It fixes a live config bug — the selector
   currently marks 75-180 degrees unsupported when matchers solve most of it —
   but its bands rest on `minima-loftr`, which is not shippable and whose real
   behaviour is unexplained.
4. **Promote `minima-loftr`?** Blocked on explaining the CHECK-03 failure.

## 5. The pattern worth reporting on its own

Four times a synthetic or label-derived result has failed against real data:

| claim | source | reality |
|---|---|---|
| TMC-2 GSD 4.41 m | PDS label | ~4.99 m, four matchers |
| unsolved above 60 deg azimuth | 4-matcher synthetic benchmark | wrong in both directions |
| illumination breaks matching | synthetic generator | not on real NAC terrain |
| minima-loftr is the best illumination matcher | synthetic generator | fails CHECK-03 on real TMC-2 |

Every one was caught by measuring rather than by review. The control gates in
particular have twice caught a result whose accuracy metrics looked normal —
which is failure mode #19 in the research document, and we have a live example
rather than a citation.
