# No single matcher covers the required regimes

Evidence: `reports/default_matcher_selection.json` (720 runs, MATCH-03/04/05)
and `reports/illumination_sweep.json` (600 runs, MATCH-09, branch
`part2/unsolved-band`). Both ran under criteria frozen before measurement.

**This is a proposal. Nothing here is applied.** `configs/regimes.yaml` still
ships `default_matcher: aliked-lightglue`.

---

## The capability matrix

| regime | who can do it | who cannot |
|---|---|---|
| same modality, sun < 60 deg | everyone | -- |
| **sun 75 deg** | **minima-loftr 10/10, alone** | xoftr 0/10, eloftr 3/10, aliked 2/10 |
| **sun 90-120 deg** | **minima-loftr 6-8/10, alone** | everything else 0-1/10 |
| **sun 135-180 deg** | minima-loftr 10/10; xoftr 10/10 at 180 | eloftr, aliked 0/10 |
| **cross-modal** | **xoftr 10/10, minima-loftr 10/10** | the other seven, 0/10 |
| **2x resolution gap** | xoftr, eloftr, aliked, sift-lg 10/10 | **minima-loftr 0/10** |

The two strongest matchers are **complementary, not ranked**:

- `xoftr` owns cross-modal and the resolution gap, and fails across 75-135 deg.
- `minima-loftr` owns the whole illumination range, and cannot cross a 2x
  resolution gap at all.

Neither dominates the other, and no third candidate covers either gap. That is
not a tie to be broken by a tighter threshold; it is the shape of the problem.
It is also what section 28 of the research document predicted from the
literature -- a hybrid, regime-dependent architecture rather than a single
off-the-shelf matcher -- reached here independently from our own runs.

## Why "which matcher is the default" is the wrong question

The frozen protocol asked for one default and returned exactly one eligible
candidate, `xoftr`. Everything since has been the same fact from new angles:

1. `xoftr` cannot meet the demo budget as a universal default -- 922 s against
   180 s, 5.3x over (`reports/runtime_budget.json`).
2. `xoftr` cannot cover 75-135 deg of sun azimuth, where only `minima-loftr`
   works.
3. The cheap matchers ARE adequate on the path we actually ship: on real
   TMC-2 to SELENE, `eloftr` registers 9/9 in 107.5 s and `aliked-lightglue`
   8/9 in 100.7 s, both inside budget, against `xoftr`'s 948.3 s.
4. `aliked-lightglue`'s failures are concentrated where it should never be
   routed: all 15 of its false confidences in the selection benchmark fall in
   regimes outside its lane, and in its lane it is 50/50 with zero. The
   illumination sweep says the same thing independently -- 22 of 30 confident
   wrong answers, clustered at 75 deg and 180 deg.

A single default is forced to be simultaneously fast enough for every window
and capable enough for the worst regime. Nothing measured is both.

## The proposed routing

| condition | matcher | why |
|---|---|---|
| cross-modality | `xoftr` | only matcher at 10/10; cost paid rarely |
| sun azimuth >= 75 deg | `minima-loftr` | only matcher that works there at all |
| 2x+ scale gap, same modality | `eloftr` | 10/10, and `minima-loftr` is 0/10 here |
| otherwise | `eloftr` | 9/9 on real data, inside budget |

`aliked-lightglue` is not in this table. It is not disqualified -- it is simply
not the best in any cell.

## What blocks this

**`minima-loftr` is not in `shippable_matchers`.** It is licence-clean (MINIMA
Apache-2.0 fine-tuning LoFTR Apache-2.0, `reports/licence_audit.json`) but has
never been put through a promotion decision. Three of the four rows above, and
every illumination band above 60 deg, depend on it.

Without it the regime map is barely better than the wrong one it replaces:

| | old (wrong) | new, shippable only | new, with minima-loftr |
|---|---|---|---|
| solved to | 60 deg | 60 deg | **75 deg** |
| 90-120 deg | unsolved | unsolved | **degraded, 6-8/10** |
| 135-180 deg | unsolved | unsolved | **solved, 10/10** |

**Promoting `minima-loftr` converts 8 of 12 sampled azimuths from unsupported
to supported.** That is the largest single capability gain available in Part 2,
and it is a promotion decision, not a research problem.

## CORRECTION (post-dates the proposal above)

`minima-loftr` has since been run on the 9 real TMC-2 -> SELENE TC windows and
**fails CHECK-03 on 5 of them** -- median perturbation error 1.77 px against a
1.5 px tolerance, where xoftr, eloftr and aliked-lightglue score 0.05, 0.08 and
0.01. On one window it recovered a known (3, 4) px move as (1.21, -0.85): the
y component has the wrong sign. Its accuracy metrics were fine throughout, so
only the control gate saw it.

**So the illumination lane in the table above has no matcher supported by real
data.** Promotion of `minima-loftr` is blocked pending an explanation.

A SECOND CORRECTION, to this document and to commit 6675c44. Both described
those 9 windows as "similar-illumination pairs". That was never checked and is
not supported: TMC-2's label gives sun_azimuth 104.27 deg, but SELENE TC is
`SLN-L-TC-5-ORTHO-MAP-V2.0`, a MOSAICKED ortho map whose label carries no
illumination angle because it is assembled from many orbits. The regime of
those pairs is UNKNOWN, not easy, and the failure above therefore cannot be
placed on the azimuth axis at all.

That has a consequence beyond this matcher: **the illumination bands in
`configs/regimes.yaml` cannot be validated against TMC-2 <-> TC**, because the
reference product cannot supply a sun azimuth. `regime.py` correctly marks such
pairs `illumination_known=False` and falls back, but the bands stay
SYNTHETIC-ONLY until we register pairs that carry per-product sun geometry --
the LRO NAC illumination ladder that `regimes.yaml` already names as the
validation set and that has never been built.

## What is still unmeasured, and must not be assumed

- `minima-loftr` has **never been run on real data**. Every number above for it
  is synthetic. The routing should not ship before it gets the same real
  TMC-2 to SELENE check that `xoftr`, `eloftr` and `aliked-lightglue` have had.
- Its runtime is known only synthetically (~0.47 s at 512 px). Its scaling
  exponent is unmeasured, and `xoftr`'s k = 2.82 is the warning that a dense
  matcher can be far worse at real window sizes than at 512 px.
- Routing adds a failure mode of its own: a pair sent to the wrong lane gets a
  matcher chosen for conditions it is not in. The selector already refuses to
  guess when sun geometry is incomplete, and that behaviour becomes more
  load-bearing under this proposal, not less.
- All regime boundaries are synthetic, from one terrain generator.
