# Ablation Experiment 3 acceptance protocol — geometry-guided filtering (P2-T12)

**Status: FROZEN. Committed before the experiment was written or run.**
Check `git log` on this file against whatever result file Experiment 3 produces:
this document's commit must precede the result's. If any threshold, baseline,
metric or seed below is edited after results exist, the experiment is void and
must be re-run on new seeds.

Frozen against `origin/main` at `4962176` (2026-09-22). Every configuration
value quoted below is quoted **by value** from that commit, so that a later edit
to `configs/` cannot silently move the baseline out from under the result.

---

## 1. Why this document exists

Research document §33 names Experiment 3 "the novel-contribution measurement":
geometry-guided filtering is the one ablation stage for which no directly
comparable published result exists. Section 29 states the gap precisely — no
source demonstrates a general "compute incidence/emission/phase angle and DEM
slope per pixel, then use it to constrain or reweight candidate matches"
pipeline for cross-sensor lunar images. It is therefore the claim judges will
attack hardest, and the one claim we cannot defend by pointing at a paper.

The criterion it inherited cannot survive that attack.

### 1.1 The old wording, and why it cannot discriminate

PLAN.md line 953 reads:

> *"on a repetitive-crater-field subset, false-positive inliers drop measurably
> versus RANSAC-only, and **inlier precision** (not just count) is the reported
> metric."*

Two defects, each sufficient on its own.

**(a) "Measurably" is not a quantity.** A drop of 1% passes. A drop of 50%
passes. A drop attributable entirely to having thinned the match pool passes.
No run can fail a bar that names no number, so the bar grades nothing.

**(b) "RANSAC-only" is not a pipeline.** It does not say which robust driver, at
which reprojection threshold, at which iteration budget, on which data, at which
outlier rate, scored against which ground truth. Every one of those is a free
parameter, and a free parameter in a baseline is a dial that can be turned until
the treatment looks good.

### 1.2 Why "beats RANSAC" is the wrong claim even when fully specified

`reports/estimator_benchmark.json` (900 runs, committed on branch
`part2/magsac`) already settled this, and the result is the reason this protocol
is shaped the way it is.

| Outlier structure | magsac | ransac | usac_accurate |
|---|---|---|---|
| Agree with **nothing** (uniform), 150 runs each | 150/150 succeed | 149/150 | 149/150 |
| Agree with **each other** (structured), 150 runs each | 36/150 succeed | 41/150 | 40/150 |

Pooled over all 300 runs per driver, the structured half produced **109–114
confident wrong answers per driver** — runs where the estimator returned the
decoys' transform and reported a clean fit. Broken out by outlier fraction, the
structured half degrades identically for all three:

| Structured outlier fraction | magsac | ransac | usac_accurate |
|---|---|---|---|
| 0.3 | 0/30 wrong | 0/30 | 0/30 |
| 0.5 | 24/30 wrong | 19/30 | 20/30 |
| 0.7 | 30/30 wrong | 30/30 | 30/30 |
| 0.8 | 30/30 wrong | 30/30 | 30/30 |
| 0.9 | 30/30 wrong | 30/30 | 30/30 |

Scattered outliers are not the hard case. Outliers that agree with each other
are, and past 50% every driver fails totally.

**A consensus method cannot prefer the minority by construction.** No choice of
driver, threshold or iteration budget changes that, because the majority *is*
the criterion the method optimises. So "geometry filtering beats MAGSAC on
repetitive terrain" is a claim about a method that is definitionally unable to do
the thing being asked of it. Passing such a bar establishes nothing.

The scale sanity check does not rescue it either. The decoy in that benchmark
sits at scale 1.02 against a truth of 1.08 — 5.5% off, well inside
`estimate.scale_tolerance: 0.25` (`configs/default.yaml:28`). It is not a defence
and is not cited as one anywhere below.

**Consequence for this protocol.** Passing must mean the geometry carried
information the photometric match score did not. Section 5.3's count-matched
control exists solely to make that the only way to pass.

### 1.3 Disclosure: prior favourable evidence exists, and it does not vote

`tests/test_geometry_filter.py` on `origin/main` already measures this filter
favourably: on one real SLDEM tile it takes a match pool at 3.7% precision that
fools RANSAC into a confident wrong transform, and returns precision 1.0 with
every true match retained. That test was read while writing this protocol.

It is therefore demoted to a **consistency check with no vote**, exactly as
`docs/default_matcher_protocol.md` §1 demoted `reports/rift_benchmark.json`:

- every threshold in §5 is sourced **outside** it;
- the decision runs on **fresh seeds 301–315** (§4.4) that no one has run;
- it could not support the claim anyway — it is a single tile, at one location,
  same modality on both sides, same Sun on both sides, pure translation, SIFT
  only, no seed variation. It shows the mechanism can work once. It cannot
  measure a rate.

---

## 2. CLAIM

> **On repetitive lunar terrain, comparing DEM-derived slope and aspect at the
> two endpoints of each candidate match supplies information that the
> photometric match score does not, and using it reduces the rate of
> confidently-wrong registrations below the repository's brokenness tolerance —
> without discarding correct registrations, and by more than removing the same
> number of matches blindly achieves.**

The claim is FALSE if any one of the following is observed:

1. The geometry arm's confidently-wrong rate on the repetitive set exceeds 4 runs
   in 90 (§5.1).
2. The geometry arm does not accept at least as many correct registrations as the
   baseline it replaces (§5.2) — i.e. it bought precision by rejecting
   everything.
3. The geometry arm's confidently-wrong rate is not lower than the
   **count-matched blind-thinning** control's by a margin outside sampling noise
   (§5.3) — i.e. the improvement is explained by removing matches, not by which
   matches were removed.
4. Any negative control in §7 fails.

All four are conjunctive. Failing one fails the experiment; there is no weighted
score and nothing averages out. This mirrors `evaluate.quality.assess`, which is
a worst-of conjunction for the same reason: a single-signal gate leaks one
failure family or the other.

**A failure here is a publishable result, not an emergency.** §33 asks whether
this stage helps, not to assume it does. "Geometry filtering does not rescue
structured consensus failure past N% outliers" is the novel-contribution
measurement just as much as a pass is, and §9's per-configuration reporting is
designed so that the boundary is the deliverable either way.

---

## 3. BASELINE

Four arms. One code path, one generator, one gate, one ground truth. Nothing gets
a private path.

### 3.1 Values, pinned here by value

Every arm below runs these unless the arm's own row overrides them.

| Setting | Value | Source at `4962176` |
|---|---|---|
| Matcher | `aliked-lightglue` | `configs/regimes.yaml:22` (`default_matcher`) |
| Lowe ratio | `0.80` | `configs/default.yaml:13` (`matching.ratio_test`) |
| Mutual NN | `true` | `configs/default.yaml` (`matching.mutual_nn`) |
| Keypoint cap | `2048` | `configs/default.yaml` (`matching.max_num_keypoints`) |
| Library RANSAC | `skip_library_ransac: true` | `configs/default.yaml` |
| Estimator entry point | `chandralign.estimate.robust.estimate` | `src/chandralign/estimate/robust.py` |
| Robust driver | `magsac` → `cv2.USAC_MAGSAC` | `configs/default.yaml:21`; `robust.py:_METHODS` |
| Reprojection threshold | `3.0` px | `configs/default.yaml:22` |
| Iteration budget | `10000` | `configs/default.yaml` (`estimate.max_iters`) |
| Confidence | `0.9999` | `configs/default.yaml` (`estimate.confidence`) |
| Minimum inliers | `12` | `configs/default.yaml` (`estimate.min_inliers`) |
| Scale tolerance | `0.25` | `configs/default.yaml:28` |
| Quality gate | `chandralign.evaluate.quality.assess` | `src/chandralign/evaluate/quality.py` |
| Accept boundary | `min_inliers: 8`, `min_inlier_ratio: 0.325`, `min_coverage: 0.15` | `configs/default.yaml:87` (`tiers.low`) |
| Uniformity grid | `8`, `top_k_per_cell: 6` | `configs/default.yaml` |
| Device | as `device: auto` resolves, identical across arms | `configs/default.yaml` |

### 3.2 The four arms

| Arm | What it is | Difference from B0 |
|---|---|---|
| **B0** | The shipped pipeline | — (the table above, verbatim) |
| **B1** | The mitigation we already ship for failure mode #11 | `matching.ratio_test` → `0.70` (`ratio_test_repetitive`, `configs/default.yaml:14`) |
| **B2** | Count-matched blind thinning — the information control | Removes exactly as many matches as E3 removed on the same configuration and seed, chosen lowest-confidence-first; see §5.3 |
| **E3** | The treatment | `estimate.geometry_filter.filter_matches` inserted between matching and estimation, at its committed constants: `MAX_SLOPE_DIFFERENCE_DEG = 8.0`, `MAX_ASPECT_DIFFERENCE_DEG = 60.0`, `ASPECT_MEANINGFUL_SLOPE_DEG = 3.0` (`geometry_filter.py:48,49,51`) |

**B1 is not optional.** Research document §32 gives failure mode #11 an existing
mitigation — "stricter ratio-test threshold in repetitive regions" — and this
repository already ships it as `ratio_test_repetitive: 0.70`. If geometry
filtering does not beat a config value we already have, it has not earned the DEM
dependency, the geolocation dependency, or the code.

### 3.3 E3-reweight is judged separately and must earn itself

Research document §29 and P2-T12 describe two mechanisms: **reject** matches
whose terrain disagrees, and **reweight** survivors by geometric plausibility
into MAGSAC. Only reject is implemented at `4962176`.

- The §2 gate is applied to **E3-reject**, the committed behaviour.
- **E3-reweight**, if built, is a fifth arm. It is reported always, and it
  justifies shipping only if it passes the same conjunction *and* strictly beats
  E3-reject on §5.1. Otherwise it is extra complexity buying nothing (YAGNI).

Declaring which of the two is "the shipped one" after seeing results is forbidden
by §10. It is fixed here: E3-reject.

### 3.4 Cross-experiment hygiene

MATCH-03/04/05 (`docs/default_matcher_protocol.md`, branch
`part2/default-matcher`) may change `default_matcher`. That result **cannot**
retroactively move this baseline: the matcher here is pinned by value to
`aliked-lightglue`. If the default changes before Experiment 3 runs, Experiment 3
runs on **both** matchers and reports both tables. A baseline that moves when a
neighbouring experiment concludes is not a baseline.

---

## 4. TEST DATA

Three sets, two synthetic and one real. The synthetic sets decide; the real set
checks behaviour and has no vote on accuracy (§5.4).

### 4.1 Repetitive set — synthetic, exact ground truth (the primary set)

Generator: `chandralign.synth.make_pair`, unchanged, at `4962176`.

It is the right generator for this experiment specifically because
`synth.lunar_dem` draws crater radii from an inverse-cube size-frequency law,
which its own docstring states "produces the repetitive self-similar terrain that
failure mode #11 is about", and because `make_pair` returns full `GeometryLayers`
(DEM elevation, slope, aspect, incidence) for **both** planes alongside the exact
`H_true`. The filter therefore has real geometry to read and the scorer has exact
truth, from one call.

Six configurations. Sun angles are relative to `make_pair`'s reference default
`sun_ref = (135.0, 45.0)`.

| ID | `n_craters` | `sun_src` azimuth | `scale` | Why it is here |
|---|---|---|---|---|
| R1 | 200 | +0° | 1.0 | repetitive terrain, nothing else varying |
| R2 | 200 | +30° | 1.0 | routine between-orbit Sun drift |
| R3 | 400 | +0° | 1.0 | denser crater field, the ambiguity turned up |
| R4 | 400 | +30° | 1.0 | both stressors together |
| R5 | 400 | +90° | 1.0 | shadows move and partly invert; photometric evidence is at its weakest and geometry at its most useful |
| R6 | 400 | +0° | 2.0 | TMC-2 → SELENE TC, a real shipped route at a real scale ratio (§1.4: TMC-2:SELENE-TC ≈ 1:2) |

15 seeds × 6 configurations = **90 runs per arm**.

**Manipulation check (runs before anything is scored).** If arm B0's
confidently-wrong rate on this set is **below 0.20**, the set did not reproduce
the failure mode and the experiment is **void, not passed** — a treatment cannot
be shown to fix a failure that did not occur. 0.20 is four times the §5.1 pass
threshold, so it is arithmetic from an already-sourced number and not a new free
parameter. The closest committed measurement (0.73–0.76 over the structured runs,
1.00 at outlier fractions ≥ 0.7) suggests a genuinely repetitive set clears it
easily; if it does not, the fix is a harder set, declared and re-frozen, not a
lower bar.

### 4.2 Decoy set — synthetic, structured outliers at known rates

The repetitive set produces whatever false-consensus rate the terrain happens to
produce. The decoy set fixes that rate by construction, so the result can be read
as a function of outlier fraction and the boundary can be located.

Construction, reproducing `reports/estimator_benchmark.json`'s structured case
exactly, with terrain attached so the filter has something to read:

| Parameter | Value | Source |
|---|---|---|
| Correspondences | 200 | `estimator_benchmark.json:n_points` |
| Image shape | 512 × 512 | `estimator_benchmark.json:shape` |
| Inlier noise | 0.5 px | `estimator_benchmark.json:noise_px` |
| Truth transform | scale 1.08, rotation 6.0°, shift (12.0, −7.0) | `estimator_benchmark.json:truth` |
| Decoy transform | scale 1.02, rotation −4.0°, shift (−35.0, 28.0) | `estimator_benchmark.json:decoy` |
| Outlier fractions | 0.5, 0.7, 0.9 | the three at which every driver already fails |

Source points are drawn on a `synth.lunar_dem` height field so that both
endpoints of every correspondence have a real slope and aspect; true endpoints
are placed by the truth transform, structured outliers by the decoy transform.
Outlier fraction 0.3 is excluded: all three drivers already score 30/30 there, so
it cannot discriminate.

15 seeds × 3 fractions = **45 runs per arm**.

### 4.3 Negative-control set — synthetic, cases that must keep working

| ID | Configuration | What it protects |
|---|---|---|
| N1 | identity: source is the reference, `H = I` | the CHECK-01..06 control gates |
| N2 | `n_craters` 70 (generator default), +0°, scale 1.0 | the ordinary case |
| N3 | `n_craters` 70, +30°, scale 1.0 | ordinary case under routine Sun drift |
| N4 | `n_craters` 70, +0°, scale 2.0 | a real shipped route, normal texture |

15 seeds × 4 configurations = **60 runs per arm**.

**N5, the no-DEM replay:** every run in N1–N4 repeated with the DEM withheld. Not
a fifth configuration — a replay of the same 60, used by §7.3.

### 4.4 Seeds

**Seeds 301–315**, fifteen consecutive integers, fixed here before any run.

They are fresh by inspection of every seed this repository has used:
`reports/estimator_benchmark.json` used 0–9; `scripts/bench_matchers.py` fitted
the `tiers.low` gate on 3, 11 and 29; `docs/default_matcher_protocol.md` reserved
101–110. None recurs, and none of 301–315 has been run by anything.

### 4.5 Real-data set — behaviour only, no accuracy vote

Pairs already registered in `data/pairs/registered_pairs.json` (22 usable of 25):
`TMC-2 → SELENE TC` (the 3 headline pairs), `OHRC → TMC-2`, `OHRC → SELENE TC`.
DEM: SLDEM2015 tiles under `data/raw/dem/sldem2015`, loaded through
`chandralign.io.dem.find_tiles`.

Three questions only, all yes/no:

1. Does the filter run to completion and report `applied: True` where a DEM
   covers the footprints?
2. Does the registration already established for these pairs survive it — does
   the transform stay within `gates.perturbation_tolerance_px: 1.5`
   (`configs/default.yaml`) of the committed result?
3. Does the run still pass the control gates?

**No accuracy number from this set is reported as evidence for or against the
claim.** §5 is decided on synthetic data, for the reason in §5.4.

### 4.6 Run count

195 runs per arm × 4 arms = **780 runs**, plus the 60-run N5 replay per arm, plus
E3-reweight's 195 if it is built. Comparable in scale to the 900-run estimator
benchmark and the 560-run default-matcher benchmark.

---

## 5. GROUND TRUTH and METRIC

### 5.0 Ground truth

**Synthetic (§4.1–4.3): exact, by construction.** `make_pair` returns `H_true`
mapping source pixels to reference pixels exactly. It grades runs. It is never an
input to any arm.

A **registration is WRONG** when

```
RMSE_grid(H_hat, H_true) > 2.0 px
```

where `RMSE_grid` is the root-mean-square Euclidean distance between `H_hat(p)`
and `H_true(p)` over a 16×16 regular grid of source pixels `p`.

A **correspondence `(s_i, r_i)` is FALSE** when

```
|| H_true(s_i) - r_i || > 2.0 px
```

**Source of 2.0 px:** `BAD_PX = 2.0` in `scripts/bench_matchers.py:48` ("above
this, a transform is wrong enough to matter"), and the default
`tolerance_px: float = 2.0` of `inlier_precision` in
`src/chandralign/estimate/geometry_filter.py:159`. Both committed before this
protocol. Not chosen here.

**Real (§4.5): none exists, stated plainly.** These pairs have no exact truth —
TMC-2 → SELENE TC is scored on SYSTEM corners only. Nor can the labels supply
one: this repository's own `scripts/audit_gsd.py` found NAC labels off by up to
2.33×, which is why `configs/default.yaml` carries `unverified_scale_slack: 3.0`.
A number derived from a label already known to be that wrong cannot grade a
sub-2-pixel claim. Hence §4.5's three yes/no questions and nothing more.

### 5.1 PRIMARY METRIC — confidently-wrong rate

For a set of N runs in one arm:

```
FC = |{ r : accepted(r) = true  AND  RMSE_grid(H_hat_r, H_true_r) > 2.0 px }| / N
```

where `accepted(r)` is `chandralign.evaluate.quality.assess(...).accepted` at the
§3.1 values.

This is primary rather than inlier precision because it is what the product
actually does wrong. Research document §32 names it twice: failure mode #12
(false correspondences surviving RANSAC) and failure mode #19 (confident numbers
that are not real). A run that fails loudly is a nuisance. A run that is wrong and
says it is fine is the failure the whole control-gate layer exists for.

**PASS THRESHOLD: FC ≤ 0.05 — at most 4 of the 90 repetitive-set runs, and at
most 2 of the 45 decoy-set runs.**

**Source of 0.05:** `gates.null_max_inlier_ratio: 0.05` in
`configs/default.yaml:104`, committed before this protocol, whose own comment
reads *"a null test scoring above this means we are broken."* It is the only
number already in this repository that answers the question "what rate of
confident nonsense means the system is broken." The mapping is stated honestly:
the committed constant bounds a ratio over points and this bounds a rate over
runs. Reusing the repository's existing 1-in-20 brokenness tolerance is a choice.
Inventing a fresh number after seeing the results would be a worse one, and §10
forbids it.

The integer counts are fixed now so that rounding cannot be argued later:
0.05 × 90 = 4.5 → **≤ 4**; 0.05 × 45 = 2.25 → **≤ 2**.

The bar is failable and non-trivial: the closest committed measurement of this
failure is FC = 0.73–0.76 across the 150 structured runs per driver (109–114
wrong), and 1.00 at outlier fractions 0.7, 0.8 and 0.9 for all three drivers.
Clearing 0.05 is roughly a fifteenfold reduction on the pooled figure and a total
reversal at the high fractions.

### 5.2 SECOND GATE — correct acceptances must not fall

```
CA = |{ r : accepted(r) = true  AND  RMSE_grid(H_hat_r, H_true_r) <= 2.0 px }|
```

**PASS: `CA(E3) >= CA(B0)` on the repetitive set and on the decoy set.**

FC alone is gameable to zero by rejecting every run, which is why this gate is not
optional and why it needs no external constant — it is a monotonicity
requirement, not a threshold. A filter that improves precision by refusing to
register anything is worthless, and this is the line that catches it on the hard
sets. §7 catches it on the easy ones.

### 5.3 THIRD GATE — the information control

**Reported metric** (inlier precision, the one thing the old wording got right),
via the committed `estimate.geometry_filter.inlier_precision`:

```
P = |{ i in inlier set : ||H_true(s_i) - r_i|| <= 2.0 px }| / |inlier set|
```

Precision is reported for every arm and every run. It does not gate, because it
rises trivially as the match pool shrinks — which is exactly the confound this
gate removes.

**B2, the count-matched blind thinner.** For each configuration and seed, E3
reports `n_rejected = n_rejected_slope + n_rejected_aspect` (fields already on
`FilterReport`). B2 then runs the same configuration and seed and removes
**exactly `n_rejected` matches** by a rule that cannot see terrain:
lowest-confidence-first. A second variant removes them uniformly at random under
an RNG seeded by the run identifier. Both variants are reported; the gate uses
whichever performs **better**, so the control is never weakened by picking the
weaker straw man.

**PASS: `FC(E3) < FC(B2)`, with the 95% confidence interval on the difference of
the two proportions excluding zero (two-proportion test, α = 0.05, two-sided).**

**Source:** research document §32 failure mode #19 — fabricated or inflated
evaluation numbers, a failure this field has already produced in public — and this
repository's committed `scripts/sabotage.py` discipline, whose premise is that a
result which would appear anyway under a deliberately uninformative substitute is
not evidence of anything. Blind thinning is that substitute. If geometry cannot
beat it, then what helped was removing matches, not knowing which to remove, and
the novel contribution is not novel.

**Pre-registered power, so that a small gap cannot be spun afterwards.** At 90
runs per arm, a two-proportion test at α = 0.05 two-sided has 80% power against an
absolute FC difference of about 0.21 near a 50% baseline. **Differences smaller
than that are reported as "not established at this N" — never as an improvement.**
If that outcome occurs and a real effect is still suspected, the answer is more
seeds declared in a new frozen amendment, not a softer reading of these.

**PASS also requires `FC(E3) < FC(B1)`** by the same test, for the §3.2 reason: a
DEM dependency must beat a config value we already ship.

### 5.4 Why the decision is synthetic

Every number in §5.1–5.3 needs exact per-match ground truth: which
correspondences were false, and whether the returned transform was wrong. §5.0
establishes that the real pairs do not have it and, given a 2.33× label error
already measured in this repository, cannot be made to have it at this precision.
Deciding a sub-2-pixel precision claim on data whose truth is looser than the
claim would be the §32 failure mode #19 pattern with extra steps. The real set
therefore answers behaviour questions only.

### 5.5 Reported but not gating

Reported for every run and every arm, because they are what the problem statement
asks for and what the boundary is made of: RMSE, inlier count, inlier ratio,
spatial coverage (grid-cell occupancy, `uniformity.grid: 8`), runtime, assigned
confidence tier, and every `FilterReport` field (`applied`, `reason`, `n_in`,
`n_kept`, `n_rejected_slope`, `n_rejected_aspect`, `n_unknown_terrain`,
`median_slope_difference_deg`, `median_aspect_difference_deg`).

---

## 6. PASS THRESHOLD — the whole gate in one place

Conjunctive. All five. No weighting, no averaging.

| # | Gate | Threshold | Source of the number |
|---|---|---|---|
| 1 | Confidently-wrong rate, repetitive set | `FC ≤ 0.05` → **≤ 4 of 90** | `gates.null_max_inlier_ratio: 0.05`, `configs/default.yaml:104` |
| 2 | Confidently-wrong rate, decoy set | `FC ≤ 0.05` → **≤ 2 of 45** | same |
| 3 | Correct acceptances | `CA(E3) >= CA(B0)`, both hard sets | monotonicity; no constant needed |
| 4 | Information control | `FC(E3) < FC(B2)` **and** `FC(E3) < FC(B1)`, difference CI excluding zero | §32 #19; `scripts/sabotage.py` premise |
| 5 | Negative controls | all of §7 | §7, per row |

Supporting constants, all pre-existing: "wrong" = **2.0 px**
(`scripts/bench_matchers.py:48`; `geometry_filter.py:159`). Set validity =
baseline `FC ≥ 0.20` on the repetitive set (§4.1), four times gate 1.

---

## 7. NEGATIVE CONTROL

A filter that raises precision by rejecting everything is worthless. Gate 3
catches that on the hard sets. These catch it everywhere else, and they catch the
subtler version: a filter that quietly discards correct registrations on the
cases that already worked.

### 7.1 Correct registrations must survive nearly intact

On the negative-control set (§4.3, 60 runs), for every run:

**PASS: the filter retains at least 90% of the true inliers the baseline found**

```
|{ i in kept : ||H_true(s_i) - r_i|| <= 2.0 px }|
------------------------------------------------------  >=  0.90
|{ i in matched : ||H_true(s_i) - r_i|| <= 2.0 px }|
```

**Source of 0.90:** `gates.identity_min_inlier_ratio: 0.9`
(`configs/default.yaml:110`), committed, whose comment states the principle
directly — *"an image against itself should keep ~every match."* That is exactly
the requirement here, applied to a correct registration rather than a literal
identity, and it was not chosen for this experiment.

### 7.2 Accept-rate regression: at most 1 run in 60

**PASS: at most one run in the 60 that B0 accepted correctly is not accepted
correctly by E3, and its cause is diagnosed in the report.**

**Source:** the near-zero tolerance construction of
`docs/default_matcher_protocol.md` §4.3, which set the same kind of bound at "at
most 1" of 80 runs on the grounds that a component which lies confidently is worse
than one that fails loudly. The mirror-image reasoning applies: a component that
silently deletes correct work is worse than one that declines to act. One in sixty
is a fault to explain, not a budget to spend.

Additionally, on every N1–N4 run the accepted result must still clear the
committed accept boundary — `min_inliers: 8`, `min_inlier_ratio: 0.325`,
`min_coverage: 0.15` (`configs/default.yaml:87`). Coverage is called out
explicitly: the problem statement mandates uniformly distributed match points, so
a filter that improves precision by collapsing the matches into one corner has
failed the deliverable even where its RMSE improves.

### 7.3 With no DEM it must be a bit-for-bit no-op

On the N5 replay (§4.3), for every run: `applied: False`, `n_kept == n_in`, and
**every reported metric identical to the B0 run on the same configuration and
seed**. Not "similar" — identical.

**Source:** the committed contract of `filter_matches` itself, which returns the
match set untouched with `reason` = *"no DEM supplied: physics filter disabled"*.
The docstring's own words are *"It never silently does nothing"*; this control is
what makes that testable rather than aspirational. Any drift here means the filter
has a side effect nobody declared.

### 7.4 The existing control gates, with the filter in the chain

CHECK-01..06 must still pass with the filter inserted, at their committed
tolerances: `null_max_inlier_ratio: 0.05`, `perturbation_shift_xy: [3, 4]`,
`perturbation_tolerance_px: 1.5`, `identity_max_rmse_px: 0.25`,
`identity_min_inlier_ratio: 0.9` (`configs/default.yaml:104–110`).

The null test is the important one here. A filter with a bug that correlates its
keep-decision with the estimator's own output could manufacture agreement out of
noise, and `scripts/sabotage.py` exists in this repository precisely because a
cross-validator was once found that could be made vacuous while every test stayed
green.

---

## 8. LIMITS — what this experiment cannot establish

Stated before the result, so that none of it reads as an excuse afterwards.

**8.1 The synthetic DEM is the same DEM that made the image.** This is the single
most important limit and it bounds the headline. In `make_pair`, both views are
rendered from one height field and the filter then reads that same height field.
Slope and aspect therefore agree at a true match *perfectly, by construction*. The
synthetic result is an **upper bound on what geometry filtering can do**, not an
estimate of what it will do.

**8.2 Real DEM resolution is the unmeasured gap.** SLDEM2015 is used in this
repository at ~59 m/px (`tests/test_geometry_filter.py`, `plane(...) gsd_m=59.0`).
OHRC is 0.25 m/px (§1.4). Per-match slope lookup on real OHRC is sampling a
surface roughly 236× coarser than the features being matched. Nothing in §4.1–4.3
measures that, and §4.5 cannot (no ground truth). **The protocol cannot conclude
that geometry filtering works on real OHRC data.** It can conclude that it works
where the DEM resolves the terrain, and that is a narrower claim which must be
stated as such in any demo or write-up.

**8.3 One generator, one crater law, one photometric model.** `synth.lunar_dem`
uses a single inverse-cube size-frequency law; `synth.render` is Lambertian, which
the Moon is not — the filter's own docstring already concedes the point for its
brightness term. Other repetitive terrains named in §32 (mare ridges, ejecta
patterns) are not represented.

**8.4 It tests these constants, not the idea.** A negative result is a result
about `MAX_SLOPE_DIFFERENCE_DEG = 8.0` / `MAX_ASPECT_DIFFERENCE_DEG = 60.0` /
`ASPECT_MEANINGFUL_SLOPE_DEG = 3.0` on this data. It does not refute
geometry-guided filtering in general, and must not be reported as though it did.
Re-tuning those constants after seeing results and re-running is permitted only
under §10 — new frozen document, new seeds.

**8.5 Slope and aspect only.** §29 and §33 describe incidence, emission and phase
angle as well. The committed filter compares slope and aspect and down-weights on
a Lambertian brightness prediction. Emission and phase are not exercised at all,
so this experiment measures a proper subset of the stated novel contribution and
the write-up must say which subset.

**8.6 It says nothing about the pairings we do not hold.** Research document §19
places OHRC↔IIRS in the Extreme tier where two independent teams already failed.
No arm here touches it. Experiment 3's result must not be quoted as covering it.

**8.7 It is not an accuracy result.** Sub-pixel accuracy is Experiment 5 (§33,
§31). Scale bridging is Experiment 4. A pass here means fewer confident lies on
repetitive terrain — nothing about RMSE on the pairs that already work beyond §7's
"did not get worse."

**8.8 Passing does not establish a mechanism.** Gate 4 establishes that the
geometry carried information the blind control did not. It does not establish
*which* geometric signal did the work; the per-run split of `n_rejected_slope`
against `n_rejected_aspect` (§5.5) is reported as a hypothesis for a later
experiment, not as a finding of this one.

---

## 9. Reporting

The deliverable is the boundary, not the verdict. Regardless of pass or fail, the
result file reports FC, CA and precision **per configuration and per outlier
fraction**, all four arms, never only pooled. The pooled number decides §6; the
per-configuration table is the thing worth having, because it locates where
geometry stops helping — and on a stage §33 calls the novel-contribution
measurement, that boundary is the contribution whichever side of the gate it falls
on.

The write-up states, in this order: the claim (§2), the result against each of the
five gates, and §8 in full. §8 is not an appendix.

---

## 10. Standing commitment

No threshold, baseline arm, metric definition, data configuration, seed or
priority in this document may be changed once results exist. If one is changed,
**the experiment is void and must be re-run on new seeds**, under a new frozen
document, and the void run must be reported as void rather than discarded
silently.

This includes, explicitly, all of the following, each of which is a dial that
would otherwise be turnable after the fact:

- the 2.0 px "wrong" bar, and the 0.05 / 0.20 / 0.90 thresholds;
- the choice of B0's matcher, driver, reprojection threshold and iteration budget;
- the filter's three constants (`8.0`, `60.0`, `3.0` degrees);
- the six repetitive configurations, the three decoy outlier fractions, and seeds
  301–315;
- which of E3-reject and E3-reweight the gate applies to (§3.3: reject);
- the decision that the real-data set does not vote on accuracy (§5.4).

The gate returns a verdict mechanically. **Whether to ship geometry filtering is
the team's decision, not this experiment's** — the experiment produces the
measurement and its limits.
