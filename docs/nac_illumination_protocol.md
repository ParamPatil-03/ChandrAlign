# Real LRO NAC illumination validation — protocol (MATCH-09)

**Status: FROZEN. Committed before the measurement was written or run.**
Check `git log` on this file against `reports/lro_nac_illumination_validation.json`.
If a threshold below is edited after results exist, the experiment is void.

**This is a validation experiment. It changes no production behaviour.**
`default_matcher`, the routing rules, `shippable_matchers` and the quality-gate
thresholds are all out of scope and must not move as a result of it.

---

## 0. THE BOUNDARY OF THE CLAIM — read before quoting any result

This experiment validates **incidence difference** on real imagery. It does
**not** validate the solar-azimuth bands in `configs/regimes.yaml`.

| | status after this experiment |
|---|---|
| incidence difference, real NAC evidence | **measured** |
| solar azimuth difference, real evidence | **still not validated** |

LRO's ODE metadata gives incidence, emission and phase per product. It does
**not** give solar azimuth, and azimuth cannot be derived from incidence: two
scenes can share an incidence angle and differ by 180 degrees of azimuth, one
lit from the morning side and one from the evening side. Getting true azimuth
needs SPICE or an ephemeris, which this experiment does not use.

So no azimuth threshold may be rewritten from this result, in either direction,
and no single azimuth difference may be attributed to any pair here. A report
sentence of the form "d_azimuth X is solved" is out of scope by construction.

What it *can* settle is whether large illumination differences break matching on
real lunar terrain at all, which is the claim the bands actually make and which
has never been tested outside the synthetic generator.

## 1. Why this is worth doing now

Three times, a synthetic result has failed to survive contact with real data:

- TMC-2's label GSD (4.41 m) against ~4.99 m measured three ways;
- the old `unsolved_illumination` rule, wrong in both directions;
- `minima-loftr`, the best illumination matcher on synthetic pairs, failing
  CHECK-03 on 5 of 9 real TMC-2 windows.

Every illumination claim in this repository is currently synthetic, from one
terrain generator. `configs/regimes.yaml` has named the LRO NAC illumination
ladder as its validation set since it was written. The products are on disk and
have never been used.

## 2. Data, frozen

Nine LRO NAC CDR products (`LRO-L-LROC-3-CDR-V1.0`), 52224 x 5064, 16-bit
scaled I/F, already in `data/raw/lro/nac/`. Incidence from ODE
(`data/pairs/lro_nac_candidates.json`), the only source we have for it: the
CDR labels carry **no** illumination angle and **no** geolocation.

    incidence:  2.2  7.7  13.5  27.2  35.9  40.9  53.8  79.6  81.8 deg

**The 15 overlapping pairs, all of them, fixed here:**

| pair | inc gap | inc A / B | overlap lat x lon |
|---|---|---|---|
| M102000149LC / M102014464RC | 2.2 | 79.6 / 81.8 | 2.00 x 0.07 |
| M106719774LC / M175124932LC | 13.7 | 27.2 / 40.9 | 0.96 x 0.04 |
| M1417360906LC / M106719774LC | 19.5 | 7.7 / 27.2 | 0.13 x 0.09 |
| M172765160RC / M1415013176LC | 22.4 | 13.5 / 35.9 | 0.22 x 0.08 |
| M109080308LC / M106719774LC | 25.0 | 2.2 / 27.2 | 0.94 x 0.02 |
| M104362199LC / M102000149LC | 25.8 | 53.8 / 79.6 | 1.70 x 0.15 |
| M106719774LC / M104362199LC | 26.6 | 27.2 / 53.8 | 2.00 x 0.15 |
| M104362199LC / M102014464RC | 28.0 | 53.8 / 81.8 | 1.73 x 0.13 |
| M109080308LC / M175124932LC | 38.7 | 2.2 / 40.9 | 0.94 x 0.03 |
| M175124932LC / M102014464RC | 40.8 | 40.9 / 81.8 | 0.96 x 0.05 |
| M1417360906LC / M104362199LC | 46.1 | 7.7 / 53.8 | 0.19 x 0.02 |
| M106719774LC / M102000149LC | 52.3 | 27.2 / 79.6 | 1.64 x 0.09 |
| M106719774LC / M102014464RC | 54.5 | 27.2 / 81.8 | 1.67 x 0.20 |
| M1417360906LC / M102014464RC | 74.1 | 7.7 / 81.8 | 0.49 x 0.17 |
| M109080308LC / M102014464RC | **79.6** | 2.2 / 81.8 | 0.94 x 0.10 |

**Windows: 5 per pair, deterministic** — evenly spaced along the overlapping
latitude range at fractions 1/6, 2/6, 3/6, 4/6, 5/6, at the centre of the
overlapping longitude range. 512 px square at native resolution. No seed and no
random selection: the pair set is the sample, and re-running must reproduce it
exactly. 15 pairs x 5 windows = **75 window-pairs per matcher**.

## 3. Matchers

`xoftr`, `minima-loftr`, `eloftr`, `aliked-lightglue`, `sift-nn`.

Identical pipeline for every one: same window extraction, same coarse
alignment, same estimator (`estimate.robust`), same quality gate, same control
gates. `sift-nn` is the floor and is expected to fail early; it is present so
the experiment can demonstrate it is capable of failing.

## 4. How windows are paired, and why it is done this way

NAC CDRs are not map-projected and carry no per-pixel geolocation, so the
transform between two scenes of the same ground is unknown up front.

1. **Prior.** Each product's ODE bounding box is treated as a linear lat/lon
   frame over (line, sample), giving an approximate predicted location for a
   given lat/lon in each image. Crude -- it ignores rotation and pushbroom
   distortion -- so it is used only to get within a search margin.
2. **Coarse lock, matcher-independent.** A MIND template search
   (`preprocess.phase_congruency.mind`, ours, illumination-robust, no learned
   weights) at reduced resolution locks the window pair onto common ground.
3. **The matcher under test** then runs on the coarse-aligned pair.

Step 2 exists to separate two failures that would otherwise be reported as one:
*could not locate the overlap* and *could not match across the illumination
difference*. Without it, a matcher that never found the ground would be
indistinguishable from one that found it and failed, and the experiment would
answer neither question. A window-pair whose coarse lock fails is recorded as
`coarse_failed` and excluded from every matcher's score, identically.

## 5. There is no ground-truth transform, and none will be invented

These are real scenes with no absolute truth available. Accordingly:

**The accuracy measurement is the perturbation gate (CHECK-03)**, which is the
only known truth obtainable here: it injects a **known (3, 4) px shift** into
the real source image, re-runs the whole pipeline, and measures how far the
recovered transform moved from that known move. It is a controlled
transformation *within* real imagery, so it needs no external reference. It is
also the gate that caught `minima-loftr` on TMC-2.

Reported but explicitly **not** treated as accuracy:

- inlier RMSE -- self-consistency of the fit, not correctness;
- quality tier -- a decision about evidence, not a measured error;
- cross-matcher agreement -- corroboration, not truth.

**No result from this experiment may be described as sub-pixel accurate.** A
passing quality gate is not an accuracy claim, and this document says so in
advance so that the report cannot quietly upgrade it.

## 6. Success, frozen

A window-pair **succeeds** for a matcher when all three hold:

1. the estimator accepts (`EstimateResult.ok`);
2. every control gate passes -- null grey, null noise, perturbation, identity,
   shared-mask;
3. the perturbation gate recovers the known (3, 4) px move to within **1.5 px**
   (`gates.perturbation_tolerance_px`, `configs/default.yaml`, pre-existing).

Condition 3 is implied by 2 and is stated separately because it is the
load-bearing one.

**Band verdicts**, per incidence-gap band, over that band's window-pairs:

| verdict | threshold |
|---|---|
| solved | >= 90% succeed |
| degraded | >= 60% succeed |
| unsolved | below 60% |

Both numbers are taken unchanged from `docs/unsolved_band_protocol.md`, which
took them from `docs/default_matcher_protocol.md` §4.2 and §5. Nothing is
invented here.

**Bands, fixed in advance with their sample sizes**, which are uneven and small
at the extremes:

| band | pairs | window-pairs | note |
|---|---|---|---|
| 0-15 deg | 2 | 10 | **thin** |
| 15-30 deg | 6 | 30 | |
| 30-60 deg | 5 | 25 | |
| 60-90 deg | 2 | 10 | **thin** |

A band of 10 window-pairs from 2 scene pairs cannot distinguish a property of
the illumination gap from a property of those two particular scenes. The thin
bands are reported with that caveat attached, every time, and a verdict from
them is provisional. Eleven further products in the same bounding box are
listed in `data/pairs/lro_nac_candidates.json` and are not downloaded;
fetching them is the way to thicken these bands, and is out of scope here.

## 7. The three questions, and what would answer each

1. **Does the synthetic illumination behaviour survive on real terrain?**
   Answered by the band table. Monotonicity is *not* assumed -- the synthetic
   result was explicitly non-monotonic in azimuth, and the incidence axis may
   behave differently again.
2. **Does `minima-loftr` work in the hard illumination regime it was proposed
   for?** Answered by its success rate in the 30-60 and 60-90 bands
   specifically. This must **not** be inferred from its TMC-2 failure, which
   happened in a regime whose illumination difference is unknown.
3. **Does `xoftr`'s synthetic 75-135 degree failure appear on real data?**
   Answered by its success rate at large incidence gaps -- while remembering
   that the synthetic failure was on the *azimuth* axis and this is the
   *incidence* axis, so agreement is evidence and disagreement is not a
   contradiction.

## 8. Output

`reports/lro_nac_illumination_validation.json`, carrying at minimum:

    "source": "measured"
    "axis": "incidence_difference"
    "azimuth_status": "not_validated"

Per window-pair: both product ids, incidence of each, incidence gap, window
index, overlap size, matcher, success, tier, every control-gate result, the
perturbation recovery and error, inlier count and ratio, coarse-lock z-score,
runtime, and the rejection reason where rejected.

## 9. Standing commitment

No threshold, pair, window rule, matcher or band boundary above may change once
results exist. If the run reveals the method itself is faulty -- as the first
runtime benchmark did, where an OOM in one process corrupted every later
measurement -- the fix is to repair the harness and re-run the whole thing,
disclosing it, not to keep partial rows.

**A negative result is a result.** If the synthetic illumination map does not
survive on real NAC imagery, that is the finding, and it is reported as the
finding rather than explained away.
