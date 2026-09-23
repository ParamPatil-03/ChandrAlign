# Real NAC solar-azimuth validation — protocol (MATCH-09)

**Status: FROZEN. Committed before the measurement was written or run**, and
before the imagery finished downloading. Check `git log` against
`reports/lro_nac_azimuth_validation.json`.

**Changes no production behaviour.** `default_matcher`, the routing rules,
`shippable_matchers` and the quality-gate thresholds are out of scope.

---

## 0. What this can and cannot establish

`configs/regimes.yaml` indexes its illumination bands on **solar azimuth
difference**. Every number in those bands comes from one synthetic terrain
generator. This is the first real-data test of that axis.

| azimuth difference | status |
|---|---|
| 0-15 deg | already measured (`reports/lro_nac_illumination_validation.json`) |
| **~90 deg** | **structurally unavailable — see below** |
| **126.7 deg** | **1 scene pair, tested here** |
| **175-180 deg** | **3 scene pairs, tested here** |

**Why ~90 degrees cannot be obtained, and will not be by searching harder.** An
equatorial target imaged from a polar orbit is lit from the east or from the
west, never from the side: the sun is near azimuth 90 or 270 and passes through
the intermediate angles only in the few orbits around local noon, when the
incidence is also near zero and the azimuth is geometrically ill-defined. Four
LROC photometric sites were scanned (`b95e354`); three sit within 1 degree of the
equator and produced azimuths in two tight clusters with nothing between. Reiner
Gamma, at 7 degrees, produced the single 126.7 degree pair used here. Reaching
90 degrees needs a high-latitude site, and none is published.

This matters because the synthetic model says **90 degrees is the worst case**,
not 180. So this experiment tests the band the model predicts is *easier* and
cannot reach the one it predicts is *hardest*.

## 1. The pairs, frozen

Six products, two sites. Chosen for azimuth difference with **incidence and
pixel scale held**, so azimuth is the variable that moves.

| pair | d_azimuth | incidence A/B | scale ratio | site |
|---|---|---|---|---|
| M1132219897R / M1180506270R | **126.7** | 15 / 19 | 1.00 | Reiner Gamma |
| M117338434L / M131494509L | **180.0** | 84 / 79 | 1.00 | Apollo 11 |
| M111443315R / M122054682L | **175.3** | 26 / 28 | 1.06 | Apollo 11 |

The two Apollo 11 pairs give opposed sun at **high** and at **low** incidence.
If difficulty depends on incidence and azimuth jointly rather than on azimuth
alone, that is where it shows.

Azimuths are **computed**, not scraped: the ground azimuth from the target to
the sub-solar point by spherical trigonometry, from LROC's sub-solar latitude
and longitude (`data/pairs/apollo11_pho_azimuth.json`,
`data/pairs/pho_sites_azimuth.json`). LROC's own `Sub solar azimuth` field is
in an instrument frame and gave physically impossible ground directions; that
error and its correction are recorded in `06789d9`.

## 2. Method

Identical to `docs/nac_illumination_protocol.md` sections 3-6, unchanged, so the
two experiments are comparable:

- same 5 matchers: `xoftr`, `minima-loftr`, `eloftr`, `aliked-lightglue`, `sift-nn`
- 5 deterministic windows per pair, 512 px, at latitude fractions 1/6 .. 5/6
- MIND coarse lock, matcher-independent, z >= 6 or the window-pair is excluded
  identically for every matcher
- **success** = estimator accepts AND all five control gates pass AND the
  perturbation gate recovers the known (3, 4) px move within 1.5 px
- the perturbation gate is again the only known-truth accuracy measure; inlier
  RMSE and quality tier are reported and are **not** accuracy

## 3. THE PRE-REGISTERED PREDICTION

This is the point of the experiment. The synthetic sweep
(`reports/illumination_sweep.json`, 600 runs) makes sharp per-matcher
predictions at these angles. They are written down here, before the data exists,
so the model can be **falsified** rather than merely compared against.

At **126.7 deg** (interpolating the 120 and 135 degree columns):

| matcher | synthetic prediction |
|---|---|
| minima-loftr | succeeds, roughly 7-10 of 10 |
| xoftr | **fails or nearly fails**, 0-4 of 10 |
| eloftr | **fails**, 0 of 10 |
| aliked-lightglue | **fails**, 0 of 10 |
| sift-nn | fails, 0 of 10 |

At **175-180 deg**:

| matcher | synthetic prediction |
|---|---|
| minima-loftr | succeeds, 10 of 10 |
| xoftr | succeeds, 10 of 10 |
| eloftr | **fails**, 0 of 10 |
| aliked-lightglue | **fails**, 0 of 10 |
| sift-nn | fails, 0 of 10 |

**What each outcome would mean, decided now:**

- **Predictions hold** → the bands survive their first real test on this axis,
  at two of the three angles they cover. The 90 degree band stays unvalidated.
- **Everything succeeds, including eloftr and aliked-lightglue** → the same
  result already found on the incidence axis, where the synthetic collapse did
  not appear on real terrain. Two axes agreeing that the synthetic model is
  pessimistic would make the bands themselves the thing to re-derive, not the
  routing built on them.
- **Everything fails** → the coarse lock or the harness is suspect before the
  matchers are, because `sift-nn` succeeding anywhere else in this project at
  180 degrees has never happened and its failure here would carry no
  information.
- **The 180 degree pairs split by incidence** (one succeeds, one does not) →
  azimuth alone does not index difficulty, and a one-dimensional band structure
  is the wrong shape regardless of its numbers.

No threshold moves on any of these outcomes. This protocol records what would
be concluded; `configs/regimes.yaml` changes only by a separate decision.

## 4. Limits

- Three scene pairs. Two sites. Any band verdict rests on one or two pairs and
  is provisional in exactly the way section 6 of the incidence protocol
  describes.
- The 126.7 degree point is a **single** scene pair. One pair cannot separate a
  property of the azimuth difference from a property of that terrain.
- ~90 degrees is untested and untestable from this archive.
- Emission angle is not controlled. It varies between the paired observations
  and is recorded per window, not held.
- Both sites are near-equatorial mare or mare-adjacent terrain. Polar terrain,
  which is where the published literature reports classical matchers failing,
  is not represented at all.
