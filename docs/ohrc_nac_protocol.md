# Protocol: OHRC -> LRO NAC on real data (frozen before any measurement)

## Why

Research doc section 19/25: OHRC <-> LRO NAC is the **Easy / credibility-floor** pairing
(solved in arXiv:2509.04775: SuperGlue, 0.57-0.92 px RMSE). We have never run it.

It is also, with the NAC products we hold, the first **real-data illumination test of
learned matchers** beyond 60 deg: the OHRC scene has a grazing sun (incidence 82.7 deg,
ground azimuth 269.8 deg), and the five overlapping NAC products span:

| NAC | NAC px (m, across x along) | NAC incidence | NAC ground sun az | d_az vs OHRC | d_incidence | role |
|---|---|---|---|---|---|---|
| M102014464RC | 1.29 x 1.17 | 81.8 | 270.2 | 0.4 | 0.9 | **same-sun control** |
| M106719774LC | 1.28 x 1.26 | 27.2 | 265.8 | 4.0 | 55.5 | large incidence difference |
| M175124932LC | 0.24 x 0.56 | 40.9 | 92.0 | **177.8** | 41.8 | **opposed sun** |
| M1417360906LC | 0.75 x 0.82 | 7.7 | 285.3 | 15.4 | **75.0** | NAC sun near overhead |
| M109080308LC | 0.53 x 0.54 | 2.2 | 197.4 | 72.4 | **80.5** | NAC sun overhead (azimuth ~meaningless) |

(NAC pixel sizes and incidences: LROC, `data/pairs/tmc2_nac_lroc_meta.json`; ground
azimuths: `data/pairs/nac_sun_azimuth_computed.json`; OHRC: its PDS4 label.)
M102000149RC also overlaps but has no LROC corner metadata here; excluded.

## Method (script: `scripts/register_ohrc_nac.py`)

Per NAC product, **5 OHRC windows** of 2048 x 2048 native px (~0.61 km), centres spread
evenly along the OHRC centre column where the window lies inside the NAC footprint.

1. **Prior, system-level only:** OHRC's system geometry grid (independent of any
   reference) and LROC's NAC corner coordinates give orientation and scale; position
   is SEARCHED over +-4 km (OHRC system error ~2 km measured; NAC product error up to
   ~1.25 km measured in PR #18).
2. **Coarse lock:** `cascade.register_step_dense` (MIND search + sub-pixel) with both
   images block-averaged to ~4 m. Lock requires z >= `cascade.min_z` (10).
3. **Fine stage:** OHRC warped (anti-aliased) onto the NAC pixel grid, matched with the
   matcher under test, then `pipeline.fine_stage` with the configured defaults
   (terrain filter, uniform points, sub-pixel). All five control gates run on the
   same pipeline; `quality.assess` gives the tier.

**Matchers on identical windows:** `eloftr` (the routed default), `minima-loftr` (the
synthetic winner above 60 deg; licence-clean, not yet shippable), `sift` (the baseline
every paper uses).

## Success (frozen)

A window succeeds for a matcher when ALL hold:
- coarse lock z >= 10;
- the fine stage returns a model;
- all five control gates pass (incl. the known (3, 4) px shift recovered within 1.5 px);
- tier >= LOW;
- **consistent location:** the implied OHRC-system -> NAC offset is within 150 m of the
  median offset of that product's other locked windows. (PR #18 measured within-product
  scatter of 23-87 m for true locks; a false lock lands at an unrelated place.)

Verdict per (product, matcher), from its 5 windows: **solved** >= 90% (5/5), **degraded**
>= 60% (3-4/5), else **unsolved** -- the same bars as the unsolved-band protocol.

## Predictions, registered now

1. Same-sun control (M102014464RC): `eloftr` solved. If it is not, the harness, not the
   illumination, is the problem, and the other rows are not interpreted.
2. Opposed sun (M175124932LC, d_az 178 deg): the synthetic sweep gives `eloftr` 0/10 and
   `minima-loftr` 10/10 at 180 deg. Predicted: `eloftr` unsolved, `minima-loftr` solved.
3. Large incidence differences (M106719774LC, M1417360906LC, M109080308LC): the sweep held
   elevation fixed, so there is **no prediction**; exploratory.
4. `sift`: below `eloftr` wherever the sun differs (arXiv:2509.04775's finding).

## Reported, not decided on

- Inlier RMSE of the fine stage in NAC px (the kind of number arXiv:2509.04775 reports;
  an internal residual, NOT accuracy).
- Delivered-point count and coverage; runtime.

## Limits

One OHRC scene, five NAC products, 5 windows each: 2-3 windows' difference is not
meaningful. One site (~0 N, 23.5 E). The known-shift gate measures precision of the
lock on this pair, not absolute accuracy; the consistency check guards against false
locks, it does not measure accuracy either.

## Amendment 1 (before any of the four non-control products was measured)

The frozen placement ("along the OHRC centre column") found **zero** windows for four of
the five products: the NAC strips are 1.2-6.5 km wide and cross OHRC's 3.6 km strip
off-centre, so OHRC's centre column misses them. Nothing was measured for those four;
this is a harness limit, not a result.

Change, placement only: for each candidate OHRC row, use the OHRC column **closest to the
centre** whose 2048 px window lies inside the NAC footprint (candidate columns every 256
px). Rows are then spread evenly as before. Success rules, thresholds, matchers and
predictions are unchanged. (Correction, recorded after the run: the control's window rows
also shifted by up to 125 px, because rows are now snapped to the new candidate set. Its
result was 5/5 for all three matchers both times.)

## Amendment 2 (before the opposed-sun product was measured)

Run 2 (placement per amendment 1) still found **zero** windows for M175124932LC, the
opposed-sun product and prediction 2's test. Cause, a harness bug: the inside-footprint
test padded each side by the window's FULL extent in NAC px instead of HALF of it. On a
strip 5064 px wide at 0.24 m (1.2 km), no window can pass that test. Fix: pad = 1.1 x the
half-extent. Every product's placement changes, so **all five are re-run (run 3) and
run 3 is the result**; runs 1-2 are kept in `docs/ohrc_nac_results.md` for the record,
not counted. Success rules, thresholds, matchers and predictions are unchanged.
