# Protocol: terrain parallax in the fine stage (ALIGN-08), frozen before measuring

## Why (measured 2026-09-24, while testing ALIGN-05)

The refill test left 17 empty grid cells on the three TMC-2 -> TC windows of tile N00 (none on
N03/N09). They are **not** empty for lack of data or texture: 80-100% of each has valid data
in both images, and re-matching each one returns 124-313 matches that agree with each other
to ~1 px but sit 5-25 px from the affine model, almost purely along-track. Phase correlation
per cell (no matcher involved) gives the same offsets.

Cause: the TMC-2 product is the **aft** view (`nca`; fore/nadir/aft = ncf/ncn/nca, +-26 deg
along-track). An oblique view displaces each point along-track by ~height x tan 26 deg, which
a 2-D affine cannot represent. Test (prediction written before running): per-cell offset
from the model against SLDEM height above the window's best-fit plane gives r = -0.74 to
-0.91 on all 5 windows tested, slope 1.03-1.26x the predicted tan 26 deg / 7.4 m. N00 has
+-100-180 m of relief (offset sd 3.6-7.2 px); N03/N09 +-10-24 m (0.3-0.8 px).

So on relief the delivered affine is wrong by up to ~25 px locally, the correct matches
there are discarded as outliers (hence the empty cells), and neither the tier nor the gates
can see it (the known-shift gate measures lock precision, not model fit).

## The stage (`pipeline.parallax`, default off)

Needs a DEM and a ground model (as `geometry_filter` does); skipped with a reason otherwise.
For every match, h = SLDEM height at its source point minus the median over the first
robust estimate's inliers. Model: `ref = A . src + h . p`, A affine (6), p a 2-vector in
px/m (the parallax per metre of height). Fitted by least squares on the first estimate's
inliers, then re-selecting inliers from ALL matches at the configured reprojection
threshold (3 px), repeated until the set is stable (max 5 rounds).

What changes: WHICH matches are inliers (and so which control points are delivered, and
the TPS through them). What does not change: the delivered affine `model` stays the first
robust estimate, so the gates and tier certify exactly what they did before. `p`, h0 and
the affine+parallax residuals are reported in `stages["parallax"]`.

## Measurement

All 9 TMC-2 -> TC windows (`--tile all --windows 3`), parallax on vs off (the off run is
`reports/tmc2_tc_registration.json`). Accuracy is measured independently of the matcher:
per 8x8 cell with full valid data, the phase-correlation offset between the two images
(same method as above), compared with each model's predicted offset at the cell centre.

## Decision (all must hold to switch it on)

(a) Every N00 window ends with <= 1 empty cell.
(b) On every N00 window, RMS of (phase-corr offset - affine+parallax prediction) over cells
    is <= 50% of RMS against the affine alone; on N03/N09 it is no worse by > 0.2 px.
(c) No window loses registration, a gate, or tier.
(d) p is physical: along-track (|p_y| >= 5 |p_x| in the north-up TC frame) and |p| within
    0.5-1.5x tan 26 deg / gsd on every N00 window.

Limits: one TMC-2 product (aft view), one DEM (SLDEM2015, 59 m/px, coarser than the 7.4 m
grid, so relief finer than ~8 px is not modelled). The DEM is also an input to the model;
the phase-correlation measure is not, so (b) is not circular. It does not check the TPS.

## Result (2026-09-25): NOT adopted by rule (b) -- one window short

`reports/tmc2_tc_parallax.json` (all 9 windows, parallax on; baseline
`reports/tmc2_tc_registration.json`). RMS = phase-correlation offset minus model, per cell.

| window | empty cells | RMS affine | RMS affine+parallax | ratio | p / predicted |
|---|---|---|---|---|---|
| N00 1562 | 1 -> 0 | 3.62 px | 2.24 px | **0.62** | 0.90 |
| N00 3125 | 6 -> 0 | 7.30 | 2.57 | 0.35 | 0.95 |
| N00 4687 | 10 -> 0 | 6.43 | 2.37 | 0.37 | 1.01 |
| N03/N09 (6) | 0 -> 0 | 0.25-1.07 | 0.21-0.49 | 0.28-0.87 | -- |

(a) pass: 17 -> 0 empty cells. (c) pass: every window keeps registration, tier and all five
gates. (d) pass: p along-track, 0.90-1.01x the 26 deg prediction on N00. (b) **fails**: N00
1562 improves 38%, the rule asked for 50%. Every window improves; none gets worse. Inliers
on N00 rise 53-85% (8,743 -> 13,339; 6,941 -> 12,611; 7,742 -> 14,351).

By the frozen rule the stage stays **off**. Not re-decided on these numbers.

Observed, not tested: all three N00 windows keep ~2.2-2.6 px after the correction, against
0.2-0.5 px on flat ground. SLDEM2015 is 59 m/px (8 TC px), so relief finer than that is not
modelled; a finer DEM would test this. The inlier re-selection hit its 5-round cap on N00
(still growing), so its inlier counts are a lower bound.

## Amendment 1 (2026-09-25, frozen BEFORE any run on these windows): replication on fresh windows

The run above stays the record and stays "not adopted". Rules (a)-(d), their thresholds and
the stage (incl. its 5-round cap) are **unchanged**. Only the data is new.

Fresh N00 windows (all hilly, chosen from SLDEM relief alone, before any registration):
- rows 1562, 3125, 4687 moved to the strip's LEFT and RIGHT edges (`--col-offset -1232` and
  `+1232`: TMC-2 columns 0-1536 and 2464-4000). Each shares 20% of its area with the old
  centre window at that row, and none with any other.
- row 6250, centre (rows 5482-7018): no overlap with any earlier window; relief ~+-90 m.

Each window is run with parallax off (baseline) and on. Rules (a), (b) (N00 clause), (c)
and (d) are applied to these 7 windows. **Adopt (`pipeline.parallax: true`) only if all
four hold on the fresh set**; the first run's 1562 shortfall is reported next to it either way.
A window that fails to register in the BASELINE is excluded from (a), (b), (d) and reported.

## Amendment 1 result (2026-09-25): NOT adopted again

`reports/tmc2_tc_parallax_fresh.json` (7 fresh N00 windows, off vs on).

| window (row, col) | empty cells | RMS affine | RMS affine+parallax | ratio | p / predicted |
|---|---|---|---|---|---|
| 1562, 768 | 4 -> 0 | 4.92 px | 2.69 px | 0.55 | 0.87 |
| 3125, 768 | 4 -> 0 | 5.80 | 2.94 | 0.51 | 1.02 |
| 4687, 768 | 0 -> 0 | 0.21 | 0.19 | 0.89 | 0.44, not along-track |
| 1562, 3232 | 0 -> 0 | 3.33 | 1.67 | 0.50 | 0.97 |
| 3125, 3232 | 5 -> 0 | 5.34 | 2.84 | 0.53 | 0.97 |
| 4687, 3232 | 12 -> 2 | 12.25 | 6.00 | 0.49 | 0.82 |
| 6250, centre | 11 -> 1 | 8.86 | 7.87 | 0.89 | 0.98 |

(c) passes (nothing lost). (a) fails (4687/3232 keeps 2 empty cells), (b) fails (4 of 7
windows cut the error by less than 50%), (d) fails on 4687/768 -- which turned out flat
(0.21 px), because relief was checked at the strip centre, not at that column. **Stays off.**
This is the second failure; per the standing rule, work on it stops here and is reported.

What both runs show together (16 windows): no window got worse on any measure, empty
cells 53 -> 3, inliers up ~50-95% on hilly windows, p matches the 26 deg geometry. But the
correction typically halves the error rather than removing it (2-3 px left on most hilly
windows, 6-8 px on two). Not tested: whether a finer DEM than SLDEM2015 (59 m/px) removes
the rest; row 6250's small gain (8.9 -> 7.9 px) with a physical p suggests something else
there too.

## Amendment 2 (2026-09-25, frozen BEFORE any run with the new DEM): a finer height model

Question: is the ~2-3 px left after the correction the DEM's resolution? Only the height
model changes: `--dem tc_dtm`, the SELENE TC stereo DTM (`DTM_MAP_02_N00E021S03E024SC`,
4096 px/deg ~7.4 m, same grid as the TC reference; fetched by `fetch_selene.py --bundle dtm`)
in place of SLDEM2015 (59 m). It is used for the terrain filter, for h in the stage and for h
in the evaluation (so the model's and the evaluation's heights agree). It is NOT independent
of TC -- it is the height model TC's ortho map was made with, which is why it is the right one
for parallax against TC; it is recorded as not independent.

Rules (a)-(d), thresholds, stage and 5-round cap unchanged. Windows: the 10 N00 windows used so
far, filtered by ONE criterion fixed now and computed from SLDEM2015 before any run: SLDEM
relief (p95 - p5 of height minus its best-fit plane) over the window's own footprint (TMC-2
system corners) >= 100 m. This corrects amendment 1's selection error (relief read at the
strip centre, not the window). Rules (a), (b) N00 clause, (c) and (d) apply to the selected
windows; the baseline (parallax off) is the SLDEM run already recorded for each.
**Adopt parallax, with the TC DTM where a tile is held, only if all four hold.**
Because these windows have been seen (with SLDEM), a pass here is a replication with a new
input, not fresh data; that is stated with the result.

### Amendment 2a (same day, still BEFORE any run with the new DEM): the selection criterion was wrong

Computing it showed that relief over the SYSTEM-corner footprint does not match relief where
the windows actually registered (4687/col 768: 127 m by system corners, 15 m at its registered
position; 3125/768: 89 m vs 243 m) -- the window-placement lesson again: use the best known
position, not the raw system one. Criterion replaced by: SLDEM relief (p95 - p5 about the
best-fit plane) over the window's REGISTERED position in its baseline (parallax-off) run
>= 100 m. This uses only the DEM and the baseline registration, nothing from the parallax
runs. Result, computed now: 9 selected (178-526 m); **4687/col 768 excluded (15 m)**.
Everything else in amendment 2 unchanged.

## Amendment 2 result (2026-09-25): NOT adopted (fails b, c) -- the cause of (c) is the terrain filter

`reports/tmc2_tc_parallax_tcdtm.json` (9 selected windows, TC DTM, parallax on; baselines
are the SLDEM parallax-off runs).

- (a) pass: empty cells 53 -> 0. (d) pass: p = 0.97-1.04x the 26 deg prediction on all 9
  (0.82-1.02x with SLDEM). Fit residual on the stage's inliers 0.74-0.98 px (1.0-1.3 with SLDEM).
- (b) fail: phase-correlation RMS ratio 0.31-0.48 on 7 windows, but 0.57 (1562) and 0.94 (6250).
- (c) fail: 7 of 9 windows lose tier, two to REJECTED (3125/768, 4687/3232). **Every drop is
  the terrain filter (ALIGN-03), not parallax:** fed the 7.4 m DTM it discards 33-40% of matches
  (1-2% with SLDEM), which lowers the FIRST estimate's inlier ratio, and the tier is graded
  on that first estimate, before the parallax stage runs. Amendment 2 coupled the two by
  declaring one DEM for both; that coupling, not the correction, is what failed.

Third failure: work stops here and is reported.

Observed, not tested:
1. The terrain filter's slope/aspect thresholds were set against 59 m slopes; at 7.4 m, local
   slopes are steeper and a third of real matches fail them.
2. The phase-correlation measure seems to have a ~2 px floor on hilly ground (2.1-2.5 px
   left even where the stage's inliers fit to < 1 px): a 200 px cell spans several heights,
   so it has no single offset. Rule (b)'s 50% cannot be met by a window whose affine RMS is
   below ~4 px (1562: 3.6 px). At 6250 the measure barely moves (8.8 -> 8.3 px) while empty
   cells go 11 -> 0 and the fit is 0.79 px, so that measure is unreliable there.
A clean test would use the TC DTM for the parallax stage only (SLDEM for the terrain filter)
and held-out matches (as docs/tps_protocol.md) instead of per-cell phase correlation.
