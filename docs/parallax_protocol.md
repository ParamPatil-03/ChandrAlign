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
