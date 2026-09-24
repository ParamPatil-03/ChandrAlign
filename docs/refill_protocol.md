# Protocol: empty-cell refill (ALIGN-05). Frozen before measuring.

## Question

Uniform delivered points leave some grid cells empty (TMC-2 -> SELENE TC: three N00 windows at
coverage 0.859-0.984). Research doc section 30 step 3: re-search ONLY those cells. Does a
local second pass fill them with points the model agrees with?

## Method

New pipeline stage `refill` (off until decided). After the uniform thinning, for each empty
grid cell: crop that cell (padded by 24 px) from the source and reference fine-frame images,
run the SAME matcher on the crop pair, map the matches back, and keep only those within
**2 px** of the pipeline's affine model (the model is never refitted, so gates, tiers and
scale checks cannot change). At most top-k (6) per cell, best confidence first. Refilled
points then go through the same per-point sub-pixel step. They are counted separately
(`refilled`), as `uniformity.merge_refill` requires.

Data: the nine TMC-2 -> TC windows (`scripts/register_tmc2_tc.py --tile all --windows 3`),
refill on vs off, current defaults otherwise.

## Decision

Refill ON if: (a) coverage rises on at least 2 of the windows whose coverage is < 1;
(b) no window loses registration, tier or a gate; (c) the refilled points' residual to the
model is within the 2 px bound by construction, and their RMS residual after sub-pixel
refinement is no more than 1.5x that of the primary delivered points (they must not be
worse evidence than what they supplement).

## Result (2026-09-24): not adopted

`reports/tmc2_tc_refill.json`. Windows with empty cells: 1562 (1 empty), 3125 (6), 4687 (10).
The local second pass filled 1 cell on 1 window (4687: coverage 0.844 -> 0.859; the refilled
point's residual 1.59 px vs 1.72 px for primary points). Rule (a) needs 2 of 3 windows ->
**not adopted; `refill` stays off.** Nothing else changed (model and gates untouched).

**Hypothesis below tested 2026-09-25 and WRONG** (see docs/parallax_protocol.md): the empty cells
have 80-100% valid data and ordinary texture; they are where terrain parallax of the TMC-2
aft view (26 deg) moves the true matches 5-25 px off the affine, so they fail its threshold.
Original hypothesis: most empty cells lie where one image has no valid data (rotated-window
corners, the TC tile edge near N00), where no matcher can find anything; if so, coverage should
be reported over cells with valid overlap in both images, which is a reporting change, not a
refill.
