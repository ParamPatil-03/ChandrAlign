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
