# Protocol: is OHRC -> NAC M175124932LC (0.40 m) biased? A closure test (frozen before running)

Question (docs/source_pixel_accuracy.md): on the finest held NAC, the MATCH-07 MI check finds its
peak 4.2-5.9 NAC px (~2 m) CROSS-TRACK from our registered model, consistently; on coarser NACs
MI agrees with the model to 0.07-0.23 px. Is the registration biased, or is MI misled (shadows
under different suns)? Absolute checks cannot answer it: NAC corners are rounded to 0.01 deg.

## Third reference
LRO NAC M111443315LC (LROC archive, the PS's named source; 0.52 m, incidence 26 deg), downloaded
2026-09-25 with approval.

## Closure test
For each OHRC window (2048 px, shipped behaviour of scripts/register_ohrc_nac.py) registered to
BOTH A = M175124932LC and B = M111443315LC:
- T_A: OHRC px -> A px and T_B: OHRC px -> B px (the pipeline's registrations);
- T_AB: A px -> B px, registered DIRECTLY (A crop around T_A(centre), warped onto B's grid by the
  prior T_B . T_A^-1, then eloftr + the robust estimate on the pair; the prior only places it).
Closure error at the window centre c: e = T_B(c) - T_AB(T_A(c)), in metres (B's pixel size),
split into A's cross-track (sample) and along-track (line) directions.

## Reading (fixed now)
- |median cross-track e| >= 1.0 m on >= 2 windows, same sign  -> the loop does not close cross-track;
  with the MI evidence, the OHRC -> A registration is taken as biased.
- |median e| < 0.5 m -> the loop closes; the MI peak on A is taken as misled (not a bias).
- otherwise: inconclusive.
The MI check is also run on OHRC -> B and A -> B, reported.
