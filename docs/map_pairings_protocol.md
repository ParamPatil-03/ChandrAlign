# Protocol: the remaining pairings against map-projected references (frozen before running)

Goal (research doc Opportunity #1, the "most defensible gap"): one generic pipeline registering
ANY Chandrayaan-2 instrument against the named references. Done so far: OHRC/TMC-2 -> NAC,
TMC-2 -> SELENE TC, IIRS -> WAC, IIRS -> NAC. This adds, with data already held (no download;
docs/usefulness_checks.md U4): TMC-2 -> SELENE MI, IIRS -> SELENE MI, OHRC -> SELENE MI,
TMC-2 -> LRO WAC, OHRC -> LRO WAC.

## References (both map-projected lat/lon grids)
- LRO WAC Global Morphologic Mosaic clip, 643 nm, 100 m/px (`data/raw/lro/wac_mosaic`).
- SELENE MI map tile `MI_MAP_03_N01E023N00E024SC` (0-1 N, 23-24 E), 2048 px/deg (14.8 m). One band,
  fixed now: **749 nm** for OHRC and TMC-2 (their 500-800 nm / panchromatic response); **1548 nm**
  (MI's longest NIR band, inside IIRS's 0.8-5 um range) for IIRS.

## Method (`scripts/register_to_map.py`, one engine for all five)
- Prior: SYSTEM corners plus our measured system offsets (the window-placement lesson: best
  known position): OHRC (741 E, 2064 N) m (median vs NAC); TMC-2 per row vs SELENE TC; IIRS per
  line vs the WAC mosaic. Position is still searched: the reference crop is the prior footprint
  +- max(3 km, 30 reference px).
- Coarse: `cascade.register_step_dense` (MIND). The FINER image is the template, block-averaged
  to the coarser pixel size; lock needs z >= 10.
- Fine, in the COARSER image's pixel grid (the information limit): the finer image warped
  (anti-aliased) onto it. Matchers, fixed now: eloftr for OHRC / TMC-2 (their routed matcher),
  xoftr tiled 640 for IIRS (its cross-modal route), and sift as the baseline; then
  `pipeline.fine_stage` with configured defaults, all five control gates, tier, source-px
  conversion and the MATCH-07 MI check.
- Windows: up to 5 per pairing, spread along the part of the source inside the reference with
  a margin; size fixed per source: OHRC 8192 px, TMC-2 1536 px (4000 for WAC), IIRS 256 lines.

## Success (per window, per matcher)
Lock; fine model; all five gates pass; tier >= LOW; MI check not flagged; and the implied system
offset (reference ground minus system ground at the window centre) within max(2 reference-pixel
diagonals, 150 m) of the independently measured offset used in the prior (NAC for OHRC, TC for
TMC-2, WAC for IIRS). The prior uses the same offsets, so this last check is only a false-lock
guard (the lock is searched over the whole crop), not an accuracy measure.

Verdict per pairing (routed matcher): solved >= 90% of windows, degraded >= 60%, else unsolved.
Accuracy is reported in SOURCE pixels (docs/source_pixel_accuracy.md): for OHRC -> WAC one
reference pixel is ~330 OHRC pixels, so that pairing can only be located, not sub-pixel.

## Predictions (written before running)
TMC-2 -> MI solved (3x gap, same kind of camera). OHRC -> WAC unsolved (the OHRC strip is ~37 WAC
px wide, like the 2 km NAC in IIRS -> NAC). No prediction for the other three.
