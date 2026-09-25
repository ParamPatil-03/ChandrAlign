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

## Result (2026-09-25): by the frozen rule, none solved -- but mostly for a reason the rule did not anticipate

Reports: `reports/map_{tmc2,iirs,ohrc}_mi.json`, `reports/map_{tmc2,ohrc}_wac.json`. Routed matcher:

| Pairing | Locks | Success (rule) | Why the others fail |
|---|---|---|---|
| TMC-2 -> SELENE MI | 5/5 (z 24-52) | 0/5 unsolved | all 5 are 210-317 m from the TC-derived offset (bound 150 m); 3 also REJECTED (coverage 0.08-0.25, known-shift gate) |
| IIRS -> SELENE MI | 3/5 | xoftr 2/5 unsolved (sift 3/5 degraded) | xoftr 154.6 m (bound 150); 2 windows no lock (one at the tile edge) |
| OHRC -> SELENE MI | 5/5 (z 21-24) | 2/5 unsolved | 3 REJECTED: an 8192 px OHRC window is only ~170 MI px, 41-68 matches, known shift 1.95-2.03 px (tolerance 1.5) |
| TMC-2 -> LRO WAC | 5/5 (z 13-21) | 1/5 unsolved | 4 are HIGH with no MI flag but 239-369 m from the TC-derived offset (bound 283 m); 1 MI-flagged |
| OHRC -> LRO WAC | 0/5 (z 5-8) | 0/5 unsolved | as predicted: an OHRC window is ~25 WAC px, no lock |

Predictions: TMC-2 -> MI "solved" was WRONG (by the rule); OHRC -> WAC "unsolved" was right.

**What the evidence says beyond the rule (not a re-decision):** the offsets are SYSTEMATIC, not
scattered, so they are not false locks -- the reference products disagree with each other:

| Through | new reference minus old reference, per window | spread |
|---|---|---|
| TMC-2: MI vs SELENE TC | (+97..+121 E, -194..-265 N) m | 8 / 23 m |
| TMC-2: WAC vs SELENE TC | (+79..+277 E, -197..-305 N) m | 72 / 42 m |
| IIRS: MI vs WAC | (+102..+109 E, +106..+110 N) m | 3 / 2 m |
| OHRC: MI vs NAC | (-50..-60 E, -18..-28 N) m | 4 / 4 m |

The consistency bound assumed the references agree to ~150 m; they differ by 60-300 m here. So
the "vs known offset" check was the wrong guard for pairs of DIFFERENT references (it guarded
against false locks, and there were none: every flagged or scattered result was SIFT, and the MI
check flagged all of SIFT's wild results -- 546 m, 8.8 km, 23.9 km, 227 km). A re-test would need a
new, pre-declared guard (e.g. within-pairing consistency, as docs/iirs_wac_protocol.md used).

Genuine limits shown: OHRC against coarse references is information-limited (~170 MI px or ~25 WAC
px per window); IIRS needs the tile to cover the window. TMC-2 -> WAC reaches HIGH on 4/5 windows.

## Amendment 1 (frozen BEFORE recomputing): within-pairing consistency
Replaces only the cross-reference bound (references disagree by 60-300 m). A window succeeds if it
locks, has a fine model, passes all five gates, tier >= LOW, MI check not flagged, AND its implied
offset is within max(2 reference-pixel diagonals, 150 m) of the MEDIAN implied offset over that
pairing's windows passing everything else (>= 3 such windows needed; else none succeed) -- the
guard docs/iirs_wac_protocol.md used. Computed from the same reports; no rerun, no other change.

## Amendment 1 result (`reports/map_pairings_amendment1.json`)
TMC-2 -> WAC 4/5 **degraded**; IIRS -> MI 3/5 **degraded**; TMC-2 -> MI 0/5 (only 2 windows pass the
gates+tier, < 3 needed); OHRC -> MI 0/5 (2 pass); OHRC -> WAC 0/5 (no lock). Second attempt of this
protocol: stops here. Remaining causes are real limits: few matches / low coverage at MI's 15 m
(TMC-2, OHRC) and OHRC's small footprint at coarse scale.
