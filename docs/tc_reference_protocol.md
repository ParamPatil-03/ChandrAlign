# Protocol: SELENE TC as the reference for OHRC and IIRS (frozen before running)

The PS names SELENE images as a reference. TMC-2 -> TC is solved; OHRC -> TC exists only as a
cascade step and IIRS -> TC not at all. Same engine and rules as docs/map_pairings_protocol.md
(amendment 1): `scripts/register_to_map.py --reference tc --tile <tile>`, SELENE TC ortho map tiles
(TCO_MAP_02, 4096 px/deg, 7.4 m), up to 5 windows per (source, tile) whose footprint lies inside the
tile: OHRC on N03 and N00 (it straddles 0 deg), IIRS on N09, N03 and N00. Matchers: eloftr + sift
for OHRC, xoftr (tiled 640) + sift for IIRS.

Success (per window, routed matcher): lock (z >= 10), fine model, all five gates, tier >= LOW, MI
check not flagged, and the implied offset within max(2 TC-pixel diagonals, 150 m) = 150 m of the
MEDIAN implied offset over that pairing's windows (all tiles) passing everything else (>= 3 needed).
Verdict: solved >= 90%, degraded >= 60%, else unsolved. Reported beside it (not ruled): the
systematic difference from the independent offsets (OHRC vs NAC, IIRS vs WAC), and accuracy in
source px.

Predictions: IIRS -> TC solved (a well-textured 7.4 m reference, like IIRS -> NAC's wide NAC but
with far more overlap); OHRC -> TC at least degraded (an 8192 px OHRC window is ~340 TC px, twice
OHRC -> MI's ~170).

## Result 1 (2026-09-25): both unsolved by the rule
- OHRC -> TC: all 10 windows lock (MIND z 13-24) but the fine stage gets 7-10 matches on a ~342 px
  frame (inlier ratio ~0.08): known-shift gate fails, MI flags. Information-limited, as OHRC -> MI.
- IIRS -> TC: 12/15 windows pass the gates (9 HIGH), 11 pass all but consistency, only 3 fall within
  150 m of the single median -- because IIRS's system error DRIFTS along the strip: the implied
  offset runs smoothly east 747 -> 1574 m over lines 0-4096, tracking our WAC-derived drift
  (920 -> 1486 m), with TC - WAC ~ +300 m north. One window (line 3072, east 3088 m) is off the trend.

## Amendment 1 (frozen BEFORE recomputing; IIRS only): consistency with a linear drift
Replace "within 150 m of the median" by "within 150 m of a robust straight-line fit of the implied
offset (east, north) against source line", fitted over the windows passing everything else
(Theil-Sen per axis; >= 4 needed). Everything else unchanged; computed from the same reports.
OHRC -> TC stays unsolved (information limit; a dense-lock-only variant is not attempted).
