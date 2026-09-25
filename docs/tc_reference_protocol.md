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
