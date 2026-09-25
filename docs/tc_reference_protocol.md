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

## Amendment 1 result: IIRS -> TC 8/15 -- unsolved (degraded needs 60%); stop
`reports/tc_reference_verdict.json`. 8 of the 9 windows passing gates/tier/MI sit 0-40 m from a
linear drift (192 m east, 51 m north per 1000 IIRS lines); the 9th (line 3072, tier LOW) is 1763 m
off -- a lock only this consistency check catches. The other 6: 2 MI-flagged, 1 REJECTED, 1 fine
frame too small, 1 no lock, 1 at z 10.4. Second attempt of this protocol: stops here. What is shown:
where IIRS -> TC locks, it is coherent to tens of metres along 4000 lines and agrees with the WAC
registration up to a near-constant ~300 m TC - WAC difference. OHRC -> TC: information-limited.

## Amendment 2 (user-authorised third attempt; frozen BEFORE running)
Diagnosis from the reports: every failing IIRS window had half its latitude span outside the held
TC tiles (the unheld 3-6 N gap and tile edges); IIRS's 25 km width also runs past the tiles' east
edge at 24 E on every window (fine: only the covered part is matched).
- IIRS -> TC: windows placed only where the whole 256-line footprint lies inside a held tile in
  LATITUDE (east clipping allowed); rerun; same success rule as amendment 1 (linear drift, 150 m).
- OHRC -> TC: the dense MIND lock is the answer (as docs/iirs_nac_protocol.md amendment 1), checked
  by: a FRACTIONAL known shift of the reference crop (3.5, 4.25) px, cubic, recovered within 0.5 TC
  px (the integer shift was found tautological there); constant and noise nulls must not lock; the
  MI check on the dense model not flagged; consistency: within 150 m of a linear drift fit over the
  pairing's passing windows (>= 4). Same 10 windows.
Verdicts as before. If this fails, the TC question stops for good.

## Amendment 2 result (2026-09-25) -- final for this question (`reports/tc_reference_verdict_a2.json`)
- **IIRS -> TC: 11/15 = degraded** (was 8/15). All 15 windows lock with full-latitude placement; 14 pass
  the gates (12 HIGH); the 11 passing gates, tier and MI sit 0-22 m from a linear drift of IIRS's
  system error (187 m east, 53 m north per 1000 lines). Misses: 1 REJECTED, 3 MI-flagged (the N00
  tile's south end, MI peak at the +3 px search edge).
- **OHRC -> TC (dense lock): 0/10 by the rule, yet the lock is solid**: all 10 lock (z 12.8-24.2);
  fractional known shift recovered to 0.01-0.16 TC px; both nulls pass on 10/10; implied offsets agree
  to ~30 m (east 526-559, north 2229-2250 m). The MI check flags 7/10 -- at NMI ~1.00-1.04, where the
  NMI surface is nearly flat (1.0 = no shared information), as on the finest NACs. Only 3 pass all
  checks (< 4 needed): unsolved by the frozen rule.
- Follow-up (a new rule, NOT applied here): the MI check should abstain ("inconclusive") when its
  NMI surface is flat instead of flagging; it would need its own frozen test on known-good and
  known-bad pairs.
- TC vs the other references: OHRC offsets vs NAC-derived differ by ~(-200, +180) m; IIRS vs WAC by
  ~+300 m north -- consistent with the reference-to-reference differences found before.
