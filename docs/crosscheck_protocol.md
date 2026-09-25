# Protocol: the independent cross-check gate in the product path (audit C-04). Frozen before measuring.

## Question

`control_gates.crosscheck_gate` catches consistently wrong answers that the five control gates pass by
design (RIFT2 caught 8 of 12 wrong-but-accepted synthetic results; the standard gates caught 2 of 12,
`reports/rift_crosscheck.json`). But nothing called it. Does it keep that catch rate when it runs inside
`pipeline.register_bundle`, and what does it cost on real, correct windows?

## The rule under test (fixed now)

- **Checker:** RIFT2 (`matching/rift.py`, our own implementation, licence-clear), run on the same
  ImagePlanes as the primary matcher, whenever the primary is not RIFT2. When the primary IS RIFT2 there
  is no checker from a different family in the product yet: the gate records "no checker", treated as
  inconclusive (below).
- **The checker's own verdict:** `robust.estimate` on its matches, then `quality.assess` (inliers, ratio,
  coverage, geometry, scale; no control gates). It counts only if the estimator accepted AND the tier is
  not REJECTED. This is exactly the rule `scripts/rift_crosscheck.py` measured.
- **Agreement:** `crosscheck_gate`, the RMS transform gap over a 16x16 grid of source pixels, flagged
  above `gates.crosscheck_flag_px` = 2.0 px (unchanged from the measured value).
- **Flag:** the gate `independent_crosscheck` fails, and the result is REJECTED with failure mode 12.
- **Checker cannot lock:** "inconclusive". This is not a pass and not a fail: the tier is capped at
  `gates.crosscheck_inconclusive_cap` = **MEDIUM**. HIGH requires independent confirmation; LOW and MEDIUM
  results are unchanged.
- Config switch `gates.crosscheck`, default **on**.

## Measurements

1. **Synthetic replay (the audit's bar).** Every (method, regime, seed) that `reports/rift_crosscheck.json`
   counts as accepted is re-run through `register_bundle(matcher=method)` on the same
   `synth.make_pair` inputs (`scripts/bench_rift.py` REGIMES, seeds 3/11/29, 512 px), with the gate on.
   Truth grades only. Reported:
   - **caught:** the 12 wrong-but-accepted cases that come back REJECTED. The ones rejected BY THE
     CROSS-CHECK are counted separately from those rejected by any other gate.
   - **false alarms:** correct results (< 2 px) that the cross-check flags.
   - **Bar:** caught by the cross-check >= **8/12**.
2. **Real windows (false-alarm and inconclusive rates):** the committed TMC-2 -> TC (15 windows), OHRC -> NAC
   (25) and IIRS -> WAC (5) windows, with the cross-check computed on each window's fine-stage frame and
   recorded next to the window's existing result. The success rules of those scripts are NOT changed by this
   measurement. There is no bar: the rates are reported as measured. A flag on a window the other
   evidence (probes, MI) calls correct is a false alarm.
3. **Periodic-aliasing trap:** two DIFFERENT real crops of similarly spaced repetitive terrain (not
   `np.roll` of one crop, which is a true correspondence), registered through `register_bundle` with the
   default matcher. Bar: not HIGH or MEDIUM with a wrong transform; REJECTED, or capped by the gate.

## Result 1: synthetic replay end to end (2026-09-26): bar MET

`scripts/crosscheck_replay.py --out reports/crosscheck_replay.json` (commit `76b4453` code): 105 accepted cases
of `reports/rift_benchmark.json` re-run through `register_bundle`, graded by the exact transform.

| outcome | count |
|---|---|
| wrong, flagged by the cross-check (REJECTED) | **8 / 12** (bar >= 8) |
| wrong, REJECTED by another gate (perturbation) | 2 / 12 |
| wrong, not rejected | 2 / 12: +90 deg lighting, 2.2 px (LOW) and 2.5 px (MEDIUM); RIFT2 abstained |
| correct, flagged (false alarm) | **0 / 64** checked |
| correct, confirmed | 64 (49 HIGH, 11 MEDIUM, 4 LOW) |
| correct, checker inconclusive (tier capped at MEDIUM) | 29 (+30 deg lighting 15, 2x scale 12, +90 deg 2) |

Before C-04, the product path accepted all 12 wrong results. After it, 10 are REJECTED, and the 2 remaining are
2.2-2.5 px wrong and neither is HIGH. The cost is the cap: a correct result whose regime RIFT2 cannot handle (+30 deg
lighting, a 2x scale gap) cannot be HIGH. That is the rule as frozen ("HIGH needs an independent method to
agree"). A second checker family (e.g. the MIND dense step) would recover them, and is listed as follow-up.

Measurements 2 (real windows) and 3 (periodic trap): below, once run.
