# OHRC -> LRO NAC on real data: results

> **Superseded in part:** the 55 deg and opposed-sun "no lock" rows below were a window-placement
> error; see `docs/illumination_fix_results.md`, section Q8 (both register 5/5).

Protocol frozen before measuring: `docs/ohrc_nac_protocol.md` (two placement-only
amendments, both before the affected products were measured). Result = **run 3**,
`reports/ohrc_nac_registration.json`. OHRC `ch2_ohr_ncp_20240330T0035085365`: grazing sun,
incidence 82.7 deg, ground azimuth 269.8 deg. 5 OHRC windows (2048 px, ~0.61 km) per
NAC product; three matchers on identical windows after one shared coarse lock.

## Result

| NAC | d_az / d_incidence vs OHRC | coarse lock (MIND, z >= 10) | eloftr | minima-loftr | sift |
|---|---|---|---|---|---|
| M102014464RC (same sun, control) | 0.4 / 0.9 deg | **5/5** (z 25-55) | **5/5 solved** | **5/5 solved** | **5/5 solved** |
| M106719774LC | 4 / 55 deg | 0/5 (z 5.0-5.6) | not reached | not reached | not reached |
| M175124932LC (opposed sun) | 178 / 42 deg | 0/5 (z 4.3-5.3) | not reached | not reached | not reached |
| M1417360906LC (NAC sun near overhead) | 15 / 75 deg | 3/5 | 0/3 locked | **2/3 locked** | 0/3 locked |
| M109080308LC (NAC sun overhead) | 72 / 80 deg | 4/5 | 0/4 locked | **3/4 locked** | 0/4 locked |

Verdicts over all 5 windows (frozen rule): control solved for all three; everything else
unsolved except minima-loftr on M109080308LC (3/5, **degraded**).

**Every locked window is location-consistent**: within 10 m of its product's median
offset (the rule allows 150 m). The implied OHRC system-geolocation error is ~2.06-2.21 km
north in every product, matching the ~2.23 km measured independently against SELENE TC;
the east component differs per NAC product (LRO product-level bias, as found in PR #18).

## What it shows

1. **The credibility floor holds.** Same-sun OHRC -> NAC registers 5/5 for all three
   matchers. Known-shift error: eloftr 0.011-0.043 px on 4 windows, 0.68 on the fifth
   (the weakest lock, z 25.5); SIFT 0.001-0.066 px. Inlier RMSE of the fine stage (an
   internal residual, the kind of number arXiv:2509.04775 reports, NOT accuracy):
   eloftr 0.49-1.44 NAC px against the paper's 0.57-0.92 px for SuperGlue.
2. **Under large illumination differences the COARSE LOCK is the first thing to break.**
   With a 55 deg incidence difference or an opposed sun, the MIND coarse search never
   locked (z ~5), so no fine matcher was even tested there. Prediction 2 (opposed sun:
   eloftr fails, minima-loftr succeeds) is therefore **untested, not confirmed**.
3. **Where the coarse lock did succeed at 75-80 deg incidence difference, only
   minima-loftr registers**: 5 of 7 locked windows, against 0 of 7 for eloftr and for
   SIFT. This is the first real-data evidence that the default matcher fails beyond
   ~60 deg while minima-loftr does not -- the same direction as the synthetic sweep.
   eloftr and SIFT fail by the known-shift gate or the tier rule, i.e. they are
   REJECTED, not confidently wrong.
4. **Tiers are LOW even on the clean control** (~3,180 inliers, ratio 0.99, coverage
   1.0): the NAC pixel size has one source (LROC), so the scale check is `unverified`,
   which caps the tier. A second agreeing source (ODE `Map_resolution`, already
   downloaded) would lift it -- the open audit item on NAC pixel sizes.

## What it means for the pipeline

- The default route cannot handle large illumination differences on OHRC <-> NAC, and
  the first fix is the **coarse lock**, not the matcher: a lock that survives a 55 deg /
  opposed-sun change is needed before any matcher choice matters.
- After a lock, **minima-loftr is the matcher that works beyond ~60 deg** on real data.
  It is licence-clean but not shippable yet; promoting it (as a fallback when the
  default is rejected) is now supported by real evidence, and is a separate decision.

## Limits

One OHRC scene, one site, 5 windows per product: 1-2 windows' difference is not
meaningful (runs 2 and 3, with slightly different windows, gave minima-loftr 3/5 -> 2/5
on M1417360906LC and 5/5 -> 3/5 on M109080308LC). The known-shift gate measures the
precision of the lock on this pair, not absolute accuracy.

## Runs not counted (kept for the record)

- Run 1 (frozen placement): only the control had windows; all three matchers 5/5.
- Run 2 (amendment 1): opposed-sun product still had no windows (padding bug);
  minima-loftr 3/5 and 5/5 on the two overhead-sun products, eloftr and SIFT 0.
