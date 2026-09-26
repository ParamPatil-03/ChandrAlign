# IIRS -> WAC mosaic on fresh scenes: results (2026-09-26)

Protocol: `docs/iirs_fresh_scenes_protocol.md` (frozen before results; amendments 1-2 made before any
fresh scene was registered). Data: `reports/iirs_fresh_scenes.json` (measured).

## Verdict

**0 of 4 fresh scenes solved** (2 degraded, 2 unsolved). The pre-registered claim therefore stays
"IIRS -> WAC is validated on ONE scene". The control (the evidence scene) reproduced 5/5 solved in the
same run.

| scene | region | windows solved | verdict | tiers | what stopped the rest |
|---|---|---|---|---|---|
| 20240523 (control) | nearside, 0-8 N, 23 E | 5/5 | solved | 5 HIGH | -- |
| 20260201 | Imbrium mare, 37-43 N | 0/5 | unsolved | 2 HIGH, 3 REJECTED | `null_random_noise` gate on 3 windows, leaving 2 < 3 gate-passing windows for the median |
| 20240605 | far-side south (SPA), 39-45 S | 3/5 | degraded | 4 LOW, 1 REJECTED | 1 window 532 m off the median; 1 rejected (known-shift 1.9 px) |
| 20231227 | 60-66 N, no DEM (SLDEM ends at 60) | 0/5 | unsolved | 1 LOW, 4 REJECTED | no coarse lock on 4 windows |
| 20260217 | far-side equator, 5 S-1 N | 4/5 | degraded | 4 LOW, 1 REJECTED | no coarse lock on 1 window |

## What the failures are, and are not

- **No confident wrong answer.** Every window that passed the gates lies within 240 m of its scene's
  median except one (20240605, line 1024: 532 m, on the along-track drift amendment 1 describes).
  The failures are refusals, not wrong registrations.
- **Imbrium: registrations agree but a gate refuses them.** All 5 windows registered with implied
  offsets within ~300 m of each other and known-shift error 0.08-0.29 px; 3 fail only
  `null_random_noise` (the matcher also "registers" random noise against those WAC windows, so a
  match there cannot be told from chance). Reproduced exactly on a re-run. This is the gate working
  as designed on low-texture mare; it is not loosened after the fact.
- **High latitude: the coarse lock fails.** 4 of 5 windows at 60-66 N get no MIND coarse lock, so
  matching never starts. The terrain filter was off there (no DEM), but it runs after the coarse
  lock, so the missing DEM does not explain this.
- **Fresh scenes register at LOW, not HIGH.** Every registered fresh window is LOW, except the two
  Imbrium windows that passed their gates (HIGH).

## Next (not done; each needs its own frozen protocol)

1. Coarse lock at high latitude and on the equatorial window that failed.
2. Why fresh scenes tier LOW (inlier count, coverage or ratio) when the evidence scene tiers HIGH.
3. `null_random_noise` on mare: whether a texture-aware window placement avoids windows where the
   null test cannot separate signal from chance (a placement rule, not a looser gate).

## Follow-up (2026-09-26, exploratory, no new registrations)

Checked against prior work (M3 control to the WAC mosaic and terrain model, Gaddis et al. LPSC 2016;
IIRS -> WAC on 2 scenes with fit-residual RMSE only, arXiv 2509.04775). Two cheap fixes were tried
on the measured results above; neither is adopted:

1. **Along-track drift model instead of the scene median** (M3's approach to smooth pointing drift):
   leave-one-out linear fit of the implied offset against IIRS line, 240 m rule. Control 5/5,
   20240605 2/5 (was 3/5), 20260217 4/5, others unchanged. Offsets on 20240605 jump rather than
   drift, so the model does not help. Rule unchanged.
2. **No coarse lock at 60-66 N:** the MIND coarse-lock z is 6.7-8.5 there (threshold 10; fresh
   scenes elsewhere 12-30; evidence 40-57). The true location is inside the clip (the locked window
   sits 2.9 km E, 8.8 km N of its prior), so this is weak evidence, not a search-range bug. Lowering
   the threshold without a null study would admit wrong locks. A real fix would be an
   illumination-robust coarse step (e.g. a learned matcher, or a DEM-shaded reference at the IIRS sun
   angle) with its own protocol and new scenes -- not done.
