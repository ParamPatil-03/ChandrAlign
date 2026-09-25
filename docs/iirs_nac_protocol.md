# Protocol: IIRS -> LRO NAC on real data (frozen before any measurement)

Research doc: IIRS <-> LRO NAC is open (no paper; "~40-160x scale gap"). Routing sends it to
the cascade (`routing.choose("IIRS", "NAC")`: 80 m vs 0.5 m, 160x, past the 4x limit).
The information is bounded by IIRS: a NAC strip (~5 km wide) is ~60 IIRS pixels across,
so the answer can only be located to about an IIRS pixel (80 m).

## Data

- IIRS `ch2_iir_nci_20240523T1600301891` (13,101 x 250 px), its PREP-06 composite, SYSTEM
  corners.
- NAC CDRs already held whose footprint lies >= 50% inside the IIRS strip (after the IIRS
  offset below): **M1415013176LC** (100%, incidence 36 deg, 0.76 m), **M172765160RC** (100%,
  13.5 deg, 0.49 m), **M1417360906LC** (60%, 7.7 deg, 0.79 m). M109080308LC (28%) is excluded
  by that rule.
- NAC ground geometry: LROC's four corners (`view_lroc` metadata pages; fetched 2026-09-25 for
  the two not already in `data/pairs/tmc2_nac_lroc_meta.json`), bilinear in (line, sample)
  with each product's OWN line/sample count. **The corners are rounded to 0.01 deg (~300 m)**,
  so NAC geolocation from them carries up to ~150-200 m of error per point.

## Prior (the geodetic bridge, MATCH-11 style; declared)

IIRS's system geolocation is ~12.8 km off. The prior corrects it by our own measured IIRS ->
WAC offset (`reports/iirs_wac_mosaic.json`, xoftr, 5/5 HIGH), linearly interpolated in IIRS
line (east 1195-1478 m, north 12766-12876 m over lines 2112-4160). Position is still searched.

## Method (`scripts/register_iirs_nac.py`)

- **Windows:** per NAC, the NAC lines whose ground lies inside the (corrected) IIRS strip,
  split into 3 equal windows along the line axis. A window = that NAC crop (all samples).
- **Coarse lock:** `cascade.register_step_dense` (MIND): the NAC crop block-averaged to the
  IIRS pixel size (the finer image is the template) searched over the IIRS composite lines
  that cover it +- 60 lines (~5 km), full width. Accepted if z >= 10 (`cascade.min_z`).
- **Fine stage:** the NAC crop block-averaged and warped (anti-aliased) onto the IIRS grid of
  the covered region; the IIRS region is the source, the warped NAC the reference. Matchers:
  the routed matcher of the 80-100 m cross-modal regime, **xoftr tiled 640** (this is what the
  pair becomes once NAC is at IIRS scale: IIRS vs a broadband image, the IIRS <-> WAC regime),
  plus `minima-loftr` and `sift` on identical windows. `pipeline.fine_stage` with the
  configured defaults; all five control gates; tier (NAC pixel size is unverified, so the
  tier is expected to be capped as for OHRC -> NAC).

## Success (per window, per matcher)

Coarse lock accepted; fine model returned; all five gates pass (incl. the known (3, 4) px
shift within 1.5 px); tier >= LOW; and the implied IIRS offset (NAC ground of the fine frame
centre minus IIRS system ground there) within **350 m** of the IIRS -> WAC offset at that line.
350 m = one IIRS pixel diagonal (113 m) + the NAC corner rounding (~212 m) + slack. It is a
false-lock guard between two independent registrations, not an accuracy measure: a false
lock at this scale is off by many IIRS pixels.

Verdict per NAC: solved 3/3, degraded 2/3, else unsolved. Overall: solved on >= 2 of 3 NACs.

## Predictions

1. The coarse lock succeeds on at least 2 NACs (the NAC template, ~60 x 100+ px at IIRS scale,
   is smaller than IIRS <-> WAC's but the IIRS <-> WAC locks had z 50-64).
2. xoftr >= sift on success count. No prediction for minima-loftr.

## Limits

One IIRS scene; three NACs; three windows each. The known-shift gate measures precision of the
lock, not absolute accuracy. NAC corners at 0.01 deg limit the absolute check to ~300 m.

## Run 1 result (2026-09-25): unsolved

`reports/iirs_nac_registration.json`. 0/3 NACs solved; 1 success in 27 window-matcher runs
(sift, M1415013176LC window 2: LOW, known shift 0.068 px, 149 m from the IIRS -> WAC offset).

- M1415013176LC (5 km wide = ~38 IIRS px): the MIND coarse lock succeeds on 3/3 windows
  (z 22.7-23.9) and agrees with the IIRS -> WAC prediction to 1.3-2.5 IIRS px. The keypoint fine
  stage then has a 38 x 134 px frame: 6-16 matches, which fail the perturbation gate or the tier.
- M172765160RC (~2 km wide = ~22 IIRS px): lock on 1/3 (z 7.5, 7.6, 11.2); the locked one's fine
  frame is < 32 px wide (skipped).
- M1417360906LC: its centre line lies at IIRS columns 0-7 (the strip's edge): z 3.8-4.7, no lock.

Prediction 1 (lock on >= 2 NACs) failed: 1 NAC. Prediction 2 (xoftr >= sift): failed (0 vs 1).

## Amendment 1 (frozen BEFORE running it): the dense lock as the registration

At IIRS scale a NAC is too small for keypoints (the cascade's own finding: "a 128x128
block-averaged patch held 6 SIFT keypoints"); the dense step is the method built for it.
Same 9 windows, same prior, same lock (z >= 10), same 350 m bound. Per window, the dense
step's transform is the answer, and it must pass its own checks (the control gates' logic,
applied to the dense step):
- **known shift:** the IIRS region shifted by exactly (3, 4) px, the step re-run with the same
  prior; the locked position must move by (3, 4) within **0.5 IIRS px**;
- **null, constant:** the IIRS region replaced by its mean: must NOT lock (z < 10 or no answer);
- **null, noise:** the IIRS region replaced by Gaussian noise (seed 0): must NOT lock.
Implied offset: NAC crop centre -> IIRS px by the dense transform; NAC ground there minus IIRS
system ground, vs the IIRS -> WAC offset at that line (<= 350 m). No tier (quality.assess grades
keypoint sets); the step's own precision (robust spread of its sub-pixel estimates) is reported.
Verdicts as before (per NAC 3/3 solved; overall solved on >= 2 of 3 NACs). The two narrow/edge
NACs are NOT excluded; they are expected to fail again for lack of overlap.

## Amendment 1 result (2026-09-25): solved on 1 NAC of 3 -> overall NOT solved by the frozen rule

`reports/iirs_nac_dense.json`.

| NAC | width at IIRS scale | locks (z) | checks | vs IIRS -> WAC | verdict |
|---|---|---|---|---|---|
| M1415013176LC | ~38 px (5 km) | 3/3 (22.7-23.9) | 3/3 pass | 105, 145, 196 m | **solved** |
| M172765160RC | ~22 px (2 km) | 1/3 (7.5, 7.6, 11.2) | 1/1 pass | 115 m | unsolved |
| M1417360906LC | at the strip edge | 0/3 (3.8-4.7) | - | - | unsolved |

Every window that locked passed all three checks and the 350 m bound; every window that failed,
failed to lock (z < 10), i.e. refused rather than guessed. Dense precision (robust spread of
the step's sub-pixel estimates) 0.09-0.18 IIRS px. The overall rule (solved on >= 2 NACs) is not
met: 1 of 3. Second failure of this protocol; work stops here and is reported.

**A weakness in the frozen known-shift check, found in the result:** its error was exactly
0.000 on every window. The dense search is exactly translation-equivariant, so an INTEGER shift
of the reference reproduces the same answer to the bit: the check proves the lock is not
anchored to the prior, not that it is precise. Diagnostic run afterwards (NOT part of the
decision): fractional shifts (3.5, 4.25), (-2.25, 1.75), (0.5, -3.5) by cubic interpolation are
recovered to **0.02-0.11 IIRS px** on all 4 locked windows (12/12). That is precision on this
pair, not absolute accuracy. A future protocol for dense steps should use fractional shifts.

What limits it: overlap. At ~97 m/px a NAC strip is 22-38 IIRS px wide; the 5 km one locks
every time, the 2 km one mostly does not, and one lying on the IIRS strip's edge never does.
Absolute position agrees with the independent IIRS -> WAC registration to 105-196 m (< 2 IIRS px),
inside the ~300 m that the NAC corners' 0.01 deg rounding allows.
