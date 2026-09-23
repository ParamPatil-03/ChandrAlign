# TMC-2 -> LRO NAC registration, and whether solar AZIMUTH difference breaks it

**Status: FROZEN. Committed before `scripts/register_tmc2_nac.py` exists or any
window has been run.** `git log` on this file must precede the script and
`reports/tmc2_nac_registration.json`. No threshold below may change once any
result exists. PLAN.md is not a source of criteria here.

---

## 1. Question

Research doc section 27 ranks TMC-2 <-> LRO NAC as Opportunity #2 (best known
competitor: 2.8% inlier ratio). Two questions, in order:

1. Does TMC-2 -> NAC register at all with this repository's cascade step?
2. At a matched ~10x scale gap, does a ~180 deg solar AZIMUTH difference make it
   fail where a ~2-4 deg difference does not?

## 2. Facts checked before writing this (not taken from the brief)

- `routing.choose("TMC2", "NAC")` returns `route="cascade", model_name=None`,
  regime `extreme_scale`. `cascade.plan()` returns ONE direct step for every
  product here: the NAC short side spans 402-1276 TMC-2 pixels, against
  `cascade.min_footprint_px = 112`.
- **`cascade.min_z` is not in `configs/default.yaml`.** It is the code default in
  `cascade.register_step_dense`, `config.get("cascade.min_z", 10.0)`, i.e.
  **10.0**. The same z >= 10 was the success bar when the 112 px footprint was
  measured (configs/default.yaml comment). The earlier NAC<->NAC attempt used a
  6.0 bar from a different script; it does not apply here.
- **LROC field frames, verified on all 11 products** (`data/pairs/tmc2_nac_lroc_meta.json`):
  "North azimuth" and "Sub solar azimuth" are both image-frame angles, and
  `(north_az - sub_solar_az) mod 360` reproduces the GROUND sun azimuth
  computed independently by spherical trigonometry from the sub-solar point, to
  1-21 deg on the 9 usable products. The opposite sign is 150-175 deg off.
  Every NAC image is therefore MIRRORED relative to a map (east lies 90 deg
  counter-clockwise of north), and the page's corner coordinates agree on every
  product. The azimuth fields are only good to ~+-20 deg, too coarse to orient
  a template, so **orientation comes from the corner coordinates, not from
  North azimuth.**
- Orientation varies per product: M117338434LC, M122054682LC and M131494509LC
  have north DOWN in the array. No fixed assumption could be used.
  `scripts/nac_illumination_validation.py::latlon_to_pixel` assumed every NAC is
  a descending pass (north up by line); 6 of the 9 usable products are ascending.
- **Overlap verified on the ground:** the four primary products' LROC corners
  project inside the TMC-2 strip with >= 624 columns (~3 km) to spare.
  M102014464RC crosses the strip edge (col -38); M1417360906LC is excluded anyway.

## 3. Groups (from `data/pairs/tmc2_nac_candidates.json`, recomputed and confirmed)

TMC-2 sun at the overlap: azimuth ~93.3 deg, incidence ~43.8 deg.

| group | product | d_az | NAC incidence | NAC px (m) | scale |
|---|---|---|---|---|---|
| **same sun (control)** | M111443315RC | 2.1 | 26.2 | 0.518 | 9.6x |
| | M131494509LC | 3.5 | 79.2 | 0.476 | 10.5x |
| **opposed sun (test)** | M117338434LC | 176.4 | 83.6 | 0.483 | 10.3x |
| | M122054682LC | 177.4 | 27.6 | 0.490 | 10.2x |
| secondary, same sun | M175124932LC | 1.3 | 40.9 | 0.400 | 12.5x |
| secondary, opposed, ~4x | M106719774LC, M104362199LC, M102000149LC, M102014464RC | 172.5-176.9 | 27-82 | 1.18-1.27 | ~4x |

Incidence is balanced across the primary groups: each has one moderate-sun
(26-28 deg) and one low-sun (79-84 deg) product. Excluded: M109080308LC,
M1417360906LC (NAC incidence < 10 deg, azimuth ill-defined).

## 4. Method, fixed

- **Step:** `cascade.register_step_dense`, unchanged. The NAC (finer) is the
  step's `src_img`; the TMC-2 region is `ref_img`. The step maps NAC px -> TMC-2 px;
  its inverse is reported as TMC-2 -> NAC, the direction the PS asks for.
- **Block factor:** `round(4.92 / NAC across pixel)`, where 4.92 m is TMC-2's
  verified across-track pixel (`estimate.scale.pixel_scale`); the residual scale is
  carried by the prior.
- **Prior (orientation only; position is searched):** NAC line and sample axes in
  local east/north metres from the LROC corners (line axis from the 26 km long
  side, ~1 deg; sample axis perpendicular, handedness from the corners), NAC pixel
  sizes from LROC's "Scaled pixel width/height", mapped into TMC-2 pixels by the
  Jacobian of TMC-2's **SYSTEM** corner model at the window centre. ISRO's
  refined grid is never used.
- **Windows:** per product, 5 windows of 5064 lines x 5064 samples (the full
  width), centred at line fractions 1/6, 2/6, 3/6, 4/6, 5/6 (the fractions
  `scripts/nac_illumination_validation.py` already committed).
- **Search region:** the window's predicted TMC-2 footprint from the SYSTEM model,
  plus a **7 km** margin on every side (`--margin-km` default of
  `scripts/register_tmc2_tc.py`; the measured system error is 4.5-4.9 km),
  clipped to the strip.
- **Exclusion, identical for every group:** a window is excluded (reported, not
  counted) if under 90% of its NAC pixels are valid (the TMC-2 -> TC script's
  rule), or if its predicted TMC-2 footprint leaves the strip's columns.

## 5. Success per window (every condition required)

1. the step returns a result, which requires MIND z >= 10.0 (`cascade.min_z`);
2. **known-shift test:** the TMC-2 search region's content is moved by a known
   (3, 4) TMC-2 px (`gates.perturbation_shift_xy`) and the step re-run; the recovered
   move of the NAC window centre must be within **1.5 TMC-2 px**
   (`gates.perturbation_tolerance_px`) of (3, 4).
   The shift goes into the TMC-2 image, not the NAC. A 3-4 px NAC move is 0.3-0.4
   TMC-2 px, below the resolution the match is made at, so it would test nothing.

There is no ground truth. None is invented.

### AMENDMENT 1 (2026-09-23, before the full run): section 5.2 as frozen is vacuous

Found on a smoke run of ONE product (M111443315RC, same-sun control, 5 windows),
before any other product was run. Every window recovered the (3, 4) shift as
exactly (3.000, 4.000), error 0.000 px. That is not precision. An INTEGER shift of
the search image only re-indexes its pixels, so the MIND correlation map
translates by exactly (3, 4) and the sub-pixel estimators see the identical
patch. The test cannot fail except at region borders, so it checks nothing.

**Replacement, used for every group:** the NAC window's content is moved by
**(37, 43) NAC px** (reflect-padded, `control_gates._shift_content`). At the ~10x
block factor that is a FRACTIONAL move of about 3.7 x 4.3 TMC-2 px, so the
block-averaged template really changes and the sub-pixel stage must recover it.
With T0 the step's NAC -> TMC-2 transform, the expected move of the window centre
is `-A0 (37, 43)`, where A0 is T0's linear part. The error is the distance between
the recovered and expected moves, in TMC-2 px. Tolerance unchanged: **1.5 TMC-2 px**
(`gates.perturbation_tolerance_px`). The vacuous integer test is still run and
reported, labelled vacuous, and does not count. The smoke windows are re-run
under the amended rule with everything else.

## 6. Controls

- **Null:** per window, the NAC window replaced by flat grey, and by Gaussian
  noise with its own mean and standard deviation. A returned result is a LOCK
  and means the method invents structure. If nulls lock on more than 10% of
  windows, no finding from this experiment is reported.
- **Orientation:** per primary window, the prior with the NAC LINE axis reversed.
  If it locks on half or more of the windows where the true prior succeeded,
  the orientation is not being discriminated. That is reported, and the
  orientation claims in section 2 are then unverified by matching.
- **Independent geolocation (descriptive):** a lock implies TMC-2's system
  offset in metres, taken from LROC's NAC geolocation. The committed TMC-2 -> SELENE
  TC run measured that same offset from SELENE (`reports/tmc2_tc_registration.json`:
  east 622-703 m, north -4642 to -4827 m over TMC-2 rows 4687-21500, varying by
  <= 190 m). Each lock's offset is compared with the TC value linearly
  interpolated at its TMC-2 row. A difference over **500 m** (100 TMC-2 px, 2.5x the
  whole along-strip variation) is flagged a probable false lock, and a secondary
  tally counts those as failures.

## 7. Verdicts (`docs/unsolved_band_protocol.md` section 7)

Per group: `solved` >= 90% of counted windows, `degraded` >= 60%, else `unsolved`.

**STOP RULE.** If the SAME-SUN primary control succeeds on at most half of its
counted windows, that is a method problem: the report says so and draws **no**
illumination conclusion.

## 8. Pre-registered prediction, written before any run

The synthetic model (`configs/regimes.yaml` representation_bands) finds MIND
near-invariant at an opposed sun: description correlation +0.975 at 180 deg
against +1.000 at 0 deg. It therefore **predicts opposed-sun success ~=
same-sun success.**

| outcome | meaning |
|---|---|
| same-sun <= 50% | method problem; no illumination conclusion (section 7 stop) |
| same-sun passes, opposed-sun in the SAME verdict band | prediction RIGHT: a 180 deg azimuth difference does not break MIND-based TMC-2 -> NAC at ~10x |
| same-sun passes, opposed-sun in a LOWER band | prediction WRONG: an opposed sun breaks it, despite the synthetic invariance |
| opposed-sun in a HIGHER band | no illumination penalty; the difference is noise or a product effect |

With 2 products and 10 windows per group, product effects cannot be separated
from illumination. A one-band difference is reported as suggestive, not
established.

## 9. What this cannot test

- ~90 deg azimuth difference, the synthetic model's worst case: no overlapping
  NAC gives one (equatorial targets from a polar orbit are lit from the east or
  the west).
- Accuracy against truth: none exists. The known-shift test measures sensitivity
  and precision, not correctness. A consistently wrong lock passes it, which is
  why section 6's geolocation cross-check exists.
