# TMC-2 -> LRO NAC: results

Protocol: `docs/tmc2_nac_protocol.md` (frozen `926a423`; amendment 1 `df9355d`,
made before any product other than one same-sun control had run).
Data: `reports/tmc2_nac_registration.json` (`"source": "measured"`), from
`scripts/register_tmc2_nac.py`. 45 windows, 9 NAC products, none excluded.

## 1. Does TMC-2 -> NAC register at all? Yes: 44 of 45 windows lock, 43 pass every test

| group | products | scale | counted | success | verdict | median MIND z |
|---|---|---|---|---|---|---|
| same sun, d_az 2-4 deg (control) | M111443315RC, M131494509LC | ~10x | 10 | **9** | solved | 30.2 |
| opposed sun, d_az ~177 deg (test) | M117338434LC, M122054682LC | ~10x | 10 | **10** | solved | 31.6 |
| same sun, secondary | M175124932LC | 12.5x | 5 | 5 | solved | 32.7 |
| opposed sun, secondary | 4 products | ~4x | 20 | 19 | solved | 49.4 |

Success = the step returns a result (MIND z >= 10) AND a fractional known shift
(37, 43 NAC px, ~3.7 x 4.3 TMC-2 px) is recovered within 1.5 TMC-2 px. Recovered-
shift errors are 0.016-1.14 px on successful windows (median ~0.15); the two failures are
M131494509LC at 4/6 (known shift off by 8.96 px, lock z 20.2) and M102014464RC
at 3/6 (only 1 of 11 sub-pixel estimates succeeded, z 20.8).

The best known competitor attempt at this pairing reached a 2.8% inlier ratio
(research doc section 27). This is a dense MIND step, not a feature matcher, so it
has no inlier ratio to compare. The comparable claim is **44 of 45 windows locked
unambiguously** (43 also passing the known-shift test), at z 20-70 against a bar of 10.

## 2. Does azimuth difference break it at matched ~10x scale? No

Same sun 9/10, opposed sun 10/10, both `solved`; median z 30.2 vs 31.6.
Incidence is balanced across the groups (26/79 deg vs 28/84 deg). The one
same-sun failure is on the low-sun (79 deg) control product, so if anything the
opposed-sun group did marginally better. With 2 products per group that
difference is not meaningful. What is supported: **a ~177 deg sun-azimuth
difference produces no measurable penalty for MIND-based TMC-2 -> NAC at ~10x.**

## 3. Was the pre-registered prediction right? Yes

The synthetic model predicted opposed ~= same-sun (MIND description correlation
+0.975 at 180 deg). Outcome: both `solved`, the same band, which is the row the
protocol pre-registered as "prediction RIGHT". Unlike the four synthetic or
label-derived claims listed in `docs/part2_evidence_summary.md` section 5, this one
survived real data.

## 4. Controls

| control | result |
|---|---|
| null (flat grey, noise as the NAC), 90 runs | **0 locks**, z 4.6-6.9 |
| orientation (NAC line axis reversed), 20 primary windows | **0 locks**, z 5.2-6.7: the prior's orientation is being discriminated, not ignored |
| stop rule (same-sun <= 50%) | not triggered (90%) |
| exclusions | none, identical rule for all groups |
| frozen known-shift test (integer move of the TMC-2 region) | error exactly 0.000 on every window: **vacuous**, as amendment 1 found; reported, not counted |

### The geolocation cross-check flags many locks, and the flags are about the reference, not the locks

As frozen, a lock is flagged a "probable false lock" when the TMC-2 system offset
it implies (from LROC's NAC corners) differs from the offset measured against
SELENE TC by over 500 m. Secondary tally: same sun 9/9, opposed sun **5/10**,
12.5x 5/5, ~4x **2/19**.

Evidence that the flags come from **LROC's per-product geolocation error**, not
from false locks (post hoc, stated as such):

1. **Within a product the implied offset is smooth**: it changes by at most
   ~210 m across 5 windows spanning ~20 km, e.g. M122054682LC east
   1153 -> 1228 m. A false lock lands at an unrelated place per window.
2. **Products that overlap the same TMC-2 rows disagree with EACH OTHER by more
   than the flag.** M117338434LC and M122054682LC (both opposed, rows
   ~10.4k-14.6k) imply east offsets of ~550 and ~1190 m. M102000149LC and
   M102014464RC (rows ~9.5k-17.8k) imply ~1250 and ~50 m. TMC-2 has ONE system
   offset per row, so at least one reference in each pair is 600-1200 m off
   whatever the locks do.
3. Every flagged lock passes the fractional known-shift test, and no null or
   flipped control locks anywhere.

The flagged products are 2009-2010 LROC early-mission images, whose corner
coordinates are also rounded to 0.01 deg (~300 m). The frozen 500 m bar was set
from the along-strip variation of the TC-measured offset (<= 190 m) and did
**not** account for the reference's own error. **That was a protocol error of
mine.** The cross-check can detect a gross false lock (kilometres), not a
few-hundred-metre one, on these products.

## 5. What this does not test

- **~90 deg azimuth difference**, the synthetic model's worst case. No
  overlapping NAC provides it; this site is lit from the east or the west.
- **Accuracy against truth.** None exists. The known-shift test measures
  precision and sensitivity; a consistently wrong lock would pass it. The
  geolocation check, the only external reference, is limited to about +-1 km
  here (section 4).
- **Other matchers.** Only `register_step_dense` (MIND) was run. `register_step`
  with a learned matcher was optional and was not run.
- Two products per primary group, at one site (Apollo 11): product effects cannot
  be separated from illumination.

## 6. Errors found in the brief's facts

1. **`cascade.min_z` is not in `configs/default.yaml`.** It is the code default in
   `cascade.register_step_dense`, 10.0; the earlier NAC<->NAC attempt's 6.0 bar
   came from another script.
2. **The four primary products are not in `data/pairs/nac_ode_geometry.json`.**
   Their geometry was fetched from the LROC pages (`data/pairs/tmc2_nac_lroc_meta.json`).
3. **LROC's "North azimuth" is an image-frame angle, and the images are mirrored.**
   `(north_az - sub_solar_az) mod 360` reproduces the ground sun azimuth on
   every product, but only to 1-21 deg, too coarse to orient a template. The
   corner coordinates were used instead.
4. **Orientation differs by product.** Three of the four primary NACs
   (M117338434LC, M122054682LC, M131494509LC) have north DOWN in the array.
   `scripts/nac_illumination_validation.py::latlon_to_pixel` assumes latitude
   decreases with line number (north up) for every NAC, which is wrong for those
   three. Its docstring's reason ("a descending pass") is wrong too: orientation
   does not follow the pass direction (M131494509LC is ascending and north-down;
   7 of the 9 usable products are ascending, 6 of them north-up). This may
   contribute to the earlier "~7000 lines off" prior; it was not tested.
   **Correction to my own frozen protocol:** section 2 says "6 of the 9 usable
   products are ascending", tying orientation to pass direction. Both the count
   (7) and the framing are wrong. Neither affected the measurement, which took
   orientation from each product's corners.
5. **TMC-2 pixel:** the repository's verified value is 4.92 x 5.037 m
   (`estimate.scale.pixel_scale`), not ~4.99 m.
6. **`main` (c018729) was red on arrival.** `tests/test_regime.py` hard-coded
   `aliked-lightglue` as the default after PR #13 made it `eloftr`. It is fixed
   here by asserting against the configured default; the test still fails if the
   selector ignores the default.
