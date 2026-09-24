# Fixing registration under large sun differences: results

Protocol frozen before measuring: `docs/illumination_fix_protocol.md`. Data: OHRC
`ch2_ohr_ncp_20240330T0035085365` against five LRO NAC products, 25 windows.

## Q1 - the coarse lock: MIND at 2 m replaces MIND at 4 m

Accepted = z >= 10 AND north offset within 300 m of 2,150 m AND within 150 m of the
product's median. Over the four non-control products (20 windows):

| candidate | control | accepted (hard) | false locks |
|---|---|---|---|
| A MIND 4 m (was the default) | 5/5 | 7/20 | **1** |
| B phase congruency 4 m | 4/5 | 0/20 | 2 |
| **C MIND 2 m** | **5/5** | **9/20** | **0** |
| D phase congruency 2 m | 4/5 | 2/20 | 3 |
| A then B / C / D on failure | 5/5 | 7 / 8 / 7 | 2 / 1 / 1 |

**C adopted** (`scripts/register_ohrc_nac.py`: `COARSE_M = 2.0`). The old 4 m lock had
produced one false lock that the consistency check caught. Phase congruency is WORSE
than MIND on this real data (the synthetic bands had suggested otherwise).
**Unsolved by every candidate:** M106719774LC (55 deg incidence difference) and
M175124932LC (opposed sun), z 4.3-5.7 on every window. The opposed-sun prediction (Q4)
therefore remains **untested**.

## Q2 - the matcher after the lock: minima-loftr as the fallback

With the 2 m lock, same success rules as `docs/ohrc_nac_protocol.md`:

| NAC | eloftr | minima-loftr | xoftr | sift |
|---|---|---|---|---|
| same-sun control | 5/5 | 5/5 | 5/5 | 5/5 |
| 80 deg incidence (M109080308LC) | 0/5 | **5/5** | 0/5 | 0/5 |
| 75 deg incidence (M1417360906LC), 4 locked | 0 | 0 | 0 | 0 |
| 55 deg / opposed | no lock | | | |

- **xoftr adds nothing** beyond the control (the synthetic sweep predicted it fails 75-120 deg).
- **minima-loftr adds 5 windows** the default cannot register. Fallback list: `[minima-loftr]`.
- On M1417360906LC minima-loftr is borderline: known-shift errors 2.2-3.1 px (limit 1.5)
  and once a noise-null gate failure, where at 4 m it had passed 2 of 3 with errors
  0.48 and 1.44. Recorded as **unsolved, near the edge**.

**Implemented:** `configs/regimes.yaml` `fallback_matchers: [minima-loftr]`;
`routing.MatcherChoice.fallbacks` and `.candidates()`; `xoftr` and `minima-loftr` added
to `shippable_matchers` (both Apache-2.0, `reports/licence_audit.json`). The fallback runs
ONLY when the default is rejected or fails a gate, so no accepted result can change.
**Checked on real data:** `--matchers routed` on the 80 deg product: eloftr rejected ->
minima-loftr -> **5/5 solved**.

## Q3 - the LOW grade: stays LOW, correctly

The planned independent source failed on inspection: PR #18's coarse step searches
POSITION only and takes scale and rotation from its prior, which was built from LROC's
pixel sizes -- so its transforms reproduce LROC to 0.5% by construction. ODE's
`Map_resolution` is the same LROC/SPICE computation. There is therefore **no independent
NAC pixel size yet**, the scale check is honestly `unverified`, and the tier cap at LOW
is the system correctly saying "matches are good, scale not independently confirmed".
The real fix is a TMC-2 -> NAC run whose fine stage fits scale freely; not done here.

## Net effect on OHRC <-> LRO NAC (the shipped, routed behaviour)

| NAC | sun difference | before (4 m lock, eloftr) | after (2 m lock, routed) |
|---|---|---|---|
| control | ~1 deg | 5/5 | 5/5 |
| M109080308LC | 80 deg incidence, 72 deg azimuth | 0/5 | **5/5** |
| M1417360906LC | 75 deg incidence | 0/5 | 0/5 (borderline) |
| M106719774LC | 55 deg incidence | 0/5 (no lock) | 0/5 (no lock) |
| M175124932LC | 178 deg azimuth | 0/5 (no lock) | 0/5 (no lock) |

## Still open

- A coarse lock that survives 55 deg incidence / opposed sun (every descriptor tried fails).
- Fallback in `scripts/register_tmc2_tc.py` (not needed so far: eloftr passes every
  TMC-2 window) and in any future entry point.
- An independent NAC pixel-size measurement (Q3).

Candidate selection and evaluation used the same 25 windows, so the adopted settings'
rates are optimistic.

## Q5-Q7 (research-driven follow-up, same frozen-protocol discipline)

Research consulted: crater-neighbourhood matching for multi-illumination lunar orbiter
images (Remote Sensing 17(13):2302, 2025); rendered-DEM / LIMA matching used in lunar
terrain-relative navigation; photoclinometry-assisted matching (ISPRS J. 159, 2020);
Geo-LoFTR (arXiv:2502.09795). The rendering, photoclinometry and Geo-LoFTR routes need a
DEM at the image's resolution (ours: SLDEM, 59 m) or training; not feasible here. The
research doc's own OHRC geodetic bridge (opportunity #9) was feasible and tested first.

| problem | finding | status |
|---|---|---|
| M106719774LC, 55 deg incidence, "no lock" | with the geodetic bridge (OHRC vs TC + NAC vs TC, both measured independently; validated on the control to 128 m) eloftr 3/5, minima-loftr 3/5, **routed 4/5**, all within 6 m of each other | **the failure was the +-4 km search, not the sun -> fixed by the bridge** |
| M175124932LC, opposed sun 178 deg | eloftr fails outright; **minima-loftr locates all 5 windows consistently** (within ~50 m; ~130 m from the bridge) but its known-shift error is 1.5-2.3 px (gate 1.5); dense MIND refinement cannot help (0-1 of 10 estimates succeed) | **located, not precise enough: unsolved** |
| M1417360906LC, 75 deg | locks, minima-loftr errors 2.2-3.1 px; dense refinement mixed | **unsolved (precision)** |
| dense_refine stage | same-sun precision 10x better on the control; on unseen TMC-2 -> TC median -72% but one window worse | **built, off by the frozen rule; re-test with TPS** |

What would solve the remaining two (next research step): a high-resolution DEM of the
site (e.g. an LROC NAC DTM near Apollo 11, ~1-2 m) to render each image's own sun, or a
precision stage robust to reversed shading; both need data or work not available here.
