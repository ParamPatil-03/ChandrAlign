# Protocol: fixing registration under large sun differences (frozen before measuring)

Follows `docs/ohrc_nac_results.md` (run 3). Same data: OHRC `ch2_ohr_ncp_20240330T0035085365`
against five LRO NAC products, same 25 windows (the run-3 placement), same fine stage.

## Q1. The coarse lock (the first thing that broke)

MIND at ~4 m locked 0/5 windows on the 55-deg-incidence product and 0/5 on the opposed-sun
product. Candidates, each grounded in evidence we already hold:

| id | descriptor | resolution | why |
|---|---|---|---|
| A | MIND | 4 m | the current method (baseline) |
| B | phase congruency | 4 m | our real-relief test: PC held +0.92 across OHRC's and TMC-2's suns (raw -0.98); the synthetic bands give PC 60-135 deg azimuth |
| C | MIND | 2 m | 4 m averages away most craters that a grazing sun lights; 2 m keeps them |
| D | phase congruency | 2 m | B and C together |

**A window's lock is accepted** when z >= 10 AND the implied OHRC offset is **consistent**:
- within 300 m of 2,150 m north (the OHRC system error north, measured twice
  independently: 2.21 km on the control, 2.23 km against SELENE TC; east is excluded
  because each NAC product has its own east bias, up to ~1 km in PR #18), AND
- within 150 m of the median of that product's other accepted locks (when there are >= 2).
A z >= 10 lock failing either test is a **false lock**.

**Decision:** adopt the candidate, or the ordered fallback "A, then the best other if
A's z < 10", that gives the most accepted locks over the four non-control products,
provided it (i) keeps the control at 5/5, and (ii) produces **zero** false locks. If no
candidate beats A, the coarse lock is recorded as unsolved for these sun differences.

## Q2. The matcher after a lock

On every window the Q1 winner locks: `eloftr`, `minima-loftr`, **`xoftr`** (added: licence-
clean, our cross-modal matcher; the synthetic sweep predicts it fails 75-120 deg and
succeeds 150-180 deg), `sift`. Success per window as frozen in
`docs/ohrc_nac_protocol.md` (gates, tier >= LOW, consistent location).

**Decision (fallback policy):** keep `eloftr` as the default. Add an ordered fallback
used ONLY when the default's result is REJECTED or fails a gate: the fallback list is the
other learned matchers ordered by their success count here, keeping only those that add
at least one success the earlier ones did not. A fallback is never used when the default
succeeded, so it cannot change any currently accepted result.

## Q3. The LOW grade

Tiers are capped at LOW because the NAC pixel size is `unverified`: LROC's scaled pixel
and ODE's `Map_resolution` agree, but both come from the LROC team's SPICE geometry, so
they are ONE source, not two. The independent second source is a measurement from
pixels: PR #18's TMC-2 -> NAC locks (TMC-2's own pixel measured from OHRC, 4.920 x
5.037 m) give each NAC product's pixel size through `nac_to_tmc`.

**Decision:** for each NAC product with >= 3 PR #18 locks, the measured pixel size (median
over windows, per axis) is added to `configs/measured_scales.yaml` if every window agrees
with LROC's value within 10%. A product without that measurement stays `unverified`.
This measurement is NOT used to check the TMC-2 -> NAC pairing that produced it.

## Q4. The opposed-sun prediction

Tested only if Q1 produces accepted locks on M175124932LC; otherwise it stays untested.

## Limits

Five products, 25 windows, one site. Candidate selection and evaluation use the same
windows, so the winner's success rate is optimistic; it is reported as such.

## Q5 (added after Q1-Q4, frozen before its run): the geodetic bridge

Q1 left two products with no lock under any descriptor (z ~5 everywhere). A +-4 km
search needs a clear peak; weak similarity may exist without one. Q5 separates "no
similarity" from "search too wide" by removing the search.

**Bridge prediction (research doc opportunity #9, FEATURES MATCH-11), all from measurements
independent of the OHRC <-> NAC pair:** OHRC's system error against SELENE TC,
(+551, +2229) m east/north (`reports/cascade_ohrc_tc.json`), plus each NAC product's
geolocation difference from TC, measured through TMC-2 in PR #18 (median per product of
system_offset - tc_measured_offset): M106719774LC (+557, -143), M175124932LC (+217, -288),
and for validation M102014464RC (-597, +75).

**Method:** the OHRC window is placed at its bridge-predicted position in the NAC (no
coarse search) and the fine frame is widened by 200 m on every side; the matcher then
has to find the residual itself. Matchers: `eloftr`, `minima-loftr`.

**Success per window:** fine stage returns a model; all five gates pass; tier >= LOW;
and the window's implied OHRC offset (from the FINAL transform) is within 150 m of the
median over that product's gate-passing windows for that matcher (>= 3 such windows
required, else no window counts). Verdicts as before.

**Validation first:** the same run on the control (M102014464RC) must give 5/5 for
eloftr, and its bridge prediction must lie within 200 m of its Q2-measured offset;
otherwise the bridge is not trusted and the two products are not interpreted.

**Interpretation:** success on a product => its earlier failure was the search, and the
bridge fixes it; failure => no usable similarity at that sun geometry for these
matchers, and the product is recorded as unsolved by the methods available.

### Q5 amendment 1 (before M175124932LC was measured)

The first complete Q5 run measured the control and M106719774LC, but every
M175124932LC window died of CUDA out-of-memory: its NAC pixels are 0.24 m across, so the
bridge frame (+200 m each side) was ~4,200 px wide and eloftr asked for ~45 GB. No
result exists for that product; the harness could not run it.

Change, harness only: the fine stage works on the NAC grid block-averaged per axis so no
axis is finer than 0.5 m (`FINE_MIN_M`). Only M175124932LC changes (0.24 x 0.56 m ->
0.72 x 0.56 m); every other product's pixels are already >= 0.5 m, so their measured
results stand. Success rules unchanged. Only M175124932LC is re-run.

### Q5 amendment 2 (before any M175124932LC result existed)

With amendment 1 the M175124932LC frames were 1,195-1,301 x 1,857 px (2.4 Mpx) and still
ran out of GPU memory; nothing was measured. Frames up to ~1.3 Mpx have run on this GPU
(M109080308LC, Q2). Change, harness only: after amendment 1, the finer axis is coarsened
by further whole-pixel blocks until the frame is at most 1.5 Mpx (`FINE_MAX_PX`). No
measured product is affected (the control and M106719774LC frames are < 0.7 Mpx).

## Q5 result (recorded before Q6 was designed)

- Validation: control eloftr 5/5; bridge prediction 128 m from the measured offset (< 200 m).
- M106719774LC (55 deg incidence): eloftr 3/5, minima-loftr 3/5 (degraded); routed
  (eloftr then minima-loftr) 4/5; every locked window within 6 m of the others.
  **Its Q1 failure was the +-4 km search, not the illumination.**
- M175124932LC (opposed sun): eloftr fails outright (6-34 matches, scattered offsets).
  minima-loftr puts all 5 windows at mutually consistent positions (within ~50 m; ~130 m
  from the bridge) with 300-1,045 matches, but its known-shift error is 1.48-2.29 px (gate
  1.5 px): 1/5 passes, fewer than the 3 the rule needs -> unsolved. **It locates the
  opposed-sun pair; it is not precise enough.**

## Q6 (frozen before its run): illumination-robust dense refinement of the model

The remaining failures (M175124932LC; M1417360906LC in Q2, errors 2.2-3.1 px) are PRECISION
failures after a correct lock. The coarse step already refines on illumination-robust
channels (`cascade.register_step_dense` step 4: NCC/phase on raw intensity AND NCC on each
MIND channel, median of all that succeed). Q6 applies the same estimator to the WHOLE fine
frame after the first robust estimate: warp the source with the model, estimate the
residual translation (median over raw + MIND-channel estimates, at least 3 succeeding,
each within 1.5 px), update the model's translation, repeat once. New pipeline stage
`dense_refine` (default OFF). The control gates run it too (they call fine_stage).

**Runs:** stage ON for: control and M175124932LC (bridge mode, as Q5), and M1417360906LC
(normal 2 m lock, as Q2). Stage OFF results for the same windows already exist (Q5, Q2).

**Decision:** `dense_refine` becomes a default if (a) the control stays 5/5 for eloftr
and its median known-shift error rises by no more than 0.05 px, AND (b) it adds at least
one success on M175124932LC or M1417360906LC. Otherwise it stays off and both products
are recorded unsolved.

## Q6 result

| product | matcher | known-shift error, stage off -> on | successes off -> on |
|---|---|---|---|
| control | eloftr | median **0.568 -> 0.053 px** | 5/5 -> 5/5 |
| control | minima-loftr | 0.14-1.83 -> 0.04-0.68 px | 4/5 -> 5/5 |
| M175124932LC (opposed) | minima-loftr | one window 1.76 -> 0.59; elsewhere the estimator fails (0-1 of 10 estimates succeed) | 0 -> 0 |
| M1417360906LC (75 deg) | minima-loftr | mixed (0.66 -> 0.32; 2.18 -> 2.89) | 0 -> 0 |

Rule (a) holds (control 5/5, median error far lower); rule (b) fails (no new success on
either hard product). **By the frozen rule `dense_refine` stays OFF, and both products are
recorded unsolved at the precision stage**: under opposed sun / 75 deg the illumination-
robust estimators themselves fail, not only the matchers.

The 10x precision gain on same-sun data was seen on the control, i.e. on data used to
look at the result, so it is NOT adopted on that basis. Q7 tests it on data it has not
seen.

## Q7 (frozen before its run): dense_refine as a PRECISION stage, on unseen data

Data: the nine committed TMC-2 -> SELENE TC windows (`scripts/register_tmc2_tc.py --tile
all --windows 3`), current defaults, with `--stage dense_refine=on`; compared with the
committed `reports/tmc2_tc_registration.json` (same code, stage off).

Adopt `dense_refine: true` as a default iff ALL hold: no window loses registration, tier
or a gate; the median known-shift error falls by at least 20%; the largest known-shift
error does not rise.

## Q7 result

Nine TMC-2 -> TC windows, dense_refine on: no window lost; median known-shift error
0.081 -> 0.023 px (72% lower); but the largest rose 0.208 -> 0.345 px (row 4687, the N00
window whose residual to the affine model is largest). **Rule 3 fails -> not adopted.**
A single translation update cannot fix a non-affine residual; this is ALIGN-02's (TPS)
question, and dense_refine should be re-tested together with TPS.

## Q8 (frozen before its run): the geodetic bridge as an automatic fallback

Option considered and rejected first: rendering each image's sun from a high-resolution
DEM. The only one near (LROC NAC_DTM_APOLLO11, 2 m) covers 0.68 km2 of the OHRC /
M175124932LC overlap (about one window; the rules need 3) and none of M1417360906LC's.

**Shipped behaviour under test:** the 2 m MIND coarse lock; ONLY if it fails (z < 10) and a
bridge prediction exists for the NAC product, the window is re-run at the bridge-predicted
position (Q5 method); then the routed matcher (eloftr, falling back to minima-loftr).
Bridge predictions are COMPUTED from committed reports, never typed in: OHRC vs SELENE TC
= median `ohrc_system_offset_m` of `reports/cascade_ohrc_tc.json`; NAC vs TC = median
(system_offset - tc_measured_offset) over the successful windows of that product in
`reports/tmc2_nac_registration.json`. Products without a TMC-2 measurement get no bridge.

**Success per window:** as Q5 (gates, tier >= LOW, and the FINAL transform's implied offset
within 150 m of the median over that product's gate-passing windows, >= 3 required),
applied to every window whether it was locked or bridged.

**Adopt automatic bridging** if, on all five products, no window that succeeds without it
fails with it, and it adds at least one success.

### Q8 amendment 1 (after run 1, which is void)

Run 1 placed windows by OHRC's APPARENT (system) position. For the two bridged products the
bridge moves each window ~2 km, off their narrow NAC strips: M106719774LC's frames fell
entirely outside (5 skipped) and M175124932LC's were clipped to 111-182 px (0 matches).
Nothing was measured for either. Change, placement only: for every product WITH a bridge
prediction, windows are placed by the predicted true position (as Q5 did). The coarse lock
still runs first on each window; the bridge is still used only if it fails. Products
without a prediction are placed as before. Run 2 is the result; all five are re-run.

## Q8 result (run 2) -- and a correction to Q1, Q5 and run 3

With windows placed by the PREDICTED true position, the coarse lock itself succeeded on every
window of both "unsolvable" products; the bridge fallback was never triggered:

| NAC | sun difference | coarse lock | routed result | used |
|---|---|---|---|---|
| M102014464RC control | ~1 deg | 5/5 | **5/5** | eloftr |
| M106719774LC | 55 deg incidence | 5/5 | **5/5** (err 0.31-0.85 px) | eloftr |
| M175124932LC | **178 deg azimuth** | 5/5 | **5/5** (err 0.05-0.68 px) | minima-loftr (eloftr failed first) |
| M109080308LC | 80 deg incidence | 5/5 | **5/5** | minima-loftr |
| M1417360906LC | 75 deg incidence | 4/5 | 0/5 (err 0.66-3.09 px) | -- |

**Rule:** automatic bridging added no success (it never triggered), so the fallback is NOT
adopted. What changed the outcome was placement.

**Correction.** The Q1 / run-3 / Q5 "no lock" failures on M106719774LC and M175124932LC
were a harness error, not illumination: windows were placed where OHRC's SYSTEM position
fell inside the NAC strip, but OHRC's true ground is ~2.2 km away, and these strips are
only 1.2-1.6 km wide -- the window's content was partly or wholly outside the NAC image.
Q5 appeared to "fix" them because bridge mode also moved placement. Conclusions drawn from
those runs ("the coarse lock breaks first", "opposed sun located but not precise") are
withdrawn. Prediction 2 is now TESTED and CONFIRMED: at 178 deg eloftr fails and
minima-loftr registers 5/5.

The pipeline lesson: pair / window selection must use the best KNOWN geolocation (system
position corrected by measured offsets), not the raw system position, whenever the
reference strip is narrower than the source's system error.

## Q9 (frozen before its run): automatic window enlargement on rejection

Shipped behaviour = Q8 (2 m lock, routed matcher, best-known placement) PLUS: a window whose
routed result is rejected (not registered, a gate failed, or tier REJECTED) is re-run ONCE
with a window twice as large per side (2048 -> 4096 OHRC px), same centre. The larger
result replaces the smaller only if it is registered, passes every gate and tiers >= LOW.
Success rules as Q8 (consistency from the final transforms, >= 3 gate-passing windows).

**Adopt** if, over all five NAC products, no window that succeeds under Q8 fails under Q9,
and Q9 adds at least one success. Evidence it should: the exploratory 4096 px check on
M1417360906LC (4/5).

## Q9 result: not adopted

All windows that succeeded under Q8 still succeed (20/25, none lost); no success added.
On M1417360906LC (75 deg) enlargement triggered on all 5 windows: 1 no lock at 4096 px;
2 now pass every gate (known-shift 0.23 and 0.55 px); 2 are still rejected (2.5 and 3.1 px).
Two gate-passing windows are fewer than the 3 the consistency rule needs, so none counts
-> 0/5. The exploratory 4/5 used windows CENTRED to fit 4096 px inside the strip; Q9 keeps
the 2048-px centres, and 2 of its enlarged windows run past the strip edge. **By the frozen
rule, automatic enlargement is not adopted; the 75 deg case remains unsolved in the
shipped behaviour** (solvable in principle: exploratory 4/5).

## Q10 (frozen before its run): enlargement re-centred to fit the strip

As Q9, except the 4096 px window is placed at the candidate centre nearest the original
window whose 4096 px window lies inside the NAC footprint (the placement rule of
amendment 2, applied at 4096 px) -- what the exploratory 4/5 check did. If no such centre
exists, no enlargement. Adopt on the Q9 rule (none lost, at least one added).

## Q10 result: not adopted -- 75 deg closed as a known limit

M1417360906LC only (the other products never trigger enlargement): 1 window enlarged and
passed (0.34 px); 3 re-centred 4096 px windows still rejected; 1 no lock -> 0/5. The
exploratory 4/5 used windows chosen for 4096 px from the start; the outcome at this sun
geometry depends strongly on WHICH ground a window covers. Iterating further would tune to
one product. **Closed: the 75 deg case (NAC sun near overhead vs OHRC grazing) is a known
limit -- sometimes registrable with larger windows, not reliably.** Shipped behaviour stays
Q8: 20/25 OHRC -> NAC windows.
