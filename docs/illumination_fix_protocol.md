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
