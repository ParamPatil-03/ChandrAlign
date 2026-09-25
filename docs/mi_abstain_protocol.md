# Protocol: when should the MI check abstain? (frozen before running)

Problem (docs/tc_reference_protocol.md amendment 2, docs/source_pixel_accuracy.md): at NMI ~1.00-1.04 the
NMI surface is nearly flat and the MATCH-07 check flags models that every other check supports
(OHRC -> TC dense lock: fractional shift 0.01-0.16 px, nulls pass, positions agree to ~30 m, yet 7/10
flagged). A flag should mean "MI sees a different alignment", not "MI cannot see".

## Rule R (no NMI threshold to tune)
`alignment_check` also finds the NMI peak separately on the two halves of the frame (split across its
longer axis). If the two halves' peaks disagree by more than the flag distance (1.0 px), MI has no
reliable answer there: the check ABSTAINS -- `abstain: true`, `flag: false`, reported "inconclusive"
(neither a pass nor a flag). Otherwise it flags exactly as before.

## Validation, on cases whose truth is known (OHRC -> TC NOT used)
- V1 known-good, delivered models: TMC-2 -> TC (15 windows), IIRS -> WAC (5), OHRC -> coarse NAC
  (M102014464RC, M106719774LC: 10; MI and model agree there) = 30. Must abstain on <= 10% (3).
- V2 known-bad: each of those 30 models offset by 2 px in 8 directions = 240. Must be FLAGGED (not
  abstained) on >= 95% (228). An abstain here is a miss.
- V3 reported only: abstain / flag rates on the low-information dumps (OHRC -> M175124932LC, M109080308LC).
Adopt (default in `alignment_check`) only if V1 and V2 both hold.

## Then, and only if adopted: OHRC -> TC re-evaluated
Rerun `register_to_map.py --source ohrc --reference tc --dense` on the same 10 windows with rule R.
The success rule of amendment 2 is unchanged except that an ABSTAINING MI check does not fail a window
(it carries no evidence either way); the fractional known shift, both nulls and the linear-drift
consistency still decide. Reported as "MI inconclusive" wherever it abstains.

## Result (2026-09-25): NOT adopted -- V2 fails (`reports/mi_abstain_check.json`)
- V1 known-good: abstained on 2/30 (limit 3) -- pass.
- V2 known-bad 2 px: flagged 225/240 = 93.8% (need 95%) -- **fail**. All 15 misses are ABSTENTIONS
  on 3 hilly TMC-2 windows (1562, 3125/c3232, 4687/c3232): there terrain parallax makes the two halves'
  true local offsets differ, so the halves disagree although MI is informative. Rule R confuses
  "MI cannot see" with "the image is locally distorted". Without R those biased models were flagged.
- V3 (reported): on the low-information OHRC -> NAC dumps it abstains on 5/10 and still flags 5/10;
  where it still flags (e.g. M175124932LC, halves agreeing to 0-0.7 px), MI's disagreement with the
  model is consistent across the frame -- weak support that the finest-NAC offset is real.
Rule R stays implemented but OFF (`similarity.abstain: false`). OHRC -> TC is NOT re-evaluated.
A better abstain test would have to separate flat MI from relief (e.g. require the halves to
disagree AND the full-frame NMI surface to be flat); that is a new test, not attempted here.
