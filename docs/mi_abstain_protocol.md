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
