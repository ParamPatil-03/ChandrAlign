# Part 2 verification (2026-09-25)

1. **Static check** (pyflakes over `src/chandralign` and `scripts`): no undefined names or real errors;
   only unused imports (removed in Part 2 files; Part 1 files left to their owner).
2. **Code review** of the certification path (control gates, quality tiers): gate maths and the
   integer-shift helper verified correct for both signs. Fixed: a REJECTED result with NO control gates
   carried an empty failure_modes list, breaking CHECK-07's contract -> now failure mode 19
   (fabricated / unverified evaluation, PLAN.md s15), with a test. Fixed: the perturbation gate's
   message hard-coded "5 px".
3. **Tests**: the full suite passes.
4. **Reproduction from scratch** on current code of every adopted result, compared with the committed
   reports under tolerances fixed before running (`scripts/verify_reproduction.py`,
   `reports/reproduction_check.json`): TMC-2 -> TC 15/15, OHRC -> NAC 25/25, IIRS -> WAC 5/5,
   IIRS -> NAC 9/9, TMC-2 -> NAC 4/4 groups -- **0 not reproduced**.

Known LIMITS (measured, documented, not bugs): OHRC -> NAC tiers capped at LOW because NAC pixel size is
unverified; OHRC 75 deg illumination (0/5); a possible ~2 m bias on the finest NAC (unresolved);
IIRS -> NAC limited to NACs wide enough; the WAC / MI / TC pairings (degraded or information-limited).
