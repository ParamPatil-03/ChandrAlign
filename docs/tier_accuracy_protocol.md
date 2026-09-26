# Protocol: an independent accuracy signal in the confidence tier (audit I-08). Frozen before scoring.

## Question

`evaluate/quality.py` grades only inlier count, ratio and coverage (plus scale). None of these is an accuracy
figure, so TMC-2 -> WAC is HIGH with a 24-34 TMC-2-px fit RMS (`reports/map_tmc2_wac.json`), and HIGH/MEDIUM
are "UNVALIDATED guesses" fitted on synthetic runs. Can the tier say something about accuracy?

## The signal (fixed now)

**Matcher-free NCC probes** (`scripts/parallax_probe_eval.probes`, moved into the library): a 31 px template
every 48 px of the source, searched +-32 px in the reference, and kept only with a clear, unambiguous peak.
Each probe's measured displacement is compared with the DELIVERED geometry's prediction. The statistic is p50
and p95 of |error|, in **source px** (reference px divided by the model's local scale). The probes share no
code with the matcher; they do share the two images (see the audit's caveat on shading bias).

## Thresholds, taken from the problem statement, not fitted to our windows

| tier the signal allows | probe p50 (src px) | probe p95 (src px) | reading |
|---|---|---|---|
| HIGH | <= 0.5 | <= 1.5 | sub-pixel at the median, tail within 1.5 px |
| MEDIUM | <= 1.0 | <= 3.0 | sub-pixel at the median |
| LOW | anything measured | - | locates the scene; not sub-pixel |

- **Unmeasured** (fewer than `tiers.accuracy.min_probes` = 20 accepted probes): the signal allows at most
  MEDIUM. HIGH requires a measured accuracy, the same rule as an unconfirmed scale.
- The signal never REJECTS on its own: probes can be biased by illumination, so a large probe error caps the
  tier at LOW; it does not refuse the result.
- It joins the other signals worst-of, as `accuracy`.

Disclosure: when this was written, the probe numbers of three TMC-2 -> TC N00 windows from the Track B batch
(p50 ~0.3, p95 ~1.1-1.2 TMC-2 px) had been seen. The thresholds come from the sub-pixel mandate and were not
adjusted to them.

## Measurement

Re-grade every committed window of the Track B batch (TMC-2 -> TC 15, OHRC -> NAC 25, IIRS -> WAC 5) with the
signal. Report per pairing how the tiers move (and why), plus the TMC-2 -> WAC map pairing if it is re-run.
There is no pass/fail bar: this measures what the tier now says. The audit's "re-derive HIGH/MEDIUM on real
windows" is answered by reporting how the real windows distribute under PS-derived thresholds, rather than by
fitting thresholds to 45 windows that are all believed correct (with no negatives, a fit would be
circular).

## Result (2026-09-26)

Re-grading with the accuracy signal (Track B batches; `reports/trackb_batch*_summary.json`):
- **TMC-2 -> TC (15):** 12 HIGH / 3 MEDIUM, unchanged. Every window measured sub-pixel at p50.
- **OHRC -> NAC (routed):** 20 LOW / 4 REJECTED, unchanged (already LOW). OHRC probes: median p95 2.38 OHRC px
  (eloftr), so the signal would also have held them at LOW.
- **IIRS -> WAC (xoftr):** 4 HIGH / 1 MEDIUM; probe p95 ~0.23 IIRS px.

**A limitation found by measurement, and its fix.** In the synthetic C-04 replay (512 px pairs rotated 8 deg,
through `register_bundle`) NO result reached HIGH: the probes compare axis-aligned patches, almost none is
accepted on a rotated pair, and "unmeasured" allows at most MEDIUM. Real product runs are resampled into a
common frame first, so they are unaffected (above), but a raw rotated pair could never be HIGH. Fix: the probes
first warp the source through the delivered geometry (the lesson of C-02). See the commit that follows.
