# Protocol: benchmark the RoMa family already inside vismatch 1.3.2 (audit I-05 / G-03). Frozen before measuring.

## Question

Every current matcher except MIND is LoFTR-family, and they share failure modes. The pinned `vismatch 1.3.2`
also ships dense matchers built on DINOv2 coarse features, the strongest published signal for large
appearance change, and none of them was ever measured here. Do any of them solve what the current pool does
not: OHRC at 75 deg incidence (M1417360906LC, 0/5), opposed sun (M175124932LC), TMC-2 -> MI (0/5)? Do they
do it without new confident wrong answers?

## Candidates (benchmark only: ship_mode off, recorded in each output)

| model | components | licence status for SHIPPING |
|---|---|---|
| `minima-roma` | MINIMA (Apache-2.0) + RoMa (MIT) + DINOv2 (Apache-2.0) | audit row needed (I-06 allowlist) |
| `roma` | RoMa (MIT) + DINOv2 (Apache-2.0) | audit row needed |
| `matchanything-roma` | MatchAnything, depends on the vismatch pin like eloftr | audit row needed |
| `ufm`, `gim-dkm` | unknown | full licence audit needed |
| `romav2` | DINOv3 backbone, Meta DINOv3 licence (not OSI) | benchmark only, never shippable |
| `master`, `duster` | CC BY-NC-SA | excluded, not run |

Nothing here is promoted to shipping without a `reports/licence_audit.json` row (verdict pass). This benchmark
decides only whether an audit is worth doing.

## Fairness fixes applied BEFORE measuring (audit M-06 and G-03)

1. **Keypoint budget:** dense models get `matching.dense_max_num_keypoints` = 5000 samples, not 2048.
   RoMa samples exactly that many, and 2048 handicaps it on the inlier and coverage gates.
2. **Certainty floor:** dense-model matches below `matching.dense_min_certainty` = 0.05 are dropped. A dense
   model returns N samples even on pure noise, so without a floor it can fail the noise null gate by
   construction rather than by finding structure. 0.05 is RoMa's own default sampling threshold, not a
   value tuned here.
3. Both fixes apply only to dense models (`roma` family, `gim-dkm`, `ufm`); nothing else changes.

## Measurements (the audit's list; each is one argument to an existing script)

S. **Synthetic.** `scripts/bench_rift.py --methods <candidates>` (8 regimes x seeds 3/11/29, exact truth):
   successes (< 2 px) and **false confidences** (accepted while > 2 px wrong), next to the committed rows for
   eloftr / xoftr / minima-loftr in `reports/rift_benchmark.json`.
R1. **OHRC -> NAC, all 5 NACs:** `register_ohrc_nac.py --auto-bridge --matchers eloftr minima-loftr minima-roma`
   (roma added on M1417360906LC only, the audit's partial-evidence case). Per matcher: success / unconfirmed
   (MI) / failed, known-shift error, probe p50/p95 in OHRC px.
R2. **TMC-2 -> TC 15 windows:** `register_tmc2_tc.py --matcher minima-roma` vs the eloftr batch run.
R3. **IIRS -> WAC:** `register_iirs_wac.py --matchers xoftr minima-roma matchanything-roma`.
R4. **TMC-2 -> MI:** `register_to_map.py --source tmc2 --reference mi --matchers eloftr minima-loftr minima-roma`.

Order, by value to the open cases: S, R1, R4, R3, R2. One GPU job at a time, on an otherwise idle machine
(audit: RoMa stalled at 14.5 GB RAM next to other jobs).

## Decision rule (fixed now)

A candidate is **recommended for a licence audit and a fallback slot** only if all three hold:
1. **zero** false confidences in S;
2. on R1 it solves (success, MI not flagged) strictly more OHRC windows than `minima-loftr`, OR the same
   number with a lower median probe p95;
3. it opens no new failure on R2-R4 (no window that eloftr / xoftr solved becomes a confident wrong answer).
Otherwise its results are recorded and it is not recommended. Speed is reported (seconds per window), and
it does not decide.
