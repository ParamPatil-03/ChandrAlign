# Protocol: MATCH-07 mutual information as an alignment check (frozen before running)

PLAN P2-T22's corrected bar: MI must rank known misalignments (0, 1, 2, 4 px) monotonically on real
CROSS-MODAL pairs, and flag a 2 px error the control gates miss. The gates measure the PRECISION
of the lock (a re-run on a shifted copy agrees with itself); a model that is consistently 2 px off
passes them. docs/usefulness_checks.md U1 already showed NMI ranks shifts on 15/15 TMC-2 -> TC
windows (same broadband modality, different sensors).

## Method (`matching/similarity.py`)
`nmi(a, b, mask)`: normalised MI (Studholme), 64 bins. `alignment_check(src, ref, model, ...)`:
warp the source through the model plus every shift on a +-3 px grid (1 px step), fit a quadratic
to NMI around the best grid cell -> sub-pixel `peak_offset_px`; `flag = |peak_offset| >= 1.0 px`.

## Data
TMC-2 -> SELENE TC: the 15 adopted fine frames (source, reference, model). IIRS -> LRO WAC mosaic
(truly cross-modal: IIRS PREP-06 composite vs broadband 643 nm): the 5 adopted windows, xoftr,
rerun with `--dump-points` (same code path as reports/iirs_wac_mosaic.json).

## Decision (all must hold for MATCH-07 DONE and the check to be reported on every registration)
(a) Ranking: mean NMI over 8 directions strictly decreases over shifts 0, 1, 2, 4 px on >= 14/15
    TMC-2 windows and 5/5 IIRS windows.
(b) Bias detection: the delivered model offset by 2 px in each of 8 directions is flagged in >= 95%
    of cases per pairing, and the recovered offset is within 0.5 px of the injected one (median).
(c) False flags: the delivered (unbiased) models are flagged on <= 10% of windows per pairing.

## Result (2026-09-25): ADOPTED -- `reports/mi_check.json`

| | TMC-2 -> TC (15) | IIRS -> WAC, cross-modal (5) |
|---|---|---|
| (a) NMI ranks 0 > 1 > 2 > 4 px | 15/15 | 5/5 |
| (b) 2 px-biased model flagged | 120/120 | 40/40 |
| (b) bias recovered, median error | 0.13 px | 0.19 px |
| (c) delivered models falsely flagged | 0/15 | 0/5 |

All pass. Every registration script (TMC-2 -> TC, OHRC -> NAC, IIRS -> WAC) now reports `mi_check`:
NMI at the delivered model, where NMI peaks (sub-pixel, also in source px) and a flag when the
peak is >= 1 px away. It is a matcher-free check of ACCURACY (bias), which the control gates --
measuring only the lock's repeatability -- cannot see. Limits: global over the frame (local relief
errors average out); needs texture; a +-3 px search, so larger errors show as `at_search_edge`.

Observation on first use (not part of the decision): on OHRC -> NAC M109080308LC (sun ~2 deg
incidence; the low-coverage windows, 0.48-0.64), the check flags 3 of 5 windows the pipeline graded
LOW-but-successful: NMI peaks 1.2-1.6 NAC px from the model (2.2-2.9 OHRC px). But NMI there is
~1.01-1.03 (1.0 = no shared information) vs 1.12-1.21 on the validated pairs, so the NMI surface is
nearly flat and the flag may be noise. Neither "these windows are wrong" nor "the check is
unreliable at low NMI" is shown; it is carried into the OHRC accuracy/coverage investigation.
