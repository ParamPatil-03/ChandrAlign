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
