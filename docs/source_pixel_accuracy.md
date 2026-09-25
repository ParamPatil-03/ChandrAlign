# Accuracy in SOURCE-image pixels (the problem statement's unit)

The PS asks for "sub-pixel accuracy of source image". Our registrations measure errors in the
frame the fine stage works in, usually the REFERENCE grid, whose pixels are not the source's.
`src/chandralign/evaluate/source_px.py` converts (worst direction as the headline);
`scripts/source_px_summary.py` applies it to every adopted result -> `reports/source_px_summary.json`.
New runs of `register_tmc2_tc.py`, `register_ohrc_nac.py` and `register_iirs_wac.py` now write a
`source_px` block themselves (exact, from the source->frame transform).

Source pixels (verified): OHRC 0.305 x 0.309 m, TMC-2 4.92 x 5.04 m, IIRS 100.0 x 79.5 m.

## Result (2026-09-25)

| Pairing (adopted config) | source px per frame px | Measure | frame px | **source px (worst)** | sub-pixel? |
|---|---|---|---|---|---|
| TMC-2 -> TC, hilly (9) | 1.49-1.50 (exact) | NCC-probe error, window median | 0.35 med / 0.51 max | **0.53 / 0.76** | yes |
| | | NCC-probe error, 90th pct | 0.70 / 0.95 | 1.04 / 1.43 | no |
| | | known (3,4) px shift recovered | 0.12 / 1.29 | 0.18 / **1.93** | median yes, one window no |
| TMC-2 -> TC, flat (6) | 1.47-1.49 (exact) | NCC-probe error, window median | 0.26 / 0.30 | **0.38 / 0.44** | yes |
| | | known shift | 0.08 / 0.18 | 0.12 / 0.26 | yes |
| OHRC -> NAC (20) | **1.75-4.23** (pixel sizes) | known shift | 0.37 / 1.34 | **0.94 / 3.56** | median yes (barely), max no |
| | | fit residual RMS of delivered points | 1.43 / 2.02 | **3.49 / 6.08** | **no** |
| IIRS -> WAC (5) | 1.26 (pixel sizes) | known shift | 0.03 / 0.06 | 0.04 / 0.08 | yes |
| | | fit residual RMS | 0.26 / 0.49 | 0.32 / 0.61 | yes |
| IIRS -> NAC, dense (4) | 1.00 | step precision | 0.10 / 0.18 | 0.10 / 0.18 | yes (+ fractional shifts 0.02-0.11) |

## What it means

- **IIRS (both references): sub-pixel in IIRS pixels on every measure.**
- **TMC-2 -> TC: the MODEL is sub-pixel in TMC-2 pixels** (probe medians 0.38-0.76 src px); its 90th
  percentile on hills is ~1 src px, and one hilly window's known-shift recovery is 1.9 src px.
- **OHRC -> NAC is NOT sub-pixel in OHRC pixels.** The NAC frame is 1.75-4.2x coarser than OHRC, so
  every frame-pixel error is multiplied by that. Reported before in NAC pixels, this was invisible.
  Honest reading: against a reference 2-4x coarser than the source, OHRC-pixel accuracy is
  information-limited by the reference; the finest NACs (0.4-0.5 m) help most (not yet tested).
- **Two kinds of "RMSE".** The per-point residual RMS of delivered match points (the classic RMSE)
  is 1-2+ source px for TMC-2 and OHRC even where the model is sub-pixel: single matches scatter at
  about one REFERENCE pixel, and those references are coarser than the source. Reports must name
  which one they give. The PS's "RMSE" should be reported as both, labelled.

Limits: OHRC and IIRS->WAC conversions use pixel sizes (their runs did not store the transform);
NAC pixel size is LROC's and not independently verified. Probes are a proxy, not ground truth.
