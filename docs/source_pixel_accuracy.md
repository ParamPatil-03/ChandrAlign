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

## OHRC -> NAC: why it is not sub-pixel in OHRC pixels (diagnosed 2026-09-25)

`scripts/ohrc_accuracy_diag.py` -> `reports/ohrc_accuracy_diag.json` (the 4 solved products rerun
with `--dump-points`). The ~3.5 OHRC-px figure is mostly POINT SCATTER; the MODEL is better:

| NAC | frame px = OHRC px x | point scatter | model error (matcher-free NCC probes, median) |
|---|---|---|---|
| M102014464RC (5) | 4.3 | 2.8-6.1 OHRC px | 0.41-1.04 NAC px = **1.8-4.5 OHRC px** |
| M106719774LC (5) | 4.2 | 3.6-4.4 OHRC px | 0.32-0.52 NAC px = **1.3-2.2 OHRC px** |
| M109080308LC, M175124932LC | 1.8 / 2.4 | 2.8-4.1 OHRC px | 0-3 probes accepted: not measurable |

Conclusion: against a NAC 4x coarser than OHRC, the model is sub-NAC-pixel but 1.3-4.5 OHRC px --
an information limit of the reference, not a pipeline defect. Sub-OHRC-pixel accuracy needs a NAC
of ~0.5 m or finer, well lit, AND model error <= ~0.55 NAC px; the finer NACs held are near-overhead
sun (2 deg) or give almost no probes, so it cannot be demonstrated with the data held. Terrain
parallax could not be tested (no DEM heights in these dumps).

### MI check on the same frames (matcher-free; MATCH-07's alignment_check, PR #27)
- Coarse NACs (M102014464RC, M106719774LC; 10 windows): NMI 1.04-1.18, peak 0.07-0.23 NAC px from
  the model = **0.29-0.95 OHRC px**: by this measure the model's GLOBAL offset is sub-OHRC-pixel on
  10/10 windows. (The NCC probes' 1.3-4.5 OHRC px include each probe's own local noise.)
- Finest NAC M175124932LC (0.40 m): NMI only ~1.006 at the model, but a searched (+-10 px) peak
  sits 4.2-5.9 NAC px CROSS-TRACK away (NMI ~1.02), consistently on 3/3 windows checked. Either the
  registration to this NAC is biased by ~2 m, or MI is misled (e.g. shadows under the two different
  suns). Unresolved without an independent reference. Candidate: M111443315LC (LROC, 0.52 m,
  incidence 26 deg, overlaps ~8 km of the OHRC strip; 529 MB) -- not downloaded.
- M109080308LC (2 deg sun): NMI 1.008-1.031, peaks 0.8-4.2 NAC px: too little shared information to judge.

## OHRC coverage on the near-overhead-sun NAC (diagnosed 2026-09-25)
M109080308LC (incidence 2 deg) left 48-64% of the grid covered. Its empty cells have the SAME
band-pass texture as occupied ones (0.040 vs 0.041), and re-matching them returns ~39 matches per
cell that scatter by ~34 px inside the cell (TMC-2's parallax cells scattered ~1 px): random, not
displaced -- the matcher cannot find true correspondences on a shading-free NAC. Refill (ALIGN-05)
cannot help (it keeps only matches that agree with the model). Over the SAME OHRC windows, coverage
follows the reference's sun: M175124932LC (41 deg) 0.92-0.98, M111443315LC (26 deg) 0.69-0.80,
M109080308LC (2 deg) 0.48-0.64. Fix = choose a well-lit reference (a pair-selection rule), not a
matching change. (M102014464RC's few empty cells are texture-free: 0.0005 vs 0.041, a data limit.)
