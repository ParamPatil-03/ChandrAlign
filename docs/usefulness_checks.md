# Are MATCH-07, MATCH-08, ALIGN-07 and the WAC / SELENE-MI pairings useful? (frozen before running)

An earlier verdict called these "not useful" on reasoning alone; the user challenged it (they
also matter for demonstrating the prototype). Each is now tested; criteria fixed before running.
Data: the 15 TMC-2 -> SELENE TC fine frames of the adopted run (`--dump-points`: source, reference,
model, inliers), plus footprints of the held WAC mosaic clip and SELENE MI tile.

## U1. MATCH-07 mutual information as a registration-quality score
Normalised MI (Studholme, 64 bins) between the source frame and the reference frame, over valid
pixels, with the source shifted by 0, 0.5, 1, 2 and 5 px (8 directions each). Useful if, on >= 14
of 15 windows, NMI is highest at 0 shift AND falls monotonically with shift size (mean over the 8
directions). That makes it a matcher-free, cross-modal quality metric worth reporting.

## U2. ALIGN-07 farthest-point sampling vs the grid (ALIGN-04)
On each window's inliers, select the same number of points as the grid keeps, by (a) grid top-k,
(b) farthest-point sampling. Also on a clustered case: the same inliers with 70% of one half
removed (the OHRC situation). Measures: grid coverage, largest empty circle (max Delaunay gap),
coefficient of variation of nearest-neighbour distances. Useful if FPS lowers the largest empty
circle by >= 20% at equal point count on the median window, or on the clustered case.

## U3. MATCH-08 craters as an independent check (feasibility only; the feature is a stub)
Detect bright/dark crater-rim pairs? Too much to build for a check. Instead: blob detection
(cv2.SimpleBlobDetector on the Laplacian-of-Gaussian-filtered image, dark blobs 3-40 px) in source
and reference; map source blobs by the model; repeatability = share with a reference blob within
2 px. Useful as an independent landmark check if repeatability >= 30% at the true model and
<= 5% at a 10 px wrong model, on >= 12 of 15 windows.

## U4. Pairings with data already held (no download)
Overlap of the held CH-2 products with the WAC mosaic clip (4S-4N, 22.3-25.3E) and the SELENE MI
tile (0-1N, 23-24E), from system geolocation corrected by our measured offsets. A pairing is
"testable now" if >= 1 product overlaps by >= 20% of its footprint (or of the tile).

## Result (2026-09-25) -- `reports/usefulness_checks.json`

| Check | Criterion | Result | Verdict |
|---|---|---|---|
| U1 MI score | peak at 0 and monotone on >= 14/15 | **15/15**, even at 0.5 px (e.g. 1.1521 -> 1.1478 -> 1.1369 -> 1.1100 -> 1.0646) | **USEFUL** (earlier "not useful" was wrong) |
| U2 farthest-point sampling | max empty circle -20% | -16% normal, -11% clustered | not useful by the rule |
| U3 craters (blob stand-in) | repeatability >= 0.30 & wrong <= 0.05 on >= 12/15 | 7/15 (0.16-0.55); wrong-model 0.000-0.007 on all 15 | not useful by the rule |
| U4 pairings | overlap with held data | OHRC in WAC clip 100%, 55% inside the MI tile; TMC-2 21% in WAC, covers 65% of MI tile; IIRS covers 40% of MI tile | **all testable now, no download** |

Reported, not ruled:
- U2: FPS spaces points far more evenly (nearest-neighbour CV 0.73 -> 0.13) but cannot fill holes
  (clustered coverage 0.984 -> 0.953). Evenness was not the pre-declared measure.
- U3: even a crude blob detector separates the true model from a 10 px error completely (true >=
  0.16 vs wrong <= 0.007 on 15/15). Its absolute repeatability is low, so the test does not show
  that crater matching is useful; a real crater detector (rim + shadow with the sun direction) was
  not tested and might do better.

Correction to the earlier verdict: MATCH-07 and the WAC / SELENE-MI pairings are useful (MI as a
matcher-free cross-modal quality metric; the pairings complete the "any CH-2 instrument against
any named reference" demonstration with data already held). ALIGN-07 and MATCH-08 are not shown
useful by these tests.
