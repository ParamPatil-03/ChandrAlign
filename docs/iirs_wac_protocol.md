# Protocol: IIRS -> LRO WAC on real data (frozen before any measurement)

Research doc sections 19/25: IIRS <-> LRO WAC is the **Medium / credibility-floor** pairing
(solved by a SIFT-based preprint and arXiv:2509.04775 section 3.2: RMS under one IIRS
pixel). It is also the first real test of our cross-modal route (IIRS is hyperspectral,
WAC is a broadband camera): `routing.choose("IIRS", "WAC")` -> **xoftr, tiled at 640 px**.

## Data

IIRS `ch2_iir_nci_20240523T1600301891` (13,101 x 250 px, 256 bands) against the two SUNLIT
WAC products that overlap it (the night-side ones are excluded, as `build_pairs` flags):
M106705467MC and M106698280MC (ODE incidence 28.2 / 28.3 deg; 504 x 1024 px; ODE
Map_resolution ~167 m). WAC labels carry no geometry, so the WAC footprint is ODE's
four-corner polygon.

## Method (`scripts/register_iirs_wac.py`)

- IIRS enters as its PREP-06 band composite (`iirs_composite_plane`, one band selection per
  product).
- **Windows:** 5 per WAC product, 512 IIRS lines each, spread along the part of the strip
  inside the WAC footprint (by IIRS SYSTEM corners, `load_corner_model(corners="system")`,
  never the reference-fitted grid).
- **Prior:** IIRS system corners give IIRS pixel -> lat/lon. WAC pixel -> lat/lon comes from
  ODE's polygon, whose vertex order relative to the image axes is unknown: all **8**
  assignments of image corners to polygon corners (4 rotations x 2 mirrorings) are tried.
- **Coarse lock:** `cascade.register_step_dense` (MIND) of the IIRS window, block-averaged to
  WAC's pixel size, over the WHOLE WAC image, once per orientation. The orientation with
  the highest z is kept; it is accepted if z >= 10 AND at least 2 units above the
  second-best orientation (otherwise the orientation is ambiguous: no lock).
- **Fine stage:** IIRS composite warped (anti-aliased) onto the WAC grid; routed matcher
  (xoftr tiled 640) plus, on identical windows, `minima-loftr`, `rift2` and `sift`;
  `pipeline.fine_stage` with the configured defaults; all five control gates; tier.

## Success (per window, per matcher)

Coarse lock accepted; fine model returned; all five gates pass (incl. the known (3, 4) px
shift within 1.5 px); tier >= LOW; and the final transform's implied IIRS offset within
**1 WAC pixel-diagonal (~240 m)** of the median over that product's gate-passing windows
(>= 3 required). The consistency bound is looser than OHRC's 150 m because a WAC pixel is
~167 m: a correct lock can only be located to about a pixel here.

Verdicts: solved >= 90%, degraded >= 60%, else unsolved.

## Predictions

1. The routed matcher (xoftr) is solved on at least one product (it is the only candidate
   measured 10/10 on synthetic cross-modal data).
2. `sift` below xoftr (the published pairing used SIFT, but on hand-picked pairs).
No prediction for `rift2` or `minima-loftr`.

## Limits

One IIRS scene, two WAC products, 5 windows each. The known-shift gate measures
precision of the lock on this pair, not absolute accuracy.
