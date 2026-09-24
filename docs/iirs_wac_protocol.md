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

## Run 1 result (2026-09-24): no coarse lock on any window

10/10 windows (two WAC products): best-orientation z 8.9-10.6, runner-up within 0.0-1.3 ->
no window passes (z >= 10 AND >= 2 above the runner-up). No fine stage ran.
`reports/iirs_wac_registration.json`.

**Not yet interpreted as a matching result.** Hypothesis, NOT tested: the WAC products used
are raw CDR frames (not map-projected) from a ~90 deg field-of-view camera, so the pixel
scale varies strongly across the 258 km frame; the prior fits ONE affine to ODE's four
footprint corners, which gives the wrong local scale/shape at each window, and MIND cannot
match a template of the wrong scale. The published IIRS <-> WAC result is understood to use
map-projected WAC. Test: repeat with the map-projected LROC WAC Global Morphologic Mosaic
(100 m/px, 643 nm) clipped to this area -- a download, pending approval.

## Amendment 1 (approved download; before any mosaic measurement)

Reference changed to test the run-1 hypothesis: the **LROC WAC Global Morphologic Mosaic**
(643 nm, 100 m/px, simple cylindrical, north up; USGS/ASC
`Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif`), a clip 4 S - 4 N, 22.3 - 25.3 E
read by byte ranges (`data/raw/lro/wac_mosaic/wac_mosaic_100m_clip.{npy,json}`, SHA-256 in
the sidecar). The geometry is exact, so there is ONE orientation: the "2 above the
runner-up" condition no longer applies (lock = z >= 10). Windows: 5, spread along the IIRS
strip inside the clip (0.3 deg margin). Consistency bound 240 m (~2.4 mosaic pixels) is
unchanged. Everything else unchanged. Run 1 (raw CDR frames) stays recorded.

## Mosaic run 1 result, and amendment 2

Run 1 on the mosaic: 4/5 windows lock (z 50-64) and xoftr and sift register 4/5, minima-loftr
and rift2 3/5 (`reports/iirs_wac_mosaic_run1.json`). **The run-1 hypothesis is confirmed:
the raw WAC frames were the problem.** The locked windows measure IIRS's system
geolocation error at ~12.8 km north, ~1.2-1.5 km east, consistent along the strip.

Two harness errors, fixed for run 2 (all 5 windows re-run; run 1 kept for the record):
1. The one unlocked window (z 7.1) is the northernmost: windows were kept 0.3 deg (~9 km)
   inside the clip by IIRS's SYSTEM position, less than its measured 12.8 km error, so its
   true ground fell outside the clip. Margin -> 0.6 deg (~18 km).
2. The mosaic's pixel size was passed as `unverified`; a map-projected grid is exact by
   construction and is treated as verified everywhere else (`tc_pixel_scale`). This only
   affects the reported tier, never success (tier >= LOW either way).

## Mosaic run 2 result, and amendment 3 (placement only)

Run 2: the same 4 windows register (xoftr now HIGH with the map grid verified); the
northern window again fails to lock (z 7.2) -- because a window is 512 IIRS lines
(~0.67 deg tall): half of it (0.34 deg) plus the 0.42 deg system error exceeds a 0.6 deg
margin measured from the window CENTRE. Margin -> 1.2 deg (half window + error + slack).
Run 3 is the result; runs 1-2 kept (`reports/iirs_wac_mosaic_run{1,2}.json`).
