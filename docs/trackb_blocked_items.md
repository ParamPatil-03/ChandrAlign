# Track B: items that cannot be finished with the data held (audit 2026-09-26)

Each item below is scoped to its first concrete step, names what blocks it, and says how its result would be
measured. None of them is claimed as done.

## G-04: independent-truth check points for every pairing

**Blocked by:** data. Every accuracy figure today (probes, MI, check points) sees the same two images, so it
can share a shading bias with the matcher. Truth needs points located by something that shares nothing with
them.

**Plan:**
1. OHRC -> NAC: download the LROC NAC DTM and orthophoto pairs (LROC RDR `NAC_DTM_*`) that overlap the five
   held NACs. An ortho tie point (a crater rim or boulder located in the ORTHO and in OHRC by a person, or
   by a crater-centroid fit run independently in each image) is independent of the NAC EDR's shading. Target:
   30-50 points per NAC.
2. TMC-2 -> TC: crater centroids fitted separately in TMC-2 and in the TC ortho tile (`matching/craters.py`
   already fits blobs; use ellipse fits on rims >= 20 px), paired by the delivered geometry within 3 px and
   then scored on their own centroids.
3. Report the section 7 table again: p50/p95 in SOURCE px against those points, per pairing.

**Effort:** 2-3 days including downloads and a manual QA pass on the tie points.

## G-05 part 2: rendered references for opposite and overhead sun

**Blocked by:** data. Rendering NAC under OHRC's sun needs a DTM near OHRC resolution. Held: LOLA and SLDEM2015
(~60 m) and the SELENE TC DTM (~10 m, not over the OHRC scenes). None of these can shade a 0.3 m image.

**Plan:** with the LROC NAC DTMs from G-04 (2 m posts): hillshade under OHRC's sun azimuth and incidence
(`geometry/solar.py` has both), blend with NAC albedo (NAC / its own Lambertian shading), and register OHRC
to the rendered image. Measure MI flags and probe p95 in OHRC px on M175124932LC (opposed sun) and
M1417360906LC (75 deg incidence), the two unsolved cases.

## G-10: rigorous sensor model plus DEM bundle adjustment (the audit's "longer term")

**Blocked by:** scope (weeks), not data.

**Plan, in steps each worth doing alone:**
1. RPC bias compensation per TMC-2 strip: fit an affine correction in image space to the ISRO RPCs /
   corner model using the TMC-2 -> TC tie points already produced, with heights from the TC DTM. This
   generalises ALIGN-08 from per-window to per-strip.
2. Pushbroom attitude polynomials (roll/pitch/yaw vs line, degree 2-3) solved over all windows of a strip at
   once, so windows share one physical model instead of 15 independent fits.
3. Joint adjustment across overlapping windows and pairings (TMC-2 <-> OHRC <-> NAC), with closure error as
   the check.

**Measure:** closure error across windows, probe error on the hilly N00 windows (today p95 1.05-1.47
TMC-2 px), and the offset versus the ISRO refined grid (today 239-636 m).

## Partial items (done as far as held data allows)

- **I-16 (fresh data):** fresh WINDOWS of the held products are scored (docs/fresh_windows_protocol.md).
  Fresh PRODUCTS (a second OHRC scene, a second TMC-2 orbit) still need downloading from PRADAN. IIRS -> WAC
  has no fresh window inside the held mosaic clip.
- **I-17 / G-08:** measured leave-one-out, because every held product is already in geoprior's tables. A
  product not yet registered is still needed for the literal test.
- **G-05 part 1:** done. MIND fallback, not adopted (docs/illumination_refinement_protocol.md).
