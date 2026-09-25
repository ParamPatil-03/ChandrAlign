# Protocol: deliver the TMC-2 terrain model as a standard RPC (frozen before running)

Question: can the ALIGN-08 terrain model (ref = A.src + (h - h0).p) be delivered in the standard
form GIS tools orthorectify with -- an RPC (RPC00B) plus a DEM, applied by GDAL -- on the Moon?

Why RPC: it is the industry-standard sensor model for satellite imagery (GDAL RFC 22; gdalwarp
-rpc -to RPC_DEM=...). Our model is linear in (lon, lat, h) once the reference is a lat/lon map
grid (TC is simple cylindrical), so it is EXACTLY a first-order RPC (denominators = 1): no fit error.
Open question the docs do not answer: GDAL's RPC transformer does not state the body/datum of
its lat/long, and the Moon is not WGS84.

Test (one real window, TMC-2 -> TC row 4687, TC DTM):
1. RPC from the model; evaluated by our own RPC00B code it must reproduce ParallaxModel.apply to
   <= 0.001 px (checks the conversion).
2. GDAL (rasterio 1.5.1 / GDAL 3.12.4) warps two coordinate rasters (x, y of every source pixel)
   through the RPC with RPC_DEM = the TC DTM (IAU_2015:30100) onto the reference grid. The result
   is GDAL's source map. Compared with models.parallax_source_map on interior pixels:
   **RMS <= 0.1 px and 99th percentile <= 0.25 px** -> RPC export adopted as the delivery format.
Otherwise: not adopted; the ParallaxModel + our remap stays the only delivery, and why is recorded.

## Amendment 1 (before running): compare like with like

GDAL RFC 22: RPC line/sample are relative to the pixel CENTRE, the same convention as ours (numpy/
cv2, centres at integers), so no half-pixel shift is expected. But an RPC maps a GROUND point
(lon, lat, h) to the image, so h is the height AT THE GROUND POINT, i.e. at the reference pixel.
The ALIGN-08 stage samples h at the SOURCE pixel's position in the (coarse-aligned) TC frame,
up to ~25 px along-track from the ground point. The two differ on slopes. So test 2 compares
GDAL's source map with the DIRECT formula s = A^-1 (r - (h(r) - h0) p), h(r) from the same DEM
(this isolates the question asked: does GDAL's RPC + lunar DEM work?). Same thresholds.
Reported separately, not decided here: how far the stage's convention (h at the source pixel,
`parallax_source_map`) is from the ground-height convention on this window.

## Result (2026-09-25): NOT adopted -- GDAL's RPC transformer is Earth-bound

`reports/rpc_export_check.json` (window 4687, GDAL 3.12.4 / rasterio 1.5.1).

- Test 1 passes: the first-order RPC reproduces the model to 7e-13 px (the export is exact).
- Test 2 cannot run: GDAL assumes RPC lat/long are EPSG:4326 (WGS84) whatever CRS is passed, and
  refuses the operation "EPSG:4326 -> Moon (2015)" ("Cannot find coordinate operations"). So a
  lunar RPC is not usable in GDAL as a lunar product. **Not adopted.**

Diagnostics (not part of the decision):
- Labelling the image, DEM and output all EPSG:4326 (numbers unchanged: an RPC is pure polynomial
  in lat/lon/h) makes GDAL orthorectify exactly as our formula: max 0.0001 px over 1,076,369 px.
  So an RPC export WORKS numerically in GDAL, but only by calling lunar coordinates terrestrial;
  offered as an optional export with that caveat, never as the default.
- **Height convention: 0.91 px RMS, 2.67 px p99, 3.93 px max** between the stage's convention
  (h sampled at the SOURCE pixel's position) and h at the ground point (the reference pixel),
  on this hilly window. The ground point is the physical one. Which predicts the images better
  is a separate, measurable question (next protocol).

Decision on delivery: the canonical terrain model stays models.ParallaxModel +
parallax_source_map (our code, now cross-checked against GDAL to 1e-4 px); geometry/rpc.py is
kept as an optional, caveated GIS export.
