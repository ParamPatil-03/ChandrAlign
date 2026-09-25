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
