"""RPC00B export of the ALIGN-08 terrain model (docs/rpc_export_protocol.md).

An RPC (rational polynomial coefficients, GDAL RFC 22) maps a ground point (longitude,
latitude, height) to image (sample, line) and is how GIS tools orthorectify satellite images
with a DEM (`gdalwarp -rpc -to RPC_DEM=dem.tif`). Once the reference is a lat/lon map grid
(SELENE TC is simple cylindrical), the parallax model ref = A.src + (h - h0).p inverts to

    src = A^-1 (G^-1 [lon, lat] - (h - h0) p)

which is affine in (lon, lat, h): a FIRST-ORDER RPC with unit denominators, so the export is
exact, not a fit. Pixel convention: RPC line/sample are relative to the pixel CENTRE (RFC 22),
the same as numpy/cv2 here (centres at integers). h is the height AT THE GROUND POINT, which is
what an RPC means and, since docs/parallax_height_protocol.md, what the ALIGN-08 stage fits
(parallax.height_at: ref). GDAL's RPC transformer assumes EPSG:4326 and refuses a lunar CRS
(GDAL 3.12.4), so this export works in GDAL only with the Moon's lat/lon labelled EPSG:4326
(numbers unchanged; verified exact to 1e-4 px): an optional export, not the default delivery.
"""
from __future__ import annotations

import numpy as np

# RPC00B term order (GDAL gdal_rpc.cpp / RFC 22): L = longitude, P = latitude, H = height
TERMS = ("1", "L", "P", "H", "LP", "LH", "PH", "LL", "PP", "HH",
         "PLH", "LLL", "LPP", "LHH", "LLP", "PPP", "PHH", "LLH", "PPH", "HHH")


def terms(L, P, H) -> np.ndarray:
    L, P, H = (np.asarray(v, float) for v in (L, P, H))
    one = np.ones_like(L)
    return np.stack([one, L, P, H, L * P, L * H, P * H, L * L, P * P, H * H, P * L * H, L ** 3, L * P * P,
                     L * H * H, L * L * P, P ** 3, P * H * H, L * L * H, P * P * H, H ** 3])


def from_parallax(matrix, p_px_per_m, h0_m, ref_px_to_lonlat, src_shape, lonlat_box, h_range) -> dict:
    """RPC00B dict for a parallax model whose REFERENCE pixels are a lat/lon grid.

    matrix: 3x3 affine part (src px -> ref px); ref_px_to_lonlat: 3x3 affine (ref px -> lon, lat);
    src_shape: (lines, samples); lonlat_box: (lon_min, lon_max, lat_min, lat_max); h_range: (min, max) m.
    """
    Ainv = np.linalg.inv(np.asarray(matrix, float))
    Gi = np.linalg.inv(np.asarray(ref_px_to_lonlat, float))
    p = np.asarray(p_px_per_m, float)
    lines, samps = src_shape
    lon0, lat0 = (lonlat_box[0] + lonlat_box[1]) / 2, (lonlat_box[2] + lonlat_box[3]) / 2
    lons, lats = max((lonlat_box[1] - lonlat_box[0]) / 2, 1e-9), max((lonlat_box[3] - lonlat_box[2]) / 2, 1e-9)
    hoff, hs = float(h0_m), max(abs(h_range[0] - h0_m), abs(h_range[1] - h0_m), 1.0)
    off = {"LINE_OFF": (lines - 1) / 2, "SAMP_OFF": (samps - 1) / 2, "LAT_OFF": lat0, "LONG_OFF": lon0,
           "HEIGHT_OFF": hoff, "LINE_SCALE": max(lines / 2, 1.0), "SAMP_SCALE": max(samps / 2, 1.0),
           "LAT_SCALE": lats, "LONG_SCALE": lons, "HEIGHT_SCALE": hs}

    def src_of(L, P, H):
        lon, lat, h = lon0 + lons * L, lat0 + lats * P, hoff + hs * H
        r = Gi @ np.array([lon, lat, 1.0])
        return (Ainv @ np.array([r[0] - (h - h0_m) * p[0], r[1] - (h - h0_m) * p[1], 1.0]))[:2]

    c0 = src_of(0, 0, 0)
    grads = [src_of(*e) - c0 for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]    # exact: the map is affine
    rpc = dict(off)
    for k, axis, o, sc in (("SAMP", 0, off["SAMP_OFF"], off["SAMP_SCALE"]), ("LINE", 1, off["LINE_OFF"], off["LINE_SCALE"])):
        num = np.zeros(20)
        num[0] = (c0[axis] - o) / sc
        num[1:4] = [g[axis] / sc for g in grads]
        den = np.zeros(20); den[0] = 1.0
        rpc[f"{k}_NUM_COEFF"], rpc[f"{k}_DEN_COEFF"] = num.tolist(), den.tolist()
    return rpc


def evaluate(rpc: dict, lon, lat, h) -> tuple[np.ndarray, np.ndarray]:
    """(sample, line) of ground points, by the RPC00B formula."""
    L = (np.asarray(lon, float) - rpc["LONG_OFF"]) / rpc["LONG_SCALE"]
    P = (np.asarray(lat, float) - rpc["LAT_OFF"]) / rpc["LAT_SCALE"]
    H = (np.asarray(h, float) - rpc["HEIGHT_OFF"]) / rpc["HEIGHT_SCALE"]
    t = terms(L, P, H)
    ratio = lambda k: np.tensordot(rpc[f"{k}_NUM_COEFF"], t, 1) / np.tensordot(rpc[f"{k}_DEN_COEFF"], t, 1)  # noqa: E731
    return rpc["SAMP_OFF"] + rpc["SAMP_SCALE"] * ratio("SAMP"), rpc["LINE_OFF"] + rpc["LINE_SCALE"] * ratio("LINE")


def to_rasterio(rpc: dict):
    """rasterio.rpc.RPC for GDAL (optional dependency: rasterio)."""
    from rasterio.rpc import RPC
    return RPC(height_off=rpc["HEIGHT_OFF"], height_scale=rpc["HEIGHT_SCALE"], lat_off=rpc["LAT_OFF"],
               lat_scale=rpc["LAT_SCALE"], long_off=rpc["LONG_OFF"], long_scale=rpc["LONG_SCALE"],
               line_off=rpc["LINE_OFF"], line_scale=rpc["LINE_SCALE"], samp_off=rpc["SAMP_OFF"],
               samp_scale=rpc["SAMP_SCALE"], line_num_coeff=rpc["LINE_NUM_COEFF"],
               line_den_coeff=rpc["LINE_DEN_COEFF"], samp_num_coeff=rpc["SAMP_NUM_COEFF"],
               samp_den_coeff=rpc["SAMP_DEN_COEFF"])
