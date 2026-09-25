"""Does GDAL orthorectify our TMC-2 terrain model, exported as an RPC, correctly on the Moon?

    python scripts/register_tmc2_tc.py --rows 4687 --dump-points <dir> --out <run.json>
    python scripts/verify_rpc_export.py <dir>/window_4687.npz <run.json>

Protocol, frozen before this ran: docs/rpc_export_protocol.md (with amendment 1).
Needs rasterio (GDAL). Writes reports/rpc_export_check.json.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from rasterio.warp import reproject  # noqa: E402

from chandralign.estimate import models  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.geometry import projection, rpc  # noqa: E402
from chandralign.io.dem import dem_patch, find_tiles  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402

MOON = CRS.from_user_input("IAU_2015:30100")
BORDER = 5


def main() -> int:
    z = np.load(sys.argv[1])
    w = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["rows"][0]
    par = w["pipeline"]["parallax"]
    A = np.vstack([np.asarray(par["affine"], float), [0.0, 0.0, 1.0]])
    p, h0 = np.asarray(par["p_px_per_m"], float), float(par["h0_m"])
    pm = models.ParallaxModel(matrix=A, p_px_per_m=tuple(p), h0_m=h0)
    tcm = projection.load_map_model(parse_label(ROOT / "data/raw/selene/tc" / f"{str(z['tc_product'])}.lbl"))
    o = np.asarray(z["tc_origin_px"], float)
    hF, wF = z["src_img"].shape

    def lonlat(x, y):
        lat, lon = tcm.pixel_to_latlon(np.asarray(y, float) + o[1], np.asarray(x, float) + o[0])
        return np.asarray(lon, float), np.asarray(lat, float)

    # reference px -> (lon, lat): TC is simple cylindrical, so this is affine; check it is
    q = np.array([[0, 0], [1, 0], [0, 1]], float)
    lo, la = lonlat(q[:, 0], q[:, 1])
    G = np.array([[lo[1] - lo[0], lo[2] - lo[0], lo[0]], [la[1] - la[0], la[2] - la[0], la[0]], [0, 0, 1.0]])
    lo4, la4 = lonlat([wF - 1.0], [hF - 1.0])
    lin_err = float(np.hypot(*(G @ [wF - 1.0, hF - 1.0, 1.0])[:2] - np.r_[lo4, la4]) * 30000.0)   # ~px
    corners = G @ np.array([[0, 0, 1], [wF, 0, 1], [0, hF, 1], [wF, hF, 1]], float).T
    box = (corners[0].min(), corners[0].max(), corners[1].min(), corners[1].max())
    dem = dem_patch(find_tiles(ROOT / "data/raw/selene/tc_dtm"), (box[2] - 0.01, box[3] + 0.01, box[0] - 0.01, box[1] + 0.01))
    hmin, hmax = float(np.nanmin(dem.heights_m)), float(np.nanmax(dem.heights_m))
    R = rpc.from_parallax(A, p, h0, G, (hF, wF), box, (hmin, hmax))

    # the ground-height formula: for every reference pixel r, s = A^-1 (r - (h(r) - h0) p)
    ys, xs = np.mgrid[0:hF, 0:wF].astype(float)
    glon, glat = (G @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]))[:2]
    gh = dem.sample(glat, glon)
    back = np.c_[xs.ravel() - (gh - h0) * p[0], ys.ravel() - (gh - h0) * p[1], np.ones(xs.size)] @ np.linalg.inv(A).T
    sx, sy = back[:, 0].reshape(hF, wF), back[:, 1].reshape(hF, wF)

    # test 1: our RPC00B evaluation reproduces the formula
    ex, ey = rpc.evaluate(R, glon, glat, gh)
    t1 = float(np.nanmax(np.hypot(ex - sx.ravel(), ey - sy.ravel())))

    # test 2: GDAL warps coordinate rasters through the RPC + lunar DEM. The protocol's test
    # uses the Moon CRS; labelling everything EPSG:4326 is run AFTER it as a diagnostic only
    # (an RPC is pure polynomial in lat/lon/h, so the label changes no number -- but it is a
    # workaround that asks users to call lunar coordinates terrestrial).
    res = 1.0 / dem.res_px_per_deg
    dst_tf = Affine(G[0, 0], G[0, 1], G[0, 2] - 0.5 * (G[0, 0] + G[0, 1]),
                    G[1, 0], G[1, 1], G[1, 2] - 0.5 * (G[1, 0] + G[1, 1]))        # pixel-corner convention
    inner0 = np.zeros((hF, wF), bool); inner0[BORDER:-BORDER, BORDER:-BORDER] = True
    inner0 &= (sx > 1) & (sx < wF - 2) & (sy > 1) & (sy < hF - 2)

    def gdal_map(crs):
        with tempfile.TemporaryDirectory() as td:
            dem_tif = Path(td) / "dem.tif"
            with rasterio.open(dem_tif, "w", driver="GTiff", height=dem.heights_m.shape[0],
                               width=dem.heights_m.shape[1], count=1, dtype="float32", crs=crs, nodata=np.nan,
                               transform=Affine(res, 0, dem.lon[0] - res / 2, 0, -res, dem.lat[0] + res / 2)) as ds:
                ds.write(dem.heights_m.astype(np.float32), 1)
            maps = []
            for src in (xs.astype(np.float32), ys.astype(np.float32)):
                dst = np.full((hF, wF), np.nan, np.float32)
                reproject(source=src, destination=dst, rpcs=rpc.to_rasterio(R), src_crs=crs, dst_crs=crs,
                          dst_transform=dst_tf, dst_nodata=np.nan, resampling=Resampling.bilinear,
                          RPC_DEM=str(dem_tif))
                maps.append(dst)
        gx, gy = maps
        inner = inner0 & np.isfinite(gx) & np.isfinite(gy)
        d = np.hypot(gx - sx, gy - sy)[inner]
        return {"pixels": int(inner.sum()), "rms_px": round(float(np.sqrt(np.mean(d ** 2))), 4),
                "p99_px": round(float(np.percentile(d, 99)), 4), "max_px": round(float(d.max()), 4),
                "mean_offset_px": [round(float(np.mean((gx - sx)[inner])), 4), round(float(np.mean((gy - sy)[inner])), 4)]}

    try:
        t2 = gdal_map(MOON)
    except Exception as exc:                          # recorded, never hidden
        t2 = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    try:
        t2_wgs84_label = gdal_map(CRS.from_epsg(4326))
    except Exception as exc:
        t2_wgs84_label = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    inner = inner0

    # diagnostic (not decided): the stage's convention (h at the source pixel) vs h at the ground point
    heights_at = lambda s: dem.sample(*lonlat(s[:, 0], s[:, 1])[::-1])  # noqa: E731
    mx, my = models.parallax_source_map(pm, heights_at, (hF, wF), step=4)
    dd = np.hypot(mx - sx, my - sy)[inner]
    diag = {"rms_px": round(float(np.sqrt(np.mean(dd ** 2))), 4), "p99_px": round(float(np.percentile(dd, 99)), 4),
            "max_px": round(float(dd.max()), 4)}

    verdict = {"t1_rpc_formula_max_px": t1, "t1_pass": t1 <= 1e-3, "t2_gdal_vs_formula": t2,
               "t2_pass": "rms_px" in t2 and t2["rms_px"] <= 0.1 and t2["p99_px"] <= 0.25}
    verdict["adopt"] = bool(verdict["t1_pass"] and verdict["t2_pass"])
    outp = {"source": "measured", "protocol": "docs/rpc_export_protocol.md", "run": run_record(),
            "gdal": rasterio.__gdal_version__, "rasterio": rasterio.__version__, "window": w["tmc_row"],
            "reference_grid_linearity_px": round(lin_err, 6), "dem": "TC DTM " + ",".join(dem.tiles),
            "verdict": verdict, "diagnostic_gdal_with_epsg4326_label": t2_wgs84_label,
            "diagnostic_stage_convention_vs_ground_height": diag, "rpc": R}
    (ROOT / "reports/rpc_export_check.json").write_text(json.dumps(outp, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in outp.items() if k not in ("rpc", "run")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
