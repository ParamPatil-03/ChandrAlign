"""The adopted IIRS -> LRO WAC (global mosaic) registration, one window at a time.

Moved VERBATIM from scripts/register_iirs_wac.py (audit 2026-09-26 C-01 follow-up) so the product
runs the same code as the committed evidence (reports/iirs_wac_mosaic.json: xoftr 5/5 HIGH);
the script is now a thin wrapper. Its DUMP_DIR global became the `dump_dir=` argument.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..estimate import scale
from ..estimate.scale import PixelScale
from ..evaluate import control_gates, quality
from ..evaluate.source_px import jacobian_from_transform, to_source_px
from ..geometry.nac import enu
from ..io import pds_raster
from ..io.dem import dem_patch
from ..matching import cascade
from ..matching.similarity import alignment_check
from ..pipeline import fine_stage
from ..preprocess.iirs_composite import iirs_composite_plane
from ..preprocess.resample import warp_affine
from .ohrc_nac import T, affine_fit, match, norm

# ---- frozen in docs/iirs_wac_protocol.md -------------------------------------------
PRODUCTS = ("M106705467MC", "M106698280MC")
LINES = 512
N_WIN = 5
MIN_Z = float(config.get("cascade.min_z", 10.0))
Z_MARGIN = 2.0
CONSISTENT_M = 240.0
WAC_FILL = -1e30


class WacGeo:
    """WAC pixel (x, y) <-> lon/lat from ODE's 4-corner polygon under one of 8 orientations."""

    def __init__(self, poly_lonlat, w, h, k, mirror):
        img = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
        v = np.array(poly_lonlat[:4], float)
        v = v[::-1] if mirror else v
        v = np.roll(v, k, axis=0)
        self.lat0, self.lon0 = float(v[:, 1].mean()), float(v[:, 0].mean())
        e = np.array([enu(la, lo, self.lat0, self.lon0) for lo, la in v])
        self.fwd = affine_fit(img, e)                       # px -> east/north m
        self.inv = np.linalg.inv(self.fwd)
        self.name = f"rot{k}{'m' if mirror else ''}"
        J = self.fwd[:2, :2]
        self.px_x, self.px_y = float(np.linalg.norm(J[:, 0])), float(np.linalg.norm(J[:, 1]))

    def to_px(self, lat, lon):
        e = np.array([enu(a, b, self.lat0, self.lon0) for a, b in zip(np.atleast_1d(lat), np.atleast_1d(lon))])
        p = np.c_[e, np.ones(len(e))] @ self.inv.T
        return p[:, 0], p[:, 1]

    def to_latlon(self, x, y):
        e = self.fwd @ np.array([x, y, 1.0])
        k = np.pi / 180 * 1737400.0
        return self.lat0 + e[1] / k, self.lon0 + e[0] / (k * np.cos(np.radians(self.lat0)))

    def pixel_to_latlon(self, rows, cols):
        ll = np.array([self.to_latlon(c, r) for r, c in zip(np.ravel(rows), np.ravel(cols))])
        return ll[:, 0], ll[:, 1]


class MosaicGeo:
    """The map-projected WAC mosaic clip: simple cylindrical, north up (amendment 1)."""

    def __init__(self, meta):
        self.m = meta
        self.name = "mosaic"
        self.lat_top, self.lon_left, self.dpp = meta["pixel_centre_lat_of_row0"], meta["pixel_centre_lon_of_col0"], meta["deg_per_px"]
        lat_c = self.lat_top - meta["shape"][0] / 2 * self.dpp
        self.px_x, self.px_y = 100.0 * float(np.cos(np.radians(lat_c))), 100.0

    def to_px(self, lat, lon):
        return (np.asarray(lon, float) - self.lon_left) / self.dpp, (self.lat_top - np.asarray(lat, float)) / self.dpp

    def to_latlon(self, x, y):
        return self.lat_top - y * self.dpp, self.lon_left + x * self.dpp

    def pixel_to_latlon(self, rows, cols):
        return self.lat_top - np.asarray(rows, float) * self.dpp, self.lon_left + np.asarray(cols, float) * self.dpp


class Offset:
    def __init__(self, geo, ox, oy):
        self.geo, self.ox, self.oy = geo, ox, oy

    def pixel_to_latlon(self, rows, cols):
        return self.geo.pixel_to_latlon(np.asarray(rows, float) + self.oy, np.asarray(cols, float) + self.ox)


def polygon_of(record):
    wkt = record.get("Footprint_C0_geometry") or record.get("Footprint_geometry")
    pts = [tuple(float(v) for v in p.split()) for p in re.findall(r"-?\d+\.?\d*\s+-?\d+\.?\d*", wkt)]
    return pts[:-1] if pts[0] == pts[-1] else pts


def run_window(iirs, im, sel, wac, wimg, wok, geos, r0, matchers, device, stages, dem_tiles, route_opts, *,
               dump_dir=None, capture=None):
    """One IIRS window registered to LRO WAC (the adopted IIRS -> WAC path, docs/iirs_wac_protocol.md).

    Returns the evidence record scripts/register_iirs_wac.py writes; `capture` (a dict) also receives
    each matcher's registration for the product layer.
    """
    out = {"iirs_line0": int(r0)}
    S = iirs.array_shape[1]
    plane = iirs_composite_plane(iirs, pds_raster.Window(int(r0), 0, LINES, S), sel)
    src, src_ok = plane.array.astype(np.float32), plane.valid_mask
    if src_ok.mean() < 0.9:
        out["status"] = "skipped: IIRS window has invalid pixels"; return out
    g = np.linspace(0, LINES - 1, 7); gs = np.linspace(0, S - 1, 5)
    gx, gy = np.meshgrid(gs, g)
    lat, lon = im.pixel_to_latlon(r0 + gy.ravel(), gx.ravel())
    lat_c, lon_c = (float(v) for v in im.pixel_to_latlon(np.array([r0 + LINES / 2]), np.array([S / 2])))
    iirs_px = float(np.median([v for v in (iirs.gsd_m,) if v]))
    tries = []
    for geo in geos:                                           # 8 orientations of ODE's polygon
        x, y = geo.to_px(lat, lon)
        A = affine_fit(np.c_[gx.ravel(), gy.ravel()], np.c_[x, y])
        f = max(1, round(max(geo.px_x, geo.px_y) / iirs_px))
        prior = A @ np.linalg.inv(cascade.downsample_transform(f))
        diag = {}
        st = cascade.register_step_dense(src, wimg, f, src="IIRS", ref=wac.product_id,
                                         ref_pixel_m=max(geo.px_x, geo.px_y), prior=prior, diag=diag)
        tries.append((float(diag.get("z") or 0.0), geo, st))
    tries.sort(key=lambda t: -t[0])
    (z1, geo, st), z2 = tries[0], (tries[1][0] if len(tries) > 1 else float("-inf"))
    out["coarse"] = {"best": geo.name, "z": round(z1, 1), "runner_up_z": None if len(tries) == 1 else round(z2, 1)}
    if st is None or z1 < MIN_Z or (len(tries) > 1 and z1 - z2 < Z_MARGIN):
        out["status"] = "no coarse lock"; out["results"] = {}; return out
    T_c = np.asarray(st.model.matrix, float)                   # IIRS window px -> WAC px
    h, w = wimg.shape
    cor = np.array([[0, 0, 1], [S, 0, 1], [0, LINES, 1], [S, LINES, 1]], float) @ T_c.T
    o = np.maximum(np.floor(cor[:, :2].min(0)).astype(int) + 2, 0)
    e_ = np.minimum(np.ceil(cor[:, :2].max(0)).astype(int) - 2, [w, h])
    wF, hF = int(e_[0] - o[0]), int(e_[1] - o[1])
    if wF < 48 or hF < 48:
        out["status"] = "skipped: fine frame off the WAC image"; out["results"] = {}; return out
    Wf = T(-o[0], -o[1]) @ T_c
    fsrc = warp_affine(src, Wf, (wF, hF))
    fref = wimg[o[1]:o[1] + hF, o[0]:o[0] + wF].copy()
    fsrc_ok = cv2.warpAffine(src_ok.astype(np.float32), Wf[:2], (wF, hF), flags=cv2.INTER_NEAREST) > 0.5
    fref_ok = wok[o[1]:o[1] + hF, o[0]:o[0] + wF].copy()
    out["fine_frame_px"] = [wF, hF]
    dem = None
    if stages["geometry_filter"] and dem_tiles:
        dem = dem_patch(dem_tiles, (lat_c - 0.4, lat_c + 0.4, lon_c - 0.4, lon_c + 0.4))
    mapped = isinstance(geo, MosaicGeo)                       # a map grid is exact by construction
    exp = scale.expected_scale(scale.pixel_scale(iirs),
                               PixelScale(wac.product_id, geo.px_x, geo.px_y, (geo.px_x,) * 2, (geo.px_y,) * 2,
                                          ("map_projection",) if mapped else ("ode_footprint",), mapped,
                                          ("map-projected mosaic grid",) if mapped else ("from ODE's footprint polygon",)))
    gsd = max(geo.px_x, geo.px_y)
    results = {}
    for name in matchers:
        t0 = time.perf_counter(); r = {}
        opts = route_opts if name == "xoftr" else {}
        try:
            if name == "rift2":
                from chandralign import synth
                from chandralign.matching import rift
                mk = lambda a: synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool),  # noqa: E731
                                                shadow_mask=np.zeros(a.shape, bool), gsd_m=gsd, meta=None, geo=None)
                pa, pb = mk(fsrc), mk(fref); ms = rift.match(pa, pb)
            elif opts:
                from chandralign import synth
                from chandralign.matching import adapter
                mk = lambda a: synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool),  # noqa: E731
                                                shadow_mask=np.zeros(a.shape, bool), gsd_m=gsd, meta=None, geo=None)
                pa, pb = mk(fsrc), mk(fref); ms = adapter.match(pa, pb, model_name=name, device=device, **opts)
            else:
                ms, pa, pb = match(name, fsrc, fref, device, gsd)
            n = int(len(ms.src_pts))
            if n < 4:
                results[name] = {"status": f"only {n} matches", "success": False}; continue
            fr = fine_stage(ms, fsrc, fref, centre=(wF / 2, hF / 2), flags=stages,
                            ground_model=Offset(geo, o[0], o[1]), dem=dem)
            r.update(matches=n, inliers=fr.inlier_count, inlier_ratio=round(fr.inlier_ratio, 4),
                     coverage=round(fr.coverage, 3), inlier_rmse_px=None if fr.rmse_px is None else round(fr.rmse_px, 3))
            if not fr.ok:
                results[name] = {**r, "status": "fine stage: no transform", "success": False}; continue
            if dump_dir is not None and name == "xoftr":
                np.savez_compressed(Path(dump_dir) / f"window_{int(r0)}.npz", src_img=fsrc, ref_img=fref,
                                    src_ok=fsrc_ok, ref_ok=fref_ok, model=np.asarray(fr.model.matrix, float))
            Tt = T(o[0], o[1]) @ np.asarray(fr.model.matrix, float) @ Wf
            fx, fy = (Tt @ [S / 2, LINES / 2, 1])[:2]
            la_f, lo_f = geo.to_latlon(fx, fy)
            ef = enu(la_f, lo_f, lat_c, lon_c)
            verdict = scale.check(Tt, exp, centre=(S / 2, LINES / 2))
            gates = control_gates.run_all(control_gates.pipeline_from(name, device=device, gsd_m=gsd, stages=stages, **opts),
                                          fsrc, fref, pa, pb)
            q = quality.assess(inlier_count=fr.inlier_count, inlier_ratio=fr.inlier_ratio, spatial_coverage=fr.coverage,
                               model=fr.model, scale_ok=verdict.ok, scale_status=verdict.status,
                               gates=gates.gates, require_gates=True)
            r.update(status="registered", tier=q.tier, gates=gates.gates, scale_status=verdict.status,
                     known_shift_error_px=gates.to_dict().get("perturbation_sensitivity", {}).get("error_px"),
                     gates_pass=bool(gates.all_passed), tier_ok=q.tier in ("HIGH", "MEDIUM", "LOW"),
                     implied_offset_m={"east": round(float(ef[0]), 1), "north": round(float(ef[1]), 1)})
            J_src = jacobian_from_transform(Wf)                      # IIRS px -> fine-frame (WAC) px
            r["source_px"] = {"known_shift": to_source_px(r["known_shift_error_px"], J_src, "exact (Wf)"),
                              "inlier_rmse": to_source_px(r["inlier_rmse_px"], J_src, "exact (Wf)")}
            mi = alignment_check(fsrc, fref, np.asarray(fr.model.matrix, float), src_ok=fsrc_ok, ref_ok=fref_ok)
            if mi.get("peak_offset_px") is not None:
                mi["peak_offset_src_px"] = to_source_px(float(np.hypot(*mi["peak_offset_px"])), J_src, "exact (Wf)")
            r["mi_check"] = mi                                     # MATCH-07 (docs/mi_protocol.md)
            if capture is not None:
                capture[name] = dict(src=fsrc, ref=fref, src_ok=fsrc_ok, ref_ok=fref_ok, fine=fr, gates=gates,
                                     quality=q, scale=verdict, Wf=Wf, origin=(int(o[0]), int(o[1])), T_total=Tt,
                                     gsd=gsd, geo=geo, window_origin=(int(r0), 0), mi=mi)
        except Exception as exc:                                  # recorded, never hidden
            r.update(status=f"error: {type(exc).__name__}: {exc}"[:300], success=False)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = r
    out["results"] = results
    out["status"] = "locked"
    return out




def load_mosaic(root=None):
    """The map-projected WAC mosaic clip (amendment 1): (meta, image 0..1, valid mask, MosaicGeo)."""
    mdir = Path(root or config.ROOT) / "data/raw/lro/wac_mosaic"
    mmeta = json.loads((mdir / "wac_mosaic_100m_clip.json").read_text(encoding="utf-8"))
    raw = np.load(mdir / "wac_mosaic_100m_clip.npy").astype(np.float32)
    wok = raw > mmeta["nodata"]
    wimg = norm(raw, wok)
    return mmeta, wimg, wok, MosaicGeo(mmeta)


def mosaic_windows(iirs, im, geo, wimg, n_win=N_WIN):
    """IIRS window start lines safely inside the mosaic clip (amendment 3), spread along-track."""
    L, S = iirs.array_shape
    rows = np.arange(0, L - LINES, 64)
    lat_all, lon_all = im.pixel_to_latlon(rows + LINES / 2, np.full(rows.shape, S / 2))
    H, W = wimg.shape
    la_top, lo_l = geo.to_latlon(0, 0); la_bot, lo_r = geo.to_latlon(W - 1, H - 1)  # noqa: E702
    M = 1.2            # amendment 3: half a window (0.34 deg) + IIRS system error (0.42 deg) + slack
    box_ok = [(la_bot + M < la < la_top - M) and (lo_l + 0.3 < lo < lo_r - 0.3) for la, lo in zip(lat_all, lon_all)]
    inside = rows[box_ok]
    if len(inside) == 0:
        return []
    picks = np.linspace(inside.min(), inside.max(), n_win).astype(int)
    return [int(inside[np.argmin(np.abs(inside - p))]) for p in picks]
