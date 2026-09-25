"""Any Chandrayaan-2 instrument -> a map-projected reference (LRO WAC mosaic, SELENE MI tile).

    python scripts/register_to_map.py --source tmc2 --reference mi
    python scripts/register_to_map.py --source ohrc --reference wac

Protocol, frozen before this script existed: docs/map_pairings_protocol.md. One engine for all
pairings: the FINER image is the template of a MIND dense lock at the coarser pixel size; the fine
stage then works in the COARSER image's grid (the information limit), followed by
pipeline.fine_stage, all five control gates, the tier, the source-pixel conversion and the MATCH-07
MI check. Priors are SYSTEM corners plus our measured system offsets; position is searched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from chandralign import config  # noqa: E402
from chandralign.estimate import scale  # noqa: E402
from chandralign.estimate.scale import PixelScale  # noqa: E402
from chandralign.evaluate import control_gates, quality  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.evaluate.source_px import jacobian_from_transform, to_source_px  # noqa: E402
from chandralign.geometry import geoprior, projection  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.dem import dem_patch, find_tiles  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.matching import cascade, routing  # noqa: E402
from chandralign.matching.similarity import alignment_check  # noqa: E402
from chandralign.pipeline import fine_stage, stage_flags  # noqa: E402
from chandralign.preprocess.iirs_composite import iirs_composite_plane, product_band_selection  # noqa: E402
from chandralign.preprocess.resample import warp_affine  # noqa: E402
from register_ohrc_nac import T, affine_fit, match, norm  # noqa: E402
from register_tmc2_nac import MOON_R_M, enu  # noqa: E402

# ---- frozen in docs/map_pairings_protocol.md ----------------------------------------
N_WIN = 5
WIN = {"ohrc": 8192, "tmc2": 1536, "iirs": 256}
WIN_WAC = {"tmc2": 4000}
MI_BAND_NM = {"ohrc": 749.0, "tmc2": 749.0, "iirs": 1548.0}
MATCHERS = {"ohrc": ["eloftr", "sift"], "tmc2": ["eloftr", "sift"], "iirs": ["xoftr", "sift"]}   # the frozen default
REF_INSTRUMENT = {"wac": "WAC", "mi": "MI", "tc": "TC"}
MARGIN_M, MARGIN_PX = 3000.0, 30
MIN_Z = float(config.get("cascade.min_z", 10.0))
K = np.pi / 180 * MOON_R_M


class MapRef:
    """A lat/lon grid reference: array (normalised), valid mask, pixel <-> lat/lon."""

    def __init__(self, name, arr, ok, lat0, lon0, dpp, kind="", meta=None):
        self.name, self.arr, self.ok, self.lat0, self.lon0, self.dpp = name, arr, ok, lat0, lon0, dpp
        self.kind, self.meta = kind, meta        # meta: the tile's SceneMeta where a label is held (routing, I-09)
        lat_c = lat0 - arr.shape[0] / 2 * dpp
        self.px_m = (dpp * K * np.cos(np.radians(lat_c)), dpp * K)             # (x, y) metres

    def pixel_to_latlon(self, rows, cols):
        return self.lat0 - np.asarray(rows, float) * self.dpp, self.lon0 + np.asarray(cols, float) * self.dpp

    def latlon_to_pixel(self, lat, lon):
        return (self.lat0 - np.asarray(lat, float)) / self.dpp, (np.asarray(lon, float) - self.lon0) / self.dpp

    def pixel_scale(self):
        x, y = self.px_m
        return PixelScale(self.name, x, y, (x, x), (y, y), ("map_projection",), True, ("map-projected grid",))


def load_ref(kind, source, tile=None):
    if kind == "tc":                                   # SELENE TC ortho map tile (docs/tc_reference_protocol.md)
        meta = parse_label(ROOT / "data/raw/selene/tc" / f"TCO_MAP_02_{tile}.lbl")
        raw = pds_raster.read_raster(meta).astype(np.float32)
        ok = raw > 0
        mm = projection.load_map_model(meta)
        la, lo = mm.pixel_to_latlon([0, 1], [0, 1])
        return MapRef(meta.product_id, norm(raw, ok), ok, float(la[0]), float(lo[0]), float(la[0] - la[1]), "tc", meta)
    if kind == "wac":
        d = ROOT / "data/raw/lro/wac_mosaic"
        m = json.loads((d / "wac_mosaic_100m_clip.json").read_text(encoding="utf-8"))
        raw = np.load(d / "wac_mosaic_100m_clip.npy").astype(np.float32)
        ok = raw > m["nodata"]
        return MapRef("WAC_GLOBAL_MOSAIC_100M", norm(raw, ok), ok, m["pixel_centre_lat_of_row0"],
                      m["pixel_centre_lon_of_col0"], m["deg_per_px"], "wac")
    lbl = ROOT / "data/raw/selene/mi/MI_MAP_03_N01E023N00E024SC.lbl"
    meta = parse_label(lbl)
    waves = [414.0, 749.0, 901.0, 950.0, 1001.0, 1000.0, 1049.0, 1248.0, 1548.0]   # label CENTER_FILTER_WAVELENGTH
    band = int(np.argmin(np.abs(np.array(waves) - MI_BAND_NM[source])))
    raw = pds_raster.read_raster(meta, None, bands=band).astype(np.float32)
    ok = raw > 0
    mm = projection.load_map_model(meta)
    la, lo = mm.pixel_to_latlon([0, 1], [0, 1])
    return MapRef(f"{meta.product_id}_{waves[band]:.0f}nm", norm(raw, ok), ok, float(la[0]), float(lo[0]),
                  float(la[0] - la[1]), "mi", meta)


class Source:
    """A CH-2 product: window reader, system ground model, known system offset (east, north m)."""

    def __init__(self, kind):
        self.kind = kind
        self.meta = parse_label(next((ROOT / f"data/raw/ch2/{kind}").rglob("*_d_img_d18.xml")))
        self.sysm = (projection.load_grid_model(self.meta) if kind == "ohrc"
                     else projection.load_corner_model(self.meta, corners="system"))
        assert self.sysm.independent_of_references
        self.L, self.S = self.meta.array_shape[-2], self.meta.array_shape[-1]
        ps = scale.pixel_scale(self.meta)
        self.px_m = (ps.across_m, ps.along_m)
        self.sel = product_band_selection(self.meta) if kind == "iirs" else None
        # MATCH-11 default: the best-known correction, fitted from our committed registrations
        self.prior = geoprior.load(self.meta.product_id)

    def offset(self, row):
        e, n = self.prior.offset_at(row)
        return float(e), float(n)

    def ground(self, rows, cols, corrected=True):
        rows, cols = np.asarray(rows, float).ravel(), np.asarray(cols, float).ravel()
        lat, lon = self.sysm.pixel_to_latlon(rows, cols)
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        if not corrected:
            return lat, lon
        e, n = np.array([self.offset(r) for r in rows]).T
        lat = lat + n / K
        return lat, lon + e / (K * np.cos(np.radians(lat)))

    def read(self, r0, c0, h, w):
        if self.kind == "iirs":
            p = iirs_composite_plane(self.meta, pds_raster.Window(int(r0), 0, int(h), self.S), self.sel)
            return p.array.astype(np.float32), p.valid_mask
        raw = pds_raster.read_raster(self.meta, pds_raster.Window(int(r0), int(c0), int(h), int(w))).astype(np.float32)
        ok = np.isfinite(raw) & (raw > 0)
        return norm(raw, ok), ok


class Frame:
    """Ground model for fine-frame pixels (for the terrain filter)."""

    def __init__(self, fn, oy, ox):
        self.fn, self.oy, self.ox = fn, oy, ox

    def pixel_to_latlon(self, rows, cols):
        return self.fn(np.asarray(rows, float) + self.oy, np.asarray(cols, float) + self.ox)


def fit_to_ref(src, ref, r0, c0, h, w):
    """Affine: source-window px -> full reference px, from the corrected prior."""
    gy, gx = np.meshgrid(np.linspace(0, h - 1, 7), np.linspace(0, w - 1, 5), indexing="ij")
    la, lo = src.ground(r0 + gy.ravel(), c0 + gx.ravel())
    rr, cc = ref.latlon_to_pixel(la, lo)
    return affine_fit(np.c_[gx.ravel(), gy.ravel()], np.c_[cc, rr])


def dense_checks(S, Rc, f, prior, T_c, centre, ref_name, px):
    """docs/tc_reference_protocol.md amendment 2: the dense lock's own checks."""
    def run(r):
        dg = {}
        st = cascade.register_step_dense(S, r, f, src="src", ref=ref_name, ref_pixel_m=px, prior=prior, diag=dg)
        return st, float(dg.get("z") or 0.0)
    locked = (T_c @ centre)[:2]
    sh = cv2.warpAffine(Rc, np.float32([[1, 0, 3.5], [0, 1, 4.25]]), Rc.shape[::-1], flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
    st_s, z_s = run(sh)
    moved = None if st_s is None or z_s < MIN_Z else (np.asarray(st_s.model.matrix, float) @ centre)[:2] - locked
    err = None if moved is None else float(np.hypot(moved[0] - 3.5, moved[1] - 4.25))
    st_c, z_c = run(np.full_like(Rc, float(Rc.mean())))
    st_n, z_n = run(np.random.default_rng(0).normal(0, 1, Rc.shape).astype(np.float32))
    out = {"fractional_shift": {"z": round(z_s, 1), "error_px": None if err is None else round(err, 3),
                                "pass": err is not None and err <= 0.5},
           "null_constant": {"z": round(z_c, 1), "pass": st_c is None or z_c < MIN_Z},
           "null_noise": {"z": round(z_n, 1), "pass": st_n is None or z_n < MIN_Z}}
    out["all_pass"] = all(v["pass"] for v in out.values())
    return out


def run_window(src, ref, r0, c0, h, w, matchers, device, stages, dem_tiles, route_opts, dense=False):
    out = {"src_row0": int(r0), "src_col0": int(c0), "window": [int(w), int(h)]}
    A = fit_to_ref(src, ref, r0, c0, h, w)
    cor = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ A.T
    ref_finer = max(ref.px_m) < max(src.px_m) / 1.5
    mpx = int(max(MARGIN_M / min(ref.px_m), MARGIN_PX))
    H, W = ref.arr.shape
    if not ref_finer:                          # (A) reference coarser: the source window is the template
        S, S_ok = src.read(r0, c0, h, w)
        if S_ok.mean() < 0.9:
            out["status"] = "skipped: source window has invalid pixels"; return out
        y0, x0 = np.maximum(np.floor(cor[:, 1::-1].min(0)).astype(int) - mpx, 0)
        y1, x1 = np.minimum(np.ceil(cor[:, 1::-1].max(0)).astype(int) + mpx, [H, W])
        Rc, Rc_ok = ref.arr[y0:y1, x0:x1], ref.ok[y0:y1, x0:x1]
        f = max(1, round(max(ref.px_m) / max(src.px_m)))
        prior = T(-x0, -y0) @ A @ np.linalg.inv(cascade.downsample_transform(f))
        diag = {}
        st = cascade.register_step_dense(S, Rc, f, src=src.kind, ref=ref.name, ref_pixel_m=max(ref.px_m),
                                         prior=prior, diag=diag)
        z = float(diag.get("z") or 0.0)
        out["coarse"] = {"z": round(z, 1), "factor": f, "template": "source"}
        if st is None or z < MIN_Z:
            out["status"] = "no coarse lock"; out["results"] = {}; return out
        T_c = np.asarray(st.model.matrix, float)                        # source px -> Rc px
        c2 = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ T_c.T
        o = np.maximum(np.floor(c2[:, :2].min(0)).astype(int) + 2, 0)
        e_ = np.minimum(np.ceil(c2[:, :2].max(0)).astype(int) - 2, [Rc.shape[1], Rc.shape[0]])
        wF, hF = int(e_[0] - o[0]), int(e_[1] - o[1])
        if wF < 32 or hF < 32:
            out["status"] = "skipped: fine frame too small"; out["results"] = {}; return out
        Wf = T(-o[0], -o[1]) @ T_c                                      # source px -> frame px
        if dense:                                                         # amendment 2: the lock is the answer
            ctr = np.array([w / 2, h / 2, 1.0])
            chk = dense_checks(S, Rc, f, prior, T_c, ctr, ref.name, max(ref.px_m))
            fs = warp_affine(S, Wf, (wF, hF))
            fs_ok = cv2.warpAffine(S_ok.astype(np.float32), Wf[:2], (wF, hF), flags=cv2.INTER_NEAREST) > 0.5
            mi = alignment_check(fs, Rc[o[1]:o[1] + hF, o[0]:o[0] + wF], np.eye(3), src_ok=fs_ok,
                                 ref_ok=Rc_ok[o[1]:o[1] + hF, o[0]:o[0] + wF])
            rx, ry = (T(x0, y0) @ T_c @ ctr)[:2]
            la_r, lo_r = ref.pixel_to_latlon([ry], [rx])
            la_s, lo_s = src.ground([r0 + h / 2], [c0 + w / 2], corrected=False)
            ef = enu(float(la_r[0]), float(lo_r[0]), float(la_s[0]), float(lo_s[0]))
            out["dense"] = {"checks": chk, "mi_check": mi, "precision_ref_px": round(float(st.rmse_px), 4),
                            "implied_offset_m": {"east": round(float(ef[0]), 1), "north": round(float(ef[1]), 1)},
                            "known_offset_m": dict(zip(("east", "north"), (round(float(v), 1) for v in src.offset(r0 + h / 2)))),
                            "pass_checks": bool(chk["all_pass"] and not mi.get("flag")),
                            "source_px_per_ref_px": round(max(ref.px_m) / min(src.px_m), 1)}
            out["status"] = "locked"; out["results"] = {}
            return out
        fsrc = warp_affine(S, Wf, (wF, hF))
        fsrc_ok = cv2.warpAffine(S_ok.astype(np.float32), Wf[:2], (wF, hF), flags=cv2.INTER_NEAREST) > 0.5
        fref = Rc[o[1]:o[1] + hF, o[0]:o[0] + wF].copy()
        fref_ok = Rc_ok[o[1]:o[1] + hF, o[0]:o[0] + wF].copy()
        to_ref = lambda M: T(x0 + o[0], y0 + o[1]) @ M @ Wf              # noqa: E731  source px -> ref px
        frame_ground = Frame(ref.pixel_to_latlon, y0 + o[1], x0 + o[0])
        J = jacobian_from_transform(Wf)
        centre_src = np.array([w / 2, h / 2, 1.0])
        grid_px = max(ref.px_m)
    else:                                      # (B) source coarser: the reference patch is the template
        pad = int(np.ceil(mpx * max(ref.px_m) / min(src.px_m)))
        ra, rb = max(0, r0 - pad), min(src.L, r0 + h + pad)
        S, S_ok = src.read(ra, 0, rb - ra, src.S)
        y0, x0 = np.maximum(np.floor(cor[:, 1::-1].min(0)).astype(int), 0)
        y1, x1 = np.minimum(np.ceil(cor[:, 1::-1].max(0)).astype(int), [H, W])
        if y1 - y0 < 64 or x1 - x0 < 64:
            out["status"] = "skipped: window barely overlaps the reference"; return out
        Rc, Rc_ok = ref.arr[y0:y1, x0:x1], ref.ok[y0:y1, x0:x1]
        if Rc_ok.mean() < 0.9:
            out["status"] = "skipped: reference patch has invalid pixels"; return out
        f = max(1, round(max(src.px_m) / max(ref.px_m)))
        Rin = np.linalg.inv(A) @ T(x0, y0)                              # Rc px -> source window px
        prior = (T(c0, r0 - ra) @ Rin) @ np.linalg.inv(cascade.downsample_transform(f))
        diag = {}
        st = cascade.register_step_dense(Rc, S, f, src=ref.name, ref=src.kind, ref_pixel_m=max(src.px_m),
                                         prior=prior, diag=diag)
        z = float(diag.get("z") or 0.0)
        out["coarse"] = {"z": round(z, 1), "factor": f, "template": "reference"}
        if st is None or z < MIN_Z:
            out["status"] = "no coarse lock"; out["results"] = {}; return out
        T_c = np.asarray(st.model.matrix, float)                        # Rc px -> source-region px
        c2 = np.array([[0, 0, 1], [Rc.shape[1], 0, 1], [0, Rc.shape[0], 1], [Rc.shape[1], Rc.shape[0], 1]], float) @ T_c.T
        o = np.maximum(np.floor(c2[:, :2].min(0)).astype(int) + 2, 0)
        e_ = np.minimum(np.ceil(c2[:, :2].max(0)).astype(int) - 2, [S.shape[1], S.shape[0]])
        wF, hF = int(e_[0] - o[0]), int(e_[1] - o[1])
        if wF < 32 or hF < 32:
            out["status"] = "skipped: fine frame too small"; out["results"] = {}; return out
        Wr = T(-o[0], -o[1]) @ T_c                                      # Rc px -> frame px
        fsrc = S[o[1]:o[1] + hF, o[0]:o[0] + wF].copy()
        fsrc_ok = S_ok[o[1]:o[1] + hF, o[0]:o[0] + wF].copy()
        fref = warp_affine(Rc, Wr, (wF, hF))
        fref_ok = cv2.warpAffine(Rc_ok.astype(np.float32), Wr[:2], (wF, hF), flags=cv2.INTER_NEAREST) > 0.5
        fref = norm(fref, fref_ok)
        to_ref = lambda M: T(x0, y0) @ np.linalg.inv(Wr) @ M @ T(-o[0], -o[1])   # noqa: E731  region px -> ref px
        frame_ground = Frame(lambda rr, cc: src.ground(rr + ra, cc), o[1], o[0])
        J = np.eye(2)
        centre_src = np.array([o[0] + wF / 2, o[1] + hF / 2, 1.0])
        r0 = ra                                                          # source-region rows start here
        grid_px = max(src.px_m)
    out["fine_frame_px"] = [wF, hF]
    # the frame centre in SOURCE product px (case B: r0 is now the region start, c0 = 0)
    lat_c, lon_c = (float(v[0]) for v in src.ground([r0 + centre_src[1]], [c0 + centre_src[0]]))
    dem = None
    if stages["geometry_filter"] and dem_tiles:
        dem = dem_patch(dem_tiles, (lat_c - 0.3, lat_c + 0.3, lon_c - 0.3, lon_c + 0.3))
    exp = scale.expected_scale(scale.pixel_scale(src.meta), ref.pixel_scale())
    results = {}

    def evaluate(name):
        t0 = time.perf_counter(); r = {}
        opts = route_opts if name == "xoftr" else {}
        try:
            if opts:
                from chandralign import synth
                from chandralign.matching import adapter
                mk = lambda a: synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool),  # noqa: E731
                                                shadow_mask=np.zeros(a.shape, bool), gsd_m=grid_px, meta=None, geo=None)
                pa, pb = mk(fsrc), mk(fref); ms = adapter.match(pa, pb, model_name=name, device=device, **opts)
            else:
                ms, pa, pb = match(name, fsrc, fref, device, grid_px)
            n = int(len(ms.src_pts))
            if n < 4:
                results[name] = {"status": f"only {n} matches", "success": False}; return results[name]
            fr = fine_stage(ms, fsrc, fref, centre=(wF / 2, hF / 2), flags=stages, ground_model=frame_ground, dem=dem)
            r.update(matches=n, inliers=fr.inlier_count, inlier_ratio=round(fr.inlier_ratio, 4),
                     coverage=round(fr.coverage, 3), inlier_rmse_px=None if fr.rmse_px is None else round(fr.rmse_px, 3))
            if not fr.ok:
                results[name] = {**r, "status": "fine stage: no transform", "success": False}; return results[name]
            Tt = to_ref(np.asarray(fr.model.matrix, float))                   # source(-window/region) px -> ref px
            rx, ry = (Tt @ centre_src)[:2]
            la_r, lo_r = ref.pixel_to_latlon([ry], [rx])
            sr, sc = r0 + centre_src[1], c0 + centre_src[0]
            la_s, lo_s = src.ground([sr], [sc], corrected=False)
            ef = enu(float(la_r[0]), float(lo_r[0]), float(la_s[0]), float(lo_s[0]))
            known = np.array(src.offset(sr))
            bound = max(2 * float(np.hypot(*ref.px_m)), 150.0)
            gates = control_gates.run_all(control_gates.pipeline_from(name, device=device, gsd_m=grid_px,
                                                                      stages=stages, **opts), fsrc, fref, pa, pb)
            verdict = scale.check(Tt, exp, centre=tuple(centre_src[:2]))
            q = quality.assess(inlier_count=fr.inlier_count, inlier_ratio=fr.inlier_ratio, spatial_coverage=fr.coverage,
                               model=fr.model, scale_ok=verdict.ok, scale_status=verdict.status,
                               gates=gates.gates, require_gates=True)
            mi = alignment_check(fsrc, fref, np.asarray(fr.model.matrix, float), src_ok=fsrc_ok, ref_ok=fref_ok)
            ks = gates.to_dict().get("perturbation_sensitivity", {}).get("error_px")
            r.update(status="registered", tier=q.tier, gates=gates.gates, scale_status=verdict.status,
                     known_shift_error_px=ks, gates_pass=bool(gates.all_passed),
                     tier_ok=q.tier in ("HIGH", "MEDIUM", "LOW"), mi_check=mi,
                     implied_offset_m={"east": round(float(ef[0]), 1), "north": round(float(ef[1]), 1)},
                     known_offset_m={"east": round(float(known[0]), 1), "north": round(float(known[1]), 1)},
                     vs_known_m=round(float(np.hypot(*(ef - known))), 1), bound_m=round(bound, 1),
                     source_px={"known_shift": to_source_px(ks, J, "exact"),
                                "inlier_rmse": to_source_px(fr.rmse_px, J, "exact"),
                                "mi_peak_offset": None if mi.get("peak_offset_px") is None else
                                to_source_px(float(np.hypot(*mi["peak_offset_px"])), J, "exact")})
            r["success"] = bool(r["gates_pass"] and r["tier_ok"] and not mi.get("flag") and r["vs_known_m"] <= bound)
        except Exception as exc:                                            # recorded, never hidden
            r.update(status=f"error: {type(exc).__name__}: {exc}"[:300], success=False)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = r
        return r

    # audit I-09: "routed" = routing's matcher, then its fallbacks only if rejected (routing.run_candidates),
    # routed on the REAL tile's pixel size where a label is held (MI, TC), not the nominal registry value
    for name in matchers:
        if name != "routed":
            evaluate(name)
            continue
        choice = routing.choose(src.meta, ref.meta if ref.meta is not None else REF_INSTRUMENT[ref.kind])
        if choice.route != "direct":
            results["routed"] = {"status": f"routing chose {choice.route}: {choice.reason}"[:300], "success": False,
                                 "routing": choice.as_provenance()}
            continue
        rc = routing.run_candidates(choice, lambda m: results[m] if m in results else evaluate(m),
                                    lambda r: bool(r.get("success")))
        results["routed"] = {**results[rc["used"]], "used": rc["used"], "tried": rc["tried"],
                             "routing": choice.as_provenance()}
    out["results"] = results
    out["status"] = "locked"
    return out


def windows_for(src, ref, n):
    """Up to n source windows whose corrected footprint lies inside the reference."""
    h = WIN_WAC.get(src.kind, WIN[src.kind]) if ref.name.startswith("WAC") else WIN[src.kind]
    w = src.S if src.kind == "iirs" else min(h, src.S)
    c0 = 0 if src.kind == "iirs" else (src.S - w) // 2
    H, W = ref.arr.shape
    rows = np.arange(0, src.L - h, max(h // 4, 32))
    ok = []
    for r0 in rows:
        A = fit_to_ref(src, ref, r0, c0, h, w)
        cor = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ A.T
        if src.kind == "iirs":
            # docs/tc_reference_protocol.md amendment 2: the whole latitude span on held data;
            # east/west clipping allowed (only the covered part is matched)
            inside = (np.ptp(np.clip(cor[:, 0], 0, W)) > 64) and cor[:, 1].min() >= 0 and cor[:, 1].max() < H
        else:
            inside = (cor[:, 0].min() >= 0 and cor[:, 1].min() >= 0 and cor[:, 0].max() < W and cor[:, 1].max() < H)
        if inside:
            ok.append(int(r0))
    if not ok:
        return [], (h, w, c0)
    picks = np.linspace(min(ok), max(ok), min(n, len(ok))).astype(int)
    return sorted({int(min(ok, key=lambda r: abs(r - p))) for p in picks}), (h, w, c0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=["ohrc", "tmc2", "iirs"])
    ap.add_argument("--reference", required=True, choices=["wac", "mi", "tc"])
    ap.add_argument("--tile", default=None, help="TC tile, e.g. N03E021N00E024SC (with --reference tc)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--matchers", nargs="+", default=None,
                    help="matchers to run (default: the frozen MATCHERS[source]); 'routed' = routing + fallbacks (I-09)")
    ap.add_argument("--dense", action="store_true", help="amendment 2: the dense lock is the answer (reference-coarser case)")
    args = ap.parse_args()
    src = Source(args.source)
    ref = load_ref(args.reference, args.source, args.tile)
    stages = stage_flags()
    dem_tiles = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")
    route_opts = dict(routing.choose("IIRS", "WAC").fine_stage_options)
    matchers = args.matchers or MATCHERS[args.source]
    picks, (h, w, c0) = windows_for(src, ref, N_WIN)
    print(f"{args.source} -> {ref.name}: windows at rows {picks} (window {w} x {h} px)", flush=True)
    wins = []
    for r0 in picks:
        t = time.perf_counter()
        wd = run_window(src, ref, r0, c0, h, w, matchers, args.device, stages, dem_tiles, route_opts,
                        dense=args.dense)
        wd["seconds"] = round(time.perf_counter() - t, 1)
        wins.append(wd)
        print(json.dumps({k: wd.get(k) for k in ("src_row0", "status", "coarse")} |
                         {"ok": {m: (v.get("status", "")[:18], v.get("tier"), v.get("vs_known_m"),
                                     (v.get("mi_check") or {}).get("flag"), v.get("success"))
                                 for m, v in (wd.get("results") or {}).items()}}), flush=True)
    summary = {}
    for name in matchers:
        ok = sum(bool((wd.get("results") or {}).get(name, {}).get("success")) for wd in wins)
        rate = ok / len(wins) if wins else 0.0
        summary[name] = {"success": ok, "windows": len(wins),
                         "verdict": "solved" if wins and rate >= 0.9 else "degraded" if rate >= 0.6 else "unsolved"}
    print(json.dumps(summary))
    out = ROOT / (args.out or f"reports/map_{args.source}_{args.reference}{'_' + args.tile[:3] if args.tile else ''}.json")
    out.write_text(json.dumps({"source": "measured", "protocol": "docs/map_pairings_protocol.md",
                               "pairing": f"{src.meta.product_id} -> {ref.name}", "pipeline_stages": stages,
                               "prior": src.prior.as_dict(),
                               "run": run_record(), "summary": summary, "windows": wins}, indent=2, default=str),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
