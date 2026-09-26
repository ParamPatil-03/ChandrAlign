"""MATCH-10 on real data: OHRC -> TC by two independent routes, which must agree.

    .venv/Scripts/python scripts/cascade_ohrc_tc.py --windows 3

Needs OHRC and TMC-2 in data/raw/ch2 and the N00 TC tile in data/raw/selene/tc.

Real products have no analytic truth. What they do have is a second, independent
route to the same answer:

    cascade  OHRC --16x--> TMC-2 --1.5x--> TC      (goes THROUGH TMC-2's pixels)
    direct   OHRC --24x--> TC                      (OHRC and TC only)

Both are feasible by the planner's measured footprint rule (727 and 495 coarse
pixels). They share no intermediate, so if they put OHRC in the same place on
TC the cascade is consistent on real data -- and if they do not, the size of
the disagreement is the honest number to report.

PRIORS ARE SYSTEM-LEVEL ONLY: OHRC's system grid, TMC-2's SYSTEM corners, TC's
map projection. TMC-2's refined geolocation was fitted to SELENE by ISRO and is
never used (the rule agreed with Member A). Priors give orientation and scale
only; every position is SEARCHED over +-9 km, because system geolocation is km
off (TMC-2 4.7 km, OHRC ~2 km).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.estimate import models  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.io.evidence import evidence_label  # noqa: E402
from chandralign.matching import cascade as cc  # noqa: E402

TC_TILE = "TCO_MAP_02_N00E021S03E024SC"
# Measured: at OHRC's location TMC-2's SYSTEM position is 5.1 km from where it
# really is, and OHRC's is ~2 km off too. Step 1 compares two system
# geolocations, so their errors ADD: a first version searched +-6 km and
# found the match on the region's edge. +-9 km covers the measured sum.
MARGIN_M = 9000.0


def affine_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    X = np.c_[src, np.ones(len(src))]
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
    A = np.eye(3)
    A[:2, :] = sol.T
    return A


def prior(src_model, src_origin, n_src, factor, dst_model, dst_origin):
    """Linear map (downsampled src px -> dst px) from geolocation models only."""
    g = np.linspace(0, n_src / factor - 1, 9)
    gx, gy = np.meshgrid(g, g)
    rows = src_origin[0] + (gy.ravel() + 0.5) * factor - 0.5
    cols = src_origin[1] + (gx.ravel() + 0.5) * factor - 0.5
    lat, lon = src_model.pixel_to_latlon(rows, cols)
    r, c = dst_model.latlon_to_pixel(lat, lon)
    return affine_fit(np.c_[gx.ravel(), gy.ravel()], np.c_[np.asarray(c) - dst_origin[1],
                                                          np.asarray(r) - dst_origin[0]])


def region_around(model, lat, lon, half_px, shape):
    r, c = model.latlon_to_pixel(np.array([lat]), np.array([lon]))
    r, c = int(round(float(np.asarray(r)[0]))), int(round(float(np.asarray(c)[0])))
    y0, x0 = max(r - half_px, 0), max(c - half_px, 0)
    y1, x1 = min(r + half_px, shape[0]), min(c + half_px, shape[1])
    return (y0, x0), (y1 - y0, x1 - x0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--ohrc-px", type=int, default=3072, help="OHRC window side, native px")
    ap.add_argument("--out", default="reports/cascade_ohrc_tc.json")
    args = ap.parse_args()

    ohrc = parse_label(evidence_label("OHRC", ROOT))
    tmc = parse_label(evidence_label("TMC2", ROOT))
    tc = parse_label(ROOT / "data/raw/selene/tc" / f"{TC_TILE}.lbl")
    om = projection.load_grid_model(ohrc)                        # system-level for OHRC
    assert om.independent_of_references
    tm = projection.load_corner_model(tmc, corners="system")     # NEVER the refined geolocation
    assert tm.independent_of_references
    cm = projection.load_map_model(tc)
    tc_m = float(tc.gsd_m)
    n = args.ohrc_px

    # OHRC windows that sit safely inside the N00 tile (lat < 0 with margin).
    L, S = ohrc.array_shape
    rows = np.arange(n, L - n, 500)
    lats, _ = om.pixel_to_latlon(rows, np.full(rows.shape, S / 2.0))
    ok_rows = rows[np.asarray(lats) < -0.12]
    picks = np.linspace(ok_rows.min(), ok_rows.max(), args.windows).astype(int)

    results = []
    for rc in picks:
        t0 = time.perf_counter()
        o_org = (int(rc) - n // 2, S // 2 - n // 2)
        ohrc_win = pds_raster.read_raster(ohrc, pds_raster.Window(o_org[0], o_org[1], n, n)).astype(np.float32)
        lat_c, lon_c = (float(np.asarray(v)[0]) for v in om.pixel_to_latlon(np.array([rc]), np.array([S / 2.0])))
        row = {"ohrc_row": int(rc), "lat": round(lat_c, 4)}

        # --- cascade step 1: OHRC -> TMC-2 (16x)
        t_half = int((MARGIN_M + n * 0.305) / 4.92)
        t_org, t_shape = region_around(tm, lat_c, lon_c, t_half, tmc.array_shape)
        tmc_reg = pds_raster.read_raster(tmc, pds_raster.Window(t_org[0], t_org[1], *t_shape)).astype(np.float32)
        p1 = prior(om, o_org, n, 16, tm, t_org)
        dg1, dg2, dgd = {}, {}, {}
        s1 = cc.register_step_dense(ohrc_win, tmc_reg, 16, src="OHRC", ref="TMC2", ref_pixel_m=4.92,
                                    prior=p1, diag=dg1)

        # --- cascade step 2: TMC-2 -> TC (1.5x). Source = a TMC-2 window around
        # where step 1 put OHRC, so it is a real piece of TMC-2, not the whole region.
        s2 = None
        if s1 is not None:
            land = models.apply(s1.model, np.array([[n / 2.0, n / 2.0]]))[0] + np.array([t_org[1], t_org[0]])
            # 700 px (3.4 km) of TMC-2: ~470 TC px, far above the 112 px footprint
            # threshold. A first version used 1400 px (6.9 km); near latitude 0 that
            # window reached past the edge of the TC tile, so its true place was not
            # inside the reference at all (diagnosed: search region cut at the tile).
            w2 = 700
            t2_org = (int(land[1]) - w2 // 2, int(land[0]) - w2 // 2)
            tmc_win = pds_raster.read_raster(tmc, pds_raster.Window(t2_org[0], t2_org[1], w2, w2)).astype(np.float32)
            la2, lo2 = (float(np.asarray(v)[0]) for v in tm.pixel_to_latlon(np.array([t2_org[0] + w2 / 2]),
                                                                              np.array([t2_org[1] + w2 / 2])))
            c_half = int((MARGIN_M + w2 * 4.92) / tc_m)
            c_org, c_shape = region_around(cm, la2, lo2, c_half, tc.array_shape)
            tc_reg = pds_raster.read_raster(tc, pds_raster.Window(c_org[0], c_org[1], *c_shape)).astype(np.float32)
            p2 = prior(tm, t2_org, w2, 1, cm, c_org)
            s2 = cc.register_step_dense(tmc_win, tc_reg, 1, src="TMC2", ref="TC", ref_pixel_m=tc_m, prior=p2,
                                        diag=dg2)

        # --- direct: OHRC -> TC (24x)
        d_half = int((MARGIN_M + n * 0.305) / tc_m)
        d_org, d_shape = region_around(cm, lat_c, lon_c, d_half, tc.array_shape)
        tc_d = pds_raster.read_raster(tc, pds_raster.Window(d_org[0], d_org[1], *d_shape)).astype(np.float32)
        pd = prior(om, o_org, n, 24, cm, d_org)
        sd = cc.register_step_dense(ohrc_win, tc_d, 24, src="OHRC", ref="TC", ref_pixel_m=tc_m, prior=pd,
                                    diag=dgd)

        row["cascade_steps_registered"] = [s1 is not None, s2 is not None]
        row["direct_registered"] = sd is not None
        row["diagnostics"] = {"step1": dg1, "step2": dg2, "direct": dgd}
        if s1 is not None and s2 is not None and sd is not None:
            # Put every transform into ABSOLUTE pixel frames before chaining/comparing.
            def shifted(step, src_off, ref_off):
                T = np.eye(3); T[:2, 2] = [ref_off[1], ref_off[0]]          # region px -> absolute
                U = np.eye(3); U[:2, 2] = [-src_off[1], -src_off[0]]        # absolute -> window px
                return cc.StepResult(step.src, step.ref,
                                     models.TransformModel(kind="affine", matrix=T @ step.model.matrix @ U),
                                     step.rmse_px, step.ref_pixel_m)
            a1 = shifted(s1, o_org, t_org)            # OHRC abs px -> TMC-2 abs px
            a2 = shifted(s2, t2_org, c_org)           # TMC-2 abs px -> TC abs px
            ad = shifted(sd, o_org, d_org)            # OHRC abs px -> TC abs px
            ch = cc.chain([a1, a2], at=(o_org[1] + n / 2.0, o_org[0] + n / 2.0))

            g = np.linspace(0, n - 1, 9)
            gx, gy = np.meshgrid(g, g)
            pts = np.c_[gx.ravel() + o_org[1], gy.ravel() + o_org[0]]
            gap = np.hypot(*(models.apply(ch.model, pts) - models.apply(ad.model, pts)).T)
            row.update({
                "cascade_vs_direct_tc_px": {"rmse": round(float(np.sqrt(np.mean(gap ** 2))), 3),
                                            "max": round(float(gap.max()), 3)},
                "cascade_vs_direct_m": round(float(np.sqrt(np.mean(gap ** 2))) * tc_m, 2),
                "cascade_error": {"per_step": ch.per_step, "total_tc_px": ch.total_rmse_px,
                                  "total_m": ch.total_rmse_m},
                "direct_error_tc_px": round(sd.rmse_px, 4),
            })
            # How far OHRC's own SYSTEM geolocation is from where both routes put it.
            la, lo = om.pixel_to_latlon(pts[:, 1], pts[:, 0])
            r_sys, c_sys = cm.latlon_to_pixel(la, lo)
            off = models.apply(ad.model, pts) - np.c_[np.asarray(c_sys), np.asarray(r_sys)]
            ground = np.c_[off[:, 0] * tc_m * math.cos(math.radians(lat_c)), -off[:, 1] * tc_m]
            row["ohrc_system_offset_m"] = {"east": round(float(ground[:, 0].mean())),
                                           "north": round(float(ground[:, 1].mean())),
                                           "total": round(float(np.hypot(*ground.mean(axis=0))))}
        row["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(row)
        print(json.dumps(row), flush=True)

    out = ROOT / args.out
    out.write_text(json.dumps({"source": "measured", "tile": TC_TILE,
                               "routes": {"cascade": "OHRC -16x-> TMC-2 -1.5x-> TC", "direct": "OHRC -24x-> TC"},
                               "priors": "system-level only (OHRC system grid, TMC-2 system corners, TC map)",
                               "run": run_record(),
                               "rows": results}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
