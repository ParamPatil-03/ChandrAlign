"""TMC-2 -> LRO NAC registration, same-sun vs opposed-sun (docs/tmc2_nac_protocol.md).

    .venv/Scripts/python scripts/register_tmc2_nac.py                 # all groups
    .venv/Scripts/python scripts/register_tmc2_nac.py --products M111443315RC

WHAT IS FIXED ELSEWHERE, NOT HERE
Every threshold, window position, group and control is frozen in
docs/tmc2_nac_protocol.md, committed before this file. This script implements it
and decides nothing.

THE STEP
`cascade.register_step_dense`, unchanged: the NAC (finer) is block-averaged to
TMC-2 resolution, oriented by a 2x2 prior, and searched for with MIND over a
TMC-2 region; position is never taken from the prior. It maps NAC px -> TMC-2 px;
the inverse (TMC-2 -> NAC) is reported too, since that is what the PS asks for.

THE PRIOR
From LROC's corner coordinates (data/pairs/tmc2_nac_lroc_meta.json, frame checked
there), mapped through TMC-2's SYSTEM corner model. ISRO's refined grid is never
read. The LROC azimuth fields are NOT used for orientation: they agree with the
ground sun to only ~20 deg.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chandralign import config  # noqa: E402
from chandralign.evaluate.control_gates import _shift_content  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.matching import cascade, routing  # noqa: E402
from chandralign.geometry.nac import (  # noqa: E402,F401
    LINES, MOON_R_M, NULL_BELOW, PIX_OFFSET, SAMPLES, SCALE, WIN_LINES, Nac, enu)

# ---- frozen in docs/tmc2_nac_protocol.md -------------------------------------------
# The NAC layout (LINES, SAMPLES, ..., NULL_BELOW, WIN_LINES), enu and Nac moved to
# chandralign.geometry.nac (audit 2026-09-26 C-01); imported below, names unchanged.
FRACTIONS = (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)       # section 4
TMC_ACROSS_M, TMC_ALONG_M = 4.92, 5.037               # estimate.scale.pixel_scale, verified
MARGIN_M = 7000.0                                     # section 4: 7 km search margin
MIN_VALID = 0.90                                      # section 4 exclusion
SHIFT = tuple(int(v) for v in config.get("gates.perturbation_shift_xy", [3, 4]))
SHIFT_TOL = float(config.get("gates.perturbation_tolerance_px", 1.5))
NAC_SHIFT = (37, 43)                                  # amendment 1: ~3.7 x 4.3 TMC-2 px, fractional
MIN_Z = float(config.get("cascade.min_z", 10.0))      # the step's own bar (code default)
SOLVED, DEGRADED = 0.90, 0.60                         # section 7
NULL_VOID = 0.10                                      # section 6
FALSE_LOCK_M = 500.0                                  # section 6

GROUPS = {
    "same_sun": ["M111443315RC", "M131494509LC"],
    "opposed_sun": ["M117338434LC", "M122054682LC"],
    "secondary_same_sun_12x": ["M175124932LC"],
    "secondary_opposed_4x": ["M106719774LC", "M104362199LC", "M102000149LC", "M102014464RC"],
}
PRIMARY = ("same_sun", "opposed_sun")

# TMC-2 SYSTEM offset measured independently against SELENE TC
# (reports/tmc2_tc_registration.json, committed), by TMC-2 row: (east m, north m).
TC_OFFSET = {4687: (622, -4642), 16000: (659, -4753), 18750: (673, -4805), 21500: (703, -4827)}


def tc_offset_at(row: float) -> tuple[float, float]:
    rows = sorted(TC_OFFSET)
    e = np.interp(row, rows, [TC_OFFSET[r][0] for r in rows])
    n = np.interp(row, rows, [TC_OFFSET[r][1] for r in rows])
    return float(e), float(n)


def tmc_jacobian(sysm, lat0: float, lon0: float) -> np.ndarray:
    """d(col, row)/d(east, north) of TMC-2's SYSTEM model at a point, per metre."""
    k = math.pi / 180 * MOON_R_M
    d = 50.0
    dlat, dlon = d / k, d / (k * math.cos(math.radians(lat0)))
    r0, c0 = sysm.latlon_to_pixel([lat0], [lon0])
    re, ce = sysm.latlon_to_pixel([lat0], [lon0 + dlon])
    rn, cn = sysm.latlon_to_pixel([lat0 + dlat], [lon0])
    return np.array([[(ce[0] - c0[0]) / d, (cn[0] - c0[0]) / d],
                     [(re[0] - r0[0]) / d, (rn[0] - r0[0]) / d]])


def norm(a, valid=None):
    v = a[valid] if valid is not None and valid.any() else a
    lo, hi = np.percentile(v, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def step(nac_img, region, factor, lin, pid):
    diag: dict = {}
    t = time.perf_counter()
    r = cascade.register_step_dense(nac_img, region, factor, src=pid, ref="TMC2",
                                    ref_pixel_m=TMC_ACROSS_M, prior=lin, diag=diag)
    diag["seconds"] = round(time.perf_counter() - t, 1)
    return r, diag


def apply(M, x, y):
    p = np.asarray(M, float) @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def run_window(nac: Nac, frac: float, tmc, sysm, primary: bool) -> dict:
    centre_line = int(round(frac * LINES))
    out: dict = {"nac": nac.pid, "fraction": round(frac, 4), "centre_line": centre_line}
    l0, img, valid = nac.window(centre_line)
    out["valid_fraction"] = round(float(valid.mean()), 4)
    if valid.mean() < MIN_VALID:
        out["excluded"] = f"only {valid.mean():.0%} of NAC pixels valid (< {MIN_VALID:.0%})"
        return out

    cx = cy = WIN_LINES / 2.0                               # window centre, window px (x=sample, y=line)
    lat0, lon0 = nac.latlon(l0 + cy, cx)
    u, v, mirrored = nac.axes(lat0, lon0)
    factor = max(1, int(round(TMC_ACROSS_M / nac.px_w)))
    J = tmc_jacobian(sysm, lat0, lon0)
    lin = J @ np.column_stack([v * nac.px_w, u * nac.px_h]) * factor   # small px -> TMC px
    out.update(lat=round(lat0, 4), lon=round(lon0, 4), block_factor=factor, mirrored=mirrored,
               prior_2x2=[[round(float(x), 5) for x in row] for row in lin])

    # predicted TMC-2 footprint of the window, SYSTEM model; exclusion + search region
    L, S = tmc.array_shape
    corners = [(l0, 0), (l0, SAMPLES - 1), (l0 + WIN_LINES - 1, 0), (l0 + WIN_LINES - 1, SAMPLES - 1)]
    lats, lons = zip(*[nac.latlon(a, b) for a, b in corners])
    rr, cc = sysm.latlon_to_pixel(np.array(lats), np.array(lons))
    out["predicted_tmc_cols"] = [round(float(cc.min())), round(float(cc.max()))]
    if cc.min() < 0 or cc.max() > S - 1:
        out["excluded"] = "predicted TMC-2 footprint leaves the strip's columns"
        return out
    mr, mc = MARGIN_M / TMC_ALONG_M, MARGIN_M / TMC_ACROSS_M
    r0, r1 = max(0, int(rr.min() - mr)), min(L, int(math.ceil(rr.max() + mr)))
    c0, c1 = max(0, int(cc.min() - mc)), min(S, int(math.ceil(cc.max() + mc)))
    raw = pds_raster.read_raster(tmc, pds_raster.Window(r0, c0, r1 - r0, c1 - c0)).astype(np.float32)
    rvalid = raw > 0
    region = norm(raw, rvalid)
    region[~rvalid] = float(np.median(region[rvalid])) if rvalid.any() else 0.0
    out["region"] = {"row0": r0, "col0": c0, "rows": r1 - r0, "cols": c1 - c0,
                     "valid_fraction": round(float(rvalid.mean()), 4)}
    nimg = norm(img, valid)

    # ---- the registration --------------------------------------------------------
    res, diag = step(nimg, region, factor, lin, nac.pid)
    out["step"] = diag
    out["locked"] = res is not None
    if res is not None:
        T0 = np.asarray(res.model.matrix, float)                    # NAC window px -> region px
        g = np.array([[1, 0, c0], [0, 1, r0], [0, 0, 1]], float) @ T0   # -> global TMC px
        Wn = np.array([[1, 0, 0], [0, 1, -l0], [0, 0, 1]], float)   # global NAC px -> window px
        nac_to_tmc = g @ Wn
        out["nac_to_tmc"] = [[round(float(x), 8) for x in row] for row in nac_to_tmc]
        out["tmc_to_nac"] = [[round(float(x), 8) for x in row] for row in np.linalg.inv(nac_to_tmc)]
        out["subpixel_spread_tmc_px"] = round(float(res.rmse_px), 3)

        # ---- known-shift test, AS FIRST FROZEN: integer move of the TMC-2 region.
        # Vacuous (protocol amendment 1): it only re-indexes pixels. Reported, not counted.
        moved, mdiag = step(nimg, _shift_content(region, SHIFT[0], SHIFT[1]), factor, lin, nac.pid)
        rec = None
        if moved is not None:
            rec = apply(moved.model.matrix, cx, cy) - apply(T0, cx, cy)
            err = float(np.hypot(*(rec - np.array(SHIFT, float))))
        out["known_shift_vacuous_integer_tmc"] = {
            "shift": list(SHIFT), "z": mdiag.get("z"),
            "recovered": None if rec is None else [round(float(x), 3) for x in rec],
            "error_px": None if rec is None else round(err, 3), "counts": False}

        # ---- known-shift test, AMENDMENT 1: move the NAC content by NAC_SHIFT ------
        # T1(p) = T0(p - s), so the window centre should move by -A0 s in TMC-2 px.
        moved, mdiag = step(_shift_content(nimg, NAC_SHIFT[0], NAC_SHIFT[1]), region, factor, lin, nac.pid)
        expected = -(T0[:2, :2] @ np.array(NAC_SHIFT, float))
        rec = None
        if moved is not None:
            rec = apply(moved.model.matrix, cx, cy) - apply(T0, cx, cy)
            err = float(np.hypot(*(rec - expected)))
        out["known_shift"] = {"nac_shift_px": list(NAC_SHIFT), "z": mdiag.get("z"),
                              "expected_tmc_px": [round(float(x), 3) for x in expected],
                              "recovered_tmc_px": None if rec is None else [round(float(x), 3) for x in rec],
                              "error_px": None if rec is None else round(err, 3),
                              "tolerance_px": SHIFT_TOL,
                              "passed": bool(rec is not None and err <= SHIFT_TOL)}

        # ---- independent geolocation check (descriptive) --------------------------
        p = apply(g, cx, cy)                                         # (col, row) global TMC px
        slat, slon = sysm.pixel_to_latlon(np.array([p[1]]), np.array([p[0]]))
        off = enu(lat0, lon0, float(slat[0]), float(slon[0]))       # truth - system, metres
        tc_e, tc_n = tc_offset_at(float(p[1]))
        dis = float(np.hypot(off[0] - tc_e, off[1] - tc_n))
        out["geolocation"] = {"tmc_row": round(float(p[1]), 1), "tmc_col": round(float(p[0]), 1),
                              "system_offset_m": {"east": round(float(off[0])), "north": round(float(off[1]))},
                              "tc_measured_offset_m": {"east": round(tc_e), "north": round(tc_n)},
                              "disagreement_m": round(dis), "probable_false_lock": dis > FALSE_LOCK_M}
    out["success"] = bool(out["locked"] and out.get("known_shift", {}).get("passed"))

    # ---- null controls --------------------------------------------------------------
    rng = np.random.default_rng(0)
    grey = np.full_like(nimg, float(nimg.mean()))
    noise = np.clip(rng.normal(float(nimg.mean()), float(nimg.std()) or 0.1, nimg.shape), 0, 1).astype(np.float32)
    nulls = {}
    for name, a in (("flat_grey", grey), ("noise", noise)):
        r, d = step(a, region, factor, lin, nac.pid)
        nulls[name] = {"locked": r is not None, "z": d.get("z"), "failed": d.get("failed")}
    out["null"] = nulls

    # ---- orientation control (primary groups): NAC line axis reversed -----------------
    if primary:
        flip = lin @ np.diag([1.0, -1.0])
        r, d = step(nimg, region, factor, flip, nac.pid)
        out["flipped_prior"] = {"locked": r is not None, "z": d.get("z")}
    return out


def verdict(rate):
    return None if rate is None else ("solved" if rate >= SOLVED else "degraded" if rate >= DEGRADED else "unsolved")


def summarise(rows, meta):
    groups = {}
    for g, pids in GROUPS.items():
        W = [r for r in rows if r["nac"] in pids]
        counted = [r for r in W if "excluded" not in r]
        ok = [r for r in counted if r["success"]]
        ok_geo = [r for r in ok if not r.get("geolocation", {}).get("probable_false_lock")]
        n = len(counted)
        rate = len(ok) / n if n else None
        groups[g] = {
            "products": pids, "windows": len(W), "counted": n,
            "excluded": [{"nac": r["nac"], "fraction": r["fraction"], "why": r["excluded"]}
                         for r in W if "excluded" in r],
            "locked": sum(r["locked"] for r in counted), "success": len(ok),
            "success_rate": None if rate is None else round(rate, 3), "verdict": verdict(rate),
            "success_excluding_probable_false_locks": len(ok_geo),
            "median_z": (round(float(np.median([r["step"]["z"] for r in counted if r["step"].get("z") is not None])), 1)
                         if counted else None),
            "null_locks": sum(any(v["locked"] for v in r["null"].values()) for r in counted),
            "flipped_prior_locks_where_true_prior_succeeded":
                (sum(bool(r.get("flipped_prior", {}).get("locked")) for r in ok) if g in PRIMARY else None),
            "d_az_vs_tmc2": {p: next(c["d_az_vs_tmc2"] for c in meta["candidates"] if c["nac"] == p) for p in pids},
        }
    all_counted = [r for r in rows if "excluded" not in r]
    null_rate = (sum(any(v["locked"] for v in r["null"].values()) for r in all_counted) / len(all_counted)
                 if all_counted else 0.0)
    same = groups["same_sun"]
    stop = same["success_rate"] is None or same["success_rate"] <= 0.5
    return groups, null_rate, stop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=None)
    ap.add_argument("--out", default="reports/tmc2_nac_registration.json")
    args = ap.parse_args()

    meta = json.loads((ROOT / "data/pairs/tmc2_nac_lroc_meta.json").read_text())["products"]
    cand = json.loads((ROOT / "data/pairs/tmc2_nac_candidates.json").read_text())
    tmc = parse_label(next((ROOT / "data/raw/ch2/tmc2").rglob("*_d_img_d18.xml")))
    sysm = projection.load_corner_model(tmc, corners="system")
    choice = routing.choose("TMC2", "NAC")

    rows = []
    for g, pids in GROUPS.items():
        for pid in pids:
            if args.products and pid not in args.products:
                continue
            nac = Nac(pid, meta[pid])
            for f in FRACTIONS:
                r = run_window(nac, f, tmc, sysm, primary=g in PRIMARY)
                r["group"] = g
                rows.append(r)
                print(json.dumps({k: r.get(k) for k in ("nac", "fraction", "excluded", "locked", "success")}
                                 | {"z": r.get("step", {}).get("z"),
                                    "shift_err": r.get("known_shift", {}).get("error_px"),
                                    "geo_dis_m": r.get("geolocation", {}).get("disagreement_m"),
                                    "null": {k: v["locked"] for k, v in r.get("null", {}).items()},
                                    "flip": r.get("flipped_prior", {}).get("locked")}), flush=True)

    groups, null_rate, stop = summarise(rows, cand)
    report = {
        "source": "measured",
        "protocol": "docs/tmc2_nac_protocol.md",
        "pairing": "TMC-2 (ch2_tmc_nca_20250207T1102039417) -> LRO NAC CDR",
        "routing": {"route": choice.route, "model_name": choice.model_name, "regime": choice.regime},
        "method": "cascade.register_step_dense (MIND search + sub-pixel median), NAC as the finer src",
        "prior": "LROC corner coordinates through TMC-2 SYSTEM corners; position searched, never taken from the prior",
        "thresholds": {"min_z": MIN_Z, "known_shift_px": list(SHIFT), "shift_tolerance_px": SHIFT_TOL,
                       "solved": SOLVED, "degraded": DEGRADED, "null_void_rate": NULL_VOID,
                       "false_lock_m": FALSE_LOCK_M, "margin_m": MARGIN_M},
        "null_lock_rate": round(null_rate, 4),
        "void_by_nulls": null_rate > NULL_VOID,
        "stop_rule_triggered": stop,
        "groups": groups,
        "windows": rows,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({g: {k: v[k] for k in ("counted", "success", "success_rate", "verdict", "median_z",
                                              "null_locks", "success_excluding_probable_false_locks")}
                      for g, v in groups.items()}, indent=1))
    print("null lock rate", round(null_rate, 3), "stop rule:", stop, "\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
