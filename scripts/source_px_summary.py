"""Every adopted accuracy number, re-expressed in SOURCE-image pixels (the problem statement's unit).

    python scripts/source_px_summary.py

Reads the committed reports of the adopted configurations; writes reports/source_px_summary.json.
Conversion: src/chandralign/evaluate/source_px.py (worst direction is the headline). Where a run
stored its transform (TMC-2) the Jacobian is exact; otherwise it is built from the verified pixel
sizes of source and frame, and says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.estimate import scale  # noqa: E402
from chandralign.evaluate import source_px as sp  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def px_of(instrument_dir):
    ps = scale.pixel_scale(parse_label(next((ROOT / instrument_dir).rglob("*_d_img_d18.xml"))))
    return (ps.across_m, ps.along_m), ps.verified


def stats(vals):
    v = [x for x in vals if x is not None]
    return None if not v else {"n": len(v), "median": round(float(np.median(v)), 4), "max": round(float(np.max(v)), 4)}


def summarise(name, rows, metrics):
    """rows: list of dicts with metric values in frame px and a jacobian 'J' + 'method'."""
    out = {"pairing": name, "windows": len(rows), "metrics": {}}
    for m, label in metrics.items():
        frame = [r.get(m) for r in rows]
        src = [None if r.get(m) is None else sp.to_source_px(r[m], r["J"], r["method"])["worst"] for r in rows]
        s = stats(src)
        out["metrics"][m] = {"meaning": label, "frame_px": stats(frame), "source_px_worst": s,
                             "sub_pixel_in_source": None if s is None else bool(s["max"] < 1.0),
                             "sub_pixel_median": None if s is None else bool(s["median"] < 1.0)}
    fac = [sp.factors(r["J"])["worst"] for r in rows]
    out["source_px_per_frame_px_worst"] = {"min": round(min(fac), 3), "max": round(max(fac), 3)}
    out["method"] = sorted({r["method"] for r in rows})
    return out


def main() -> int:
    res = []
    ohrc_px, _ = px_of("data/raw/ch2/ohrc")
    iirs_px, _ = px_of("data/raw/ch2/iirs")

    # TMC-2 -> SELENE TC (adopted: parallax on, TC DTM, h at the ground point)
    raw = {(w["tmc_row"], w.get("tmc_col")): w for w in load("reports/tmc2_tc_registration_height_ref.json")["rows"]}
    probe = {(w["tmc_row"], w["tmc_col"]): w for w in load("reports/tmc2_tc_parallax_height_ref.json")["windows"]}
    rows = []
    for k, w in raw.items():
        if w.get("status") != "registered":
            continue
        J = sp.jacobian_from_transform(w["registration_result"]["model"]["matrix"])   # TMC-2 px -> TC px
        p = probe.get(k, {})
        rows.append({"J": J, "method": "exact (stored transform)", "hilly": w.get("tile") == "N00",
                     "known_shift": w["gate_detail"]["perturbation_sensitivity"]["error_px"],
                     "probe_median": p.get("median_err_parallax_px"), "probe_p90": p.get("p90_err_parallax_px"),
                     "fit_rms": (w.get("pipeline") or {}).get("parallax", {}).get("rms_px")})
    m = {"known_shift": "recovery of a known (3, 4) px shift (precision of the lock)",
         "probe_median": "median |NCC-probe offset - model| per window (accuracy proxy, matcher-free)",
         "probe_p90": "90th percentile of the same",
         "fit_rms": "RMS of the parallax model on its inliers"}
    for sub, sel in (("hilly (N00)", True), ("flat (N03/N09)", False)):
        res.append(summarise(f"TMC-2 -> SELENE TC, {sub}", [r for r in rows if r["hilly"] == sel], m))

    # OHRC -> LRO NAC (shipped behaviour: reports/ohrc_nac_q8_auto_bridge.json, routed matcher)
    meta = {}
    for f in ("tmc2_nac_lroc_meta.json", "iirs_nac_lroc_meta.json"):
        meta.update(load(f"data/pairs/{f}")["products"])
    rows = []
    for w in load("reports/ohrc_nac_q8_auto_bridge.json")["windows"]:
        r = (w.get("results") or {}).get("routed") or {}
        if not r.get("success") and not (r.get("gates_pass") and r.get("tier_ok") and w.get("consistent")):
            continue
        mm = meta[w["nac"]]
        bx, by = w.get("fine_block") or [1, 1]
        frame = (float(mm["scaled_pixel_width"]) * bx, float(mm["scaled_pixel_height"]) * by)
        rows.append({"J": sp.jacobian_from_pixel_sizes(ohrc_px, frame),
                     "method": "pixel sizes (OHRC verified; NAC LROC scaled pixel, unverified)",
                     "known_shift": r.get("known_shift_error_px"), "inlier_rmse": r.get("inlier_rmse_px")})
    res.append(summarise("OHRC -> LRO NAC", rows,
                         {"known_shift": "recovery of a known (3, 4) px shift (precision of the lock)",
                          "inlier_rmse": "RMS of the model on its delivered points (fit residual)"}))

    # IIRS -> LRO WAC (mosaic, 100 m/px simple cylindrical near the equator)
    rows = []
    for w in load("reports/iirs_wac_mosaic.json")["windows"]:
        r = (w.get("results") or {}).get("xoftr") or {}
        if not r.get("success"):
            continue
        rows.append({"J": sp.jacobian_from_pixel_sizes(iirs_px, (100.0, 100.0)),
                     "method": "pixel sizes (IIRS verified; mosaic 100 m, cos(lat) ~ 1 at the clip)",
                     "known_shift": r.get("known_shift_error_px"), "inlier_rmse": r.get("inlier_rmse_px")})
    res.append(summarise("IIRS -> LRO WAC (xoftr)", rows,
                         {"known_shift": "recovery of a known (3, 4) px shift", "inlier_rmse": "fit residual RMS"}))

    # IIRS -> LRO NAC (dense lock, measured in IIRS px already)
    rows = []
    for w in load("reports/iirs_nac_dense.json")["windows"]:
        d = w.get("dense") or {}
        if not d.get("success"):
            continue
        rows.append({"J": np.eye(2), "method": "already in IIRS (source) px",
                     "precision": d.get("precision_iirs_px")})
    res.append(summarise("IIRS -> LRO NAC (dense)", rows,
                         {"precision": "robust spread of the dense step's sub-pixel estimates"}))

    out = {"source": "derived from committed measured reports", "run": run_record(),
           "source_pixel_m": {"OHRC": ohrc_px, "IIRS": iirs_px}, "pairings": res,
           "note": "IIRS -> NAC fractional known shifts (0.02-0.11 IIRS px, 12/12) are in docs/iirs_nac_protocol.md"}
    (ROOT / "reports/source_px_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for p in res:
        print(f'\n{p["pairing"]}: {p["windows"]} windows; source px per frame px (worst) {p["source_px_per_frame_px_worst"]}')
        for k, v in p["metrics"].items():
            print(f'  {k:13} frame {v["frame_px"]}  ->  source(worst) {v["source_px_worst"]}  sub-px max:{v["sub_pixel_in_source"]} median:{v["sub_pixel_median"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
