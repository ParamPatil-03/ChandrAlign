"""Does solar AZIMUTH difference break matching on real lunar terrain? (MATCH-09)

    .venv/Scripts/python scripts/nac_azimuth_validation.py

Criterion frozen in docs/nac_azimuth_protocol.md, committed before this script
existed and before the imagery finished downloading. That document also carries
a PRE-REGISTERED PREDICTION of each matcher's result at each angle, so the
synthetic model can be falsified rather than compared against afterwards.

This is the companion to scripts/nac_illumination_validation.py and shares its
machinery deliberately -- same window rule, same coarse lock, same estimator,
same gates, same success definition -- so the incidence and azimuth results are
directly comparable. Only the pair selection differs: there, pairs came from an
overlap search and varied incidence; here they are three FIXED pairs chosen so
that incidence and pixel scale are held and azimuth is what moves.

AZIMUTHS ARE COMPUTED, NOT SCRAPED. LROC publishes a "Sub solar azimuth" field
which is in an instrument frame: for M102014464R it reads 180.24 (due south)
where the sub-solar point is 81.8 deg WEST of the target, so the ground azimuth
must be near 270. The value used here is the spherical-trigonometry azimuth from
the target to the sub-solar point, which reproduces 270.23 for that product.

WHAT THIS CANNOT REACH: ~90 deg of azimuth difference, the angle the synthetic
model calls worst. An equatorial target imaged from a polar orbit is lit from
the east or the west, never from the side. Four photometric sites were scanned
and three sit within a degree of the equator. See the protocol, section 0.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from nac_illumination_validation import (  # noqa: E402
    BANDS, DEGRADED, LINES, MATCHERS, MIN_COARSE_Z, SAMPLES, SEARCH_LINES, SOLVED,
    WIN, WINDOW_FRACTIONS, _norm, coarse_lock, latlon_to_pixel, read_window, test_window)

# Protocol 1. Fixed before the imagery arrived. d_azimuth computed by spherical
# trigonometry from LROC sub-solar lat/lon; incidence and pixel scale held so
# that azimuth is the variable that moves.
PAIRS = [
    {"a": "M1132219897R", "b": "M1180506270R", "d_azimuth": 126.7,
     "inc_a": 15, "inc_b": 19, "scale_ratio": 1.00, "site": "ReinerGamma"},
    {"a": "M117338434L", "b": "M131494509L", "d_azimuth": 180.0,
     "inc_a": 84, "inc_b": 79, "scale_ratio": 1.00, "site": "Apollo11"},
    {"a": "M111443315R", "b": "M122054682L", "d_azimuth": 175.3,
     "inc_a": 26, "inc_b": 28, "scale_ratio": 1.06, "site": "Apollo11"},
]

# Protocol 3, written down before any run. Ranges, not points: the synthetic
# sweep sampled 120 and 135 deg, so 126.7 is interpolated and given latitude.
PREDICTION = {
    126.7: {"minima-loftr": (0.7, 1.0), "xoftr": (0.0, 0.4), "eloftr": (0.0, 0.0),
            "aliked-lightglue": (0.0, 0.0), "sift-nn": (0.0, 0.0)},
    180.0: {"minima-loftr": (1.0, 1.0), "xoftr": (1.0, 1.0), "eloftr": (0.0, 0.0),
            "aliked-lightglue": (0.0, 0.0), "sift-nn": (0.0, 0.0)},
    175.3: {"minima-loftr": (1.0, 1.0), "xoftr": (1.0, 1.0), "eloftr": (0.0, 0.0),
            "aliked-lightglue": (0.0, 0.0), "sift-nn": (0.0, 0.0)},
}


def find_products():
    """product id (no suffix letter case) -> path, for whatever is on disk."""
    out = {}
    for p in (ROOT / "data/raw/lro/nac").rglob("*.IMG"):
        out[p.stem.upper().rstrip("C")] = p           # M117338434LC.IMG -> M117338434L
    return out


def geometry_for(pid_full: str) -> dict | None:
    """ODE bbox + centre for one product, cached next to the experiment."""
    cache = ROOT / "data/pairs/nac_azimuth_geometry.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    if pid_full in store:
        return store[pid_full]
    from chandralign.io.ode_client import fetch_product_record
    try:
        r = fetch_product_record("nac." + pid_full.lower() + "c")
    except Exception as exc:
        print(f"   ODE lookup failed for {pid_full}: {type(exc).__name__} {str(exc)[:60]}")
        return None
    store[pid_full] = {"bbox": [float(r["Minimum_latitude"]), float(r["Maximum_latitude"]),
                                float(r["Westernmost_longitude"]), float(r["Easternmost_longitude"])],
                       "center_lat": float(r["Center_latitude"]),
                       "center_lon": float(r["Center_longitude"]),
                       "map_resolution": float(r["Map_resolution"]),
                       "incidence": float(r["Incidence_angle"])}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(store, indent=1), encoding="utf-8")
    return store[pid_full]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchers", default=",".join(MATCHERS))
    ap.add_argument("--windows", type=int, default=len(WINDOW_FRACTIONS))
    ap.add_argument("--out", default="reports/lro_nac_azimuth_validation.json")
    args = ap.parse_args()
    matchers = [m for m in args.matchers.split(",") if m]
    fracs = WINDOW_FRACTIONS[:args.windows]

    have = find_products()
    missing = [p[k] for p in PAIRS for k in ("a", "b") if p[k] not in have]
    if missing:
        print("not on disk yet: " + ", ".join(missing))
        return 2

    rows, skipped = [], []
    for p in PAIRS:
        ga, gb = geometry_for(p["a"]), geometry_for(p["b"])
        if ga is None or gb is None:
            skipped.append({**p, "why": "ODE geometry unavailable"})
            continue
        lat_lo = max(ga["bbox"][0], gb["bbox"][0])
        lat_hi = min(ga["bbox"][1], gb["bbox"][1])
        lon_c = 0.5 * (max(ga["bbox"][2], gb["bbox"][2]) + min(ga["bbox"][3], gb["bbox"][3]))
        if lat_hi <= lat_lo:
            skipped.append({**p, "why": "ODE bounding boxes do not overlap in latitude"})
            continue
        print(f"\n{p['a']} / {p['b']}   d_azimuth {p['d_azimuth']}   "
              f"incidence {p['inc_a']}/{p['inc_b']}   scale {p['scale_ratio']}   [{p['site']}]")
        for wi, fr in enumerate(fracs):
            lat = lat_lo + fr * (lat_hi - lat_lo)
            la, sa = latlon_to_pixel(ga["bbox"], lat, lon_c)
            lb, _ = latlon_to_pixel(gb["bbox"], lat, lon_c)
            src, src_ok = read_window(have[p["a"]], la, sa, WIN // 2)
            reg, reg_ok = read_window(have[p["b"]], lb, SAMPLES // 2, SEARCH_LINES)
            if src is None or reg is None or src_ok.mean() < 0.9 or reg_ok.mean() < 0.9:
                skipped.append({**p, "window": wi, "why": "window outside the product or too much null"})
                continue
            lock = coarse_lock(_norm(src, src_ok), _norm(reg, reg_ok))
            if lock is None or lock[2] < MIN_COARSE_Z:
                skipped.append({**p, "window": wi,
                                "why": f"coarse lock failed (z={lock[2]:.1f})" if lock else "lock not possible"})
                continue
            dy, dx, z = lock
            ref, ref_ok = reg[dy:dy + WIN, dx:dx + WIN], reg_ok[dy:dy + WIN, dx:dx + WIN]
            if ref.shape != (WIN, WIN):
                skipped.append({**p, "window": wi, "why": "locked window ran off the search region"})
                continue
            s_n, r_n = _norm(src, src_ok), _norm(ref, ref_ok)
            for m in matchers:
                rec = test_window(m, s_n, src_ok, r_n, ref_ok)
                rec.update({k: p[k] for k in ("a", "b", "d_azimuth", "inc_a", "inc_b", "scale_ratio", "site")})
                rec.update(window=wi, coarse_z=round(z, 2))
                rows.append(rec)
                print(json.dumps({k: rec[k] for k in ("d_azimuth", "window", "matcher", "success",
                                                      "tier", "perturbation_error_px", "inliers")}), flush=True)

    # ---- prediction scored against outcome --------------------------------
    print("\n=== PRE-REGISTERED PREDICTION vs MEASURED ===")
    print(f"{'d_az':>7}  {'matcher':<20}{'predicted':>14}{'measured':>12}   verdict")
    scored = []
    for p in PAIRS:
        dz = p["d_azimuth"]
        for m in matchers:
            rs = [r for r in rows if r["d_azimuth"] == dz and r["matcher"] == m]
            if not rs:
                continue
            rate = sum(r["success"] for r in rs) / len(rs)
            lo, hi = PREDICTION.get(dz, {}).get(m, (None, None))
            if lo is None:
                v = "no prediction"
            else:
                v = "HELD" if lo - 1e-9 <= rate <= hi + 1e-9 else (
                    "FALSIFIED (better than predicted)" if rate > hi else "FALSIFIED (worse)")
            scored.append({"d_azimuth": dz, "matcher": m, "n": len(rs), "rate": round(rate, 3),
                           "predicted": [lo, hi], "verdict": v})
            n_ok = sum(r["success"] for r in rs)
            pred = f"{lo:.0%}-{hi:.0%}" if lo is not None else "--"
            print(f"{dz:>7.1f}  {m:<20}{pred:>14}{f'{n_ok}/{len(rs)}':>12}   {v}")

    print("\n=== median perturbation error, px (the only known-truth measure) ===")
    for m in matchers:
        e = [r["perturbation_error_px"] for r in rows
             if r["matcher"] == m and r.get("perturbation_error_px") is not None]
        print(f"   {m:<20} {np.median(e):>7.2f} px over {len(e)} windows" if e else f"   {m:<20}   no data")

    if skipped:
        print(f"\n{len(skipped)} window-pairs excluded before any matcher ran:")
        for s in skipped[:12]:
            print(f"   {s['a']}/{s['b']}  window {s.get('window','-')}  {s['why']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "measured", "axis": "solar_azimuth_difference",
        "incidence_status": "held, not varied: 15/19, 84/79 and 26/28 deg",
        "scale_status": "held: ratios 1.00, 1.00, 1.06",
        "ninety_degree_status": ("structurally unavailable -- an equatorial target imaged from a "
                                 "polar orbit is lit from the east or west, never from the side; "
                                 "4 photometric sites scanned, 3 within 1 deg of the equator"),
        "azimuth_source": ("computed by spherical trigonometry from LROC sub-solar lat/lon; LROC's "
                           "own 'Sub solar azimuth' field is in an instrument frame and is not "
                           "ground azimuth"),
        "protocol": "docs/nac_azimuth_protocol.md",
        "prediction_registered_before_run": True,
        "pairs": PAIRS, "prediction": {str(k): v for k, v in PREDICTION.items()},
        "prediction_scored": scored, "thresholds": {"solved": SOLVED, "degraded": DEGRADED},
        "excluded": skipped, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(rows)} tests, {len(skipped)} excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
