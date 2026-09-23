"""Do the illumination findings survive on REAL lunar imagery? (MATCH-09 validation)

    .venv/Scripts/python scripts/nac_illumination_validation.py
    .venv/Scripts/python scripts/nac_illumination_validation.py --pairs 3 --windows 2

The criterion is frozen in docs/nac_illumination_protocol.md, committed before
this script existed. This measures; it does not decide, and it changes no
production setting.

WHAT THIS VALIDATES, AND WHAT IT DOES NOT
  incidence difference, on real NAC imagery ........ measured here
  solar azimuth difference ......................... STILL NOT VALIDATED

ODE gives incidence, emission and phase per product; it does not give solar
azimuth, and azimuth cannot be recovered from incidence -- two scenes can share
an incidence angle and be lit from opposite sides of noon. So nothing here may
rewrite an azimuth band. The report carries axis and azimuth_status as FIELDS
rather than prose, so a summary cannot drop them.

WHY IT EXISTS
Every illumination claim in this repository is synthetic, from one terrain
generator, and synthetic has failed against real data three times now (the
TMC-2 label GSD; the old unsolved band; minima-loftr passing every accuracy
check on TMC-2 while failing CHECK-03 on 5 of 9 windows). The nine NAC products
configs/regimes.yaml names as its validation set are on disk, unused.

NO GROUND TRUTH IS INVENTED. Real scenes have none. The accuracy measurement is
the PERTURBATION GATE: it injects a known (3, 4) px shift into the real source
and measures how the recovered transform responds. That is a controlled
transformation inside real imagery, needing no external reference. Inlier RMSE
and quality tier are recorded and are NOT accuracy.
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from chandralign.contracts import ImagePlane  # noqa: E402
from chandralign.estimate import robust  # noqa: E402
from chandralign.evaluate import control_gates, quality  # noqa: E402
from chandralign.matching import adapter, classical  # noqa: E402
from chandralign.preprocess.phase_congruency import mind  # noqa: E402
from chandralign.refine import uniformity  # noqa: E402

# NAC CDR geometry, read from the attached PDS3 label and identical across the nine.
LINES, SAMPLES, PIX_OFFSET, SCALE = 52224, 5064, 5064, 3.05185094759972e-05
NULL_BELOW = -32752                      # VALID_MINIMUM; NULL and the saturations sit below
WIN = 512                                # window side, native px (protocol 2)
SEARCH_LINES = 4000                     # coarse search half-range in LINES (protocol 2a)
# The ODE bbox prior is the bounding box of a ROTATED footprint, measured wrong
# by ~7000 lines, so latitude->line is kept as a coarse prior and
# longitude->sample is DISCARDED: the search covers the full sample width.
MAX_SCALE_RATIO = 1.5                   # protocol 2a: above this, a resolution gap
                                        # confounds the illumination question
COARSE_BLOCK = 4                         # block-average factor for the coarse lock
MIN_COARSE_Z = 6.0                       # below this the lock is not trusted; pair excluded
WINDOW_FRACTIONS = (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)     # protocol 2, deterministic
MATCHERS = ["xoftr", "minima-loftr", "eloftr", "aliked-lightglue", "sift-nn"]
BANDS = [(0, 15), (15, 30), (30, 60), (60, 90)]
SOLVED, DEGRADED = 0.90, 0.60            # protocol 6, from unsolved_band_protocol.md
NAC_GSD_M = 0.55                         # nominal, for the gate pipeline only


def load_products():
    """product id -> (path, incidence_deg, bbox) from ODE, for what is on disk."""
    cand = json.loads((ROOT / "data/pairs/lro_nac_candidates.json").read_text())
    on_disk = {p.stem.upper(): p for p in (ROOT / "data/raw/lro/nac").rglob("*.IMG")}
    out = {}
    for r in cand["results"]:
        pid = r["product_id"].split(".")[-1].upper()
        if pid in on_disk:
            out[pid] = (on_disk[pid], float(r["incidence_deg"]),
                        [float(x) for x in r["bbox"]])       # lat_lo, lat_hi, lon_lo, lon_hi
    return out


def product_gsd():
    """ODE Map_resolution, m/px, per product (data/pairs/nac_ode_geometry.json)."""
    f = ROOT / "data/pairs/nac_ode_geometry.json"
    return {k: v["map_resolution"] for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


def pairs_of(products, max_scale_ratio=MAX_SCALE_RATIO):
    """Overlapping pairs, sorted by incidence gap, SCALE-CLEAN only (protocol 2a).

    A pair whose two products differ by more than `max_scale_ratio` in ground
    sampling carries a resolution gap on top of the illumination gap, and a
    failure on it cannot be attributed to illumination -- minima-loftr scores
    0/10 on resolution_2x, so that attribution would be plainly wrong. The cost
    is that the 60-90 deg band empties: both its pairs are 1.56x and 2.29x.
    """
    gsd = product_gsd()
    ids = sorted(products, key=lambda k: products[k][1])
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = products[ids[i]], products[ids[j]]
            lat_lo, lat_hi = max(a[2][0], b[2][0]), min(a[2][1], b[2][1])
            lon_lo, lon_hi = max(a[2][2], b[2][2]), min(a[2][3], b[2][3])
            if lat_hi > lat_lo and lon_hi > lon_lo:
                ga, gb = gsd.get(ids[i]), gsd.get(ids[j])
                ratio = max(ga, gb) / min(ga, gb) if ga and gb else None
                if ratio is not None and ratio > max_scale_ratio:
                    continue
                out.append({"a": ids[i], "b": ids[j], "inc_a": a[1], "inc_b": b[1],
                            "inc_gap": round(abs(a[1] - b[1]), 2),
                            "gsd_a": ga, "gsd_b": gb,
                            "scale_ratio": None if ratio is None else round(ratio, 3),
                            "overlap": [lat_lo, lat_hi, lon_lo, lon_hi],
                            "overlap_deg": [round(lat_hi - lat_lo, 3), round(lon_hi - lon_lo, 3)]})
    return sorted(out, key=lambda p: p["inc_gap"])


def latlon_to_pixel(bbox, lat, lon):
    """The crude prior of protocol 4.1: the ODE bbox as a linear lat/lon frame.

    Ignores rotation and pushbroom distortion, so it only gets us within the
    search margin -- the MIND lock does the rest. NAC line 0 is the NORTH end
    of a descending pass, so latitude decreases with line number.
    """
    lat_lo, lat_hi, lon_lo, lon_hi = bbox
    f_line = (lat_hi - lat) / max(lat_hi - lat_lo, 1e-9)
    f_samp = (lon - lon_lo) / max(lon_hi - lon_lo, 1e-9)
    return int(round(f_line * (LINES - 1))), int(round(f_samp * (SAMPLES - 1)))


def read_window(path, line, samp, half):
    """(image, valid) centred on (line, samp), clipped to the product."""
    arr = np.memmap(path, dtype="<i2", mode="r", offset=PIX_OFFSET, shape=(LINES, SAMPLES))
    l0, l1 = max(0, line - half), min(LINES, line + half)
    s0, s1 = max(0, samp - half), min(SAMPLES, samp + half)
    if l1 - l0 < half or s1 - s0 < half // 2:
        return None, None
    raw = np.asarray(arr[l0:l1, s0:s1], np.float32)
    valid = raw > NULL_BELOW
    return np.where(valid, raw * SCALE, 0.0), valid


def _norm(a, valid):
    v = a[valid] if valid is not None and valid.any() else a
    if v.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(v, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def block(a, f):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def coarse_lock(tpl, region):
    """Integer offset of `tpl` inside `region`, on MIND channels. (dy, dx, z).

    Matcher-independent by design (protocol 4): it must not be able to make one
    matcher look better than another. MIND is ours, needs no weights, and is
    built for exactly the radiometric difference under test here.
    """
    t, r = block(tpl, COARSE_BLOCK), block(region, COARSE_BLOCK)
    if min(r.shape) <= min(t.shape) or min(t.shape) < 16:
        return None
    mt, mr = mind(t).astype(np.float32), mind(r).astype(np.float32)
    total = None
    for c in range(mt.shape[2]):
        res = cv2.matchTemplate(mr[:, :, c], mt[:, :, c], cv2.TM_CCOEFF_NORMED)
        total = res if total is None else total + res
    total /= mt.shape[2]
    _, peak, _, loc = cv2.minMaxLoc(total)
    med, sd = float(np.median(total)), float(total.std())
    z = (peak - med) / sd if sd > 0 else 0.0
    return loc[1] * COARSE_BLOCK, loc[0] * COARSE_BLOCK, z


def run_matcher(name, src_plane, ref_plane):
    if name == "sift-nn":
        return classical.match(src_plane, ref_plane, detector="sift")
    return adapter.match(src_plane, ref_plane, model_name=name, device="cuda")


def test_window(name, src_img, src_ok, ref_img, ref_ok):
    """One matcher on one already-aligned window pair. Returns the record."""
    plane = lambda a, ok: ImagePlane(array=a, valid_mask=ok, shadow_mask=np.zeros(a.shape, bool),
                                     gsd_m=NAC_GSD_M, meta=None, geo=None)
    sp, rp = plane(src_img, src_ok), plane(ref_img, ref_ok)
    t0 = time.perf_counter()
    try:
        ms = run_matcher(name, sp, rp)
    except Exception as exc:
        return {"matcher": name, "error": f"{type(exc).__name__}: {exc}"[:160], "success": False}
    secs = time.perf_counter() - t0

    res = robust.estimate(ms.src_pts, ms.ref_pts, centre=(src_img.shape[1] / 2, src_img.shape[0] / 2))
    n = len(ms.src_pts)
    cov = uniformity.coverage_of(ms.src_pts[res.inlier_mask], src_img.shape, grid=8) if res.inlier_count else 0.0

    gates = control_gates.run_all(
        control_gates.pipeline_from(name, device="cuda", gsd_m=NAC_GSD_M),
        src_img, ref_img, sp, rp)
    pert = gates.to_dict().get("perturbation_sensitivity", {})
    tier = quality.assess(inlier_count=res.inlier_count,
                          inlier_ratio=res.inlier_count / n if n else 0.0,
                          spatial_coverage=cov, model=res.model,
                          scale_ok=res.scale_status not in ("inconsistent", "degenerate"),
                          scale_status=res.scale_status,
                          gates=gates.gates, require_gates=True).tier

    all_gates = all(gates.gates.values())
    # Protocol 6: accept + every gate + the perturbation move recovered within tolerance.
    return {"matcher": name, "matches": n, "inliers": res.inlier_count,
            "inlier_ratio": round(res.inlier_count / n, 4) if n else 0.0,
            "coverage": round(cov, 4), "tier": tier, "accepted": bool(res.ok),
            "gates": gates.gates, "gates_all_passed": all_gates,
            "perturbation_recovered": pert.get("recovered"),
            "perturbation_error_px": pert.get("error_px"),
            "inlier_rmse_px": None,      # self-consistency only; not accuracy (protocol 5)
            "success": bool(res.ok and all_gates),
            "reject_reason": None if (res.ok and all_gates) else
                             ("estimator refused" if not res.ok else
                              "gate: " + ",".join(k for k, v in gates.gates.items() if not v)),
            "seconds": round(secs, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchers", default=",".join(MATCHERS))
    ap.add_argument("--pairs", type=int, default=0, help="limit pair count (debug only)")
    ap.add_argument("--windows", type=int, default=len(WINDOW_FRACTIONS))
    ap.add_argument("--out", default="reports/lro_nac_illumination_validation.json")
    args = ap.parse_args()
    matchers = [m for m in args.matchers.split(",") if m]

    products = load_products()
    pairs = pairs_of(products)
    if args.pairs:
        pairs = pairs[:args.pairs]
    fracs = WINDOW_FRACTIONS[:args.windows]
    print(f"{len(products)} products, {len(pairs)} overlapping pairs, {len(fracs)} windows each, "
          f"{len(matchers)} matchers -> {len(pairs) * len(fracs) * len(matchers)} tests\n")

    rows, skipped = [], []
    for p in pairs:
        pa, pb = products[p["a"]][0], products[p["b"]][0]
        lat_lo, lat_hi, lon_lo, lon_hi = p["overlap"]
        lon_c = 0.5 * (lon_lo + lon_hi)
        for wi, fr in enumerate(fracs):
            lat = lat_lo + fr * (lat_hi - lat_lo)
            la, sa = latlon_to_pixel(products[p["a"]][2], lat, lon_c)
            lb, sb = latlon_to_pixel(products[p["b"]][2], lat, lon_c)
            src, src_ok = read_window(pa, la, sa, WIN // 2)
            # Latitude gives the line prior; the sample prior is discarded, so the
            # search spans the whole swath (protocol 2a).
            reg, reg_ok = read_window(pb, lb, SAMPLES // 2, SEARCH_LINES)
            if src is None or reg is None or src_ok.mean() < 0.9 or reg_ok.mean() < 0.9:
                skipped.append({**{k: p[k] for k in ("a", "b", "inc_gap")}, "window": wi,
                                "why": "window outside the product or too much null"})
                continue
            lock = coarse_lock(_norm(src, src_ok), _norm(reg, reg_ok))
            if lock is None or lock[2] < MIN_COARSE_Z:
                skipped.append({**{k: p[k] for k in ("a", "b", "inc_gap")}, "window": wi,
                                "why": f"coarse lock failed (z={lock[2]:.1f})" if lock else "coarse lock not possible",
                                "coarse_z": None if lock is None else round(lock[2], 2)})
                continue
            dy, dx, z = lock
            ref, ref_ok = reg[dy:dy + WIN, dx:dx + WIN], reg_ok[dy:dy + WIN, dx:dx + WIN]
            if ref.shape != (WIN, WIN):
                skipped.append({**{k: p[k] for k in ("a", "b", "inc_gap")}, "window": wi,
                                "why": "locked window ran off the search region"})
                continue
            s_n, r_n = _norm(src, src_ok), _norm(ref, ref_ok)
            for m in matchers:
                rec = test_window(m, s_n, src_ok, r_n, ref_ok)
                rec.update({k: p[k] for k in ("a", "b", "inc_a", "inc_b", "inc_gap", "overlap_deg", "scale_ratio")})
                rec.update(window=wi, coarse_z=round(z, 2))
                rows.append(rec)
                print(json.dumps({k: rec[k] for k in ("a", "b", "inc_gap", "window", "matcher",
                                                      "success", "tier", "perturbation_error_px",
                                                      "inliers", "seconds") if k in rec}), flush=True)

    # ---- band table, protocol 6 ------------------------------------------------
    print(f"\n=== success rate by INCIDENCE gap band (azimuth NOT validated) ===")
    print(f"{'band':<12}{'n/band':>8}  " + "".join(f"{m[:16]:>18}" for m in matchers))
    band_table = {}
    for lo, hi in BANDS:
        sel = [r for r in rows if lo <= r["inc_gap"] < hi]
        npairs = len({(r["a"], r["b"]) for r in sel})
        line = f"{f'{lo}-{hi} deg':<12}{npairs:>4} pr  "
        band_table[f"{lo}-{hi}"] = {"scene_pairs": npairs, "thin": npairs < 3, "matchers": {}}
        for m in matchers:
            rs = [r for r in sel if r["matcher"] == m]
            ok = sum(r["success"] for r in rs)
            rate = ok / len(rs) if rs else None
            verdict = "--" if rate is None else ("solved" if rate >= SOLVED else
                                                 "degraded" if rate >= DEGRADED else "unsolved")
            band_table[f"{lo}-{hi}"]["matchers"][m] = {
                "n": len(rs), "success": ok, "rate": None if rate is None else round(rate, 3),
                "verdict": verdict}
            line += f"{f'{ok}/{len(rs)} {verdict}':>18}"
        print(line + ("   <- THIN, provisional" if npairs < 3 else ""))

    print("\n=== median perturbation error, px (the only known-truth measure here) ===")
    for m in matchers:
        e = [r["perturbation_error_px"] for r in rows
             if r["matcher"] == m and r.get("perturbation_error_px") is not None]
        print(f"   {m:<20} {np.median(e):>7.2f} px over {len(e)} windows" if e else f"   {m:<20}   no data")

    if skipped:
        print(f"\n{len(skipped)} window-pairs excluded before any matcher ran "
              f"(identically for all matchers, protocol 4):")
        for s in skipped[:10]:
            print(f"   {s['a']} / {s['b']}  gap {s['inc_gap']:>5.1f}  window {s['window']}  {s['why']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "measured",
        "axis": "incidence_difference",
        "azimuth_status": "not_validated",
        "azimuth_note": ("ODE supplies incidence, emission and phase but not solar azimuth, and "
                         "azimuth cannot be derived from incidence: two scenes can share an "
                         "incidence angle and be lit from opposite sides of noon. No azimuth "
                         "band may be rewritten from this result."),
        "protocol": "docs/nac_illumination_protocol.md",
        "ground_truth": ("none available; the accuracy measure is the perturbation gate's "
                         "recovery of a known (3,4) px shift injected into the real source"),
        "products": {k: {"incidence_deg": v[1], "bbox": v[2]} for k, v in products.items()},
        "pairs": pairs, "matchers": matchers,
        "window_px": WIN, "window_fractions": list(fracs),
        "thresholds": {"solved": SOLVED, "degraded": DEGRADED, "min_coarse_z": MIN_COARSE_Z},
        "band_table": band_table, "excluded": skipped, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(rows)} tests, {len(skipped)} excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
