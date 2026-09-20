"""Measure the accuracy of every completed Part 1 step on the real data, repeatedly.

Each check compares our code with an INDEPENDENT reference (a different code path,
the publisher's own checksum or statistics, or a Monte-Carlo estimate) and counts
correct outputs. Every check is run --repeats times; the report states how many
runs gave the identical result, so a pass by luck or a non-deterministic output shows.

    python scripts/accuracy_report.py              # writes reports/accuracy.md
    python scripts/accuracy_report.py --repeats 10 --samples 500

Numbers in the report are produced by this script, never typed in (PLAN.md rule H1).
Timings are reported separately and excluded from the repeatability comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapefile

from chandralign.geometry.footprint import check_overlap, footprint_of
from chandralign.geometry.solar import azimuth_difference, illumination_delta, scene_illumination
from chandralign.io.instruments import detect_instrument
from chandralign.io.pds_label import parse_label, read_array_layout, read_pds3_image_info
from chandralign.io.pds_raster import Window, check_attached_pds3_header, read_raster, verify_raster
from chandralign.io.tiling import global_to_tile, tile_grid, tile_to_global

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def products() -> dict[str, Path]:
    g = lambda p: sorted(RAW.glob(p))[0]
    return {
        "OHRC": g("ch2/ohrc/products/*/data/calibrated/*/*_d_img_d18.xml"),
        "TMC2": g("ch2/tmc2/products/*/data/calibrated/*/*_d_img_d18.xml"),
        "IIRS": g("ch2/iirs/products/*/data/calibrated/*/*_d_img_d18.xml"),
        "NAC_2022": RAW / "lro/nac/nac.m1417360906lc/M1417360906LC.XML",
        "NAC_2009": RAW / "lro/nac/nac.m102000149rc/M102000149RC.XML",
        "TC_N": RAW / "selene/tc/TCO_MAP_02_N03E021N00E024SC.lbl",
        "TC_S": RAW / "selene/tc/TCO_MAP_02_N00E021S03E024SC.lbl",
    }


# ============================================================================= checks
# Each returns (result_dict_compared_across_repeats, timing_dict_not_compared).

def check_camera_detection(samples: int, rng):
    """Every product ID in ISRO's archive index, plus every image product we hold."""
    folder_camera = {"ohrc": "OHRC", "tmc2": "TMC2", "iirs": "IIRS"}
    total = correct = 0
    wrong = []
    for shp in sorted(RAW.glob("ch2/*/shapefiles/*/*.shp")):
        if shp.stem.endswith(("_np", "_sp")):
            continue                      # polar copies of the same products
        expected = folder_camera[shp.parts[-4]]
        for rec in shapefile.Reader(str(shp)).iterRecords(fields=["PRODUCT_ID"]):
            total += 1
            try:
                ok = detect_instrument(rec[0]) == expected
            except Exception:
                ok = False
            correct += ok
            if not ok and len(wrong) < 5:
                wrong.append(rec[0])
    for name, label in products().items():
        total += 1
        expected = {"TC_N": "TC", "TC_S": "TC", "NAC_2022": "NAC", "NAC_2009": "NAC"}.get(name, name)
        correct += detect_instrument(Path(read_array_layout(label).raster_path).name) == expected
    return {"correct": correct, "total": total, "examples_wrong": wrong}, {}


_RX = lambda tag: re.compile(rf"<(?:\w+:)?{tag}(?:\s[^>]*)?>\s*([^<]+?)\s*</(?:\w+:)?{tag}>")


def check_label_fields(samples: int, rng):
    """Parsed CH-2 fields vs an independent plain-text regex read of the same XML."""
    total = correct = 0
    mismatches = []
    for name in ("OHRC", "TMC2", "IIRS"):
        label = products()[name]
        text = label.read_text(encoding="utf-8")
        meta = parse_label(label)
        grab = lambda tag, i=0: _RX(tag).findall(text)[i]
        axes = dict(zip([a.lower() for a in _RX("axis_name").findall(text)], map(int, _RX("elements").findall(text))))
        system = text[text.index("System_Level_Coordinates"):text.index("Refined_Corner_Coordinates")]
        corner = lambda c, k: float(_RX(f"{c}_{k}").findall(system)[0])
        expected = {
            "product_id": grab("file_name").rsplit(".", 1)[0],
            "array_shape": (axes["line"], axes["sample"]),
            "n_bands": axes.get("band", 1),
            "gsd_m": float(grab("pixel_resolution")),
            "sub_solar_azimuth_deg": float(grab("sun_azimuth")),
            "solar_incidence_deg": float(grab("solar_incidence")),
            "acquisition_utc": grab("start_date_time"),
            "corner_latlon": [(corner(c, "latitude"), corner(c, "longitude"))
                              for c in ("upper_left", "upper_right", "lower_right", "lower_left")],
        }
        for field, value in expected.items():
            total += 1
            ok = getattr(meta, field) == value
            correct += ok
            if not ok:
                mismatches.append(f"{name}.{field}")
    return {"correct": correct, "total": total, "mismatches": mismatches}, {}


def check_pixels(samples: int, rng):
    """Random single pixels via read_raster vs a hand-computed seek/read of the file."""
    total = correct = 0
    latencies = []
    for name, label in products().items():
        layout = read_array_layout(label)
        meta = parse_label(label)
        names = [a.lower() for a in layout.axis_names]
        item = np.dtype(layout.dtype).itemsize
        lines, samples_ = meta.array_shape
        with layout.raster_path.open("rb") as fh:
            for _ in range(samples):
                r, c = int(rng.integers(lines)), int(rng.integers(samples_))
                band = int(rng.integers(meta.n_bands))
                t = time.perf_counter()
                got = read_raster(meta, Window(r, c, 1, 1), bands=band if meta.n_bands > 1 else None)
                latencies.append(time.perf_counter() - t)
                idx = {"band": band, "line": r, "sample": c}
                flat = 0
                for axis, size in zip(names, layout.shape):
                    flat = flat * size + idx[axis]
                fh.seek(layout.offset_bytes + flat * item)
                want = np.frombuffer(fh.read(item), layout.dtype)[0]
                total += 1
                correct += bool(got.reshape(-1)[0] == want)
    return ({"correct": correct, "total": total},
            {"median_ms": 1000 * statistics.median(latencies), "p95_ms": 1000 * float(np.percentile(latencies, 95))})


def check_checksums(samples: int, rng):
    """Publisher's own MD5s: whole file (ISRO, NASA) and pixels-only (NASA header)."""
    results = {}
    for name in ("OHRC", "TMC2", "IIRS", "NAC_2022", "NAC_2009"):
        results[f"{name} file md5"] = verify_raster(products()[name], check_md5=True)["md5_checked"]
    for name in ("NAC_2022", "NAC_2009"):
        results[f"{name} pixel md5"] = check_attached_pds3_header(products()[name], check_md5=True)["pixel_md5_checked"]
    return {"correct": sum(results.values()), "total": len(results)}, {}


def check_selene_statistics(samples: int, rng):
    """Our pixels vs the MINIMUM/MAXIMUM/AVERAGE/STDEV JAXA wrote in each label."""
    worst = 0.0
    matched = total = 0
    for name in ("TC_N", "TC_S"):
        label = products()[name]
        info = read_pds3_image_info(label)
        px = read_raster(label).astype(np.float64)
        v = px[px != info["dummy"]]
        ours = {"minimum": v.min(), "maximum": v.max(), "average": v.mean(), "stdev": v.std()}
        for k, val in ours.items():
            # JAXA prints 6 decimals; agreement to that precision counts as exact.
            err = abs(val - float(info[k]))
            worst = max(worst, err)
            total += 1
            matched += err <= 5e-7
    return {"correct": matched, "total": total, "max_abs_error": round(worst, 9)}, {}


def check_tiling(samples: int, rng):
    """Grid coverage of every real product, and tile<->product round-trip error."""
    covered = 0
    for name, label in products().items():
        lines, samples_ = parse_label(label).array_shape
        grid = tile_grid(lines, samples_)
        rows = np.zeros(lines, bool)
        cols = np.zeros(samples_, bool)
        for w in grid:
            rows[w.row:w.row + w.height] = True
            cols[w.col:w.col + w.width] = True
        covered += bool(rows.all() and cols.all())
    pts = rng.random((samples * 100, 2)) * 1024
    origins = [(int(rng.integers(0, 160000)), int(rng.integers(0, 12000))) for _ in range(20)]
    err = max(float(np.max(np.abs(global_to_tile(tile_to_global(pts, o), o) - pts))) for o in origins)
    return {"correct": covered, "total": len(products()), "roundtrip_max_error_px": float(f"{err:.3g}")}, {}


def check_sun_consistency(samples: int, rng):
    """Label incidence vs 90 - label elevation (two independent label fields)."""
    worst = 0.0
    for name in ("OHRC", "TMC2", "IIRS"):
        text = products()[name].read_text(encoding="utf-8")
        elevation = float(_RX("sun_elevation").findall(text)[0])
        ill = scene_illumination(parse_label(products()[name]))
        worst = max(worst, abs(ill.incidence_deg + elevation - 90.0))
    sym = all(azimuth_difference(a, b) == azimuth_difference(b, a) <= 180
              for a, b in rng.uniform(0, 360, (samples, 2)))
    d = illumination_delta(parse_label(products()["OHRC"]), parse_label(products()["TMC2"]))
    return {"incidence_plus_elevation_max_error_deg": round(worst, 9),
            "azimuth_difference_symmetric": sym, "ohrc_vs_tmc2_d_azimuth": round(d["d_azimuth_deg"], 6)}, {}


def _monte_carlo_fraction(small, large, n, rng):
    """Area-weighted Monte-Carlo share of `small` covered by `large` -- independent of our projection."""
    from shapely.geometry import Point
    minx, miny, maxx, maxy = small.bounds
    hit = weight = 0.0
    while weight == 0.0 or n > 0:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if small.contains(Point(x, y)):
            w = math.cos(math.radians(y))          # area weight on a sphere
            weight += w
            hit += w * large.contains(Point(x, y))
            n -= 1
    return hit / weight


def check_overlap_accuracy(samples: int, rng):
    """check_overlap() vs an area-weighted Monte-Carlo estimate, for every real pair."""
    metas = {k: parse_label(v) for k, v in products().items()}
    pairs = [("OHRC", "TMC2"), ("OHRC", "IIRS"), ("OHRC", "NAC_2009"), ("OHRC", "NAC_2022"),
             ("OHRC", "TC_N"), ("OHRC", "TC_S"), ("TMC2", "TC_N"), ("TMC2", "TC_S"),
             ("TMC2", "NAC_2022"), ("IIRS", "NAC_2009"), ("NAC_2022", "TC_N")]
    errors, decisions = {}, 0
    n = samples * 20
    for a, b in pairs:
        c = check_overlap(metas[a], metas[b])
        fa, fb = footprint_of(metas[a]).polygon, footprint_of(metas[b]).polygon
        small, large = (fa, fb) if c.src_area_km2 <= c.ref_area_km2 else (fb, fa)
        mc = _monte_carlo_fraction(small, large, n, rng)
        errors[f"{a}/{b}"] = abs(c.fraction_of_smaller - mc) * 100
        decisions += c.ok == (mc >= c.min_overlap)
    se = 100 * 0.5 / math.sqrt(n)   # worst-case Monte-Carlo standard error, percentage points
    return ({"decisions_agree": decisions, "total": len(pairs),
             "max_error_pp": round(max(errors.values()), 2), "mean_error_pp": round(statistics.mean(errors.values()), 2),
             "monte_carlo_se_pp": round(se, 2)}, {})


def check_geolocation(samples: int, rng):
    """Pixel<->ground models vs the labels' own numbers; reprojection and inverse round trips."""
    from chandralign.geometry.projection import (
        from_map, geolocation_model, load_grid_model, scene_crs, surface_distance_m, to_map,
    )
    metas = {k: parse_label(v) for k, v in products().items()}
    correct = total = 0

    # SELENE map model must reproduce the corner coordinates written in each label.
    for name in ("TC_N", "TC_S"):
        m = metas[name]; model = geolocation_model(m); L, S = m.array_shape
        for (r, c), want in zip([(0, 0), (0, S - 1), (L - 1, S - 1), (L - 1, 0)], m.corner_latlon):
            lat, lon = model.pixel_to_latlon(r, c)
            total += 1; correct += abs(lat - want[0]) < 1e-6 and abs(lon - want[1]) < 1e-6

    shift_m, grid_nodes, inverse_ok, inverse_n = {}, 0, 0, 0
    for name in ("OHRC", "TMC2", "IIRS"):
        m = metas[name]
        grid = load_grid_model(m)
        raw = np.loadtxt(grid_path_of(m), delimiter=",", skiprows=1)
        raw = raw[~np.all(raw == 0.0, axis=1)]
        lat, lon = grid.pixel_to_latlon(raw[:, 3], raw[:, 2])
        ok = (np.abs(lat - raw[:, 1]) < 1e-9) & (np.abs(lon - raw[:, 0]) < 1e-9)
        total += len(ok); correct += int(ok.sum()); grid_nodes += len(ok)
        # inverse (ground -> pixel) on random pixels, both models
        L, S = m.array_shape
        rows, cols = rng.uniform(0, L - 1, samples), rng.uniform(0, S - 1, samples)
        for model in (grid, geolocation_model(m)):
            r2, c2 = model.latlon_to_pixel(*model.pixel_to_latlon(rows, cols))
            good = (np.abs(r2 - rows) < 1e-6) & (np.abs(c2 - cols) < 1e-6)
            inverse_ok += int(good.sum()); inverse_n += len(good)
        # how far the reference-independent model is from ISRO's refined grid
        ind = geolocation_model(m)
        shift_m[name] = round(float(np.median(surface_distance_m(*grid.pixel_to_latlon(rows, cols),
                                                                   *ind.pixel_to_latlon(rows, cols)))), 1)
    total += inverse_n; correct += inverse_ok

    # map projection round trip of random points around each scene
    worst = 0.0
    for m in metas.values():
        if not m.corner_latlon:
            continue
        lat0 = float(np.mean([c[0] for c in m.corner_latlon])); lon0 = float(np.mean([c[1] for c in m.corner_latlon]))
        lat = lat0 + rng.uniform(-2, 2, samples); lon = lon0 + rng.uniform(-2, 2, samples)
        crs = scene_crs(lat0, lon0)
        lat2, lon2 = from_map(*to_map(lat, lon, crs), crs)
        err = np.maximum(np.abs(lat2 - lat), np.abs(lon2 - lon))
        worst = max(worst, float(err.max()))
        total += samples; correct += int((err < 1e-6).sum())

    return ({"correct": int(correct), "total": int(total), "grid_nodes_checked": grid_nodes,
             "reprojection_max_error_deg": float(f"{worst:.3g}"),
             "median_m_independent_vs_refined": shift_m}, {})


def grid_path_of(meta):
    from chandralign.geometry.projection import grid_path
    return grid_path(meta)


def check_shadow_iou(samples: int, rng):
    """Detected shadows vs a mask traced BY HAND on a real OHRC crater crop (PREP-04)."""
    import cv2
    from chandralign.preprocess.shadow_mask import detect_shadows, mask_iou
    crop_path = ROOT / "tests" / "fixtures" / "images" / "ohrc_shadow_crop.npy"
    mask_path = ROOT / "tests" / "fixtures" / "images" / "ohrc_shadow_crop_mask.png"
    if not mask_path.exists():
        return {"correct": 0, "total": 0, "note": "no hand-drawn mask"}, {}
    drawn = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    truth = (drawn[:, :, 2] > 200) & (drawn[:, :, 1] < 60) & (drawn[:, :, 0] < 60)
    crop = np.load(crop_path).astype(float)
    mask = detect_shadows(crop, np.ones(crop.shape, bool)).mask
    agreed = int((mask == truth).sum())
    # Where we disagree, is the pixel actually dark? Sunlit ground is ~38 DN, shadow ~4.
    hand_only, code_only = truth & ~mask, mask & ~truth
    return ({"correct": agreed, "total": int(truth.size),
             "iou": round(mask_iou(mask, truth), 4),
             "hand_only_median_DN": float(np.median(crop[hand_only])) if hand_only.any() else None,
             "code_only_median_DN": float(np.median(crop[code_only])) if code_only.any() else None},
            {})


def check_iirs_composite(samples: int, rng):
    """Repeatable features in the IIRS composite vs the best single band (PREP-06).

    Two images of the same terrain with independent noise: only real features match.
    """
    import cv2
    from chandralign.io.pds_raster import read_raster
    from chandralign.preprocess.iirs_composite import product_band_selection
    from chandralign.preprocess.radiometric import clahe, percentile_stretch

    meta = parse_label(products()["IIRS"])
    sel = product_band_selection(meta)
    cube = read_raster(meta, Window(6000, 0, 256, 250)).astype(np.float64)
    valid = np.ones(cube.shape[1:], bool)
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher()

    def prep(img):
        return (clahe(percentile_stretch(img.astype(np.float32), valid)[0], valid) * 255).astype(np.uint8)

    def matches(a, b):
        ka, da = sift.detectAndCompute(prep(a), None)
        kb, db = sift.detectAndCompute(prep(b), None)
        good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return len(ka), 0
        pa = np.float32([ka[g.queryIdx].pt for g in good])
        pb = np.float32([kb[g.trainIdx].pt for g in good])
        _, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        return len(ka), int(inl.sum())

    best = max(sel.bands, key=lambda b: sel.snr[b])
    kp_single, single = matches(cube[best], cube[best + 1])
    blended = []
    for half in (sel.bands[0::2], sel.bands[1::2]):
        idx = list(half)
        w = sel.snr[idx] ** 2
        scaled = (cube[idx] - sel.signal[idx][:, None, None]) / sel.noise[idx][:, None, None]
        blended.append((scaled * (w / w.sum())[:, None, None]).sum(0))
    kp_comp, paired = matches(*blended)
    return ({"correct": paired, "total": paired + single, "bands_kept": len(sel.bands),
             "composite_matches": paired, "best_single_band_matches": single,
             "raw_keypoints_composite": kp_comp, "raw_keypoints_single": kp_single}, {})


CHECKS = [
    ("3", "Camera detection", "Every product ID in ISRO's archive index + every product we hold", check_camera_detection),
    ("4", "CH-2 label fields", "Parsed fields vs a separate plain-text read of the same XML", check_label_fields),
    ("5", "Pixel reading", "Random pixels vs hand-computed byte offsets (all 7 products)", check_pixels),
    ("5, 7", "Publisher checksums", "ISRO / NASA MD5 of whole files and NASA pixels-only MD5", check_checksums),
    ("6", "SELENE pixel statistics", "Our min/max/mean/std vs the values JAXA wrote in the label", check_selene_statistics),
    ("8", "Tiling", "Grid covers every pixel of every product; point round-trip error", check_tiling),
    ("9", "Sun geometry", "Incidence vs 90 - elevation (two label fields); azimuth maths", check_sun_consistency),
    ("10", "Footprint overlap", "check_overlap vs area-weighted Monte-Carlo, all 11 real pairs", check_overlap_accuracy),
    ("11", "Geolocation / projection", "SELENE corners vs label; every ISRO grid node; ground->pixel inverse; "
     "map round trip", check_geolocation),
("13b", "Shadow detection", "Detected shadow vs a mask traced by hand on a real OHRC crater crop",
     check_shadow_iou),
    ("13c", "IIRS composite", "Repeatable features (independent-noise pair) vs the best single band",
     check_iirs_composite),
]


def fingerprint(result: dict) -> str:
    return hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--samples", type=int, default=500, help="random pixels per product, etc.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "accuracy.md"))
    args = ap.parse_args()

    rows = []
    for step, name, method, fn in CHECKS:
        fps, timings, first = [], [], None
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            rng = np.random.default_rng(args.seed)        # same samples each run -> results must match
            result, timing = fn(args.samples, rng)
            first = first or result
            fps.append(fingerprint(result))
            timings.append(timing)
        identical = sum(f == fps[0] for f in fps)
        elapsed = (time.perf_counter() - t0) / args.repeats
        rows.append((step, name, method, first, identical, timings[0], elapsed))
        print(f"step {step:5} {name:26} {identical}/{args.repeats} identical  {first}")

    lines = [
        "# Part 1 accuracy report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `scripts/accuracy_report.py "
        f"--repeats {args.repeats} --samples {args.samples} --seed {args.seed}`. Every number below is measured; "
        "regenerate rather than edit.",
        "",
        "Each check compares our code with an independent reference and was run "
        f"{args.repeats} times with the same random samples; **Repeatable** counts runs whose result was identical "
        "to the first.",
        "",
        "| Step | Check | Method | Correct | Accuracy | Other measurements | Repeatable | Time / run |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for step, name, method, res, identical, timing, elapsed in rows:
        correct = res.get("correct", res.get("decisions_agree"))
        total = res.get("total")
        acc = f"{100 * correct / total:.2f}%" if total else "–"
        cnt = f"{correct:,} / {total:,}" if total else "–"
        other = {k: v for k, v in res.items() if k not in ("correct", "total", "decisions_agree")}
        other.update({k: round(v, 2) for k, v in timing.items()})
        shown = {k: v for k, v in other.items() if v is not None and not (isinstance(v, list) and not v)}
        other_s = "; ".join(f"{k} = {v}" for k, v in shown.items()) or "–"
        lines.append(f"| {step} | {name} | {method} | {cnt} | {acc} | {other_s} | "
                     f"{identical}/{args.repeats} | {elapsed:.1f} s |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
