"""Protocol v3: does per-point refinement put DELIVERED match points closer to the truth?

    python scripts/verify_point_refinement.py              # 9 real TMC-2 windows, eloftr, GPU

Frozen before running: docs/pipeline_stages_protocol.md, section "v3".

HOW THE TRUTH IS MADE (as scripts/verify_subpixel.py does for PREC-06)
A real TMC-2 window is block-averaged BLOCK x BLOCK twice, the second copy
starting an integer number of native pixels later, so the pair differs by EXACTLY
SHIFT coarse px, with no interpolation model to flatter any method. Independent
sensor-realistic noise is added to each copy (verify_subpixel's noise model).
The real matcher matches the pair; pipeline.fine_stage delivers uniform points
with per-point refinement off, then on. Each delivered point's error is
|(ref - src) - SHIFT|.

WHAT IT DOES NOT TEST: illumination change, parallax, different sensors. It
measures how precisely a delivered point is LOCATED -- the same limit PREC-06 states.
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

import numpy as np  # noqa: E402

from chandralign import synth  # noqa: E402
from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.io.evidence import evidence_label  # noqa: E402
from chandralign.matching import adapter  # noqa: E402
from chandralign.pipeline import fine_stage  # noqa: E402
from verify_subpixel import native_noise_ratio  # noqa: E402  (the SAME noise model as PREC-06)

SHIFT = (0.25, -0.75)          # coarse px; exact multiples of 1/BLOCK
BLOCK = 4
COARSE = 768                   # coarse window side -> 3072 native px


def norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def plane(a: np.ndarray) -> synth.ImagePlane:
    return synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool), shadow_mask=np.zeros(a.shape, bool),
                            gsd_m=5.0 * BLOCK, meta=None, geo=None)


def exact_pair(meta, row0: int, col0: int, seed: int):
    sx, sy = int(round(-SHIFT[0] * BLOCK)), int(round(-SHIFT[1] * BLOCK))
    pad = max(abs(sx), abs(sy)) + 1
    n = COARSE * BLOCK
    big = pds_raster.read_raster(meta, pds_raster.Window(row0 - pad, col0 - pad, n + 2 * pad, n + 2 * pad)
                                 ).astype(np.float64)
    if (big <= 0).mean() > 0.001:
        return None
    cut = lambda ox, oy: big[pad + oy:pad + oy + n, pad + ox:pad + ox + n] \
        .reshape(COARSE, BLOCK, COARSE, BLOCK).mean(axis=(1, 3))
    a, b = cut(0, 0), cut(sx, sy)
    ratio = native_noise_ratio(big[pad:pad + n, pad:pad + n])
    rng = np.random.default_rng(seed)
    sig = ratio * a.std()
    return norm(a + rng.normal(0, sig, a.shape)), norm(b + rng.normal(0, sig, b.shape)), ratio


def errors(fr) -> np.ndarray:
    return np.hypot(*((fr.control_ref - fr.control_src) - np.array(SHIFT)).T)


def stats(e: np.ndarray) -> dict:
    return {"n": int(len(e)), "median": round(float(np.median(e)), 4),
            "p90": round(float(np.percentile(e, 90)), 4), "max": round(float(e.max()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, default=9)
    ap.add_argument("--matcher", default="eloftr")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/point_refinement_verification.json")
    args = ap.parse_args()

    meta = parse_label(evidence_label("TMC2", ROOT))
    L, S = meta.array_shape
    n = COARSE * BLOCK
    col0 = (S - n) // 2
    rows = np.linspace(5000, L - n - 5000, args.windows).astype(int)

    results = []
    for i, r0 in enumerate(rows):
        t0 = time.perf_counter()
        pair = exact_pair(meta, int(r0), col0, seed=1000 + i)
        if pair is None:
            results.append({"row": int(r0), "status": "skipped: invalid pixels"})
            continue
        a, b, ratio = pair
        ms = adapter.match(plane(a), plane(b), model_name=args.matcher, device=args.device)
        centre = (a.shape[1] / 2.0, a.shape[0] / 2.0)
        off = fine_stage(ms, a, b, centre=centre, flags={"geometry_filter": False, "uniformity": True, "subpixel": False})
        on = fine_stage(ms, a, b, centre=centre, flags={"geometry_filter": False, "uniformity": True, "subpixel": True})
        all_inl = fine_stage(ms, a, b, centre=centre, flags={"geometry_filter": False, "uniformity": False, "subpixel": False})
        eo, en = errors(off), errors(on)
        u = on.stages["uniformity"]
        from chandralign.refine import uniformity as un
        k = int(__import__("chandralign.config", fromlist=["get"]).get("uniformity.top_k_per_cell", 6))
        cy, cx = un._cell_index(on.control_src, a.shape, u["grid"])
        row = {"row": int(r0), "native_noise_ratio": round(ratio, 4), "matches": int(len(ms.src_pts)),
               "inliers": on.inlier_count,
               "unrefined": stats(eo), "refined": stats(en),
               "subpixel_better": bool(np.median(en) < np.median(eo) and np.percentile(en, 90) <= np.percentile(eo, 90)),
               "uniformity": {"delivered": u["kept"], "coverage": round(on.coverage, 4),
                              "coverage_all_inliers": round(all_inl.coverage, 4),
                              "max_per_cell": int(np.bincount(cy * u["grid"] + cx).max()), "top_k": k},
               "seconds": round(time.perf_counter() - t0, 1)}
        row["uniformity_ok"] = bool(abs(row["uniformity"]["coverage"] - row["uniformity"]["coverage_all_inliers"]) < 1e-9
                                    and row["uniformity"]["max_per_cell"] <= k)
        results.append(row)
        print(json.dumps(row), flush=True)

    done = [r for r in results if "refined" in r]
    better = sum(r["subpixel_better"] for r in done)
    verdict = {
        "subpixel": {"windows_better": better, "of": len(done), "rule": ">= 8 of 9",
                     "default_on": better >= 8 and len(done) == 9},
        "uniformity": {"windows_ok": sum(r["uniformity_ok"] for r in done), "of": len(done),
                       "default_on": all(r["uniformity_ok"] for r in done) and len(done) == 9},
    }
    print(json.dumps(verdict, indent=1))
    out = ROOT / args.out
    out.write_text(json.dumps({"source": "measured", "protocol": "docs/pipeline_stages_protocol.md#v3",
                               "product": meta.product_id, "shift_coarse_px": SHIFT, "block": BLOCK,
                               "coarse_window_px": COARSE, "matcher": args.matcher,
                               "run": run_record(), "verdict": verdict, "windows": results}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
