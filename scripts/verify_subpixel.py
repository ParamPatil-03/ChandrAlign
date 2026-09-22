"""PREC-06: can the pipeline recover a known sub-pixel shift on a REAL CH-2 image?

    .venv/Scripts/python scripts/verify_subpixel.py
    .venv/Scripts/python scripts/verify_subpixel.py --windows 10 --block 100

Needs the OHRC product in data/raw/ch2/ohrc (see data/manifest.json).

HOW THE SHIFT IS MADE
Two images are block-averaged BLOCK x BLOCK from the same real OHRC strip, the
second starting (-dx*BLOCK, -dy*BLOCK) native pixels later. With BLOCK = 100
that is the target shift (0.37, -0.62) EXACTLY, and it is formed the way a
detector integrates light over a pixel -- no interpolation model anywhere, so
no method is flattered by sharing one. (A Fourier shift would flatter phase
correlation; a bicubic shift would flatter anything fitted with a smooth kernel.)

TWO CONDITIONS, REPORTED SEPARATELY
  clean   both images from the SAME native pixels. Optimistic: their noise is
          identical, which no real repeat acquisition gives you.
  noisy   independent Gaussian noise added to each image so that a COARSE
          pixel is as noisy relative to its texture as a NATIVE OHRC pixel is
          (noise/signal 0.05-0.13, measured from neighbouring-pixel differences).
          A real sensor at the coarse resolution has its own noise, which does
          not shrink because OHRC's pixels were averaged. This is the one to quote.

          (A first version divided the noise by the block size -- correct
          arithmetic for averaging, wrong model of a sensor -- and the "noisy"
          results came out identical to "clean". Caught before reporting.)

WHAT IT DOES NOT TEST
Illumination change, parallax and different sensors. Same image, same sun:
this measures how precisely a method LOCATES a shift, which is the claim
PREC-06 makes. Registration accuracy across real pairs is a separate result
(scripts/register_tmc2_tc.py).

PREC-06 done_when: at least two of the four methods recover (0.37, -0.62) to
< 0.1 px, and Metrics.subpixel_recovery_err_px is filled from a real
measurement on real imagery. Both are written to reports/subpixel_verification.json.
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

import numpy as np  # noqa: E402

from chandralign import compute  # noqa: E402
from chandralign.contracts import Metrics  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.refine import subpixel as sp  # noqa: E402

TARGET = (0.37, -0.62)
# PREC-06 names "the four methods". Which VARIANT represents each was fixed
# BEFORE the OHRC run that followed it, on the crater-field dev fixture only
# (ncc_gaussian_iter 0.012 px mean there, phase_iter 0.048), and the single-pass
# variants stay in every table beside them. The first OHRC run -- single-pass
# only, NOT MET with 1 of 4 -- is kept in git history (9b487d2) as the baseline.
FOUR = {"ncc": "ncc_gaussian_iter", "phase": "phase_iter", "corner": "corner", "ecc": "ecc"}
ALL = ["ncc_parabola", "ncc_gaussian", "ncc_gaussian_iter", "phase", "phase_iter", "corner", "ecc"]


def native_noise_ratio(a: np.ndarray) -> float:
    """Noise std / signal std in native pixels.

    Differences of horizontal neighbours contain texture AND noise; their median
    absolute deviation is dominated by the noise on smooth ground. MAD/0.6745/sqrt2
    is the standard robust per-pixel noise estimate.
    """
    d = np.diff(a, axis=1).ravel()
    sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    return float(sigma / max(a.std(), 1e-9))


def block_pair(meta, row0: int, col0: int, n: int, blk: int, shift: tuple[float, float]):
    sx, sy = int(round(-shift[0] * blk)), int(round(-shift[1] * blk))
    if abs(sx + shift[0] * blk) > 1e-6 or abs(sy + shift[1] * blk) > 1e-6:
        raise ValueError(f"{shift} is not a multiple of 1/{blk}: the shift would not be exact")
    pad = max(abs(sx), abs(sy)) + 1
    big = pds_raster.read_raster(meta, pds_raster.Window(row0 - pad, col0 - pad,
                                                          n * blk + 2 * pad, n * blk + 2 * pad)).astype(np.float64)
    if (big <= 0).mean() > 0.001:
        return None, None, None
    cut = lambda ox, oy: big[pad + oy:pad + oy + n * blk, pad + ox:pad + ox + n * blk] \
        .reshape(n, blk, n, blk).mean(axis=(1, 3))
    return cut(0, 0), cut(sx, sy), big[pad:pad + n * blk, pad:pad + n * blk]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--size", type=int, default=64, help="patch size in coarse pixels")
    ap.add_argument("--extra-shifts", type=int, default=6, help="random exact shifts per window, besides the target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/subpixel_verification.json")
    args = ap.parse_args()

    meta = parse_label(next((ROOT / "data/raw/ch2/ohrc").rglob("*_d_img_d18.xml")))
    L, S = meta.array_shape
    span = args.size * args.block
    if span + 400 > S:
        raise SystemExit(f"a {args.size}x{args.block} patch ({span} px) does not fit OHRC's width {S}")
    rng = np.random.default_rng(args.seed)
    rows = np.linspace(1000, L - span - 1000, args.windows).astype(int)
    col0 = (S - span) // 2

    shifts = [TARGET] + [tuple(np.round(rng.uniform(-1.5, 1.5, 2) * args.block) / args.block)
                         for _ in range(args.extra_shifts)]
    records = []
    t0 = time.perf_counter()
    for r0 in rows:
        ratio = None
        for k, shift in enumerate(shifts):
            ref, mov, native = block_pair(meta, int(r0), int(col0), args.size, args.block, shift)
            if ref is None:
                break
            if ratio is None:
                ratio = native_noise_ratio(native)
            # A coarse pixel exactly as noisy, relative to its texture, as a native
            # OHRC pixel: the sensor-realistic model (see the module docstring).
            sig_coarse = ratio * ref.std()
            for cond in ("clean", "noisy"):
                a, b = ref, mov
                if cond == "noisy":
                    nrng = np.random.default_rng(args.seed + 1000 * k + int(r0))
                    a = ref + nrng.normal(0, sig_coarse, ref.shape)
                    b = mov + nrng.normal(0, sig_coarse, mov.shape)
                for m in ALL:
                    e = sp.estimate(a, b, m)
                    records.append({
                        "row": int(r0), "shift": list(shift), "target": k == 0, "condition": cond,
                        "method": m, "ok": bool(e.ok),
                        "error_px": float(np.hypot(e.dx - shift[0], e.dy - shift[1])) if e.ok else None,
                        "texture_snr": round(float(ref.std() / max(sig_coarse, 1e-12)), 1),
                    })
        print(f"row {r0:>6}: native noise/signal {ratio if ratio is not None else float('nan'):.3f}", flush=True)

    def stats(sel):
        errs = np.array([r["error_px"] for r in sel if r["ok"]])
        fails = sum(not r["ok"] for r in sel)
        if len(errs) == 0:
            return {"n": 0, "failed": fails}
        return {"n": int(len(errs)), "failed": fails, "median": round(float(np.median(errs)), 4),
                "mean": round(float(errs.mean()), 4), "p95": round(float(np.percentile(errs, 95)), 4),
                "max": round(float(errs.max()), 4)}

    summary: dict = {}
    print(f"\ntarget {TARGET}, {args.block}x block averaging -> "
          f"{0.305 * args.block:.1f} m pixels, {args.size}x{args.size} patches, "
          f"{len(set(r['row'] for r in records))} windows")
    for cond in ("clean", "noisy"):
        print(f"\n[{cond}] error on the TARGET shift (px)          | all shifts")
        print(f"{'method':<14}{'median':>8}{'mean':>8}{'p95':>8}{'max':>8}{'fail':>6} | {'median':>7}{'p95':>8}")
        summary[cond] = {}
        for m in ALL:
            tgt = stats([r for r in records if r["condition"] == cond and r["method"] == m and r["target"]])
            allr = stats([r for r in records if r["condition"] == cond and r["method"] == m])
            summary[cond][m] = {"target": tgt, "all_shifts": allr}
            if tgt.get("n"):
                print(f"{m:<14}{tgt['median']:>8.3f}{tgt['mean']:>8.3f}{tgt['p95']:>8.3f}{tgt['max']:>8.3f}"
                      f"{tgt['failed']:>6} | {allr['median']:>7.3f}{allr['p95']:>8.3f}")
            else:
                print(f"{m:<14}  all failed ({tgt['failed']})")

    # PREC-06 verdict on the NOISY condition, which is the honest one.
    passing = [name for name, m in FOUR.items()
               if summary["noisy"][m]["target"].get("n")
               and summary["noisy"][m]["target"]["max"] < 0.1]
    best = min(FOUR.values(), key=lambda m: summary["noisy"][m]["target"].get("median", 9e9))
    metrics = Metrics(subpixel_recovery_err_px=summary["noisy"][best]["target"]["median"], source="measured")
    verdict = {
        "done_when": "at least two of the four methods recover (0.37, -0.62) to < 0.1 px on real imagery",
        "criterion_used": "max error over all windows < 0.1 px, noisy condition",
        "methods_passing": passing,
        "met": len(passing) >= 2,
        "best_method": best,
        "Metrics.subpixel_recovery_err_px": metrics.subpixel_recovery_err_px,
        "Metrics.source": metrics.source,
    }
    print(f"\nPREC-06: {len(passing)}/4 methods recover {TARGET} to < 0.1 px in EVERY window "
          f"(noisy): {passing} -> {'MET' if verdict['met'] else 'NOT MET'}")
    print(f"Metrics.subpixel_recovery_err_px = {metrics.subpixel_recovery_err_px} px "
          f"(median, {best}, noisy, real OHRC)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "measured",
        "product": meta.product_id,
        "target_shift": TARGET, "block": args.block, "patch": args.size,
        "coarse_pixel_m": round(0.305 * args.block, 2),
        "shift_construction": "block-averaged from native OHRC with an integer native offset (exact, no interpolation)",
        "device": compute.classical_device(),
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": verdict, "summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
