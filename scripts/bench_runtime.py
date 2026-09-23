"""How does matcher cost scale with window size, and does tiling recover it?

    .venv/Scripts/python scripts/bench_runtime.py

The criterion is frozen in docs/runtime_budget_protocol.md. This measures; it
does not decide.

WHY THIS EXACT MEASUREMENT
The real TMC-2 -> SELENE run gives the end-to-end picture already
(reports/tmc2_tc_registration.json, tmc2_tc_xoftr.json): 9 windows cost 107.5 s
with eloftr and 948.3 s with xoftr, against a 180 s demo budget. What those runs
CANNOT say is whether the cost is recoverable, and that turns on one number --
the exponent in

    time  ~  (pixels) ** k

  k ~ 1   cost is proportional to area. Tiling a 1536 px window into 9 512 px
          tiles costs the same as doing it whole; tiling buys nothing.
  k ~ 2   cost is quadratic in area -- the signature of dense attention over
          all pixel pairs. Then 9 tiles cost 9x(1/9)^2 = 1/9 of the whole, and
          tiling is most of an order of magnitude.

So this sweeps window size and fits k. Everything else follows from it.

TWO COSTS THAT ARE NOT THE MATCHER, measured separately because they dominate:

  warm-up   the first call loads weights and initialises CUDA. A demo pays it
            once; a product run does not pay it per window. Measured on the
            real run at ~76 s for xoftr, which is 40% of a 180 s budget spent
            before any work happens.
  gates     control_gates.run_all re-runs the matcher about five times per
            window (two nulls, perturbation, identity, masks). They are 54-71%
            of the real end-to-end time, so a slower matcher costs roughly FIVE
            times its own difference. Whether they belong per window or once per
            product is a design question this script does not answer -- it only
            makes the size of it visible.

SYNTHETIC PAIRS, deliberately: a timing scaling law is a property of the model
and the input size, not of the terrain, and synthetic pairs can be generated at
any size on demand. The 1536 px point is cross-checked against the real runs.
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

from chandralign import synth  # noqa: E402
from chandralign.matching import adapter  # noqa: E402

SIZES = [512, 768, 1024, 1536]
METHODS = ["xoftr", "eloftr", "aliked-lightglue"]
REPEATS = 3
DEMO_BUDGET_S = 180.0          # research doc section 38; frozen in the protocol
GATE_MATCHER_CALLS = 5         # control_gates.run_all re-runs the matcher this often
# The size the matcher actually receives on the real TMC-2 -> SELENE path: a
# 1536 px TMC-2 window warped into TC space by 4.99/7.40 = 0.67.
REAL_MATCH_PX = 1024


def _measure_in_process(method, size, repeats):
    """(cold_s, warm_median_s, peak_vram_bytes). Runs in a CHILD process; see below."""
    import torch

    src, ref, _ = synth.make_pair(out_shape=(size, size), rot_deg=8.0, seed=7)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    adapter.match(src, ref, model_name=method, device="cuda")
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        adapter.match(src, ref, model_name=method, device="cuda")
        warm.append(time.perf_counter() - t0)
    return cold, float(np.median(warm)), int(torch.cuda.max_memory_allocated())


def time_one(method, size, repeats=REPEATS):
    """(cold_s, warm_median_s, peak_vram_bytes) measured in a FRESH PROCESS.

    Process isolation is not tidiness, it is correctness. A first version of
    this script ran every (method, size) in one process and produced nonsense:
    aliked-lightglue timed at 16 s for a 512 px pair that
    scripts/select_default_matcher.py measures at 0.36 s, and the fitted
    exponent came out at 3.34, which no attention mechanism explains. The cause
    was that xoftr and eloftr had already hit CUDA OOM at 1536 px earlier in
    the same process, and everything measured afterwards was carrying degraded
    allocator state. An OOM must not be able to reach the next measurement, and
    the only reliable boundary is a process.
    """
    code = (
        "import sys, json; sys.path.insert(0, r'%s')\n"
        "sys.argv = ['x']\n"
        "import importlib.util as u\n"
        "spec = u.spec_from_file_location('br', r'%s'); m = u.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "try:\n"
        "    c, w, v = m._measure_in_process(%r, %d, %d)\n"
        "    print('RESULT' + json.dumps({'cold': c, 'warm': w, 'vram': v}))\n"
        "except Exception as e:\n"
        "    print('RESULT' + json.dumps({'error': type(e).__name__ + ': ' + str(e)[:120]}))\n"
    ) % (str(ROOT / "src"), str(Path(__file__).resolve()), method, size, repeats)
    import subprocess

    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")), None)
    if line is None:
        return {"error": (r.stderr.strip().splitlines() or ["no output"])[-1][:120]}
    return json.loads(line[len("RESULT"):])


def fit_exponent(sizes, times):
    """k in time ~ pixels**k, by least squares on the logs."""
    px = np.array([s * s for s in sizes], float)
    t = np.array(times, float)
    keep = t > 0
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(px[keep]), np.log(t[keep]), 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--out", default="reports/runtime_budget.json")
    ap.add_argument("--from-report", action="store_true",
                    help="re-do the analysis on an existing report without re-timing")
    args = ap.parse_args()
    methods = [m for m in args.methods.split(",") if m]
    sizes = [int(s) for s in args.sizes.split(",")]

    if args.from_report:
        saved = json.loads((ROOT / args.out).read_text(encoding="utf-8"))
        rows, fits = saved["rows"], saved["scaling_exponent_k"]
        for m in methods:
            ok = [(r["size"], r["warm_s"]) for r in rows if r["method"] == m and "warm_s" in r]
            print(f"\n{m}   time ~ pixels**{fits.get(m)}   " +
                  "  ".join(f"{s}px {w:.2f}s" for s, w in ok))
        return _report(rows, fits, methods, args)

    rows, fits = [], {}
    for m in methods:
        print(f"\n{m}")
        print(f"{'size':>7}{'pixels':>12}{'cold s':>10}{'warm s':>10}{'ms/Mpx':>12}{'VRAM MB':>10}")
        warm_by_size, sizes_ok = [], []
        for s in sizes:
            res = time_one(m, s)
            if "error" in res:
                print(f"{s:>7}   {res['error'][:70]}")
                rows.append({"method": m, "size": s, "error": res["error"]})
                continue
            cold, warm, vram = res["cold"], res["warm"], res["vram"]
            mpx = s * s / 1e6
            print(f"{s:>7}{s*s:>12,}{cold:>10.2f}{warm:>10.2f}{warm / mpx * 1000:>12.0f}{vram/1e6:>10.0f}")
            rows.append({"method": m, "size": s, "pixels": s * s, "cold_s": round(cold, 3),
                         "warm_s": round(warm, 3), "peak_vram_mb": round(vram / 1e6, 1)})
            warm_by_size.append(warm)
            sizes_ok.append(s)
        k = fit_exponent(sizes_ok, warm_by_size)
        fits[m] = round(k, 3)
        verdict = ("proportional to area -- tiling buys nothing" if k < 1.25 else
                   "superlinear in area -- tiling helps" if k < 1.75 else
                   "quadratic in area -- tiling is a large win")
        print(f"   time ~ pixels**{k:.2f}   ({verdict})")

    return _report(rows, fits, methods, args)


def _report(rows, fits, methods, args):
    sizes = sorted({r["size"] for r in rows})
    # Analysis is done at the size the matcher ACTUALLY SEES, which is not the
    # window size. register_tmc2_tc.py cuts a 1536 px TMC-2 window and warps it
    # into TC space; TMC-2's ~4.99 m pixel against TC's 7.40 m is a factor of
    # 0.67, so the pair reaching the matcher is about 1030 px. Analysing at 1536
    # would answer a question the pipeline never asks -- and on this GPU it also
    # has no answer, since two of the three matchers cannot allocate it.
    print(f"\n=== tiling a {REAL_MATCH_PX} px pair as 4 x 512 px, predicted from measurement ===")
    print(f"{'method':<20}{'whole s':>10}{'4 tiles s':>12}{'speedup':>10}")
    tiling = {}
    for m in methods:
        w = next((r["warm_s"] for r in rows if r["method"] == m and r["size"] == REAL_MATCH_PX and "warm_s" in r), None)
        t512 = next((r["warm_s"] for r in rows if r["method"] == m and r["size"] == 512 and "warm_s" in r), None)
        if w is None or t512 is None:
            continue
        tiled = 4 * t512
        tiling[m] = {"whole_s": round(w, 3), "four_tiles_s": round(tiled, 3),
                     "speedup": round(w / tiled, 2) if tiled else None}
        print(f"{m:<20}{w:>10.2f}{tiled:>12.2f}{w / tiled:>9.2f}x")
    print("   An upper bound. Tiling costs accuracy at tile edges, needs the "
          "matches merged across tiles, and neither is implemented or measured.")

    # The budget question, using the real 9-window demo path.
    print(f"\n=== the {DEMO_BUDGET_S:.0f}s demo budget, 9 real windows at {REAL_MATCH_PX} px ===")
    print(f"{'method':<20}{'match only':>12}{'+ gates':>12}{'verdict':>28}")
    budget = {}
    for m in methods:
        w = next((r["warm_s"] for r in rows if r["method"] == m and r["size"] == REAL_MATCH_PX and "warm_s" in r), None)
        if w is None:
            continue
        match_only = 9 * w
        with_gates = match_only * (1 + GATE_MATCHER_CALLS)
        fits_budget = "PASS" if with_gates <= DEMO_BUDGET_S else (
            "CONDITIONAL (match only fits)" if match_only <= DEMO_BUDGET_S else "over budget both ways")
        budget[m] = {"match_only_s": round(match_only, 1), "with_gates_s": round(with_gates, 1),
                     "verdict": fits_budget}
        print(f"{m:<20}{match_only:>11.1f}s{with_gates:>11.1f}s{fits_budget:>28}")
    print("   Gates modelled as ~5 extra matcher runs per window. The measured "
          "real split (reports/tmc2_tc_*.json) is the authority; this is the "
          "same arithmetic made explicit, not a second measurement.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "synthetic",
        "what": "matcher runtime vs window size, to decide whether the real-data cost is recoverable",
        "protocol": "docs/runtime_budget_protocol.md",
        "demo_budget_s": DEMO_BUDGET_S, "gate_matcher_calls": GATE_MATCHER_CALLS,
        "repeats": REPEATS, "sizes": sizes,
        "scaling_exponent_k": fits, "tiling_1536_as_9x512": tiling,
        "nine_window_demo": budget, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
