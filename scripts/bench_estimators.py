"""Does MAGSAC++ actually beat plain RANSAC here? (ALIGN-01 accept criterion)

    .venv/Scripts/python scripts/bench_estimators.py
    .venv/Scripts/python scripts/bench_estimators.py --seeds 0,1,2 --outliers 0.7

PLAN.md's accept bar for ALIGN-01 is "MAGSAC beats plain RANSAC on a synthetic
set with 70% outliers". configs/default.yaml already ships `method: magsac`, so
that choice is currently an assumption, not a measurement. This measures it.

WHAT IS VARIED, AND WHY EACH AXIS EXISTS

outlier fraction   0.3 .. 0.9. The stated bar is one point on this curve; a
                   single point cannot say whether the ordering is stable.

outlier KIND       the axis that decides the answer.
                   `uniform`    wrong match lands anywhere -- the textbook case.
                                Outliers agree with nothing, so any consensus
                                method finds the one real consensus.
                   `structured` wrong matches agree with EACH OTHER on a second,
                                plausible transform. This is failure mode #12 on
                                repetitive terrain: crater fields and mare ridges
                                produce false matches that are mutually
                                consistent, so the wrong answer has a consensus
                                too. Benchmarking only `uniform` would report a
                                difficulty this pipeline never faces.

reprojection       1, 3, 10 px. MAGSAC++'s actual published claim is not "more
threshold          accurate at the right threshold" but "does not need the right
                   threshold": it marginalises over the inlier noise scale
                   instead of taking it as a parameter. Our threshold is a
                   config constant (estimate.reproj_threshold_px: 3.0) applied
                   across instrument pairs whose true noise differs, so
                   threshold sensitivity is the property that matters to us,
                   not peak accuracy.

iteration budget   10^4, 10^5, 10^6, at 90% outliers only. Without this axis
                   the 90% column is uninterpretable: a 4-point sample is
                   all-inlier about once in 10^4 draws there, so a 10^4 cap
                   sits on the edge of solvable and any ordering measured at it
                   ranks sampling budgets rather than estimators. It is also
                   the axis that found estimate.max_iters was set too low.

WHAT THIS BENCHMARK IS ALLOWED TO CONCLUDE
It compares robust DRIVERS, and only those. The structured-outlier collapse it
reports is a property of consensus estimation itself, not of any driver, so
"no method wins there" must not be read as "this needs a better estimator".

SCORED AGAINST TRUTH (grading only -- never an input)
  rmse_px      transform error over a 16x16 grid against the true homography
  precision    of the points the estimator called inliers, how many really were
  recall       of the true inliers, how many it found
  accepted     what the shipped pipeline would have done with this match set
  false_conf   accepted AND more than 2 px wrong -- the expensive failure

ALL SYNTHETIC: correspondence sets, not images. This measures the estimator in
isolation, with the matcher's own errors removed. A matcher-in-the-loop number
is a different claim and is measured by scripts/bench_matchers.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign import synth  # noqa: E402
from chandralign.estimate import models, robust  # noqa: E402

SHAPE = (512, 512)
CENTRE = (SHAPE[1] / 2, SHAPE[0] / 2)
BAD_PX = 2.0
N_POINTS = 200
NOISE_PX = 0.5
METHODS = ["magsac", "ransac", "usac_accurate"]
THRESHOLDS = [1.0, 3.0, 10.0]
OUTLIER_FRACTIONS = [0.3, 0.5, 0.7, 0.8, 0.9]
KINDS = ["uniform", "structured"]

# The true transform every run must recover, and the decoy the structured
# outliers agree on. The decoy is deliberately PLAUSIBLE -- a similar rotation
# and scale, displaced -- because an implausible decoy would be thrown out by
# geometry alone and would not test consensus at all.
TRUTH = dict(scale=1.08, rot_deg=6.0, shift=(12.0, -7.0))
DECOY = dict(scale=1.02, rot_deg=-4.0, shift=(-35.0, 28.0))


def make_correspondences(outlier_fraction, kind, seed, n=N_POINTS, noise=NOISE_PX):
    """(src, ref, truth_H, is_inlier). Truth is exact; noise is on the inliers only."""
    rng = np.random.default_rng(seed)
    H = synth.homography(centre=CENTRE, **TRUTH)
    src = rng.uniform(0, SHAPE[0] - 1, size=(n, 2))
    n_out = int(round(n * outlier_fraction))
    is_inlier = np.ones(n, bool)
    is_inlier[rng.permutation(n)[:n_out]] = False

    ref = synth.transform_points(H, src) + rng.normal(0, noise, size=(n, 2))
    if n_out:
        bad = ~is_inlier
        if kind == "uniform":
            ref[bad] = rng.uniform(0, SHAPE[0] - 1, size=(n_out, 2))
        elif kind == "structured":
            decoy = synth.homography(centre=CENTRE, **DECOY)
            ref[bad] = synth.transform_points(decoy, src[bad]) + rng.normal(0, noise, size=(n_out, 2))
        else:
            raise ValueError(f"unknown outlier kind {kind!r}")
    return src, ref, H, is_inlier


def grid_rmse(model, H):
    g = np.linspace(0, SHAPE[0] - 1, 16)
    gx, gy = np.meshgrid(g, g)
    p = np.c_[gx.ravel(), gy.ravel()]
    return float(np.sqrt(np.mean(np.sum((models.apply(model, p) - synth.transform_points(H, p)) ** 2, axis=1))))


def one(method, thresh, outlier_fraction, kind, seed, max_iters=None):
    src, ref, H, is_inlier = make_correspondences(outlier_fraction, kind, seed)
    t0 = time.perf_counter()
    # kind="homography" fixes the model so the ONLY thing varying is the robust
    # driver. Leaving model selection on would mix two effects in one number.
    res = robust.estimate(src, ref, kind="homography", method=method,
                          reproj_threshold=thresh, centre=CENTRE, max_iters=max_iters)
    secs = time.perf_counter() - t0

    rmse = None if res.model is None else grid_rmse(res.model, H)
    found = res.inlier_mask.astype(bool)
    tp = int(np.count_nonzero(found & is_inlier))
    return {
        "method": method, "threshold_px": thresh, "outlier_fraction": outlier_fraction,
        "kind": kind, "seed": seed,
        "rmse_px": None if rmse is None else round(rmse, 4),
        "inliers": int(found.sum()), "true_inliers": int(is_inlier.sum()),
        "precision": round(tp / max(int(found.sum()), 1), 4),
        "recall": round(tp / max(int(is_inlier.sum()), 1), 4),
        "accepted": bool(res.ok),
        "success": bool(res.ok and rmse is not None and rmse < BAD_PX),
        "false_confidence": bool(res.ok and (rmse is None or rmse > BAD_PX)),
        "seconds": round(secs, 4),
        "max_iters": max_iters,
    }


def budget_sweep(methods, seeds, budgets=(10_000, 100_000, 1_000_000),
                 fraction=0.9, thresh=3.0, kind="uniform"):
    """Success at a fixed outlier rate as the iteration cap rises.

    This axis exists because without it the 90% column below is unreadable. A
    4-point sample is all-inlier about once in 10^4 draws at 90% outliers, so a
    10000 cap sits exactly on the edge of solvable and any ordering measured
    there is an ordering of sampling budgets, not of estimators.
    """
    rows = []
    for n in budgets:
        for m in methods:
            rs = [one(m, thresh, fraction, kind, s, max_iters=n) for s in seeds]
            ok = [r for r in rs if r["success"]]
            rows.append({"max_iters": n, "method": m,
                         "success": len(ok), "of": len(rs),
                         "median_rmse_px": round(float(np.median([r["rmse_px"] for r in ok])), 4) if ok else None,
                         "median_seconds": round(float(np.median([r["seconds"] for r in rs])), 4)})
    return rows


def _fmt(v, nd=2):
    return "  --  " if v is None else f"{v:>6.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--thresholds", default=",".join(str(t) for t in THRESHOLDS))
    ap.add_argument("--outliers", default=",".join(str(f) for f in OUTLIER_FRACTIONS))
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--no-budget-sweep", dest="budget_sweep", action="store_false",
                    help="skip the iteration-budget axis (it is the slow one: plain "
                         "RANSAC at 10^6 iterations costs about 3 s per fit)")
    ap.add_argument("--out", default="reports/estimator_benchmark.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [m for m in args.methods.split(",") if m]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    fractions = [float(f) for f in args.outliers.split(",")]
    kinds = [k for k in args.kinds.split(",") if k]

    rows = [one(m, t, f, k, s)
            for k in kinds for f in fractions for t in thresholds for m in methods for s in seeds]

    cell = defaultdict(list)
    for r in rows:
        cell[(r["kind"], r["outlier_fraction"], r["threshold_px"], r["method"])].append(r)

    # 1. The stated bar, and the curve around it.
    for kind in kinds:
        print(f"\n=== {kind} outliers: success rate, and median px error of the successes ===")
        print(f"{'thresh':>7} {'method':<15}" + "".join(f"{f'{int(f*100)}% out':>18}" for f in fractions))
        for t in thresholds:
            for m in methods:
                line = f"{t:>7.0f} {m:<15}"
                for f in fractions:
                    rs = cell[(kind, f, t, m)]
                    ok = [x for x in rs if x["success"]]
                    med = np.median([x["rmse_px"] for x in ok]) if ok else None
                    line += f"{len(ok)}/{len(rs)} {_fmt(med, 3):>11}"
                print(line)
            print()

    # 2. Threshold sensitivity -- MAGSAC's real claim. Spread of the success
    #    rate across thresholds, at each outlier level, per method.
    print("=== threshold sensitivity: success rate at 1 / 3 / 10 px ===")
    for kind in kinds:
        print(f"\n  {kind}")
        print(f"{'':>6}{'method':<16}" + "".join(f"{f'{int(f*100)}%':>22}" for f in fractions))
        for m in methods:
            line = f"{'':>6}{m:<16}"
            for f in fractions:
                rates = []
                for t in thresholds:
                    rs = cell[(kind, f, t, m)]
                    rates.append(sum(x["success"] for x in rs) / max(len(rs), 1))
                line += "".join(f"{r:>7.2f}" for r in rates) + " "
            print(line)

    # 3. The expensive failure: accepted by the quality path AND wrong.
    print("\n=== false confidence (estimator accepted a result > 2 px wrong) ===")
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        fc = [r for r in rs if r["false_confidence"]]
        by_kind = {k: sum(1 for r in fc if r["kind"] == k) for k in kinds}
        print(f"   {m:<16} {len(fc):>4}/{len(rs)}   " + "  ".join(f"{k}: {v}" for k, v in by_kind.items()))

    print("\n=== median runtime per fit (s) ===")
    for m in methods:
        rs = [r["seconds"] for r in rows if r["method"] == m]
        print(f"   {m:<16} {np.median(rs):.4f}")

    # 4. Iteration budget: the axis that decides whether the 90% column above
    #    is measuring estimators or sampling luck.
    budget = budget_sweep(methods, seeds) if args.budget_sweep else []
    if budget:
        print(f"\n=== iteration budget, {int(0.9*100)}% uniform outliers ===")
        print(f"{'max_iters':>10} {'method':<16}{'success':>10}{'median px':>11}{'median s':>11}")
        for r in budget:
            med = "--" if r["median_rmse_px"] is None else f"{r['median_rmse_px']:.3f}"
            print(f"{r['max_iters']:>10} {r['method']:<16}{r['success']:>7}/{r['of']:<2}{med:>11}{r['median_seconds']:>11.4f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "synthetic",
        "what": "robust estimator comparison on synthetic correspondence sets, model fixed to homography",
        "n_points": N_POINTS, "noise_px": NOISE_PX, "bad_px": BAD_PX,
        "truth": TRUTH, "decoy": DECOY, "shape": list(SHAPE),
        "seeds": seeds, "rows": rows, "budget_sweep": budget}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
