"""Where does RIFT2 earn a place? RIFT2 vs every other matcher, regime by regime (MATCH-06).

    .venv/Scripts/python scripts/bench_rift.py
    .venv/Scripts/python scripts/bench_rift.py --seeds 3,11,29 --methods rift2,sift,xoftr

RIFT2 was added as a CANDIDATE, not a default (configs/regimes.yaml). This
decides its role from measurement: every method runs on the same synthetic
pairs, with the exact transform known, across the regimes that matter:

    same lighting | lighting +30 | +90 | opposite | cross-modal (nonlinear
    sensor response) | cross-modal + opposite sun | 2x resolution gap |
    low texture

success       = estimator accepted AND transform error < 2 px (against truth)
false conf.   = quality gate ACCEPTED a result that is > 2 px wrong
Ground truth grades the runs; it is never an input to any method.

ALL SYNTHETIC. Real-data RIFT2 results are a separate run and a separate claim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign import synth  # noqa: E402
from chandralign.estimate import models, robust  # noqa: E402
from chandralign.evaluate import quality  # noqa: E402
from chandralign.matching import adapter, classical, rift  # noqa: E402
from chandralign.refine import uniformity  # noqa: E402

SHAPE = (512, 512)
BAD_PX = 2.0
REGIMES = {
    "same_lighting":        dict(rot_deg=8.0),
    "lighting_+30":         dict(rot_deg=8.0, sun_src=(165, 40)),
    "lighting_+90":         dict(rot_deg=8.0, sun_src=(225, 45)),
    "lighting_opposite":    dict(rot_deg=8.0, sun_src=(315, 45)),
    "cross_modal":          dict(rot_deg=8.0, cross_modal=True),
    "cross_modal+opposite": dict(rot_deg=8.0, cross_modal=True, sun_src=(315, 45)),
    "resolution_2x":        dict(rot_deg=8.0, scale=2.0),
    "low_texture":          dict(rot_deg=8.0, n_craters=6),
}
METHODS = ["sift", "rift2", "rift2-mim", "aliked-lightglue", "eloftr", "xoftr",
           "matchanything-eloftr", "minima-loftr"]


# The published Python RIFT2 (canyagmur/RIFT2-multimodal-matching-rotation-python),
# pinned, BENCHMARK ONLY: the repository has no licence, so it is cloned outside
# this repo and never vendored or shipped (the CHECK-10 rule).
RIFT2_REF_DIR = Path("C:/chandralign-data/reference/rift2-python")
RIFT2_REF_SHA = "7d92f005f84c25dd23c07670c2aad90e0f3997e5"


def _rift2_reference(src, ref):
    """Run the reference exactly as its demo.py does (its features, its matcher:
    ratio 0.95, no mutual check), with the keypoint budget capped at ours (1500)
    so the comparison is like for like. Its output then goes through OUR
    estimator and quality gate, the same as every other method here."""
    import subprocess
    from types import SimpleNamespace
    sha = subprocess.run(["git", "-C", str(RIFT2_REF_DIR), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if sha != RIFT2_REF_SHA:
        raise RuntimeError(f"reference RIFT2 is at {sha or 'nothing'}, expected {RIFT2_REF_SHA}")
    if str(RIFT2_REF_DIR) not in sys.path:
        sys.path.insert(0, str(RIFT2_REF_DIR))
    from src.RIFT2 import RIFT2 as RefRIFT2            # noqa: E402  (the reference's own package)
    from src.matcher_functions import match_keypoints_nn
    u8 = lambda a: (np.clip(np.asarray(a, np.float64), 0, 1) * 255).astype(np.uint8)
    kp1, d1, kp2, d2 = RefRIFT2(npt=1500)(u8(src.array), u8(ref.array))
    p1, p2, _ = match_keypoints_nn(d1, d2, kp1, kp2, lowes_ratio=0.95, mutual=False)
    return SimpleNamespace(src_pts=np.asarray(p1, np.float64).reshape(-1, 2),
                           ref_pts=np.asarray(p2, np.float64).reshape(-1, 2), device="cpu")


def run_method(name, src, ref):
    if name == "rift2-ref":
        return _rift2_reference(src, ref)
    if name == "sift":
        return classical.match(src, ref, detector="sift")
    if name == "rift2":
        return rift.match(src, ref)
    if name == "rift2-mim":
        return rift.match(src, ref, params=rift.RiftParams(orientation="mim"))
    return adapter.match(src, ref, model_name=name, device="cuda", ship_mode=False)  # benchmarking candidates, not shipping


def one(name, regime, seed):
    kw = dict(REGIMES[regime])
    n_craters = kw.pop("n_craters", 60)
    src, ref, H = synth.make_pair(out_shape=SHAPE, n_craters=n_craters, seed=seed, **kw)
    t0 = time.perf_counter()
    try:
        ms = run_method(name, src, ref)
    except Exception as exc:                                  # recorded, not hidden
        return {"method": name, "regime": regime, "seed": seed, "error": f"{type(exc).__name__}: {exc}"[:160]}
    secs = time.perf_counter() - t0
    res = robust.estimate(ms.src_pts, ms.ref_pts, centre=(SHAPE[1] / 2, SHAPE[0] / 2))
    n = len(ms.src_pts)
    rmse = None
    if res.model is not None:
        g = np.linspace(0, SHAPE[0] - 1, 16)
        gx, gy = np.meshgrid(g, g)
        p = np.c_[gx.ravel(), gy.ravel()]
        rmse = float(np.sqrt(np.mean(np.sum((models.apply(res.model, p) - synth.transform_points(H, p)) ** 2, axis=1))))
    cov = uniformity.coverage_of(ms.src_pts[res.inlier_mask], SHAPE, grid=8) if res.inlier_count else 0.0
    tier = quality.assess(inlier_count=res.inlier_count, inlier_ratio=res.inlier_count / n if n else 0.0,
                          spatial_coverage=cov, model=res.model,
                          scale_ok=res.scale_status not in ("inconsistent", "degenerate"),
                          scale_status=res.scale_status).tier
    return {"method": name, "regime": regime, "seed": seed, "matches": n, "inliers": res.inlier_count,
            "rmse_px": None if rmse is None else round(rmse, 4), "accepted": bool(res.ok), "tier": tier,
            "success": bool(res.ok and rmse is not None and rmse < BAD_PX),
            "false_confidence": bool(tier != "REJECTED" and (rmse is None or rmse > BAD_PX) and res.ok),
            "seconds": round(secs, 2), "device": getattr(ms, "device", None),
            # stored so methods can be cross-checked against each other afterwards
            "matrix": None if res.model is None else np.asarray(res.model.matrix, float).round(8).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="3,11,29")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--out", default="reports/rift_benchmark.json")
    ap.add_argument("--benchmark-models", action="store_true",
                    help="allow unaudited / benchmark-only matchers (G-03); recorded as ship_mode false")
    args = ap.parse_args()
    if args.benchmark_models:
        from chandralign.matching import licence as _licence
        _licence.enable_benchmark_mode()
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [m for m in args.methods.split(",") if m]
    regimes = [r for r in args.regimes.split(",") if r]

    rows = []
    for regime in regimes:
        for name in methods:
            for seed in seeds:
                r = one(name, regime, seed)
                rows.append(r)
                print(json.dumps(r), flush=True)

    # success counts and median error, method x regime
    cell = defaultdict(list)
    for r in rows:
        cell[(r["method"], r["regime"])].append(r)
    print(f"\n{'method':<22}" + "".join(f"{g[:14]:>16}" for g in regimes))
    for name in methods:
        line = f"{name:<22}"
        for regime in regimes:
            rs = cell[(name, regime)]
            ok = [x for x in rs if x.get("success")]
            med = np.median([x["rmse_px"] for x in ok]) if ok else None
            line += f"{len(ok)}/{len(rs)}" .rjust(7) + (f" {med:6.3f}px" if med is not None else "         ")
        print(line)
    fc = [r for r in rows if r.get("false_confidence")]
    print(f"\nfalse confidence (gate accepted a > {BAD_PX} px result): {len(fc)}")
    for r in fc:
        print(f"   {r['method']}/{r['regime']}/s{r['seed']}: {r['rmse_px']} px as {r['tier']}")
    errs = [r for r in rows if "error" in r]
    for r in errs:
        print(f"   ERROR {r['method']}/{r['regime']}/s{r['seed']}: {r['error']}")

    out = ROOT / args.out
    from chandralign.evaluate.run_record import run_record
    out.write_text(json.dumps({"source": "synthetic", "bad_px": BAD_PX, "regimes": REGIMES, "run": run_record(),
                               "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
