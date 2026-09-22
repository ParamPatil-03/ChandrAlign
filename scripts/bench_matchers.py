"""Compare matchers across illumination regimes, and validate the quality gate.

Feature MATCH-03/04/05 selection evidence (P2-T04) plus CHECK-07/08 validation.

ALL RESULTS ARE SYNTHETIC. The imagery is generated (chandralign.synth), so
every row is tagged source="synthetic" and must never be presented as
performance on real Chandrayaan-2 data. What it CAN honestly show is relative
behaviour across a controlled illumination sweep, because the same terrain is
rendered under different Suns with the ground-truth transform known exactly.

TWO THINGS ARE MEASURED, AND THEY MUST NOT BE CONFUSED:

1. Matcher accuracy, scored against the TRUE transform over a grid -- not by
   reprojection error on each matcher's own inliers. A set of mutually
   consistent WRONG matches has excellent reprojection error, which is exactly
   how four registrations in the first run were accepted while being 3.7 to
   13.1 px wrong.

2. Whether the quality gate catches those. The gate is given only what exists
   at run time on real data (counts, ratios, coverage, geometry). Ground-truth
   error is recorded to GRADE the gate and is never an input to it -- otherwise
   this would test nothing.

    .venv/Scripts/python scripts/bench_matchers.py
    .venv/Scripts/python scripts/bench_matchers.py --models eloftr --seeds 1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chandralign import compute, synth  # noqa: E402
from chandralign.estimate import models, robust  # noqa: E402
from chandralign.evaluate import control_gates, quality  # noqa: E402
from chandralign.matching import adapter, classical  # noqa: E402
from chandralign.refine import uniformity  # noqa: E402

SHAPE = (384, 384)
CENTRE = (SHAPE[1] / 2.0, SHAPE[0] / 2.0)
BAD_PX = 2.0          # above this, a transform is wrong enough to matter

CASES = {
    "same_sun": dict(sun_ref=(135, 45), sun_src=(135, 45)),
    "sun_+30deg": dict(sun_ref=(135, 45), sun_src=(165, 40)),
    "sun_+90deg": dict(sun_ref=(135, 45), sun_src=(225, 45)),
    "opposite_sun": dict(sun_ref=(135, 45), sun_src=(315, 45)),
    "low_sun_10deg": dict(sun_ref=(135, 10), sun_src=(165, 10)),
}

DEFAULT_MODELS = ["sift", "xfeat", "aliked-lightglue", "eloftr"]


def transform_rmse(model, h_true, shape=SHAPE, n=24) -> float:
    ys, xs = np.mgrid[0:shape[0]:complex(n), 0:shape[1]:complex(n)]
    grid = np.stack([xs.ravel(), ys.ravel()], 1)
    pred = models.apply(model, grid)
    true = synth.transform_points(h_true, grid)
    return float(np.sqrt(((pred - true) ** 2).sum(1).mean()))


def run_one(model_name: str, case: str, seed: int = 3, gates_on: bool = True) -> dict:
    src, ref, h_true = synth.make_pair(out_shape=SHAPE, rot_deg=8.0,
                                       shift=(0.37, -0.62), n_craters=45,
                                       seed=seed, **CASES[case])
    started = time.perf_counter()
    if model_name == "sift":
        ms = classical.match(src, ref, detector="sift")
    else:
        ms = adapter.match(src, ref, model_name=model_name)
    match_s = time.perf_counter() - started

    res = robust.estimate(ms.src_pts, ms.ref_pts, expected_scale=1.0, centre=CENTRE)
    n_matches = int(len(ms.src_pts))
    ratio = (res.inlier_count / n_matches) if n_matches else 0.0
    coverage = (uniformity.coverage_of(ms.src_pts[res.inlier_mask], SHAPE, grid=8)
                if res.inlier_count else 0.0)
    scale_ok = robust.FM_SCALE_CONFUSION not in res.failure_modes

    # CHECK-01..04, 06: run with the SAME matcher on the SAME pair, on every
    # benchmark execution, and fed to the verdict. Ground truth is still never an
    # input -- the gates see only what a real run would see.
    gates = None
    if gates_on:
        gates = control_gates.run_all(control_gates.pipeline_from(model_name, gsd_m=src.gsd_m),
                                      src.array, ref.array, src, ref, seed=seed)
    verdict = quality.assess(inlier_count=res.inlier_count, inlier_ratio=ratio,
                             spatial_coverage=coverage, model=res.model,
                             scale_ok=scale_ok, scale_status=res.scale_status,
                             gates=gates.gates if gates else None)

    row = {
        "model": model_name, "case": case, "seed": seed,
        "device": getattr(ms, "device", "unknown"),
        "keypoints_src": getattr(ms, "n_keypoints_src", None),
        "matches": n_matches,
        "inliers": res.inlier_count,
        "inlier_ratio": round(ratio, 4),
        "spatial_coverage": round(coverage, 4),
        "scale_estimated": (round(res.model.scale_estimated, 4)
                            if res.model is not None and res.model.scale_estimated else None),
        "estimator_ok": bool(res.ok),
        "tier": verdict.tier,
        "limiting_signal": verdict.limiting_signal,
        "match_seconds": round(match_s, 3),
        "source": "synthetic",
        "gates": gates.gates if gates else None,
        "gates_failed": [g.name for g in gates.failed] if gates else None,
    }
    # Recorded to GRADE the gate. Never an input to it.
    row["transform_rmse_px"] = (round(transform_rmse(res.model, h_true), 4)
                                if res.model is not None else None)
    row["reason"] = None if res.ok else (res.notes[-1] if res.notes else "rejected")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--seeds", default="3")
    ap.add_argument("--out", default="reports/synthetic_matcher_bench.json")
    ap.add_argument("--no-gates", action="store_true", help="skip the control gates (not for reported runs)")
    args = ap.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    case_names = [c.strip() for c in args.cases.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"{'model':<19}{'case':<15}{'seed':>4}{'match':>7}{'inlier':>7}"
          f"{'ratio':>7}{'cover':>7}{'RMSE':>9}{'tier':>10}")
    print("-" * 92)

    rows = []
    for model_name in model_names:
        for seed in seeds:
            for case in case_names:
                try:
                    row = run_one(model_name, case, seed=seed, gates_on=not args.no_gates)
                except Exception as exc:
                    rows.append({"model": model_name, "case": case, "seed": seed,
                                 "tier": "ERROR", "source": "synthetic",
                                 "reason": f"{type(exc).__name__}: {exc}"})
                    print(f"{model_name:<19}{case:<15}{seed:>4}{'':>7}{'':>7}"
                          f"{'':>7}{'':>7}{'ERROR':>9}{str(exc)[:30]:>10}")
                    continue
                rows.append(row)
                rmse = (f"{row['transform_rmse_px']:.3f}"
                        if row["transform_rmse_px"] is not None else "n/a")
                print(f"{row['model']:<19}{row['case']:<15}{row['seed']:>4}"
                      f"{row['matches']:>7}{row['inliers']:>7}{row['inlier_ratio']:>7.2f}"
                      f"{row['spatial_coverage']:>7.2f}{rmse:>9}{row['tier']:>10}")

    accepted_bad = [r for r in rows if r.get("tier") in ("HIGH", "MEDIUM", "LOW")
                    and (r.get("transform_rmse_px") or 0) > BAD_PX]
    rejected_good = [r for r in rows if r.get("tier") == "REJECTED"
                     and r.get("transform_rmse_px") is not None
                     and r["transform_rmse_px"] <= BAD_PX]

    print(f"\nGATE VALIDATION (a transform is 'bad' above {BAD_PX} px)")
    print(f"  accepted but bad  (false confidence): {len(accepted_bad)}")
    for r in accepted_bad:
        print(f"    {r['model']}/{r['case']}/seed{r['seed']}: {r['transform_rmse_px']} px "
              f"as {r['tier']} (ratio {r['inlier_ratio']}, coverage {r['spatial_coverage']})")
    print(f"  rejected though good (over-caution): {len(rejected_good)}")
    for r in rejected_good:
        print(f"    {r['model']}/{r['case']}/seed{r['seed']}: {r['transform_rmse_px']} px "
              f"rejected on {r['limiting_signal']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"source": "synthetic",
         "warning": "synthetic imagery; not a claim about real Chandrayaan-2 data",
         "gate_validation": {"bad_threshold_px": BAD_PX,
                             "accepted_but_bad": len(accepted_bad),
                             "rejected_though_good": len(rejected_good)},
         "capability": compute.capability(),
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
