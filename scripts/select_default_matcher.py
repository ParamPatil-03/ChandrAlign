"""Which matcher should be the default? (MATCH-03/04/05)

    .venv/Scripts/python scripts/select_default_matcher.py
    .venv/Scripts/python scripts/select_default_matcher.py --decide-only

THE RULE IS NOT IN THIS FILE BY ACCIDENT.
docs/default_matcher_protocol.md fixes the candidate pool, the eligibility
thresholds, the seeds and the selection priority, and it was committed BEFORE
this script existed -- check `git log`. This script only APPLIES that rule. If
you find yourself wanting to adjust a number here to change the outcome, the
experiment is void; re-run it on new seeds with a new protocol.

WHY THIS EXPERIMENT EXISTS
configs/regimes.yaml ships `default_matcher: aliked-lightglue`, accepted under
PLAN.md P2-T03's bar: "beats SIFT on the synthetic illumination-shifted pair".
SIFT fails outright -- zero successful seeds -- at every sun-azimuth difference
of 30 degrees or more, so that bar is passed automatically and separates
nothing. The default has never been compared against an alternative that could
also pass it.

That does NOT mean the incumbent is wrong. It means nothing establishes that it
is right. Note the rule below can return the incumbent.

THE BIAS THIS IS BUILT AGAINST
An earlier 3-seed table (reports/rift_benchmark.json) already suggested XoFTR
beats the incumbent, and it was seen before the protocol was written. So every
threshold comes from OUTSIDE that table -- the 2.0 px bar already committed in
scripts/bench_*.py, failure modes #12/#14/#19, the 3-minute demo constraint --
and the decision runs on seeds 101-110, which nothing has yet looked at.

ALL SYNTHETIC. The leading candidate is then checked on real pairs separately
(protocol section 6); that check is about behaviour, not accuracy, because
those pairs have no exact truth.
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
from chandralign.matching import adapter, classical, licence  # noqa: E402
from chandralign.refine import uniformity  # noqa: E402

SHAPE = (512, 512)
# Protocol 4.2: the accuracy bar already used across this repository before this
# experiment. Not chosen here.
BAD_PX = 2.0

# Protocol 5: fixed before any run.
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

# Protocol 2, minus the two eliminated at the licence gate
# (reports/licence_audit.json): eloftr and matchanything-eloftr.
CANDIDATES = ["aliked-lightglue", "disk-lightglue", "xfeat", "sift-lightglue",
              "sift-nn", "xoftr", "minima-loftr"]

# Protocol 3. CORE regimes decide eligibility; STRETCH regimes only break ties.
# Classed by what the product must do, NOT by what anything scored.
CORE = {
    "same_lighting":        dict(rot_deg=8.0),
    "lighting_+30":         dict(rot_deg=8.0, sun_src=(165, 40)),
    "cross_modal":          dict(rot_deg=8.0, cross_modal=True),
    "resolution_2x":        dict(rot_deg=8.0, scale=2.0),
    "low_texture":          dict(rot_deg=8.0, n_craters=6),
}
STRETCH = {
    "lighting_+90":         dict(rot_deg=8.0, sun_src=(225, 45)),
    "lighting_opposite":    dict(rot_deg=8.0, sun_src=(315, 45)),
    "cross_modal+opposite": dict(rot_deg=8.0, cross_modal=True, sun_src=(315, 45)),
}
REGIMES = {**CORE, **STRETCH}

# Protocol 4: eligibility gates. Each has a source outside this experiment.
MIN_CORE_SUCCESS = 0.90      # a default must work on the pairings we ship
MAX_FALSE_CONF = 0.02        # failure modes #12/#19: a confident lie is the worst outcome
MAX_SECONDS = 5.0            # research doc section 38, the 3-minute live demo


def run_method(name, src, ref):
    if name == "sift-nn":
        return classical.match(src, ref, detector="sift")
    return adapter.match(src, ref, model_name=name, device="cuda")


def one(name, regime, seed):
    kw = dict(REGIMES[regime])
    n_craters = kw.pop("n_craters", 60)
    src, ref, H = synth.make_pair(out_shape=SHAPE, n_craters=n_craters, seed=seed, **kw)
    t0 = time.perf_counter()
    try:
        ms = run_method(name, src, ref)
    except Exception as exc:                          # recorded, never hidden
        return {"method": name, "regime": regime, "seed": seed,
                "error": f"{type(exc).__name__}: {exc}"[:160], "success": False,
                "false_confidence": False, "seconds": None}
    secs = time.perf_counter() - t0

    res = robust.estimate(ms.src_pts, ms.ref_pts, centre=(SHAPE[1] / 2, SHAPE[0] / 2))
    n = len(ms.src_pts)
    rmse = None
    if res.model is not None:
        g = np.linspace(0, SHAPE[0] - 1, 16)
        gx, gy = np.meshgrid(g, g)
        p = np.c_[gx.ravel(), gy.ravel()]
        rmse = float(np.sqrt(np.mean(np.sum(
            (models.apply(res.model, p) - synth.transform_points(H, p)) ** 2, axis=1))))
    cov = uniformity.coverage_of(ms.src_pts[res.inlier_mask], SHAPE, grid=8) if res.inlier_count else 0.0
    tier = quality.assess(inlier_count=res.inlier_count,
                          inlier_ratio=res.inlier_count / n if n else 0.0,
                          spatial_coverage=cov, model=res.model,
                          scale_ok=res.scale_status not in ("inconsistent", "degenerate"),
                          scale_status=res.scale_status).tier
    return {"method": name, "regime": regime, "seed": seed, "matches": n,
            "inliers": res.inlier_count, "rmse_px": None if rmse is None else round(rmse, 4),
            "accepted": bool(res.ok), "tier": tier,
            "success": bool(res.ok and rmse is not None and rmse < BAD_PX),
            "false_confidence": bool(res.ok and tier != "REJECTED" and (rmse is None or rmse > BAD_PX)),
            "seconds": round(secs, 3)}


def score(rows, name):
    """Apply the protocol to one candidate. Returns its scorecard."""
    mine = [r for r in rows if r["method"] == name]
    core = [r for r in mine if r["regime"] in CORE]
    core_ok = [r for r in core if r["success"]]
    fc = [r for r in mine if r["false_confidence"]]
    secs = [r["seconds"] for r in mine if r.get("seconds") is not None]

    # STRETCH coverage: regimes solved on a majority of seeds. Counted from the
    # rows actually present, not from the module default, so a short run
    # (--seeds) reports honestly instead of silently scoring every regime zero.
    stretch_solved = 0
    for g in STRETCH:
        runs = [r for r in mine if r["regime"] == g]
        if runs and sum(r["success"] for r in runs) > len(runs) / 2:
            stretch_solved += 1

    core_rate = len(core_ok) / max(len(core), 1)
    fc_rate = len(fc) / max(len(mine), 1)
    med_s = float(np.median(secs)) if secs else float("inf")
    restricted = licence.restriction_reason(name)

    fails = []
    if restricted:
        fails.append(f"licence: {restricted}")
    if core_rate < MIN_CORE_SUCCESS:
        fails.append(f"core success {core_rate:.0%} < {MIN_CORE_SUCCESS:.0%}")
    if fc_rate > MAX_FALSE_CONF:
        fails.append(f"false confidence {fc_rate:.1%} > {MAX_FALSE_CONF:.0%} ({len(fc)} runs)")
    if med_s > MAX_SECONDS:
        fails.append(f"median {med_s:.2f}s > {MAX_SECONDS}s")

    return {"method": name, "eligible": not fails, "failed_rules": fails,
            "core_success_rate": round(core_rate, 4),
            "core_median_rmse_px": round(float(np.median([r["rmse_px"] for r in core_ok])), 4) if core_ok else None,
            "false_confidences": len(fc), "false_confidence_rate": round(fc_rate, 4),
            "stretch_regimes_solved": stretch_solved, "of_stretch": len(STRETCH),
            "median_seconds": round(med_s, 3) if secs else None,
            "errors": sum(1 for r in mine if "error" in r)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", default="reports/default_matcher_selection.json")
    ap.add_argument("--decide-only", action="store_true",
                    help="re-apply the rule to an existing report without re-running")
    args = ap.parse_args()
    out = ROOT / args.out
    names = [c for c in args.candidates.split(",") if c]
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.decide_only:
        rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    else:
        rows = []
        for name in names:
            for regime in REGIMES:
                for seed in seeds:
                    r = one(name, regime, seed)
                    rows.append(r)
                    print(json.dumps(r), flush=True)

    cards = [score(rows, n) for n in names]

    print("\n=== per-candidate scorecard (protocol section 4) ===")
    print(f"{'candidate':<22}{'core ok':>9}{'core px':>9}{'false':>7}{'stretch':>9}{'sec':>8}   eligibility")
    for c in sorted(cards, key=lambda c: (not c["eligible"], c["method"])):
        px = "--" if c["core_median_rmse_px"] is None else f"{c['core_median_rmse_px']:.3f}"
        sec = "--" if c["median_seconds"] is None else f"{c['median_seconds']:.2f}"
        verdict = "ELIGIBLE" if c["eligible"] else "; ".join(c["failed_rules"])
        print(f"{c['method']:<22}{c['core_success_rate']:>8.0%}{px:>9}"
              f"{c['false_confidences']:>7}{c['stretch_regimes_solved']:>6}/{c['of_stretch']}"
              f"{sec:>8}   {verdict}")

    eligible = [c for c in cards if c["eligible"]]
    print(f"\n{len(eligible)} of {len(cards)} candidates eligible")

    # Protocol 5, applied mechanically and in the stated order.
    ranked = sorted(eligible, key=lambda c: (
        c["false_confidence_rate"],
        c["core_median_rmse_px"] if c["core_median_rmse_px"] is not None else float("inf"),
        -c["stretch_regimes_solved"],
        c["median_seconds"] if c["median_seconds"] is not None else float("inf")))

    incumbent = "aliked-lightglue"
    recommendation = ranked[0]["method"] if ranked else None
    if ranked:
        print("\n=== ranked by the protocol's fixed priority ===")
        print("   (1 false-confidence rate, 2 core error, 3 stretch coverage, 4 runtime)")
        for i, c in enumerate(ranked, 1):
            print(f"   {i}. {c['method']:<22} fc={c['false_confidence_rate']:.1%}  "
                  f"core={c['core_median_rmse_px']}px  stretch={c['stretch_regimes_solved']}/{c['of_stretch']}  "
                  f"{c['median_seconds']}s")
        inc = next((c for c in cards if c["method"] == incumbent), None)
        print(f"\nRECOMMENDATION: {recommendation}")
        if recommendation == incumbent:
            print("   the incumbent default is confirmed by the rule; no change")
        else:
            why = "is not eligible" if inc and not inc["eligible"] else "ranks lower"
            print(f"   the incumbent ({incumbent}) {why}")
            print("   NOT APPLIED. Protocol section 6 requires the real-data check first,")
            print("   and section 8 leaves the decision to the team.")
    else:
        print("\nRECOMMENDATION: none -- no candidate met the eligibility rules.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "synthetic",
        "protocol": "docs/default_matcher_protocol.md",
        "protocol_frozen_before_this_ran": True,
        "incumbent": incumbent, "recommendation": recommendation, "applied": False,
        "seeds": seeds, "bad_px": BAD_PX,
        "eligibility": {"min_core_success": MIN_CORE_SUCCESS,
                        "max_false_confidence_rate": MAX_FALSE_CONF,
                        "max_median_seconds": MAX_SECONDS},
        "core_regimes": list(CORE), "stretch_regimes": list(STRETCH),
        "licence_eliminated": ["eloftr", "matchanything-eloftr"],
        "scorecards": cards, "ranked": [c["method"] for c in ranked], "rows": rows},
        indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
