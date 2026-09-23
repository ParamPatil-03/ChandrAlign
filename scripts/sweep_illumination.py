"""Where does sun azimuth difference actually stop being solvable? (MATCH-09)

    .venv/Scripts/python scripts/sweep_illumination.py
    .venv/Scripts/python scripts/sweep_illumination.py --decide-only

THE RULE IS NOT IN THIS FILE BY ACCIDENT.
docs/unsolved_band_protocol.md fixes the matcher pool, the azimuth samples, the
seeds, the success definition, the membership rule and the two verdict
boundaries, and it was committed BEFORE this script existed -- check `git log`.
This script only APPLIES that rule. If you find yourself wanting to adjust a
number here to change the outcome, the experiment is void; re-run it on fresh
seeds with a new protocol.

WHY THIS EXPERIMENT EXISTS
configs/regimes.yaml declared every pair with >= 60 deg of sun azimuth
difference `unsolved`, on the evidence of a four-matcher benchmark (sift,
xfeat, aliked-lightglue, eloftr) in which all four fail above 60 deg. A pool
where nothing can pass cannot locate where passing stops. The same defect had
already been found twice in this project: "beat SIFT on the illumination-
shifted pair" (SIFT scores zero there) and "MAGSAC beats RANSAC at 70%
outliers" (every estimator succeeds there).

WHY A SINGLE THRESHOLD IS THE WRONG SHAPE
regimes.yaml's own representation_bands comment, forty lines above the band
this replaces, states the measured finding: difficulty is NOT monotonic in
azimuth. At 90 deg the lit and shadowed facets swap and raw correlation
collapses to +0.010; at 180 deg the scene approaches a contrast inversion,
which MIND-style descriptors are invariant to (+0.975). `min_d_azimuth_deg`
cannot express "harder in the middle than at the end".

ALL SYNTHETIC. Both views are re-lightings of one height field, so they differ
in illumination and nothing else. Real pairs differ in far more; a band solved
here is not thereby solved on real data (protocol section 9).
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
from chandralign.estimate import models, robust  # noqa: E402
from chandralign.evaluate import quality  # noqa: E402
from chandralign.matching import adapter, classical, licence  # noqa: E402
from chandralign.refine import uniformity  # noqa: E402

SHAPE = (512, 512)

# Protocol 5: the accuracy bar already committed in scripts/bench_matchers.py
# before this experiment. Not chosen here.
BAD_PX = 2.0

# Protocol 6.4: fixed before any run. 0-9, 3/11/29, 101-110 and 301-315 are spent.
SEEDS = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210]

# Protocol 6.2. Denser than the old 0/30/60/90/120/150/180 bands through the
# transition, because a 30 deg grid cannot locate an edge finer than 30 deg.
AZIMUTHS = [0.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]

# Protocol 6.1. sift-nn is the floor and is EXPECTED to fail above 30 deg; if it
# does not, the harness is wrong, not SIFT.
MATCHERS = ["xoftr", "minima-loftr", "aliked-lightglue", "eloftr", "sift-nn"]

# Protocol 6.2/6.3: elevation held fixed so azimuth difference is the only
# variable. sun_ref is make_pair's own default.
SUN_REF = (135.0, 45.0)
ELEVATION_DEG = 45.0
ROT_DEG = 8.0
N_CRATERS = 60

# Protocol 7, both from docs/default_matcher_protocol.md, which predates this:
#   9/10 -- section 4.2's 90% bar ("one core failure in ten ... is the most that
#           can be called working")
#   6/10 -- section 5 criterion 3, "solved on a majority of seeds", the bar that
#           document already applies to lighting_+90 and lighting_opposite
SOLVED_MIN_SEEDS = 9
DEGRADED_MIN_SEEDS = 6

# Worst last. Used by the worse-of-two rule for azimuths between two samples.
SEVERITY = {"solved": 0, "degraded": 1, "unsolved": 2}


def worse(a: str, b: str) -> str:
    return a if SEVERITY[a] >= SEVERITY[b] else b


def members(names: list[str]) -> list[str]:
    """Protocol 4: a matcher may make a band `solved` only if it is licence-clean.

    This is licence.restriction_reason, the same gate select_default_matcher.py
    applies to this pool -- NOT licence.shippable(), which is a narrower
    allowlist that xoftr and minima-loftr have not yet been promoted onto. The
    report names which matcher solved each band precisely so that a band resting
    on a not-yet-promoted matcher stays visible.
    """
    return [n for n in names if licence.restriction_reason(n) is None]


def run_method(name, src, ref):
    if name == "sift-nn":
        return classical.match(src, ref, detector="sift")
    return adapter.match(src, ref, model_name=name, device="cuda")


def release_models() -> None:
    """Drop cached weights before moving to the next matcher.

    NOT a micro-optimisation. `adapter._load` is lru_cache(maxsize=8), so every
    matcher this sweep touches stays resident on the GPU. On a 6 GB card the
    third one lands with VRAM already 93% full, and the FIRST run of this sweep
    measured aliked-lightglue at 53.7 s per pair against 0.20 s when it runs
    alone -- a 270x slowdown that is contention, not the matcher. Left in place
    it would both take hours and write a `seconds` column describing this
    script's memory management rather than the matchers.

    Accuracy verdicts were unaffected either way; this keeps the timings
    honest and the run finite.
    """
    adapter._load.cache_clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:                                # CPU-only environment
        pass


def one(name: str, d_az: float, seed: int) -> dict:
    """One run: one matcher, one azimuth difference, one seed."""
    src, ref, H = synth.make_pair(
        out_shape=SHAPE, n_craters=N_CRATERS, seed=seed, rot_deg=ROT_DEG,
        sun_ref=SUN_REF, sun_src=(SUN_REF[0] + d_az, ELEVATION_DEG))

    t0 = time.perf_counter()
    try:
        ms = run_method(name, src, ref)
    except Exception as exc:                          # recorded, never hidden
        return {"method": name, "d_azimuth_deg": d_az, "seed": seed,
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
    return {"method": name, "d_azimuth_deg": d_az, "seed": seed, "matches": n,
            "inliers": res.inlier_count, "rmse_px": None if rmse is None else round(rmse, 4),
            "accepted": bool(res.ok), "tier": tier,
            "success": bool(res.ok and rmse is not None and rmse < BAD_PX),
            "false_confidence": bool(res.ok and tier != "REJECTED" and (rmse is None or rmse > BAD_PX)),
            "seconds": round(secs, 3)}


def per_sample(rows: list[dict], names: list[str], azimuths: list[float]) -> list[dict]:
    """Protocol 7: one verdict per azimuth, with the attribution that earns it."""
    eligible = set(members(names))
    out = []
    for az in azimuths:
        here = [r for r in rows if r["d_azimuth_deg"] == az]
        scores, n_seeds = {}, {}
        for name in names:
            mine = [r for r in here if r["method"] == name]
            scores[name] = sum(1 for r in mine if r["success"])
            n_seeds[name] = len(mine)

        # Only licence-clean matchers can earn a verdict. Scored out of the runs
        # actually present, so a short run (--seeds) reports honestly.
        scaled = {n: (scores[n] / n_seeds[n] * len(SEEDS)) if n_seeds[n] else 0.0
                  for n in names if n in eligible}
        best = max(scaled.values()) if scaled else 0.0

        if best >= SOLVED_MIN_SEEDS:
            verdict = "solved"
        elif best >= DEGRADED_MIN_SEEDS:
            verdict = "degraded"
        else:
            verdict = "unsolved"

        # Every member that independently reaches the verdict's own bar.
        bar = {"solved": SOLVED_MIN_SEEDS, "degraded": DEGRADED_MIN_SEEDS,
               "unsolved": None}[verdict]
        earned = sorted([n for n in scaled if bar is not None and scaled[n] >= bar],
                        key=lambda n: (-scores[n], n))

        out.append({
            "d_azimuth_deg": az,
            "expectation": verdict,
            "solved_by": earned,
            "best_seeds": None if not scaled else int(round(best)),
            "scores": {n: f"{scores[n]}/{n_seeds[n]}" for n in names},
            "median_rmse_px": {
                n: (round(float(np.median([r["rmse_px"] for r in here
                                           if r["method"] == n and r["rmse_px"] is not None])), 3)
                    if any(r["method"] == n and r["rmse_px"] is not None for r in here) else None)
                for n in names},
            "non_member_scores": {n: f"{scores[n]}/{n_seeds[n]}"
                                  for n in names if n not in eligible},
        })
    return out


def to_bands(samples: list[dict]) -> list[dict]:
    """Protocol 7.1: merge adjacent samples that agree, keep the range explicit.

    Ranges are inclusive and cover only the SAMPLED points. The gaps between
    bands are deliberately left unclaimed here; regime.py resolves a query that
    lands in one by taking the WORSE of the two neighbours, which is the rule
    the protocol fixed before any of this was measured. Encoding the gap as if
    it had been measured would be the same over-claim this experiment exists to
    remove.
    """
    bands: list[dict] = []
    for s in samples:
        if bands and bands[-1]["expectation"] == s["expectation"]:
            prev = bands[-1]
            prev["d_azimuth_deg"][1] = s["d_azimuth_deg"]
            # Only a matcher that earned the verdict at EVERY sample in the band
            # keeps the credit; one good azimuth does not carry a whole band.
            prev["solved_by"] = sorted(set(prev["solved_by"]) & set(s["solved_by"])) \
                if prev["solved_by"] and s["solved_by"] else \
                sorted(set(prev["solved_by"]) | set(s["solved_by"]))
            # The band is only as good as its worst sample.
            if s["best_seeds"] is not None:
                prev["best_seeds"] = min(prev["best_seeds"], s["best_seeds"]) \
                    if prev["best_seeds"] is not None else s["best_seeds"]
            continue
        bands.append({"d_azimuth_deg": [s["d_azimuth_deg"], s["d_azimuth_deg"]],
                      "expectation": s["expectation"],
                      "solved_by": list(s["solved_by"]),
                      "best_seeds": s["best_seeds"]})
    return bands


def report(samples: list[dict], bands: list[dict], names: list[str]) -> None:
    eligible = set(members(names))
    print("\n=== per-azimuth verdict (protocol section 7) ===")
    head = "  d_az  verdict     " + "".join(f"{n[:16]:>17}" for n in names)
    print(head)
    for s in samples:
        row = f"  {s['d_azimuth_deg']:>4.0f}  {s['expectation']:<10}"
        for n in names:
            mark = "" if n in eligible else "*"
            row += f"{s['scores'][n] + mark:>17}"
        print(row)
    print("  * not licence-clean: scored but cannot earn a verdict")

    print("\n=== bands (protocol section 7.1) ===")
    for b in bands:
        lo, hi = b["d_azimuth_deg"]
        span = f"{lo:.0f}" if lo == hi else f"{lo:.0f}-{hi:.0f}"
        who = ", ".join(b["solved_by"]) or "(none)"
        print(f"  {span:>9} deg  {b['expectation']:<10} {who}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchers", default=",".join(MATCHERS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--azimuths", default=",".join(str(a) for a in AZIMUTHS))
    ap.add_argument("--out", default="reports/illumination_sweep.json")
    ap.add_argument("--decide-only", action="store_true",
                    help="re-apply the rule to an existing report without re-running")
    args = ap.parse_args()

    out = ROOT / args.out
    names = [c for c in args.matchers.split(",") if c]
    seeds = [int(s) for s in args.seeds.split(",")]
    azimuths = [float(a) for a in args.azimuths.split(",")]

    if args.decide_only:
        rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    else:
        rows = []
        for name in names:
            # One matcher resident at a time. See release_models().
            release_models()
            for az in azimuths:
                for seed in seeds:
                    r = one(name, az, seed)
                    rows.append(r)
                    print(json.dumps(r), flush=True)
        release_models()

    samples = per_sample(rows, names, azimuths)
    bands = to_bands(samples)
    report(samples, bands, names)

    payload = {
        "source": "synthetic",
        "what": ("sun-azimuth sweep re-deriving the unsolved_illumination band "
                 "(MATCH-09). Both views are re-lightings of one synthetic height "
                 "field: they differ in illumination and in nothing else."),
        "protocol": "docs/unsolved_band_protocol.md",
        "supersedes": "reports/synthetic_matcher_bench.json",
        "accuracy_bar_px": BAD_PX,
        "accuracy_bar_source": "BAD_PX, scripts/bench_matchers.py",
        "verdict_rule": {
            "solved": f"at least one licence-clean matcher succeeds on >= {SOLVED_MIN_SEEDS}/{len(SEEDS)} seeds",
            "degraded": f"best licence-clean matcher succeeds on >= {DEGRADED_MIN_SEEDS}/{len(SEEDS)} seeds",
            "unsolved": f"no licence-clean matcher reaches {DEGRADED_MIN_SEEDS}/{len(SEEDS)} seeds",
            "solved_source": "docs/default_matcher_protocol.md section 4.2 (90%)",
            "degraded_source": "docs/default_matcher_protocol.md section 5 criterion 3 (majority of seeds)",
            "between_samples": "an azimuth between two samples takes the WORSE verdict",
        },
        "membership": {
            "rule": "licence.restriction_reason(name) is None",
            "members": members(names),
            "excluded": [n for n in names if n not in set(members(names))],
            "note": ("xoftr and minima-loftr are licence-clean but not yet on "
                     "regimes.yaml shippable_matchers; promotion is MATCH-03/04/05's "
                     "decision. A band resting only on a matcher whose promotion is "
                     "refused is void and must be re-derived."),
        },
        "fixed": {"shape": list(SHAPE), "n_craters": N_CRATERS, "rot_deg": ROT_DEG,
                  "sun_ref": list(SUN_REF), "elevation_deg": ELEVATION_DEG,
                  "cross_modal": False, "scale": 1.0},
        "default_matcher": "aliked-lightglue",
        "seeds": seeds, "azimuths_deg": azimuths, "matchers": names,
        "samples": samples,
        "bands": bands,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
