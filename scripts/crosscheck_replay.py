"""Audit C-04, measurement 1: replay rift_crosscheck.py's cases END TO END through register_bundle.

    .venv/Scripts/python scripts/crosscheck_replay.py --out reports/crosscheck_replay.json

Protocol: docs/crosscheck_protocol.md (frozen before this ran). Every (method, regime, seed) that
reports/rift_crosscheck.json counts as ACCEPTED (from reports/rift_benchmark.json: estimator ok, tier
not REJECTED, method not RIFT2) is re-run on the same synth.make_pair inputs through
pipeline.register_bundle(matcher=method) -- matching, fine stage, the five control gates, the
cross-check and the tier. Truth (the exact synthetic transform) GRADES only.

    caught_by_crosscheck   wrong (> 2 px) and the cross-check flagged it
    caught_by_other        wrong, REJECTED, but not by the cross-check
    missed                 wrong and not REJECTED (a confident wrong answer)
    false_alarm            correct (< 2 px) and the cross-check flagged it
Benchmark-only matchers are allowed here (ship_mode off, as scripts/bench_*.py do).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from chandralign import config, synth  # noqa: E402
from chandralign.estimate import models  # noqa: E402
from chandralign.pipeline import register_bundle  # noqa: E402

BAD_PX = 2.0


def truth_error(model, H, shape):
    if model is None or model.matrix is None:
        return None
    g = np.linspace(0, shape[0] - 1, 16)
    gx, gy = np.meshgrid(g, g)
    p = np.c_[gx.ravel(), gy.ravel()]
    return float(np.sqrt(np.mean(np.sum((models.apply(model, p) - synth.transform_points(H, p)) ** 2, axis=1))))


def main() -> int:
    from bench_rift import REGIMES, SHAPE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-wrong", action="store_true", help="just the wrong-but-accepted cases")
    args = ap.parse_args()
    config.load("default")["ship_mode"] = False          # benchmark-only matchers, as bench_rift.py

    bench = json.loads((ROOT / "reports/rift_benchmark.json").read_text(encoding="utf-8"))["rows"]
    rows = [r for r in bench if "error" not in r and not r["method"].startswith("rift2")
            and r["accepted"] and r["tier"] != "REJECTED"]
    wrong_in_bench = {(r["method"], r["regime"], r["seed"]) for r in rows
                      if r["rmse_px"] is None or r["rmse_px"] > BAD_PX}
    if args.only_wrong:
        rows = [r for r in rows if (r["method"], r["regime"], r["seed"]) in wrong_in_bench]
    print(f"{len(rows)} accepted cases, {len(wrong_in_bench)} of them wrong in the benchmark", flush=True)

    out, recs = Counter(), []
    for i, r in enumerate(rows):
        kw = dict(REGIMES[r["regime"]])
        n_craters = kw.pop("n_craters", 60)
        src, ref, H = synth.make_pair(out_shape=SHAPE, n_craters=n_craters, seed=r["seed"], **kw)
        t0 = time.perf_counter()
        try:
            b = register_bundle(src, ref, matcher=r["method"], device="cuda")
        except Exception as exc:                          # recorded, never hidden
            recs.append({**{k: r[k] for k in ("method", "regime", "seed")}, "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(i, r["method"], r["regime"], r["seed"], "ERROR", exc, flush=True)
            continue
        res = b.result
        err = truth_error(res.model, H, SHAPE)
        wrong = err is None or err > BAD_PX
        xc = res.provenance.get("crosscheck", {})
        flagged = res.gates.get("independent_crosscheck") is False
        tier = res.confidence_tier
        outcome = ("caught_by_crosscheck" if wrong and flagged else
                   "caught_by_other" if wrong and tier == "REJECTED" else
                   "missed" if wrong else
                   "false_alarm" if flagged else
                   "confirmed" if xc.get("verdict") == "agree" else
                   "unchecked_correct")
        out[outcome] += 1
        key = (r["method"], r["regime"], r["seed"])
        rec = {"method": r["method"], "regime": r["regime"], "seed": r["seed"], "bench_wrong": key in wrong_in_bench,
               "bench_error_px": r["rmse_px"], "error_px": None if err is None else round(err, 3), "tier": tier,
               "failed_gates": [g for g, ok in res.gates.items() if not ok], "crosscheck": xc,
               "outcome": outcome, "seconds": round(time.perf_counter() - t0, 1)}
        recs.append(rec)
        print(i, rec["method"], rec["regime"], rec["seed"], "err", rec["error_px"], tier, xc.get("verdict"),
              xc.get("gap_px"), outcome, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"source": "synthetic", "protocol": "docs/crosscheck_protocol.md",
                                        "totals": dict(out), "runs": recs}, indent=1), encoding="utf-8")

    bw = [x for x in recs if x.get("bench_wrong")]
    by_x = sum(x.get("outcome") == "caught_by_crosscheck" for x in bw)
    rejected = sum(x.get("tier") == "REJECTED" for x in bw)
    print(f"\ntotals: {dict(out)}")
    print(f"the benchmark's {len(bw)} wrong-but-accepted cases: flagged by the cross-check {by_x}; "
          f"REJECTED by anything {rejected}; still wrong in this run: {sum(bool(x.get('error_px') is None or x['error_px'] > BAD_PX) for x in bw)}")
    print(f"false alarms on correct results: {out['false_alarm']}/{out['false_alarm'] + out['confirmed']} checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
