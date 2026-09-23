"""fp16 synthetic check for xoftr (docs/verify_default_protocol.md, 3.B.3).

    .venv/Scripts/python scripts/xoftr_fp16_synthetic.py --precision fp16
    .venv/Scripts/python scripts/xoftr_fp16_synthetic.py --precision fp32   # control

Reuses scripts/select_default_matcher.py UNCHANGED -- same generator, regimes,
seeds 101-110, estimator, quality gate and grading -- and replaces only the one
function that calls the matcher, so the precision is the single difference. The
fp32 run is the control: it must reproduce the committed xoftr rows, or any
fp16 difference could be environment drift. Run each precision in its own process.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import select_default_matcher as sdm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    args = ap.parse_args()

    def run_method(name, src, ref):
        return sdm.adapter.match(src, ref, model_name=name, device="cuda", precision=args.precision)

    sdm.run_method = run_method
    rows = []
    for regime in sdm.REGIMES:
        for seed in sdm.SEEDS:
            r = sdm.one("xoftr", regime, seed)
            rows.append(r)
            print(json.dumps(r), flush=True)

    core = ["same_lighting", "lighting_+30", "resolution_2x", "low_texture", "cross_modal"]
    per = {}
    for g in sdm.REGIMES:
        R = [r for r in rows if r["regime"] == g]
        per[g] = {"gate_accepted_success": sum(bool(r.get("success")) and r.get("tier") != "REJECTED"
                                               for r in R),
                  "false_confidences": sum(bool(r.get("false_confidence")) for r in R), "of": len(R)}
    passes = (all(per[g]["gate_accepted_success"] >= 9 for g in core)
              and sum(per[g]["false_confidences"] for g in core) == 0)
    out = ROOT / f"reports/xoftr_fp16_synthetic_{args.precision}.json"
    out.write_text(json.dumps({
        "source": "synthetic", "protocol": "docs/verify_default_protocol.md 3.B.3",
        "precision": args.precision, "seeds": sdm.SEEDS, "per_regime": per,
        "criterion_core": core, "passes_3B3": passes, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(per, indent=1), "\npasses 3.B.3:", passes, "\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
