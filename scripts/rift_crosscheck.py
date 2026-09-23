"""Can RIFT2 serve as an INDEPENDENT cross-check on the learned matchers? (MATCH-06 role)

    .venv/Scripts/python scripts/rift_crosscheck.py            # reads reports/rift_benchmark.json

WHY THIS QUESTION
The RIFT2 benchmark found 12 results that the quality gate ACCEPTED while they
were 2.2-5.7 px wrong -- all from learned matchers (ALIKED-LightGlue,
MatchAnything, MINIMA). The control gates caught only 2 of them: those results
are CONSISTENTLY wrong -- move the input 5 px and the wrong answer moves 5 px
too -- and no check that looks only at one method's own output can see a
consistent error. A second method that fails DIFFERENTLY can. RIFT2 is
classical, untrained and phase-based, so it shares no training data and no
architecture with the learned matchers.

THE RULE (usable at run time, no ground truth):
  - RIFT2 is a checker only when its OWN result passed the quality gate;
    otherwise it ABSTAINS -- it never vetoes on the strength of a failed run
  - an accepted result from any other method is FLAGGED when its transform
    differs from RIFT2's by more than 2 px RMS over the image

SCORED AGAINST TRUTH (grading only):
  caught        a wrong accepted result that RIFT2 flagged
  missed        a wrong accepted result RIFT2 did not flag, or abstained on
  false alarm   a CORRECT accepted result that RIFT2 flagged -- the cost
A cross-check that raises false alarms is worse than none, so both are reported.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chandralign.evaluate.control_gates import crosscheck_gate  # noqa: E402
FLAG_PX = 2.0
SHAPE = (512, 512)


def main() -> int:
    data = json.loads((ROOT / "reports/rift_benchmark.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if "error" not in r]
    if not all("matrix" in r for r in rows):
        sys.exit("report has no stored transforms; re-run scripts/bench_rift.py")
    rift = {(r["regime"], r["seed"]): r for r in rows if r["method"] == "rift2"}

    out = Counter()
    per_regime: dict[str, Counter] = {}
    detail = []
    for r in rows:
        if r["method"] in ("rift2", "rift2-mim") or r["tier"] == "REJECTED" or not r["accepted"]:
            continue                                    # only results the gate let through
        wrong = r["rmse_px"] is None or r["rmse_px"] > FLAG_PX
        chk = rift.get((r["regime"], r["seed"]))
        can_check = chk is not None and chk["accepted"] and chk["tier"] != "REJECTED" and chk["matrix"] is not None
        # The SAME gate the pipeline uses, so this measures what ships.
        g = crosscheck_gate(np.asarray(r["matrix"]), np.asarray(chk["matrix"]) if can_check else None,
                            can_check, SHAPE, flag_px=FLAG_PX)
        verdict = "abstain" if g is None else ("agree" if g.passed else "flag")
        outcome = ("caught" if (wrong and verdict == "flag") else
                   "missed" if wrong else
                   "false_alarm" if verdict == "flag" else
                   "confirmed" if verdict == "agree" else "unchecked_correct")
        out[outcome] += 1
        per_regime.setdefault(r["regime"], Counter())[outcome] += 1
        if wrong or outcome == "false_alarm":
            detail.append((r["method"], r["regime"], r["seed"], r["rmse_px"], verdict, outcome))

    print(f"accepted results from non-RIFT methods, cross-checked against RIFT2 (flag > {FLAG_PX} px):\n")
    for k in ("caught", "missed", "false_alarm", "confirmed", "unchecked_correct"):
        print(f"   {k:<18} {out[k]}")
    wrong_total = out["caught"] + out["missed"]
    print(f"\n   wrong-but-accepted results caught: {out['caught']}/{wrong_total}")
    correct_checked = out["confirmed"] + out["false_alarm"]
    print(f"   false alarms on correct results:   {out['false_alarm']}/{correct_checked} checked")
    print("\nper regime (caught / missed / false alarm / confirmed / unchecked):")
    for g, c in per_regime.items():
        print(f"   {g:<22} {c['caught']:>2} / {c['missed']:>2} / {c['false_alarm']:>2} / {c['confirmed']:>2} / {c['unchecked_correct']:>2}")
    print("\nevery wrong-but-accepted result, and every false alarm:")
    for m, g, s, e, v, o in sorted(detail, key=lambda x: (x[5], x[1])):
        print(f"   {o:<12} {m:<22}{g:<22}s{s:<3} {e if e is not None else 'n/a':>7} px  rift2: {v}")
    (ROOT / "reports/rift_crosscheck.json").write_text(json.dumps({
        "source": "synthetic", "flag_px": FLAG_PX, "rule": "RIFT2 checks only when its own result passed the gate",
        "totals": dict(out), "per_regime": {k: dict(v) for k, v in per_regime.items()},
        "detail": [dict(zip(("method", "regime", "seed", "rmse_px", "rift2", "outcome"), d)) for d in detail]},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
