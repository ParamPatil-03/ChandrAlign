"""Independent re-derivation of the default-matcher decision (verify protocol section 2).

    .venv/Scripts/python scripts/verify_default_rescore.py

Reads ONLY committed raw rows; recomputes nothing from the scorecards. Writes
reports/verify_default_rescore.json. The synthetic part is post hoc (the rows
were seen before this was written) and can refute a ranking, not promote one.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_real_check import score  # noqa: E402

BAD_PX = 2.0
MIN_CORE, MAX_FC, MAX_S = 0.90, 0.02, 5.0          # docs/default_matcher_protocol.md 4.2-4.4, unchanged


def ok(r):
    return r.get("rmse_px") is not None and r["rmse_px"] < BAD_PX


def fc(r):          # as stored: estimator accepted, quality gate did not reject, >2 px off
    return bool(r.get("accepted")) and r.get("tier") != "REJECTED" and not ok(r)


def success(r):     # as stored: estimator accepted and <2 px. Ignores the quality tier.
    return bool(r.get("accepted")) and ok(r)


def success_gated(r):
    return success(r) and r.get("tier") != "REJECTED"


def scorecard(rows, m, core, stretch):
    R = [r for r in rows if r["method"] == m and r["regime"] in core | stretch]
    Rc = [r for r in R if r["regime"] in core]
    secs = [r["seconds"] for r in R if r.get("seconds") is not None]
    ok_core = [r["rmse_px"] for r in Rc if success(r)]
    card = {
        "method": m, "runs": len(R),
        "core_success": round(sum(map(success, Rc)) / len(Rc), 4),
        "core_success_gate_accepted": round(sum(map(success_gated, Rc)) / len(Rc), 4),
        "false_confidences": sum(map(fc, R)),
        "false_confidence_rate": round(sum(map(fc, R)) / len(R), 4),
        "false_confidences_by_regime": {g: sum(fc(r) for r in R if r["regime"] == g)
                                        for g in sorted(core | stretch)},
        "core_median_rmse_px": round(st.median(ok_core), 4) if ok_core else None,
        "stretch_solved": sum(1 for g in stretch
                              if sum(success(r) for r in R if r["regime"] == g) > 5),
        "median_seconds": round(st.median(secs), 3),
        "errors": sum(1 for r in R if "error" in r),
    }
    card["eligible"] = (card["core_success"] >= MIN_CORE and card["false_confidence_rate"] <= MAX_FC
                        and card["median_seconds"] <= MAX_S)
    return card


def rank(cards):
    """Protocol section 5, in order: fewest false confidences, lowest core median
    error, most stretch regimes solved, fastest."""
    el = [c for c in cards if c["eligible"]]
    return [c["method"] for c in sorted(el, key=lambda c: (c["false_confidence_rate"],
                                                           c["core_median_rmse_px"],
                                                           -c["stretch_solved"],
                                                           c["median_seconds"]))]


def main():
    d = json.loads((ROOT / "reports/default_matcher_selection.json").read_text())
    rows = d["rows"]
    methods = sorted({r["method"] for r in rows})
    core, stretch = set(d["core_regimes"]), set(d["stretch_regimes"])
    sm_core, sm_stretch = core - {"cross_modal"}, stretch - {"cross_modal+opposite"}

    stored = {c["method"]: c["false_confidences"] for c in d["scorecards"]}
    orig = [scorecard(rows, m, core, stretch) for m in methods]
    same = [scorecard(rows, m, sm_core, sm_stretch) for m in methods]
    inconsistent = [(r["method"], r["regime"], r["seed"]) for r in rows
                    if success(r) and r.get("tier") == "REJECTED"]

    real_files = {"xoftr": "tmc2_tc_xoftr", "eloftr": "tmc2_tc_registration",
                  "aliked-lightglue": "tmc2_tc_aliked", "minima-loftr": "tmc2_tc_minima",
                  "sift-lightglue": "tmc2_tc_sift_lightglue", "disk-lightglue": "tmc2_tc_disk_lightglue"}
    real = {m: score(ROOT / "reports" / f"{f}.json")["summary"] for m, f in real_files.items()}
    post_hoc = {"xoftr", "eloftr", "aliked-lightglue", "minima-loftr"}   # seen before the protocol

    out = {
        "source": "synthetic",
        "note": ("synthetic sections re-derived from reports/default_matcher_selection.json rows; "
                 "real_check sections from measured reports/tmc2_tc_*.json. Protocol: "
                 "docs/verify_default_protocol.md"),
        "recount_matches_stored_false_confidences": all(
            c["false_confidences"] == stored[c["method"]] for c in orig),
        "definition_inconsistency": {
            "what": ("the selector's `success` requires only estimator acceptance and <2 px; "
                     "`false_confidence` additionally requires the quality tier not REJECTED. "
                     "A run the quality gate rejected can therefore count as a success."),
            "runs_affected": len(inconsistent),
            "examples": inconsistent[:10],
        },
        "original_protocol": {"core": sorted(core), "stretch": sorted(stretch),
                              "scorecards": orig, "ranked_eligible": rank(orig)},
        "same_modality_only": {
            "why": ("regime.select returns matcher=None for cross-modal pairs and checks scale "
                    "first, so default_matcher is only reached on the same-modality branch"),
            "core": sorted(sm_core), "stretch": sorted(sm_stretch),
            "scorecards": same, "ranked_eligible": rank(same)},
        "real_check_tmc2_tc": {m: {**s, "verdict_is_post_hoc": m in post_hoc}
                               for m, s in real.items()},
    }
    dst = ROOT / "reports/verify_default_rescore.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("recount == stored:", out["recount_matches_stored_false_confidences"])
    print("success-despite-REJECTED runs:", len(inconsistent))
    print("original ranking:", out["original_protocol"]["ranked_eligible"])
    print("same-modality ranking:", out["same_modality_only"]["ranked_eligible"])
    for m, s in real.items():
        print(f"real {m:18s} passes_3A={s['passes_3A']}  rejected={s['rejected_windows']}  "
              f"max_vs_xoftr={s['max_vs_xoftr_rms_px']}  total={s['total_seconds']}s")
    print("wrote", dst)


if __name__ == "__main__":
    main()
