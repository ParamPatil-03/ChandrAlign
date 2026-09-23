"""Score xoftr speed variants against docs/verify_default_protocol.md section 3.B.

    .venv/Scripts/python scripts/score_xoftr_speed.py reports/xoftr_speed_*.json

Criterion 3.B.2 compares each window's composed transform with the COMMITTED
xoftr run (reports/tmc2_tc_xoftr.json) and requires the difference to be no
larger than the committed eloftr-vs-xoftr difference on that window. The V0
re-run's own difference is the run-to-run noise floor.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_real_check import ELOFTR_VS_XOFTR, grid_rms, rows  # noqa: E402

BUDGET_S = 180.0
TIER_RANK = {"REJECTED": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def score(path, v0_path=ROOT / "reports/xoftr_speed_V0_fp32_whole.json"):
    X = rows(ROOT / "reports/tmc2_tc_xoftr.json")
    R = rows(path)
    V0 = rows(v0_path) if Path(v0_path).exists() else {}
    wins, fails = [], []
    for k, env in ELOFTR_VS_XOFTR.items():
        r = R.get(k)
        if r is None or r.get("status") != "registered":
            fails.append(f"{k}: not registered")
            continue
        d = grid_rms(r, X[k])
        tier0 = V0.get(k, {}).get("tier", X[k]["tier"])
        w = {"tmc_row": k, "tier": r["tier"], "tier_v0": tier0,
             "gates_failed": [g for g, v in r["gates"].items() if not v],
             "vs_committed_xoftr_px": round(d, 3), "envelope_px": env,
             "perturbation_error_px": r["gate_detail"]["perturbation_sensitivity"].get("error_px"),
             "match_s": r["match_seconds"], "gate_s": r["gate_seconds"], "total_s": r["seconds"],
             "inliers": r["inliers"], "inlier_ratio": r["inlier_ratio"],
             "inlier_rmse_px": r["inlier_rmse_px"], "coverage": r["coverage"],
             "tmc_pixel_m": r["tmc_pixel_from_tc_m"]}
        if w["gates_failed"]:
            fails.append(f"{k}: gates {w['gates_failed']}")
        if TIER_RANK[r["tier"]] < TIER_RANK[tier0]:
            fails.append(f"{k}: tier {r['tier']} worse than V0 {tier0}")
        if d > env:
            fails.append(f"{k}: {d:.3f} px from committed xoftr > envelope {env}")
        wins.append(w)
    total = sum(w["total_s"] for w in wins)
    first = wins[0]["total_s"] if wins else 0.0
    return {"file": str(path), "match_options": json.loads(Path(path).read_text()).get("match_options"),
            "total_s": round(total, 1),
            "total_without_first_window_s": round(total - first, 1),
            "match_s": round(sum(w["match_s"] for w in wins), 1),
            "gate_s": round(sum(w["gate_s"] for w in wins), 1),
            "under_budget": total < BUDGET_S, "criteria_failures": fails,
            "acceptable_3B": total < BUDGET_S and not fails and len(wins) == 9,
            "max_vs_committed_px": max((w["vs_committed_xoftr_px"] for w in wins), default=None),
            "windows": wins}


if __name__ == "__main__":
    allres = []
    for p in sys.argv[1:]:
        s = score(p)
        allres.append(s)
        print(f"\n== {p}  options={s['match_options']}")
        print(f"   total {s['total_s']} s (match {s['match_s']}, gates {s['gate_s']}; "
              f"without window 1: {s['total_without_first_window_s']})  under 180 s: {s['under_budget']}")
        print(f"   max vs committed xoftr {s['max_vs_committed_px']} px; failures: {s['criteria_failures']}")
        print(f"   ACCEPTABLE (3.B): {s['acceptable_3B']}")
        for w in s["windows"]:
            print("     ", {k: w[k] for k in ("tmc_row", "tier", "vs_committed_xoftr_px", "envelope_px",
                                               "perturbation_error_px", "match_s", "gate_s", "inliers",
                                               "inlier_rmse_px")})
