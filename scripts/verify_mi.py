"""MATCH-07: MI ranks misalignments and flags a biased model (docs/mi_protocol.md).

    python scripts/verify_mi.py <TMC-2 -> TC dump dir> <IIRS -> WAC dump dir>

Writes reports/mi_check.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.matching.similarity import _nmi_at, alignment_check  # noqa: E402

DIRS = [(np.cos(a), np.sin(a)) for a in np.arange(8) * np.pi / 4]


def check_window(z):
    src, ref = z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32)
    so, ro = np.asarray(z["src_ok"], bool), np.asarray(z["ref_ok"], bool)
    M = np.asarray(z["model"], float)
    if M.shape == (2, 3):
        M = np.vstack([M, [0, 0, 1.0]])
    rank = {}
    for s in (0, 1, 2, 4):
        vals = []
        for dx, dy in (DIRS if s else [(0.0, 0.0)]):
            T = M.copy(); T[0, 2] += s * dx; T[1, 2] += s * dy
            vals.append(_nmi_at(src, ref, T, so, ro, 12, 64))
        rank[s] = float(np.mean(vals))
    seq = [rank[s] for s in (0, 1, 2, 4)]
    base = alignment_check(src, ref, M, src_ok=so, ref_ok=ro)
    biased = []
    for dx, dy in DIRS:
        T = M.copy(); T[0, 2] += 2 * dx; T[1, 2] += 2 * dy
        c = alignment_check(src, ref, T, src_ok=so, ref_ok=ro)
        rec = np.array(c["peak_offset_px"]) if c["peak_offset_px"] else np.array([np.nan, np.nan])
        biased.append({"flag": c["flag"], "recovery_err_px": round(float(np.hypot(*(rec + 2 * np.array([dx, dy])))), 3)})
    return {"nmi_by_shift": {str(k): round(v, 5) for k, v in rank.items()},
            "ranked": bool(all(seq[i] > seq[i + 1] for i in range(3))),
            "unbiased": base, "biased": biased}


def pairing(d: Path):
    return {f.stem: check_window(np.load(f)) for f in sorted(d.glob("window_*.npz"))}


def main() -> int:
    res = {"TMC-2 -> TC": pairing(Path(sys.argv[1])), "IIRS -> WAC": pairing(Path(sys.argv[2]))}
    verdict = {}
    for name, ws in res.items():
        n = len(ws)
        ranked = sum(w["ranked"] for w in ws.values())
        flags = [b["flag"] for w in ws.values() for b in w["biased"]]
        errs = [b["recovery_err_px"] for w in ws.values() for b in w["biased"]]
        false = sum(bool(w["unbiased"]["flag"]) for w in ws.values())
        need_rank = 14 if name.startswith("TMC") else n
        verdict[name] = {"windows": n, "ranked": ranked, "a_pass": ranked >= need_rank,
                         "biased_flagged": f"{sum(flags)}/{len(flags)}", "b_flag_pass": sum(flags) >= 0.95 * len(flags),
                         "median_recovery_err_px": round(float(np.median(errs)), 3),
                         "b_recovery_pass": float(np.median(errs)) <= 0.5,
                         "false_flags": f"{false}/{n}", "c_pass": false <= 0.10 * n}
    verdict["adopt"] = all(v["a_pass"] and v["b_flag_pass"] and v["b_recovery_pass"] and v["c_pass"]
                           for v in verdict.values() if isinstance(v, dict))
    out = {"source": "measured", "protocol": "docs/mi_protocol.md", "run": run_record(), "verdict": verdict, "windows": res}
    (ROOT / "reports/mi_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
