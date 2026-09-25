"""Rule R validation (docs/mi_abstain_protocol.md): abstain on flat MI, never on a real 2 px error.

    python scripts/verify_mi_abstain.py <tmc2 dumps> <iirs-wac dumps> <ohrc-nac dumps>
Writes reports/mi_abstain_check.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.matching.similarity import alignment_check  # noqa: E402

DIRS = [(np.cos(a), np.sin(a)) for a in np.arange(8) * np.pi / 4]


def frames(files):
    for f in files:
        z = np.load(f)
        M = np.asarray(z["model"], float)
        M = np.vstack([M, [0, 0, 1.0]]) if M.shape == (2, 3) else M
        yield f.stem, z["src_img"].astype(np.float32), z["ref_img"].astype(np.float32), \
            np.asarray(z["src_ok"], bool), np.asarray(z["ref_ok"], bool), M


def check(s, r, so, ro, M):
    return alignment_check(s, r, M, src_ok=so, ref_ok=ro, abstain=True)


def main() -> int:
    t, i, o = (Path(a) for a in sys.argv[1:4])
    good = sorted(t.glob("window_*.npz")) + sorted(i.glob("window_*.npz")) + \
        sorted(o.glob("M102014464RC_*_eloftr.npz")) + sorted(o.glob("M106719774LC_*_eloftr.npz"))
    low = sorted(o.glob("M175124932LC_*_eloftr.npz")) + sorted(o.glob("M109080308LC_*_eloftr.npz"))
    v1, v2, v3 = [], [], []
    for name, s, r, so, ro, M in frames(good):
        c = check(s, r, so, ro, M)
        v1.append({"window": name, "abstain": c.get("abstain"), "flag": c["flag"], "halves_px": c.get("half_disagreement_px")})
        for dx, dy in DIRS:
            B = M.copy(); B[0, 2] += 2 * dx; B[1, 2] += 2 * dy
            cb = check(s, r, so, ro, B)
            v2.append({"window": name, "flag": cb["flag"], "abstain": cb.get("abstain")})
        print(name, v1[-1], "biased flagged", sum(x["flag"] for x in v2[-8:]), "/8", flush=True)
    for name, s, r, so, ro, M in frames(low):
        c = check(s, r, so, ro, M)
        v3.append({"window": name, "nmi": c["nmi"], "abstain": c.get("abstain"), "flag": c["flag"],
                   "halves_px": c.get("half_disagreement_px")})
        print("low-info", v3[-1], flush=True)
    ab1 = sum(bool(x["abstain"]) for x in v1)
    fl2 = sum(bool(x["flag"]) for x in v2)
    verdict = {"V1_abstain": f"{ab1}/{len(v1)}", "V1_pass": ab1 <= 0.10 * len(v1),
               "V2_flagged": f"{fl2}/{len(v2)}", "V2_pass": fl2 >= 0.95 * len(v2),
               "V3_abstain": f"{sum(bool(x['abstain']) for x in v3)}/{len(v3)}",
               "V3_flag": f"{sum(bool(x['flag']) for x in v3)}/{len(v3)}"}
    verdict["adopt"] = bool(verdict["V1_pass"] and verdict["V2_pass"])
    (ROOT / "reports/mi_abstain_check.json").write_text(json.dumps({"source": "measured", "protocol": "docs/mi_abstain_protocol.md",
                                                                    "run": run_record(), "verdict": verdict, "V1": v1, "V2": v2, "V3": v3},
                                                                   indent=1), encoding="utf-8")
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
