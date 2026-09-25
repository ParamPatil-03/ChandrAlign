"""The release's measured sub-pixel recovery error (audit I-10: Metrics.subpixel_recovery_err_px).

`subpixel_recovery_err_px` was declared, displayed by the report, and never filled. It is a property
of the RELEASE, not of one registration: how far the delivered points land from the truth on a known
warp of real CH-2 texture. It is read from the committed known-warp confirmation run
(`scripts/known_warp_harness.py --set fresh`, docs/refinement_geometry_protocol.md): the WORST p95 of
the delivered points over the scored cases, in reference px. Missing file -> None, never a guess (H1).
"""
from __future__ import annotations

import functools
import json

from .. import config

SELFTEST = "reports/known_warp_fresh.json"


@functools.lru_cache(maxsize=1)
def subpixel_recovery() -> dict:
    path = config.ROOT / SELFTEST
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    except (OSError, ValueError, KeyError):
        return {"value_px": None, "source": f"{SELFTEST} unavailable"}
    p95 = [c["points"]["library"]["p95"] for c in cases
           if c.get("scored") and c.get("status") == "ok" and "library" in c.get("points", {})]
    if not p95:
        return {"value_px": None, "source": f"{SELFTEST} has no scored cases"}
    return {"value_px": float(max(p95)), "statistic": "worst p95 of delivered points, reference px",
            "cases": len(p95), "source": SELFTEST}
