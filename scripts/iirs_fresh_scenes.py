"""IIRS -> WAC mosaic on fresh scenes (docs/iirs_fresh_scenes_protocol.md, frozen before results).

    .venv/Scripts/python scripts/iirs_fresh_scenes.py --scenes <product id> ... [--out reports/iirs_fresh_scenes.json]

Per scene: unzip the PRADAN product if needed, read the label, exclude a night-side scene
(incidence >= 90 deg), cut the WAC global-mosaic clip (io.wac_mosaic), register 5 windows on the
product path (workflows.products -> workflows.iirs_wac, xoftr), and apply the evidence success rule
(registered, gates pass, tier >= LOW, implied offset within 240 m of the scene's median over >= 3
gate-passing windows). Run it on the evidence scene first as a control: it must reproduce 5/5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.io.wac_mosaic import clip_for_iirs  # noqa: E402
from chandralign.workflows.products import register_products  # noqa: E402

RAW = ROOT / "data" / "raw" / "ch2"
CONSISTENT_M = 240.0            # docs/iirs_wac_protocol.md: one WAC pixel-diagonal
N_WIN = 5


def label_of(pid: str) -> Path:
    base = RAW / "iirs" / "products"
    hit = next(iter(sorted(base.rglob(f"{pid}.xml"))), None)
    if hit is None:
        z = RAW / f"{pid}.zip"
        if not z.is_file():
            raise FileNotFoundError(f"{pid}: neither extracted nor {z}")
        with zipfile.ZipFile(z) as zf:
            top = {n.split("/")[0] for n in zf.namelist()}
            zf.extractall(base if top == {pid} else base / pid)
        hit = next(iter(sorted(base.rglob(f"{pid}.xml"))))
    return hit


def score(windows: list[dict], matcher: str) -> tuple[list[dict], dict]:
    rows = []
    for w in windows:
        r = (w.get("results") or {}).get(matcher) or {}
        rows.append({"iirs_line0": w.get("iirs_line0"), "status": r.get("status") or w.get("status"),
                     "tier": r.get("tier"), "gates_pass": bool(r.get("gates_pass")), "tier_ok": bool(r.get("tier_ok")),
                     "implied_offset_m": r.get("implied_offset_m"), "known_shift_error_px": r.get("known_shift_error_px"),
                     "inlier_rmse_px": r.get("inlier_rmse_px"), "source_px": r.get("source_px"),
                     "mi_peak_offset_src_px": ((r.get("mi_check") or {}).get("peak_offset_src_px") or {}).get("value")
                     if isinstance((r.get("mi_check") or {}).get("peak_offset_src_px"), dict)
                     else (r.get("mi_check") or {}).get("peak_offset_src_px"),
                     "coarse": w.get("coarse")})
    gp = [x for x in rows if x["status"] == "registered" and x["gates_pass"] and x["tier_ok"] and x["implied_offset_m"]]
    med = (np.median([[x["implied_offset_m"]["east"], x["implied_offset_m"]["north"]] for x in gp], axis=0)
           if len(gp) >= 3 else None)
    for x in rows:
        io = x["implied_offset_m"]
        x["success"] = bool(x["status"] == "registered" and x["gates_pass"] and x["tier_ok"] and med is not None
                            and io and np.hypot(io["east"] - med[0], io["north"] - med[1]) <= CONSISTENT_M)
    ok = sum(x["success"] for x in rows)
    rate = ok / len(rows) if rows else 0.0
    return rows, {"success": ok, "windows": len(rows),
                  "verdict": "solved" if rate >= 0.9 else "degraded" if rate >= 0.6 else "unsolved",
                  "median_offset_m": None if med is None else [round(float(v), 1) for v in med]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True, help="IIRS product ids (ch2_iir_nci_..._d_img_d18)")
    ap.add_argument("--out", default="reports/iirs_fresh_scenes.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out_path = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    report = {"source": "measured", "protocol": "docs/iirs_fresh_scenes_protocol.md", "run": run_record(), "scenes": {}}
    for pid in args.scenes:
        t0 = time.perf_counter()
        entry: dict = {}
        try:
            label = label_of(pid)
            meta = parse_label(label)
            entry["label_incidence_deg"] = meta.solar_incidence_deg
            entry["corner_latlon"] = meta.corner_latlon
            if meta.solar_incidence_deg is not None and meta.solar_incidence_deg >= 90:
                entry["excluded"] = f"night side: label incidence {meta.solar_incidence_deg:.1f} deg (protocol)"
            else:
                clip = clip_for_iirs(label, ROOT / "data/raw/lro/wac_mosaic", progress=print)
                entry["reference_clip"] = str(clip.relative_to(ROOT))
                run = register_products(label, clip, windows=N_WIN, device=args.device, progress=print)
                matcher = run.matcher_choice.get("matcher") or "xoftr"
                rows, verdict = score([w.window for w in run.windows], matcher)
                entry.update(matcher=matcher, windows=rows, summary=verdict,
                             tiers=[(w.bundle.result.confidence_tier if w.bundle else w.failure["confidence_tier"])
                                    for w in run.windows])
        except Exception as exc:                                  # recorded, never hidden
            entry["error"] = f"{type(exc).__name__}: {exc}"[:400]
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        report["scenes"][pid] = entry
        print(json.dumps({pid: entry.get("summary") or entry.get("excluded") or entry.get("error")}), flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")   # after every scene
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
