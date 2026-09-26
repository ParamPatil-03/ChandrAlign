"""Command-line entry point for registration and evaluation workflows (UI-01)."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chandralign", description="Lunar image registration")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser(
        "register", help="register a source product against a reference product",
        description="Registers two real products on the validated path for their pairing "
                    "(TMC-2 -> SELENE TC, OHRC -> LRO NAC): windows placed where the products "
                    "overlap, coarse lock, routed matcher, fine stage, control gates, tier. Any "
                    "other pairing is refused with the reason. --mock runs a synthetic pair.")
    register.add_argument("--src", type=Path, help="source PDS3/PDS4 label")
    register.add_argument("--ref", type=Path, help="reference PDS3/PDS4 label")
    register.add_argument("--out", type=Path, required=True, help="new run directory")
    register.add_argument("--windows", type=int, default=3,
                          help="windows registered across the overlap (default 3)")
    register.add_argument("--matcher", default=None,
                          help="--mock only: the matcher to use (default sift). Real products use "
                               "the matcher routing chose for the pairing, as the evidence did.")
    register.add_argument("--cpu", action="store_true", help="force CPU execution")
    register.add_argument("--mock", action="store_true", help="run an offline synthetic pair instead of labels")
    register.add_argument("--seed", type=int, default=7, help="synthetic-pair seed with --mock")
    register.set_defaults(handler=_register)

    clip = sub.add_parser("fetch-wac-clip", help="cut the LROC WAC global-mosaic reference for an IIRS scene",
                          description="Downloads only the needed part of the public LROC WAC global 100 m mosaic "
                                      "(USGS) around an IIRS scene, for use as --ref in register.")
    clip.add_argument("--iirs", type=Path, required=True, help="the IIRS PDS4 label (.xml)")
    clip.add_argument("--out-dir", type=Path, default=Path("data/raw/lro/wac_mosaic"), help="where to write the clip")
    clip.set_defaults(handler=_fetch_wac_clip)

    report = sub.add_parser("report", help="render the HTML report for a run")
    report.add_argument("--run", type=Path, required=True, help="run directory")
    report.add_argument("--out", type=Path, help="output HTML (default: RUN/report.html)")
    report.set_defaults(handler=_report)
    demo = sub.add_parser("demo", help="offline demonstration: a SYNTHETIC registration and its report")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"), help="demo output directory")
    demo.add_argument("--seed", type=int, default=7, help="synthetic-pair seed")
    demo.set_defaults(handler=_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


def _new_run_dir(path: Path) -> Path:
    out = path.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _register(args) -> int:
    out = _new_run_dir(args.out)
    if args.mock:
        return _register_mock(args, out)
    if args.src is None or args.ref is None:
        raise SystemExit("--src and --ref are required unless --mock is used")
    if args.matcher:
        raise SystemExit("--matcher applies to --mock only: real products use the matcher routing "
                         "chose for their pairing, which is what the committed evidence measured")
    if args.windows < 1:
        raise SystemExit("--windows must be at least 1")
    from .workflows.products import UnsupportedPairing, register_products
    try:
        run = register_products(args.src, args.ref, windows=args.windows,
                                device="cpu" if args.cpu else None, progress=print)
    except UnsupportedPairing as exc:
        raise SystemExit(f"refused: {exc}")
    except ValueError as exc:                 # no overlap: failure mode 10, before any matching
        raise SystemExit(f"refused: {exc}")
    arguments = {"command": "register", "src": str(args.src), "ref": str(args.ref),
                 "windows": args.windows, "device": "cpu" if args.cpu else "auto"}
    summary = write_product_run(out, run, arguments)
    for w in summary["window_results"]:
        print(f"  {w['dir']}: {w['confidence_tier']}  {w['status']}")
    print(f"{summary['accepted']}/{summary['windows']} windows accepted: {out}")
    return 0


def write_product_run(out: Path, run, arguments: dict) -> dict:
    """Write every window of a workflows.products.ProductRun, plus summary.json; return the summary."""
    from .product import provenance, run_export
    rows = []
    for k, w in enumerate(run.windows, 1):
        wdir = out / f"window_{k:02d}"
        extra = {"window": _window_brief(w.window), "pairing": run.pairing, "workflow": run.workflow}
        if w.bundle is None:
            rec = run_export.write_unregistered(wdir, w.failure, extra=extra)
        else:
            manifest = provenance.build(w.bundle, run_arguments=arguments)
            rec = run_export.write_run(wdir, w.bundle, manifest=manifest, src_model=w.src_model,
                                       ref_model=w.ref_model, heights_at=w.bundle.heights_at, extra=extra)
        rows.append({"dir": wdir.name, "confidence_tier": rec["confidence_tier"],
                     "failure_modes": rec["failure_modes"], "status": w.window.get("status"),
                     "consistent": w.window.get("consistent"), "metrics": rec.get("metrics"),
                     "gates": rec.get("gates"), "geometry_used": rec.get("geometry_used"),
                     "matcher": rec.get("matcher"), "exports": rec.get("exports")})
    summary = {**run.summary(), "run_arguments": arguments, "window_results": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                                      encoding="utf-8")
    return summary


def _window_brief(rec: dict) -> dict:
    """The evidence record without its bulky per-matcher detail (that stays in result.json's stages)."""
    keep = ("tmc_row", "tmc_col", "ohrc_row", "ohrc_col", "nac", "lat", "lon", "status", "coarse_z",
            "coarse", "system_offset_m", "scale_status", "tier", "vs_isro_refined_grid_m", "source_px",
            "mi_check", "consistent", "consistency_rule", "coarse_lock_failed", "tmc_pixel_from_tc_m")
    out = {k: rec[k] for k in keep if k in rec}
    routed = (rec.get("results") or {}).get("routed")
    if routed:
        out["routed"] = {k: routed.get(k) for k in ("used", "tried", "tier", "known_shift_error_px",
                                                     "implied_offset_m", "source_px", "mi_check")}
    return out


def _register_mock(args, out: Path) -> int:
    from . import synth
    from .pipeline import register_bundle
    from .product import provenance, run_export
    src, ref, _ = synth.make_pair(out_shape=(256, 256), shift=(3.4, -2.2), seed=args.seed,
                                  n_craters=35, shadows=False)
    _materialize_mock_inputs(src, ref, out)
    model = _MockGroundModel(src.gsd_m)
    bundle = register_bundle(src, ref, matcher=args.matcher or "sift", device="cpu" if args.cpu else None)
    manifest = provenance.build(bundle, run_arguments={"command": "register", "mock": True, "seed": args.seed,
                                                       "matcher": args.matcher or "sift"})
    run_export.write_run(out, bundle, manifest=manifest, src_model=model, ref_model=model)
    print(f"{bundle.result.confidence_tier} (synthetic): {out}")
    return 0


def _demo(args) -> int:
    """Offline: a synthetic registration and its HTML report. Everything in it says SYNTHETIC."""
    out = _new_run_dir(args.out)
    args.mock, args.matcher, args.cpu = True, None, True
    _register_mock(args, out)
    from .product import report
    print(report.render_run(out, None))
    return 0


def _fetch_wac_clip(args) -> int:
    from .io.wac_mosaic import clip_for_iirs
    path = clip_for_iirs(args.iirs, args.out_dir, progress=print)
    print(f"reference clip: {path}  (use it as --ref)")
    return 0


def _report(args) -> int:
    from .product import report
    path = report.render_run(args.run, args.out)
    print(path)
    return 0


def _materialize_mock_inputs(src, ref, out: Path) -> None:
    for role, plane in (("src", src), ("ref", ref)):
        raster = out / f"mock-{role}.npy"
        label = out / f"mock-{role}.json"
        np.save(raster, plane.array)
        label.write_text(json.dumps({"synthetic": True, "seed_role": role}) + "\n", encoding="utf-8")
        plane.meta = replace(plane.meta, raster_path=raster, label_path=label)


class _MockGroundModel:
    """Explicit synthetic lunar grid used only by ``register --mock``."""
    def __init__(self, gsd_m: float):
        self.gsd_m = float(gsd_m)

    def pixel_to_latlon(self, rows, cols):
        metres_per_degree = np.pi * 1_737_400.0 / 180.0
        return (-80.0 - np.asarray(rows) * self.gsd_m / metres_per_degree,
                30.0 + np.asarray(cols) * self.gsd_m /
                (metres_per_degree * np.cos(np.deg2rad(80.0))))


if __name__ == "__main__":
    raise SystemExit(main())
