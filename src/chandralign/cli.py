"""Command-line entry point for registration and evaluation workflows (UI-01)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chandralign", description="Lunar image registration")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="register a source image against a reference")
    register.add_argument("--src", type=Path, help="source PDS3/PDS4 label")
    register.add_argument("--ref", type=Path, help="reference PDS3/PDS4 label")
    register.add_argument("--out", type=Path, required=True, help="new run directory")
    register.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                          help="configuration YAML (default: configs/default.yaml)")
    register.add_argument("--matcher", default="sift", help="sift, akaze, orb, brisk, rift2 or learned model")
    register.add_argument("--cpu", action="store_true", help="force CPU execution")
    register.add_argument("--tile-size", type=int, default=1024, help="first-tile size for large products")
    register.add_argument("--overlap", type=int, default=128, help="tile overlap in pixels")
    register.add_argument("--src-band", type=int, help="zero-based source band")
    register.add_argument("--ref-band", type=int, help="zero-based reference band")
    register.add_argument("--mock", action="store_true", help="run an offline synthetic pair instead of labels")
    register.add_argument("--seed", type=int, default=7, help="synthetic-pair seed with --mock")
    register.set_defaults(handler=_register)

    benchmark = sub.add_parser("benchmark", help="run curated benchmark tiers")
    benchmark.add_argument("--tiers", default="easy,medium,hard,extreme", help="comma-separated tiers")
    benchmark.set_defaults(handler=_pending)
    ablate = sub.add_parser("ablate", help="run the configured stage ablation")
    ablate.add_argument("--stages", default="all", help="comma-separated stages or 'all'")
    ablate.set_defaults(handler=_pending)
    report = sub.add_parser("report", help="render the HTML report for a run")
    report.add_argument("--run", type=Path, required=True, help="run directory")
    report.set_defaults(handler=_pending)
    demo = sub.add_parser("demo", help="run the offline demonstration")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"), help="demo output directory")
    demo.set_defaults(handler=_pending)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


def _register(args) -> int:
    from . import config
    from .evaluate import failure_log
    from .pipeline import register_bundle
    from .product import matchpoints, provenance, warp
    from .viz import coverage_plot, match_plot, sidebyside, swipe

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    configuration = _load_config(args.config)

    if args.mock:
        from . import synth
        src, ref, _ = synth.make_pair(out_shape=(256, 256), shift=(3.4, -2.2), seed=args.seed,
                                      n_craters=35, shadows=False)
        _materialize_mock_inputs(src, ref, out)
        src_model = ref_model = _MockGroundModel(src.gsd_m)
    else:
        if args.src is None or args.ref is None:
            raise SystemExit("--src and --ref are required unless --mock is used")
        from .io.pds_label import parse_label
        from .io.tiling import iter_tiles
        src_meta, ref_meta = parse_label(args.src), parse_label(args.ref)
        src = next(iter_tiles(src_meta, tile=args.tile_size, overlap=args.overlap, band=args.src_band))
        ref = next(iter_tiles(ref_meta, tile=args.tile_size, overlap=args.overlap, band=args.ref_band))
        src_model = ref_model = None

    bundle = register_bundle(src, ref, matcher=args.matcher, device="cpu" if args.cpu else None)
    manifest = provenance.build(bundle, config_data=configuration,
                                ship_mode=bool(configuration.get("ship_mode", True)))
    provenance.write(out / "provenance.json", manifest)
    matchpoints.export_bundle(out, bundle, src_model=src_model, ref_model=ref_model,
                              grid=int(configuration.get("uniformity", {}).get("grid", 8)))
    try:
        warp.export_bundle(out / "registered.tif", bundle, ref_model=ref_model, provenance=manifest)
    except RuntimeError as exc:
        if "rasterio" not in str(exc).lower():
            raise
    sidebyside.render(bundle, out / "side-by-side.png")
    match_plot.render(bundle, out / "matches.png")
    coverage_plot.render(bundle, out / "coverage.png",
                         grid=int(configuration.get("uniformity", {}).get("grid", 8)))
    registered, _ = warp.warp_array(bundle)
    swipe.render_checkerboard(bundle.ref.array, registered, out / "checkerboard.png")
    (out / "result.json").write_text(json.dumps(_result_record(bundle), indent=2,
                                                  sort_keys=True) + "\n", encoding="utf-8")
    failure_log.log_run(out / "failure-log.jsonl", bundle)
    print(f"{bundle.result.confidence_tier}: {out}")
    return 0


def _pending(args) -> int:
    raise SystemExit(f"'{args.command}' is declared but its planned feature is not implemented yet")


def _load_config(path: Path) -> dict:
    import yaml
    if not path.is_file():
        raise SystemExit(f"configuration not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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


def _result_record(bundle) -> dict:
    result = bundle.result
    return {
        "confidence_tier": result.confidence_tier,
        "metrics": _jsonable(result.metrics),
        "gates": result.gates,
        "failure_modes": result.failure_modes,
        "notes": result.notes,
        "stages": _jsonable(bundle.stages),
        "geometry_used": "parallax" if bundle.parallax is not None else
                         "tps" if bundle.tps is not None else result.model.kind if result.model else None,
    }


def _jsonable(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
