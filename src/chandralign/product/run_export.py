"""Write one registration's run folder: the single export path for the CLI and the API.

Before this module the CLI and the API each carried their own copy of the export sequence,
and they drifted (audit 2026-09-26 I-12, I-14): a REJECTED result with no geometry crashed
both, so its rejection reasons were lost; a REJECTED result WITH a model shipped an unflagged
`registered.tif`; a reference with no ground model (LRO NAC) crashed the exporters; and the
API never wrote the failure log.

The rules here:
- `result.json`, `provenance.json` and `failure-log.jsonl` are ALWAYS written.
- A registered product (`registered.tif`) is issued only for an accepted tier, and carries
  the tier, failure modes and gates in its tags and sidecar.
- Without a ground model the exports that need one are skipped, and `result.json` says why
  (`exports_skipped`); pixel coordinates are still exported. Nothing is skipped silently.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import matchpoints, warp

REJECTED = "REJECTED"


def write_run(out_dir: str | Path, bundle, *, manifest: dict, src_model=None, ref_model=None,
              grid: int = 8, heights_at=None, extra: Optional[dict] = None) -> dict:
    """Write every product of `bundle` into `out_dir`; return the result record.

    `manifest` is the provenance record (product.provenance.build, or a synthetic marker).
    `src_model` / `ref_model` describe the FULL products (default: from each plane's
    SceneMeta); each plane's tile_origin is applied by the exporters. `extra` is merged into
    result.json (e.g. the effective run arguments).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    skipped: dict[str, str] = {}
    written: list[str] = []

    _write_json(out / "provenance.json", manifest)
    written.append("provenance.json")

    src_model, why_src = _usable(src_model, bundle.src)
    ref_model, why_ref = _usable(ref_model, bundle.ref)
    tier = bundle.result.confidence_tier

    # match points: pixel coordinates are exact with or without a ground model
    if ref_model is not None:
        matchpoints.export_bundle(out, bundle, src_model=src_model, ref_model=ref_model, grid=grid)
        written += ["matches.csv", "matches.geojson"]
    else:
        _pixel_only_csv(out, bundle, src_model, grid)
        written.append("matches.csv")
        skipped["matches.geojson"] = f"no ground model for the reference: {why_ref}"

    # the registered product: only for an accepted result with a geometry and a ground model
    has_geometry = any(g is not None for g in (bundle.parallax, bundle.tps, bundle.result.model))
    if tier == REJECTED:
        skipped["registered.tif"] = f"{REJECTED}: no registered product is issued for a rejected result"
    elif not has_geometry:
        skipped["registered.tif"] = "the result carries no geometry to warp with"
    elif ref_model is None:
        skipped["registered.tif"] = f"no ground model for the reference: {why_ref}"
    else:
        try:
            warp.export_bundle(out / "registered.tif", bundle, ref_model=ref_model,
                               heights_at=heights_at, provenance=manifest)
            written += ["registered.tif", "registered.json"]
        except RuntimeError as exc:           # the optional 'product' extra (rasterio) is absent
            if "rasterio" not in str(exc).lower():
                raise
            skipped["registered.tif"] = str(exc)

    written += _figures(out, bundle, has_geometry, grid, heights_at, skipped)

    from ..evaluate import failure_log
    failure_log.log_run(out / "failure-log.jsonl", bundle)
    written.append("failure-log.jsonl")

    record = result_record(bundle)
    record.update(extra or {})
    record["exports"] = sorted(set(written + ["result.json", "report.html"]))
    record["exports_skipped"] = skipped
    if why_src:
        record["source_ground_model"] = f"unavailable: {why_src}"
    _write_json(out / "result.json", record)
    from . import report
    report.render_run(out)                    # the run's own report, from what was just written
    return record


def write_unregistered(out_dir: str | Path, failure: dict, *, extra: Optional[dict] = None) -> dict:
    """A window that never reached a registration (no overlap, no coarse lock, no transform):
    result.json and the failure log still say so, with the failure mode (rule H3)."""
    from types import SimpleNamespace
    from ..evaluate import failure_log
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stub = SimpleNamespace(confidence_tier=failure["confidence_tier"], failure_modes=list(failure["failure_modes"]),
                           notes=[failure.get("status", "")], gates={})
    failure_log.log_run(out / "failure-log.jsonl", stub)
    record = {"confidence_tier": failure["confidence_tier"], "failure_modes": list(failure["failure_modes"]),
              "status": failure.get("status", ""), "gates": {}, "metrics": None,
              "exports": ["failure-log.jsonl", "result.json"],
              "exports_skipped": {"*": "the window never reached a registration: " + str(failure.get("status", ""))}}
    record.update(extra or {})
    _write_json(out / "result.json", record)
    return record


def result_record(bundle) -> dict:
    """The run summary written as result.json (and returned by the API)."""
    result = bundle.result
    geometry = ("parallax" if bundle.parallax is not None else
                "tps" if bundle.tps is not None else
                result.model.kind if result.model is not None else None)
    return {
        "confidence_tier": result.confidence_tier,
        "metrics": _jsonable(result.metrics),
        "gates": result.gates,
        "failure_modes": list(result.failure_modes),
        "notes": list(result.notes),
        "matcher": result.matches.method,
        "regime": result.matches.regime,
        "match_stage": result.matches.stage,
        "stages": _jsonable(getattr(bundle, "stages", {})),
        "geometry_used": geometry,
        "provenance": _jsonable(result.provenance),
        # How the bundle's frames map to the products (px x, y): lets a user take any exported
        # point, or the model, back to native source / reference pixels.
        "frames": {"src_to_product": _jsonable(getattr(bundle, "src_to_product", None)),
                   "ref_to_product": _jsonable(getattr(bundle, "ref_to_product", None))},
    }


def _usable(model, plane) -> tuple[Any, str]:
    """(model, "") if a ground model exists for `plane`, else (None, why)."""
    from ..geometry.projection import GeolocationUnavailable
    try:
        if model is None:
            if getattr(plane, "meta", None) is None:
                return None, "the plane has no SceneMeta"
            from ..geometry.projection import geolocation_model
            model = geolocation_model(plane.meta)
        model.pixel_to_latlon(np.array([0.0]), np.array([0.0]))   # probe: some models fail lazily
        return model, ""
    except GeolocationUnavailable as exc:
        return None, str(exc)


def _pixel_only_csv(out: Path, bundle, src_model, grid: int) -> None:
    from ..geometry.projection import window_model
    delivered = bundle.delivered
    if src_model is not None:
        src_model = window_model(src_model, getattr(bundle.src, "tile_origin", (0, 0)))
    matchpoints.write_csv(out / "matches.csv", delivered, np.ones(len(delivered.src_pts), bool),
                          src_model=src_model, ref_model=None,
                          image_shape=tuple(np.asarray(bundle.src.array).shape[:2]), grid=grid)


def _figures(out: Path, bundle, has_geometry: bool, grid: int, heights_at, skipped: dict) -> list[str]:
    from ..viz import coverage_plot, match_plot, sidebyside, swipe
    sidebyside.render(bundle, out / "side-by-side.png")
    match_plot.render(bundle, out / "matches.png")
    coverage_plot.render(bundle, out / "coverage.png", grid=grid)
    names = ["side-by-side.png", "matches.png", "coverage.png"]
    _quicklook(out / "reference.png", bundle.ref.array)
    names.append("reference.png")
    why = ("the result carries no geometry to warp with" if not has_geometry else
           "the parallax geometry needs terrain heights to warp"
           if bundle.parallax is not None and heights_at is None else None)
    if why:
        skipped["checkerboard.png"] = skipped["registered.png"] = why
        return names
    registered, _ = warp.warp_array(bundle, heights_at=heights_at)
    swipe.render_checkerboard(bundle.ref.array, registered, out / "checkerboard.png")
    _quicklook(out / "registered.png", registered)   # the source warped onto the reference grid
    return names + ["checkerboard.png", "registered.png"]


def _quicklook(path: Path, array) -> None:
    """8-bit greyscale PNG of a 0..1 plane, pixel for pixel (the UI's swipe compares these)."""
    import cv2
    a = np.nan_to_num(np.asarray(array, np.float32), nan=0.0)
    cv2.imwrite(str(path), np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
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
