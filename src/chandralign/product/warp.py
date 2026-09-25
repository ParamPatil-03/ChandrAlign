"""Registered GeoTIFF and JSON sidecar export (OUT-01)."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import warnings

import numpy as np

from ..estimate import models


def best_geometry(bundle):
    """Return ``(name, model)``: the geometry the fine stage chose by check-point error
    (``bundle.geometry``, audit C-03 / I-01); for bundles without that choice, the old priority."""
    chosen = getattr(bundle, "geometry", None)
    if chosen is not None and getattr(bundle, "geometry_model", None) is not None:
        return chosen, bundle.geometry_model
    if chosen == "parallax" and bundle.parallax is not None:
        return "parallax", bundle.parallax
    if chosen == "tps" and bundle.tps is not None:
        return "tps", bundle.tps
    if chosen is not None and bundle.result.model is not None:
        return bundle.result.model.kind, bundle.result.model
    if bundle.parallax is not None:
        return "parallax", bundle.parallax
    if bundle.tps is not None:
        return "tps", bundle.tps
    if bundle.result.model is not None:
        return bundle.result.model.kind, bundle.result.model
    raise ValueError("registration bundle contains no usable geometry")


def warp_array(bundle, *, heights_at=None, interpolation: int | None = None) -> tuple[np.ndarray, str]:
    """Warp the source array onto the reference pixel grid."""
    import cv2
    src = np.asarray(bundle.src.array)
    h, w = np.asarray(bundle.ref.array).shape[:2]
    name, model = best_geometry(bundle)
    interp = cv2.INTER_CUBIC if interpolation is None else interpolation

    if name in ("parallax", "parallax_tps"):
        if heights_at is None:
            raise ValueError(f"heights_at is required to warp with the {name} model")
        source_map = models.parallax_tps_source_map if name == "parallax_tps" else models.parallax_source_map
        map_x, map_y = source_map(model, heights_at, (h, w))
        warped = cv2.remap(src, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT)
    elif name == "tps":
        delivered = bundle.delivered
        if len(delivered.src_pts) < 3:
            raise ValueError("at least three delivered points are required to invert TPS")
        # the inverse fitted by the fine stage on the SAME points and weights as the forward TPS
        inverse = (model.tps_params or {}).get("inverse")
        if inverse is None:
            smoothing = float((model.tps_params or {}).get("smoothing", 0.0))
            inverse = models.fit_tps(delivered.ref_pts, delivered.src_pts, smoothing=smoothing)
        map_x, map_y = _tps_source_map(inverse, (h, w))
        warped = cv2.remap(src, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT)
    else:
        matrix = np.asarray(model.matrix, np.float64)
        warped = cv2.warpPerspective(src, matrix, (w, h), flags=interp,
                                     borderMode=cv2.BORDER_CONSTANT)
    return warped, name


def _tps_source_map(inverse, shape: tuple[int, int], max_points: int = 262_144):
    """Build a dense inverse map in bounded-memory row chunks."""
    h, w = shape
    map_x = np.empty((h, w), np.float32)
    map_y = np.empty((h, w), np.float32)
    rows_per_chunk = max(1, max_points // max(w, 1))
    for y0 in range(0, h, rows_per_chunk):
        y1 = min(h, y0 + rows_per_chunk)
        yy, xx = np.mgrid[y0:y1, :w]
        source = models.apply(inverse, np.c_[xx.ravel(), yy.ravel()])
        map_x[y0:y1] = source[:, 0].reshape(y1 - y0, w)
        map_y[y0:y1] = source[:, 1].reshape(y1 - y0, w)
    return map_x, map_y


def export_bundle(path: str | Path, bundle, *, ref_model=None, crs=None,
                  heights_at=None, provenance: dict | None = None) -> tuple[Path, Path]:
    """Warp a RegistrationBundle and write GeoTIFF plus a JSON sidecar."""
    try:
        import rasterio
        from rasterio.control import GroundControlPoint
        from rasterio.errors import NotGeoreferencedWarning
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError("GeoTIFF export requires the 'product' extra: pip install -e .[product]") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    warped, geometry_name = warp_array(bundle, heights_at=heights_at)
    ref_model = ref_model or _ground_model(bundle.ref)
    crs = crs or _moon_crs()
    profile = {"driver": "GTiff", "height": warped.shape[0], "width": warped.shape[1],
               "count": 1 if warped.ndim == 2 else warped.shape[2], "dtype": warped.dtype,
               "compress": "deflate"}
    transform = _affine_for(ref_model, Affine)
    if transform is not None:
        profile.update(crs=crs, transform=transform)

    with warnings.catch_warnings():
        # The writer warns before we can assign GCPs. Suppress only that creation-
        # time warning; the completed dataset is verified to contain lunar GCPs.
        if transform is None:
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dataset:
            if warped.ndim == 2:
                dataset.write(warped, 1)
            else:
                dataset.write(np.moveaxis(warped, -1, 0))
            if transform is None:
                dataset.gcps = (_gcps_for(ref_model, warped.shape[:2], GroundControlPoint), crs)
            dataset.update_tags(geometry=geometry_name,
                                source_product=str(bundle.src.meta.product_id),
                                reference_product=str(bundle.ref.meta.product_id))

    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "source": _jsonable(bundle.src.meta), "reference": _jsonable(bundle.ref.meta),
        "geometry": _geometry_record(geometry_name, best_geometry(bundle)[1]),
        "metrics": _jsonable(bundle.result.metrics), "provenance": provenance or {},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, sidecar


def _ground_model(plane):
    from ..geometry.projection import geolocation_model
    if getattr(plane, "meta", None) is None:
        raise ValueError("reference SceneMeta or an explicit ref_model is required")
    return geolocation_model(plane.meta)


def _moon_crs():
    from ..geometry.projection import moon_geographic
    return moon_geographic()


def _affine_for(model, Affine):
    required = ("resolution_px_per_deg", "line_offset", "sample_offset", "center_lat", "center_lon")
    if not all(hasattr(model, name) for name in required):
        return None
    res = float(model.resolution_px_per_deg)
    return Affine(1 / res, 0, float(model.center_lon) - (float(model.sample_offset) + 0.5) / res,
                  0, -1 / res, float(model.center_lat) + (float(model.line_offset) + 0.5) / res)


def _gcps_for(model, shape, GCP, grid: int = 5):
    h, w = shape
    rows = np.linspace(0, h - 1, min(grid, h))
    cols = np.linspace(0, w - 1, min(grid, w))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    lat, lon = model.pixel_to_latlon(rr.ravel(), cc.ravel())
    return [GCP(row=float(r), col=float(c), x=float(x), y=float(y))
            for r, c, x, y in zip(rr.ravel(), cc.ravel(), lon, lat)]


def _geometry_record(name, model):
    if name == "parallax":
        return model.as_dict()
    params = model.tps_params or {}
    return {"kind": name,
            "matrix": None if model.matrix is None else np.asarray(model.matrix).tolist(),
            "n_control": params.get("n_control"), "smoothing": params.get("smoothing"),
            "cv_rms_px": params.get("cv_rms_px")}


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
