"""Lossless match-point CSV and lunar GeoJSON export (OUT-02, OUT-03)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import MatchSet

FIELDS = ("idx", "src_x", "src_y", "ref_x", "ref_y", "src_lat", "src_lon",
          "ref_lat", "ref_lon", "confidence", "is_inlier", "method", "regime",
          "cascade_stage", "grid_cell")
# Written when the bundle maps its frames to the products (bundle.src_to_product /
# ref_to_product): each point's position in the source and reference PRODUCTS, in their native
# pixels. src_lat/src_lon are always the SOURCE product's own label geolocation (before
# registration); ref_lat/ref_lon are where the registration puts the point on the Moon.
PRODUCT_FIELDS = ("src_product_x", "src_product_y")
REF_PRODUCT_FIELDS = ("ref_product_x", "ref_product_y")


def export_bundle(out_dir: str | Path, bundle, *, src_model=None, ref_model=None,
                  grid: int = 8, crs: Any = "IAU_2015:30100") -> tuple[Path, Path]:
    """Export the final points from ``pipeline.register_bundle()``.

    `src_model` / `ref_model` describe the FULL products; each plane's `tile_origin` is applied.

    This deliberately accepts a bundle rather than a ``RegistrationResult`` so
    callers cannot accidentally export the evidence matches. Part 2 documents
    ``bundle.delivered.confidence`` as placeholder ones until selection carries
    the original per-point scores through the fine stage.
    """
    delivered = bundle.delivered
    if delivered.stage != "delivered":
        raise ValueError("bundle.delivered must contain the final delivered points")
    # Models describe the full products; the points are in each WINDOW's pixels (audit C-07),
    # or, for a source registered in a resampled frame, mapped by bundle.src_to_product (C-01).
    from ..geometry.projection import framed_model, window_model
    to_product = getattr(bundle, "src_to_product", None)
    src_model = src_model or _model_from_plane(bundle.src)
    src_model = (framed_model(src_model, to_product) if to_product is not None
                 else window_model(src_model, getattr(bundle.src, "tile_origin", (0, 0))))
    ref_model = window_model(ref_model or _model_from_plane(bundle.ref),
                             getattr(bundle.ref, "tile_origin", (0, 0)))
    shape = tuple(np.asarray(bundle.src.array).shape[:2])
    mask = np.ones(len(delivered.src_pts), dtype=bool)
    out_dir = Path(out_dir)
    ref_to_product = getattr(bundle, "ref_to_product", None)
    csv_path = write_csv(out_dir / "matches.csv", delivered, mask, src_model=src_model,
                         ref_model=ref_model, image_shape=shape, grid=grid, src_to_product=to_product,
                         ref_to_product=ref_to_product)
    geojson_path = write_geojson(out_dir / "matches.geojson", delivered, mask,
                                 src_model=src_model, ref_model=ref_model,
                                 image_shape=shape, grid=grid, crs=crs, src_to_product=to_product,
                                 ref_to_product=ref_to_product)
    return csv_path, geojson_path


def write_csv(path: str | Path, matches: MatchSet, inlier_mask: np.ndarray, *,
              src_model=None, ref_model=None, image_shape: tuple[int, int] | None = None,
              grid: int = 8, src_to_product=None, ref_to_product=None) -> Path:
    """Write sub-pixel match coordinates without rounding them.

    With `src_to_product` (3x3, src frame px -> source product px) two more columns give each
    point's position in the source PRODUCT: src_product_x, src_product_y.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows(matches, inlier_mask, src_model=src_model, ref_model=ref_model,
                 image_shape=image_shape, grid=grid, src_to_product=src_to_product,
                 ref_to_product=ref_to_product)
    fields = (FIELDS + (PRODUCT_FIELDS if src_to_product is not None else ())
              + (REF_PRODUCT_FIELDS if ref_to_product is not None else ()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: str | Path) -> tuple[MatchSet, np.ndarray]:
    """Read an exported CSV back into its MatchSet and inlier mask."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return (MatchSet(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, np.float32),
                         "", "same_modal_normal", ""), np.zeros(0, bool))
    for field in FIELDS:
        if field not in rows[0]:
            raise ValueError(f"missing match-point column: {field}")
    for field in ("method", "regime", "cascade_stage"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"column {field!r} must be constant for one MatchSet")
    src = np.array([[float(r["src_x"]), float(r["src_y"])] for r in rows])
    ref = np.array([[float(r["ref_x"]), float(r["ref_y"])] for r in rows])
    confidence = np.array([float(r["confidence"]) for r in rows], np.float32)
    mask = np.array([r["is_inlier"].strip().lower() in ("1", "true") for r in rows])
    first = rows[0]
    return (MatchSet(src, ref, confidence, first["method"], first["regime"],
                     first["cascade_stage"]), mask)


def write_geojson(path: str | Path, matches: MatchSet, inlier_mask: np.ndarray, *,
                  ref_model, src_model=None, image_shape: tuple[int, int] | None = None,
                  grid: int = 8, crs: Any = "IAU_2015:30100", src_to_product=None,
                  ref_to_product=None) -> Path:
    """Write reference-ground match points as GeoJSON for direct QGIS use."""
    if ref_model is None:
        raise ValueError("ref_model is required: GeoJSON geometry must be ground coordinates")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows(matches, inlier_mask, src_model=src_model, ref_model=ref_model,
                 image_shape=image_shape, grid=grid, src_to_product=src_to_product,
                 ref_to_product=ref_to_product)
    features = []
    for row in rows:
        lon, lat = row["ref_lon"], row["ref_lat"]
        properties = {key: value for key, value in row.items() if key not in ("ref_lon", "ref_lat")}
        features.append({"type": "Feature", "id": row["idx"],
                         "geometry": {"type": "Point", "coordinates": [lon, lat]},
                         "properties": properties})
    crs_name = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    document = {"type": "FeatureCollection", "name": path.stem,
                "crs": {"type": "name", "properties": {"name": crs_name}},
                "features": features}
    path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _rows(matches: MatchSet, inlier_mask: np.ndarray, *, src_model=None, ref_model=None,
          image_shape=None, grid: int = 8, src_to_product=None, ref_to_product=None) -> list[dict]:
    src = np.asarray(matches.src_pts, np.float64)
    ref = np.asarray(matches.ref_pts, np.float64)
    confidence = np.asarray(matches.confidence, np.float32)
    mask = np.asarray(inlier_mask)
    if src.ndim != 2 or src.shape[1:] != (2,) or ref.shape != src.shape:
        raise ValueError("match coordinates must both have shape (N, 2)")
    if confidence.shape != (len(src),) or mask.shape != (len(src),):
        raise ValueError("confidence and inlier_mask must have shape (N,)")
    if grid <= 0:
        raise ValueError("grid must be positive")
    src_lat, src_lon = _latlon(src_model, src)
    ref_lat, ref_lon = _latlon(ref_model, ref)
    cells = _grid_cells(src, image_shape, grid)
    rows = []
    for i in range(len(src)):
        rows.append({
            "idx": i, "src_x": float(src[i, 0]), "src_y": float(src[i, 1]),
            "ref_x": float(ref[i, 0]), "ref_y": float(ref[i, 1]),
            "src_lat": src_lat[i], "src_lon": src_lon[i],
            "ref_lat": ref_lat[i], "ref_lon": ref_lon[i],
            "confidence": float(confidence[i]), "is_inlier": bool(mask[i]),
            "method": matches.method, "regime": matches.regime,
            "cascade_stage": matches.stage, "grid_cell": cells[i],
        })
    for M, pts, fields in ((src_to_product, src, PRODUCT_FIELDS), (ref_to_product, ref, REF_PRODUCT_FIELDS)):
        if M is None:
            continue
        prod = np.c_[pts, np.ones(len(pts))] @ np.asarray(M, np.float64).T
        for row, (x, y) in zip(rows, prod[:, :2]):
            row[fields[0]], row[fields[1]] = float(x), float(y)
    return rows


def _latlon(model, points: np.ndarray) -> tuple[list, list]:
    if model is None:
        return [""] * len(points), [""] * len(points)
    lat, lon = model.pixel_to_latlon(points[:, 1], points[:, 0])
    return np.asarray(lat, float).tolist(), np.asarray(lon, float).tolist()


def _grid_cells(points: np.ndarray, shape, grid: int) -> list[str]:
    if shape is None:
        return [""] * len(points)
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("image_shape values must be positive")
    cx = np.clip((points[:, 0] / w * grid).astype(int), 0, grid - 1)
    cy = np.clip((points[:, 1] / h * grid).astype(int), 0, grid - 1)
    return [f"{int(y)},{int(x)}" for y, x in zip(cy, cx)]


def _model_from_plane(plane):
    meta = getattr(plane, "meta", None)
    if meta is None:
        raise ValueError("a ground model is required when the bundle plane has no SceneMeta")
    from ..geometry.projection import geolocation_model
    return geolocation_model(meta)
