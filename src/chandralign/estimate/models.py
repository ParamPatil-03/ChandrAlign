"""Transform models: affine, homography, thin-plate spline (ALIGN-02).

Everything here operates on the TransformModel contract object so the cascade
(MATCH-10) can compose hops and the exporter (Member C) can warp a product
without knowing which model type was chosen.

On TPS: a single global homography cannot express local non-rigid distortion
from crater-driven relief, which is exactly the residual structure we expect on
lunar terrain. TPS is therefore available, but opt-in -- it has enough freedom
to absorb genuine mis-registration into a plausible-looking warp, so it is only
justified once `residual_structure` shows the residuals are spatially organised
rather than random.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..contracts import TransformModel


def apply(model: TransformModel, pts: np.ndarray) -> np.ndarray:
    """Map (N,2) source points through the model into reference coordinates."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if model.kind == "tps":
        return _tps_apply(model, pts)
    if model.matrix is None:
        raise ValueError(f"{model.kind} model has no matrix")
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    out = (model.matrix @ homo.T).T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[:, :2] / w


@dataclass
class ParallaxModel:
    """ALIGN-08: ref = matrix . src + (h - h0_m) * p_px_per_m, h the DEM height (m) of the point.

    The terrain-aware geometry of an oblique view (TMC-2 fore/aft, 26 deg). It is NOT a
    contract TransformModel: contracts.py is frozen (PLAN.md section 5), and a new model kind
    needs all three members to agree. Until then it travels in FineResult.parallax and in
    RegistrationResult.provenance["terrain_model"].
    """

    matrix: np.ndarray                  # 3x3 affine part
    p_px_per_m: tuple[float, float]
    h0_m: float
    dem: str = ""
    height_at: str = "src"              # where h was sampled in the fit: "src" point or "ref" (ground) point

    def apply(self, pts: np.ndarray, heights) -> np.ndarray:
        """The model with GIVEN heights (at the source points for "src", at the reference points for "ref")."""
        if heights is None:
            raise ValueError("a parallax model needs the DEM height (m) of each point")
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        h = np.asarray(heights, np.float64).reshape(-1)
        base = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(self.matrix, float)[:2].T
        return base + (h - float(self.h0_m))[:, None] * np.asarray(self.p_px_per_m, float)

    def predict(self, pts: np.ndarray, heights_at, iters: int = 6) -> np.ndarray:
        """Reference positions of source points, heights looked up by `heights_at(frame px)` in the
        model's own convention ("ref": fixed point r = A.s + (h(r) - h0).p, from r = A.s)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.height_at != "ref":
            return self.apply(pts, heights_at(pts))
        r = self.apply(pts, np.full(len(pts), float(self.h0_m)))
        for _ in range(iters):
            h = np.asarray(heights_at(r), np.float64)
            r = self.apply(pts, np.where(np.isfinite(h), h, self.h0_m))
        return r

    def as_dict(self) -> dict:
        return {"kind": "affine_parallax", "matrix": [[round(float(v), 8) for v in row] for row in self.matrix],
                "p_px_per_m": [round(float(v), 7) for v in self.p_px_per_m], "h0_m": round(float(self.h0_m), 2),
                "dem": self.dem, "height_at": self.height_at}


def parallax_source_map(model: ParallaxModel, heights_at, shape: tuple[int, int],
                        step: int = 8, iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """For every REFERENCE pixel, the SOURCE pixel a parallax model sends there.

    Returns (map_x, map_y) for cv2.remap(src, map_x, map_y, ...), i.e. the source warped
    into the reference frame. The model runs source -> reference and the height depends on
    the point, so each reference pixel r is inverted by fixed-point iteration
    s <- A^-1 (r - (h(s) - h0) p); it converges while |grad h . p| < 1 (slopes under ~60 deg for
    TMC-2's 26 deg view). Solved on a grid every `step` px, bilinear in between.
    `heights_at(pts)` -> heights (m) at (N, 2) frame positions; NaN is treated as h0. A model fitted
    with height_at="ref" needs no iteration: s = A^-1 (r - (h(r) - h0) p) directly (the RPC form).
    """
    from scipy.ndimage import map_coordinates
    H, W = shape
    ys, xs = np.mgrid[0:H + step:step, 0:W + step:step].astype(np.float64)
    r = np.c_[np.minimum(xs.ravel(), W - 1), np.minimum(ys.ravel(), H - 1)]
    Ainv = np.linalg.inv(np.asarray(model.matrix, float))
    p, h0 = np.asarray(model.p_px_per_m, float), float(model.h0_m)
    back = lambda q: (np.c_[q, np.ones(len(q))] @ Ainv.T)[:, :2]  # noqa: E731
    if model.height_at == "ref":
        h = np.asarray(heights_at(r), np.float64)
        s = back(r - (np.where(np.isfinite(h), h, h0) - h0)[:, None] * p)
    else:
        s = back(r)
        for _ in range(iters):
            h = np.asarray(heights_at(s), np.float64)
            s = back(r - (np.where(np.isfinite(h), h, h0) - h0)[:, None] * p)
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
    at = [np.minimum(gy, H - 1).ravel() / step, np.minimum(gx, W - 1).ravel() / step]
    mx = map_coordinates(s[:, 0].reshape(ys.shape), at, order=1).reshape(H, W)
    my = map_coordinates(s[:, 1].reshape(ys.shape), at, order=1).reshape(H, W)
    return mx.astype(np.float32), my.astype(np.float32)


def compose(first: TransformModel, second: TransformModel) -> TransformModel:
    """Chain two hops: apply `first`, then `second`.

    Used by the scale-bridging cascade, where a source reaches its reference
    through intermediate instruments and the hops must combine into one
    transform without silently losing the per-hop scale bookkeeping.
    """
    if first.kind == "tps" or second.kind == "tps":
        raise NotImplementedError(
            "TPS hops cannot be composed as matrices; compose them by resampling "
            "control points instead, and record that in the cascade notes."
        )
    if first.matrix is None or second.matrix is None:
        raise ValueError("cannot compose a model with no matrix")
    kind = "homography" if "homography" in (first.kind, second.kind) else "affine"
    matrix = second.matrix @ first.matrix
    scales = [s for s in (first.scale_estimated, second.scale_estimated) if s is not None]
    expected = [s for s in (first.scale_expected, second.scale_expected) if s is not None]
    return TransformModel(
        kind=kind,
        matrix=matrix,
        scale_estimated=float(np.prod(scales)) if scales else None,
        scale_expected=float(np.prod(expected)) if expected else None,
    )


def inverse(model: TransformModel) -> TransformModel:
    if model.kind == "tps" or model.matrix is None:
        raise NotImplementedError("inverse is defined for matrix models only")
    inv = np.linalg.inv(model.matrix)
    est = None if model.scale_estimated in (None, 0) else 1.0 / model.scale_estimated
    exp = None if model.scale_expected in (None, 0) else 1.0 / model.scale_expected
    return TransformModel(kind=model.kind, matrix=inv,
                          scale_estimated=est, scale_expected=exp)


def local_jacobian(matrix: np.ndarray, at: tuple[float, float] = (0.0, 0.0)) -> np.ndarray | None:
    """2x2 Jacobian of a 3x3 transform at a point, or None where it is singular.

    A homography's local linear map varies across the frame, so it is evaluated
    at a chosen point (normally the tile centre).
    """
    matrix = np.asarray(matrix, np.float64)
    x, y = float(at[0]), float(at[1])
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if abs(w) < 1e-12:
        return None
    q = (matrix @ np.array([x, y, 1.0]))[:2] / w
    return (matrix[:2, :2] - np.outer(q, matrix[2, :2])) / w


def estimated_scale(matrix: np.ndarray, at: tuple[float, float] = (0.0, 0.0)) -> float:
    """Local AREA scale of a 3x3 transform at a point: sqrt|det J|.

    One number, and deliberately so -- it is the geometric mean of the two axis
    scales, which is exactly what survives an unknown rotation. It says nothing
    about whether the two axes are scaled DIFFERENTLY; local_axis_scales does.
    """
    jac = local_jacobian(matrix, at)
    if jac is None:
        return float("nan")
    return float(np.sqrt(abs(np.linalg.det(jac))))


def local_axis_scales(matrix: np.ndarray, at: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """(largest, smallest) axis scale of a transform at a point: J's singular values.

    Rotation-invariant, so it can be compared against pixel geometry without
    knowing how the two images are oriented. Their ratio is the transform's
    anisotropy; for a pushbroom camera whose along-track spacing differs from
    its cross-track spacing, a correct transform is NOT isotropic.
    """
    jac = local_jacobian(matrix, at)
    if jac is None:
        return float("nan"), float("nan")
    s = np.linalg.svd(jac, compute_uv=False)
    return float(s[0]), float(s[-1])


def residuals(model: TransformModel, src_pts: np.ndarray,
              ref_pts: np.ndarray) -> np.ndarray:
    """Per-point reprojection error in reference pixels."""
    pred = apply(model, src_pts)
    return np.linalg.norm(pred - np.asarray(ref_pts, np.float64).reshape(-1, 2), axis=1)


def residual_structure(src_pts: np.ndarray, resid: np.ndarray,
                       grid: int = 4) -> float:
    """How spatially organised the residuals are, in [0, 1].

    Random residuals mean the model fits and the error is noise. Residuals that
    cluster by region mean the model is the wrong shape -- the signature of
    local relief a global transform cannot express. Computed as the fraction of
    total residual variance explained by cell means (a one-way ANOVA effect
    size over a coarse grid), so it is scale-free and needs no threshold tuning
    to interpret: near 0 is unstructured, near 1 is strongly structured.
    """
    src_pts = np.asarray(src_pts, np.float64).reshape(-1, 2)
    resid = np.asarray(resid, np.float64).ravel()
    if len(resid) < grid * grid or resid.std() < 1e-12:
        return 0.0
    x0, y0 = src_pts.min(0)
    x1, y1 = src_pts.max(0)
    cx = np.clip(((src_pts[:, 0] - x0) / max(x1 - x0, 1e-9) * grid).astype(int), 0, grid - 1)
    cy = np.clip(((src_pts[:, 1] - y0) / max(y1 - y0, 1e-9) * grid).astype(int), 0, grid - 1)
    cell = cy * grid + cx
    total = float(((resid - resid.mean()) ** 2).sum())
    between = 0.0
    for c in np.unique(cell):
        sel = resid[cell == c]
        if len(sel) >= 2:
            between += len(sel) * (sel.mean() - resid.mean()) ** 2
    return float(np.clip(between / total, 0.0, 1.0)) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Thin-plate spline
# ---------------------------------------------------------------------------
def fit_tps(src_pts: np.ndarray, ref_pts: np.ndarray,
            smoothing: float = 1.0) -> TransformModel:
    """Fit a TPS warp through scipy's RBF interpolator.

    `smoothing` > 0 matters: an exact-interpolation TPS passes through every
    control point including the wrong ones, which would let outliers dictate
    the warp. Fit TPS on inliers only, and keep smoothing non-zero.
    """
    from scipy.interpolate import RBFInterpolator

    src = np.asarray(src_pts, np.float64).reshape(-1, 2)
    ref = np.asarray(ref_pts, np.float64).reshape(-1, 2)
    if len(src) < 3:
        raise ValueError("TPS needs at least 3 correspondences")
    interp = RBFInterpolator(src, ref, kernel="thin_plate_spline",
                             smoothing=float(smoothing))
    return TransformModel(kind="tps", matrix=None,
                          tps_params={"interpolator": interp,
                                      "n_control": int(len(src)),
                                      "smoothing": float(smoothing)})


TPS_SMOOTHING_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def fit_tps_cv(src_pts: np.ndarray, ref_pts: np.ndarray, grid=TPS_SMOOTHING_GRID,
               folds: int = 5, seed: int = 0) -> TransformModel:
    """fit_tps with the smoothing chosen by k-fold cross-validation on the points given.

    The choice uses only these points, so the fitted warp can be judged honestly on any
    OTHER points (docs/tps_protocol.md). The chosen value is kept in tps_params.
    """
    src = np.asarray(src_pts, np.float64).reshape(-1, 2)
    ref = np.asarray(ref_pts, np.float64).reshape(-1, 2)
    if len(src) < 3 * folds:
        return fit_tps(src, ref, smoothing=max(grid))
    idx = np.random.default_rng(seed).permutation(len(src))
    parts = np.array_split(idx, folds)
    best = None
    for s in grid:
        err = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            err.append(apply(fit_tps(src[tr], ref[tr], smoothing=s), src[te]) - ref[te])
        e = float(np.sqrt(np.mean(np.sum(np.concatenate(err) ** 2, axis=1))))
        if best is None or e < best[1]:
            best = (s, e)
    model = fit_tps(src, ref, smoothing=best[0])
    model.tps_params["cv_rms_px"] = round(best[1], 4)
    return model


def _tps_apply(model: TransformModel, pts: np.ndarray) -> np.ndarray:
    params = model.tps_params or {}
    interp = params.get("interpolator")
    if interp is None:
        raise ValueError("TPS model carries no fitted interpolator")
    return np.asarray(interp(pts), np.float64)
