"""Which geometry is delivered, and how accurate it is on points it never saw (C-03, I-01; G-02).

WHY THIS EXISTS (docs/AUDIT_2026-09-26.md)
- I-01: the product ALWAYS warped with the TPS when one existed, and that TPS was fitted on
  the <= 384 thinned points. On a known RIGID warp that TPS was 3-5x worse than the affine
  (0.089-0.119 vs 0.021-0.036 px); on a non-rigid one it was far better. Neither choice is
  right always, so the choice is made per registration, by cross-validation.
- G-02: the delivered geometry is fitted on ALL (sampled) REFINED inliers, not on the thinned
  export points and not on the matcher's unrefined positions.
- C-03: the "RMSE" reported was the affine's in-sample residual on inliers pre-screened at
  3 px -- 1.14 px on a harness whose delivered TPS was truly 0.075 px wrong. The number
  reported now is the CROSS-VALIDATED CHECK-POINT error of the geometry actually delivered.

THE FIT SET. Up to `geometry.max_fit_points` first-estimate inliers, sampled evenly over a
grid (a crowded corner cannot dominate), refined per point by the caller. The 384 uniform
points stay the EXPORTED match points; they are not what the geometry is fitted on.

THE CANDIDATES, each fitted on the fit set with Huber IRLS (a stray inlier cannot pull it):
    affine    6 parameters
    parallax  affine + DEM height x parallax (ALIGN-08), where the fine stage fitted one
    tps       thin-plate spline, smoothing chosen by the same cross-validation

HOW THEY ARE SCORED. k-fold cross-validation, every candidate on the same folds: each fold is
predicted by the candidate fitted on the other folds, so every fit point is a check point once.
`geometry.cv_scheme` picks the folds:
    stratified     each grid cell's points are dealt into the k folds (the audit's "hold out
                   ~20% stratified by grid cell"): a check point keeps neighbours in training,
                   as every point of the delivered geometry does
    spatial_block  whole grid cells per fold: prediction across a cell-sized gap
Every figure includes the check points' own matching noise, so it is an upper bound on the
geometry's error, not the error itself.

THE CHOICE. Lowest check-point RMS wins, except that a richer model must beat a simpler one
by `geometry.min_gain` (fraction) to displace it: a TPS has enough freedom to look better by
chance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .. import config
from ..contracts import TransformModel
from . import models

SIMPLICITY = ("affine", "parallax", "tps")        # simplest first; "parallax_tps" is judged against both parents


@dataclass
class Selection:
    name: str                                    # "affine" | "parallax" | "tps"
    model: object                                # TransformModel, or models.ParallaxModel
    checkpoint_rmse_px: Optional[float]          # of the chosen model, reference px
    candidates: dict = field(default_factory=dict)
    n_fit: int = 0
    notes: list[str] = field(default_factory=list)
    tps: Optional[TransformModel] = None         # the fitted TPS candidate, chosen or not
    affine: Optional[TransformModel] = None      # the refitted affine candidate
    parallax: Optional[models.ParallaxModel] = None

    def record(self) -> dict:
        return {"delivered": self.name, "checkpoint_rmse_px_ref": _r(self.checkpoint_rmse_px),
                "n_fit_points": self.n_fit, "candidates": self.candidates, "notes": self.notes}


def _r(v, n=4):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def _rms(e) -> float:
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(np.sum(e.reshape(len(e), -1) ** 2, axis=1)))) if len(e) else float("nan")


def cells(pts: np.ndarray, shape: tuple[int, int], grid: int) -> np.ndarray:
    h, w = shape
    cx = np.clip((np.asarray(pts)[:, 0] * grid / max(w, 1)).astype(int), 0, grid - 1)
    cy = np.clip((np.asarray(pts)[:, 1] * grid / max(h, 1)).astype(int), 0, grid - 1)
    return cy * grid + cx


def stratified_sample(pts: np.ndarray, shape: tuple[int, int], n_max: int, grid: int = 8,
                      seed: int = 0) -> np.ndarray:
    """Indices of at most n_max points, as evenly spread over the grid cells as the points allow."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) <= n_max:
        return np.arange(len(pts))
    rng = np.random.default_rng(seed)
    cid = cells(pts, shape, grid)
    groups = [rng.permutation(np.flatnonzero(cid == c)) for c in np.unique(cid)]
    take, level = [], 0
    while len(take) < n_max:                     # deal one point per cell per round
        added = False
        for g in groups:
            if level < len(g):
                take.append(g[level])
                added = True
                if len(take) == n_max:
                    break
        if not added:
            break
        level += 1
    return np.sort(np.asarray(take, int))


def folds_for(pts: np.ndarray, shape: tuple[int, int], k: int, grid: int, scheme: str,
              seed: int = 0) -> np.ndarray:
    """Fold id per point (see the module docstring for the two schemes)."""
    rng = np.random.default_rng(seed)
    cid = cells(pts, shape, grid)
    out = np.zeros(len(cid), int)
    if scheme == "spatial_block":
        uniq = rng.permutation(np.unique(cid))
        fold_of = {c: i % k for i, c in enumerate(uniq)}
        return np.array([fold_of[c] for c in cid], int)
    if scheme != "stratified":
        raise ValueError(f"unknown geometry.cv_scheme {scheme!r}")
    offset = 0
    for c in np.unique(cid):                     # deal each cell's points round-robin, continuing
        idx = rng.permutation(np.flatnonzero(cid == c))
        out[idx] = (np.arange(len(idx)) + offset) % k
        offset += len(idx)
    return out


def _huber_weights(e: np.ndarray, k: float = 1.345) -> np.ndarray:
    c = k * (1.4826 * float(np.median(e)) + 1e-9)
    return np.where(e <= c, 1.0, c / np.maximum(e, 1e-12))


def _wlstsq(X: np.ndarray, y: np.ndarray, rounds: int) -> np.ndarray:
    """Huber-IRLS linear least squares, y ~ X B."""
    w = np.ones(len(X))
    B = None
    for _ in range(max(1, rounds + 1)):
        sw = np.sqrt(w)[:, None]
        B = np.linalg.lstsq(X * sw, y * sw, rcond=None)[0]
        w_new = _huber_weights(np.hypot(*(X @ B - y).T))
        if np.allclose(w_new, w, atol=1e-3):
            break
        w = w_new
    return B


def fit_affine_robust(src, ref, rounds: int = 2) -> TransformModel:
    src, ref = np.asarray(src, float).reshape(-1, 2), np.asarray(ref, float).reshape(-1, 2)
    B = _wlstsq(np.c_[src, np.ones(len(src))], ref, rounds)
    return TransformModel(kind="affine", matrix=np.vstack([B.T, [0.0, 0.0, 1.0]]))


def fit_parallax_robust(src, ref, h, h0: float, height_at: str, rounds: int = 2) -> models.ParallaxModel:
    """ref = A.src + (h - h0).p with h the DEM height at the model's convention point (known here)."""
    src, ref = np.asarray(src, float).reshape(-1, 2), np.asarray(ref, float).reshape(-1, 2)
    B = _wlstsq(np.c_[src, np.ones(len(src)), np.asarray(h, float) - h0], ref, rounds)
    return models.ParallaxModel(matrix=np.vstack([B[:3].T, [0.0, 0.0, 1.0]]),
                                p_px_per_m=(float(B[3, 0]), float(B[3, 1])), h0_m=float(h0), height_at=height_at)


def fit_tps_robust(src: np.ndarray, ref: np.ndarray, smoothing: float, rounds: int = 2) -> TransformModel:
    """TPS with Huber IRLS: a point with a large residual gets proportionally more smoothing.

    scipy's RBFInterpolator takes a per-point smoothing; smoothing_i = s / w_i is the
    weighted-least-squares form of the TPS penalty, so reweighting needs no custom solver.
    """
    from scipy.interpolate import RBFInterpolator
    src = np.asarray(src, float).reshape(-1, 2)
    ref = np.asarray(ref, float).reshape(-1, 2)
    w = np.ones(len(src))
    s = max(float(smoothing), 1e-6)
    interp = None
    for _ in range(max(1, rounds + 1)):
        interp = RBFInterpolator(src, ref, kernel="thin_plate_spline", smoothing=s / w)
        w_new = _huber_weights(np.hypot(*(interp(src) - ref).T))
        if np.allclose(w_new, w, atol=1e-3):
            break
        w = w_new
    return TransformModel(kind="tps", matrix=None,
                          tps_params={"interpolator": interp, "n_control": int(len(src)), "smoothing": s,
                                      "robust": "huber_irls", "weights": w})


def _cv(fit: Callable, predict: Callable, n: int, folds: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-point check-point error: each fold predicted by the model fitted on the others."""
    err = np.full(n, np.nan)
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        if tr.sum() < 10:
            continue
        m = fit(tr)
        err[te] = np.hypot(*(predict(m, te) - ref[te]).T)
    return err


def _summary(e: np.ndarray, scored_by: str, **extra) -> dict:
    ok = e[np.isfinite(e)]
    return {"checkpoint_rmse_px": _r(_rms(ok[:, None])) if len(ok) else None,
            "p50_px": _r(np.median(ok)) if len(ok) else None,
            "p95_px": _r(np.percentile(ok, 95)) if len(ok) else None, "scored_by": scored_by, **extra}


def select(src: np.ndarray, ref: np.ndarray, shape: tuple[int, int], *,
           parallax: Optional[models.ParallaxModel] = None,
           heights_at: Optional[Callable[[np.ndarray], np.ndarray]] = None,
           with_tps: bool = True) -> Selection:
    """Fit and cross-validate the candidates on the refined fit set (src -> ref); choose one."""
    src = np.asarray(src, float).reshape(-1, 2)
    ref = np.asarray(ref, float).reshape(-1, 2)
    n = len(src)
    grid = int(config.get("uniformity.grid", 8))
    k = int(config.get("geometry.cv_folds", 5))
    scheme = str(config.get("geometry.cv_scheme", "stratified"))
    min_gain = float(config.get("geometry.min_gain", 0.05))
    rounds = int(config.get("geometry.irls_rounds", 2))
    grid_s = tuple(config.get("geometry.tps_smoothing_grid", list(models.TPS_SMOOTHING_GRID)))
    folds = folds_for(src, shape, k, grid, scheme)
    how = f"{scheme} {k}-fold CV ({grid}x{grid} cells), Huber-IRLS fits"
    cand: dict[str, dict] = {}
    built: dict[str, object] = {}
    notes: list[str] = []

    e = _cv(lambda tr: fit_affine_robust(src[tr], ref[tr], rounds),
            lambda m, te: models.apply(m, src[te]), n, folds, ref)
    cand["affine"] = _summary(e, how)
    built["affine"] = fit_affine_robust(src, ref, rounds)

    if parallax is not None and heights_at is not None:
        at = ref if parallax.height_at == "ref" else src
        h = np.asarray(heights_at(at), float)
        known = np.isfinite(h)
        if known.sum() >= 30:
            s_k, r_k, h_k, f_k = src[known], ref[known], h[known], folds[known]
            e = _cv(lambda tr: fit_parallax_robust(s_k[tr], r_k[tr], h_k[tr], parallax.h0_m, parallax.height_at, rounds),
                    lambda m, te: m.apply(s_k[te], h_k[te]), len(s_k), f_k, r_k)
            cand["parallax"] = _summary(e, how, n_with_height=int(known.sum()))
            built["parallax"] = fit_parallax_robust(s_k, r_k, h_k, parallax.h0_m, parallax.height_at, rounds)
            built["parallax"].dem = parallax.dem
        else:
            notes.append(f"parallax not scored: {int(known.sum())} fit points have a DEM height")

    min_tps = int(config.get("geometry.min_tps_points", 30))
    if with_tps and n >= min_tps:
        best = None
        for s in grid_s:
            e = _cv(lambda tr: fit_tps_robust(src[tr], ref[tr], s, rounds),
                    lambda m, te: models.apply(m, src[te]), n, folds, ref)
            rms = _rms(e[np.isfinite(e)][:, None])
            if best is None or rms < best[1]:
                best = (float(s), rms, e)
        s_best, _, e = best
        cand["tps"] = _summary(e, how, smoothing=s_best)
        tps = fit_tps_robust(src, ref, s_best, rounds)
        tps.tps_params["cv_rms_px"] = cand["tps"]["checkpoint_rmse_px"]
        # the inverse (reference -> source), fitted on the SAME points and weights, so an
        # export warps with exactly the model that was scored here
        from scipy.interpolate import RBFInterpolator
        w = tps.tps_params["weights"]
        tps.tps_params["inverse"] = TransformModel(
            kind="tps", matrix=None,
            tps_params={"interpolator": RBFInterpolator(ref, src, kernel="thin_plate_spline", smoothing=s_best / w),
                        "n_control": n, "smoothing": s_best})
        built["tps"] = tps
    elif with_tps:
        notes.append(f"TPS not considered: {n} fit points < {min_tps}")

    # G-06 (docs/tmc2_tail_protocol.md): parallax + a TPS on its residuals, same folds, same smoothing grid
    if "parallax" in built and with_tps and bool(config.get("geometry.parallax_tps", True)):
        at = ref if parallax.height_at == "ref" else src
        h = np.asarray(heights_at(at), float)
        kn = np.isfinite(h)
        if kn.sum() >= min_tps:
            s_k, r_k, h_k, f_k = src[kn], ref[kn], h[kn], folds[kn]

            def fit_pt(tr, sm):
                par = fit_parallax_robust(s_k[tr], r_k[tr], h_k[tr], parallax.h0_m, parallax.height_at, rounds)
                res = r_k[tr] - par.apply(s_k[tr], h_k[tr])
                return models.ParallaxTPSModel(par, fit_tps_robust(s_k[tr], res, sm, rounds), parallax.height_at)
            best_pt = None
            for sm in grid_s:
                e = _cv(lambda tr: fit_pt(tr, sm), lambda m, te: m.apply(s_k[te], h_k[te]), len(s_k), f_k, r_k)
                rms = _rms(e[np.isfinite(e)][:, None])
                if best_pt is None or rms < best_pt[1]:
                    best_pt = (float(sm), rms, e)
            cand["parallax_tps"] = _summary(best_pt[2], how, smoothing=best_pt[0], n_with_height=int(kn.sum()))
            built["parallax_tps"] = fit_pt(np.ones(len(s_k), bool), best_pt[0])
            built["parallax_tps"].parallax.dem = parallax.dem

    chosen = "affine"
    for name in SIMPLICITY[1:]:
        if name in cand and cand[name]["checkpoint_rmse_px"] is not None:
            cur = cand[chosen]["checkpoint_rmse_px"]
            if cur is None or cand[name]["checkpoint_rmse_px"] < (1.0 - min_gain) * cur:
                chosen = name
    if "parallax_tps" in cand and cand["parallax_tps"]["checkpoint_rmse_px"] is not None:
        parents = [cand[n]["checkpoint_rmse_px"] for n in ("parallax", "tps") if n in cand
                   and cand[n]["checkpoint_rmse_px"] is not None]
        best_parent = min(parents + [cand[chosen]["checkpoint_rmse_px"]])
        if cand["parallax_tps"]["checkpoint_rmse_px"] < (1.0 - min_gain) * best_parent:
            chosen = "parallax_tps"
    split = _split_half(chosen, src, ref, folds, k, rounds, cand, parallax, heights_at)
    if split is not None:
        cand[chosen]["split_half_px"] = _r(split)
    notes.append(f"{chosen} chosen: " + ", ".join(f"{m} {c['checkpoint_rmse_px']}" for m, c in cand.items())
                 + f" px (a richer model must win by {min_gain:.0%})")
    return Selection(chosen, built[chosen], cand[chosen]["checkpoint_rmse_px"], cand, n, notes,
                     built.get("tps"), built.get("affine"), built.get("parallax"))


def _split_half(name, src, ref, folds, k, rounds, cand, parallax, heights_at) -> Optional[float]:
    """The chosen geometry's own estimation error, WITHOUT the check points' matching noise.

    Fit the model on each half of the fit set (half the folds each; stratified folds keep both
    halves evenly spread) and compare the two fits at every fit point. Each half-fit has about
    twice the full fit's variance, so their RMS disagreement is about 2x the full fit's error.
    Blind to a bias both halves share (e.g. a TPS smoothing away a real ripple): a lower bound
    on the geometry's error where the check-point RMSE is an upper bound."""
    a = folds < (k + 1) // 2
    if a.sum() < 15 or (~a).sum() < 15:
        return None
    if name == "affine":
        fa, fb = fit_affine_robust(src[a], ref[a], rounds), fit_affine_robust(src[~a], ref[~a], rounds)
        d = models.apply(fa, src) - models.apply(fb, src)
    elif name == "tps":
        s = cand["tps"]["smoothing"]
        fa, fb = fit_tps_robust(src[a], ref[a], s, rounds), fit_tps_robust(src[~a], ref[~a], s, rounds)
        d = models.apply(fa, src) - models.apply(fb, src)
    elif name == "parallax" and parallax is not None and heights_at is not None:
        h = np.asarray(heights_at(ref if parallax.height_at == "ref" else src), float)
        kn = np.isfinite(h)
        a2, b2 = a & kn, ~a & kn
        if a2.sum() < 15 or b2.sum() < 15:
            return None
        fa = fit_parallax_robust(src[a2], ref[a2], h[a2], parallax.h0_m, parallax.height_at, rounds)
        fb = fit_parallax_robust(src[b2], ref[b2], h[b2], parallax.h0_m, parallax.height_at, rounds)
        d = fa.apply(src[kn], h[kn]) - fb.apply(src[kn], h[kn])
    else:
        return None
    return _rms(d) / 2.0
