"""Mutual-information alignment check (MATCH-07, docs/mi_protocol.md).

MI is the classical cross-modal similarity: it asks how well one image's intensities PREDICT the
other's, not whether they are equal, so it survives a different sensor or band. It has no scale or
rotation invariance, so it is used here only AFTER registration, as a matcher-free check:

  * `nmi` -- normalised MI (Studholme) of two aligned images: a quality score;
  * `alignment_check` -- NMI searched over small shifts around the delivered model; the sub-pixel
    peak says where the images really line up, so a model that is consistently off (which the
    control gates, measuring only the lock's repeatability, cannot see) is flagged.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config


def nmi(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, bins: int = 64) -> float:
    """(H(A) + H(B)) / H(A, B): 1 for independent images, 2 for identical ones."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if mask is not None:
        x, y = x[mask], y[mask]
    else:
        x, y = x.ravel(), y.ravel()
    if x.size < 100:
        return float("nan")
    h, _, _ = np.histogram2d(x, y, bins=bins)
    p = h / h.sum()

    def ent(q):
        q = q[q > 0]
        return float(-np.sum(q * np.log(q)))
    return (ent(p.sum(1)) + ent(p.sum(0))) / ent(p.ravel())


def _nmi_at(src, ref, M, src_ok, ref_ok, border, bins):
    H, W = ref.shape
    a = cv2.warpAffine(np.asarray(src, np.float32), M[:2], (W, H), flags=cv2.INTER_LINEAR)
    ok = cv2.warpAffine(np.asarray(src_ok, np.float32), M[:2], (W, H), flags=cv2.INTER_NEAREST) > 0.5
    ok &= ref_ok
    ok[:border, :] = ok[-border:, :] = False
    ok[:, :border] = ok[:, -border:] = False
    return nmi(a, ref, ok, bins)


def _peak(src, ref, M0, src_ok, ref_ok, r, border, bins):
    """NMI over the +-r px shift grid around M0; sub-pixel peak offset, the grid, and edge flag."""
    grid = np.full((2 * r + 1, 2 * r + 1), np.nan)
    for iy, dy in enumerate(range(-r, r + 1)):
        for ix, dx in enumerate(range(-r, r + 1)):
            M = M0.copy(); M[0, 2] += dx; M[1, 2] += dy
            grid[iy, ix] = _nmi_at(src, ref, M, src_ok, ref_ok, border, bins)
    if not np.isfinite(grid).any():
        return None, grid, False
    iy, ix = np.unravel_index(int(np.nanargmax(grid)), grid.shape)

    def vertex(fm, f0, fp):                       # 1-D parabola through three samples
        d = fm - 2 * f0 + fp
        return 0.0 if not np.isfinite(d) or d >= 0 else float(np.clip(0.5 * (fm - fp) / d, -0.5, 0.5))
    ox = vertex(grid[iy, ix - 1], grid[iy, ix], grid[iy, ix + 1]) if 0 < ix < 2 * r else 0.0
    oy = vertex(grid[iy - 1, ix], grid[iy, ix], grid[iy + 1, ix]) if 0 < iy < 2 * r else 0.0
    return np.array([ix - r + ox, iy - r + oy]), grid, bool(ix in (0, 2 * r) or iy in (0, 2 * r))


def alignment_check(src: np.ndarray, ref: np.ndarray, model: np.ndarray, *,
                    src_ok: np.ndarray | None = None, ref_ok: np.ndarray | None = None,
                    search_px: int | None = None, flag_px: float | None = None, bins: int = 64,
                    abstain: bool | None = None) -> dict:
    """NMI of the source warped by `model` (3x3, source px -> reference px), and where NMI peaks.

    Returns {"nmi", "peak_offset_px": [dx, dy], "peak_nmi", "flag", ...}. peak_offset is the shift, in
    reference px, that must be ADDED to the model for the best NMI; flag = |peak_offset| >= flag_px.
    With `abstain` (docs/mi_abstain_protocol.md, rule R), the peak is also found on each half of the
    frame; if the halves disagree by more than flag_px, MI has no reliable answer: abstain=True and
    flag=False ("inconclusive").
    """
    r = int(search_px if search_px is not None else config.get("similarity.search_px", 3))
    fpx = float(flag_px if flag_px is not None else config.get("similarity.flag_px", 1.0))
    use_abstain = bool(config.get("similarity.abstain", False)) if abstain is None else bool(abstain)
    src_ok = np.ones(np.shape(src), bool) if src_ok is None else np.asarray(src_ok, bool)
    ref_ok = np.ones(np.shape(ref), bool) if ref_ok is None else np.asarray(ref_ok, bool)
    M0 = np.asarray(model, float)
    border = r + 4
    off, grid, edge = _peak(src, ref, M0, src_ok, ref_ok, r, border, bins)
    if off is None:
        return {"nmi": None, "peak_offset_px": None, "peak_nmi": None, "flag": None}
    out = {"nmi": round(float(grid[r, r]), 5), "peak_offset_px": [round(float(v), 3) for v in off],
           "peak_nmi": round(float(np.nanmax(grid)), 5), "flag": bool(np.hypot(*off) >= fpx),
           "at_search_edge": edge}
    if use_abstain:
        H, W = np.shape(ref)
        halves = []
        for k in range(2):                        # split the REFERENCE frame across its longer axis
            m = np.zeros((H, W), bool)
            if W >= H:
                m[:, k * W // 2:(k + 1) * W // 2] = True
            else:
                m[k * H // 2:(k + 1) * H // 2, :] = True
            o, _, _ = _peak(src, ref, M0, src_ok, ref_ok & m, r, border, bins)
            halves.append(o)
        if any(h is None for h in halves):
            out.update(abstain=True, flag=False, half_disagreement_px=None)
        else:
            dis = float(np.hypot(*(halves[0] - halves[1])))
            out["half_peaks_px"] = [[round(float(v), 3) for v in h] for h in halves]
            out["half_disagreement_px"] = round(dis, 3)
            out["abstain"] = bool(dis > fpx)
            if out["abstain"]:
                out["flag"] = False
    return out
