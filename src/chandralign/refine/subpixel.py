"""Sub-pixel shift estimation and per-match refinement (PREC-01..04, P2-T08).

Four independent estimators of the translation between two image patches:

    ncc_peak      PREC-01  normalised cross-correlation, then a 3-point peak fit
                           per axis (parabola by default, Gaussian optional)
    phase         PREC-02  upsampled phase correlation (skimage), Hann-windowed
    corner        PREC-03  cv2.cornerSubPix on corners found in both patches
    ecc           PREC-04  cv2.findTransformECC, translation model, seeded by an
                           integer estimate; brightness/contrast-invariant

SIGN CONVENTION -- one for the whole module, pinned by tests:
    estimate(ref, mov) returns d = (dx, dy) such that a feature at position p
    in `ref` appears at p + d in `mov`, i.e. mov(p) ~ ref(p - d).
Every estimator is converted to this convention inside its own function, so a
caller never has to remember which library returns which sign.

WHY FOUR, AND WHY THEY ARE COMPARED RATHER THAN RANKED BY ASSUMPTION
Each has a known failure shape: a parabola fitted to an NCC peak is biased
towards whole pixels ("pixel locking"); phase correlation leaks at patch edges
and assumes a pure translation; cornerSubPix needs corners and refines each
image independently; ECC needs a good starting point and can diverge. Which
one is best on lunar imagery is an empirical question, answered by
scripts/verify_subpixel.py on real OHRC data (PREC-06), not by this docstring.

Every estimator returns ok=False instead of raising when it cannot produce an
answer, so a failure is recorded rather than turned into a wrong number (H1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .. import config


@dataclass
class ShiftEstimate:
    dx: float
    dy: float
    method: str
    ok: bool = True
    quality: Optional[float] = None       # method-specific: NCC peak, phase error, ...
    notes: list[str] = field(default_factory=list)

    @property
    def d(self) -> np.ndarray:
        return np.array([self.dx, self.dy], float)


def _failed(method: str, why: str) -> ShiftEstimate:
    return ShiftEstimate(float("nan"), float("nan"), method, ok=False, notes=[why])


def _as_float(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D patch, got shape {a.shape}")
    return a


# ---------------------------------------------------------------------------
# PREC-01: NCC with a sub-pixel peak fit
# ---------------------------------------------------------------------------
def _peak_offset(cm1: float, c0: float, cp1: float, fit: str) -> float:
    """Sub-pixel offset of a peak from three samples at -1, 0, +1.

    parabola  the textbook fit; biased towards 0 (pixel locking) on sharp peaks
    gaussian  the same fit on log values; less biased when the peak is
              Gaussian-like, undefined for non-positive samples (falls back)
    """
    if fit == "gaussian" and min(cm1, c0, cp1) > 0:
        cm1, c0, cp1 = np.log(cm1), np.log(c0), np.log(cp1)
    denom = cm1 - 2.0 * c0 + cp1
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (cm1 - cp1) / denom, -0.5, 0.5))


def ncc_peak(ref: np.ndarray, mov: np.ndarray, max_shift: int = 4,
             fit: str = "parabola") -> ShiftEstimate:
    """PREC-01. Integer NCC peak over +-max_shift, then a per-axis 3-point fit.

    The central (H - 2m) x (W - 2m) region of `mov` is slid over `ref`; the best
    placement gives the integer shift and its neighbours give the fraction.
    """
    method = f"ncc_{fit}"
    ref, mov = _as_float(ref), _as_float(mov)
    m = int(max_shift)
    if min(ref.shape) <= 2 * m + 8 or ref.shape != mov.shape:
        return _failed(method, "patches too small for the search range, or unequal")
    tpl = mov[m:-m, m:-m]
    if float(tpl.std()) < 1e-6 or float(ref.std()) < 1e-6:
        return _failed(method, "no texture: a flat patch has no correlation peak")
    cmap = cv2.matchTemplate(ref, tpl, cv2.TM_CCOEFF_NORMED)      # (2m+1, 2m+1)
    v, u = np.unravel_index(int(np.argmax(cmap)), cmap.shape)
    notes = []
    if u in (0, cmap.shape[1] - 1) or v in (0, cmap.shape[0] - 1):
        notes.append("peak on the edge of the search range: shift may exceed max_shift")
        fx = fy = 0.0
    else:
        fx = _peak_offset(cmap[v, u - 1], cmap[v, u], cmap[v, u + 1], fit)
        fy = _peak_offset(cmap[v - 1, u], cmap[v, u], cmap[v + 1, u], fit)
    # tpl at ref offset (u*, v*) means mov(m + p) = ref(u* + p): content at ref q
    # sits in mov at q + (m - u*), so d = m - u*.
    return ShiftEstimate(m - (u + fx), m - (v + fy), method,
                         quality=float(cmap[v, u]), notes=notes)


# ---------------------------------------------------------------------------
# PREC-02: upsampled phase correlation
# ---------------------------------------------------------------------------
def phase(ref: np.ndarray, mov: np.ndarray, upsample: Optional[int] = None,
          window: bool = True) -> ShiftEstimate:
    """PREC-02. skimage phase_cross_correlation with upsampling.

    A Hann window suppresses the edge discontinuity that the implicit periodic
    extension of the FFT otherwise turns into a spurious zero-shift peak.
    skimage returns the shift that registers `mov` ONTO `ref`, which is -d.
    """
    from skimage.registration import phase_cross_correlation

    ref, mov = _as_float(ref), _as_float(mov)
    if ref.shape != mov.shape:
        return _failed("phase", "unequal patch shapes")
    if float(ref.std()) < 1e-6 or float(mov.std()) < 1e-6:
        return _failed("phase", "no texture")
    if upsample is None:
        upsample = int(config.get("subpixel.phase_upsample", 100))
    if window:
        w = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1])).astype(np.float32)
        ref = (ref - ref.mean()) * w
        mov = (mov - mov.mean()) * w
    shift, error, _ = phase_cross_correlation(ref, mov, upsample_factor=int(upsample))
    return ShiftEstimate(-float(shift[1]), -float(shift[0]), "phase", quality=float(error))


# ---------------------------------------------------------------------------
# PREC-03: cornerSubPix
# ---------------------------------------------------------------------------
def corner(ref: np.ndarray, mov: np.ndarray, max_corners: int = 150,
           win: int = 5, seed: Optional[ShiftEstimate] = None) -> ShiftEstimate:
    """PREC-03. Refine the same corners independently in both patches.

    Corners are found in `ref`, refined there, carried to `mov` by an integer
    seed, refined again, and the shift is the median displacement. Median, not
    mean: a corner that snaps to a different feature in `mov` is an outlier.
    """
    ref, mov = _as_float(ref), _as_float(mov)
    if seed is None:
        seed = ncc_peak(ref, mov)
    if not seed.ok:
        return _failed("corner", "no integer seed")
    to8 = lambda a: cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    margin = win + 4 + int(np.ceil(max(abs(seed.dx), abs(seed.dy))))
    mask = np.zeros(ref.shape, np.uint8)
    mask[margin:-margin, margin:-margin] = 255
    pts = cv2.goodFeaturesToTrack(to8(ref), max_corners, 0.01, 5, mask=mask)
    if pts is None or len(pts) < 5:
        return _failed("corner", "too few corners (smooth terrain has few)")
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    start = pts.reshape(-1, 2).copy()
    p_ref = cv2.cornerSubPix(ref, pts.copy(), (win, win), (-1, -1), crit).reshape(-1, 2)
    seed_int = np.round([seed.dx, seed.dy]).astype(np.float32)
    start_mov = p_ref + seed_int
    p_mov = cv2.cornerSubPix(mov, start_mov.reshape(-1, 1, 2).copy(), (win, win),
                             (-1, -1), crit).reshape(-1, 2)

    # cornerSubPix silently returns a point UNCHANGED when it cannot refine it
    # (measured: ~60% of corners on a block-averaged OHRC crop). Keeping those
    # made the median displacement equal the integer seed -- every answer came
    # out as a whole number. Use only corners refined in BOTH images, and drop
    # any that snapped to a different feature (displacement far from the seed).
    refined = (np.hypot(*(p_ref - start).T) > 1e-4) & (np.hypot(*(p_mov - start_mov).T) > 1e-4)
    disp = p_mov - p_ref
    sane = np.hypot(*(disp - seed_int).T) <= 1.5
    use = refined & sane
    if use.sum() < 5:
        return _failed("corner", f"only {int(use.sum())} corners refined in both patches")
    disp = disp[use]
    med = np.median(disp, axis=0)
    spread = float(np.median(np.abs(disp - med)))
    return ShiftEstimate(float(med[0]), float(med[1]), "corner", quality=spread,
                         notes=[f"{int(use.sum())}/{len(start)} corners refined in both patches, "
                                f"median absolute deviation {spread:.3f} px"])


# ---------------------------------------------------------------------------
# PREC-04: ECC
# ---------------------------------------------------------------------------
def ecc(ref: np.ndarray, mov: np.ndarray, seed: Optional[ShiftEstimate] = None,
        iterations: Optional[int] = None, eps: Optional[float] = None) -> ShiftEstimate:
    """PREC-04. Enhanced Correlation Coefficient, translation model.

    ECC is invariant to brightness and contrast change, which is why it is in the
    list for lunar pairs. It is a local optimiser: seeded by an integer NCC
    estimate, and reported as failed (not as zero) when it does not converge.
    findTransformECC finds W with mov(W x) ~ ref(x); for a translation W x = x + t,
    so t is already d in this module's convention.
    """
    ref, mov = _as_float(ref), _as_float(mov)
    if seed is None:
        seed = ncc_peak(ref, mov)
    if not seed.ok:
        return _failed("ecc", "no seed")
    iterations = int(iterations or config.get("subpixel.ecc_iterations", 50))
    eps = float(eps or config.get("subpixel.ecc_eps", 1e-6))
    warp = np.array([[1, 0, seed.dx], [0, 1, seed.dy]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        cc, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
    except cv2.error as exc:
        return _failed("ecc", f"did not converge: {str(exc).splitlines()[-1][:80]}")
    return ShiftEstimate(float(warp[0, 2]), float(warp[1, 2]), "ecc", quality=float(cc))


# ---------------------------------------------------------------------------
# Iterative refinement: remove the fit bias by re-estimating near zero
# ---------------------------------------------------------------------------
def iterate(ref: np.ndarray, mov: np.ndarray, base: Callable[..., ShiftEstimate],
            iterations: int = 3, name: Optional[str] = None) -> ShiftEstimate:
    """Estimate, shift `mov` back by the estimate, re-estimate the residual, repeat.

    Why it helps: a 3-point peak fit (and phase correlation's upsampled peak) is
    biased except near zero offset -- the "pixel locking" measured in PREC-01's
    tests. Each pass leaves a smaller residual, where that bias is smaller, so the
    accumulated estimate converges on the unbiased answer.

    The resampling uses Lanczos-4. That is interpolation INSIDE the method, which
    is legitimate; the truth it is scored against is still the exact
    block-averaged shift, so no interpolation model is shared with the test.
    Border pixels the warp has to invent are cropped before each re-estimate.
    """
    name = name or f"{base.__name__}_iter"
    ref, mov = _as_float(ref), _as_float(mov)
    first = base(ref, mov)
    if not first.ok:
        first.method = name
        return first
    d = first.d.copy()
    h, w = ref.shape
    for _ in range(iterations):
        m = np.array([[1.0, 0.0, -d[0]], [0.0, 1.0, -d[1]]], np.float32)
        back = cv2.warpAffine(mov, m, (w, h), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT)      # back(y) = mov(y + d) ~ ref(y)
        c = int(np.ceil(np.abs(d).max())) + 3
        if min(h, w) - 2 * c < 24:
            break
        r = base(ref[c:-c, c:-c], back[c:-c, c:-c])
        if not r.ok:
            break
        d = d + r.d
        if np.hypot(*r.d) < 1e-3:
            break
    return ShiftEstimate(float(d[0]), float(d[1]), name, quality=first.quality,
                         notes=first.notes + [f"iterated {iterations}x with Lanczos-4 resampling"])


# ---------------------------------------------------------------------------
# Dispatch and per-match refinement (the pipeline's use of PREC-01..04)
# ---------------------------------------------------------------------------
METHODS: dict[str, Callable[..., ShiftEstimate]] = {
    "ncc_parabola": lambda r, m: ncc_peak(r, m, fit="parabola"),
    "ncc_gaussian": lambda r, m: ncc_peak(r, m, fit="gaussian"),
    "ncc_gaussian_iter": lambda r, m: iterate(r, m, lambda a, b: ncc_peak(a, b, max_shift=3, fit="gaussian"),
                                              name="ncc_gaussian_iter"),
    "phase": phase,
    "phase_iter": lambda r, m: iterate(r, m, phase, name="phase_iter"),
    "corner": corner,
    "ecc": ecc,
}


def estimate(ref: np.ndarray, mov: np.ndarray, method: str = "phase") -> ShiftEstimate:
    if method not in METHODS:
        raise ValueError(f"unknown sub-pixel method {method!r}; choose from {sorted(METHODS)}")
    try:
        return METHODS[method](ref, mov)
    except Exception as exc:                      # a failure is a result, not a crash
        return _failed(method, f"{type(exc).__name__}: {exc}")


def lsm(ref: np.ndarray, mov: np.ndarray, motion: str = "affine", gauss: Optional[int] = None,
        iterations: int = 100, eps: float = 1e-6) -> ShiftEstimate:
    """G-01. Least-squares matching: ECC with an AFFINE (default) warp between two patches.

    ECC maximises the zero-mean normalised correlation, so it is invariant to a
    radiometric gain and offset; the affine warp absorbs what is left of the local
    geometry after the patches were brought into one frame. The shift reported is
    where the patch CENTRE of `ref` lands in `mov` (the module's convention),
    W c - c. Seeded at identity: the caller has already warped the patches, so the
    residual is small. Failure to converge is ok=False, never a zero shift.
    """
    ref, mov = _as_float(ref), _as_float(mov)
    if ref.shape != mov.shape:
        return _failed("lsm", "unequal patch shapes")
    if float(ref.std()) < 1e-6 or float(mov.std()) < 1e-6:
        return _failed("lsm", "no texture")
    gauss = int(gauss if gauss is not None else config.get("subpixel.lsm_gauss", 1))
    mode = {"affine": cv2.MOTION_AFFINE, "translation": cv2.MOTION_TRANSLATION}[motion]
    # ECC is a local optimiser: from more than ~1 px it can diverge (measured: a 1.25 px
    # truth went to 17 px). Seed with the integer NCC peak, and refuse a result that ran
    # away from its seed rather than report it as a shift.
    seed = ncc_peak(ref, mov, max_shift=3)
    s0 = np.round(seed.d) if seed.ok and np.all(np.isfinite(seed.d)) else np.zeros(2)
    w = np.float32([[1, 0, s0[0]], [0, 1, s0[1]]])
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        cc, w = cv2.findTransformECC(ref, mov, w, mode, crit, None, gauss)
    except cv2.error as exc:
        return _failed("lsm", f"did not converge: {str(exc).splitlines()[-1][:80]}")
    c = np.array([(ref.shape[1] - 1) / 2.0, (ref.shape[0] - 1) / 2.0])
    d = w[:, :2].astype(float) @ c + w[:, 2] - c
    if not np.all(np.isfinite(d)) or np.hypot(*(d - s0)) > 1.5:
        return _failed("lsm", f"diverged: {np.round(d, 2).tolist()} from seed {s0.tolist()}")
    return ShiftEstimate(float(d[0]), float(d[1]), "lsm", quality=float(cc))


METHODS["lsm"] = lsm


def _ncc(a: np.ndarray, b: np.ndarray, border: int) -> float:
    """Zero-mean normalised correlation of the central parts of two equal patches."""
    if border:
        a, b = a[border:-border, border:-border], b[border:-border, border:-border]
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else -1.0


def _mind_shift(sp: np.ndarray, r32: np.ndarray, r, size: int, half: int, m: int = 3):
    """Shift of the (warped) source patch in the reference, on channel-averaged MIND NCC.

    MIND keeps only the RELATIVE pattern of neighbour similarity, so it survives a lighting change that
    inverts intensity. Returns (d, score at the matcher's position, score at the peak) or (None, ., .)."""
    from ..preprocess.phase_congruency import mind
    region = cv2.getRectSubPix(r32, (size + 2 * m, size + 2 * m), (float(r[0]), float(r[1])))
    nz = lambda a: (a - a.min()) / max(float(a.max() - a.min()), 1e-9)          # noqa: E731
    ms, mr = mind(nz(sp)).astype(np.float32), mind(nz(region)).astype(np.float32)
    if float(ms.std()) < 1e-6 or float(mr.std()) < 1e-6:
        return None, -1.0, -1.0
    cmap = np.mean([cv2.matchTemplate(mr[..., c], ms[..., c], cv2.TM_CCOEFF_NORMED)
                    for c in range(ms.shape[-1])], axis=0)                     # (2m+1, 2m+1)
    v, u = np.unravel_index(int(np.argmax(cmap)), cmap.shape)
    if u in (0, cmap.shape[1] - 1) or v in (0, cmap.shape[0] - 1):
        return None, float(cmap[m, m]), float(cmap[v, u])
    fx = _peak_offset(cmap[v, u - 1], cmap[v, u], cmap[v, u + 1], "gaussian")
    fy = _peak_offset(cmap[v - 1, u], cmap[v, u], cmap[v + 1, u], "gaussian")
    # the template's centre sits at region (u + half, v + half), i.e. reference r + (u - m, v - m)
    return np.array([u + fx - m, v + fy - m], float), float(cmap[m, m]), float(cmap[v, u])


def _antialias(src: np.ndarray, matrix: Optional[np.ndarray], at: tuple[float, float]) -> np.ndarray:
    """Low-pass the source before it is resampled more coarsely than its own pixels.

    Sampling the source at the reference's spacing is a decimation by 1/scale when the
    reference is coarser (scale < 1); without a pre-filter the patch aliases and the
    residual estimate inherits it. Gaussian sigma 0.5*sqrt(f^2 - 1), f = 1/scale.
    """
    from ..estimate import models
    if matrix is None:
        return src
    k = models.estimated_scale(matrix, at=at)
    if not np.isfinite(k) or k >= 0.95:
        return src
    sigma = 0.5 * float(np.sqrt(max((1.0 / k) ** 2 - 1.0, 0.0)))
    return cv2.GaussianBlur(src, (0, 0), sigma) if sigma > 0.2 else src


def refine_points(src_img: np.ndarray, ref_img: np.ndarray, src_pts: np.ndarray,
                  ref_pts: np.ndarray, method: Optional[str] = None, half: Optional[int] = None,
                  max_move: Optional[float] = None, model=None,
                  return_info: bool = False):
    """Refine each match's position in `ref_img` to sub-pixel precision.

    THE GEOMETRY (audit 2026-09-26, C-02). Two images that differ in rotation or scale
    do not show the same ground in the same shape inside two axis-aligned patches, so a
    translation estimated between such patches is biased -- measured on a known warp,
    the old axis-aligned refinement was WORSE than no refinement in 11 of 15 cases, with
    p95 up to 1.39 px. So, given `model` (the source -> reference TransformModel), the
    source patch is resampled through the model's local Jacobian at the point, into the
    reference patch's geometry, before the residual is estimated. Without a model the
    patches are compared axis-aligned (correct only when the images already share a frame).

    THE ESTIMATE. `method` (default `subpixel.method`) is tried first and each name in
    `subpixel.fallbacks` after it; each candidate move is scored by the NCC of the
    warped source patch against the reference patch AT THE MOVED POSITION.

    THE ACCEPTANCE RULE. A move is applied only if it RAISES that score above the score
    at the matcher's position, and is no larger than `max_move` (`subpixel.max_move_px`
    for warped patches, 1.5 px for axis-aligned ones): a sub-pixel step should correct
    a matcher's rounding, never relocate a match, and never make the fit worse.

    Returns (refined_ref_pts, moved_mask), plus a per-point info dict with
    `return_info=True` (`score_before`, `score_after`, `method` used per point).
    """
    from ..estimate import models

    method = method or str(config.get("subpixel.method", "lsm"))
    fallbacks = [m for m in (config.get("subpixel.fallbacks", ["ncc_gaussian_iter"]) or []) if m != method]
    half = int(half if half is not None else config.get("subpixel.refine_half_px", 16))
    matrix = None
    if model is not None and getattr(model, "matrix", None) is not None:
        matrix = np.asarray(model.matrix, float)
    if max_move is None:
        max_move = float(config.get("subpixel.max_move_px", 0.75)) if matrix is not None else 1.5
    border = int(config.get("subpixel.score_border_px", 3))
    need_gain = bool(config.get("subpixel.require_score_gain", True))
    use_mind = str(config.get("subpixel.representation", "intensity")) == "auto"
    src_pts = np.asarray(src_pts, float).reshape(-1, 2)
    out = np.asarray(ref_pts, float).reshape(-1, 2).copy()
    moved = np.zeros(len(out), bool)
    size = 2 * half + 1
    c = np.array([half, half], float)
    s32 = np.asarray(src_img, np.float32)
    r32 = np.asarray(ref_img, np.float32)
    h, w = s32.shape[:2]
    s32 = _antialias(s32, matrix, (w / 2.0, h / 2.0))
    lanczos = str(config.get("subpixel.score_interp", "lanczos")) == "lanczos"

    def grab(q):
        """The reference patch centred at q, for SCORING. Lanczos, not getRectSubPix's bilinear:
        bilinear smooths by an amount that depends on the fractional offset, which biases a
        comparison of two positions at exactly the 0.1 px scale being judged."""
        if not lanczos:
            return cv2.getRectSubPix(r32, (size, size), (float(q[0]), float(q[1])))
        m = np.float32([[1, 0, q[0] - half], [0, 1, q[1] - half]])
        return cv2.warpAffine(r32, m, (size, size), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)

    before_s = np.full(len(out), np.nan)
    after_s = np.full(len(out), np.nan)
    used = np.array([""] * len(out), dtype=object)
    for i, (s, r) in enumerate(zip(src_pts, out.copy())):
        J = None if matrix is None else models.local_jacobian(matrix, (float(s[0]), float(s[1])))
        if J is not None and abs(np.linalg.det(J)) > 1e-9:
            Ji = np.linalg.inv(J)
            # patch pixel u <- source s + J^-1 (u - c): the source seen in the reference's geometry
            sp = cv2.warpAffine(s32, np.c_[Ji, s - Ji @ c].astype(np.float32), (size, size),
                                flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        else:
            sp = cv2.getRectSubPix(s32, (size, size), (float(s[0]), float(s[1])))
        rp = cv2.getRectSubPix(r32, (size, size), (float(r[0]), float(r[1])))
        base = _ncc(sp, grab(r), border)
        before_s[i] = base
        best = None
        for m in [method] + fallbacks:
            est = estimate(sp, rp, m)
            if not (est.ok and np.all(np.isfinite(est.d))) or np.hypot(est.dx, est.dy) > max_move:
                continue
            # rp is the ref neighbourhood placed where we THINK the match is; a feature at the
            # patch centre of sp appears at centre + d in rp, so the match is d away.
            q = r + est.d
            score = _ncc(sp, grab(q), border)
            if (score > base or not need_gain) and (best is None or score > best[1]):
                best = (q, score, m)
        if best is None and use_mind:
            # G-05 (docs/illumination_refinement_protocol.md): intensity refused every move; try MIND channels
            d, m_base, m_peak = _mind_shift(sp, r32, r, size, half)
            if d is not None and np.hypot(*d) <= max_move and (m_peak > m_base or not need_gain):
                best = (r + d, m_peak, "mind")
        if best is not None:
            out[i], after_s[i], used[i] = best[0], best[1], best[2]
            moved[i] = True
    if return_info:
        return out, moved, {"score_before": before_s, "score_after": after_s, "method": used}
    return out, moved
