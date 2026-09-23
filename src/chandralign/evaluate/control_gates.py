"""Control gates: prove the pipeline would notice if it were wrong (CHECK-01..04, 06, 08).

A registration result says what the pipeline found. A control gate says whether
the pipeline is CAPABLE of noticing that it found nothing. The two are
different claims, and only the second makes the first worth believing:

    CHECK-01  blank       a flat grey source against the real reference
    CHECK-02  noise       random noise with the reference's own mean and spread
    CHECK-03  perturb     the source moved by a known (3, 4) px -- |s| = 5 px --
                          the recovered transform must move by 5, not 0, not 50
    CHECK-04  identity    the reference against itself: RMSE ~0, inliers ~all
    CHECK-06  shared-mask source and reference masks must be separate arrays

01 and 02 are NULL tests: there is nothing to find, so an accepted registration
means the pipeline invents structure. 03 is a SENSITIVITY test: a pipeline that
returns the same answer whatever it is shown passes 01, 02 and 04 and fails
this. 04 is the floor: if the trivial case is not near-perfect, nothing is.

Every gate RUNS THE REAL PIPELINE, passed in as a function, on the real
reference -- not a stand-in -- so what is tested is what ran. A gate that
cannot be evaluated (e.g. the baseline registration for 03 failed) is recorded
as NOT passed with its reason: an unverified gate is not a passed gate (H1, H4).

CHECK-08's guard lives here too: require_gates() refuses a result that carries
no gate results at all. Part 3 must call it before rendering anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .. import config


@dataclass
class PipelineRun:
    """What one call of the pipeline produced, in the form the gates need."""

    ok: bool                              # did the estimator accept a transform?
    matches: int
    inliers: int
    matrix: Optional[np.ndarray] = None   # 3x3, source px -> reference px
    note: str = ""
    rmse_px: Optional[float] = None       # inlier fit error, reference px (None = not measured)

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


# The pipeline under test: (source image, reference image) -> PipelineRun.
Pipeline = Callable[[np.ndarray, np.ndarray], PipelineRun]


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass
class GateReport:
    results: list[GateResult]

    @property
    def gates(self) -> dict[str, bool]:
        """The dict RegistrationResult.gates and quality.assess(gates=...) take."""
        return {r.name: r.passed for r in self.results}

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def failed(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {r.name: {"passed": r.passed, "reason": r.reason, **r.detail} for r in self.results}


class UngatedResultError(RuntimeError):
    """A result with no control-gate results reached something that would show it."""


def _cfg(key: str, default):
    return config.get(f"gates.{key}", default)


def _safe(pipeline: Pipeline, src: np.ndarray, ref: np.ndarray) -> PipelineRun:
    """A pipeline that raises on a gate input has, for that gate, produced nothing."""
    try:
        return pipeline(src, ref)
    except Exception as exc:                             # noqa: BLE001 -- recorded, not hidden
        return PipelineRun(False, 0, 0, None, f"pipeline raised {type(exc).__name__}: {exc}")


def _null_verdict(name: str, run: PipelineRun, what: str) -> GateResult:
    limit = float(_cfg("null_max_inlier_ratio", 0.05))
    detail = {"matches": run.matches, "inliers": run.inliers,
              "inlier_ratio": round(run.inlier_ratio, 4), "estimator_accepted": run.ok}
    if not run.ok:
        return GateResult(name, True, f"{what}: no transform accepted ({run.matches} matches)", detail)
    if run.inlier_ratio <= limit:
        return GateResult(name, True, f"{what}: accepted with {run.inlier_ratio:.1%} inliers, "
                                      f"at or under the {limit:.0%} null limit", detail)
    return GateResult(name, False,
                      f"{what}: the pipeline ACCEPTED a registration with {run.inliers} inliers "
                      f"({run.inlier_ratio:.0%}) where there is nothing to find -- it invents "
                      f"structure", detail)


# ---------------------------------------------------------------------------
# CHECK-01 / CHECK-02: null tests
# ---------------------------------------------------------------------------
def blank_gate(pipeline: Pipeline, src: np.ndarray, ref: np.ndarray) -> GateResult:
    grey = np.full(np.asarray(src).shape, float(np.mean(ref)), np.float32)
    return _null_verdict("null_constant_grey", _safe(pipeline, grey, ref), "flat grey source")


def noise_gate(pipeline: Pipeline, src: np.ndarray, ref: np.ndarray, seed: int = 0) -> GateResult:
    """White noise with the REFERENCE's mean and spread, so brightness statistics
    cannot give it away -- only the absence of structure can."""
    ref = np.asarray(ref, np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(float(ref.mean()), float(ref.std()) or 1.0, np.asarray(src).shape)
    noise = np.clip(noise, float(ref.min()), float(ref.max())).astype(np.float32)
    return _null_verdict("null_random_noise", _safe(pipeline, noise, ref), "random-noise source")


# ---------------------------------------------------------------------------
# CHECK-03: perturbation sensitivity
# ---------------------------------------------------------------------------
def _shift_content(img: np.ndarray, sx: int, sy: int) -> np.ndarray:
    """Exact integer move: content at p ends up at p + (sx, sy). Reflect-padded, not wrapped."""
    img = np.asarray(img, np.float32)
    h, w = img.shape
    padded = np.pad(img, ((max(sy, 0), max(-sy, 0)), (max(sx, 0), max(-sx, 0))), mode="reflect")
    # padded(q) = img(q - pad_top_left); cropping at (max(-s, 0)) gives out(p) = img(p - s)
    # for either sign of s.
    y0, x0 = max(-sy, 0), max(-sx, 0)
    return padded[y0:y0 + h, x0:x0 + w]


def perturbation_gate(pipeline: Pipeline, src: np.ndarray, ref: np.ndarray,
                      shift: tuple[int, int] = (3, 4),
                      base: Optional[PipelineRun] = None) -> GateResult:
    """Move the source by a known shift; the transform must move by the same amount.

    If the source content moves by s, a point p in the moved source is the old
    source's p - s, so T1(p) = T0(p - s) = T0(p) - J0 s. The implied shift is
    solved from J0 s = T0(c) - T1(c) at the centre, which keeps the answer in
    SOURCE pixels even when source and reference differ in scale (TMC-2 -> TC).

    `base` may be the registration being certified, when it came from this same
    pipeline on this same (src, ref). The gate then compares the moved run with
    THAT result rather than a fresh repeat of it, which is the stronger test of
    the two: it certifies the transform actually reported. Only the one
    redundant call is skipped.
    """
    name = "perturbation_sensitivity"
    tol = float(_cfg("perturbation_tolerance_px", 1.5))
    s = np.array(shift, float)
    if base is None:
        base = _safe(pipeline, src, ref)
    if not base.ok or base.matrix is None:
        return GateResult(name, False, f"cannot evaluate: the baseline registration failed "
                                       f"({base.note or 'no transform'})", {"shift": list(shift)})
    moved = _safe(pipeline, _shift_content(src, int(s[0]), int(s[1])), ref)
    if not moved.ok or moved.matrix is None:
        return GateResult(name, False, "the pipeline lost the pair after a 5 px shift",
                          {"shift": list(shift), "note": moved.note})
    h, w = np.asarray(src).shape
    c = np.array([w / 2.0, h / 2.0, 1.0])
    T0, T1 = np.asarray(base.matrix, float), np.asarray(moved.matrix, float)
    p0, p1 = (T0 @ c)[:2] / (T0 @ c)[2], (T1 @ c)[:2] / (T1 @ c)[2]
    J0 = T0[:2, :2] - np.outer(p0, T0[2, :2])
    J0 = J0 / (T0 @ c)[2]
    try:
        s_est = np.linalg.solve(J0, p0 - p1)
    except np.linalg.LinAlgError:
        return GateResult(name, False, "baseline transform is singular", {"shift": list(shift)})
    err = float(np.hypot(*(s_est - s)))
    detail = {"shift": list(shift), "recovered": [round(float(v), 3) for v in s_est],
              "error_px": round(err, 3), "tolerance_px": tol}
    if err <= tol:
        return GateResult(name, True, f"recovered a {np.hypot(*s):.0f} px move as "
                                      f"({s_est[0]:.2f}, {s_est[1]:.2f}), {err:.2f} px off", detail)
    return GateResult(name, False, f"the transform moved by ({s_est[0]:.2f}, {s_est[1]:.2f}) px "
                                   f"for a known ({s[0]:.0f}, {s[1]:.0f}) move -- {err:.2f} px off, "
                                   f"past {tol} px", detail)


# ---------------------------------------------------------------------------
# CHECK-04: identity
# ---------------------------------------------------------------------------
def identity_gate(pipeline: Pipeline, ref: np.ndarray) -> GateResult:
    name = "identity"
    max_rmse = float(_cfg("identity_max_rmse_px", 0.25))
    min_ratio = float(_cfg("identity_min_inlier_ratio", 0.9))
    run = _safe(pipeline, ref, ref)
    if not run.ok or run.matrix is None:
        return GateResult(name, False, f"an image did not register against ITSELF ({run.note})",
                          {"matches": run.matches})
    h, w = np.asarray(ref).shape
    ys, xs = np.mgrid[0:h:complex(16), 0:w:complex(16)]
    p = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    q = np.asarray(run.matrix, float) @ p
    rmse = float(np.sqrt(np.mean(np.sum((q[:2] / q[2] - p[:2]) ** 2, axis=0))))
    detail = {"rmse_px": round(rmse, 4), "inlier_ratio": round(run.inlier_ratio, 4),
              "max_rmse_px": max_rmse, "min_inlier_ratio": min_ratio}
    problems = []
    if rmse > max_rmse:
        problems.append(f"RMSE {rmse:.3f} px > {max_rmse}")
    if run.inlier_ratio < min_ratio:
        problems.append(f"inlier ratio {run.inlier_ratio:.2f} < {min_ratio}")
    if problems:
        return GateResult(name, False, "self-registration is not near-perfect: " + "; ".join(problems), detail)
    return GateResult(name, True, f"RMSE {rmse:.3f} px, inlier ratio {run.inlier_ratio:.2f}", detail)


# ---------------------------------------------------------------------------
# CHECK-06: shared-mask guard
# ---------------------------------------------------------------------------
def shared_mask_gate(src_plane, ref_plane) -> GateResult:
    """Source and reference masks must be different arrays.

    One mask object reused for both images makes every masked-out region match
    itself perfectly -- a failure that LOOKS like excellent registration.
    np.shares_memory also catches views of the same buffer, not only `is`.
    """
    name = "masks_independent"
    shared = []
    for attr in ("valid_mask", "shadow_mask"):
        a, b = getattr(src_plane, attr, None), getattr(ref_plane, attr, None)
        if a is None or b is None:
            continue
        if a is b or np.shares_memory(np.asarray(a), np.asarray(b)):
            shared.append(attr)
    if shared:
        return GateResult(name, False, f"source and reference share the same {', '.join(shared)} "
                                       f"in memory", {"shared": shared})
    return GateResult(name, True, "source and reference masks are separate arrays")


# ---------------------------------------------------------------------------
# Independent cross-check: a second method that fails DIFFERENTLY
# ---------------------------------------------------------------------------
def transform_gap_px(m1: np.ndarray, m2: np.ndarray, shape: tuple[int, int]) -> float:
    """RMS distance between where two transforms send a 16 x 16 grid of source pixels."""
    h, w = shape
    ys, xs = np.mgrid[0:h:complex(16), 0:w:complex(16)]
    p = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    a, b = np.asarray(m1, float) @ p, np.asarray(m2, float) @ p
    return float(np.sqrt(np.mean(np.sum((a[:2] / a[2] - b[:2] / b[2]) ** 2, axis=0))))


def crosscheck_gate(primary: np.ndarray, checker: Optional[np.ndarray], checker_accepted: bool,
                    shape: tuple[int, int], flag_px: Optional[float] = None,
                    name: str = "independent_crosscheck") -> Optional[GateResult]:
    """Does an independent method agree with the primary result?

    Why it exists, measured (scripts/rift_crosscheck.py, 192 synthetic runs):
    the quality gate accepted 12 results that were 2.2-5.7 px wrong, all from
    learned matchers. They were CONSISTENTLY wrong -- shift the input 5 px and
    the wrong answer shifts 5 px -- so the control gates above caught only 2.
    RIFT2 (classical, untrained, phase-based) fails differently, and as a
    checker it caught 8 of the 12 with 0 false alarms on 64 correct results.

    Returns None -- NOT a pass, NOT a fail -- when the checker has no confident
    answer of its own. An abstaining checker has no opinion: counting it as a
    pass would claim a check that never happened, and counting it as a fail
    would reject every correct result in a regime the checker cannot handle
    (29 correct results at +30 deg lighting, where RIFT2 fails). Callers add the
    gate to RegistrationResult.gates only when it returns a result.
    """
    if checker is None or not checker_accepted:
        return None
    limit = float(_cfg("crosscheck_flag_px", 2.0)) if flag_px is None else float(flag_px)
    gap = transform_gap_px(primary, checker, shape)
    detail = {"gap_px": round(gap, 3), "flag_px": limit}
    if gap > limit:
        return GateResult(name, False, f"an independent method disagrees by {gap:.2f} px "
                                       f"(> {limit} px): the primary may be consistently wrong", detail)
    return GateResult(name, True, f"independent method agrees to {gap:.2f} px", detail)


# ---------------------------------------------------------------------------
# The real pipeline, wrapped so the gates can drive it
# ---------------------------------------------------------------------------
def pipeline_from(matcher: str = "sift", device: Optional[str] = None,
                  gsd_m: float = 1.0, stages: Optional[dict] = None, **match_kwargs) -> Pipeline:
    """Wrap the matcher + fine stage the registration actually uses.

    The gates must exercise THE SAME code path that produced the result they
    certify; a separate, simpler stand-in would test itself, not the pipeline.
    `match_kwargs` (precision, tile_px, ...) are passed to adapter.match for that
    reason: a result matched in fp16 or in tiles is certified in fp16 or in tiles.
    `stages` are pipeline.fine_stage's stage switches, for the same reason. (The
    terrain filter cannot run inside a gate -- a gate's synthetic source has no
    ground position -- and fine_stage records it as skipped.)
    """
    from .. import synth
    from ..matching import adapter, classical
    from ..pipeline import fine_stage

    def plane(a: np.ndarray):
        a = np.asarray(a, np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        # Fresh masks per call: sharing them would defeat CHECK-06's own premise.
        return synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool),
                                shadow_mask=np.zeros(a.shape, bool), gsd_m=gsd_m,
                                meta=None, geo=None)

    def run(src: np.ndarray, ref: np.ndarray) -> PipelineRun:
        s, r = plane(src), plane(ref)
        if matcher in ("sift", "akaze", "orb", "brisk"):
            ms = classical.match(s, r, detector=matcher)
        else:
            ms = adapter.match(s, r, model_name=matcher, device=device, **match_kwargs)
        n = int(len(ms.src_pts))
        if n < 4:
            return PipelineRun(False, n, 0, None, f"only {n} matches")
        fr = fine_stage(ms, s.array, r.array, flags=stages,
                        centre=(s.array.shape[1] / 2.0, s.array.shape[0] / 2.0))
        res = fr.first
        m = None if fr.model is None or fr.model.matrix is None else np.asarray(fr.model.matrix, float)
        return PipelineRun(bool(res.ok), n, int(res.inlier_count), m,
                           res.notes[-1] if res.notes else "", fr.rmse_px)

    run.__name__ = f"pipeline[{matcher}]"
    return run


# ---------------------------------------------------------------------------
# All of them, and CHECK-08's guard
# ---------------------------------------------------------------------------
def run_all(pipeline: Pipeline, src: np.ndarray, ref: np.ndarray,
            src_plane=None, ref_plane=None, seed: int = 0,
            base: Optional[PipelineRun] = None) -> GateReport:
    """Run every control gate against the pipeline that produced a result.

    The shared-mask gate needs the ImagePlanes; without them it is recorded as
    not passed rather than silently left out, so a caller cannot skip it by
    omission.
    """
    shift = tuple(int(v) for v in _cfg("perturbation_shift_xy", [3, 4]))
    results = [
        blank_gate(pipeline, src, ref),
        noise_gate(pipeline, src, ref, seed=seed),
        perturbation_gate(pipeline, src, ref, shift=shift, base=base),
        identity_gate(pipeline, ref),
    ]
    if src_plane is not None and ref_plane is not None:
        results.append(shared_mask_gate(src_plane, ref_plane))
    else:
        results.append(GateResult("masks_independent", False,
                                  "not evaluated: no ImagePlanes were given to check"))
    return GateReport(results)


def require_gates(result) -> None:
    """CHECK-08: refuse a result that carries no control-gate results.

    A FAILED gate is a legitimate thing to show (the result renders as REJECTED
    with its reason). An ABSENT gate means nobody checked, and that must never
    reach a report looking like a finding. Part 3 calls this before rendering.
    """
    gates = getattr(result, "gates", None)
    if not gates:
        raise UngatedResultError(
            "this registration result has no control-gate results; a result nobody "
            "checked cannot be rendered as if it were one (rule H4, CHECK-08)")
