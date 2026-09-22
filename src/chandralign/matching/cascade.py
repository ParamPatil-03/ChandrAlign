"""Scale-bridging cascade (MATCH-10, P2-T11).

WHAT DECIDES WHETHER A JUMP IS POSSIBLE -- AND IT IS NOT THE SCALE RATIO
A large scale gap is crossed by bringing the sharper image down to the coarser
camera's resolution and matching there. That works at any ratio as long as the
sharper image still COVERS ENOUGH of the coarser camera's pixels to carry
structure. Measured on our real products (both axes, from estimate/scale.py):

    step           ratio    sharper image's width in coarser pixels
    OHRC -> TMC-2   16x          727
    OHRC -> IIRS   260x           37      <- a 37-pixel sliver: nothing to match
    TMC-2 -> IIRS   16x          197
    OHRC -> TC      24x          495

OHRC -> TMC-2 and OHRC -> IIRS differ in ratio by 16x, but what actually
separates the feasible step from the impossible one is footprint: OHRC is
3.7 km wide, which is 727 TMC-2 pixels and 37 IIRS pixels. That is also why
two teams measured 0 inliers on direct OHRC <-> IIRS. So the planner's rule is
a footprint rule, and OHRC -> IIRS is routed via TMC-2 because of it -- the
behaviour MATCH-10's done_when asks for, arrived at from the physics.

The minimum footprint (cascade.min_footprint_px) is MEASURED, not chosen: see
scripts/measure_cascade_footprint.py.

WHAT A CHAIN REPORTS
Each step's fit error is expressed in its own reference pixels, then carried
through every LATER step into the final reference: an error of e intermediate
pixels becomes e * s final pixels, where s is the local scale of the rest of
the chain. Steps are independent fits, so the total adds in quadrature. Both
per-step and total error are reported (done_when), in final pixels and metres.

These are INTERNAL errors -- how well each fit explains its own matches. They
bound precision, not accuracy: a chain of self-consistent wrong steps would
report small numbers. Accuracy needs an independent check (the tests' analytic
truth; on real data, agreement between two different routes).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import config
from ..contracts import SceneMeta, TransformModel
from ..estimate import models


@dataclass(frozen=True)
class Node:
    """One product that can be a step in a chain."""

    name: str
    across_m: float          # pixel size, from estimate.scale.pixel_scale (both axes)
    along_m: float
    width_m: float           # ground footprint of the whole product
    length_m: float

    @property
    def pixel_m(self) -> float:
        """The coarser of the two pixel dimensions: what limits what it can see."""
        return max(self.across_m, self.along_m)

    @property
    def short_side_m(self) -> float:
        return min(self.width_m, self.length_m)


def node_from(meta: SceneMeta, name: Optional[str] = None) -> Node:
    from ..estimate.scale import pixel_scale

    ps = pixel_scale(meta)
    lines, samples = meta.array_shape
    return Node(name or str(meta.instrument), ps.across_m, ps.along_m,
                samples * ps.across_m, lines * ps.along_m)


@dataclass
class Step:
    src: str
    ref: str
    ratio: float              # coarser pixel / finer pixel
    footprint_px: float       # finer image's short side, in the coarser camera's pixels
    feasible: bool
    reason: str


@dataclass
class Plan:
    src: str
    ref: str
    route: str                # "direct" | "cascade" | "impossible"
    steps: list[Step] = field(default_factory=list)
    reason: str = ""

    @property
    def path(self) -> list[str]:
        return [self.src] + [s.ref for s in self.steps] if self.steps else [self.src, self.ref]


def min_footprint_px() -> float:
    return float(config.get("cascade.min_footprint_px", 112))     # measured; see configs/default.yaml


def step_between(a: Node, b: Node, threshold: Optional[float] = None) -> Step:
    """Can a be registered to b in one step? Direction-free: the FINER one is
    brought down to the coarser one's resolution, so what matters is how many
    of the coarser camera's pixels the finer image's footprint spans."""
    threshold = min_footprint_px() if threshold is None else float(threshold)
    fine, coarse = (a, b) if a.pixel_m <= b.pixel_m else (b, a)
    ratio = coarse.pixel_m / min(fine.across_m, fine.along_m)
    px = fine.short_side_m / coarse.pixel_m
    ok = px >= threshold
    reason = (f"{fine.name} spans {px:.0f} {coarse.name} pixels across "
              f"({fine.short_side_m / 1000:.2f} km at {coarse.pixel_m:.3g} m); "
              + ("enough to match" if ok else f"below the {threshold:.0f} px needed to match"))
    return Step(a.name, b.name, round(ratio, 2), round(px, 1), ok, reason)


def plan(src: Node, ref: Node, via: list[Node] = (), threshold: Optional[float] = None) -> Plan:
    """Fewest-step chain from src to ref through the products we hold.

    Breadth-first over feasible steps, so a direct step is always preferred when
    it is possible, and an intermediate is used only when it is needed. Among
    chains of equal length, the one whose TIGHTEST step spans the most coarse
    pixels wins -- the chain with the most room to spare.
    """
    nodes = {n.name: n for n in [src, ref, *via]}
    direct = step_between(src, ref, threshold)
    if direct.feasible:
        return Plan(src.name, ref.name, "direct", [direct], direct.reason)

    best: Optional[list[Step]] = None
    queue: deque[tuple[str, list[Step]]] = deque([(src.name, [])])
    seen_depth: dict[str, int] = {src.name: 0}
    while queue:
        here, steps = queue.popleft()
        if best is not None and len(steps) >= len(best):
            continue
        for name, node in nodes.items():
            if name == here or name in [s.src for s in steps]:
                continue
            st = step_between(nodes[here], node, threshold)
            if not st.feasible:
                continue
            chain = steps + [st]
            if name == ref.name:
                if best is None or len(chain) < len(best) or (
                        len(chain) == len(best)
                        and min(s.footprint_px for s in chain) > min(s.footprint_px for s in best)):
                    best = chain
                continue
            if seen_depth.get(name, 1 << 30) >= len(chain):
                seen_depth[name] = len(chain)
                queue.append((name, chain))

    if best is None:
        return Plan(src.name, ref.name, "impossible", [], f"no feasible chain: {direct.reason}")
    return Plan(src.name, ref.name, "cascade", best,
                f"direct is infeasible ({direct.reason}); routed via "
                + " -> ".join([src.name] + [s.ref for s in best]))


# ---------------------------------------------------------------------------
# Chaining the fitted steps
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """One fitted step: a transform from this step's source px to its reference px."""

    src: str
    ref: str
    model: TransformModel
    rmse_px: float            # fit error, in THIS step's reference pixels
    ref_pixel_m: float        # size of one reference pixel of this step, metres


@dataclass
class ChainResult:
    model: TransformModel     # composed: first source px -> final reference px
    per_step: list[dict]
    total_rmse_px: float      # final reference pixels
    total_rmse_m: float
    notes: list[str] = field(default_factory=list)


def _local_scale(model: TransformModel, at: np.ndarray) -> float:
    return float(models.estimated_scale(model.matrix, at=(float(at[0]), float(at[1]))))


def chain(steps: list[StepResult], at: Optional[tuple[float, float]] = None) -> ChainResult:
    """Compose the steps and propagate each step's error into the final frame.

    `at` is the point (first-source pixels) where the chain's local scales are
    evaluated; the source centre is the natural choice.
    """
    if not steps:
        raise ValueError("an empty chain has no transform")
    missing = [i for i, s in enumerate(steps) if s is None]
    if missing:
        raise ValueError(f"step(s) {missing} did not register: a chain with a failed step has "
                         f"no transform, and is never bridged over")
    for a, b in zip(steps, steps[1:]):
        if a.ref != b.src:
            raise ValueError(f"steps do not connect: {a.src}->{a.ref} then {b.src}->{b.ref}")

    total = steps[0].model
    for s in steps[1:]:
        total = models.compose(total, s.model)

    p = np.array(at if at is not None else (0.0, 0.0), float)
    # Where each step's reference point lies, so later scales are evaluated in place.
    points = [p]
    for s in steps:
        points.append(models.apply(s.model, points[-1][None, :])[0])

    final_px_m = steps[-1].ref_pixel_m
    per_step, sq = [], 0.0
    for i, s in enumerate(steps):
        # Scale of everything AFTER step i, at the point where step i lands.
        rest_scale = 1.0
        for j in range(i + 1, len(steps)):
            rest_scale *= _local_scale(steps[j].model, points[j])
        e_final = s.rmse_px * rest_scale
        sq += e_final ** 2
        per_step.append({
            "step": f"{s.src} -> {s.ref}",
            "rmse_px_own": round(s.rmse_px, 4),
            "rmse_m": round(s.rmse_px * s.ref_pixel_m, 3),
            "rmse_px_final": round(e_final, 4),
        })
    total_px = math.sqrt(sq)
    return ChainResult(total, per_step, round(total_px, 4), round(total_px * final_px_m, 3),
                       notes=["per-step and total errors are INTERNAL fit errors, added in "
                              "quadrature; they bound precision, not accuracy"])


# ---------------------------------------------------------------------------
# Executing one step: bring the finer image down, match, map back
# ---------------------------------------------------------------------------
def block_average(img: np.ndarray, factor: int) -> np.ndarray:
    """Area-average factor x factor blocks: how a coarser detector sees the scene."""
    img = np.asarray(img, np.float32)
    f = int(factor)
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def downsample_transform(factor: int) -> np.ndarray:
    """Pixel map from a fine image to its factor x block average: x' = (x + 0.5)/f - 0.5."""
    f = float(factor)
    return np.array([[1 / f, 0, 0.5 / f - 0.5], [0, 1 / f, 0.5 / f - 0.5], [0, 0, 1]], float)


def register_step_dense(src_img: np.ndarray, ref_img: np.ndarray, factor: int, *,
                        src: str, ref: str, ref_pixel_m: float,
                        prior: Optional[np.ndarray] = None,
                        diag: Optional[dict] = None) -> Optional[StepResult]:
    """One cascade step by DENSE matching: the method the footprint threshold
    was measured with, so a step the planner calls feasible is executed the way
    feasibility was established.

      1. block-average the finer image by `factor`
      2. orient it with `prior`'s linear part (2x2: small px -> ref px; rotation
         and residual scale from metadata/geolocation). Position is NOT taken
         from the prior -- it is searched for.
      3. MIND template search over the whole reference -> integer position
      4. sub-pixel refinement on raw intensity (iterated NCC, iterated phase,
         ECC) AND on each MIND channel; the answer is the median of all that
         succeed (at least three) and their robust spread is the step's error

    Sparse features failed here on purpose-built synthetic data: a 128x128
    block-averaged patch held 6 SIFT keypoints. Dense matching does not need
    keypoints, which is why it is the default for coarse, small-footprint steps.
    Returns None if the match is ambiguous (z < cascade.min_z) -- never a guess.
    Pass a dict as `diag` to learn WHY a step failed: a bare None cannot be debugged.
    """
    diag = diag if diag is not None else {}
    import cv2

    from ..preprocess.phase_congruency import mind
    from ..refine import subpixel

    small = block_average(src_img, factor) if factor > 1 else np.asarray(src_img, np.float32)
    ref_arr = np.asarray(ref_img, np.float32)
    lin = np.eye(2) if prior is None else np.asarray(prior, float)[:2, :2]

    # 2. orient: canvas = T(c_out) . lin . T(-c_in)
    h, w = small.shape
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) - [w / 2, h / 2]
    span = np.abs(corners @ lin.T).max(axis=0)
    size = (int(np.ceil(2 * span[0])) + 4, int(np.ceil(2 * span[1])) + 4)
    Wc = np.eye(3)
    Wc[:2, :2] = lin
    Wc[:2, 2] = np.array([size[0] / 2, size[1] / 2]) - lin @ np.array([w / 2, h / 2])
    canvas = cv2.warpAffine(small, Wc[:2], size, flags=cv2.INTER_AREA)
    valid = cv2.warpAffine(np.ones_like(small), Wc[:2], size, flags=cv2.INTER_NEAREST) > 0.5
    ys, xs = np.where(valid)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    while not valid[y0:y1, x0:x1].all():
        y0, y1, x0, x1 = y0 + 1, y1 - 1, x0 + 1, x1 - 1
    tpl = canvas[y0:y1, x0:x1]
    diag["template_px"] = list(tpl.shape)
    if tpl.shape[0] >= ref_arr.shape[0] or tpl.shape[1] >= ref_arr.shape[1]:
        diag["failed"] = "template is not smaller than the reference region"
        return None

    # 3. MIND search
    nz = lambda a: (a - a.min()) / max(float(a.max() - a.min()), 1e-9)
    fr, ft = mind(nz(ref_arr)).astype(np.float32), mind(nz(tpl)).astype(np.float32)
    cmap = np.mean([cv2.matchTemplate(fr[..., i], ft[..., i], cv2.TM_CCOEFF_NORMED)
                    for i in range(fr.shape[-1])], axis=0)
    iy, ix = np.unravel_index(int(np.argmax(cmap)), cmap.shape)
    z = (float(cmap[iy, ix]) - float(cmap.mean())) / float(cmap.std() + 1e-9)
    diag.update(z=round(z, 1), peak_xy=[int(ix), int(iy)], search_px=list(cmap.shape[::-1]))
    if ix in (0, cmap.shape[1] - 1) or iy in (0, cmap.shape[0] - 1):
        diag["note"] = "peak on the edge of the search region: the true place may lie outside it"
    if z < float(config.get("cascade.min_z", 10.0)):
        diag["failed"] = f"ambiguous match: z {z:.1f} below cascade.min_z"
        return None

    # 4. two independent sub-pixel estimators on the aligned pair
    th, tw = tpl.shape
    patch = ref_arr[iy:iy + th, ix:ix + tw]
    # Sub-pixel estimates from RAW intensity and from each MIND channel.
    #  - Raw estimators (iterated NCC, iterated phase, ECC): which is best depends
    #    on texture -- NCC won on real OHRC, phase on smooth synthetic terrain,
    #    and ECC failed whenever the true shift was fractional.
    #  - MIND channels: raw intensity fails under a lighting change. On real
    #    OHRC -> TC (OHRC's sun 7 deg up) MIND FOUND the match (z 11.3) and every
    #    raw refiner then failed. Refining on the same illumination-invariant
    #    description that found the match keeps the step alive.
    # Value = median of every estimate that succeeds; error = robust spread
    # (1.4826 x MAD per axis). Channel estimates share one image, so the spread
    # is a PRECISION figure, not an accuracy guarantee -- as the chain says.
    ests = [subpixel.estimate(tpl, patch, m) for m in ("ncc_gaussian_iter", "phase_iter", "ecc")]
    mt, mp = mind(nz(tpl)).astype(np.float32), mind(nz(patch)).astype(np.float32)
    ests += [subpixel.estimate(mt[..., i], mp[..., i], "ncc_gaussian_iter") for i in range(mt.shape[-1])]
    ok = [e for e in ests if e.ok and np.all(np.isfinite(e.d)) and np.hypot(*e.d) <= 1.5]
    diag["subpixel_ok"] = f"{len(ok)}/{len(ests)}"
    if len(ok) < 3:
        diag["failed"] = f"only {len(ok)} of {len(ests)} sub-pixel estimates succeeded"
        return None
    ds = np.array([e.d for e in ok])
    d = np.median(ds, axis=0)
    mad = 1.4826 * np.median(np.abs(ds - d), axis=0)
    err = float(np.hypot(*mad))

    # ref px = canvas px - (x0, y0) + (ix, iy) + d
    shift = np.eye(3)
    shift[:2, 2] = np.array([ix - x0, iy - y0], float) + d
    total = shift @ Wc @ (downsample_transform(factor) if factor > 1 else np.eye(3))
    return StepResult(src, ref, TransformModel(kind="affine", matrix=total), err, ref_pixel_m)


def register_step(src_img: np.ndarray, ref_img: np.ndarray, factor: int, *,
                  src: str, ref: str, ref_pixel_m: float,
                  matcher: str = "sift", device: Optional[str] = None) -> Optional[StepResult]:
    """One cascade step. The finer image is block-averaged by `factor` to the
    coarser resolution, matched with THE SAME matcher + estimator path the
    control gates exercise, and the downsampling is composed back in, so the
    returned model maps ORIGINAL source pixels to reference pixels.
    Returns None when the step does not register -- a failed step ends the
    chain; it is never bridged over."""
    from ..evaluate.control_gates import pipeline_from

    small = block_average(src_img, factor) if factor > 1 else np.asarray(src_img, np.float32)
    run = pipeline_from(matcher, device=device)(small, ref_img)
    if not run.ok or run.matrix is None:
        return None
    total = run.matrix @ downsample_transform(factor) if factor > 1 else run.matrix
    return StepResult(src, ref, TransformModel(kind="affine", matrix=total),
                      float(run.rmse_px if run.rmse_px is not None else float("nan")), ref_pixel_m)
