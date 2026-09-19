"""Spatially uniform match-point distribution (ALIGN-04..07, P2-T07).

The problem statement asks for match points "maintaining uniform distribution
across the images", which is a requirement about WHERE the matches are, not just
how many there are or how accurate they are. It is easy to miss: a matcher that
finds 2000 correspondences all inside one well-textured crater field satisfies
every count-based metric while leaving most of the frame unconstrained.

That also degrades the fit. RANSAC's consensus is dominated by whichever region
supplied the most points, so a transform can be driven almost entirely by one
corner and then extrapolate badly everywhere else -- which is exactly the
failure the PS illustrates (a dense cluster plus empty regions).

The five steps below follow PLAN.md section P2-T07 exactly. Step 3 (refilling
empty cells at a looser threshold) is the one that needs care: relaxing a
threshold to manufacture matches is how a system starts reporting coverage it
has not earned, so refilled points are marked and can be reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config


@dataclass
class UniformityResult:
    keep_mask: np.ndarray                 # which input matches survive
    coverage: float                       # fraction of grid cells holding >=1 point
    occupancy: np.ndarray                 # (grid, grid) counts after selection
    grid: int
    empty_cells: list[tuple[int, int]] = field(default_factory=list)
    max_delaunay_gap_px: float | None = None
    notes: list[str] = field(default_factory=list)


def _cell_index(pts: np.ndarray, shape: tuple[int, int], grid: int):
    """Map points to grid cells using the IMAGE extent, not the point extent.

    Using the points' own bounding box would be circular: a cluster in one
    corner would be spread across the whole grid and score perfect coverage.
    """
    h, w = shape
    cx = np.clip((pts[:, 0] / max(w, 1) * grid).astype(int), 0, grid - 1)
    cy = np.clip((pts[:, 1] / max(h, 1) * grid).astype(int), 0, grid - 1)
    return cy, cx


def adaptive_grid(shape: tuple[int, int], n_matches: int,
                  target_per_cell: int = 6, lo: int = 4, hi: int = 16) -> int:
    """Choose N for the N x N grid from image size and match count.

    A fixed grid is wrong at both ends: too coarse and it certifies clustered
    points as uniform, too fine and most cells are empty however good the match
    set is. Aim for roughly `target_per_cell` matches per cell.
    """
    if n_matches <= 0:
        return lo
    return int(np.clip(round(np.sqrt(n_matches / max(target_per_cell, 1))), lo, hi))


def coverage_of(pts: np.ndarray, shape: tuple[int, int], grid: int) -> float:
    """Fraction of grid cells containing at least one point (a PS-named metric)."""
    if len(pts) == 0:
        return 0.0
    cy, cx = _cell_index(np.asarray(pts, np.float64).reshape(-1, 2), shape, grid)
    occ = np.zeros((grid, grid), int)
    np.add.at(occ, (cy, cx), 1)
    return float((occ > 0).sum() / (grid * grid))


def max_delaunay_gap(pts: np.ndarray, shape: tuple[int, int] | None = None) -> float | None:
    """Radius of the largest empty circle in the match set, or None.

    A second, complementary coverage diagnostic: grid occupancy can look healthy
    while a diagonal band holds nothing, because a band clips the corners of
    many cells without emptying any of them.

    This uses the circumradius of each Delaunay triangle, not its longest edge.
    That matters. A Delaunay circumcircle is empty by construction, so the
    largest circumradius IS the largest hole -- whereas the longest edge is
    dominated by thin boundary triangles and does not measure emptiness at all.
    An earlier version used the longest edge and scored a deliberately holed
    point set as BETTER covered than a uniform random one, which is backwards.

    Triangles whose circumcentre falls outside `shape` are ignored when it is
    given, since a sliver on the convex hull has a huge circumradius centred
    far outside the image and describes no real gap in the data.
    """
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return None
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(pts)
    except Exception:
        return None

    largest = 0.0
    for simplex in tri.simplices:
        a, b, c = pts[simplex]
        ab, bc, ca = (np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))
        u, v = b - a, c - a
        area2 = abs(float(u[0] * v[1] - u[1] * v[0]))   # 2-D cross, avoids the NumPy 2.0 deprecation
        if area2 < 1e-9:
            continue                      # degenerate, carries no information
        radius = float(ab * bc * ca / (2.0 * area2))
        if shape is not None:
            centre = _circumcentre(a, b, c)
            if centre is None:
                continue
            h, w = shape
            if not (0 <= centre[0] <= w and 0 <= centre[1] <= h):
                continue
        largest = max(largest, radius)
    return largest


def _circumcentre(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray | None:
    d = 2.0 * float(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-9:
        return None
    a2, b2, c2 = float(a @ a), float(b @ b), float(c @ c)
    return np.array([
        (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
        (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d,
    ])


def enforce(src_pts: np.ndarray, confidence: np.ndarray, shape: tuple[int, int], *,
            grid: int | None = None, top_k: int | None = None,
            compute_delaunay: bool = True) -> UniformityResult:
    """Steps 1, 2, 4 and 5: grid, per-cell top-k, coverage, Delaunay diagnostic.

    Returns a keep mask over the input matches. Selection is by confidence
    within each cell, so thinning a dense region keeps its best evidence rather
    than an arbitrary subset.
    """
    pts = np.asarray(src_pts, np.float64).reshape(-1, 2)
    conf = np.asarray(confidence, np.float64).ravel()
    if len(pts) == 0:
        g = grid or int(config.get("uniformity.grid", 8))
        return UniformityResult(np.zeros(0, bool), 0.0, np.zeros((g, g), int), g,
                                notes=["no matches to distribute"])
    if len(conf) != len(pts):
        conf = np.ones(len(pts))

    if grid is None:
        configured = config.get("uniformity.grid", None)
        grid = int(configured) if configured else adaptive_grid(shape, len(pts))
    if top_k is None:
        top_k = int(config.get("uniformity.top_k_per_cell", 6))

    cy, cx = _cell_index(pts, shape, grid)
    keep = np.zeros(len(pts), bool)
    order = np.argsort(-conf, kind="stable")          # best first
    counts: dict[tuple[int, int], int] = {}
    for i in order:
        cell = (int(cy[i]), int(cx[i]))
        if counts.get(cell, 0) < top_k:
            counts[cell] = counts.get(cell, 0) + 1
            keep[i] = True

    occ = np.zeros((grid, grid), int)
    if keep.any():
        np.add.at(occ, (cy[keep], cx[keep]), 1)
    empty = [(int(r), int(c)) for r, c in zip(*np.where(occ == 0))]

    res = UniformityResult(
        keep_mask=keep,
        coverage=float((occ > 0).sum() / (grid * grid)),
        occupancy=occ, grid=grid, empty_cells=empty)
    if compute_delaunay:
        res.max_delaunay_gap_px = max_delaunay_gap(pts[keep], shape)
    res.notes.append(f"grid {grid}x{grid}, top-{top_k} per cell, "
                     f"{int(keep.sum())}/{len(pts)} kept, "
                     f"{len(empty)} empty cells")
    return res


def empty_cell_regions(result: UniformityResult, shape: tuple[int, int]):
    """Step 3 support: pixel bounds of each empty cell.

    The refill pass is restricted to these boxes rather than relaxing the
    threshold globally, which would let weak matches back in everywhere.
    """
    h, w = shape
    ch, cw = h / result.grid, w / result.grid
    return [(int(r * ch), int((r + 1) * ch), int(c * cw), int((c + 1) * cw))
            for r, c in result.empty_cells]


def merge_refill(base_keep: np.ndarray, refill_src: np.ndarray,
                 result: UniformityResult, shape: tuple[int, int]) -> UniformityResult:
    """Fold a refill pass back in, recording how many points were refilled.

    Refilled matches passed a LOOSER threshold, so they are counted separately.
    Coverage earned by relaxing a threshold is not the same as coverage the
    primary pass found, and a report that blurs the two overstates the result.
    """
    refill_src = np.asarray(refill_src, np.float64).reshape(-1, 2)
    if len(refill_src) == 0:
        result.notes.append("refill pass added nothing")
        return result
    cy, cx = _cell_index(refill_src, shape, result.grid)
    occ = result.occupancy.copy()
    np.add.at(occ, (cy, cx), 1)
    result.occupancy = occ
    result.coverage = float((occ > 0).sum() / (result.grid * result.grid))
    result.empty_cells = [(int(r), int(c)) for r, c in zip(*np.where(occ == 0))]
    result.notes.append(f"refill added {len(refill_src)} matches at a looser "
                        f"threshold; coverage now {result.coverage:.2f}")
    return result
