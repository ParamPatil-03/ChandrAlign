"""Delivered-point grid occupancy heatmap (UI-06)."""
from pathlib import Path

import numpy as np


def occupancy(bundle, grid: int = 8) -> tuple[np.ndarray, float]:
    if grid <= 0:
        raise ValueError("grid must be positive")
    points = np.asarray(bundle.delivered.src_pts, float).reshape(-1, 2)
    h, w = np.asarray(bundle.src.array).shape[:2]
    cells = np.zeros((grid, grid), int)
    if len(points):
        cx = np.clip((points[:, 0] / w * grid).astype(int), 0, grid - 1)
        cy = np.clip((points[:, 1] / h * grid).astype(int), 0, grid - 1)
        np.add.at(cells, (cy, cx), 1)
    return cells, float((cells > 0).sum() / cells.size)


def render(bundle, path: str | Path, *, grid: int = 8, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt
    cells, fraction = occupancy(bundle, grid)
    reported = bundle.result.metrics.spatial_coverage
    if reported is not None and not np.isclose(fraction, reported, atol=1e-9):
        raise ValueError(f"delivered coverage {fraction} disagrees with metrics {reported}")
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(cells, cmap="viridis", interpolation="nearest")
    for row, col in np.ndindex(cells.shape):
        ax.text(col, row, str(cells[row, col]), ha="center", va="center",
                color="white" if cells[row, col] < cells.max(initial=0) * 0.55 else "black", fontsize=7)
    ax.set(title=f"Delivered-point coverage: {fraction:.1%}", xlabel="grid column", ylabel="grid row")
    fig.colorbar(image, ax=ax, label="points")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path
