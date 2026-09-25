"""Evidence-match overlay with trusted and rejected matches separated (UI-05)."""
from pathlib import Path

import numpy as np


def counts(bundle) -> tuple[int, int]:
    mask = np.asarray(bundle.result.inlier_mask, bool)
    return int(mask.sum()), int(len(mask) - mask.sum())


def render(bundle, path: str | Path, *, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    src, ref = np.asarray(bundle.src.array), np.asarray(bundle.ref.array)
    matches = bundle.result.matches
    mask = np.asarray(bundle.result.inlier_mask, bool)
    if len(mask) != len(matches.src_pts):
        raise ValueError("result.inlier_mask does not match result.matches")
    canvas = np.zeros((max(src.shape[0], ref.shape[0]), src.shape[1] + ref.shape[1]), np.float32)
    canvas[:src.shape[0], :src.shape[1]] = src
    canvas[:ref.shape[0], src.shape[1]:] = ref
    segments = np.stack([matches.src_pts, matches.ref_pts + [src.shape[1], 0]], axis=1)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.imshow(canvas, cmap="gray")
    if (~mask).any():
        ax.add_collection(LineCollection(segments[~mask], colors="#ef4444", linewidths=0.45, alpha=0.5))
    if mask.any():
        ax.add_collection(LineCollection(segments[mask], colors="#22c55e", linewidths=0.65, alpha=0.7))
    n_in, n_out = counts(bundle)
    ax.set_title(f"Evidence matches — {n_in} inliers, {n_out} outliers")
    ax.axis("off")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path
