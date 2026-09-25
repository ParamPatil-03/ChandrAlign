"""Static checkerboard and embeddable before/after swipe (UI-07)."""
from html import escape
from pathlib import Path

import numpy as np


def checkerboard(before: np.ndarray, after: np.ndarray, block: int = 32) -> np.ndarray:
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape:
        raise ValueError("before and after images must have identical shapes")
    if block <= 0:
        raise ValueError("block must be positive")
    yy, xx = np.indices(before.shape[:2])
    choose_after = ((yy // block + xx // block) % 2).astype(bool)
    if before.ndim > 2:
        choose_after = choose_after[..., None]
    return np.where(choose_after, after, before)


def render_checkerboard(before, after, path: str | Path, *, block: int = 32, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt
    mixed = checkerboard(before, after, block)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(mixed, cmap="gray"); ax.set_title("Before / registered checkerboard"); ax.axis("off")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def slider_html(before_url: str, after_url: str, *, position: int = 50) -> str:
    """Self-contained markup usable by both the HTML report and web UI."""
    if not 0 <= position <= 100:
        raise ValueError("position must be within 0..100")
    before, after = escape(before_url, quote=True), escape(after_url, quote=True)
    return f'''<div class="chandralign-swipe" style="position:relative;overflow:hidden">
<img src="{before}" alt="before" style="display:block;width:100%">
<div style="position:absolute;inset:0 {100-position}% 0 0;overflow:hidden">
<img src="{after}" alt="registered" style="width:{10000/max(position, 1):.6g}% ;max-width:none">
</div><input type="range" min="0" max="100" value="{position}" aria-label="comparison position">
</div>'''
