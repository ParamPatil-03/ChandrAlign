"""Side-by-side source/reference comparison with sun-angle context (UI-04)."""
from pathlib import Path

import numpy as np


def sun_angle_label(src_meta, ref_meta) -> str:
    src = getattr(src_meta, "solar_incidence_deg", None)
    ref = getattr(ref_meta, "solar_incidence_deg", None)
    if src is None or ref is None:
        return "Δ solar incidence: not measured"
    return f"Δ solar incidence: {abs(float(src) - float(ref)):.1f}°"


def render(bundle, path: str | Path, *, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, plane, title in zip(axes, (bundle.src, bundle.ref), ("Source", "Reference")):
        ax.imshow(np.asarray(plane.array), cmap="gray")
        ax.set_title(f"{title}: {plane.meta.product_id}")
        ax.axis("off")
    fig.suptitle(sun_angle_label(bundle.src.meta, bundle.ref.meta))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
