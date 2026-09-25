"""G-07 (audit 2026-09-26): within-cell spread selection (docs/uniformity_fps_protocol.md)."""
import numpy as np


def test_spread_selection_is_more_regular_than_top_k_and_keeps_coverage(monkeypatch):
    """G-07: quality x spacing inside cells. Confidence peaks cluster the top-k picks; spread picks do not."""
    from chandralign import config
    from chandralign.refine import uniformity as u
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 400, (3000, 2))
    conf = 0.5 + 0.5 * np.exp(-((pts[:, 0] % 50 - 10) ** 2 + (pts[:, 1] % 50 - 10) ** 2) / 50.0)   # a hot spot per cell
    top = u.enforce(pts, conf, (400, 400), grid=8, top_k=6)
    monkeypatch.setitem(config.load("default")["uniformity"], "within_cell", "spread")
    spr = u.enforce(pts, conf, (400, 400), grid=8, top_k=6)
    assert spr.keep_mask.sum() == top.keep_mask.sum() and spr.coverage == top.coverage
    assert u.nn_cv(pts[spr.keep_mask]) < 0.7 * u.nn_cv(pts[top.keep_mask])
