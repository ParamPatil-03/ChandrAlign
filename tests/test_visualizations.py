from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chandralign.contracts import MatchSet, Metrics
from chandralign.viz import coverage_plot, match_plot, sidebyside, swipe


def bundle():
    image = np.arange(10000, dtype=np.float32).reshape(100, 100) / 10000
    meta_a = SimpleNamespace(product_id="SRC", solar_incidence_deg=70.0)
    meta_b = SimpleNamespace(product_id="REF", solar_incidence_deg=40.0)
    plane_a, plane_b = SimpleNamespace(array=image, meta=meta_a), SimpleNamespace(array=image, meta=meta_b)
    points = np.array([[10, 10], [60, 10], [10, 60], [60, 60]], float)
    evidence = MatchSet(points, points + 1, np.ones(4, np.float32),
                        "sift", "same_modal_normal", "direct")
    delivered = MatchSet(points[:2], points[:2] + 1, np.ones(2, np.float32),
                         "sift", "same_modal_normal", "delivered")
    result = SimpleNamespace(matches=evidence, inlier_mask=np.array([1, 1, 0, 0], bool),
                             metrics=Metrics(inlier_count=2, inlier_ratio=0.5,
                                             spatial_coverage=0.5))
    return SimpleNamespace(src=plane_a, ref=plane_b, result=result, delivered=delivered)


def test_side_by_side_labels_measured_sun_angle_and_renders(tmp_path):
    b = bundle()
    assert sidebyside.sun_angle_label(b.src.meta, b.ref.meta) == "Δ solar incidence: 30.0°"
    assert sidebyside.render(b, tmp_path / "side.png").stat().st_size > 1000


def test_match_plot_counts_match_metrics_and_renders(tmp_path):
    b = bundle()
    assert match_plot.counts(b) == (b.result.metrics.inlier_count, 2)
    assert match_plot.render(b, tmp_path / "matches.png").stat().st_size > 1000


def test_coverage_grid_shape_and_fraction_equal_metrics(tmp_path):
    b = bundle()
    cells, fraction = coverage_plot.occupancy(b, grid=2)
    assert cells.shape == (2, 2) and cells.sum() == 2
    assert fraction == b.result.metrics.spatial_coverage
    assert coverage_plot.render(b, tmp_path / "coverage.png", grid=2).stat().st_size > 1000


def test_coverage_refuses_metric_mismatch(tmp_path):
    b = bundle(); b.result.metrics.spatial_coverage = 0.25
    with pytest.raises(ValueError, match="disagrees"):
        coverage_plot.render(b, tmp_path / "bad.png", grid=2)


def test_checkerboard_and_slider_are_reusable(tmp_path):
    before = np.zeros((8, 8), np.float32); after = np.ones((8, 8), np.float32)
    mixed = swipe.checkerboard(before, after, block=2)
    assert set(np.unique(mixed)) == {0.0, 1.0}
    assert swipe.render_checkerboard(before, after, tmp_path / "checker.png", block=2).exists()
    html = swipe.slider_html("before.png", "after.png")
    assert 'type="range"' in html and "before.png" in html and "after.png" in html
