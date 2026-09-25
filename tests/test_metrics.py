import math

import numpy as np
import pytest

from chandralign.contracts import MatchSet, Metrics, TransformModel
from chandralign.evaluate.metrics import RuntimeTimer, compute_metrics


def _matches(src, ref=None, confidence=None):
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    ref = src.copy() if ref is None else np.asarray(ref, dtype=np.float64).reshape(-1, 2)
    confidence = (np.ones(len(src), dtype=np.float32) if confidence is None
                  else np.asarray(confidence, dtype=np.float32))
    return MatchSet(src, ref, confidence, "test", "same_modal_normal", "test")


def test_metrics_default_none():
    metrics = Metrics()
    values = vars(metrics).copy()
    values.pop("source")
    assert all(value is None for value in values.values())


def test_rmse_count_and_ratio_match_hand_calculation():
    matches = _matches(
        [[0, 0], [2, 0], [4, 0]],
        [[3, 4], [2, 0], [100, 100]],
    )
    model = TransformModel("affine", matrix=np.eye(3))

    metrics = compute_metrics(matches, np.array([True, True, False]), model, gsd_m=2.0)

    expected = math.sqrt((5.0 ** 2 + 0.0 ** 2) / 2)
    assert metrics.rmse_px == pytest.approx(expected)
    assert metrics.rmse_m == pytest.approx(expected * 2.0)
    assert metrics.inlier_count == 2
    assert metrics.inlier_ratio == pytest.approx(2 / 3)


def test_coverage_matches_hand_counted_grid():
    matches = _matches([[10, 10], [60, 10], [10, 60], [60, 60]])

    metrics = compute_metrics(
        matches,
        np.array([True, True, False, False]),
        image_shape=(100, 100),
        grid=2,
    )

    assert metrics.spatial_coverage == 0.5
    assert metrics.max_delaunay_gap_px is None


def test_empty_measured_match_set_is_zero_not_unmeasured():
    metrics = compute_metrics(_matches([]), np.zeros(0, dtype=bool), image_shape=(10, 10))

    assert metrics.inlier_count == 0
    assert metrics.inlier_ratio == 0.0
    assert metrics.spatial_coverage == 0.0
    assert metrics.rmse_px is None
    assert metrics.rmse_m is None


def test_missing_inputs_remain_none():
    metrics = compute_metrics(_matches([[1, 1]]), runtime_s=1.25)

    assert metrics.runtime_s == 1.25
    assert metrics.inlier_count is None
    assert metrics.inlier_ratio is None
    assert metrics.rmse_px is None
    assert metrics.spatial_coverage is None
    assert metrics.max_delaunay_gap_px is None


def test_delaunay_gap_is_computed_for_supported_point_set():
    matches = _matches([[10, 10], [90, 10], [10, 90], [90, 90], [50, 50]])
    metrics = compute_metrics(matches, np.ones(5, dtype=bool), image_shape=(100, 100), grid=2)

    assert metrics.max_delaunay_gap_px == pytest.approx(40.0)


def test_rejects_invalid_inputs():
    matches = _matches([[1, 1], [2, 2]])

    with pytest.raises(ValueError, match="inlier_mask"):
        compute_metrics(matches, np.array([True]))
    with pytest.raises(ValueError, match="runtime_s"):
        compute_metrics(matches, runtime_s=-1)
    with pytest.raises(ValueError, match="source"):
        compute_metrics(matches, source="invented")
    with pytest.raises(ValueError, match="grid"):
        compute_metrics(matches, np.ones(2, dtype=bool), image_shape=(10, 10), grid=0)


def test_runtime_timer_measures_complete_context():
    readings = iter([10.0, 10.75])
    with RuntimeTimer(clock=lambda: next(readings)) as timer:
        pass

    assert timer.runtime_s == pytest.approx(0.75)
    assert timer.stop() == pytest.approx(0.75)


def test_runtime_timer_can_be_started_without_indenting_the_pipeline():
    readings = iter([4.0, 4.2])
    timer = RuntimeTimer(clock=lambda: next(readings)).start()

    assert timer.stop() == pytest.approx(0.2)
