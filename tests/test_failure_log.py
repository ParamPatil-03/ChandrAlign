import json
from types import SimpleNamespace

import pytest

from chandralign.contracts import Metrics
from chandralign.evaluate import failure_log


def _result(**overrides):
    values = {
        "failure_modes": [],
        "confidence_tier": "HIGH",
        "gates": {"blank": True},
        "notes": [],
        "metrics": Metrics(inlier_count=20, inlier_ratio=0.8,
                           spatial_coverage=0.7, runtime_s=1.2),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_register_has_exactly_twenty_explainable_modes():
    failure_log.validate_registry()
    assert set(failure_log.FAILURE_MODES) == set(range(1, 21))
    for mode in failure_log.FAILURE_MODES.values():
        assert mode.detection and mode.mitigation
        if mode.detection_kind == "automatic":
            assert mode.hooks
        else:
            assert "manual" in mode.detection.lower()


def test_modes_resolve_in_order_and_reject_unknown_contract_ids():
    assert [mode.id for mode in failure_log.modes([13, 12, 13])] == [13, 12]
    with pytest.raises(ValueError, match="unknown failure-mode ID 21"):
        failure_log.modes([21])


def test_inspect_reports_pipeline_ids_and_a_matcher_exception():
    result = _result(
        failure_modes=[12], confidence_tier="REJECTED",
        notes=["matcher 'eloftr' failed: RuntimeError: weights unavailable"],
    )
    found = failure_log.inspect(result)
    assert [item.code for item in found] == ["FM-12", "matcher_exception"]
    assert found[1].canonical_mode == 12
    assert "weights unavailable" in found[1].evidence


def test_inspect_catches_unclassified_rejection_and_missing_gates():
    found = failure_log.inspect(_result(confidence_tier="REJECTED", gates={}))
    codes = {item.code for item in found}
    assert {"missing_control_gates", "unclassified_rejection"} <= codes
    assert all(item.canonical_mode == 19 for item in found)


def test_bundle_records_missing_optional_geometry_without_calling_it_failure_12():
    bundle = SimpleNamespace(
        result=_result(),
        stages={"geometry_filter": {"applied": False, "reason": "no DEM"},
                "parallax": {"applied": False, "reason": "no ground model"}},
    )
    found = [item for item in failure_log.inspect(bundle)
             if item.code == "optional_geometry_skipped"]
    assert len(found) == 1
    assert found[0].canonical_mode is None


def test_log_run_appends_independent_json_lines(tmp_path):
    target = tmp_path / "nested" / "failures.jsonl"
    result = _result(failure_modes=[13], confidence_tier="REJECTED")
    first = failure_log.log_run(target, result, created_utc="2026-09-25T00:00:00+00:00")
    failure_log.log_run(target, result, created_utc="2026-09-25T00:01:00+00:00")

    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert first["schema"] == "chandralign.failure-log.v1"
    assert lines[0]["failure_modes"][0]["id"] == 13
    assert lines[0]["observations"][0]["code"] == "FM-13"
