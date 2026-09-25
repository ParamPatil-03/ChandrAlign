import json

import pytest

from chandralign.evaluate.control_gates import UngatedResultError
from chandralign.pipeline import register_bundle
from chandralign.product import report
from chandralign.synth import make_pair


def _run(tmp_path, *, gates=None, source="synthetic"):
    run = tmp_path / "run"
    run.mkdir()
    result = {
        "confidence_tier": "REJECTED",
        "metrics": {
            "rmse_px": None, "inlier_count": 7, "inlier_ratio": 0.25,
            "spatial_coverage": 0.5, "subpixel_recovery_err_px": None,
            "max_delaunay_gap_px": 12.5, "runtime_s": 1.75, "source": source,
        },
        "gates": {"blank_null": True, "perturbation": False} if gates is None else gates,
        "failure_modes": [12],
        "matcher": "sift", "regime": "same_modal_normal", "match_stage": "direct",
        "geometry_used": "affine",
        "stages": {"uniformity": {"applied": True, "kept": 7},
                   "parallax": {"applied": False, "reason": "no DEM"}},
    }
    provenance = {
        "git_commit": "abc123", "ship_mode": True,
        "inputs": [{"role": "src", "product_id": "SRC<&", "mission": "CH2",
                    "instrument": "TMC2", "raster": {"path": "src.img"}},
                   {"role": "ref", "product_id": "REF", "mission": "SELENE",
                    "instrument": "TC", "raster": {"path": "ref.img"}}],
    }
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (run / "coverage.png").write_bytes(b"\x89PNG\r\n\x1a\nreport-test")
    return run


def test_saved_report_is_self_contained_honest_and_complete(tmp_path):
    target = report.render_run(_run(tmp_path))
    html = target.read_text(encoding="utf-8")
    for heading in ("Inputs", "Pipeline path", "Problem-statement metrics",
                    "Precision diagnostics", "Control gates",
                    "Confidence and failure modes", "Provenance"):
        assert heading in html
    assert "not measured" in html
    assert html.count(">synthetic</span>") == 7
    assert "data:image/png;base64," in html
    assert "FM-12" in html and "False correspondences" in html
    assert "SRC&lt;&amp;" in html and "SRC<&" not in html
    assert '<span class="badge REJECTED">REJECTED</span>' in html


def test_report_refuses_empty_control_gates(tmp_path):
    with pytest.raises(UngatedResultError, match="no control-gate"):
        report.render_run(_run(tmp_path, gates={}))


def test_report_refuses_unknown_metric_source(tmp_path):
    with pytest.raises(ValueError, match="metrics.source"):
        report.render_run(_run(tmp_path, source="probably measured"))


def test_report_supports_explicit_output_path(tmp_path):
    run = _run(tmp_path)
    target = report.render_run(run, tmp_path / "published" / "evaluation.html")
    assert target.is_file() and target.parent.name == "published"


def test_live_bundle_report_renders_all_generated_figures(tmp_path):
    src, ref, _ = make_pair(out_shape=(128, 128), shift=(2.0, -1.0), seed=9,
                            n_craters=20, shadows=False)
    bundle = register_bundle(src, ref, matcher="sift")
    target = report.render_bundle(bundle, tmp_path / "live.html")
    html = target.read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == 4
    assert bundle.result.confidence_tier in html
