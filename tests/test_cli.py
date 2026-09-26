import json

import pytest

from chandralign.cli import build_parser, main


@pytest.mark.parametrize("command", ["register", "report", "demo", "fetch-wac-clip"])
def test_each_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_register_requires_inputs_without_mock(tmp_path):
    with pytest.raises(SystemExit, match="--src and --ref"):
        main(["register", "--out", str(tmp_path / "run")])


def test_mock_register_runs_end_to_end_and_writes_run_folder(tmp_path):
    out = tmp_path / "run"
    assert main(["register", "--mock", "--cpu", "--out", str(out)]) == 0
    expected = {"matches.csv", "matches.geojson", "provenance.json", "result.json",
                "registered.tif", "registered.json", "side-by-side.png", "matches.png",
                "coverage.png", "checkerboard.png", "failure-log.jsonl"}
    assert expected <= {path.name for path in out.iterdir()}
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    # seed 7 is a well-posed synthetic pair (known shift 3.4, -2.2): it must be accepted, so a
    # quality regression fails here instead of passing as "some tier" (audit M-14)
    assert result["confidence_tier"] in ("HIGH", "MEDIUM", "LOW")
    assert result["metrics"]["source"] == "synthetic"
    assert result["metrics"]["runtime_s"] is not None
    failure_record = json.loads((out / "failure-log.jsonl").read_text(encoding="utf-8"))
    assert failure_record["schema"] == "chandralign.failure-log.v1"
    assert failure_record["confidence_tier"] == result["confidence_tier"]
    assert json.loads((out / "provenance.json").read_text(encoding="utf-8"))["ship_mode"] is True
    assert main(["report", "--run", str(out)]) == 0
    report_html = (out / "report.html").read_text(encoding="utf-8")
    assert "ChandraAlign evaluation report" in report_html
    assert "data:image/png;base64," in report_html


def test_register_refuses_to_overwrite_a_nonempty_run(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["register", "--mock", "--out", str(out)])
    assert (out / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_a_rejected_run_with_no_geometry_still_writes_its_reasons(tmp_path):
    """Audit 2026-09-26 I-12: `--matcher bogus` raised in the exporters; the reasons were lost."""
    out = tmp_path / "run"
    assert main(["register", "--mock", "--cpu", "--matcher", "bogus", "--out", str(out)]) == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["confidence_tier"] == "REJECTED" and result["failure_modes"]
    assert (out / "failure-log.jsonl").exists() and not (out / "registered.tif").exists()



def test_declared_but_unimplemented_commands_are_gone(capsys):
    """Audit M-18: benchmark / ablate only said 'not implemented yet'; the research scripts do that work."""
    for command in ("benchmark", "ablate"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([command])


def test_the_demo_is_real_and_says_synthetic(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out)]) == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["metrics"]["source"] == "synthetic"
    assert (out / "report.html").exists()


def test_real_products_refuse_a_matcher_override(tmp_path):
    with pytest.raises(SystemExit, match="--mock only"):
        main(["register", "--src", "a.xml", "--ref", "b.xml", "--matcher", "sift", "--out", str(tmp_path / "r")])


def test_an_unsupported_pairing_is_refused_with_the_reason(tmp_path, monkeypatch):
    from chandralign.workflows import products

    def refuse(*a, **k):
        raise products.UnsupportedPairing("no validated product workflow for TMC2 -> MI")
    monkeypatch.setattr(products, "register_products", refuse)
    with pytest.raises(SystemExit, match="refused: no validated product workflow"):
        main(["register", "--src", "a.xml", "--ref", "b.xml", "--out", str(tmp_path / "r")])
