import json

import pytest

from chandralign.cli import build_parser, main


@pytest.mark.parametrize("command", ["register", "benchmark", "ablate", "report", "demo"])
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
    assert main(["register", "--mock", "--cpu", "--out", str(out),
                 "--config", "configs/default.yaml"]) == 0
    expected = {"matches.csv", "matches.geojson", "provenance.json", "result.json",
                "registered.tif", "registered.json", "side-by-side.png", "matches.png",
                "coverage.png", "checkerboard.png", "failure-log.jsonl"}
    assert expected <= {path.name for path in out.iterdir()}
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["confidence_tier"] in ("HIGH", "MEDIUM", "LOW", "REJECTED")
    assert result["metrics"]["runtime_s"] is not None
    failure_record = json.loads((out / "failure-log.jsonl").read_text(encoding="utf-8"))
    assert failure_record["schema"] == "chandralign.failure-log.v1"
    assert failure_record["confidence_tier"] == result["confidence_tier"]
    assert json.loads((out / "provenance.json").read_text(encoding="utf-8"))["ship_mode"] is True


def test_register_refuses_to_overwrite_a_nonempty_run(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["register", "--mock", "--out", str(out)])
    assert (out / "keep.txt").read_text(encoding="utf-8") == "user data"
