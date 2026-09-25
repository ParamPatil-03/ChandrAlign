"""Audit I-15: the reproduction check must FAIL on a dropped window, a changed tier and a moved result.

The committed reports reproduce themselves exactly, so each test copies them into a rerun directory,
breaks one thing, and requires the check to notice (the old check skipped missing windows and
compared OHRC/IIRS on success + a 30-50 m offset only)."""
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_reproduction.py"
OLD = {"ohrc_nac": "ohrc_nac_q8_auto_bridge.json", "iirs_wac": "iirs_wac_mosaic.json",
       "iirs_nac": "iirs_nac_dense.json", "tmc2_nac": "tmc2_nac_registration.json"}


def _rerun_dir(tmp_path, mutate=None):
    d = tmp_path / "rerun"
    d.mkdir()
    tc = json.loads((ROOT / "reports/tmc2_tc_registration_height_ref.json").read_text(encoding="utf-8"))
    for part in ("cm", "cp"):
        (d / f"tmc2_tc_{part}.json").write_text(json.dumps({"rows": []}))
    files = {"tmc2_tc_c0": tc}
    for name, f in OLD.items():
        files[name] = json.loads((ROOT / "reports" / f).read_text(encoding="utf-8"))
    if mutate:
        mutate(files)
    for name, data in files.items():
        (d / f"{name}.json").write_text(json.dumps(data))
    return d


def _run(d, tmp_path):
    return subprocess.run([sys.executable, str(SCRIPT), str(d), "--out", str(tmp_path / "check.json")],
                          capture_output=True, text=True, timeout=300)


def test_the_committed_reports_reproduce_themselves(tmp_path):
    r = _run(_rerun_dir(tmp_path), tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.parametrize("what", ["drop_ohrc_window", "ohrc_tier", "ohrc_offset_1m", "tmc2_nac_window", "tmc2_tc_row"])
def test_a_regression_is_not_reproduced(tmp_path, what):
    def mutate(files):
        w = files["ohrc_nac"]["windows"]
        if what == "drop_ohrc_window":
            w.pop(3)
        elif what == "ohrc_tier":
            r = w[0]["results"]["routed"]
            r["tier"] = "HIGH" if r.get("tier") != "HIGH" else "LOW"
        elif what == "ohrc_offset_1m":
            r = next(x["results"]["routed"] for x in w if (x.get("results") or {}).get("routed", {}).get("implied_offset_m"))
            r["implied_offset_m"] = {**r["implied_offset_m"], "east": r["implied_offset_m"]["east"] + 1.0}
        elif what == "tmc2_nac_window":
            files["tmc2_nac"]["windows"].pop(0)
        elif what == "tmc2_tc_row":
            files["tmc2_tc_c0"]["rows"].pop(0)
    r = _run(_rerun_dir(tmp_path, mutate), tmp_path)
    assert r.returncode == 1 and "NOT REPRODUCED" in r.stdout, (what, r.stdout[-1500:])
