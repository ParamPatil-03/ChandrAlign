"""Tests for UI-02 / P3-T07 — FastAPI backend (api.py).

Acceptance criteria (FEATURES.csv UI-02 / PLAN.md P3-T07):
  ✓ POST /register returns 202 with a run_id and poll URL.
  ✓ GET  /runs/{id} returns status, progress_log, and result when done.
  ✓ GET  /runs/{id}/assets/{name} serves a produced file.
  ✓ GET  /pairs returns the curated benchmark list.
  ✓ A completed mock run returns a valid result within a reasonable timeout.
  ✓ A REJECTED result exposes confidence_tier and failure_modes (rule H3).
  ✓ Gates dict is non-empty on a successful result (rule H4).
  ✓ Metric source is always present (rule H5).
  ✓ /health returns ok.
  ✓ 404 on unknown run_id and missing asset.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Guard: skip the entire module if fastapi / httpx are not installed.
# ---------------------------------------------------------------------------
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (starlette TestClient needs it)")

from starlette.testclient import TestClient  # noqa: E402

from chandralign.api import CURATED_PAIRS, create_app  # noqa: E402


# Project-local tmp dir so we avoid the Windows Temp permission issue
_RUNS_TMP = Path(__file__).parent.parent / ".pytest_runs_api"


@pytest.fixture(scope="module")
def client():
    _RUNS_TMP.mkdir(exist_ok=True)
    app = create_app(runs_root=_RUNS_TMP)
    with TestClient(app) as c:
        yield c
    shutil.rmtree(_RUNS_TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["feature"] == "UI-02"


# ---------------------------------------------------------------------------
# GET /pairs
# ---------------------------------------------------------------------------
def test_get_pairs_returns_list(client):
    r = client.get("/pairs")
    assert r.status_code == 200
    data = r.json()
    assert "pairs" in data
    pairs = data["pairs"]
    assert len(pairs) >= 1, "At least one curated pair must be defined"


def test_pairs_have_required_fields(client):
    r = client.get("/pairs")
    for pair in r.json()["pairs"]:
        assert "pair_id"    in pair
        assert "label"      in pair
        assert "src_camera" in pair
        assert "ref_camera" in pair
        assert "mock_seed"  in pair


# ---------------------------------------------------------------------------
# 404 on unknown run
# ---------------------------------------------------------------------------
def test_unknown_run_404(client):
    r = client.get("/runs/does-not-exist-xyz")
    assert r.status_code == 404


def test_unknown_asset_404(client):
    # Register a job first so the run_id is valid
    post_r = client.post("/register", json={"mock": True, "seed": 99})
    assert post_r.status_code == 202
    run_id = post_r.json()["run_id"]
    # Then request a non-existent asset name (job may or may not be done)
    # Wait briefly so the run directory is created
    time.sleep(0.2)
    r = client.get(f"/runs/{run_id}/assets/no-such-file.txt")
    assert r.status_code == 404


@pytest.mark.parametrize("escape", ["..%5C..%5Ccanary.txt", "..%2F..%2Fcanary.txt",
                                    "..%5Ccanary.txt", "..%2Fcanary.txt"])
def test_asset_path_cannot_escape_the_run_folder(client, escape):
    """Audit 2026-09-26 C-06: `..` in an asset name read files outside the run folder."""
    canary = _RUNS_TMP / "canary.txt"
    canary.write_text("SECRET-CANARY", encoding="utf-8")
    (_RUNS_TMP.parent / "canary.txt").write_text("SECRET-CANARY", encoding="utf-8")
    try:
        run_id = client.post("/register", json={"mock": True, "seed": 98}).json()["run_id"]
        _wait_for_done(client, run_id)
        r = client.get(f"/runs/{run_id}/assets/{escape}")
        assert r.status_code == 404
        assert "SECRET-CANARY" not in r.text
    finally:
        canary.unlink(missing_ok=True)
        (_RUNS_TMP.parent / "canary.txt").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# POST /register — queuing and polling
# ---------------------------------------------------------------------------
def test_register_returns_202_with_run_id(client):
    r = client.post("/register", json={"mock": True, "seed": 1})
    assert r.status_code == 202
    data = r.json()
    assert "run_id" in data
    assert "poll"   in data
    assert data["status"] == "QUEUED"


def test_register_poll_url_is_valid(client):
    post_r = client.post("/register", json={"mock": True, "seed": 2})
    poll   = post_r.json()["poll"]
    r = client.get(poll)
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body["status"] in ("QUEUED", "RUNNING", "DONE", "FAILED")


def test_register_with_pair_id(client):
    first_pair = CURATED_PAIRS[0]["pair_id"]
    r = client.post("/register", json={"pair_id": first_pair, "mock": True})
    assert r.status_code == 202
    assert "run_id" in r.json()


# ---------------------------------------------------------------------------
# Full end-to-end: run completes, result is valid
# ---------------------------------------------------------------------------
def _wait_for_done(client, run_id: str, timeout: float = 120.0) -> dict:
    """Poll GET /runs/{run_id} until DONE or FAILED, return the final record."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("DONE", "FAILED"):
            return body
        time.sleep(0.5)
    pytest.fail(f"Run {run_id} did not finish within {timeout}s")


def test_mock_run_completes(client):
    post_r = client.post("/register", json={"mock": True, "seed": 7})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    assert record["status"] == "DONE", (
        f"Run failed with: {record.get('error')}\n"
        f"Log: {record.get('progress_log')}"
    )


def test_done_run_has_result(client):
    post_r = client.post("/register", json={"mock": True, "seed": 8})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    assert record["status"] == "DONE"
    result = record.get("result")
    assert result is not None, "result must be present when status=DONE"
    assert "confidence_tier" in result


def test_done_run_gates_non_empty(client):
    """Rule H4: gates dict must not be empty."""
    post_r = client.post("/register", json={"mock": True, "seed": 9})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    if record["status"] == "DONE":
        gates = record["result"].get("gates", {})
        assert len(gates) > 0, "gates dict must be non-empty (rule H4)"


def test_done_run_metrics_have_source(client):
    """Rule H5: every metric value must have a source tag."""
    post_r = client.post("/register", json={"mock": True, "seed": 10})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    if record["status"] == "DONE":
        metrics = record["result"].get("metrics", {})
        # The 'source' field is required on the Metrics dataclass
        assert "source" in metrics, (
            "metrics must carry a 'source' field (rule H5)"
        )
        assert metrics["source"] in ("measured", "external", "synthetic")


def test_done_run_progress_log_non_empty(client):
    post_r = client.post("/register", json={"mock": True, "seed": 11})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    assert len(record["progress_log"]) > 0, "progress_log must have entries"


def test_done_run_elapsed_s_recorded(client):
    post_r = client.post("/register", json={"mock": True, "seed": 12})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    if record["status"] == "DONE":
        assert record["elapsed_s"] is not None
        assert record["elapsed_s"] > 0


# ---------------------------------------------------------------------------
# Asset serving
# ---------------------------------------------------------------------------
def test_asset_served_after_run(client):
    post_r = client.post("/register", json={"mock": True, "seed": 13})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    if record["status"] != "DONE":
        pytest.skip("Run did not complete successfully")
    assets = record.get("assets", [])
    assert len(assets) > 0, "At least one asset should be produced"
    # Pick the first asset and download it
    name = assets[0]
    r = client.get(f"/runs/{run_id}/assets/{name}")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_result_json_asset_content(client):
    post_r = client.post("/register", json={"mock": True, "seed": 14})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    if record["status"] != "DONE":
        pytest.skip("Run did not complete successfully")
    r = client.get(f"/runs/{run_id}/assets/result.json")
    assert r.status_code == 200
    data = r.json()
    assert "confidence_tier" in data
    assert "metrics" in data
    assert "gates" in data


# ---------------------------------------------------------------------------
# REJECTED run surfaces failure modes (rule H3)
# ---------------------------------------------------------------------------
def test_rejected_run_has_failure_modes(client):
    """If the pipeline rejects, the response must explain why (rule H3)."""
    post_r = client.post("/register", json={"mock": True, "seed": 42})
    run_id = post_r.json()["run_id"]
    record = _wait_for_done(client, run_id)
    # We cannot force a REJECTED result on every seed, but if it happens
    # the failure_modes list must be present (even if empty list is OK when tier != REJECTED).
    result = record.get("result")
    if result and result.get("confidence_tier") == "REJECTED":
        assert "failure_modes" in result, (
            "REJECTED result must expose failure_modes (rule H3)"
        )


# ---------------------------------------------------------------------------
# GET /runs — session list
# ---------------------------------------------------------------------------
def test_list_runs(client):
    r = client.get("/runs")
    assert r.status_code == 200
    data = r.json()
    assert "runs" in data
    assert isinstance(data["runs"], list)
    assert len(data["runs"]) > 0, "At least the runs submitted above should appear"


# ---------------------------------------------------------------------------
# Audit 2026-09-26 C-08: synthetic runs are labelled synthetic; real requests never get one
# ---------------------------------------------------------------------------
def test_mock_run_is_labelled_synthetic_not_measured(client):
    run_id = client.post("/register", json={"mock": True, "seed": 21}).json()["run_id"]
    record = _wait_for_done(client, run_id)
    assert record["status"] == "DONE", record.get("error")
    assert record["result"]["metrics"]["source"] == "synthetic"


def test_real_labels_with_mock_are_refused(client):
    r = client.post("/register", json={"src": "a.xml", "ref": "b.xml", "mock": True})
    assert r.status_code == 422


def test_real_labels_are_never_answered_with_a_synthetic_run(client):
    """Before: mock defaulted to True, so {src, ref} silently ran a synthetic pair."""
    r = client.post("/register", json={"src": "no/such/src.xml", "ref": "no/such/ref.xml"})
    assert r.status_code == 202
    record = _wait_for_done(client, r.json()["run_id"])
    assert record["status"] == "FAILED"            # the labels do not exist; never a fake DONE
    assert "src.xml" in (record.get("error") or "")


def test_a_request_with_nothing_to_register_is_refused(client):
    assert client.post("/register", json={}).status_code == 422


def test_curated_pairs_use_their_own_seed(client):
    """Before: the request's default seed=7 shadowed every pair's mock_seed."""
    seeds = []
    for pair in CURATED_PAIRS[:2]:
        run_id = client.post("/register", json={"pair_id": pair["pair_id"], "mock": True}).json()["run_id"]
        record = _wait_for_done(client, run_id)
        log = " ".join(record["progress_log"])
        assert f"seed={pair['mock_seed']}" in log, log
        seeds.append(pair["mock_seed"])
    assert seeds[0] != seeds[1]


def test_api_runs_write_the_failure_log(client):
    """Audit 2026-09-26 I-14: only the CLI wrote the append-only failure log."""
    run_id = client.post("/register", json={"mock": True, "seed": 31}).json()["run_id"]
    record = _wait_for_done(client, run_id)
    assert record["status"] == "DONE", record.get("error")
    assert "failure-log.jsonl" in record["assets"]


# ---------------------------------------------------------------------------
# Audit 2026-09-26 C-01 / C-05: curated pairs are real products with evidence READ from the
# committed reports; real requests run the validated product path.
# ---------------------------------------------------------------------------
def test_pair_evidence_is_read_from_the_committed_report(client):
    import json as _json
    pairs = {p["pair_id"]: p for p in client.get("/pairs").json()["pairs"]}
    root = Path(__file__).resolve().parents[1]
    ohrc = _json.loads((root / "reports/ohrc_nac_q8_auto_bridge.json").read_text(encoding="utf-8"))
    for pid, nac in (("ohrc_nac", "M102014464RC"), ("ohrc_nac_75deg_sun", "M1417360906LC")):
        assert pairs[pid]["evidence"]["accepted"] == ohrc["summary"][nac]["routed"]["success"]
    tmc = _json.loads((root / "reports/tmc2_tc_registration_height_ref.json").read_text(encoding="utf-8"))
    assert pairs["tmc2_selene_tc"]["evidence"]["windows"] == len(tmc["rows"])
    for p in pairs.values():
        assert "runnable" in p and "Published" not in p.get("note", "")


def test_a_pair_that_cannot_run_here_is_refused_for_real_data(client, monkeypatch):
    import chandralign.api as api_mod
    monkeypatch.setattr(api_mod, "_label_for", lambda product_id: None)     # data not on this machine
    r = client.post("/register", json={"pair_id": "iirs_wac"})
    assert r.status_code == 422 and "not in data/raw" in r.json()["detail"]


def test_every_curated_pair_has_a_product_workflow(client):
    """IIRS -> WAC mosaic joined TMC-2 -> TC and OHRC -> NAC as a product workflow."""
    for p in client.get("/pairs").json()["pairs"]:
        assert p["why_not"] != "no product workflow for this pairing yet", p["pair_id"]


def test_real_data_refuses_a_matcher_override(client):
    r = client.post("/register", json={"src": "a.xml", "ref": "b.xml", "matcher": "sift"})
    assert r.status_code == 422


_TMC2_TC_HELD = next(iter(Path(__file__).resolve().parents[1].glob("data/raw/selene/tc/TCO_MAP_02_N03E021N00E024SC.lbl")), None)


@pytest.mark.slow
@pytest.mark.skipif(_TMC2_TC_HELD is None, reason="real TMC-2 / TC not downloaded")
def test_the_headline_pair_runs_on_the_validated_path_through_the_api(client):
    r = client.post("/register", json={"pair_id": "tmc2_selene_tc", "windows": 1})
    assert r.status_code == 202
    record = _wait_for_done(client, r.json()["run_id"], timeout=900)
    assert record["status"] == "DONE", record.get("error")
    result = record["result"]
    assert result["pairing"] == "TMC2 -> TC" and result["accepted"] == 1
    w = result["window_results"][0]
    assert w["confidence_tier"] in ("HIGH", "MEDIUM") and w["metrics"]["source"] == "measured"
    assert "window_01/registered.tif" in record["assets"]
    got = client.get(f"/runs/{record['run_id']}/assets/window_01/result.json")
    assert got.status_code == 200 and got.json()["confidence_tier"] == w["confidence_tier"]


def test_failure_modes_are_served_by_name(client):
    """The UI shows the canonical register, not invented codes like '#01 FEW_INLIERS'."""
    modes = {m["id"]: m for m in client.get("/failure-modes").json()["failure_modes"]}
    assert set(modes) == set(range(1, 21))
    assert modes[12]["name"] == "False correspondences survive robust fitting"
    assert modes[10]["mitigation"]
