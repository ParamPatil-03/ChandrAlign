from pathlib import Path
from types import SimpleNamespace

from chandralign.product.provenance import build, canonical_sha256, file_sha256, write


def _bundle(tmp_path: Path):
    def plane(role):
        raster = tmp_path / f"{role}.img"
        label = tmp_path / f"{role}.xml"
        raster.write_bytes((role + " raster").encode())
        label.write_text(role + " label", encoding="utf-8")
        meta = SimpleNamespace(product_id=f"PRODUCT-{role}", mission="CH2",
                               instrument="OHRC", raster_path=raster, label_path=label)
        return SimpleNamespace(meta=meta)
    return SimpleNamespace(src=plane("src"), ref=plane("ref"))


def test_manifest_is_identical_apart_from_timestamp(tmp_path):
    bundle = _bundle(tmp_path)
    kwargs = {"config_data": {"b": 2, "a": 1}, "ship_mode": True,
              "packages": (), "repo_root": tmp_path}
    first = build(bundle, created_utc="2026-01-01T00:00:00Z", **kwargs)
    second = build(bundle, created_utc="2026-01-02T00:00:00Z", **kwargs)

    assert first.pop("created_utc") != second.pop("created_utc")
    assert first == second
    assert first["inputs"][0]["product_id"] == "PRODUCT-src"
    assert first["inputs"][0]["raster"]["sha256"] == file_sha256(tmp_path / "src.img")
    assert first["ship_mode"] is True


def test_config_hash_is_canonical():
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_manifest_writer_is_stable_and_creates_parent(tmp_path):
    path = write(tmp_path / "run" / "provenance.json", {"z": 1, "a": 2})
    assert path.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "z": 1\n}\n'


def test_missing_metadata_is_refused(tmp_path):
    bundle = SimpleNamespace(src=SimpleNamespace(meta=None), ref=SimpleNamespace(meta=None))
    try:
        build(bundle, packages=(), repo_root=tmp_path)
    except ValueError as exc:
        assert "meta" in str(exc)
    else:
        raise AssertionError("missing metadata was silently accepted")


# ---------------------------------------------------------------------------
# Audit 2026-09-26 I-13: provenance did not record the matcher, the run's arguments, the
# learned-matcher stack or a dirty tree, and hashed only default.yaml although routing reads
# regimes.yaml. It also re-hashed multi-GB rasters for every window.
# ---------------------------------------------------------------------------
def test_manifest_records_the_matcher_and_the_run_arguments(tmp_path):
    bundle = _bundle(tmp_path)
    bundle.result = SimpleNamespace(provenance={"matcher": "eloftr"})
    m = build(bundle, packages=(), repo_root=tmp_path, run_arguments={"windows": 2, "device": "cuda"})
    assert m["matcher"] == "eloftr"
    assert m["run_arguments"] == {"windows": 2, "device": "cuda"}


def test_default_packages_include_the_learned_matcher_stack():
    from chandralign.product.provenance import _PACKAGES
    assert {"torch", "vismatch", "rasterio", "matplotlib", "jinja2"} <= set(_PACKAGES)


def test_default_config_hash_covers_the_routing_config(tmp_path):
    from chandralign import config
    m = build(_bundle(tmp_path), packages=(), repo_root=tmp_path)
    effective = {"default": config.load("default"), "regimes": config.load("regimes")}
    assert m["config_files"] == ["default", "regimes"]
    assert m["config_sha256"] == canonical_sha256(effective)


def test_a_dirty_working_tree_is_recorded():
    from chandralign import config
    from chandralign.product.provenance import _git_state
    state = _git_state(config.ROOT)
    assert set(state) == {"git_commit", "git_dirty"}
    assert state["git_dirty"] in (True, False)


def test_a_raster_is_hashed_once_per_version(tmp_path, monkeypatch):
    import os
    import time
    from chandralign.product import provenance
    path = tmp_path / "big.img"
    path.write_bytes(b"x" * 1000)
    opened = []
    real_open = Path.open
    def counting_open(self, *a, **k):
        if str(a[0] if a else k.get("mode", "r")).startswith("r"):   # count reads, not the test's writes
            opened.append(self.name)
        return real_open(self, *a, **k)
    monkeypatch.setattr(Path, "open", counting_open)
    first = provenance.file_sha256(path)
    assert provenance.file_sha256(path) == first and opened.count("big.img") == 1
    path.write_bytes(b"y" * 1000)
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert provenance.file_sha256(path) != first and opened.count("big.img") == 2
