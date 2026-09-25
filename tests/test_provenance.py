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
