"""Reproducible per-run provenance manifests (OUT-10)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .. import config

_PACKAGES = ("chandralign", "numpy", "scipy", "opencv-python-headless", "scikit-image",
             "pyyaml", "pyproj", "shapely", "pvl", "pds4_tools")


def build(bundle, *, config_data: dict | None = None, ship_mode: bool | None = None,
          created_utc: str | None = None, packages: Iterable[str] = _PACKAGES,
          repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build the complete provenance record for a RegistrationBundle."""
    configuration = config.load("default") if config_data is None else config_data
    root = Path(repo_root) if repo_root is not None else config.ROOT
    return {
        "schema_version": 1,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": [_input_record("src", bundle.src), _input_record("ref", bundle.ref)],
        "config_sha256": canonical_sha256(configuration),
        "git_commit": _git_commit(root),
        "packages": _package_versions(packages),
        "host": _host_record(),
        "ship_mode": bool(config.get("ship_mode", True) if ship_mode is None else ship_mode),
    }


def write(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write a stable, human-readable JSON manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_record(role: str, plane) -> dict[str, Any]:
    meta = getattr(plane, "meta", None)
    if meta is None:
        raise ValueError(f"bundle.{role}.meta is required for provenance")
    return {
        "role": role,
        "product_id": meta.product_id,
        "mission": meta.mission,
        "instrument": meta.instrument,
        "raster": _file_record(meta.raster_path),
        "label": _file_record(meta.label_path),
    }


def _file_record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"provenance input does not exist: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={root.resolve().as_posix()}", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True, timeout=10)
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for name in sorted(set(packages)):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _host_record() -> dict[str, str | None]:
    gpu = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5)
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        gpu = "; ".join(names) or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu,
    }
