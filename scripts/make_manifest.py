"""Record every file under data/raw/ with its size and SHA-256 in data/manifest.json.

Feature ADD-16 / honesty rule H2: every reported number must trace back to exact bytes.
Run after any download:  python scripts/make_manifest.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    entries = {}
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            entries[path.relative_to(RAW).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "root": "data/raw",
                "file_count": len(entries),
                "total_bytes": sum(e["size_bytes"] for e in entries.values()),
                "files": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    total_mb = sum(e["size_bytes"] for e in entries.values()) / 1e6
    print(f"{len(entries)} files, {total_mb:.1f} MB -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
