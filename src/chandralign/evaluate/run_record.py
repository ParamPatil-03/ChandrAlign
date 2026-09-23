"""How a report was produced: command line, code version, time. Owner: Member B (Part 2).

A report whose command is not recorded cannot be reproduced: the audit of
2026-09-23 found `cascade_ohrc_tc.json` made with `--windows 9` against a default
of 3, and nothing in the file said so. Every measurement script that writes a
report adds `run_record()` to it.

This is the minimum a Part 2 measurement needs. The full per-output provenance
record (input checksums, product IDs; OUT-10) is Part 3's `product/provenance.py`.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def run_record(argv: list[str] | None = None) -> dict:
    """{command, commit, uncommitted_changes, utc} for the current run."""
    argv = list(sys.argv if argv is None else argv)
    if argv:
        try:
            argv[0] = Path(argv[0]).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            pass
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "command": " ".join(["python", *argv]),
        "commit": _git("rev-parse", "HEAD"),
        # A dirty tree means the commit alone does not reproduce the numbers.
        "uncommitted_changes": None if status is None else bool(status),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
