"""The web UI shows only real results (audit 2026-09-26 C-05).

Before: app.js hard-coded GOLD / 0.78 px / 512 inliers under invented product IDs, drew
procedural craters with Math.random() match points, generated fake downloads, and swapped the
fabricated table in whenever the backend errored -- failure mode 19 in the demo surface.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "src" / "chandralign" / "ui"
JS = (UI / "app.js").read_text(encoding="utf-8")
HTML = (UI / "index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("pattern", [
    r"\bGOLD\b", r"\bSILVER\b", r"\bBRONZE\b",                 # not the contract's tiers
    r"Math\.random", r"runSimulation", r"falling back",           # simulated results / silent fallback
    r"M119283", r"MN00001", r"CH2_OHRC_2020",                     # invented product ids
    r"new Blob",                                                  # generated fake downloads
])
def test_no_fabrication_in_the_ui(pattern):
    for name, text in (("app.js", JS), ("index.html", HTML)):
        assert not re.search(pattern, text), f"{name} contains {pattern!r}"


def test_no_hard_coded_metric_values_in_the_markup():
    for element in ("m-rmse", "m-inliers", "m-cov", "m-gap", "m-rt"):
        m = re.search(rf'id="{element}">([^<]*)<', HTML)
        assert m and m.group(1) == "—", f"{element} is pre-filled with {m.group(1) if m else None!r}"


def test_the_ui_speaks_the_contract_tiers():
    from chandralign.contracts import Tier
    for tier in Tier.__args__:
        assert tier in JS, tier


def test_every_handler_the_markup_calls_exists():
    called = set(re.findall(r'on(?:click|change|input)="(\w+)\(', HTML))
    defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", JS))
    assert called and called <= defined, called - defined


def test_every_element_the_script_reads_exists():
    used = set(re.findall(r'\$\("([\w-]+)"\)', JS))
    used |= {f"sp{i}" for i in range(1, 6)} | {f"vt-{v}" for v in ("swipe", "checker", "matches", "coverage", "side")}
    present = set(re.findall(r'id="([\w-]+)"', HTML))
    assert used <= present, used - present
