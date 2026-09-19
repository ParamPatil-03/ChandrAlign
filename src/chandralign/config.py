"""Configuration loading and the instrument GSD-ratio service.

Reads configs/*.yaml. Member A owns configs/; Part 2 only reads them.

The GSD-ratio service lives here rather than in the matcher because two
different stages need it and neither should guess: the cascade planner
(MATCH-10) uses it to decide how many hops a pairing needs, and the scale
sanity check (CHECK-05) uses it to reject a transform whose estimated scale
disagrees with what the instruments physically imply -- failure mode #13.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or internally inconsistent."""


@functools.lru_cache(maxsize=None)
def load(name: str = "default") -> dict[str, Any]:
    """Load and cache configs/<name>.yaml."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"missing config: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get(dotted: str, default: Any = None, config: str = "default") -> Any:
    """Fetch a nested value, e.g. get("estimate.reproj_threshold_px")."""
    node: Any = load(config)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_device(requested: str | None = None) -> str:
    """Resolve "auto" to cuda when a GPU is actually usable, else cpu.

    torch is imported lazily so the classical OpenCV path stays importable (and
    fast to start) on a machine with no torch at all.
    """
    want = (requested or get("device", "auto") or "auto").lower()
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Instrument registry
# ---------------------------------------------------------------------------
def instrument(name: str) -> dict[str, Any]:
    """Look up one instrument's record, case-insensitively."""
    table = load("instruments").get("instruments", {})
    key = name.upper().replace("-", "").replace("_", "")
    for k, v in table.items():
        if k.upper().replace("-", "").replace("_", "") == key:
            return v
    raise ConfigError(f"unknown instrument {name!r}; known: {sorted(table)}")


def gsd_m(name: str) -> float:
    return float(instrument(name)["gsd_m"])


def gsd_ratio(src: str, ref: str) -> float:
    """Reference pixels per source pixel, from the registry.

    This is the scale a correct transform should recover, and the number the
    scale sanity check compares against. OHRC (0.25 m) against IIRS (80 m)
    gives 0.003125, i.e. the 320:1 gap, expressed as src/ref.
    """
    return gsd_m(src) / gsd_m(ref)


def pairing_status(src: str, ref: str) -> dict[str, Any]:
    """What the literature says about this pairing, if anything.

    Used by the report so a result on an already-solved pairing is presented as
    a credibility floor, and a result on an open pairing is presented as new --
    rather than both being shown as if equally novel.
    """
    table = load("instruments").get("pairings", {})
    for key in (f"{src.upper()}-{ref.upper()}", f"{ref.upper()}-{src.upper()}"):
        if key in table:
            return dict(table[key], pairing=key)
    return {"pairing": f"{src.upper()}-{ref.upper()}", "status": "unknown"}


def is_cross_modal(src_instrument: str, ref_instrument: str) -> bool:
    """True when the two instruments do not share a modality.

    Panchromatic vs hyperspectral is the case RIFT/MIND-family descriptors
    exist for; panchromatic vs panchromatic is not.
    """
    a = instrument(src_instrument).get("modality", "unknown")
    b = instrument(ref_instrument).get("modality", "unknown")
    return a != b
