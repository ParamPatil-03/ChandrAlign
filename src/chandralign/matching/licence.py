"""Licence gate for matcher selection (CHECK-10 / P2-T01).

WHY THIS IS ENFORCED IN CODE AND NOT IN A DOCUMENT
vismatch ships 71 matchers behind one string argument, and the non-commercial
ones sit directly alongside the permissive ones. `superglue` is one typo away
from `sift-lightglue`. A licence table in a markdown file cannot stop that; an
exception at construction time can.

THE COMPONENT TRAP (the reason this matches substrings, not exact names)
Model names in vismatch are compound: `superpoint-lightglue` is the SuperPoint
DETECTOR feeding the LightGlue MATCHER. LightGlue is Apache-2.0 including its
weights, so the name looks safe -- but SuperPoint's weights are Magic Leap's
non-commercial research licence, so the pair is not redistributable. Matching
on exact names would wave that straight through, and the same applies to every
`minima-superpoint-*` and `*-superpoint-*` variant. So we match on restricted
COMPONENTS appearing anywhere in the name.

Sources: PLAN.md section 2.3 licence table; SuperPoint/SuperGlue are Magic Leap
non-commercial, R2D2 is CC BY-NC-SA 3.0 (share-alike, which would additionally
force our own code open).
"""
from __future__ import annotations

from .. import config

# Restricted COMPONENTS. Any model name containing one of these, anywhere, is
# blocked while ship_mode is true.
RESTRICTED_COMPONENTS: dict[str, str] = {
    "superpoint": "Magic Leap licence: non-commercial research only",
    "superglue": "Magic Leap licence: non-commercial research only",
    "r2d2": "CC BY-NC-SA 3.0: non-commercial AND share-alike",
}


class LicenceRestrictedError(RuntimeError):
    """Raised when a non-redistributable matcher is requested in ship mode."""


def restriction_reason(model_name: str) -> str | None:
    """Return why a model is restricted, or None if it is clean."""
    name = model_name.lower().strip()
    for component, reason in RESTRICTED_COMPONENTS.items():
        if component in name:
            return f"{model_name!r} contains {component!r}: {reason}"
    return None


def is_restricted(model_name: str) -> bool:
    return restriction_reason(model_name) is not None


def assert_allowed(model_name: str, ship_mode: bool | None = None) -> None:
    """Gate a model name. Raises LicenceRestrictedError when ship_mode blocks it.

    ship_mode=None reads configs/default.yaml. Only scripts/bench_external.py
    should ever pass ship_mode=False, and its outputs are written to a directory
    the deliverable bundler excludes.
    """
    if ship_mode is None:
        ship_mode = bool(config.get("ship_mode", True))
    if not ship_mode:
        return
    reason = restriction_reason(model_name)
    if reason is not None:
        raise LicenceRestrictedError(
            f"{reason}. It may be used only for internal benchmarking with "
            f"ship_mode=false (scripts/bench_external.py), never in the shipped "
            f"pipeline. Permissive alternatives: {', '.join(shippable())}."
        )


def shippable() -> list[str]:
    """Matchers declared shippable in configs/regimes.yaml, minus any that the
    component check rejects -- so a mistake in the config cannot defeat the gate.
    """
    names = config.load("regimes").get("shippable_matchers", []) or []
    return [n for n in names if not is_restricted(n)]


def benchmark_only() -> list[str]:
    return list(config.load("regimes").get("benchmark_only_matchers", []) or [])


def audit(names: list[str]) -> dict[str, str]:
    """Report the licence verdict for a list of model names (for LICENSE_AUDIT.md)."""
    return {n: (restriction_reason(n) or "permissive: clear to ship") for n in names}
