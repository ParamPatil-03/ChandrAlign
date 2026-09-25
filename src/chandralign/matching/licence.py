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
    # Audit 2026-09-26 (I-06): these passed the old denylist.
    "master": "NAVER MASt3R weights: CC BY-NC-SA 4.0",
    "duster": "NAVER DUSt3R weights: CC BY-NC-SA 4.0",
    "gim-lightglue": "loads Magic Leap superpoint_v1.pth (non-commercial)",
    "omniglue": "uses SuperPoint sp_v6 (Magic Leap, non-commercial)",
    "romav2": "DINOv3 backbone under Meta's DINOv3 licence (not OSI); benchmark only",
}

# Our own methods, not vismatch models: nothing to license beyond this repository.
OWN_METHODS = ("sift", "akaze", "orb", "brisk", "rift2", "rift2-mim", "mind")


def allowlist() -> list[str]:
    """Audit I-06: the ONLY names ship mode accepts -- models with a passing row in
    reports/licence_audit.json, configs/regimes.yaml `shippable_matchers`, and our own methods.

    The component denylist above could only refuse what someone had thought to list: on
    2026-09-26 it passed master, duster (CC BY-NC-SA), gim-lightglue and omniglue (SuperPoint
    weights). An allowlist refuses by default; adding a model needs its licence audited first.
    """
    return sorted(set(_audited_pass()) | {str(n) for n in (config.load("regimes").get("shippable_matchers", []) or [])}
                  | set(OWN_METHODS))


def _audited_pass() -> list[str]:
    """Models with a `verdict: pass` row in reports/licence_audit.json (artefact-at-a-version audit)."""
    import json
    try:
        d = json.loads((config.ROOT / "reports" / "licence_audit.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [c["model"] for c in d.get("candidates", []) if isinstance(c, dict) and c.get("verdict") == "pass"]

# WHY eloftr AND matchanything ARE **NOT** ON THAT LIST -- read before adding them.
#
# On 2026-09-23 they were added, and it was WRONG. The reasoning failed in a way
# worth keeping, because it is the failure this whole module exists to prevent,
# one level up: a licence was read off the CURRENT UPSTREAM REPOSITORY and
# applied to the ARTEFACT WE INSTALL, which is a different thing, released at a
# different time, under a different licence.
#
# zju3dv relicensed EfficientLoFTR and MatchAnything from Apache-2.0 to the
# Project Registration License on 2026-09-15 (commits 07e9c14 and 8cd8c11,
# "Adopt Project Registration License v1.0"). The PRL is not OSI-approved and
# requires registration before organisational use -- so upstream TODAY is
# genuinely restricted, and a future upgrade could bring that in.
#
# But nothing we run comes from upstream today. Apache-2.0 section 2 grants a
# "perpetual ... irrevocable" licence, so a later relicence binds future
# releases and cannot withdraw what was already distributed:
#
#   what we RUN      vismatch 1.3.2, released 2026-08-17 -- a month BEFORE the
#                    relicence -- so its vendored copy was taken under Apache-2.0
#   what we LOAD     HF vismatch/eloftr and vismatch/matchanything-eloftr,
#                    both declaring apache-2.0, last modified 2026-02-10
#
# THE CONTROL IS THEREFORE A VERSION PIN, NOT A BAN. Banning the model would
# have been both wrong and useless: it would discard a permissively licensed
# artefact while doing nothing about the upgrade that would actually import the
# PRL. pyproject.toml pins vismatch, and test_vismatch_version_is_licence_audited
# fails if it moves past the audited release. Full chain, with dates and commit
# SHAs, in reports/licence_audit.json.
#
# THE LESSON, GENERALISED: a licence attaches to an ARTEFACT AT A VERSION, not
# to a project. Record repository, commit, weight source and date -- and read
# the LICENSE text, not an API's summary of it, which is how this got through.


class LicenceRestrictedError(RuntimeError):
    """Raised when a non-redistributable matcher is requested in ship mode."""


def restriction_reason(model_name: str) -> str | None:
    """Return why a model may not ship, or None if it is audited and clean.

    Restricted components first (a clear reason), then the allowlist (refuse by default)."""
    name = model_name.lower().strip()
    for component, reason in RESTRICTED_COMPONENTS.items():
        if component in name:
            return f"{model_name!r} contains {component!r}: {reason}"
    if name not in [n.lower() for n in allowlist()]:
        return (f"{model_name!r} is not on the audited allowlist (configs/regimes.yaml shippable_matchers; "
                f"or a verdict 'pass' row in reports/licence_audit.json -- audit its licence first)")
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


def enable_benchmark_mode(reason: str = "--benchmark-models") -> None:
    """Allow benchmark-only / unaudited models for THIS process (G-03 benchmarks). Every report records
    it (run_record()["ship_mode"] is False), so such a result can never pass as shippable."""
    import warnings
    config.load("default")["ship_mode"] = False
    warnings.warn(f"ship_mode OFF ({reason}): benchmark-only models allowed; results are not shippable",
                  stacklevel=2)


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
