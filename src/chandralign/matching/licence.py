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
    # Found by the 2026-09-26 audit (I-06): these passed the three entries above.
    "master": "NAVER MASt3R weights, CC BY-NC-SA 4.0: non-commercial AND share-alike",
    "duster": "NAVER DUSt3R weights, CC BY-NC-SA 4.0: non-commercial AND share-alike",
    "gim-lightglue": "loads Magic Leap superpoint_v1.pth: non-commercial research only",
    "omniglue": "loads SuperPoint sp_v6 weights: non-commercial research only",
}

# THE DENYLIST ALONE IS NOT THE GATE. vismatch ships ~70 models and a denylist only knows the
# ones someone thought of: four non-redistributable models passed the three entries above
# until 2026-09-26. So in ship mode a model must ALSO be on the audited allowlist,
# configs/regimes.yaml `shippable_matchers`, each entry with its licence chain in
# reports/licence_audit.json. A new model (roma, ufm, ...) is refused until someone audits it.

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

    ship_mode=None reads configs/default.yaml. In ship mode a model must be on the
    audited allowlist (shippable_matchers) AND clear of the restricted components.
    Only benchmarking code that compares candidates (scripts/select_default_matcher.py,
    scripts/bench_rift.py) passes ship_mode=False; nothing it runs is shipped.
    """
    if ship_mode is None:
        ship_mode = bool(config.get("ship_mode", True))
    if not ship_mode:
        return
    reason = restriction_reason(model_name)
    if reason is None and model_name.lower().strip() not in shippable():
        reason = (f"{model_name!r} is not on the audited allowlist (configs/regimes.yaml "
                  f"shippable_matchers); audit its code AND weight licences into "
                  f"reports/licence_audit.json before adding it")
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
