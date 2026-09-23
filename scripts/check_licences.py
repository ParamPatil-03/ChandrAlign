"""CHECK-10: fail if anything we would ship is GPL/AGPL-licensed, or a restricted model is shippable.

    python scripts/check_licences.py                  # base install
    python scripts/check_licences.py --extras learned # also the optional AI matchers

Two things are checked, because two different things can make the product
unshippable:

1. PACKAGES. The dependencies pyproject.toml declares (plus the named extras) and
   everything they pull in are walked through the INSTALLED distributions'
   metadata, and every licence is classified. GPL and AGPL fail: they would
   impose their terms on the product (PLAN.md 2.3: every dependency permissively
   licensed). LGPL passes with a note -- it is linked, not copied -- and so does
   an unstated licence, which is listed for a human to look at, not guessed.
2. MODELS. The matchers configs/regimes.yaml declares shippable, and the two it
   routes to by default, must contain no restricted model: the weights, not just
   the code, carry licences (superpoint-lightglue is non-commercial). The RAW
   config list is checked -- `licence.shippable()` filters restricted names out,
   so checking its output could never fail.

Exit status 1 on any failure, so CI fails when a GPL package is added.
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_STRONG_COPYLEFT = re.compile(r"\b(A?GPL|GNU (Affero )?General Public License)", re.I)
_WEAK = re.compile(r"\b(LGPL|Lesser General Public|Library General Public)", re.I)
_ORDER = ["ok", "weak-copyleft", "unknown", "forbidden"]

# Packages whose metadata states no licence, looked at by a person. Each entry
# says what the licence is and where that was seen, so it can be re-checked.
REVIEWED = {
    "phasepack": "MIT-style: the permission notice heads every source file "
                 "(phasecong.py, phasecongmono.py, ...); package metadata states none. Reviewed 2026-09-23.",
}


def classify(licence_text: str) -> str:
    """'ok' | 'weak-copyleft' | 'unknown' | 'forbidden' for one licence string.

    An SPDX 'A OR B' expression is judged by its most permissive alternative:
    the licensee may choose it.
    """
    text = (licence_text or "").strip()
    if not text or text.upper() in ("UNKNOWN", "NONE"):
        return "unknown"
    alternatives = re.split(r"\s+OR\s+", text)
    verdicts = []
    for alt in alternatives:
        if _WEAK.search(alt):
            verdicts.append("weak-copyleft")
        elif _STRONG_COPYLEFT.search(alt):
            verdicts.append("forbidden")
        else:
            verdicts.append("ok")
    return min(verdicts, key=_ORDER.index)


def licence_of(dist: metadata.Distribution) -> str:
    """The licence a distribution declares: SPDX expression, then classifiers, then the free-text field."""
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return expr
    classifiers = [c.split("::")[-1].strip() for c in (md.get_all("Classifier") or [])
                   if c.startswith("License ::") and c.split("::")[-1].strip() != "OSI Approved"]
    if classifiers:
        return " OR ".join(classifiers)
    free = (md.get("License") or "").strip()
    return free.splitlines()[0][:120] if free else ""


def _name(req: str) -> str:
    return re.split(r"[\s;<>=!~\[(]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")


def declared(extras: tuple[str, ...] = ()) -> list[str]:
    """chandralign's own dependencies (and extras), from pyproject.toml: what we ship."""
    import tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    reqs = list(project.get("dependencies", []))
    for e in extras:
        if e not in project.get("optional-dependencies", {}):
            raise SystemExit(f"no optional-dependency group {e!r} in pyproject.toml")
        reqs += project["optional-dependencies"][e]
    return sorted({_name(r) for r in reqs})


def closure(roots: list[str]) -> dict[str, metadata.Distribution]:
    """Installed distributions reachable from `roots`, by normalised name.
    Optional (extra-gated) requirements of a dependency are not followed."""
    seen: dict[str, metadata.Distribution] = {}
    todo = list(roots)
    while todo:
        name = todo.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen[key] = dist
        for req in dist.requires or []:
            marker = req.split(";", 1)[1] if ";" in req else ""
            if "extra" in marker:
                continue
            if marker:
                try:
                    from packaging.markers import Marker
                    if not Marker(marker).evaluate():
                        continue
                except Exception:
                    pass
            sub = _name(req)
            if sub and sub not in seen:
                todo.append(sub)
    return seen


def check_models() -> tuple[list[str], int]:
    from chandralign import config
    from chandralign.matching import licence
    cfg = config.load("regimes")
    offered = set(cfg.get("shippable_matchers") or []) | {cfg.get("default_matcher"), cfg.get("cross_modal_matcher")}
    offered.discard(None)
    return sorted(m for m in offered if licence.is_restricted(m)), len(offered)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extras", nargs="*", default=[], help="optional-dependency groups to include")
    args = ap.parse_args()

    failures = []
    roots = declared(tuple(args.extras))
    dists = closure(roots)
    missing = [r for r in roots if r not in dists]
    if missing:
        failures.append(f"declared but not installed, so not checked: {missing}")
    rows = []
    for key in sorted(dists):
        lic = licence_of(dists[key])
        verdict = classify(lic)
        if verdict == "unknown" and key in REVIEWED:
            lic, verdict = f"(reviewed) {REVIEWED[key]}", "ok"
        rows.append((key, dists[key].version, verdict, lic))
        if verdict == "forbidden":
            failures.append(f"{key} {dists[key].version}: {lic}")
    width = max((len(r[0]) for r in rows), default=10)
    for key, ver, verdict, lic in rows:
        flag = {"ok": "  ", "weak-copyleft": "~ ", "unknown": "? ", "forbidden": "XX"}[verdict]
        print(f"{flag} {key:<{width}} {ver:<14} {lic}")

    leaked, n_models = check_models()
    if leaked:
        failures.append(f"restricted models are declared shippable or routed to: {leaked}")
    print(f"\n{len(rows)} packages checked; {n_models} shippable/routed models checked, "
          f"restricted among them: {len(leaked)}")
    print("legend: XX forbidden (GPL/AGPL)  ~ weak copyleft (LGPL, allowed)  ? licence not stated")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS: nothing GPL/AGPL in the dependency closure, and no restricted model ships")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
