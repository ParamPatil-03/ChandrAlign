"""Break the code on purpose and check the tests notice (P2 test-quality tool).

WHY THIS EXISTS
A passing test suite proves the tests pass. It does not prove they would fail
if the code were wrong, and those are different claims. Member A found a
cross-validator that could be made vacuous by feeding the label's own number
back in, with every test still green; the only thing that exposed it was
deliberately breaking the code to see what still passed.

That risk is highest exactly where we are weakest: a threshold fitted on the
same runs used to score it will pass its own tests by construction. The quality
gate in evaluate/quality.py and the routing bands in matching/regime.py are both
in that category, so both are covered here.

Each sabotage replaces one line with a plausible WRONG behaviour -- the kind a
careless refactor would introduce, not a syntax error -- and the suite is
expected to go red. A sabotage that slips through names a real hole in the
tests, not a curiosity.

    .venv/Scripts/python scripts/sabotage.py
    .venv/Scripts/python scripts/sabotage.py --target regime
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# target -> (source file, test file, {description: (find, replace)})
TARGETS: dict[str, tuple[str, str, dict[str, tuple[str, str]]]] = {
    "regime": (
        "src/chandralign/matching/regime.py",
        "tests/test_regime.py",
        {
            "representation bands do nothing (always raw)": (
                '    bands = config.load("regimes").get("representation_bands") or []',
                '    return "raw", "SABOTAGE"\n'
                '    bands = config.load("regimes").get("representation_bands") or []'),
            "never flag a regime as unsolved": (
                '    unsolved = config.load("regimes").get("unsolved_illumination") or {}',
                '    return "solved", "SABOTAGE"\n'
                '    unsolved = config.load("regimes").get("unsolved_illumination") or {}'),
            "ignore the scale gap (always match directly)": (
                "    if check.extreme:", "    if False:"),
            "an unknown sun silently becomes a matching sun": (
                '        d_az = delta.get("d_azimuth_deg")',
                '        d_az = delta.get("d_azimuth_deg") or 0.0'),
            "stop declaring the evidence as synthetic": (
                "_SYNTHETIC_EVIDENCE = (", '_SYNTHETIC_EVIDENCE = ""\n_UNUSED = ('),
        },
    ),
    "scale": (
        "src/chandralign/estimate/scale.py",
        "tests/test_scale.py",
        {
            "drop the anisotropy check (area only, as before)": (
                "    if expected.verified and not (k_lo_t <= aniso <= k_hi_t):",
                "    if False:"),
            "report an unverified expectation as consistent": (
                '    if expected.verified:\n        return ScaleVerdict(True, "consistent",',
                '    if True:\n        return ScaleVerdict(True, "consistent",'),
            "a single source counts as verified": (
                '    verified = _agree(a_vals, tolerance) and "label (assumed square)" not in along',
                "    verified = True"),
            "ignore the corners (every pixel square)": (
                "    if corners is not None:", "    if False:"),
            "no slack for unverified (trust the label fully)": (
                "        a_lo, a_hi = lo / slack, hi * slack",
                "        a_lo, a_hi = lo / (1.0 + tolerance), hi * (1.0 + tolerance)"),
            "unlimited slack (unverified can never reject)": (
                '    slack = float(config.get("estimate.unverified_scale_slack", 3.0))',
                "    slack = 1e9"),
        },
    ),
    "scale-gate": (
        "src/chandralign/evaluate/quality.py",
        "tests/test_scale.py",
        {
            "an unverified scale no longer caps the tier": (
                '    if scale_status in ("unverified", "abstained"):',
                "    if False:"),
            "an inconsistent scale status is ignored": (
                '    if not scale_ok or scale_status in ("inconsistent", "degenerate"):',
                "    if not scale_ok:"),
        },
    ),
    "quality": (
        "src/chandralign/evaluate/quality.py",
        "tests/test_quality_uniformity.py",
        {
            "average the signals instead of taking the worst": (
                "    worst = max(per_signal.values(), key=TIER_ORDER.index)",
                "    worst = min(per_signal.values(), key=TIER_ORDER.index)"),
            "a rejection carries no failure mode": (
                '    if worst == "REJECTED":',
                '    if False:'),
            "a failed control gate is only a warning": (
                '            return QualityVerdict("REJECTED", signals, "control_gates",\n'
                '                                  [FM_FALSE_CORRESPONDENCE], notes)',
                '            pass'),
            "an unmeasured signal counts as good": (
                '        return "LOW"        # unmeasured cannot justify confidence, but is not fatal',
                '        return "HIGH"'),
            "skip the geometry check": (
                "    if not geom_ok:", "    if False:"),
            "ignore a failed scale check": (
                "    if not scale_ok or scale_status in (\"inconsistent\", \"degenerate\"):",
                "    if False:"),
        },
    ),
}


def run(test_file: str) -> bool:
    """True when the suite is green."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT)
    return r.returncode == 0


def check(target: str) -> list[str]:
    source_rel, test_rel, sabotages = TARGETS[target]
    source = ROOT / source_rel
    original = source.read_text(encoding="utf-8")

    if not run(test_rel):
        print(f"  {test_rel} is already failing; fix that before sabotaging")
        return ["<suite was not green to begin with>"]

    slipped = []
    try:
        for label, (find, replace) in sabotages.items():
            if find not in original:
                print(f"  {label:<52} ANCHOR MISSING (code moved; update this script)")
                slipped.append(f"{label} (anchor missing)")
                continue
            source.write_text(original.replace(find, replace, 1), encoding="utf-8")
            caught = not run(test_rel)
            print(f"  {label:<52} {'caught' if caught else '*** SLIPPED THROUGH ***'}")
            if not caught:
                slipped.append(label)
    finally:
        source.write_text(original, encoding="utf-8")
    return slipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=sorted(TARGETS) + ["all"], default="all")
    args = ap.parse_args()

    targets = sorted(TARGETS) if args.target == "all" else [args.target]
    all_slipped: dict[str, list[str]] = {}
    for target in targets:
        print(f"\n{target}  ({TARGETS[target][0]})")
        all_slipped[target] = check(target)

    total = sum(len(TARGETS[t][2]) for t in targets)
    missed = sum(len(v) for v in all_slipped.values())
    print(f"\n{total - missed}/{total} sabotages caught")
    for target, slipped in all_slipped.items():
        for label in slipped:
            print(f"  UNCAUGHT  {target}: {label}")
    if missed:
        print("\nEach uncaught sabotage is a behaviour no test pins down.")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
