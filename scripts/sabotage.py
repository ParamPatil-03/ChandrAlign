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
    "pair_tiling": (
        "src/chandralign/matching/pair_tiling.py",
        "tests/test_pair_tiling.py",
        {
            "tile matches left in tile coordinates (source)": (
                "        src_parts.append(sp + [x0, y0])", "        src_parts.append(sp)"),
            "reference offset ignores the margin": (
                "        ref_parts.append(rp + [rx0, ry0])", "        ref_parts.append(rp + [x0, y0])"),
            "reference tile gets no margin": (
                "        ry0, ry1 = max(0, y0 - margin), min(h, y1 + margin)", "        ry0, ry1 = y0, y1"),
            "grid over-splits the window": (
                "    ys = np.linspace(0, h, ny + 1)", "    ys = np.linspace(0, h, ny + 2)"),
            "an unaligned pair is tiled anyway": (
                "    if np.asarray(ref.array).shape[:2] != (h, w):", "    if False:"),
        },
    ),
    "gate_reuse": (
        "src/chandralign/evaluate/control_gates.py",
        "tests/test_pair_tiling.py",
        {
            "perturbation gate ignores the base it is given": (
                "    if base is None:\n        base = _safe(pipeline, src, ref)",
                "    base = _safe(pipeline, src, ref)"),
        },
    ),
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
                '    verified = (all_agree or geometry_agrees) and "label (assumed square)" not in along',
                "    verified = True"),
            "one non-label source is enough to outvote the label": (
                "    geometry_agrees = _agree(list(geometric.values()), tolerance)",
                "    geometry_agrees = len(geometric) >= 1"),
            "an outvoted label is kept in the tolerated range": (
                "    range_vals = list(geometric.values()) if outvoted else a_vals",
                "    range_vals = a_vals"),
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
    "gates": (
        "src/chandralign/evaluate/control_gates.py",
        "tests/test_control_gates.py",
        {
            "a null test passes whatever was accepted": (
                "    if run.inlier_ratio <= limit:", "    if True:"),
            "the perturbation gate always passes": (
                "    if err <= tol:", "    if True:"),
            "an unevaluable perturbation counts as passed": (
                '        return GateResult(name, False, f"cannot evaluate: the baseline registration failed "',
                '        return GateResult(name, True, f"cannot evaluate: the baseline registration failed "'),
            "the identity gate ignores RMSE": (
                "    if rmse > max_rmse:", "    if False:"),
            "shared masks detected by identity only, not by memory": (
                "        if a is b or np.shares_memory(np.asarray(a), np.asarray(b)):",
                "        if a is b:"),
            "a skipped mask check silently disappears": (
                '        results.append(GateResult("masks_independent", False,\n'
                '                                  "not evaluated: no ImagePlanes were given to check"))',
                "        pass"),
            "an ungated result is allowed through": (
                "    if not gates:\n        raise UngatedResultError(",
                "    if False:\n        raise UngatedResultError("),
            "noise ignores the reference's brightness": (
                "    noise = rng.normal(float(ref.mean()), float(ref.std()) or 1.0, np.asarray(src).shape)",
                "    noise = rng.normal(0.0, 1.0, np.asarray(src).shape)"),
        },
    ),
    "cascade": (
        "src/chandralign/matching/cascade.py",
        "tests/test_cascade.py",
        {
            "every step is called feasible": (
                "    ok = px >= threshold\n", "    ok = True\n"),
            "an infeasible direct step is attempted anyway": (
                "    if direct.feasible:\n        return Plan(src.name, ref.name, \"direct\"",
                "    if True:\n        return Plan(src.name, ref.name, \"direct\""),
            "later steps' scale is ignored in error propagation": (
                "            rest_scale *= _local_scale(steps[j].model, points[j])",
                "            rest_scale *= 1.0"),
            "steps are composed in the wrong order": (
                "        total = models.compose(total, s.model)",
                "        total = models.compose(s.model, total)"),
            "an ambiguous dense match is accepted": (
                '    if z < float(config.get("cascade.min_z", 10.0)):',
                "    if False:"),
            "downsampling loses the half-pixel offset": (
                "    return np.array([[1 / f, 0, 0.5 / f - 0.5], [0, 1 / f, 0.5 / f - 0.5], [0, 0, 1]], float)",
                "    return np.array([[1 / f, 0, 0.0], [0, 1 / f, 0.0], [0, 0, 1]], float)"),
            "a failed step is bridged over": (
                "    if missing:\n", "    if False:\n"),
        },
    ),
    "subpixel": (
        "src/chandralign/refine/subpixel.py",
        "tests/test_subpixel.py",
        {
            "phase correlation returns the opposite sign": (
                '    return ShiftEstimate(-float(shift[1]), -float(shift[0]), "phase", quality=float(error))',
                '    return ShiftEstimate(float(shift[1]), float(shift[0]), "phase", quality=float(error))'),
            "the peak fit silently rounds to whole pixels": (
                "    return float(np.clip(0.5 * (cm1 - cp1) / denom, -0.5, 0.5))",
                "    return 0.0"),
            "the Gaussian fit quietly falls back to the parabola": (
                '    if fit == "gaussian" and min(cm1, c0, cp1) > 0:',
                "    if False:"),
            "corners that were never refined are counted": (
                "    use = refined & sane", "    use = sane"),
            "a refinement may relocate a match": (
                "np.hypot(est.dx, est.dy) <= max_move", "True"),
            "iteration adds nothing (first estimate only)": (
                "        d = d + r.d", "        d = d"),
            "iteration resamples in the wrong direction": (
                "        m = np.array([[1.0, 0.0, -d[0]], [0.0, 1.0, -d[1]]], np.float32)",
                "        m = np.array([[1.0, 0.0, d[0]], [0.0, 1.0, d[1]]], np.float32)"),
            "a flat patch yields a number instead of a failure": (
                '        return _failed(method, "no texture: a flat patch has no correlation peak")',
                "        pass"),
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
