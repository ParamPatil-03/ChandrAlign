"""Cross-check each product's declared GSD against the ground it actually covers.

Complements scripts/audit_readers.py, which checks a label against ITSELF (file
size, dtype, scaling, special constants). This asks a question no label can
answer alone: does the stated ground sample distance agree with how much Moon
the product is recorded as covering?

    .venv/Scripts/python scripts/audit_gsd.py

WHY IT EXISTS
Every LRO NAC label we hold declares 0.5 m. Their footprints say otherwise:

    product          rows   label    footprint    implied GSD
    M109080308LC    52224   0.50 m    28.5 km       0.55 m
    M102000149LC    52224   0.50 m    62.2 km       1.19 m
    M106719774LC    52224   0.50 m    66.4 km       1.27 m

0.55 m to 1.27 m is a 2.3x spread across products that all claim the same
number. LRO flew an eccentric orbit, so NAC's real GSD tracks spacecraft
altitude; 0.5 m is the design value at the nominal 50 km, not a measurement of
any particular observation.

WHY IT MATTERS MORE THAN IT LOOKS
The scale sanity check (CHECK-05) compares a fitted transform's scale against
the ratio the instrument metadata predicts. For two NAC products that ratio is
computed as 0.5 / 0.5 = 1.0. For M102014464RC against M109080308LC the true
ratio is about 2.15. So the check would not merely miss the discrepancy -- it
would actively CERTIFY a 2.15x scale error as correct, which is the exact
failure mode (#13) it exists to catch.

It also confounds illumination experiments. That same pair is the widest
illumination gap we hold (81.8 deg against 2.2 deg incidence), so a failure to
match it reads as "extreme illumination breaks matching" when a 2.15x scale gap
is sitting underneath it. Two causes, one observation.

This is the same lesson Member A recorded for Chandrayaan-2 -- OHRC is 0.30 m
not 0.25, IIRS is 97.15 m not 80 -- with a wider spread, and with the added
wrinkle that here the LABEL repeats the nominal figure rather than correcting
it, so trusting the label is not enough.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chandralign.io.pds_label import parse_label  # noqa: E402

MOON_RADIUS_KM = 1737.4
KM_PER_DEG = 2.0 * 3.141592653589793 * MOON_RADIUS_KM / 360.0
DISAGREEMENT_WARN = 0.10           # 10% is already worth knowing about


def nac_products() -> list[tuple[str, Path]]:
    base = ROOT / "data" / "raw" / "lro" / "nac"
    out = []
    for d in sorted(base.glob("*")):
        label = next((p for p in d.glob("M*[CE].XML") if "_PYR" not in p.name), None)
        if label is not None:
            out.append((d.name, label))
    return out


def footprints() -> dict:
    path = ROOT / "data" / "pairs" / "lro_nac_candidates.json"
    if not path.exists():
        return {}
    return {e["product_id"]: e
            for e in json.loads(path.read_text(encoding="utf-8"))["results"]}


def main() -> int:
    products = nac_products()
    if not products:
        print("no NAC products held; run scripts/fetch_lro.py --download first")
        return 0
    catalogue = footprints()
    if not catalogue:
        print("no ODE footprint catalogue; run scripts/fetch_lro.py to build it")
        return 0

    print(f"{'product':<16}{'rows':>7}{'label':>8}{'footprint':>11}{'implied':>9}"
          f"{'disagree':>10}")
    print("-" * 61)
    rows = []
    for pid, label_path in products:
        entry = catalogue.get(pid)
        if entry is None:
            print(f"{pid:<16}  not in the footprint catalogue, skipped")
            continue
        meta = parse_label(label_path)
        b = [float(x) for x in entry["bbox"]]
        ground_km = (b[1] - b[0]) * KM_PER_DEG
        n_rows = meta.array_shape[0]
        implied_m = ground_km * 1000.0 / n_rows
        disagreement = abs(implied_m - meta.gsd_m) / meta.gsd_m
        flag = "  <-- " if disagreement > DISAGREEMENT_WARN else ""
        print(f"{meta.product_id:<16}{n_rows:>7}{meta.gsd_m:>7.2f}m{ground_km:>9.1f}km"
              f"{implied_m:>8.2f}m{disagreement*100:>9.0f}%{flag}")
        rows.append((meta.product_id, meta.gsd_m, implied_m, disagreement))

    if not rows:
        return 0
    implied = [r[2] for r in rows]
    spread = max(implied) / min(implied)
    bad = [r for r in rows if r[3] > DISAGREEMENT_WARN]
    print(f"\nimplied GSD spans {min(implied):.2f} m to {max(implied):.2f} m "
          f"= {spread:.2f}x, across products all declaring the same value")
    print(f"{len(bad)}/{len(rows)} disagree with their label by more than "
          f"{DISAGREEMENT_WARN*100:.0f}%")
    if spread > 1.5:
        print("\nCONSEQUENCE: a NAC<->NAC scale ratio computed from the labels is 1.0 "
              "by construction.\nThe scale sanity check (CHECK-05) would certify a "
              f"{spread:.1f}x error rather than catch it,\nand any illumination result "
              "on such a pair is confounded by scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
