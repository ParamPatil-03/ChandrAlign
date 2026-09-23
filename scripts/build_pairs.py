"""Record every genuinely overlapping source/reference pair we hold. Feature DATA-08.

    .venv/Scripts/python scripts/build_pairs.py            # write data/pairs/registered_pairs.json
    .venv/Scripts/python scripts/build_pairs.py --print    # show them without writing

Every Chandrayaan-2 product we hold is checked against every reference product we
hold, and the ones that really share ground are written out with the overlap
MEASURED rather than assumed -- area in km^2 and as a fraction of each footprint.

The headline pairing is TMC-2 <-> SELENE TC (~2:1 scale, no published prior art).
Its pairs are listed first, and the summary counts them separately, because
"we have three overlapping pairs" is a claim about THAT pairing specifically.

Overlap comes from geometry.footprint.check_overlap, which densifies the footprint
edges before projecting (a corners-only polygon understated one real overlap by
more than half). Nothing here reads pixels -- only labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chandralign.geometry.footprint import check_overlap
from chandralign.io.pds_label import parse_label
from chandralign.io.reference import load_reference

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "pairs" / "registered_pairs.json"

SOURCE_GLOBS = ["ch2/*/products/*/data/calibrated/*/*_d_img_d18.xml"]
REFERENCE_GLOBS = ["selene/tc/*.lbl", "selene/mi/*.lbl", "lro/nac/*/*.XML", "lro/wac/*/*.XML"]
HEADLINE = ("TMC2", "TC")


# Browse and pyramid sidecars sit beside the real products and carry no Array
# structure. They are skipped by name rather than by swallowing parse errors, so a
# genuinely broken product still fails loudly.
SIDECAR_MARKERS = ("_PYR.", "_BROWSE.", "_THUMB.")


def find(globs: list[str]) -> list[Path]:
    seen: list[Path] = []
    for pattern in globs:
        for path in sorted(RAW.glob(pattern)):
            if any(m in path.name.upper() for m in SIDECAR_MARKERS):
                continue
            seen.append(path)
    return seen


def describe(meta) -> dict:
    return {
        "product_id": meta.product_id,
        "instrument": str(meta.instrument),
        "mission": meta.mission,
        "gsd_m": meta.gsd_m,
        "acquisition_utc": meta.acquisition_utc,
        "label": str(meta.label_path.relative_to(ROOT)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="show", action="store_true", help="do not write the file")
    ap.add_argument("--min-overlap", type=float, default=0.05,
                    help="fraction of the SMALLER footprint that must be shared")
    args = ap.parse_args()

    sources = [parse_label(p) for p in find(SOURCE_GLOBS)]
    references = [parse_label(p) for p in find(REFERENCE_GLOBS)]
    if not sources or not references:
        raise SystemExit(f"nothing to pair: {len(sources)} source(s), {len(references)} reference(s)")

    # Whether each reference is actually LIT, measured from its raw pixels once.
    # Two of our WAC products are night-side (ODE incidence 166 and 179 deg) and
    # read ~0.000 reflectance; a pair against one of those is geometrically real and
    # practically useless, so it is recorded and flagged rather than silently listed.
    lit = {}
    for path in find(REFERENCE_GLOBS):
        try:
            ref = load_reference(path)
            lit[ref.meta.product_id] = ref.sunlit()
        except Exception as exc:                       # noqa: BLE001 - reported, not hidden
            print(f"  could not test illumination for {path.name}: {exc}")

    pairs = []
    for src in sources:
        for ref in references:
            check = check_overlap(src, ref, min_overlap=args.min_overlap)
            if not check.ok:
                continue
            pairs.append({
                "source": describe(src),
                "reference": describe(ref),
                "pairing": f"{src.instrument}<->{ref.instrument}",
                "scale_ratio": round(max(src.gsd_m, ref.gsd_m) / min(src.gsd_m, ref.gsd_m), 3),
                "overlap_km2": round(check.overlap_km2, 2),
                "fraction_of_source": round(check.fraction_of_src, 4),
                "fraction_of_reference": round(check.fraction_of_ref, 4),
                "fraction_of_smaller": round(check.fraction_of_smaller, 4),
                "reason": check.reason,
                "reference_lit": lit.get(ref.product_id, {}).get("lit"),
                "reference_incidence_deg": lit.get(ref.product_id, {}).get("incidence_deg"),
            })

    usable = [p for p in pairs if p["reference_lit"] is not False]
    headline = [p for p in pairs if p["pairing"] == f"{HEADLINE[0]}<->{HEADLINE[1]}"]
    pairs.sort(key=lambda p: (p["pairing"] != f"{HEADLINE[0]}<->{HEADLINE[1]}",
                              -p["overlap_km2"]))
    doc = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_overlap": args.min_overlap,
        "sources_checked": len(sources),
        "references_checked": len(references),
        "pair_count": len(pairs),
        "usable_pair_count": len(usable),
        "unlit_reference_pair_count": len(pairs) - len(usable),
        "headline_pairing": f"{HEADLINE[0]}<->{HEADLINE[1]}",
        "headline_pair_count": len(headline),
        "pairs": pairs,
    }

    print(f"{len(sources)} source(s) x {len(references)} reference(s) -> {len(pairs)} overlapping pair(s)")
    print(f"  of which {HEADLINE[0]}<->{HEADLINE[1]}: {len(headline)}\n")
    for p in pairs:
        print(f"  {p['pairing']:14} {p['source']['product_id'][:34]:34} x "
              f"{p['reference']['product_id'][:30]:30} "
              f"{p['overlap_km2']:9.1f} km2  {p['fraction_of_smaller'] * 100:5.1f}% of the smaller"
              f"{'' if p['reference_lit'] is not False else '   [UNLIT reference]'}")

    if not args.show:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
