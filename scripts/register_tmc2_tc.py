"""Register Chandrayaan-2 TMC-2 to SELENE TC: the first real CH-2 <-> reference result.

    .venv/Scripts/python scripts/register_tmc2_tc.py --tile N03 --windows 1
    .venv/Scripts/python scripts/register_tmc2_tc.py --tile all --windows 3

Needs data/raw/ch2/tmc2 and data/raw/selene/tc (see data/manifest.json).

THE ONE RULE (agreed with Member A): TMC-2's REFINED geolocation -- its refined
corners and its geometry grid -- was fitted by ISRO against SELENE. Using it
anywhere in this registration, or to judge it, would be circular. So:

  prior        TMC-2 SYSTEM corners (spacecraft data only) -> TC map projection
  registration pixels only, from that prior
  judged by    Part 2's own gates: MAGSAC inliers, per-axis scale check,
               quality gate. The refined geolocation is compared AFTERWARDS and
               reported as "agreement with ISRO's own SELENE fit" -- a
               reproduction check, explicitly not independent evidence.

STAGES
  1 prior    system-corner prediction of where each TMC-2 window lands in TC
  2 coarse   MIND template search over +-margin km (system geolocation of a
             CH-2 product can be km off: TMC-2's own refinement moved it 5.4 km)
  3 fine     learned feature matching on the coarse-aligned pair, MAGSAC, then
             the per-axis scale check and the quality gate on the COMPOSED
             TMC-2 -> TC transform
  4 report   system geolocation error (m), TMC-2 pixel size implied by TC, and
             distance from ISRO's refined solution

TC is SIMPLE CYLINDRICAL at 7.403 m/px (at the equator); its cross-track pixel
is 7.403 * cos(lat) on the ground, which the scale check accounts for.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from chandralign.evaluate.run_record import run_record  # noqa: E402
from chandralign.pipeline import stage_flags  # noqa: E402
from chandralign.geometry import projection  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.io.evidence import evidence_label  # noqa: E402
from chandralign.matching import routing  # noqa: E402

# The window registration moved VERBATIM to src/chandralign/workflows/tmc2_tc.py (audit 2026-09-26
# C-01), so the CLI and API run the same code as this script. What this file keeps: the flags,
# window selection, the report, and resolve_matcher (the routing entry point tests spy on).
from chandralign.workflows.tmc2_tc import (  # noqa: E402,F401
    DEM_DIRS, M_PER_DEG, PAIRING_STAGES, TILES, Options, run_window)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="N03", choices=sorted(TILES) + ["all"])
    ap.add_argument("--windows", type=int, default=1, help="TMC-2 windows per tile, spread along-track")
    ap.add_argument("--win", type=int, default=1536, help="TMC-2 window size (px)")
    ap.add_argument("--coarse", type=int, default=2, help="block factor for the coarse search")
    ap.add_argument("--margin-km", type=float, default=7.0)
    ap.add_argument("--rows", type=int, nargs="+", default=None,
                    help="explicit TMC-2 window-centre rows; the tile is chosen by latitude")
    ap.add_argument("--prior-offset-m", type=float, nargs=2, default=(0.0, 0.0), metavar=("EAST", "NORTH"),
                    help="known correction applied to the system prior, so --margin-km can shrink")
    ap.add_argument("--matcher", default=None,
                    help="override the routed matcher; omit to let matching.routing choose")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--tile-px", type=int, default=None,
                    help="split the coarse-aligned pair into tiles no larger than this")
    ap.add_argument("--gate-reuse-base", action="store_true",
                    help="perturbation gate reuses the main registration as its baseline")
    ap.add_argument("--col-offset", type=int, default=0,
                    help="with --rows: move the window this many TMC-2 columns off the strip centre")
    ap.add_argument("--dem", choices=sorted(DEM_DIRS), default="sldem2015",
                    help="height model (tc_dtm: SELENE TC stereo DTM, ~7.4 m, only where a tile is held)")
    ap.add_argument("--parallax-dem", choices=sorted(DEM_DIRS), default="tc_dtm",
                    help="height model for the parallax stage only (falls back to --dem where not held); "
                         "tc_dtm is the adopted default (docs/parallax_protocol.md amendment 3)")
    ap.add_argument("--parallax-height-at", choices=["src", "ref"], default=None,
                    help="where the parallax stage samples h (docs/parallax_height_protocol.md)")
    ap.add_argument("--dump-points", default=None, help="folder: save control points and inliers per window")
    ap.add_argument("--stage", action="append", default=[], metavar="NAME=on|off",
                    help="override a pipeline stage (geometry_filter, uniformity, subpixel); "
                         "the default comes from configs/default.yaml pipeline.*")
    ap.add_argument("--out", default="reports/tmc2_tc_registration.json")
    args = ap.parse_args()
    opts = Options(dem=args.dem, parallax_dem=args.parallax_dem, parallax_height_at=args.parallax_height_at,
                   dump_dir=args.dump_points, root=ROOT)
    if args.dump_points:
        Path(args.dump_points).mkdir(parents=True, exist_ok=True)
    # ALIGN-08: parallax is ON for this pairing (TMC-2 fore/aft views are 26 deg oblique),
    # adopted by docs/parallax_protocol.md amendment 3. Only here: the global default stays
    # off because no other pairing has been measured. --stage parallax=off still wins.
    stages = stage_flags({**PAIRING_STAGES, **{k: v.lower() in ("on", "1", "true") for k, v in
                                               (item.split("=", 1) for item in args.stage)}})
    print(f"pipeline stages: {stages}", flush=True)
    args.matcher, routed_opts, matcher_choice = resolve_matcher(args.matcher)
    print(f"matcher: {args.matcher}  ({matcher_choice['chosen_by']})", flush=True)
    # Routed fine-stage options first, explicit CLI flags on top of them.
    mk = {**routed_opts, "precision": args.precision}
    if args.tile_px:
        mk["tile_px"] = args.tile_px

    tmc = parse_label(evidence_label("TMC2", ROOT))
    sysm = projection.load_corner_model(tmc, corners="system")
    # Stage 4 ONLY: ISRO's dense geometry grid, which it fitted against SELENE. It
    # is the actual refined solution; the refined CORNERS are a 4-point bilinear
    # model stretched over ~800 km and too crude to compare against.
    refm = projection.load_grid_model(tmc)
    assert sysm.independent_of_references and not refm.independent_of_references

    rows_all = np.arange(0, tmc.array_shape[0], 250)
    lat_all, _ = sysm.pixel_to_latlon(rows_all, np.full(rows_all.shape, tmc.array_shape[1] / 2))

    results = []
    if args.rows:
        tcs = {k: parse_label(ROOT / "data/raw/selene/tc" / f"{v}.lbl") for k, v in TILES.items()}
        tcms = {k: projection.load_map_model(m) for k, m in tcs.items()}
        shift_deg = args.prior_offset_m[1] / M_PER_DEG
        for row_c in args.rows:
            la, _ = sysm.pixel_to_latlon(np.array([float(row_c)]), np.array([tmc.array_shape[1] / 2.0]))
            la = float(la[0]) + shift_deg
            key = next((k for k, m in tcms.items()
                        if min(m.pixel_to_latlon([0, tcs[k].array_shape[0] - 1], [0, 0])[0]) <= la
                        <= max(m.pixel_to_latlon([0, tcs[k].array_shape[0] - 1], [0, 0])[0])), None)
            if key is None:
                print(json.dumps({"tmc_row": row_c, "status": f"no TC tile held at lat {la:.3f}"}), flush=True)
                continue
            t = time.perf_counter()
            r = run_window(tmc, sysm, refm, tcs[key], tcms[key], int(row_c), win=args.win,
                           coarse=args.coarse, margin_km=args.margin_km, matcher=args.matcher,
                           device=args.device, prior_offset_m=tuple(args.prior_offset_m),
                           match_kwargs=mk, gate_reuse_base=args.gate_reuse_base, stages=stages,
                           col_offset=args.col_offset, opts=opts)
            r["tile"] = key
            r["seconds"] = round(time.perf_counter() - t, 1)
            results.append(r)
            print(json.dumps(r), flush=True)
    for key in ([] if args.rows else (sorted(TILES) if args.tile == "all" else [args.tile])):
        tc = parse_label(ROOT / "data/raw/selene/tc" / f"{TILES[key]}.lbl")
        tcm = projection.load_map_model(tc)
        la0, la1 = sorted(tcm.pixel_to_latlon([0, tc.array_shape[0] - 1], [0, 0])[0])
        pad = (args.margin_km + 10.0) * 1000.0 / M_PER_DEG
        inside = rows_all[(lat_all > la0 + pad) & (lat_all < la1 - pad)]
        if len(inside) == 0:
            print(f"{key}: no TMC-2 rows safely inside the tile", flush=True)
            continue
        picks = np.linspace(inside.min(), inside.max(), args.windows + 2)[1:-1].astype(int)
        for row_c in picks:
            t = time.perf_counter()
            r = run_window(tmc, sysm, refm, tc, tcm, int(row_c), win=args.win, coarse=args.coarse,
                           margin_km=args.margin_km, matcher=args.matcher, device=args.device,
                           match_kwargs=mk, gate_reuse_base=args.gate_reuse_base, stages=stages,
                           opts=opts)
            r["tile"] = key
            r["seconds"] = round(time.perf_counter() - t, 1)
            results.append(r)
            print(json.dumps(r), flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "measured",
        "pairing": "TMC-2 (ch2_tmc_nca_20250207T1102039417) <-> SELENE TC",
        "prior": "TMC-2 SYSTEM corners only; refined geolocation used for stage-4 comparison only",
        "match_options": {**mk, "gate_reuse_base": bool(args.gate_reuse_base)},
        "pipeline_stages": stages,
        "matcher_choice": matcher_choice,
        "run": run_record(),
        "rows": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def resolve_matcher(explicit: str | None, src: str = "TMC2", ref: str = "TC"):
    """(matcher, fine-stage options, provenance) for this run.

    With no --matcher, the choice comes from matching.routing.choose -- the same
    decision Part 3's CLI and API will use -- instead of a name hard-coded here.
    Until routing.py existed this script named eloftr directly and the regime
    selector had no caller outside the tests. An explicit --matcher still wins,
    and is recorded as an override so a result never claims routing chose it.
    """
    if explicit:
        return explicit, {}, {"route": "direct", "matcher": explicit,
                              "chosen_by": "--matcher override"}
    choice = routing.choose(src, ref)
    if choice.route != "direct" or not choice.model_name:
        raise SystemExit(f"routing sends {src} -> {ref} to route {choice.route!r}, not a direct "
                         f"match; this script only runs the direct TMC-2 -> TC path")
    return choice.model_name, dict(choice.fine_stage_options), choice.as_provenance()


if __name__ == "__main__":
    raise SystemExit(main())
