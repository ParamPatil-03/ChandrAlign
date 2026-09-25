"""Register two PRODUCTS on the validated path: the product entry point (CLI `register`, API).

Before this (audit 2026-09-26 C-01) the CLI and API registered the top-left 1024 px tile of the
source against the top-left tile of the reference -- unrelated ground -- with SIFT, and none of the
validated machinery ran: the TMC-2 -> SELENE TC headline pair came back REJECTED with 4 inliers
while scripts/register_tmc2_tc.py registers it HIGH, 0.46 px.

Now a product pair is dispatched to the workflow that produced the committed evidence for that
pairing (chandralign.workflows.*, moved verbatim from scripts/register_*.py), with the windows
placed where the two products overlap exactly as those scripts place them. A pairing with no
validated workflow is REFUSED with the reason, never answered with a guess.

Each window comes back as a `WindowRun`: the workflow's evidence record, and -- when the window
reached a registration -- a RegistrationBundle that product.run_export.write_run writes. The
bundle's `src` is the source RESAMPLED onto the reference grid (the frame the registration ran
in); `src_to_product` maps it back to source product pixels; `ref` is the georeferenced
reference window.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .. import config
from ..contracts import ImagePlane, MatchSet, Metrics, RegistrationResult

SUPPORTED = {("TMC2", "TC"): "tmc2_tc", ("OHRC", "NAC"): "ohrc_nac", ("IIRS", "WAC_MOSAIC"): "iirs_wac"}
# The IIRS -> WAC evidence is against LROC's map-projected WAC global mosaic (a clip at
# data/raw/lro/wac_mosaic/, reference "label" = its .json); a raw WAC CDR frame is not validated.
MOSAIC_JSON = "wac_mosaic_100m_clip.json"
OTHER_EVIDENCE = {                           # validated, but only as research scripts so far
    ("IIRS", "WAC"): "scripts/register_iirs_wac.py (validated against the WAC global mosaic: pass "
                     f"data/raw/lro/wac_mosaic/{MOSAIC_JSON} as the reference)",
    ("IIRS", "NAC"): "scripts/register_iirs_nac.py",
    ("TMC2", "NAC"): "scripts/register_tmc2_nac.py",
}

# failure modes (evaluate.failure_log) for a window that never reached a registration
FM_OVERLAP, FM_DISPLACEMENT, FM_FALSE = 10, 7, 12


class UnsupportedPairing(ValueError):
    """The pairing has no validated product workflow; the message says what exists."""


@dataclass
class WindowRun:
    window: dict                             # the workflow's evidence record for this window
    bundle: Any = None                       # pipeline.RegistrationBundle, or None if it never registered
    src_model: Any = None                    # ground model of the source PRODUCT (for exports)
    ref_model: Any = None                    # ground model of the reference, in bundle.ref's frame
    failure: Optional[dict] = None           # when bundle is None: tier, failure modes, why


@dataclass
class ProductRun:
    pairing: str
    workflow: str
    matcher_choice: dict
    overlap: dict
    windows: list[WindowRun] = field(default_factory=list)

    def summary(self) -> dict:
        tiers = [w.bundle.result.confidence_tier if w.bundle is not None else w.failure["confidence_tier"]
                 for w in self.windows]
        return {"pairing": self.pairing, "workflow": self.workflow, "matcher_choice": self.matcher_choice,
                "overlap": self.overlap, "windows": len(self.windows),
                "accepted": sum(t != "REJECTED" for t in tiers),
                "tiers": {t: tiers.count(t) for t in sorted(set(tiers))}}


def register_products(src_label, ref_label, *, windows: int = 3, device: Optional[str] = None,
                      root: Optional[Path] = None,
                      progress: Optional[Callable[[str], None]] = None) -> ProductRun:
    """Register a source product to a reference product on the validated path.

    Raises UnsupportedPairing for a pairing with no validated workflow, and ValueError when the
    two products do not overlap on the ground (failure mode 10).
    """
    from ..io.pds_label import parse_label
    root = Path(root or config.ROOT)
    say = progress or (lambda message: None)
    src = parse_label(src_label)
    ref = _mosaic_meta(Path(ref_label)) if Path(ref_label).name == MOSAIC_JSON else parse_label(ref_label)
    pair = (str(src.instrument), str(ref.instrument))
    if pair not in SUPPORTED:
        known = ", ".join(f"{a} -> {b}" for a, b in SUPPORTED)
        other = OTHER_EVIDENCE.get(pair)
        hint = (f" {pair[0]} -> {pair[1]} has real-data evidence from {other}, which is not yet a "
                f"product workflow." if other else "")
        raise UnsupportedPairing(f"no validated product workflow for {pair[0]} -> {pair[1]}; "
                                 f"validated: {known}.{hint}")
    overlap = _overlap(src, ref, root) if pair[1] != "WAC_MOSAIC" else _mosaic_overlap(src, ref)
    if not overlap["ok"]:
        raise ValueError(f"failure mode {FM_OVERLAP} (insufficient overlap): {overlap['reason']}")
    device = config.resolve_device(device)
    say(f"{pair[0]} -> {pair[1]}: {overlap['reason']}; device {device}")
    run = {"tmc2_tc": _tmc2_tc, "ohrc_nac": _ohrc_nac, "iirs_wac": _iirs_wac}[SUPPORTED[pair]]
    return run(src, ref, windows=windows, device=device, root=root, say=say, overlap=overlap)


# ============================================================================= TMC-2 -> TC

def _tmc2_tc(tmc, tc, *, windows, device, root, say, overlap) -> ProductRun:
    from ..geometry import projection
    from ..matching import routing
    from ..pipeline import stage_flags
    from . import tmc2_tc as wf

    sysm = projection.load_corner_model(tmc, corners="system")
    refm = projection.load_grid_model(tmc)       # stage 4 comparison only (the script's rule)
    tcm = projection.load_map_model(tc)
    stages = stage_flags(wf.PAIRING_STAGES)
    choice = routing.choose("TMC2", "TC")
    matcher, mk = choice.model_name, dict(choice.fine_stage_options)
    opts = wf.Options(root=root)
    run = ProductRun("TMC2 -> TC", "chandralign.workflows.tmc2_tc", choice.as_provenance(), overlap)

    # the script's window placement (--tile K --windows N): rows safely inside the tile
    win, margin_km = 1536, 7.0
    rows_all = np.arange(0, tmc.array_shape[0], 250)
    lat_all, _ = sysm.pixel_to_latlon(rows_all, np.full(rows_all.shape, tmc.array_shape[1] / 2))
    la0, la1 = sorted(tcm.pixel_to_latlon([0, tc.array_shape[0] - 1], [0, 0])[0])
    pad = (margin_km + 10.0) * 1000.0 / wf.M_PER_DEG
    inside = rows_all[(lat_all > la0 + pad) & (lat_all < la1 - pad)]
    if len(inside) == 0:
        raise ValueError(f"failure mode {FM_OVERLAP}: no {tmc.product_id} window lies safely inside "
                         f"{tc.product_id} (a {margin_km} km search margin must fit)")
    picks = np.linspace(inside.min(), inside.max(), windows + 2)[1:-1].astype(int)

    for k, row_c in enumerate(picks, 1):
        say(f"window {k}/{len(picks)}: TMC-2 row {int(row_c)} ({matcher})")
        cap: dict = {}
        t0 = time.perf_counter()
        rec = wf.run_window(tmc, sysm, refm, tc, tcm, int(row_c), win=win, coarse=2, margin_km=margin_km,
                            matcher=matcher, device=device, match_kwargs=mk, stages=stages, opts=opts,
                            capture=cap)
        if rec.get("status") != "registered":
            run.windows.append(WindowRun(rec, failure=_failure(rec, rec.get("status", ""))))
            continue
        bundle = _timed(_tmc2_tc_bundle(tmc, tc, cap), time.perf_counter() - t0, rec.get("match_seconds"))
        run.windows.append(WindowRun(rec, bundle, src_model=sysm, ref_model=tcm))
    return run


def _tmc2_tc_bundle(tmc, tc, cap):
    from ..pipeline import RegistrationBundle
    from . import tmc2_tc as wf
    fr, result, Wf, o_f = cap["fine"], cap["result"], cap["Wf"], cap["o_f"]
    r0, c0 = cap["window_origin"]
    src = ImagePlane(array=cap["src_img"], valid_mask=cap["src_ok"], shadow_mask=np.zeros(cap["src_img"].shape, bool),
                     gsd_m=tc.gsd_m, meta=tmc, preprocess_chain=["tmc2_tc: resampled onto the TC grid"])
    ref = ImagePlane(array=cap["ref_img"], valid_mask=cap["ref_ok"], shadow_mask=np.zeros(cap["ref_img"].shape, bool),
                     gsd_m=tc.gsd_m, meta=tc, tile_origin=(int(o_f[1]), int(o_f[0])))
    # src frame px -> TMC-2 window px -> TMC-2 product px
    to_product = wf.T(c0, r0) @ np.linalg.inv(Wf)
    product_transform = np.asarray(result.model.matrix, float) @ wf.T(-c0, -r0)   # TMC-2 px -> TC px
    heights_at = None
    if fr.parallax is not None and cap.get("parallax_dem") is not None:
        dem, model = cap["parallax_dem"], wf._OffsetModel(cap["tcm"], o_f)
        heights_at = lambda pts: dem.sample(*model.pixel_to_latlon(np.asarray(pts)[:, 1], np.asarray(pts)[:, 0]))  # noqa: E731
    return RegistrationBundle(
        result=_frame_result(result, fr, product_transform, "TMC-2 product px", "SELENE TC product px"),
        delivered=_delivered(fr), tps=fr.tps, parallax=fr.parallax, stages=fr.stages, src=src, ref=ref,
        src_to_product=to_product, ref_to_product=wf.T(float(o_f[0]), float(o_f[1])), heights_at=heights_at)


# ============================================================================= OHRC -> NAC

def _ohrc_nac(ohrc, nacm, *, windows, device, root, say, overlap) -> ProductRun:
    from ..evaluate import control_gates
    from ..geometry import projection
    from ..geometry.nac import Nac, NacGeo, lroc_metadata
    from ..io.dem import find_tiles
    from ..matching import routing
    from ..pipeline import stage_flags
    from . import ohrc_nac as wf

    pid = nacm.product_id
    lroc = lroc_metadata(root)
    if pid not in lroc:
        raise UnsupportedPairing(f"no LROC corner metadata for {pid} (data/pairs/*_lroc_meta.json): "
                                 f"the NAC's geometry is unknown")
    om = projection.load_grid_model(ohrc)
    nac = Nac(pid, lroc[pid], root=root)
    geo = NacGeo(nac, *nacm.array_shape)
    stages = stage_flags({})
    dem_tiles = find_tiles(root / "data" / "raw" / "dem" / "sldem2015")
    bridge = wf.bridge_predictions(root).get(pid)           # the adopted --auto-bridge (protocol Q8)
    choice = routing.choose("OHRC", "NAC")
    run = ProductRun("OHRC -> NAC", "chandralign.workflows.ohrc_nac",
                     {**choice.as_provenance(), "candidates": list(choice.candidates()),
                      "auto_bridge": bridge is not None}, overlap)

    best = wf.place_windows(ohrc, om, nac, geo, nacm.array_shape, wf.WIN, bridge=bridge)
    inside = np.array(sorted(best))
    if len(inside) == 0:
        raise ValueError(f"failure mode {FM_OVERLAP}: no {wf.WIN} px OHRC window lies inside {pid}")
    picks = (np.linspace(inside.min(), inside.max(), windows).astype(int) if len(inside) >= windows else inside)
    picks = [int(inside[np.argmin(np.abs(inside - p))]) for p in picks]

    for k, rc in enumerate(picks, 1):
        say(f"window {k}/{len(picks)}: OHRC row {rc} -> {pid}")
        cap: dict = {}
        t0 = time.perf_counter()
        rec = wf.run_window(ohrc, om, nacm, geo, nac, rc, best[rc], ["routed"], device, stages, dem_tiles,
                            capture=cap)
        if rec.get("status") == "no coarse lock" and bridge is not None:   # Q8: bridge only on failure
            failed, cap = rec.get("coarse"), {}
            rec = wf.run_window(ohrc, om, nacm, geo, nac, rc, best[rc], ["routed"], device, stages, dem_tiles,
                                bridge=bridge, capture=cap)
            rec["coarse_lock_failed"] = failed
        routed = (rec.get("results") or {}).get("routed") or {}
        used = routed.get("used")
        if used not in cap:
            why = routed.get("status") or rec.get("status", "")
            run.windows.append(WindowRun(rec, failure=_failure(rec, why)))
            continue
        c = cap[used]
        bundle = _timed(_ohrc_nac_bundle(ohrc, nacm, c, used, geo), time.perf_counter() - t0, None)
        control_gates.require_gates(bundle.result)
        run.windows.append(WindowRun(rec, bundle, src_model=om, ref_model=_FrameGround(c["ref_model"])))
    _consistency(run.windows)
    return run


class _FrameGround:
    """geometry.nac.Shifted (fine-frame px -> lat/lon) with the `clip` keyword the exporters pass."""
    def __init__(self, shifted):
        self.shifted = shifted

    def pixel_to_latlon(self, rows, cols, clip: bool = True):
        return self.shifted.pixel_to_latlon(rows, cols)


def _ohrc_nac_bundle(ohrc, nacm, c, matcher, geo):
    from ..geometry.nac import Shifted
    from ..pipeline import RegistrationBundle
    from . import ohrc_nac as wf
    fr, q, gates = c["fine"], c["quality"], c["gates"]
    r0, c0 = c["window_origin"]
    bx, by = c["block"]
    c["ref_model"] = Shifted(geo, c["origin"][0], c["origin"][1], bx, by)
    gsd = float(c["gsd"])
    src = ImagePlane(array=c["src"], valid_mask=c["src_ok"], shadow_mask=np.zeros(c["src"].shape, bool),
                     gsd_m=gsd, meta=ohrc, preprocess_chain=["ohrc_nac: resampled onto the NAC fine grid"])
    ref = ImagePlane(array=c["ref"], valid_mask=c["ref_ok"], shadow_mask=np.zeros(c["ref"].shape, bool),
                     gsd_m=gsd, meta=nacm, preprocess_chain=[f"nac: block {bx}x{by} from {tuple(c['origin'])}"])
    to_product = wf.T(c0, r0) @ np.linalg.inv(c["Wf"])
    product_transform = np.asarray(c["T_total"], float) @ wf.T(-c0, -r0)             # OHRC px -> NAC px
    n = int(len(fr.matches.src_pts))
    result = RegistrationResult(
        matches=fr.matches, inlier_mask=fr.first.inlier_mask, model=fr.model,
        metrics=Metrics(rmse_px=fr.rmse_px, inlier_count=fr.inlier_count, inlier_ratio=fr.inlier_ratio,
                        spatial_coverage=fr.coverage, source="measured"),
        confidence_tier=q.tier, gates=gates.gates, failure_modes=list(q.failure_modes),
        notes=list(fr.notes) + list(q.notes),
        provenance={"matcher": matcher, "limiting_signal": q.limiting_signal,
                    "scale_status": c["scale"].status, "mi_check": c.get("mi"), "evidence_matches": n})
    return RegistrationBundle(
        result=_frame_result(result, fr, product_transform, "OHRC product px", "LRO NAC product px"),
        delivered=_delivered(fr), tps=fr.tps, parallax=fr.parallax, stages=fr.stages, src=src, ref=ref,
        src_to_product=to_product,
        ref_to_product=wf.T(c["origin"][0], c["origin"][1]) @ np.linalg.inv(c["D"]))   # fine grid -> NAC px


def _consistency(runs: list[WindowRun], within_m: float = 150.0) -> None:
    """The protocol's product-level rule (docs/ohrc_nac_protocol.md): a window's implied offset
    must lie within 150 m of the median over the product's gate-passing windows (>= 3 needed)."""
    offs = [(w, ((w.window.get("results") or {}).get("routed") or {}).get("implied_offset_m"))
            for w in runs if w.bundle is not None and w.bundle.result.confidence_tier != "REJECTED"]
    offs = [(w, o) for w, o in offs if o]
    med = np.median([[o["east"], o["north"]] for _, o in offs], axis=0) if len(offs) >= 3 else None
    for w in runs:
        o = ((w.window.get("results") or {}).get("routed") or {}).get("implied_offset_m")
        w.window["consistent"] = bool(med is not None and o is not None
                                      and math.hypot(o["east"] - med[0], o["north"] - med[1]) <= within_m)
        w.window["consistency_rule"] = ("within 150 m of the median over >= 3 gate-passing windows"
                                        if med is not None else "not evaluable: fewer than 3 gate-passing windows")


# ============================================================================= IIRS -> WAC mosaic

def _mosaic_meta(json_path: Path):
    """SceneMeta for the WAC global mosaic clip (not a PDS product: a map-projected clip whose
    .json records the projection; provenance hashes the .npy and the .json)."""
    import json
    from ..contracts import SceneMeta
    m = json.loads(json_path.read_text(encoding="utf-8"))
    return SceneMeta(product_id="WAC_GLOBAL_MOSAIC_100M", instrument="WAC_MOSAIC", mission="LRO", gsd_m=100.0,
                     n_bands=1, wavelength_nm=(643.0, 643.0), array_shape=tuple(m["shape"]), dtype="float32",
                     corner_latlon=[], sub_solar_azimuth_deg=None, solar_incidence_deg=None, emission_deg=None,
                     phase_deg=None, acquisition_utc=None, label_path=json_path,
                     raster_path=json_path.with_suffix(".npy"))


def _mosaic_overlap(iirs, mosaic) -> dict:
    return {"ok": True, "reason": "IIRS windows are placed inside the mosaic clip (protocol amendment 3); "
                                  "none fit means no overlap", "overlap_km2": None}


def _iirs_wac(iirs, mosaic, *, windows, device, root, say, overlap) -> ProductRun:
    from ..evaluate import control_gates
    from ..geometry import projection
    from ..io.dem import find_tiles
    from ..matching import routing
    from ..pipeline import stage_flags
    from ..preprocess.iirs_composite import product_band_selection
    from . import iirs_wac as wf

    im = projection.load_corner_model(iirs, corners="system")
    sel = product_band_selection(iirs)
    choice = routing.choose("IIRS", "WAC")
    stages = stage_flags()
    dem_tiles = find_tiles(root / "data" / "raw" / "dem" / "sldem2015")
    mmeta, wimg, wok, geo = wf.load_mosaic(root)
    wac = type("Ref", (), {"product_id": mosaic.product_id})()
    matcher = choice.model_name
    run = ProductRun("IIRS -> WAC_MOSAIC", "chandralign.workflows.iirs_wac", choice.as_provenance(), overlap)
    picks = wf.mosaic_windows(iirs, im, geo, wimg, n_win=windows)
    if not picks:
        raise ValueError(f"failure mode {FM_OVERLAP}: no IIRS window lies safely inside the mosaic clip")
    for k, r0 in enumerate(picks, 1):
        say(f"window {k}/{len(picks)}: IIRS line {r0} ({matcher})")
        cap: dict = {}
        t0 = time.perf_counter()
        rec = wf.run_window(iirs, im, sel, wac, wimg, wok, [geo], r0, [matcher], device, stages, dem_tiles,
                            dict(choice.fine_stage_options), capture=cap)
        r = (rec.get("results") or {}).get(matcher) or {}
        if matcher not in cap:
            run.windows.append(WindowRun(rec, failure=_failure(rec, r.get("status") or rec.get("status", ""))))
            continue
        c = cap[matcher]
        bundle = _timed(_iirs_wac_bundle(iirs, mosaic, c, matcher, wimg, wok), time.perf_counter() - t0, None)
        control_gates.require_gates(bundle.result)
        run.windows.append(WindowRun(rec, bundle, src_model=im, ref_model=_FrameGround(wf.Offset(geo, *c["origin"]))))
    return run


def _iirs_wac_bundle(iirs, mosaic, c, matcher, wimg, wok):
    from ..pipeline import RegistrationBundle
    from . import iirs_wac as wf
    fr, q, gates = c["fine"], c["quality"], c["gates"]
    r0, c0 = c["window_origin"]
    gsd = float(c["gsd"])
    src = ImagePlane(array=c["src"], valid_mask=c["src_ok"], shadow_mask=np.zeros(c["src"].shape, bool),
                     gsd_m=gsd, meta=iirs, preprocess_chain=["iirs composite", "resampled onto the WAC mosaic grid"])
    ref = ImagePlane(array=c["ref"], valid_mask=c["ref_ok"], shadow_mask=np.zeros(c["ref"].shape, bool),
                     gsd_m=gsd, meta=mosaic, preprocess_chain=[f"mosaic clip from {tuple(c['origin'])}"])
    to_product = wf.T(c0, r0) @ np.linalg.inv(c["Wf"])
    product_transform = np.asarray(c["T_total"], float) @ wf.T(-c0, -r0)        # IIRS px -> mosaic clip px
    result = RegistrationResult(
        matches=fr.matches, inlier_mask=fr.first.inlier_mask, model=fr.model,
        metrics=Metrics(rmse_px=fr.rmse_px, inlier_count=fr.inlier_count, inlier_ratio=fr.inlier_ratio,
                        spatial_coverage=fr.coverage, source="measured"),
        confidence_tier=q.tier, gates=gates.gates, failure_modes=list(q.failure_modes),
        notes=list(fr.notes) + list(q.notes),
        provenance={"matcher": matcher, "limiting_signal": q.limiting_signal, "scale_status": c["scale"].status,
                    "mi_check": c.get("mi"), "evidence_matches": int(len(fr.matches.src_pts))})
    return RegistrationBundle(
        result=_frame_result(result, fr, product_transform, "IIRS product px", "WAC mosaic clip px"),
        delivered=_delivered(fr), tps=fr.tps, parallax=fr.parallax, stages=fr.stages, src=src, ref=ref,
        src_to_product=to_product, ref_to_product=wf.T(*c["origin"]))


# ============================================================================= shared

def _frame_result(result, fr, product_transform, frm: str, to: str):
    """The RegistrationResult in the BUNDLE's frames (src frame px -> ref window px), with the
    product-level transform the user wants recorded alongside."""
    prov = dict(result.provenance or {})
    prov["product_transform"] = {"from": frm, "to": to,
                                 "matrix": [[float(v) for v in row] for row in np.asarray(product_transform)]}
    return replace(result, model=fr.model, matches=fr.matches, inlier_mask=fr.first.inlier_mask,
                   provenance=prov)


def _timed(bundle, seconds: float, match_seconds):
    """OUT-08: runtime_s is the window's end-to-end wall time (prior, coarse lock, matching, fine
    stage, all control gates); the matcher's own time is kept in provenance when known."""
    r = bundle.result
    prov = dict(r.provenance or {})
    if match_seconds is not None:
        prov["match_seconds"] = match_seconds
    bundle.result = replace(r, metrics=replace(r.metrics, runtime_s=round(float(seconds), 3)), provenance=prov)
    return bundle


def _delivered(fr) -> MatchSet:
    cs = np.asarray(fr.control_src, float).reshape(-1, 2)
    cr = np.asarray(fr.control_ref, float).reshape(-1, 2)
    return MatchSet(src_pts=cs, ref_pts=cr, confidence=np.ones(len(cs), np.float32),
                    method=fr.matches.method, regime=fr.matches.regime, stage="delivered")


def _failure(rec: dict, why: str) -> dict:
    """A window that never reached a registration: REJECTED, with the failure mode that fits."""
    text = str(why)
    if "coarse" in text or "coarse lock" in text:
        mode = FM_DISPLACEMENT
    elif text.startswith("skipped"):
        mode = FM_OVERLAP
    else:
        mode = FM_FALSE
    return {"confidence_tier": "REJECTED", "failure_modes": [mode], "status": text}


def _overlap(src, ref, root: Path) -> dict:
    from ..geometry.footprint import FootprintError, check_overlap, footprint_of
    from ..io.reference import find_saved_record
    try:
        fp_ref = footprint_of(ref, find_saved_record(ref.raster_path.parent) or None)
        chk = check_overlap(src, fp_ref)
    except FootprintError as exc:
        return {"ok": False, "reason": f"footprint unavailable: {exc}"}
    return {"ok": bool(chk.ok), "reason": chk.reason, "overlap_km2": round(float(chk.overlap_km2), 3),
            "fraction_of_smaller": round(float(chk.fraction_of_smaller), 4)}
