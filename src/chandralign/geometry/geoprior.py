"""MATCH-11 as the pipeline default: the best-known geolocation correction for a CH-2 product.

Every registration starts from a prior position. Raw SYSTEM geolocation is off by kilometres (TMC-2
~4.8 km, IIRS ~12.8 km, OHRC ~2.2 km), and placing windows by it lost real registrations
(docs/illumination_fix_results.md, Q8). Our own registrations have MEASURED those errors against the
PS's references; this module turns them into a prior any registration can use by default:

    prior = geoprior.load(product_id)
    lat, lon = prior.correct(lat_sys, lon_sys, line)       # best-known ground of a pixel

The correction is (east, north) metres, fitted along the strip (Theil-Sen, per axis) from the
committed reports -- TMC-2 vs SELENE TC, IIRS vs the LRO WAC mosaic, OHRC vs LRO NAC -- and held
constant beyond the measured lines (no extrapolated drift). A product with no measurement gets a
zero correction labelled "system". Position is always still SEARCHED by the coarse lock: this only
places the search.

What the research doc's bridge also imagined -- tying TMC-2 and IIRS to OHRC through a shared orbit
-- needs simultaneous acquisitions; the products held are from different dates (2024-03-30,
2024-05-23, 2025-02-07), so it is not implemented or claimed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MOON_R_M = 1737400.0
K = np.pi / 180 * MOON_R_M


@dataclass
class GeoPrior:
    product_id: str
    lines: np.ndarray                       # source lines of the measurements
    east_m: np.ndarray
    north_m: np.ndarray
    source: str                             # which reference / report, or "system"
    notes: list[str] = field(default_factory=list)

    def _fit(self, v):
        if len(self.lines) == 0:
            return 0.0, 0.0
        if len(np.unique(self.lines)) < 2:
            return float(np.median(v)), 0.0
        from scipy.stats import theilslopes
        slope, icpt, _, _ = theilslopes(v, self.lines)
        return float(icpt), float(slope)

    def offset_at(self, line) -> tuple[np.ndarray, np.ndarray]:
        """(east, north) metres at `line` (scalar or array); constant beyond the measured lines."""
        line = np.asarray(line, float)
        if len(self.lines):
            line = np.clip(line, self.lines.min(), self.lines.max())
        (ie, se), (inn, sn) = self._fit(self.east_m), self._fit(self.north_m)
        return ie + se * line, inn + sn * line

    def correct(self, lat, lon, line):
        """System lat/lon of pixels on `line` -> best-known lat/lon."""
        e, n = self.offset_at(line)
        lat = np.asarray(lat, float) + n / K
        return lat, np.asarray(lon, float) + e / (K * np.cos(np.radians(lat)))

    def as_dict(self) -> dict:
        return {"product": self.product_id, "source": self.source, "measurements": int(len(self.lines)),
                "line_range": None if not len(self.lines) else [float(self.lines.min()), float(self.lines.max())],
                "notes": self.notes}


def _read(path):
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _tmc2(pid):
    d = _read("reports/tmc2_tc_registration_height_ref.json") or _read("reports/tmc2_tc_registration.json")
    if not d or pid.split("_d_img")[0] not in str(d.get("pairing", "")):
        return None
    pts = [(w["tmc_row"], w["system_offset_m"]["east"], w["system_offset_m"]["north"]) for w in d["rows"]
           if w.get("status") == "registered" and isinstance(w.get("system_offset_m"), dict)]
    return pts, "SELENE TC (reports/tmc2_tc_registration*.json)"


def _iirs(pid):
    d = _read("reports/iirs_wac_mosaic.json")
    if not d or pid.split("_d_img")[0] not in str(d.get("iirs", "")):
        return None
    pts = [(w["iirs_line0"] + 256, w["results"]["xoftr"]["implied_offset_m"]["east"],
            w["results"]["xoftr"]["implied_offset_m"]["north"]) for w in d["windows"]
           if (w.get("results") or {}).get("xoftr", {}).get("success")]
    return pts, "LRO WAC mosaic (reports/iirs_wac_mosaic.json, xoftr)"


def _ohrc(pid):
    d = _read("reports/ohrc_nac_q8_auto_bridge.json")
    if not d or pid.split("_d_img")[0] not in str(d.get("ohrc", "")):
        return None
    pts = [(w["ohrc_row"], w["system_offset_m"]["east"], w["system_offset_m"]["north"]) for w in d["windows"]
           if (w.get("results") or {}).get("routed", {}).get("success") and isinstance(w.get("system_offset_m"), dict)]
    return pts, "LRO NAC (reports/ohrc_nac_q8_auto_bridge.json, successful windows)"


def _find_label(pid: str) -> Path | None:
    base = pid.split("_d_img")[0]
    raw = ROOT / "data" / "raw" / "ch2"
    if not raw.exists():
        return None
    return next(iter(sorted(raw.rglob(f"{base}*_d_img*.xml"))), None)


def _label_refined(pid: str, label_path=None):
    """Audit I-17: ISRO's own refined-minus-system corner offsets, as a SEARCH prior.

    TMC-2 and IIRS labels carry Refined_Corner_Coordinates adjusted by ISRO against the reference
    named in isda:reference_data_used (SELENE). Refined minus system is ~5.1 km for TMC-2 and
    ~13.2 km for IIRS, which agrees with our own measured offsets. Because it is tuned against a
    reference, it NEVER enters evaluation (projection.geolocation_model keeps prefer="independent"),
    only where the coarse lock starts looking -- the lock then validates the match independently.
    Corners are placed on the first and last lines; the Theil-Sen fit spans the strip between them.
    """
    import re
    from ..io.pds_label import CORNER_ORDER, parse_label, read_corner_sets
    path = Path(label_path) if label_path else _find_label(pid)
    if path is None or not path.exists():
        return None
    try:
        sets = read_corner_sets(path)
        n_lines = int(parse_label(path).array_shape[0])
    except Exception:
        return None
    if "refined" not in sets:
        return None
    m = re.search(r"reference_data_used>\s*([^<\s][^<]*)<", path.read_text(encoding="utf-8", errors="ignore"))
    used = m.group(1).strip() if m else "unknown"
    pts = []
    for name, (la, lo), (lr, lor) in zip(CORNER_ORDER, sets["system"], sets["refined"]):
        line = 0.0 if name.startswith("upper") else float(n_lines - 1)
        pts.append((line, (lor - lo) * K * np.cos(np.radians(la)), (lr - la) * K))
    if max(np.hypot(e, n) for _, e, n in pts) < 1.0:
        return None                                  # refined == system: the label adds nothing
    return pts, f"ISRO refined corners ({used}-adjusted label; search prior only, never evaluation)"


def load(product_id: str, *, label_path=None, use_measured: bool = True) -> GeoPrior:
    """The best-known correction for a product: our MEASURED correction if one is held, else ISRO's
    refined-label correction (I-17), else zero ('system'). `use_measured=False` skips the measured
    tables -- the position of a product nobody has registered yet (leave-one-out evaluation)."""
    pid = str(product_id)
    reader = _ohrc if "_ohr_" in pid else _tmc2 if "_tmc_" in pid else _iirs if "_iir_" in pid else None
    got = reader(pid) if (reader and use_measured) else None
    notes = []
    if not got or not got[0]:
        got = _label_refined(pid, label_path)
        notes = ["no measured correction held" if use_measured else "measured tables skipped (use_measured=False)"]
    if not got or not got[0]:
        return GeoPrior(pid, np.zeros(0), np.zeros(0), np.zeros(0), "system", notes + ["no refined label corners"])
    pts, src = got
    a = np.array(pts, float)
    return GeoPrior(pid, a[:, 0], a[:, 1], a[:, 2], src, notes)
