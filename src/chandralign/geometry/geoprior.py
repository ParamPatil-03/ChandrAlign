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


def load(product_id: str) -> GeoPrior:
    """The best-known correction for a product; zero ('system') if none was measured."""
    pid = str(product_id)
    reader = _ohrc if "_ohr_" in pid else _tmc2 if "_tmc_" in pid else _iirs if "_iir_" in pid else None
    got = reader(pid) if reader else None
    if not got or not got[0]:
        return GeoPrior(pid, np.zeros(0), np.zeros(0), np.zeros(0), "system", ["no measured correction held"])
    pts, src = got
    a = np.array(pts, float)
    return GeoPrior(pid, a[:, 0], a[:, 1], a[:, 2], src)
