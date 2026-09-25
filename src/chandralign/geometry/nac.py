"""LRO NAC geometry from LROC's corner metadata (a NAC label carries no corner coordinates).

Moved VERBATIM from scripts/register_tmc2_nac.py (enu, Nac) and scripts/register_ohrc_nac.py
(NacGeo, Shifted) -- audit 2026-09-26 C-01 -- so the product can georeference a NAC reference
and run the validated OHRC -> NAC path. The corner metadata lives in data/pairs/*_lroc_meta.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import config

# ---- the NAC CDR layout (PDS3 label; frozen in docs/tmc2_nac_protocol.md) -----------
LINES, SAMPLES, PIX_OFFSET, SCALE = 52224, 5064, 5064, 3.05185094759972e-05
NULL_BELOW = -32752                                   # label VALID_MINIMUM
WIN_LINES = 5064                                      # TMC-2 -> NAC windows: 5064 x 5064
MOON_R_M = 1737400.0
LROC_META = ("tmc2_nac_lroc_meta.json", "iirs_nac_lroc_meta.json")


def _affine_fit(src, dst):
    sol, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
    A = np.eye(3); A[:2, :] = sol.T; return A  # noqa: E702


def enu(lat, lon, lat0, lon0):
    """Local east/north metres about (lat0, lon0)."""
    k = math.pi / 180 * MOON_R_M
    return np.array([(lon - lon0) * math.cos(math.radians(lat0)) * k, (lat - lat0) * k])


class Nac:
    """One NAC CDR: pixels, and its footprint from LROC's corners."""

    def __init__(self, pid: str, meta: dict, root=None):
        self.pid = pid
        self.m = meta
        self.path = next((Path(root or config.ROOT) / "data/raw/lro/nac").rglob(f"{pid}.IMG"))
        self.px_w = float(meta["scaled_pixel_width"])      # across, sample axis
        self.px_h = float(meta["scaled_pixel_height"])     # along, line axis
        c = {k: (meta[f"{k}_latitude"], meta[f"{k}_longitude"])
             for k in ("upper_left", "upper_right", "lower_left", "lower_right")}
        self.c = c

    def latlon(self, line: float, samp: float) -> tuple[float, float]:
        """Bilinear in the corners: UL = (line 0, sample 0), UR = (0, last sample)."""
        u, v = samp / (SAMPLES - 1), line / (LINES - 1)
        lat = ((1 - u) * (1 - v) * self.c["upper_left"][0] + u * (1 - v) * self.c["upper_right"][0]
               + (1 - u) * v * self.c["lower_left"][0] + u * v * self.c["lower_right"][0])
        lon = ((1 - u) * (1 - v) * self.c["upper_left"][1] + u * (1 - v) * self.c["upper_right"][1]
               + (1 - u) * v * self.c["lower_left"][1] + u * v * self.c["lower_right"][1])
        return float(lat), float(lon)

    def axes(self, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray, bool]:
        """Unit ground vectors of the LINE and SAMPLE axes, and whether the image is
        mirrored. The line axis comes from the 26 km long side (~1 deg from 0.01 deg
        corners); the sample axis is perpendicular, with its sign from the corners."""
        P = {k: enu(*v, lat0, lon0) for k, v in self.c.items()}
        u = (P["lower_left"] + P["lower_right"]) / 2 - (P["upper_left"] + P["upper_right"]) / 2
        u /= np.linalg.norm(u)
        across = (P["upper_right"] + P["lower_right"]) / 2 - (P["upper_left"] + P["lower_left"]) / 2
        v = np.array([-u[1], u[0]])                  # u rotated +90 deg (counter-clockwise)
        if v @ across < 0:
            v = -v
        # an unmirrored image (line axis south, sample axis east) has u x v = +1;
        # a mirrored one (sample axis west) has -1
        mirrored = bool(u[0] * v[1] - u[1] * v[0] < 0)
        return u, v, mirrored

    def window(self, centre_line: int):
        l0 = int(centre_line - WIN_LINES // 2)
        arr = np.memmap(self.path, dtype="<i2", mode="r", offset=PIX_OFFSET, shape=(LINES, SAMPLES))
        raw = np.asarray(arr[l0:l0 + WIN_LINES, :], np.float32)
        valid = raw > NULL_BELOW
        img = np.where(valid, raw * SCALE, 0.0).astype(np.float32)
        if valid.any():
            img[~valid] = float(np.median(img[valid]))
        return l0, img, valid


class NacGeo:
    """NAC line/sample <-> lat/lon from LROC's corners (Nac.latlon), inverted by an
    affine fit in a local metric frame; exact enough for a prior (position is searched)."""

    def __init__(self, nac: Nac, lines: int, samples: int):
        self.nac = nac
        g = np.array([(l, s) for l in np.linspace(0, lines - 1, 7) for s in np.linspace(0, samples - 1, 5)])
        ll = np.array([nac.latlon(l, s) for l, s in g])
        self.lat0, self.lon0 = ll.mean(0)
        e = np.array([enu(la, lo, self.lat0, self.lon0) for la, lo in ll])
        self.fwd = _affine_fit(g[:, ::-1], e)                 # (samp, line) -> (east, north)
        self.inv = np.linalg.inv(self.fwd)

    def to_px(self, lat, lon):
        e = np.array([enu(a, b, self.lat0, self.lon0) for a, b in zip(np.atleast_1d(lat), np.atleast_1d(lon))])
        p = np.c_[e, np.ones(len(e))] @ self.inv.T
        return p[:, 0], p[:, 1]                              # samp (x), line (y)

    def pixel_to_latlon(self, rows, cols):
        rows, cols = np.asarray(rows, float).ravel(), np.asarray(cols, float).ravel()
        ll = np.array([self.nac.latlon(r, c) for r, c in zip(rows, cols)])
        return ll[:, 0], ll[:, 1]


class Shifted:
    """Ground model for fine-frame px: (blocked) frame px -> NAC native px -> lat/lon."""

    def __init__(self, geo, ox, oy, bx=1, by=1):
        self.geo, self.ox, self.oy, self.bx, self.by = geo, ox, oy, bx, by

    def pixel_to_latlon(self, rows, cols):
        rows = (np.asarray(rows, float) + 0.5) * self.by - 0.5 + self.oy
        cols = (np.asarray(cols, float) + 0.5) * self.bx - 0.5 + self.ox
        return self.geo.pixel_to_latlon(rows, cols)



def lroc_metadata(root=None) -> dict:
    """{NAC product id: LROC corner metadata} for every NAC held (data/pairs/*_lroc_meta.json)."""
    out = {}
    for f in LROC_META:
        p = Path(root or config.ROOT) / "data/pairs" / f
        if p.is_file():
            out.update(json.loads(p.read_text(encoding="utf-8"))["products"])
    return out


def nac_ground_model(meta, root=None):
    """A ground model (pixel_to_latlon / latlon_to_pixel) for a NAC product, or None if LROC
    corner metadata for it is not held. What product.run_export needs for a NAC reference."""
    lroc = lroc_metadata(root)
    if meta.product_id not in lroc:
        return None
    return NacGroundModel(NacGeo(Nac(meta.product_id, lroc[meta.product_id], root=root), *meta.array_shape))


class NacGroundModel:
    """NacGeo in the (rows, cols) convention of geometry.projection models."""
    source = "lroc_corners"
    independent_of_references = True

    def __init__(self, geo: "NacGeo"):
        self.geo = geo

    def pixel_to_latlon(self, rows, cols, clip: bool = True):
        return self.geo.pixel_to_latlon(rows, cols)

    def latlon_to_pixel(self, lat, lon, tol_px: float = 0.0, max_iter: int = 0):
        x, y = self.geo.to_px(lat, lon)
        return y, x
