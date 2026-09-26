"""Measure TMC-2's real pixel size from pixels, using OHRC as the ruler.

    .venv/Scripts/python scripts/measure_tmc2_scale.py

Needs the OHRC and TMC-2 products in data/raw/ch2 (PRADAN; see data/manifest.json).

WHY
Three metadata sources disagree about TMC-2 (ch2_tmc_nca_20250207T1102039417):

    label           4.41 m across            (= 5 m design value x 88.2 km / 100 km)
    corners         4.881 x 5.065 m
    dense grid      4.997 x 5.066 m          (tuned against SELENE: not independent)

Metadata cannot settle which is right, so this measures it from the images.

METHOD
OHRC is the ruler: its label, corners and grid agree (0.305 x 0.309 m) and its
grid is system-level, i.e. independent of any reference image. An OHRC window is
block-averaged 32x to ~10 m, rotated into TMC-2's frame using the two grids,
then scaled per axis and template-matched over a ~15 km TMC-2 region using MIND
descriptors. The scale that matches best gives TMC-2's pixel size directly:
for J = diag(1/h) R diag(g), each row of J diag(1/g) has norm 1/h_i, with no
need to know the rotation.

MIND, not raw brightness: the two suns differ by ~166 deg of azimuth and OHRC's
is 7.3 deg above the horizon, so raw correlation finds nothing (z ~5, flat).
That is the regime matching/regime.py routes to MIND, and here MIND gave z 22.9
against phase congruency's 11.1 -- the first real-image support for that band.

RESULT (2026-09-21, five independent OHRC windows, all with z >= 12)

    TMC-2 across-track   4.920 m   (spread 4.877-4.950)
    TMC-2 along-track    5.037 m   (spread 5.015-5.053)

    label 4.41 m      +11.6% off  -> WRONG
    dense grid 4.997  -1.5% off
    fitted-scale resolution is 0.0075 (~0.04 m): the search is on a grid

Also measured: OHRC's system-level geolocation sits 1,980 m (1,953-2,014) from
TMC-2's SELENE-refined grid, near-constant over 17 km of strip -- a systematic
offset in OHRC's position, not noise.
"""
import sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from chandralign.io.pds_label import parse_label
from chandralign.io.evidence import evidence_label
from chandralign.io import pds_raster
from chandralign.geometry import projection
from chandralign.preprocess.phase_congruency import mind
from chandralign.preprocess.resample import warp_affine

B = 2                    # working resolution = 2 TMC-2 pixels
K = 16 * B               # OHRC block factor to reach it
N_DS = 192
SEARCH = 1536
OHRC_GSD = (0.3052, 0.3093)          # verified: label, corners and dense grid agree
G = (OHRC_GSD[0] * K, OHRC_GSD[1] * K)

ohrc = parse_label(evidence_label("OHRC", ROOT))
tmc = parse_label(evidence_label("TMC2", ROOT))
go, gt = projection.load_grid_model(ohrc), projection.load_grid_model(tmc)
L, S = ohrc.array_shape
TL, TS = tmc.array_shape


def norm(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def native_pixel(Alin):
    """(across, along) metres of one NATIVE TMC-2 pixel implied by a linear map
    OHRC_ds px -> working px. Rows of Alin.diag(1/g) have norm 1/h_working."""
    M = Alin @ np.diag([1.0 / G[0], 1.0 / G[1]])
    return (1.0 / np.linalg.norm(M[0])) / B, (1.0 / np.linalg.norm(M[1])) / B


def run_window(row_offset):
    r0 = L // 2 - (N_DS * K) // 2 + row_offset
    c0 = S // 2 - (N_DS * K) // 2
    full = pds_raster.read_raster(ohrc, pds_raster.Window(r0, c0, N_DS * K, N_DS * K)).astype(np.float32)
    ds = full.reshape(N_DS, K, N_DS, K).mean(axis=(1, 3))

    g = np.linspace(0, N_DS - 1, 21)
    xs, ys = np.meshgrid(g, g)
    lat, lon = go.pixel_to_latlon(r0 + (ys.ravel() + 0.5) * K - 0.5, c0 + (xs.ravel() + 0.5) * K - 0.5)
    tr, tc = gt.latlon_to_pixel(lat, lon)
    t_r0 = int(np.clip(round(np.median(tr)) - SEARCH * B // 2, 0, TL - SEARCH * B))
    t_c0 = int(np.clip(round(np.median(tc)) - SEARCH * B // 2, 0, max(0, TS - SEARCH * B)))
    wpx = min(SEARCH * B, TS - t_c0)
    region = pds_raster.read_raster(tmc, pds_raster.Window(t_r0, t_c0, SEARCH * B, wpx)).astype(np.float32)
    H, W = region.shape[0] // B, region.shape[1] // B
    region = norm(region[:H * B, :W * B].reshape(H, B, W, B).mean(axis=(1, 3)))
    mr = mind(region).astype(np.float32)
    region_ch = [mr[..., i] for i in range(mr.shape[-1])]

    A_grid, _ = cv2.estimateAffine2D(np.c_[xs.ravel(), ys.ravel()],
                                     np.c_[(tc - t_c0) / B, (tr - t_r0) / B], method=cv2.LMEDS)
    u, s, vt = np.linalg.svd(A_grid[:, :2])
    rot, base = u @ vt, float(np.mean(s))
    grid_px = native_pixel(A_grid[:, :2])
    pred = A_grid @ np.array([N_DS / 2, N_DS / 2, 1.0])

    def evaluate(sx, sy):
        Alin = np.diag([sx, sy]) @ (rot * base)
        size = int(np.ceil(N_DS * base * max(sx, sy) * 1.5)) + 4
        c_out = np.array([size / 2, size / 2])
        A = np.c_[Alin, c_out - Alin @ np.array([N_DS / 2, N_DS / 2])]
        w = warp_affine(ds, A, (size, size))           # ~unit scale: identical to bilinear
        m = cv2.warpAffine(np.ones_like(ds), A, (size, size), flags=cv2.INTER_NEAREST) > 0.5
        yy, xx = np.where(m)
        y0, y1, x0, x1 = yy.min(), yy.max(), xx.min(), xx.max()
        while not m[y0:y1, x0:x1].all():
            y0, y1, x0, x1 = y0 + 1, y1 - 1, x0 + 1, x1 - 1
        mt = mind(norm(w[y0:y1, x0:x1])).astype(np.float32)
        cmap = np.mean([cv2.matchTemplate(r, mt[..., i], cv2.TM_CCOEFF_NORMED)
                        for i, r in enumerate(region_ch)], axis=0)
        iy, ix = np.unravel_index(np.argmax(cmap), cmap.shape)
        z = (float(cmap[iy, ix]) - float(cmap.mean())) / float(cmap.std() + 1e-9)
        centre = np.array([ix + c_out[0] - x0, iy + c_out[1] - y0])
        return z, Alin, centre

    coarse = [(s_, *evaluate(s_, s_)) for s_ in np.arange(0.90, 1.121, 0.01)]
    s_best, z_iso, _, _ = max(coarse, key=lambda t: t[1])
    fine = [(sx, sy, *evaluate(sx, sy))
            for sx in np.arange(s_best - 0.03, s_best + 0.0301, 0.0075)
            for sy in np.arange(s_best - 0.03, s_best + 0.0301, 0.0075)]
    sx, sy, z, Alin, centre = max(fine, key=lambda t: t[2])
    across, along = native_pixel(Alin)
    offset_m = float(np.hypot(*(centre - pred))) * B * 5.0
    z_label = max(t[1] for t in coarse if abs(t[0] - grid_px[0] / 4.41 * 1.0) < 0.006) \
        if any(abs(t[0] - grid_px[0] / 4.41) < 0.006 for t in coarse) else None
    return dict(row=r0, z_iso=z_iso, z=z, sx=sx, sy=sy, across=across, along=along,
                grid=grid_px, offset_m=offset_m, z_label=z_label)


print(f"{'OHRC row':>9}{'z':>7}{'sx':>7}{'sy':>7}{'across m':>10}{'along m':>9}"
      f"{'grid across':>12}{'grid along':>11}{'offset':>9}", flush=True)
results = []
for off in (-28000, -14000, 0, 14000, 28000):
    t = time.perf_counter()
    r = run_window(off)
    results.append(r)
    print(f"{r['row']:>9}{r['z']:>7.1f}{r['sx']:>7.3f}{r['sy']:>7.3f}{r['across']:>10.3f}{r['along']:>9.3f}"
          f"{r['grid'][0]:>12.3f}{r['grid'][1]:>11.3f}{r['offset_m']:>8.0f}m   ({time.perf_counter() - t:.0f}s)",
          flush=True)

good = [r for r in results if r["z"] >= 10]
print(f"\n{len(good)}/{len(results)} windows with an unambiguous match (z >= 10)")
if good:
    a = np.array([r["across"] for r in good]); l = np.array([r["along"] for r in good])
    print(f"TMC-2 across-track pixel from PIXELS: {a.mean():.3f} m  (spread {a.min():.3f}-{a.max():.3f})")
    print(f"TMC-2 along-track  pixel from PIXELS: {l.mean():.3f} m  (spread {l.min():.3f}-{l.max():.3f})")
    print(f"label 4.41 m is {100 * (a.mean() / 4.41 - 1):+.1f}% off; "
          f"grid {np.mean([r['grid'][0] for r in good]):.3f} m is "
          f"{100 * (a.mean() / np.mean([r['grid'][0] for r in good]) - 1):+.1f}% off")
    o = np.array([r["offset_m"] for r in good])
    print(f"OHRC system geolocation vs TMC-2 refined grid: {o.mean():,.0f} m apart (spread {o.min():,.0f}-{o.max():,.0f})")
