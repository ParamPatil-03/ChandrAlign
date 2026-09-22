"""How big must a footprint be, in coarse pixels, to be matched? (sets cascade.min_footprint_px)

    .venv/Scripts/python scripts/measure_cascade_footprint.py

Needs OHRC and TMC-2 in data/raw/ch2.

The cascade planner (matching/cascade.py) calls a step feasible when the finer
image spans at least cascade.min_footprint_px of the coarser camera's pixels.
This measures that number instead of choosing it.

METHOD. Real OHRC is brought down to TMC-2's resolution (16x16 block average,
~4.9 m) and matched into TMC-2 with MIND over a +-2 km search region -- the
scale of real system-geolocation error (TMC-2's own refinement moved it 5.4 km,
OHRC sits ~2 km off). A full-width template first establishes where each
window truly belongs. Then the template is shrunk to W x W coarse pixels and
re-matched in the SAME region. A trial succeeds only if its peak is
unambiguous (z >= 10) AND lands within 3 px of the full-width answer: a sharp
peak in the wrong place is a failure, not a success.

The threshold is the smallest W that succeeds in EVERY window, with the next
size down reported so the margin is visible.

Two traps avoided (both hit by a first version): the truth template must be
much LARGER than any tested width, or a crop nearly as big as the truth agrees
with it by construction; and the tested widths must extend PAST the value being
checked, or the answer can only ever be the largest width tried.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from chandralign.geometry import projection  # noqa: E402
from chandralign.io import pds_raster  # noqa: E402
from chandralign.io.pds_label import parse_label  # noqa: E402
from chandralign.preprocess.phase_congruency import mind  # noqa: E402

K = 16                      # OHRC -> ~TMC-2 resolution
FULL = 320                  # OHRC window, coarse px (= 5120 OHRC px); truth template ~220 px
SEARCH_KM = 2.0
WIDTHS = [32, 48, 64, 80, 96, 112, 128, 144, 160]
ROW_OFFSETS = [-28000, -14000, 0, 14000, 28000]


def norm(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def match(region_ch, tpl):
    ft = mind(norm(tpl)).astype(np.float32)
    cm = np.mean([cv2.matchTemplate(r, ft[..., i], cv2.TM_CCOEFF_NORMED)
                  for i, r in enumerate(region_ch)], axis=0)
    iy, ix = np.unravel_index(np.argmax(cm), cm.shape)
    z = (cm[iy, ix] - cm.mean()) / (cm.std() + 1e-9)
    return np.array([ix, iy], float), float(z)


def main() -> int:
    ohrc = parse_label(next((ROOT / "data/raw/ch2/ohrc").rglob("*_d_img_d18.xml")))
    tmc = parse_label(next((ROOT / "data/raw/ch2/tmc2").rglob("*_d_img_d18.xml")))
    go, gt = projection.load_grid_model(ohrc), projection.load_grid_model(tmc)
    L, S = ohrc.array_shape
    TL, TS = tmc.array_shape
    search_px = int(SEARCH_KM * 1000 / 5.0)
    results = {w: [] for w in WIDTHS}

    for off in ROW_OFFSETS:
        r0, c0 = L // 2 - FULL * K // 2 + off, S // 2 - FULL * K // 2
        ds = pds_raster.read_raster(ohrc, pds_raster.Window(r0, c0, FULL * K, FULL * K)) \
            .astype(np.float32).reshape(FULL, K, FULL, K).mean(axis=(1, 3))
        g = np.linspace(0, FULL - 1, 15)
        gx, gy = np.meshgrid(g, g)
        lat, lon = go.pixel_to_latlon(r0 + (gy.ravel() + 0.5) * K - 0.5, c0 + (gx.ravel() + 0.5) * K - 0.5)
        tr, tc = gt.latlon_to_pixel(lat, lon)
        A, _ = cv2.estimateAffine2D(np.c_[gx.ravel(), gy.ravel()], np.c_[tc, tr])
        centre_t = A @ np.array([FULL / 2, FULL / 2, 1.0])
        # Rotate OHRC into TMC-2 orientation at TMC-2 scale (placement only; the
        # location answer comes from matching, not from either grid).
        size = FULL + 64
        Ac = A.copy()
        Ac[:, 2] += np.array([size / 2, size / 2]) - centre_t
        warped = cv2.warpAffine(ds, Ac, (size, size), flags=cv2.INTER_AREA)
        mask = cv2.warpAffine(np.ones_like(ds), Ac, (size, size), flags=cv2.INTER_NEAREST) > 0.5

        # Region: +-(search + template) around the grid's placement, plus the
        # ~2 km OHRC offset measured earlier, so the true position is inside it.
        half = search_px + size // 2 + 450
        rc, cc = int(round(centre_t[1])), int(round(centre_t[0]))
        y0, x0 = max(rc - half, 0), max(cc - half, 0)
        y1, x1 = min(rc + half, TL), min(cc + half, TS)
        region = norm(pds_raster.read_raster(tmc, pds_raster.Window(y0, x0, y1 - y0, x1 - x0)).astype(np.float32))
        mr = mind(region).astype(np.float32)
        ch = [mr[..., i] for i in range(mr.shape[-1])]

        def crop(w):
            h0 = size // 2 - w // 2
            c = warped[h0:h0 + w, h0:h0 + w]
            return c if mask[h0:h0 + w, h0:h0 + w].all() else None

        full_tpl = crop(int(FULL / 1.45))              # largest square fully inside the rotated footprint
        truth, z_full = match(ch, full_tpl)
        truth_c = truth + full_tpl.shape[1] / 2.0
        print(f"window row {r0}: full-width z {z_full:.1f}", flush=True)
        for w in WIDTHS:
            t = crop(w)
            loc, z = match(ch, t)
            err = float(np.hypot(*((loc + w / 2.0) - truth_c)))
            ok = z >= 10 and err <= 3
            results[w].append({"row": r0, "z": round(z, 1), "error_px": round(err, 1), "ok": ok})

    print(f"\n{'width (coarse px)':>18} {'km':>6} | successes | per window (z / error px)")
    threshold = None
    for w in WIDTHS:
        rs = results[w]
        n_ok = sum(r["ok"] for r in rs)
        cells = "  ".join(f"{r['z']:>5.1f}/{r['error_px']:<5.1f}{'' if r['ok'] else '*'}" for r in rs)
        print(f"{w:>18} {w * 4.92 / 1000:>6.2f} | {n_ok:>4}/{len(rs)}  | {cells}")
        if n_ok == len(rs) and threshold is None:
            threshold = w
        elif n_ok < len(rs):
            threshold = None                    # must hold for this AND every larger size
    print(f"\nsmallest width that succeeds in every window (and every larger one): {threshold}")
    out = ROOT / "reports/cascade_footprint.json"
    out.write_text(json.dumps({"source": "measured", "method": __doc__.split("METHOD.")[1].split("The threshold")[0].strip(),
                               "search_km": SEARCH_KM, "success": "z >= 10 and within 3 px of the full-width answer",
                               "threshold_px": threshold, "results": {str(k): v for k, v in results.items()}},
                              indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
