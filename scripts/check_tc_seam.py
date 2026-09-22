"""Check that SELENE TC tiles N03 and N00 are continuous across their seam (lat 0).

    .venv/Scripts/python scripts/check_tc_seam.py

Result 2026-09-22: continuous to ~3 px (~20 m) N-S, with a consistent +3 px E-W
shift, at every longitude tested including where TMC-2 crosses (23.35-23.85 E).
Rules out tile-to-tile offsets as the cause of anything larger than ~20 m.

Original notes:
Are SELENE TC tiles N03 and N00 continuous across their shared edge (lat 0)?

If the tiles are consistently georeferenced, the last row of N03 and the first
row of N00 are adjacent ground: content does not repeat across the seam and the
best horizontal alignment is zero. A north-south offset between the tiles shows
up as DUPLICATED ground (content from N03's bottom reappears inside N00's top)
or a gap; an east-west offset shows up as a nonzero horizontal shift.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
from chandralign.io.pds_label import parse_label
from chandralign.io import pds_raster
from chandralign.geometry import projection

n03 = parse_label(ROOT / "data/raw/selene/tc/TCO_MAP_02_N03E021N00E024SC.lbl")
n00 = parse_label(ROOT / "data/raw/selene/tc/TCO_MAP_02_N00E021S03E024SC.lbl")
m03, m00 = projection.load_map_model(n03), projection.load_map_model(n00)
L = n03.array_shape[0]
print("N03 last row lat :", float(m03.pixel_to_latlon([L - 1], [0])[0][0]))
print("N00 first row lat:", float(m00.pixel_to_latlon([0], [0])[0][0]))
print("N03 col 0 lon, N00 col 0 lon:", float(m03.pixel_to_latlon([0], [0])[1][0]), float(m00.pixel_to_latlon([0], [0])[1][0]))

BAND = 300                       # rows on each side of the seam
res = []
for c0 in range(4000, 11000, 1400):          # several places along the seam (lon ~22.0-23.7)
    W = 900
    top = pds_raster.read_raster(n03, pds_raster.Window(L - BAND, c0, BAND, W)).astype(np.float32)
    bot = pds_raster.read_raster(n00, pds_raster.Window(0, c0 - 100, BAND, W + 200)).astype(np.float32)
    if (top <= 0).mean() > 0.05 or (bot <= 0).mean() > 0.05:
        continue
    nz = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    # Template: N03's bottom 120 rows. Search: the full N00 top band, shifted
    # horizontally +-100 px. If the tiles are continuous there is NO copy of this
    # template inside N00; a strong match means duplicated ground (overlap).
    tpl = nz(top[-120:, :]).astype(np.float32)
    srch = nz(bot).astype(np.float32)
    cm = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
    iy, ix = np.unravel_index(np.argmax(cm), cm.shape)
    z = (cm[iy, ix] - cm.mean()) / (cm.std() + 1e-9)
    # Continuity check: correlation of the row pair straddling the seam at best
    # horizontal shift, against the typical correlation of adjacent rows inside a tile.
    inside = np.mean([np.corrcoef(top[i], top[i + 1])[0, 1] for i in range(BAND - 20, BAND - 1)])
    best = max(((np.corrcoef(top[-1], bot[0, 100 + s:100 + s + W])[0, 1], s) for s in range(-100, 101)))
    res.append((c0, float(cm[iy, ix]), float(z), int(iy), int(ix) - 100, inside, best[0], best[1]))
    print(f"col {c0:>5}: N03-bottom found inside N00? peak {cm[iy, ix]:.3f} z {z:5.1f} at row {iy:>3} "
          f"(start of N03 bottom block would be row {-120 + BAND - BAND}), dx {ix - 100:+d} | "
          f"adjacent-row corr inside tile {inside:.3f}, across seam {best[0]:.3f} at dx {best[1]:+d}")
