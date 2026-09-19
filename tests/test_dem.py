"""DATA-13 (DEM loading) and GEO-03 (slope / aspect)."""
import math
from pathlib import Path

import numpy as np
import pytest

from chandralign.geometry.dem_terrain import MOON_RADIUS_M, horn_gradients, slope_aspect, slope_aspect_from_gradients
from chandralign.io.dem import DemError, DemPatch, dem_patch, read_dem_tile

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests" / "fixtures" / "labels"
LOLA_N = LABELS / "ldem_1024_00n_15n_000_030.lbl"
LOLA_S = LABELS / "ldem_1024_15s_00s_000_030.lbl"
SLDEM_N = LABELS / "sldem2015_512_00n_30n_000_045_float.lbl"
M_PER_DEG = MOON_RADIUS_M * math.pi / 180


# ----------------------------------------------------------------------------- real labels

def test_real_lola_labels():
    for label, lat_range in [(LOLA_N, (0, 15)), (LOLA_S, (-15, 0))]:
        tile = read_dem_tile(label)
        assert tile.kind == "lola" and tile.independent_of_references
        assert tile.scale_to_m == 0.5                         # DN x 0.5 = metres
        assert (tile.lines, tile.samples) == (15360, 30720)
        lat0, lat1, lon0, lon1 = tile.bounds()                # pixel CENTRES, half a pixel inside the edges
        half = 0.5 / 1024
        assert lat0 == pytest.approx(lat_range[0] + half) and lat1 == pytest.approx(lat_range[1] - half)
        assert lon0 == pytest.approx(half) and lon1 == pytest.approx(30 - half)


def test_real_sldem_label_is_flagged_selene_derived():
    tile = read_dem_tile(SLDEM_N)
    assert tile.kind == "sldem2015"
    assert tile.independent_of_references is False            # built with SELENE TC stereo
    assert tile.scale_to_m == 1000.0                          # stored in kilometres
    assert tile.model.resolution_px_per_deg == 512


# ----------------------------------------------------------------------------- synthetic DEM products

RES = 64


def lola_like(tmp_path: Path, name: str, top_lat: float, west_lon: float, dn: np.ndarray,
              unit="METER", sample_type="LSB_INTEGER", bits=16, scale=0.5, product="LDEM_64") -> Path:
    """A LOLA-style PDS3 DEM: IMAGE nested in UNCOMPRESSED_FILE, upper-case name in the label,
    lower-case file on disk, pixel-registered simple cylindrical grid."""
    lines, samples = dn.shape
    upper = f"{product}_{name}".upper()
    text = f'''PDS_VERSION_ID = "PDS3"
PRODUCT_ID = "{upper}"
OBJECT = UNCOMPRESSED_FILE
  FILE_NAME = "{upper}.IMG"
  RECORD_TYPE = FIXED_LENGTH
  RECORD_BYTES = {samples * bits // 8}
  ^IMAGE = "{upper}.IMG"
  OBJECT = IMAGE
    LINES = {lines}
    LINE_SAMPLES = {samples}
    SAMPLE_TYPE = {sample_type}
    SAMPLE_BITS = {bits}
    UNIT = {unit}
    SCALING_FACTOR = {scale}
    OFFSET = 1737400.
  END_OBJECT = IMAGE
END_OBJECT = UNCOMPRESSED_FILE
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"
  MAP_RESOLUTION = {RES} <pix/deg>
  CENTER_LATITUDE = 0. <deg>
  CENTER_LONGITUDE = 180. <deg>
  LINE_PROJECTION_OFFSET = {top_lat * RES - 0.5} <pix>
  SAMPLE_PROJECTION_OFFSET = {(180 - west_lon) * RES - 0.5} <pix>
END_OBJECT = IMAGE_MAP_PROJECTION
END
'''
    label = tmp_path / f"{upper.lower()}.lbl"
    label.write_text(text, encoding="ascii")
    dn.astype("<i2" if sample_type == "LSB_INTEGER" else "<f4").tofile(tmp_path / f"{upper.lower()}.img")
    return label


def two_tiles(tmp_path):
    """North tile lat 0..1, south tile lat -1..0, both lon 20..22, with DN = row-major counters."""
    north = np.arange(RES * 2 * RES, dtype=np.int16).reshape(RES, 2 * RES)
    south = (np.arange(RES * 2 * RES, dtype=np.int16) + 10000).reshape(RES, 2 * RES)
    return (lola_like(tmp_path, "N", 1.0, 20.0, north), lola_like(tmp_path, "S", 0.0, 20.0, south),
            north, south)


def test_nested_label_and_filename_case(tmp_path):
    ln, _, north, _ = two_tiles(tmp_path)
    tile = read_dem_tile(ln)
    assert tile.kind == "lola" and tile.scale_to_m == 0.5
    assert tile.model.pixel_to_latlon(0, 0) == pytest.approx((1.0 - 0.5 / RES, 20.0 + 0.5 / RES))


def test_patch_stitches_across_the_equator(tmp_path):
    ln, ls, north, south = two_tiles(tmp_path)
    patch = dem_patch([ln, ls], bbox=(-0.2, 0.2, 20.5, 21.0), margin_px=0)
    assert np.all(np.isfinite(patch.heights_m))
    assert np.allclose(np.diff(patch.lat), -1 / RES) and np.allclose(np.diff(patch.lon), 1 / RES)
    assert patch.lat[0] >= 0.2 and patch.lat[-1] <= -0.2                  # covers the whole box
    # the two rows either side of the equator come from the right tiles, unscaled offset NOT added
    i_n = int(np.argmin(np.abs(patch.lat - 0.5 / RES)))
    i_s = i_n + 1
    j = int(np.argmin(np.abs(patch.lon - (20.5 + 0.5 / RES))))
    col = int(round((patch.lon[j] - 20.0) * RES - 0.5))
    assert patch.heights_m[i_n, j] == north[RES - 1, col] * 0.5
    assert patch.heights_m[i_s, j] == south[0, col] * 0.5
    assert np.abs(patch.heights_m).max() < 1e5                          # not radius (1.7e6 m)
    assert patch.source == "lola_laser_altimetry" and patch.independent_of_references


def test_patch_sampling(tmp_path):
    ln, ls, _, _ = two_tiles(tmp_path)
    patch = dem_patch([ln, ls], bbox=(-0.2, 0.2, 20.5, 21.0))
    lat, lon = patch.lat[5], patch.lon[7]
    assert patch.sample(lat, lon) == pytest.approx(patch.heights_m[5, 7])
    mid = patch.sample((patch.lat[5] + patch.lat[6]) / 2, lon)
    assert mid == pytest.approx((patch.heights_m[5, 7] + patch.heights_m[6, 7]) / 2)
    assert np.isnan(patch.sample(5.0, 20.7))                             # outside: NaN, never extrapolated


def test_uncovered_box_and_mixed_products_raise(tmp_path):
    ln, ls, north, _ = two_tiles(tmp_path)
    with pytest.raises(DemError, match="cover"):
        dem_patch([ln], bbox=(-0.2, 0.2, 20.5, 21.0))                  # south half missing
    km = lola_like(tmp_path, "K", 1.0, 20.0, north.astype(np.float32), unit="KILOMETER",
                   sample_type="PC_REAL", bits=32, scale=1, product="SLDEM2015_64")
    assert read_dem_tile(km).scale_to_m == 1000.0 and not read_dem_tile(km).independent_of_references
    with pytest.raises(DemError, match="mix"):
        dem_patch([ln, km], bbox=(0.2, 0.4, 20.5, 21.0))


# ----------------------------------------------------------------------------- slope and aspect

def plane_patch(lat0: float, gx: float, gy: float, n: int = 40, res: float = 1024) -> DemPatch:
    """A DemPatch whose heights are an exact plane: gx m/m eastward, gy m/m northward."""
    lat = lat0 - (np.arange(n) - n / 2) / res
    lon = 23.0 + np.arange(n) / res
    east = (lon[None, :] - lon[0]) * M_PER_DEG * np.cos(np.radians(lat[:, None]))
    north = (lat[:, None] - lat0) * M_PER_DEG
    return DemPatch(gx * east + gy * north, lat, lon, res, "test", True, ())


@pytest.mark.parametrize("lat0", [0.0, 45.0, -59.0])
@pytest.mark.parametrize("gx, gy, aspect", [
    (0.1, 0.0, 270.0),      # rises to the east -> faces west
    (0.0, 0.2, 180.0),      # rises to the north -> faces south
    (-0.3, 0.0, 90.0),
    (0.1, 0.1, 225.0),
    (0.0, -0.05, 0.0),
])
def test_plane_slope_within_0_1_deg(lat0, gx, gy, aspect):
    """PLAN.md P1-T09 acceptance: analytic slope within 0.1 deg (here, at three latitudes)."""
    t = slope_aspect(plane_patch(lat0, gx, gy))
    inner = t.slope_deg[1:-1, 1:-1]
    assert np.max(np.abs(inner - math.degrees(math.atan(math.hypot(gx, gy))))) < 0.1
    diff = (t.aspect_deg[1:-1, 1:-1] - aspect + 180) % 360 - 180
    assert np.max(np.abs(diff)) < 0.1


def test_flat_ground_has_no_aspect_and_borders_are_nan():
    t = slope_aspect(plane_patch(0.0, 0.0, 0.0))
    assert np.nanmax(t.slope_deg) == 0.0
    assert np.all(np.isnan(t.aspect_deg))                    # no downhill direction on flat ground
    assert np.all(np.isnan(t.slope_deg[0])) and np.all(np.isnan(t.slope_deg[:, -1]))


def test_missing_height_does_not_leak_a_guess():
    p = plane_patch(0.0, 0.1, 0.0)
    p.heights_m[10, 10] = np.nan
    t = slope_aspect(p)
    assert np.all(np.isnan(t.slope_deg[9:12, 9:12]))          # every window touching it
    assert np.isfinite(t.slope_deg[15, 15])


def test_horn_on_a_known_grid():
    z = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], float)     # rises southward by 1 m per 10 m
    gx, gy = horn_gradients(z, 10.0, 10.0)
    assert gx[1, 1] == 0.0 and gy[1, 1] == pytest.approx(-0.1)
    s, a = slope_aspect_from_gradients(gx, gy)
    assert s[1, 1] == pytest.approx(math.degrees(math.atan(0.1))) and a[1, 1] == pytest.approx(0.0)


# ----------------------------------------------------------------------------- real products

LOLA_DIR = ROOT / "data" / "raw" / "dem" / "lola"
SLDEM_DIR = ROOT / "data" / "raw" / "dem" / "sldem2015"
OHRC_BOX = (-0.45, 0.38, 23.45, 23.60)


def _have(folder: Path) -> list[Path]:
    return [l for l in sorted(folder.glob("*.lbl")) if any(folder.glob(l.stem + ".img"))] if folder.exists() else []


@pytest.mark.skipif(len(_have(LOLA_DIR)) < 2, reason="LOLA tiles not downloaded")
def test_real_lola_patch_under_the_ohrc_scene():
    patch = dem_patch(_have(LOLA_DIR), OHRC_BOX)
    assert np.all(np.isfinite(patch.heights_m))
    assert -9_200 < patch.heights_m.min() and patch.heights_m.max() < 10_800   # Moon's full height range
    t = slope_aspect(patch)
    assert np.nanmax(t.slope_deg) < 60                        # no impossible cliffs from unit errors


@pytest.mark.skipif(len(_have(LOLA_DIR)) < 2 or len(_have(SLDEM_DIR)) < 2, reason="LOLA/SLDEM not downloaded")
def test_real_lola_and_sldem_agree_on_heights():
    lola = dem_patch(_have(LOLA_DIR), OHRC_BOX)
    sldem = dem_patch(_have(SLDEM_DIR), OHRC_BOX)
    diff = sldem.sample(*np.meshgrid(lola.lat[5:-5], lola.lon[5:-5], indexing="ij")) - lola.heights_m[5:-5, 5:-5]
    assert np.nanmedian(np.abs(diff)) < 50                    # metres; unit or offset errors give km
