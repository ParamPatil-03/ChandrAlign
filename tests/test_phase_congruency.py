"""PREP-02 (phase congruency) and PREP-03 (MIND).

The acceptance here is NOT PLAN.md P1-T13's gamma test. Measured, a gamma shift
barely dents raw-intensity correlation (+0.90 at exponent 3) because it is
monotonic, so that test cannot show what it was written to show -- and phase
congruency scores LOWER than raw on it. The cases that genuinely break brightness
matching are used instead: the sun moving, and contrast inverting. Both are
measured on real lunar terrain from our own LOLA/SLDEM patch, rendered under the
two sun geometries our own products actually have.
"""
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import ImagePlane
from chandralign.geometry.dem_terrain import Terrain, hillshade, slope_aspect
from chandralign.io.dem import dem_patch, find_tiles
from chandralign.io.pds_label import parse_label
from chandralign.preprocess.phase_congruency import (
    clear_cache,
    mind,
    phase_congruency,
    phase_congruency_cached,
    phase_congruency_plane,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "tests" / "fixtures" / "images"
META = parse_label(ROOT / "tests" / "fixtures" / "labels" / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
CRATER_FIELD = np.load(IMAGES / "ohrc_crater_field.npy").astype(np.float64) / 255.0

# The real sun geometries of two products we hold (from their labels).
OHRC_SUN = (269.8, 7.3)      # azimuth, elevation: low, from the west
TMC2_SUN = (104.3, 44.0)     # high, from the east


def correlation(a, b) -> float:
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def synthetic_terrain(n=192, seed=0, relief_m=25.0, smoothing=12):
    """A bumpy surface and its slope/aspect, so it can be lit from any direction.

    Scaled to LUNAR relief: a first version used 300 m of relief over the same span
    and produced a median slope of 69 degrees -- cliffs, not the Moon, where our own
    LOLA/SLDEM patch measures 1.2-2.3 degrees. Shading saturates to black on slopes
    like that and every descriptor looks bad for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n, n))
    for _ in range(smoothing):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    z *= relief_m / max(z.std(), 1e-9)
    from chandralign.io.dem import DemPatch
    patch = DemPatch(z, np.linspace(0.1, -0.1, n), np.linspace(23.4, 23.6, n), 512, "test", True, ())
    return slope_aspect(patch)


def plane(array):
    return ImagePlane(array=array.astype(np.float32), valid_mask=np.ones(array.shape, bool),
                      shadow_mask=np.zeros(array.shape, bool), gsd_m=0.3, meta=META,
                      preprocess_chain=["tile_minmax"])


# ----------------------------------------------------------------------------- shapes and behaviour

def test_phase_congruency_outputs():
    pc = phase_congruency(CRATER_FIELD)
    assert pc.energy.shape == CRATER_FIELD.shape and pc.energy.dtype == np.float32
    assert 0.0 <= pc.energy.min() and pc.energy.max() == pytest.approx(1.0)
    assert pc.orientation_index.shape == CRATER_FIELD.shape
    assert pc.orientation_index.max() < pc.orientations          # a valid orientation index
    assert len(np.unique(pc.orientation_index)) > 1              # the MIM is not a constant map
    assert pc.label == "phase_congruency_s4_o6"


def test_mind_outputs():
    d = mind(CRATER_FIELD)
    assert d.shape == CRATER_FIELD.shape + (8,) and d.dtype == np.float32
    assert 0.0 <= d.min() and d.max() <= 1.0
    assert np.allclose(d.max(axis=-1), 1.0)                      # each pixel's best neighbour is 1


def test_cache_returns_the_same_result_fast():
    clear_cache()
    first = phase_congruency_cached(CRATER_FIELD)
    again = phase_congruency_cached(CRATER_FIELD)
    assert again is first                                        # served from the cache
    assert np.array_equal(first.energy, phase_congruency(CRATER_FIELD).energy)
    clear_cache()
    assert phase_congruency_cached(CRATER_FIELD) is not first


def test_non_images_are_refused():
    with pytest.raises(ValueError, match="2-D"):
        phase_congruency(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="2-D"):
        mind(np.zeros((9,)))


# ----------------------------------------------------------------------------- the real point

def test_sun_moving_breaks_brightness_but_not_structure():
    """PREP-02/03 acceptance: real terrain, the two real sun geometries we hold."""
    terrain = synthetic_terrain()
    a = hillshade(terrain, *OHRC_SUN)[3:-3, 3:-3]
    b = hillshade(terrain, *TMC2_SUN)[3:-3, 3:-3]
    raw = correlation(a, b)
    pc = correlation(phase_congruency(a).energy, phase_congruency(b).energy)
    md = correlation(mind(a), mind(b))
    assert raw < 0.0                                             # brightness is worse than useless
    assert pc > 0.6 and md > 0.6
    assert pc > raw + 1.0 and md > raw + 1.0


def test_contrast_inversion_is_survived():
    """The crude stand-in for a different sensor: raw flips sign, structure does not."""
    a = CRATER_FIELD
    b = 1.0 - a
    assert correlation(a, b) == pytest.approx(-1.0, abs=1e-6)
    assert correlation(phase_congruency(a).energy, phase_congruency(b).energy) > 0.95
    assert correlation(mind(a), mind(b)) > 0.95


def test_the_two_descriptions_fail_on_different_things():
    """Recorded because it decides which to use: they are complements, not alternatives."""
    terrain = synthetic_terrain(seed=1)
    east = hillshade(terrain, 0.0, 30.0)[3:-3, 3:-3]
    north = hillshade(terrain, 90.0, 30.0)[3:-3, 3:-3]
    pc_azimuth = correlation(phase_congruency(east).energy, phase_congruency(north).energy)
    mind_azimuth = correlation(mind(east), mind(north))
    assert pc_azimuth > mind_azimuth + 0.2                       # PC holds a pure azimuth change

    bright = CRATER_FIELD
    gamma = bright ** 3
    pc_gamma = correlation(phase_congruency(bright).energy, phase_congruency(gamma).energy)
    mind_gamma = correlation(mind(bright), mind(gamma))
    assert mind_gamma > pc_gamma + 0.2                           # MIND holds a response change


def test_the_plans_gamma_criterion_does_not_discriminate():
    """Kept as a measurement, not an acceptance: raw intensity SURVIVES a gamma shift,
    so the plan's 'raw correlation is much lower' premise does not hold."""
    a = CRATER_FIELD
    raw = correlation(a, a ** 3)
    assert raw > 0.80, raw                                       # monotonic: brightness order intact
    assert correlation(phase_congruency(a).energy, phase_congruency(a ** 3).energy) < raw


# ----------------------------------------------------------------------------- hillshade itself

def test_hillshade_is_a_cosine_of_the_incidence_angle():
    flat = Terrain(np.zeros((8, 8)), np.full((8, 8), np.nan), "test", True)
    assert hillshade(flat, 0.0, 90.0) == pytest.approx(1.0)       # sun overhead on level ground
    assert hillshade(flat, 0.0, 30.0) == pytest.approx(0.5, abs=1e-6)
    slope = Terrain(np.full((4, 4), 30.0), np.full((4, 4), 90.0), "test", True)   # faces east
    assert hillshade(slope, 90.0, 60.0) == pytest.approx(1.0)                     # square to the sun
    # a 30 deg slope turned away from a sun 30 deg up is grazed exactly edge-on
    assert hillshade(slope, 270.0, 30.0) == pytest.approx(0.0, abs=1e-9)
    assert hillshade(slope, 270.0, 20.0) == pytest.approx(0.0)                    # and beyond: dark


# ----------------------------------------------------------------------------- the ImagePlane step

def test_phase_congruency_plane():
    p = plane(CRATER_FIELD)
    q = phase_congruency_plane(p)
    assert np.array_equal(p.array, CRATER_FIELD.astype(np.float32))    # input untouched
    assert q.array.shape == p.array.shape and q.array.dtype == np.float32
    assert q.preprocess_chain == ["tile_minmax", "phase_congruency_s4_o6"]
    assert q.valid_mask is not p.valid_mask and q.valid_mask is not q.shadow_mask
    assert q.meta is p.meta and q.tile_origin == p.tile_origin


def test_no_data_does_not_create_structure():
    img = CRATER_FIELD.copy()
    valid = np.ones(img.shape, bool)
    valid[:, :64] = False
    img[:, :64] = 0.0                                             # a hard edge at the boundary
    pc = phase_congruency(img, valid)
    edge_energy = pc.energy[:, 62:66].mean()
    interior = pc.energy[:, 100:200].mean()
    assert edge_energy < 3 * interior                             # filled, not a bright false edge


# ----------------------------------------------------------------------------- real elevation data

REAL_DEM = find_tiles(ROOT / "data" / "raw" / "dem" / "sldem2015")


@pytest.mark.skipif(len(REAL_DEM) < 2, reason="SLDEM tiles not downloaded")
def test_real_lunar_terrain_under_our_own_two_suns():
    """The headline measurement, on real lunar relief rather than synthetic bumps."""
    patch = dem_patch(REAL_DEM, (-0.45, 0.38, 23.45, 23.60))
    terrain = slope_aspect(patch)
    a = hillshade(terrain, *OHRC_SUN)[3:-3, 3:-3]
    b = hillshade(terrain, *TMC2_SUN)[3:-3, 3:-3]
    raw = correlation(a, b)
    pc = correlation(phase_congruency(a).energy, phase_congruency(b).energy)
    md = correlation(mind(a), mind(b))
    assert raw < -0.8, f"expected brightness to invert, got {raw}"
    assert pc > 0.7, pc
    assert md > 0.7, md
