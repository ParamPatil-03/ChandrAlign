"""IIRS band quality and the single-plane composite. Owner: Member A (Part 1). Features: PREP-05, PREP-06.

IIRS records 256 infrared bands. Many are too noisy to match on, and no single
band looks like a camera image -- two independent teams measured ZERO inliers
matching raw IIRS cubes (failure mode #4/#5). So we score every band, drop the
bad ones, and blend the rest into one plane.

    selection = product_band_selection(meta)          # score once per product
    plane = iirs_composite_plane(meta, window, selection)
    plane.preprocess_chain    # ['iirs_composite_126of256_snr>=30_refl', 'stretch_p1-99', ...]

Scoring, from the data itself:
    signal = median of the band
    noise  = robust (MAD) spread of a diagonal double difference, which cancels
             smooth terrain and leaves pixel-to-pixel noise
    SNR    = signal / noise
A band is dropped when its SNR is below `min_snr`, when more than
`max_dead_fraction` of its pixels are dead (<= 0), or when it lies beyond
`max_wavelength_nm`. The dead-pixel test is a FRACTION, not "any": across 1,280
real lines only 3 bands exceed 1% dead pixels (bands 0-2, at 9.7%, 4.5%, 1.6%),
while 217 bands carry a handful of scattered dead pixels, as any detector does.

TWO THINGS THIS IS NOT, both easy to overclaim:

1. It is not a panchromatic response. Panchromatic cameras (OHRC, TMC-2, NAC)
   see roughly 400-800 nm; IIRS starts at 800 nm, so there is NO overlap. This
   is a spatial-structure proxy -- an image whose terrain pattern a panchromatic
   matcher can work with -- not a reconstruction of what a pan camera would see.

2. Per-band wavelengths are NOT in the label or the ENVI header. Only the
   overall range (800-5000 nm) is stated, so band -> wavelength is interpolated
   linearly across the 256 bands and marked approximate. The data supports it:
   brightness bottoms out near band 125 (~2.9 um) and then climbs to band 255,
   which is the expected crossover from reflected sunlight to the Moon's own
   thermal emission. Bands past `max_wavelength_nm` (3000 nm by default) are
   excluded because they show TEMPERATURE, not surface appearance.

Measured on the real cube, matching two composites built from disjoint halves of
the bands (same terrain, independent noise -- so only real features can match):
a single band gives 426-811 repeatable matches across five crops, a composite
gives 904-1134. Raw keypoint COUNT is higher for a single noisy band, because
SIFT fires on noise; see the module tests.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from chandralign.contracts import ImagePlane, SceneMeta
from chandralign.io.pds_raster import Window, read_raster
from chandralign.preprocess.radiometric import percentile_stretch

DEFAULT_MIN_SNR = 30.0
DEFAULT_MAX_DEAD_FRACTION = 0.01       # a band with >1% dead pixels is unusable
DEFAULT_MAX_DEAD_BANDS_PER_PIXEL = 0.2  # a pixel dead in >20% of kept bands is not a measurement
DEFAULT_MAX_WAVELENGTH_NM = 3000.0     # beyond this, thermal emission dominates
SAMPLE_WINDOWS = 5                     # windows spread along the strip for scoring
SAMPLE_LINES = 256


class CompositeError(ValueError):
    """The cube cannot be reduced to a usable plane."""


@dataclass(frozen=True)
class BandSelection:
    """Which bands are used, why, and with what weight. Computed once per product."""
    bands: tuple[int, ...]             # kept band indices, ascending
    weights: tuple[float, ...]         # same order, summing to 1
    snr: np.ndarray                    # (n_bands,) for every band, kept or not
    signal: np.ndarray
    noise: np.ndarray
    wavelength_nm: np.ndarray          # approximate, interpolated (see module docstring)
    wavelength_is_approximate: bool
    dead_fraction: np.ndarray          # (n_bands,) share of pixels <= 0
    rejected: dict[str, tuple[int, ...]]   # reason -> band indices
    min_snr: float
    max_wavelength_nm: Optional[float]

    @property
    def label(self) -> str:
        return (f"iirs_composite_{len(self.bands)}of{len(self.snr)}"
                f"_snr>={self.min_snr:g}"
                + ("_refl" if self.max_wavelength_nm else ""))


def band_noise(cube: np.ndarray) -> np.ndarray:
    """Per-band pixel-to-pixel noise, robust to terrain.

    The diagonal double difference z(i+1,j+1) - z(i+1,j) - z(i,j+1) + z(i,j) cancels
    any locally planar surface, so what is left is noise; MAD then ignores the
    craters and edges that survive it.
    """
    if cube.ndim != 3 or min(cube.shape[1:]) < 2:
        raise CompositeError(f"expected a (band, line, sample) cube of at least 2x2, got {cube.shape}")
    d = (cube[:, 1:, 1:] - cube[:, 1:, :-1] - cube[:, :-1, 1:] + cube[:, :-1, :-1]) / 2.0
    med = np.median(d, axis=(1, 2), keepdims=True)
    return 1.4826 * np.median(np.abs(d - med), axis=(1, 2))


def approximate_wavelengths(n_bands: int, wavelength_nm: Optional[tuple[float, float]]) -> tuple[np.ndarray, bool]:
    """Band index -> wavelength, interpolated across the label's stated range.

    Returns (wavelengths, is_approximate). With no stated range the wavelengths are
    all NaN and no wavelength-based rejection can be applied.
    """
    if not wavelength_nm:
        return np.full(n_bands, np.nan), True
    lo, hi = wavelength_nm
    return np.linspace(float(lo), float(hi), n_bands), True


def select_bands(cube: np.ndarray, wavelength_nm: Optional[tuple[float, float]] = None,
                 min_snr: float = DEFAULT_MIN_SNR,
                 max_wavelength_nm: Optional[float] = DEFAULT_MAX_WAVELENGTH_NM,
                 max_dead_fraction: float = DEFAULT_MAX_DEAD_FRACTION,
                 max_bands: Optional[int] = None) -> BandSelection:
    """Score every band and choose the ones worth blending.

    max_bands keeps only the best that many by SNR. Measured on five real crops,
    keeping ~40 beats keeping all by 1-2%, which is too small to fix as a default:
    all passing bands are kept unless a caller asks otherwise.
    """
    cube = np.asarray(cube, dtype=np.float64)
    n_bands = cube.shape[0]
    noise = band_noise(cube)
    signal = np.median(cube, axis=(1, 2))
    snr = np.where(noise > 0, signal / noise, 0.0)
    waves, approximate = approximate_wavelengths(n_bands, wavelength_nm)

    dead_fraction = np.array([float((cube[b] <= 0).mean()) for b in range(n_bands)])
    dead = dead_fraction > max_dead_fraction
    low_snr = snr < min_snr
    thermal = (waves > max_wavelength_nm) if max_wavelength_nm is not None else np.zeros(n_bands, bool)

    keep = ~(dead | low_snr | thermal)
    rejected = {
        "dead_pixels": tuple(np.where(dead)[0].tolist()),
        "low_snr": tuple(np.where(low_snr & ~dead)[0].tolist()),
        "thermal": tuple(np.where(thermal & ~dead & ~low_snr)[0].tolist()),
    }
    bands = np.where(keep)[0]
    if max_bands is not None and len(bands) > max_bands:
        bands = np.sort(bands[np.argsort(-snr[bands])[:max_bands]])
        rejected["beyond_max_bands"] = tuple(sorted(set(np.where(keep)[0].tolist()) - set(bands.tolist())))
    if len(bands) == 0:
        raise CompositeError(
            f"no band passes: {len(rejected['dead_pixels'])} have dead pixels, "
            f"{len(rejected['low_snr'])} are below SNR {min_snr:g}, {len(rejected['thermal'])} are thermal"
        )

    # Inverse-variance weighting of the noise-normalised bands: weight = SNR^2.
    w = snr[bands] ** 2
    return BandSelection(tuple(int(b) for b in bands), tuple(w / w.sum()), snr, signal, noise,
                         waves, approximate, dead_fraction, rejected, min_snr, max_wavelength_nm)


def composite(cube: np.ndarray, selection: BandSelection) -> np.ndarray:
    """Blend the selected bands into one plane, in units of each band's own noise.

    Each band is centred on its median and divided by its noise, so bands of very
    different brightness contribute comparably, then combined with the selection's
    inverse-variance weights.
    """
    cube = np.asarray(cube, dtype=np.float64)
    if cube.shape[0] != len(selection.snr):
        raise CompositeError(f"cube has {cube.shape[0]} bands, the selection was made on {len(selection.snr)}")
    bands = np.asarray(selection.bands)
    weights = np.asarray(selection.weights)[:, None, None]
    centred = cube[bands] - selection.signal[bands][:, None, None]
    scaled = centred / np.maximum(selection.noise[bands][:, None, None], 1e-12)
    return (scaled * weights).sum(axis=0)


def _sample_windows(meta: SceneMeta, count: int, lines: int) -> list[Window]:
    total_lines, samples = meta.array_shape
    height = min(lines, total_lines)
    if total_lines <= height:
        return [Window(0, 0, height, samples)]
    starts = np.linspace(0, total_lines - height, count).astype(int)
    return [Window(int(r), 0, height, samples) for r in np.unique(starts)]


def product_band_selection(meta: SceneMeta, windows: Optional[Sequence[Window]] = None,
                           **kwargs) -> BandSelection:
    """Score bands once for a whole product, from windows spread along the strip.

    Every tile must use the SAME bands and weights, or neighbouring tiles would be
    built from different data and stop being comparable.
    """
    if meta.n_bands <= 1:
        raise CompositeError(f"{meta.product_id} has {meta.n_bands} band; the composite is for IIRS cubes")
    windows = list(windows) if windows is not None else _sample_windows(meta, SAMPLE_WINDOWS, SAMPLE_LINES)
    stacked = np.concatenate([read_raster(meta, w).astype(np.float64) for w in windows], axis=1)
    return select_bands(stacked, meta.wavelength_nm, **kwargs)


def iirs_composite_plane(meta: SceneMeta, window: Window,
                         selection: Optional[BandSelection] = None) -> ImagePlane:
    """One window of an IIRS cube as a single-plane ImagePlane, ready for matching."""
    selection = selection or product_band_selection(meta)
    cube = read_raster(meta, window)
    blended = composite(cube, selection)
    kept = cube[np.asarray(selection.bands)]
    # A pixel counts as a measurement when it is finite in every kept band and
    # alive (> 0) in most of them: scattered single dead pixels are diluted by the
    # blend, but a pixel dead across many bands is not data.
    dead_share = (kept <= 0).mean(axis=0)
    valid = np.all(np.isfinite(kept), axis=0) & (dead_share <= DEFAULT_MAX_DEAD_BANDS_PER_PIXEL)
    array, _, _ = percentile_stretch(blended.astype(np.float32), valid)
    return ImagePlane(
        array=array,
        valid_mask=valid,
        shadow_mask=np.zeros(valid.shape, dtype=bool),
        gsd_m=meta.gsd_m,
        meta=meta,
        tile_origin=(window.row, window.col),
        preprocess_chain=[selection.label, "stretch_p1-99"],
    )
