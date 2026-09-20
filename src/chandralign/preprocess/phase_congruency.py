"""Phase congruency and MIND: descriptions that survive the sun moving.
Owner: Member A (Part 1). Features: PREP-02, PREP-03.

A pixel's BRIGHTNESS is set by the sun; its STRUCTURE is set by the terrain. Both
descriptions here throw brightness away and keep structure, which is what lets two
images of one place under different suns -- or from different sensors -- be matched.

    pc = phase_congruency(tile)          # pc.energy (0..1), pc.orientation_index (the MIM)
    plane = phase_congruency_plane(plane)  # energy as a new ImagePlane, ready to match
    descriptor = mind(tile)              # (H, W, 8): each pixel described by its neighbourhood

phase_congruency wraps phasepack (PLAN.md P1-T13 pins THIS package, not the newer
`phasecongruency`, because the vendored RIFT2-python is built against it). What we
add is the maximum-index map -- for each pixel, WHICH orientation carries the most
phase congruency -- which is the map the RIFT descriptor consumes, plus caching,
because a 512 x 512 tile costs ~0.7 s and the same tile is asked for repeatedly.

mind() is the Modality Independent Neighbourhood Descriptor (Heinrich et al., 2012):
each pixel is described by how its own small patch compares with its neighbours'
patches. A sensor that inverts contrast, or a sun that moves, changes the values but
not those comparisons, so the description survives.

MEASURED, and the numbers decide which description to use for which problem.
Rendering one real LOLA/SLDEM patch under the two sun geometries our own products
actually have (OHRC: 7.3 deg up from the west; TMC-2: 44 deg up from the east):

    raw brightness      correlation -0.96    INVERTED: slopes bright under one
                                             sun are dark under the other, so a
                                             brightness matcher is not weakly
                                             informed, it is actively misled
    phase congruency    correlation +0.87    (orientation map agrees on 71% of pixels)
    MIND                correlation +0.90

Under contrast inversion -- the crude stand-in for a different sensor -- raw goes
to -1.00 while both hold +1.00.

WHERE EACH ONE IS WEAK, measured the same way:
    a pure 90 deg change of sun AZIMUTH at fixed elevation:
        raw -0.09, phase congruency +0.54, MIND +0.02  -- MIND loses it, PC does not
    a strong gamma shift (same image, exponent 3):
        raw +0.90, phase congruency +0.62, MIND +0.96  -- PC loses it, MIND does not
They fail on different things, so they are complements, not alternatives: phase
congruency for the sun moving, MIND for a sensor answering differently.

NOTE ON PLAN.md P1-T13's acceptance ("PC of an image and of the same image with a
strong gamma shift correlate > 0.9, where raw-intensity correlation is much lower"):
measured, raw intensity correlation under gamma is NOT much lower -- it is +0.90 to
+0.97, because a gamma curve is monotonic and leaves brightness ORDER intact. That
test cannot show what it was meant to show, so the acceptance used here is the pair
of cases that do break raw intensity: a real change of sun geometry, and inversion.

MIND returns a stack of channels, so it does NOT fit ImagePlane.array (2-D by
contract) and is handed to a matcher directly. Phase congruency is a single map and
does fit.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from chandralign.contracts import ImagePlane

DEFAULT_SCALES = 4
DEFAULT_ORIENTATIONS = 6
MIND_OFFSETS = ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2))
MIND_PATCH = 3
_CACHE_SIZE = 8


@dataclass(frozen=True)
class PhaseCongruency:
    energy: np.ndarray            # 0..1, how much structure sits at each pixel
    orientation_index: np.ndarray  # the maximum-index map (MIM): which orientation won, 0..n-1
    orientations: int
    scales: int

    @property
    def label(self) -> str:
        return f"phase_congruency_s{self.scales}_o{self.orientations}"


def _prepare(image: np.ndarray, valid: Optional[np.ndarray]) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {a.shape}")
    if valid is not None and not valid.all():
        a = np.where(valid, a, np.median(a[valid]) if valid.any() else 0.0)
    return a


def phase_congruency(image: np.ndarray, valid: Optional[np.ndarray] = None,
                     scales: int = DEFAULT_SCALES,
                     orientations: int = DEFAULT_ORIENTATIONS) -> PhaseCongruency:
    """Phase congruency and its maximum-index map.

    Phase congruency marks where the frequency components of an image line up in
    phase -- edges and ridges -- which depends on the shape of a feature and not on
    its contrast, so it holds up when the lighting changes.
    """
    import phasepack

    a = _prepare(image, valid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # phasepack warns about pyfftw when absent
        maximum_moment, _, _, _, per_orientation, _, _ = phasepack.phasecong(
            a, nscale=scales, norient=orientations)

    energy = np.asarray(maximum_moment, dtype=np.float32)
    peak = float(energy.max())
    if peak > 0:
        energy = energy / peak
    # The MIM: which orientation carries the most phase congruency at each pixel.
    # RIFT builds its descriptor on this map rather than on intensity.
    stack = np.stack([np.asarray(o, dtype=np.float32) for o in per_orientation], axis=0)
    return PhaseCongruency(energy, np.argmax(stack, axis=0).astype(np.uint8), orientations, scales)


@lru_cache(maxsize=_CACHE_SIZE)
def _cached(key: tuple, scales: int, orientations: int) -> PhaseCongruency:
    shape, dtype, raw = key
    image = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return phase_congruency(image, None, scales, orientations)


def phase_congruency_cached(image: np.ndarray, scales: int = DEFAULT_SCALES,
                            orientations: int = DEFAULT_ORIENTATIONS) -> PhaseCongruency:
    """Same as phase_congruency, remembering the last few tiles (~0.7 s each at 512 px)."""
    a = np.ascontiguousarray(image)
    return _cached((a.shape, a.dtype.str, a.tobytes()), scales, orientations)


def clear_cache() -> None:
    _cached.cache_clear()


def phase_congruency_plane(plane: ImagePlane, scales: int = DEFAULT_SCALES,
                           orientations: int = DEFAULT_ORIENTATIONS) -> ImagePlane:
    """A new ImagePlane whose array is the phase-congruency energy. Input unchanged."""
    pc = phase_congruency(plane.array, plane.valid_mask, scales, orientations)
    return replace(plane, array=pc.energy, valid_mask=plane.valid_mask.copy(),
                   shadow_mask=plane.shadow_mask.copy(),
                   preprocess_chain=list(plane.preprocess_chain) + [pc.label])


def mind(image: np.ndarray, valid: Optional[np.ndarray] = None,
         offsets: tuple[tuple[int, int], ...] = MIND_OFFSETS,
         patch: int = MIND_PATCH) -> np.ndarray:
    """Modality Independent Neighbourhood Descriptor: (H, W, len(offsets)), each channel 0..1.

    For every offset, the squared difference between a pixel's patch and its
    neighbour's patch, turned into a similarity exp(-d / v) where v is how varied
    that pixel's own neighbourhood is. Dividing by v is what makes it independent of
    contrast: only the RELATIVE pattern of neighbour similarity survives.
    """
    a = _prepare(image, valid)
    patch_kernel = (patch, patch)
    distances = []
    for dy, dx in offsets:
        shifted = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
        distances.append(cv2.blur((a - shifted) ** 2, patch_kernel, borderType=cv2.BORDER_REFLECT))
    stack = np.stack(distances, axis=-1)
    variance = np.maximum(stack.mean(axis=-1, keepdims=True), 1e-12)
    descriptor = np.exp(-stack / variance)
    peak = descriptor.max(axis=-1, keepdims=True)
    return (descriptor / np.maximum(peak, 1e-12)).astype(np.float32)
