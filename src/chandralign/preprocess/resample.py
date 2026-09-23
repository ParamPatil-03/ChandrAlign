"""Affine resampling that does not alias when it shrinks. Owner: Member B (Part 2).

WHY THIS EXISTS
`cv2.warpAffine` does not implement `INTER_AREA`: asked for it, it silently runs
bilinear (checked: the outputs are identical). Bilinear reads a 2x2 neighbourhood
per output pixel, so a warp that shrinks the image by 1.5x (TMC-2 -> SELENE TC)
or 3x (the coarse template) skips input pixels and aliases fine texture into
false structure. Five call sites asked for area averaging and got none of it,
while the image they were compared against was properly block-averaged -- the two
sides of a match were filtered differently.

WHAT IT DOES
When the warp shrinks the image, a Gaussian pre-blur removes what the output grid
cannot represent, then the warp runs bilinear. The blur follows scikit-image's
anti-aliasing rule, sigma = (1/s - 1) / 2 for a reduction by s, per axis of the
warp's linear part (its singular values), so a warp that shrinks one axis only
blurs along that axis only. At s >= 1 nothing is blurred and the result is exactly
cv2.warpAffine's.
"""
from __future__ import annotations

import numpy as np


def antialias_sigmas(linear: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(sigma_1, sigma_2, V) for a 2x2 linear map (source px -> output px).

    V's columns are the SOURCE-frame directions of the map's singular values;
    sigma_i is the blur needed along V[:, i] before sampling.
    """
    u, s, vt = np.linalg.svd(np.asarray(linear, float))
    sig = [max(0.0, (1.0 / si - 1.0) / 2.0) if si > 0 else 0.0 for si in s]
    return sig[0], sig[1], vt.T


def _oriented_blur(img: np.ndarray, s1: float, s2: float, v: np.ndarray) -> np.ndarray:
    import cv2
    if abs(s1 - s2) < 1e-3:                     # isotropic: one separable Gaussian
        return cv2.GaussianBlur(img, (0, 0), s1, borderType=cv2.BORDER_REFLECT)
    # Anisotropic: an oriented Gaussian kernel with the given sigmas along v's columns.
    cov = v @ np.diag([s1 ** 2 + 1e-6, s2 ** 2 + 1e-6]) @ v.T
    r = int(np.ceil(3 * max(s1, s2)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    pts = np.stack([xx.ravel(), yy.ravel()])
    k = np.exp(-0.5 * np.sum(pts * (np.linalg.inv(cov) @ pts), axis=0)).reshape(xx.shape)
    return cv2.filter2D(img, -1, (k / k.sum()).astype(np.float32), borderType=cv2.BORDER_REFLECT)


def warp_affine(img: np.ndarray, matrix: np.ndarray, dsize: tuple[int, int],
                border_value: float = 0.0) -> np.ndarray:
    """cv2.warpAffine (bilinear) with anti-aliasing whenever the map shrinks the image.

    `matrix` is 2x3 or 3x3, mapping source px -> output px, as for cv2.warpAffine.
    """
    import cv2
    m = np.asarray(matrix, float)[:2]
    src = np.asarray(img, np.float32)
    s1, s2, v = antialias_sigmas(m[:, :2])
    if max(s1, s2) >= 0.05:                     # below this the kernel is a single tap
        src = _oriented_blur(src, s1, s2, v)
    return cv2.warpAffine(src, m, (int(dsize[0]), int(dsize[1])), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
