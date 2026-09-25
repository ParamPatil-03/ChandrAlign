"""Errors in SOURCE-image pixels.

The problem statement asks for "sub-pixel accuracy of source image". Our registrations measure
their errors in the frame the fine stage works in -- usually the REFERENCE grid (TC, NAC, the
WAC mosaic), whose pixels are not the source's: an OHRC pixel is ~0.3 m but the NAC frame's
is 0.5-1.3 m, so 0.3 frame px can be 0.9 OHRC px. This converts, honestly:

    e_src = J^-1 e_frame,   J = d(frame px) / d(source px), the linear part of the transform.

A reported error is a magnitude whose direction is not kept, so the conversion gives the WORST
case over directions (1 / smallest singular value of J) as the headline, and the typical
(1 / sqrt(det J)) beside it. Where a transform is not stored, J is built from the two pixel
sizes (source metres / frame metres per axis); that assumes aligned axes and is labelled so.
"""
from __future__ import annotations

import numpy as np


def jacobian_from_transform(matrix) -> np.ndarray:
    """J = linear part of a source-px -> frame-px affine (3x3 or 2x3)."""
    return np.asarray(matrix, float)[:2, :2]


def jacobian_from_pixel_sizes(src_px_m: tuple[float, float], frame_px_m: tuple[float, float]) -> np.ndarray:
    """J from (across/x, along/y) pixel sizes in metres, axes assumed aligned."""
    return np.diag([src_px_m[0] / frame_px_m[0], src_px_m[1] / frame_px_m[1]])


def factors(J) -> dict:
    """Source px per frame px: worst case over directions, and typical."""
    s = np.linalg.svd(np.asarray(J, float), compute_uv=False)
    return {"worst": float(1.0 / s.min()), "typical": float(1.0 / np.sqrt(s[0] * s[1]))}


def to_source_px(err_frame_px, J, method: str) -> dict | None:
    """{"worst", "typical", "factor_worst", "method"} for an error magnitude in frame px."""
    if err_frame_px is None:
        return None
    f = factors(J)
    e = float(err_frame_px)
    return {"worst": round(e * f["worst"], 4), "typical": round(e * f["typical"], 4),
            "factor_worst": round(f["worst"], 4), "method": method}
