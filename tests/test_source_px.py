"""evaluate/source_px.py: frame-pixel errors expressed in source pixels (the PS's unit)."""
import numpy as np

from chandralign.evaluate import source_px as sp


def test_a_coarser_frame_multiplies_the_error_in_source_pixels():
    # OHRC 0.3 m source on a 1.2 m NAC frame: 1 frame px = 4 source px
    J = sp.jacobian_from_pixel_sizes((0.3, 0.3), (1.2, 1.2))
    out = sp.to_source_px(0.5, J, "pixel sizes")
    assert np.isclose(out["worst"], 2.0) and np.isclose(out["typical"], 2.0)


def test_worst_case_takes_the_weaker_axis_and_ignores_rotation():
    J = sp.jacobian_from_pixel_sizes((100.0, 79.5), (100.0, 100.0))        # IIRS on the WAC mosaic
    th = np.radians(37.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    f, fr = sp.factors(J), sp.factors(R @ J)
    assert np.isclose(f["worst"], 100 / 79.5) and np.isclose(fr["worst"], f["worst"])
    assert sp.to_source_px(None, J, "x") is None
