"""preprocess/resample.py: warping that shrinks must not alias."""
import cv2
import numpy as np
import pytest

from chandralign.preprocess.resample import antialias_sigmas, warp_affine


def test_opencv_warpaffine_ignores_inter_area():
    """The reason the module exists. If OpenCV ever implements INTER_AREA in
    warpAffine this fails, and the helper can be revisited."""
    a = np.random.default_rng(0).random((200, 200)).astype(np.float32)
    m = np.array([[0.6, 0.05, 2], [-0.05, 0.6, 3]], np.float32)
    area = cv2.warpAffine(a, m, (120, 120), flags=cv2.INTER_AREA)
    linear = cv2.warpAffine(a, m, (120, 120), flags=cv2.INTER_LINEAR)
    assert np.array_equal(area, linear)


def test_no_blur_when_not_shrinking():
    a = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    for m in (np.array([[1.0, 0, 0.3], [0, 1.0, -0.2]]), np.array([[1.4, 0.1, 0], [-0.1, 1.4, 0]])):
        ours = warp_affine(a, m, (80, 80))
        cv2_ = cv2.warpAffine(a, m.astype(np.float32), (80, 80), flags=cv2.INTER_LINEAR)
        assert np.array_equal(ours, cv2_)


def test_shrinking_suppresses_aliasing():
    """Stripes at 0.4 cycles/px cannot exist on a grid 3x coarser. Correctly
    downsampled they vanish to a flat mean; bilinear folds them into a false,
    lower-frequency pattern."""
    x = np.arange(600)
    stripes = (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * x))[None, :].repeat(60, 0).astype(np.float32)
    m = np.array([[1 / 3, 0, 0], [0, 1 / 3, 0]])
    ours = warp_affine(stripes, m, (200, 20))[5:-5, 5:-5]
    plain = cv2.warpAffine(stripes, m.astype(np.float32), (200, 20), flags=cv2.INTER_LINEAR)[5:-5, 5:-5]
    reference = cv2.resize(stripes, (200, 20), interpolation=cv2.INTER_AREA)[5:-5, 5:-5]
    assert plain.std() > 0.2                       # bilinear aliases
    assert ours.std() < 0.35 * plain.std()         # anti-aliased: most of it gone
    assert abs(ours.mean() - reference.mean()) < 0.02


def test_blur_follows_the_axis_that_shrinks():
    s1, s2, v = antialias_sigmas(np.diag([1 / 3, 1.0]))    # x shrinks 3x, y does not
    blurs = dict(zip((abs(v[0, 0]) > 0.5, abs(v[0, 1]) > 0.5), (s1, s2)))
    assert blurs[True] == pytest.approx(1.0)                # along x: (3 - 1) / 2
    assert min(s1, s2) == 0.0


def test_constant_image_stays_constant():
    """The blur must not change brightness. Checked only where the output samples
    the source well away from its edge (the border is legitimately zero)."""
    a = np.full((90, 90), 0.37, np.float32)
    m = np.array([[0.5, 0.2, 5], [-0.2, 0.5, 5]])
    out = warp_affine(a, m, (40, 40))
    inv = cv2.invertAffineTransform(m.astype(np.float32))
    yy, xx = np.mgrid[0:40, 0:40]
    sx, sy = (inv @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])).reshape(2, 40, 40)
    inside = (sx > 8) & (sx < 81) & (sy > 8) & (sy < 81)
    assert inside.sum() > 400
    assert np.allclose(out[inside], 0.37, atol=1e-5)


def test_one_axis_shrink_blurs_only_that_axis():
    """x shrinks 3x, y does not: stripes along x vanish, stripes along y survive
    exactly as a plain bilinear warp leaves them."""
    n = 300
    x = np.arange(n)
    x_stripes = (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * x))[None, :].repeat(n, 0).astype(np.float32)
    y_stripes = np.ascontiguousarray(x_stripes.T * 0 + (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * x))[:, None])
    m = np.array([[1 / 3, 0, 0], [0, 1.0, 0]])
    ox = warp_affine(x_stripes, m, (n // 3, n))[10:-10, 10:-10]
    oy = warp_affine(y_stripes, m, (n // 3, n))[10:-10, 10:-10]
    py = cv2.warpAffine(y_stripes, m.astype(np.float32), (n // 3, n), flags=cv2.INTER_LINEAR)[10:-10, 10:-10]
    assert ox.std() < 0.12
    assert np.abs(oy - py).max() < 0.02
