"""MATCH-07: NMI and the alignment check on a synthetic cross-modal pair (inverted contrast)."""
import cv2
import numpy as np

from chandralign.matching.similarity import alignment_check, nmi


def _pair(shift=(0.0, 0.0)):
    rng = np.random.default_rng(0)
    ref = cv2.GaussianBlur(rng.random((220, 220)).astype(np.float32), (0, 0), 3.0)
    M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
    src = 1.0 - cv2.warpAffine(ref, M, (220, 220), flags=cv2.INTER_CUBIC)   # other "sensor": inverted
    return src, ref


def test_nmi_is_higher_aligned_than_misaligned_across_a_contrast_inversion():
    src, ref = _pair()
    assert nmi(src, ref) > nmi(np.roll(src, 3, axis=1), ref) + 0.02


def test_alignment_check_finds_a_two_pixel_bias_and_passes_the_true_model():
    src, ref = _pair(shift=(2.0, -1.0))          # src(x) = ref(x - shift): the true model moves by +shift
    ok = alignment_check(src, ref, np.array([[1, 0, 2.0], [0, 1, -1.0], [0, 0, 1.0]]))
    bad = alignment_check(src, ref, np.eye(3))
    assert not ok["flag"] and np.hypot(*ok["peak_offset_px"]) < 0.3
    assert bad["flag"] and np.allclose(bad["peak_offset_px"], [2.0, -1.0], atol=0.3)
