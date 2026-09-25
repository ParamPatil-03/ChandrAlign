"""MATCH-12: the rotation search finds the angle a non-invariant matcher needs, and maps back."""
import cv2
import numpy as np

from chandralign import synth
from chandralign.contracts import MatchSet
from chandralign.matching.rotation import rotate_plane, rotation_search


def _plane(a):
    a = np.asarray(a, np.float32)
    return synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool), shadow_mask=np.zeros(a.shape, bool),
                            gsd_m=1.0, meta=None, geo=None)


def _upright_only(a, b):
    """A matcher that, like LoFTR, only works when both images share an orientation: NCC of
    small patches searched near the same position."""
    A, B = np.asarray(a.array, np.float32), np.asarray(b.array, np.float32)
    src, ref = [], []
    for y in range(30, min(A.shape[0], B.shape[0]) - 30, 16):
        for x in range(30, min(A.shape[1], B.shape[1]) - 30, 16):
            t = A[y - 8:y + 9, x - 8:x + 9]
            win = B[max(0, y - 20):y + 21, max(0, x - 20):x + 21]
            if t.std() < 1e-3 or win.shape[0] < 17 or win.shape[1] < 17:
                continue
            r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
            if r[iy, ix] > 0.9:
                src.append((x, y)); ref.append((max(0, x - 20) + ix + 8, max(0, y - 20) + iy + 8))
    s, r_ = np.array(src, float).reshape(-1, 2), np.array(ref, float).reshape(-1, 2)
    return MatchSet(src_pts=s, ref_pts=r_, confidence=np.ones(len(s), np.float32),
                    method="stub", regime="same_modal_normal", stage="direct")


def test_rotate_plane_maps_points_as_it_says():
    a = np.zeros((50, 80), np.float32); a[10, 20] = 1.0
    rp, M = rotate_plane(_plane(a), 90)
    y, x = np.unravel_index(int(np.argmax(rp.array)), rp.array.shape)
    assert np.allclose((M @ [20, 10, 1])[:2], (x, y), atol=1.0)


def test_rotation_search_recovers_a_quarter_turn_and_maps_back():
    rng = np.random.default_rng(0)
    ref = cv2.GaussianBlur(rng.random((200, 200)).astype(np.float32), (0, 0), 2.0)
    src = np.ascontiguousarray(np.rot90(ref))                   # src[i, j] = ref[j, n-1-i]
    assert len(_upright_only(_plane(src), _plane(ref)).src_pts) < 20   # the plain matcher fails
    ms = rotation_search(_plane(src), _plane(ref), _upright_only, step_deg=30, min_inliers=40)
    assert ms.rotation_search["applied"] and ms.rotation_search["angle_deg"] in (90.0, 270.0)
    want = np.c_[199 - ms.src_pts[:, 1], ms.src_pts[:, 0]]      # ref (x, y) of each source point
    assert len(ms.src_pts) > 50 and np.median(np.hypot(*(ms.ref_pts - want).T)) < 0.5


def test_a_good_upright_match_is_returned_unchanged():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.random((200, 200)).astype(np.float32), (0, 0), 2.0)
    ms = rotation_search(_plane(img), _plane(img), _upright_only, step_deg=30, min_inliers=40)
    assert ms.rotation_search == {"applied": False, "angle_deg": 0.0, "inliers": ms.rotation_search["inliers"]}
