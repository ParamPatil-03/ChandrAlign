"""CHECK-07: an impossible pair comes back REJECTED with failure_modes, end to end, and nothing crashes."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from chandralign import synth
from chandralign.pipeline import register

ROOT = Path(__file__).resolve().parents[1]


def _plane(a):
    a = np.asarray(a, np.float32)
    return synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool), shadow_mask=np.zeros(a.shape, bool),
                            gsd_m=5.0, meta=None, geo=None)


def _terrain(seed, n=320):
    rng = np.random.default_rng(seed)
    return cv2.normalize(cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 2.0), None, 0, 1, cv2.NORM_MINMAX)


def test_an_impossible_pair_is_rejected_with_a_named_failure_mode():
    r = register(_plane(_terrain(1)), _plane(_terrain(2)))          # two unrelated surfaces
    assert r.confidence_tier == "REJECTED"
    assert r.failure_modes, "a rejection must say why (CHECK-07)"
    assert r.gates, "and must carry its control-gate results (CHECK-08)"


def test_a_genuine_pair_is_not_rejected():
    a = _terrain(3)
    b = cv2.warpAffine(a, np.float32([[1, 0, 3.3], [0, 1, -2.1]]), a.shape[::-1], flags=cv2.INTER_CUBIC)
    r = register(_plane(a), _plane(b))
    assert r.confidence_tier in ("HIGH", "MEDIUM", "LOW"), (r.confidence_tier, r.failure_modes, r.notes)
    assert np.allclose(r.model.matrix[:2, 2], [3.3, -2.1], atol=0.3)


def test_the_process_exits_0_on_an_impossible_pair():
    code = ("import numpy as np, cv2\n"
            "from chandralign import synth\nfrom chandralign.pipeline import register\n"
            "def p(s):\n"
            "    a = cv2.normalize(cv2.GaussianBlur(np.random.default_rng(s).random((320, 320)).astype(np.float32), (0, 0), 2.0), None, 0, 1, cv2.NORM_MINMAX)\n"
            "    return synth.ImagePlane(array=a, valid_mask=np.ones(a.shape, bool), shadow_mask=np.zeros(a.shape, bool), gsd_m=5.0, meta=None, geo=None)\n"
            "r = register(p(1), p(2))\nprint(r.confidence_tier, r.failure_modes)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().startswith("REJECTED")
