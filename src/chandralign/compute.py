"""Which device actually did the work.

WHY THIS EXISTS
A machine can have a working GPU while every number it produces was computed on
the CPU. That was literally true here: torch sees an RTX 4050 and
config.resolve_device() returns "cuda", but the classical path is OpenCV-only,
the pip OpenCV wheel is built without CUDA (0 CUDA devices), and the chain never
imports torch at all. Reporting those results next to "GPU confirmed" would be
false by adjacency, which is the same class of error as failure mode #19.

So device is RECORDED per run from what the code actually did, never inferred
from what the machine could have done. This feeds RegistrationResult.provenance
and answers judge questions #23 and #26 (how much GPU is required, how long does
inference take) with a measurement instead of an assumption.
"""
from __future__ import annotations

import platform
import sys


def opencv_cuda_devices() -> int:
    """CUDA devices visible to OpenCV. pip wheels are built without CUDA."""
    try:
        import cv2
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def torch_status() -> dict:
    """Torch/CUDA status, without importing torch if it is not already loaded.

    Deliberately does not import torch on demand: the classical path must stay
    torch-free, and a probe that imported it would make 'did this run touch
    torch?' unanswerable.
    """
    mod = sys.modules.get("torch")
    if mod is None:
        return {"imported": False, "cuda_available": None, "device_name": None}
    try:
        avail = bool(mod.cuda.is_available())
        return {"imported": True,
                "version": getattr(mod, "__version__", None),
                "cuda_available": avail,
                "device_name": mod.cuda.get_device_name(0) if avail else None}
    except Exception:
        return {"imported": True, "cuda_available": None, "device_name": None}


def capability() -> dict:
    """What this machine COULD use. Not a claim about any particular result."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv_cuda_devices": opencv_cuda_devices(),
        "torch": torch_status(),
    }


def classical_device() -> str:
    """Device the OpenCV classical path runs on.

    Always cpu unless OpenCV was built with CUDA, which the pip wheel is not.
    Stated as a fact about the build rather than a hope about the hardware.
    """
    return "cuda" if opencv_cuda_devices() > 0 else "cpu"


def provenance(device_used: str, stage: str = "match") -> dict:
    """A provenance fragment recording what actually ran where."""
    return {
        f"{stage}_device": device_used,
        "capability": capability(),
        "note": ("device_used is what this run actually used; capability is only "
                 "what the machine offers and must not be read as the device used"),
    }
