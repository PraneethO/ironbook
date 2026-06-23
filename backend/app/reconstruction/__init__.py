"""Reconstruction backends and the `.splat` format codec."""
from __future__ import annotations

import os

from .base import NotAvailable, ReconstructionBackend
from .brush_backend import BrushBackend
from .colmap_gsplat import ColmapGsplatBackend
from .depth_reconstructor import DepthReconstructor
from .fallback import FallbackReconstructor
from .gaussian_3dgs import Gaussian3DGSBackend
from .msplat_backend import MsplatBackend


_BACKENDS_BY_NAME = {
    "brush": BrushBackend,
    "msplat": MsplatBackend,
    "gaussian_3dgs": Gaussian3DGSBackend,
    "depth": DepthReconstructor,
    "fallback": FallbackReconstructor,
    "colmap_gsplat": ColmapGsplatBackend,
}


def select_backend() -> ReconstructionBackend:
    """Return the best available backend, in order of fidelity.

    Set GSW_FORCE_BACKEND=<name> to pin one (used by tests to force the fast,
    hermetic `fallback` instead of the real COLMAP+MLX pipeline).

    1. CUDA COLMAP + gsplat — true 3DGS on an NVIDIA GPU; not present here.
    2. COLMAP + brush — proven, cross-platform 3DGS on the Apple GPU (Metal via
       wgpu). The active path on this M5 Max; standard trainer + exporter, no
       hand-rolled format code. Falls back to depth if COLMAP can't solve poses.
    3. COLMAP + msplat — the older Apple-Silicon trainer (kept as a fallback).
    4. On-device monocular-depth reconstruction — real, input-dependent 2.5D
       photo-to-3D; used if no 3DGS trainer is installed.
    5. Procedural fallback — a fixed colored room, last resort.
    """
    forced = os.environ.get("GSW_FORCE_BACKEND")
    if forced and forced in _BACKENDS_BY_NAME:
        return _BACKENDS_BY_NAME[forced]()

    cuda = ColmapGsplatBackend()
    if cuda.is_available():  # pragma: no cover - no NVIDIA GPU here
        return cuda
    brush = BrushBackend()
    if brush.is_available():
        return brush
    msplat = MsplatBackend()
    if msplat.is_available():
        return msplat
    gs = Gaussian3DGSBackend()
    if gs.is_available():
        return gs
    depth = DepthReconstructor()
    if depth.is_available():
        return depth
    return FallbackReconstructor()


__all__ = [
    "NotAvailable",
    "ReconstructionBackend",
    "BrushBackend",
    "ColmapGsplatBackend",
    "MsplatBackend",
    "Gaussian3DGSBackend",
    "DepthReconstructor",
    "FallbackReconstructor",
    "select_backend",
]
