"""Reconstruction backend abstraction.

A backend turns a set of uploaded images into a navigable `.splat` asset.
The real path (COLMAP poses + gsplat training) and the CPU fallback both
implement this interface so the job worker is agnostic to which one runs.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable, List, Optional

# progress_cb(fraction: float, message: Optional[str]) -> None
ProgressCallback = Callable[[float, Optional[str]], None]


class NotAvailable(RuntimeError):
    """Raised when a backend's external dependencies (GPU/COLMAP) are missing.

    The message is kept friendly so it can surface to users without leaking
    pipeline jargon.
    """


class ReconstructionBackend(abc.ABC):
    """Abstract reconstruction backend."""

    #: short key reported by /api/health, e.g. "fallback" or "colmap_gsplat"
    name: str = "base"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True if this backend can run in the current environment."""

    @abc.abstractmethod
    def reconstruct(
        self,
        project_dir: Path,
        images: List[Path],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        """Build a `.splat` scene from `images`.

        Args:
            project_dir: the project storage dir (asset is written inside it).
            images: absolute paths to the processed input images.
            progress_cb: optional callback(fraction 0..1, message) for the
                optimization phase; backends may call it repeatedly.

        Returns:
            Path to the written `asset.splat`.
        """
