"""The *real* reconstruction backend: COLMAP poses + gsplat training.

This is the integration target described in DECISIONS.md. It is NOT used on
this machine (no GPU / no COLMAP), so `is_available()` returns False and
`reconstruct()` raises `NotAvailable`. The code below documents exactly where
each real step plugs in, so swapping the fallback for this is a config change
once a GPU worker exists.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from .base import NotAvailable, ProgressCallback, ReconstructionBackend


def _has_colmap() -> bool:
    return shutil.which("colmap") is not None


def _has_cuda() -> bool:
    # Real implementation would probe torch.cuda.is_available(). We avoid a
    # torch import at runtime (not installed here) and treat absence as "no GPU".
    try:  # pragma: no cover - exercised only on a GPU box
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class ColmapGsplatBackend(ReconstructionBackend):
    """Structure-from-Motion (COLMAP) + Splatfacto-style gsplat training.

    Pipeline (when available):
      1. preprocessing   -> undistort / resize images, write a COLMAP image dir.
      2. pose_estimation -> `colmap feature_extractor` + `exhaustive_matcher`
                            + `mapper` to recover camera intrinsics/extrinsics
                            and a sparse point cloud.
      3. structure       -> densify / clean the sparse cloud; init Gaussians.
      4. optimization    -> gsplat training loop (Splatfacto): optimize means,
                            scales, rotations, opacities, SH colors against the
                            posed images; report loss/progress via progress_cb.
      5. compression     -> prune low-opacity splats, quantize.
      6. viewer_asset    -> export the trained model to the 32-byte `.splat`
                            layout (see splat_format.write_splats).
    """

    name = "colmap_gsplat"

    def is_available(self) -> bool:
        return _has_colmap() and _has_cuda()

    def reconstruct(
        self,
        project_dir: Path,
        images: List[Path],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        if not self.is_available():
            # Friendly, no jargon — surfaced to users if ever selected.
            raise NotAvailable(
                "High-detail 3D reconstruction needs a graphics-accelerated "
                "server, which isn't available here yet."
            )

        # --- Real pipeline would go here. Sketch only. ---------------------
        # work = project_dir / "colmap"
        # run_colmap_feature_extractor(images, work)
        # run_colmap_matcher(work)
        # cameras, points = run_colmap_mapper(work)
        # model = gsplat_init_from_points(points, cameras)
        # for step in range(num_steps):
        #     loss = model.train_step()
        #     if progress_cb:
        #         progress_cb(step / num_steps, f"Refining detail (loss {loss:.3f})")
        # arrays = model.export_splats()
        # from .splat_format import write_splats
        # asset = project_dir / "asset.splat"
        # write_splats(asset, arrays)
        # return asset
        raise NotAvailable(  # pragma: no cover - unreachable on this machine
            "High-detail 3D reconstruction is not configured on this server."
        )
