"""Real 3D Gaussian Splatting backend for Apple Silicon.

Pipeline (this is the genuine article, not a 2.5D approximation):

    photos  ->  COLMAP SfM (camera poses + sparse cloud)
            ->  3DGS trainer on the Apple GPU (MLX)  ->  trained .ply
            ->  convert to 32-byte .splat  ->  viewer

COLMAP runs on CPU (Homebrew build). Training runs on the GPU via MLX (Apple's
array framework) in a separate venv that lives in the research/scratch dir
(it pulls in heavy ML deps we keep out of the API server's environment).

If COLMAP can't solve camera poses for a given photo set (too little overlap),
this backend falls back to the on-device depth reconstructor so the user still
gets *something* navigable instead of a dead end.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .base import NotAvailable, ProgressCallback, ReconstructionBackend
from .colmap_runner import colmap_available, count_registered_images, run_colmap
from .depth_reconstructor import DepthReconstructor
from .ply_to_splat import ply_to_splat
from .splat_cleanup import clean_splat

# --- configuration (env-overridable) -------------------------------------
_RESEARCH_DIR = Path(
    os.environ.get(
        "SPLAT_RESEARCH_DIR",
        "/Users/praneethotthi/Documents/gaussian-splatting/_splat_research",
    )
)
# cwd for the trainer process (the splat-apple repo, imported as PYTHONPATH=.).
_TRAINER_DIR = Path(os.environ.get("SPLAT_APPLE_DIR", str(_RESEARCH_DIR / "splat-apple")))
_TRAINER_PY = Path(os.environ.get("SPLAT_TRAINER_PY", str(_RESEARCH_DIR / ".venv" / "bin" / "python")))
# MLX training script with adaptive densification (lives in the research root).
_TRAIN_SCRIPT = Path(os.environ.get("SPLAT_TRAIN_SCRIPT", str(_RESEARCH_DIR / "run_train_mlx_densify.py")))
# The MLX rasterizer kernel: "cpp" (Metal C++, lower mem, recommended) or "python".
_TRAINER_BACKEND = os.environ.get("SPLAT_TRAINER_BACKEND", "cpp")
# Built Metal rasterizer extension (proves MLX path is compiled & ready).
_RASTERIZER_SO_GLOB = "mlx_gs/renderer/_rasterizer_metal*.so"
# Training iterations (quality/time tradeoff). 7000 ≈ ~20 min / ~23 dB on truck.
_TRAIN_ITERS = int(os.environ.get("SPLAT_TRAIN_ITERS", "7000"))
# Minimum images COLMAP must register to trust the reconstruction.
_MIN_REGISTERED = int(os.environ.get("SPLAT_MIN_REGISTERED", "5"))


class Gaussian3DGSBackend(ReconstructionBackend):
    """COLMAP + MLX 3D Gaussian Splatting on Apple Silicon."""

    name = "gaussian_3dgs"

    def is_available(self) -> bool:
        return (
            colmap_available()
            and _TRAINER_DIR.is_dir()
            and _TRAINER_PY.exists()
            and _TRAIN_SCRIPT.exists()
            and any(_TRAINER_DIR.glob(_RASTERIZER_SO_GLOB))
        )

    # -- trainer invocation (the one spot tied to the trainer's CLI) ------
    def _run_trainer(self, scene_dir: Path, out_ply: Path, log_path: Path,
                     progress_cb: Optional[ProgressCallback]) -> None:
        """Train 3DGS on `scene_dir` (COLMAP layout) -> `out_ply`.

        scene_dir contains: images/  and  sparse/0/{cameras,images,points3D}.bin
        """
        cmd = [
            str(_TRAINER_PY), "-u", str(_TRAIN_SCRIPT),
            "--data_dir", str(scene_dir),
            "--img_folder", "images",
            "--iters", str(_TRAIN_ITERS),
            "--rasterizer", _TRAINER_BACKEND,
            "--densify_from", "500",
            "--densify_until", str(max(1000, int(_TRAIN_ITERS * 0.7))),
            "--densify_every", "300",
            "--grad_thresh", "0.0002",
            "--out", str(out_ply),
        ]
        # The trainer imports the splat-apple package via PYTHONPATH=. (cwd=repo).
        env = dict(os.environ, PYTHONPATH=".")
        if progress_cb:
            progress_cb(0.05, "Optimizing the 3D scene on the GPU")
        with open(log_path, "ab") as log:
            log.write(("\n$ " + " ".join(cmd) + "\n").encode())
            log.flush()
            proc = subprocess.run(
                cmd, cwd=str(_TRAINER_DIR), env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0 or not out_ply.exists():
            raise RuntimeError("3D optimization did not finish successfully")

    # -- main -------------------------------------------------------------
    def reconstruct(
        self,
        project_dir: Path,
        images: List[Path],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        work = project_dir / "recon3d"
        work.mkdir(parents=True, exist_ok=True)
        asset = project_dir / "asset.splat"

        def emit(frac: float, msg: Optional[str] = None, stage: Optional[str] = None) -> None:
            if progress_cb:
                progress_cb(frac, msg, stage)

        # 1) Camera poses + sparse cloud (COLMAP, stage-aware). Fall back to depth.
        try:
            sparse_model = run_colmap(images, work, progress_cb=emit)
            if count_registered_images(sparse_model) < _MIN_REGISTERED:
                raise NotAvailable(
                    "We could only place a few of the photos. Try more photos with "
                    "more overlap and varied angles."
                )
        except NotAvailable:
            emit(0.0, "Not enough overlap for full 3D — building a quick depth world instead",
                 "optimization")
            return DepthReconstructor().reconstruct(
                project_dir, images, lambda f, m=None: emit(f, m, "optimization")
            )

        # 2) Train 3DGS on the GPU -> .ply
        out_ply = work / "point_cloud.ply"
        emit(0.0, "Optimizing the 3D scene on the GPU", "optimization")
        self._run_trainer(
            work, out_ply, work / "train.log",
            lambda f, m=None: emit(f, m, "optimization"),
        )

        # 3) Convert trained .ply -> .splat for the viewer
        emit(0.5, "Preparing your world for the viewer", "viewer_asset")
        ply_to_splat(out_ply, asset)
        # Trim floaters/haze (same cleanup as the msplat path).
        try:
            clean_splat(asset)
        except Exception:  # noqa: BLE001 - best-effort
            pass
        emit(1.0, None, "viewer_asset")
        return asset
