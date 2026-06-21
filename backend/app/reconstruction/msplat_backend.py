"""msplat backend — fully GPU-resident 3D Gaussian Splatting on Apple Silicon.

msplat (github.com/rayanht/msplat) is a fused-Metal trainer: projection, sort,
rasterization, SSIM, backward, Adam, and densification all run as Metal compute
kernels with no per-iteration CPU round-trip. On this M5 Max it trains the truck
scene to ~24 dB PSNR in ~41 s (7000 iters, ~169 it/s) at ~98% GPU / ~9% CPU —
roughly 20-30x faster than the MLX path and actually saturating the GPU.

Pipeline:  photos -> COLMAP poses -> msplat (Metal GPU) -> .splat (viewer-ready)

msplat reads the selected COLMAP sparse model directly and, when the output
path ends in `.splat`, writes the antimatter15 32-byte format the viewer
already renders (verified byte-compatible — no conversion step needed). If
COLMAP can't solve poses, we fall back to the on-device depth reconstructor.
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

from .base import NotAvailable, ProgressCallback, ReconstructionBackend
from .colmap_runner import colmap_available, count_registered_images, run_colmap
from .depth_reconstructor import DepthReconstructor
from .splat_cleanup import clean_splat

_RESEARCH_DIR = Path(
    os.environ.get(
        "SPLAT_RESEARCH_DIR",
        "/Users/praneethotthi/Documents/gaussian-splatting/_splat_research",
    )
)
_MSPLAT_DIR = Path(os.environ.get("MSPLAT_DIR", str(_RESEARCH_DIR / "msplat")))
_MSPLAT_BIN = Path(os.environ.get("MSPLAT_BIN", str(_MSPLAT_DIR / "build" / "msplat")))
_METALLIB = _MSPLAT_DIR / "build" / "default.metallib"
# Training iterations. This is the single biggest quality lever. 7000 was far
# too few — it leaves opacity unconverged (foggy haze) and floaters unpruned.
# 30000 is the standard 3DGS recipe and, with capped densification below, runs
# in ~100 s here (~290 it/s) while landing ~+1.4 dB / +0.05 SSIM over 7k.
_ITERS = int(os.environ.get("MSPLAT_ITERS", "30000"))
# Densification gradient threshold. msplat's default over-densifies real scenes
# (1.1M gaussians on the train set → per-tile overflow + a 37 MB asset the web
# viewer chokes on). 0.0005 caps it near ~300k for ~98% of the quality.
_DENSIFY_GRAD_THRESH = os.environ.get("MSPLAT_DENSIFY_GRAD_THRESH", "0.0005")
# Progressive downscale levels (msplat ramps from this many levels down up to
# full res). 2 starts at 1/16 px, which jams the initial cloud into tiny tiles
# (gaussians dropped) — 1 is the sweet spot: gentle start, full-res finish,
# no unified-memory swapping even on 300-image sets. "auto" picks by count.
_DOWNSCALES_ENV = os.environ.get("MSPLAT_DOWNSCALES", "auto")
_MIN_REGISTERED = int(os.environ.get("SPLAT_MIN_REGISTERED", "5"))
# Rough wall-clock estimate (s) for the heartbeat, since msplat is silent.
# Scales with iteration count (~290 it/s on this GPU at capped density).
_EST_TRAIN_SECONDS = float(os.environ.get("MSPLAT_EST_SECONDS", str(_ITERS / 280.0)))


def _downscales_for(n_images: int) -> int:
    if _DOWNSCALES_ENV != "auto":
        return int(_DOWNSCALES_ENV)
    if n_images >= 40:
        return 1   # gentle progressive start, ramps to full res
    return 0       # small sets: full res from the start

class MsplatBackend(ReconstructionBackend):
    name = "msplat"

    def is_available(self) -> bool:
        return colmap_available() and _MSPLAT_BIN.exists() and _METALLIB.exists()

    def _run_msplat(
        self,
        scene_dir: Path,
        out_splat: Path,
        log_path: Path,
        downscales: int,
        progress_cb: Optional[ProgressCallback],
        image_dir: Optional[Path] = None,
    ) -> None:
        cmd = [
            str(_MSPLAT_BIN), str(scene_dir),
            "-n", str(_ITERS),
            "--num-downscales", str(downscales),
            "--densify-grad-thresh", _DENSIFY_GRAD_THRESH,
            "--bg-color", "0", "0", "0",
            "-o", str(out_splat),
        ]
        if image_dir is not None:
            cmd.extend(["--colmap-image-path", str(image_dir)])
        # msplat prints no per-iteration progress, so ease a heartbeat toward
        # ~0.95 over the estimated training time, then snap to done. Keeps the
        # bar moving instead of looking hung during the GPU phase.
        stop = threading.Event()

        def _heartbeat() -> None:
            t0 = time.monotonic()
            tau = max(4.0, _EST_TRAIN_SECONDS / 2.0)
            while not stop.wait(0.5):
                frac = min(0.95, 1.0 - math.exp(-(time.monotonic() - t0) / tau))
                if progress_cb:
                    progress_cb(frac, None)

        beat = threading.Thread(target=_heartbeat, name="msplat-heartbeat", daemon=True)
        beat.start()
        try:
            with open(log_path, "ab") as log:
                log.write(("\n$ " + " ".join(cmd) + "\n").encode())
                log.flush()
                proc = subprocess.run(
                    cmd, cwd=str(_MSPLAT_DIR),
                    stdout=log, stderr=subprocess.STDOUT,
                )
        finally:
            stop.set()
            beat.join(timeout=1.0)
        if proc.returncode != 0 or not out_splat.exists():
            raise RuntimeError("3D optimization did not finish successfully")

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

        # 1) Camera poses (COLMAP). It reports its own stages (pose_estimation,
        #    structure) with fine sub-progress. Fall back to depth if unsolvable.
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

        # 2) Train on the GPU -> .splat directly (viewer-ready, no conversion).
        downscales = _downscales_for(len(images))
        emit(0.0, "Optimizing the 3D scene on the GPU", "optimization")
        self._run_msplat(
            sparse_model, asset, work / "train.log", downscales,
            lambda f, m=None: emit(f, m, "optimization"),
            image_dir=work / "images",
        )

        # Trim floaters/haze that wreck the viewer's auto-framing and loom as
        # translucent planes when you fly through them. Cheap, in-place, safe.
        emit(0.3, "Cleaning up stray bits", "compression")
        try:
            before, after = clean_splat(asset)
            if before != after:
                emit(0.9, f"Removed {before - after:,} stray gaussians", "compression")
        except Exception:  # noqa: BLE001 - cleanup is best-effort, never fatal
            pass

        emit(1.0, "Preparing your world for the viewer", "viewer_asset")
        return asset
