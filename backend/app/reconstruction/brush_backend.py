"""Brush backend — proven, cross-platform 3D Gaussian Splatting on the GPU.

Brush (github.com/ArthurBrussee/brush) is an open-source 3DGS engine built on
the Burn ML framework + wgpu. It runs natively on Apple Silicon (Metal) with no
CUDA dependency, and is the same well-tested code path the public tools use —
unlike the hand-rolled msplat trainer + exporter it replaces, which were the
source of every artifact we chased (haze, needles, corrupted rotations).

Pipeline:  photos -> COLMAP poses -> brush (Metal GPU) -> standard 3DGS .ply
           -> ply_to_splat (32-byte viewer format) -> floater/needle cleanup

Brush reads a COLMAP dataset directly (images/ + sparse/0/) and exports a
standard Inria-format .ply, which our existing, correct ply_to_splat converter
turns into the viewer's .splat. If COLMAP can't solve poses, we fall back to the
on-device depth reconstructor so the user still gets something navigable.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

from .base import NotAvailable, ProgressCallback, ReconstructionBackend
from .colmap_runner import colmap_available, count_registered_images, run_colmap
from .colmap_sanitize import sanitize_sparse_points
from .depth_reconstructor import DepthReconstructor
from .ply_to_splat import ply_to_splat
from .sky_masks import generate_sky_masks
from .splat_cleanup import clean_splat

_RESEARCH_DIR = Path(
    os.environ.get(
        "SPLAT_RESEARCH_DIR",
        "/Users/praneethotthi/Documents/gaussian-splatting/_splat_research",
    )
)
_BRUSH_DIR = Path(os.environ.get("BRUSH_DIR", str(_RESEARCH_DIR / "brush")))
# The headless trainer binary (brush-cli -> `brush_cli`). Release build.
_BRUSH_BIN = Path(
    os.environ.get("BRUSH_BIN", str(_BRUSH_DIR / "target" / "release" / "brush-cli"))
)
# Training steps. With the tighter growth schedule below, fewer iterations are
# enough to settle the scene without spending minutes polishing floaters.
_ITERS = int(os.environ.get("BRUSH_ITERS", "8000"))
# Hard cap on splat count. Hitting a very high cap early was the failure mode
# that produced the random low-opacity cloud in outdoor train runs.
_MAX_SPLATS = int(os.environ.get("BRUSH_MAX_SPLATS", "450000"))
_GROWTH_STOP_ITER = int(os.environ.get("BRUSH_GROWTH_STOP_ITER", "2500"))
_GROWTH_SELECT_FRACTION = float(os.environ.get("BRUSH_GROWTH_SELECT_FRACTION", "0.08"))
_MEAN_NOISE_WEIGHT = float(os.environ.get("BRUSH_MEAN_NOISE_WEIGHT", "0.0"))
_OPAC_DECAY = float(os.environ.get("BRUSH_OPAC_DECAY", "0.001"))
_MIN_REGISTERED = int(os.environ.get("SPLAT_MIN_REGISTERED", "5"))
# Rough wall-clock estimate (s) for the progress heartbeat fallback, refined by
# parsing brush's per-refine "iter N" log lines when they appear. Measured at
# ~32 it/s once the cloud hits the cap on this GPU.
_EST_TRAIN_SECONDS = float(os.environ.get("BRUSH_EST_SECONDS", str(_ITERS / 32.0)))

_ITER_RE = re.compile(rb"iter\s+(\d+)")


class BrushBackend(ReconstructionBackend):
    name = "brush"

    def is_available(self) -> bool:
        return colmap_available() and _BRUSH_BIN.exists()

    def _run_brush(self, scene_dir: Path, export_dir: Path, export_name: str,
                   log_path: Path, progress_cb: Optional[ProgressCallback]) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(_BRUSH_BIN), str(scene_dir),
            "--total-train-iters", str(_ITERS),
            "--max-splats", str(_MAX_SPLATS),  # cap densification (speed + asset size)
            "--growth-stop-iter", str(_GROWTH_STOP_ITER),
            "--growth-select-fraction", str(_GROWTH_SELECT_FRACTION),
            "--mean-noise-weight", str(_MEAN_NOISE_WEIGHT),
            "--opac-decay", str(_OPAC_DECAY),
            "--export-path", str(export_dir),
            "--export-name", export_name,
            "--export-every", str(_ITERS),   # single export at the end
            "--eval-every", str(_ITERS + 1),  # no held-out eval (no split)
        ]

        # Brush logs "Refine iter N, M splats." lines to stdout. Parse the latest
        # iter for real progress; fall back to a time-based ease so the bar never
        # looks hung between refines.
        stop = threading.Event()
        t0 = time.monotonic()

        def _heartbeat() -> None:
            tau = max(4.0, _EST_TRAIN_SECONDS / 2.0)
            last_iter = 0
            while not stop.wait(0.5):
                frac_time = 1.0 - math.exp(-(time.monotonic() - t0) / tau)
                frac_iter = 0.0
                try:
                    data = log_path.read_bytes()
                    hits = _ITER_RE.findall(data[-4096:])
                    if hits:
                        last_iter = max(last_iter, int(hits[-1]))
                        frac_iter = last_iter / float(_ITERS)
                except OSError:
                    pass
                frac = min(0.95, max(frac_time, frac_iter))
                if progress_cb:
                    progress_cb(frac, None)

        beat = threading.Thread(target=_heartbeat, name="brush-heartbeat", daemon=True)
        beat.start()
        # brush's indicatif progress bar is suppressed on a non-TTY, and
        # env_logger defaults to error-level — so without RUST_LOG it writes
        # nothing to a piped stdout. Set info level to get the "Refine iter N"
        # lines our heartbeat parses for real progress.
        env = dict(os.environ, RUST_LOG=os.environ.get("RUST_LOG", "info"))
        try:
            with open(log_path, "ab") as log:
                log.write(("\n$ " + " ".join(cmd) + "\n").encode())
                log.flush()
                proc = subprocess.run(
                    cmd, cwd=str(_BRUSH_DIR), env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                )
        finally:
            stop.set()
            beat.join(timeout=1.0)
        if proc.returncode != 0:
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

        # 1) Camera poses (COLMAP). Reports its own stages. Fall back to depth.
        try:
            sparse_model = run_colmap(images, work, progress_cb=emit)
            if count_registered_images(sparse_model) < _MIN_REGISTERED:
                raise NotAvailable(
                    "We could only place a few of the photos. Try more photos with "
                    "more overlap and varied angles."
                )
            stats = sanitize_sparse_points(sparse_model)
            if stats.applied:
                emit(
                    1.0,
                    f"Removed {stats.dropped:,} unstable sparse points before training",
                    "structure",
                )
        except NotAvailable:
            emit(0.0, "Not enough overlap for full 3D — building a quick depth world instead",
                 "optimization")
            return DepthReconstructor().reconstruct(
                project_dir, images, lambda f, m=None: emit(f, m, "optimization")
            )

        # 2) Train on the GPU with brush -> standard 3DGS .ply.
        mask_count = generate_sky_masks(work / "images", work / "masks")
        if mask_count:
            emit(0.0, f"Masked sky in {mask_count:,} photos", "optimization")

        export_dir = work / "brush_exports"
        export_name = "out.ply"
        emit(0.0, "Optimizing the 3D scene on the GPU", "optimization")
        self._run_brush(
            work, export_dir, export_name, work / "train.log",
            lambda f, m=None: emit(f, m, "optimization"),
        )

        out_ply = export_dir / export_name
        if not out_ply.exists():
            # brush interpolates {iter} into the name when present; otherwise it's
            # literal. Be forgiving and grab the newest .ply it wrote.
            plys = sorted(export_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime)
            if not plys:
                raise RuntimeError("3D optimization produced no output")
            out_ply = plys[-1]

        # 3) Convert standard .ply -> viewer .splat (our correct, normalizing path).
        emit(0.5, "Preparing your world for the viewer", "viewer_asset")
        ply_to_splat(out_ply, asset)

        # 4) Trim floaters/needles/haze that wreck the viewer's auto-framing.
        emit(0.7, "Cleaning up stray bits", "compression")
        try:
            before, after = clean_splat(asset, strict=True)
            if before != after:
                emit(0.9, f"Removed {before - after:,} stray gaussians", "compression")
        except Exception:  # noqa: BLE001 - cleanup is best-effort, never fatal
            pass

        emit(1.0, None, "viewer_asset")
        return asset
