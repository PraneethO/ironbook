"""Run COLMAP structure-from-motion to recover camera poses + a sparse cloud.

Produces the standard 3D Gaussian Splatting dataset layout the trainers expect:

    <out>/images/*.jpg
    <out>/sparse/0/{cameras,images,points3D}.bin

COLMAP here is the Homebrew build (CPU; "without CUDA"). GPU SIFT needs a
display/EGL context that isn't available in a headless server, so we run SIFT
on the CPU (`use_gpu 0`) for robustness. Sparse mapping is CPU anyway.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from .base import NotAvailable

# progress_cb(fraction_within_stage, message, stage_key)
ProgressCallback = Callable[..., None]

_RE_EXTRACT = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_RE_MATCH = re.compile(r"Processing image \[(\d+)/(\d+)\]")
_RE_REGISTER = re.compile(r"num_reg_frames=(\d+)")

# Allow override; default to the Homebrew location / PATH lookup.
COLMAP_BIN = os.environ.get("COLMAP_BIN") or shutil.which("colmap") or "/opt/homebrew/bin/colmap"

# GPU SIFT works headless on Apple Silicon (verified) and is much faster than
# CPU. "1" by default; set COLMAP_USE_GPU=0 to force CPU.
_USE_GPU = "0" if os.environ.get("COLMAP_USE_GPU") == "0" else "1"
# Cap feature-extraction resolution to keep it fast (images are already ≤1600).
_MAX_IMAGE_SIZE = os.environ.get("COLMAP_MAX_IMAGE_SIZE", "1600")
# Above this many photos, exhaustive matching (O(n²)) gets painful, so use
# sequential matching (assumes a roughly ordered capture). Override with
# COLMAP_MATCHER=exhaustive|sequential|auto.
_SEQUENTIAL_THRESHOLD = int(os.environ.get("COLMAP_SEQUENTIAL_THRESHOLD", "60"))


def colmap_available() -> bool:
    return Path(COLMAP_BIN).exists() or shutil.which("colmap") is not None


def _run(args: List[str], log_path: Path,
         on_line: Optional[Callable[[str], None]] = None) -> None:
    """Run a COLMAP subcommand, streaming output to log_path (and on_line).

    Streaming lets callers parse COLMAP's per-image progress so the UI shows
    continuous movement during the long pose-solve. Raises on failure.
    """
    with open(log_path, "ab") as log:
        log.write(("\n$ " + " ".join(args) + "\n").encode())
        log.flush()
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line.encode(errors="replace"))
            if on_line:
                on_line(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"COLMAP step failed: {args[1] if len(args) > 1 else args}")


def run_colmap(
    images: List[Path],
    out_dir: Path,
    progress_cb: Optional[ProgressCallback] = None,
    matcher: str = "auto",
) -> Path:
    """Run feature extraction → matching → mapping. Returns the best sparse model.

    Uses GPU SIFT and (for large sets) sequential matching to keep the CPU pose
    step from dominating. Raises NotAvailable (friendly) on missing COLMAP or
    unsolvable poses.
    """
    if not colmap_available():
        raise NotAvailable("The 3D pose solver isn't installed on this machine.")
    if len(images) < 5:
        raise NotAvailable(
            "We need more photos with overlap to find the camera positions. "
            "Try 20+ photos taken while walking around the subject."
        )

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse"
    img_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "database.db"
    log_path = out_dir / "colmap.log"

    # Stage the images into the expected <out>/images/ folder.
    for src in images:
        dst = img_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    total = len(images)

    def emit(frac: float, msg: Optional[str], stage: str) -> None:
        if progress_cb:
            progress_cb(max(0.0, min(1.0, frac)), msg, stage)

    # 1) Feature extraction (GPU SIFT). Pose phase: 0..45%.
    emit(0.0, "Looking at your photos", "pose_estimation")

    def _fe_line(line: str) -> None:
        m = _RE_EXTRACT.search(line)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            emit(0.45 * k / max(1, n), None, "pose_estimation")

    _run([
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(img_dir),
        "--ImageReader.single_camera", "1",
        # SIMPLE_RADIAL is COLMAP's default and is supported by the 3DGS
        # trainer's loader (OPENCV's extra distortion params are not).
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--FeatureExtraction.max_image_size", _MAX_IMAGE_SIZE,
        "--FeatureExtraction.use_gpu", _USE_GPU,
    ], log_path, on_line=_fe_line)

    # 2) Matching. Sequential (O(n)) for large ordered sets, else exhaustive.
    if matcher == "auto":
        use_sequential = total >= _SEQUENTIAL_THRESHOLD
    else:
        use_sequential = matcher == "sequential"
    matcher_cmd = "sequential_matcher" if use_sequential else "exhaustive_matcher"
    emit(0.45, "Matching features between photos", "pose_estimation")

    def _match_line(line: str) -> None:
        m = _RE_MATCH.search(line)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            emit(0.45 + 0.5 * k / max(1, n), None, "pose_estimation")

    _run([
        COLMAP_BIN, matcher_cmd,
        "--database_path", str(db_path),
        "--FeatureMatching.use_gpu", _USE_GPU,
    ], log_path, on_line=_match_line)
    emit(1.0, "Matched the photos", "pose_estimation")

    # 3) Sparse mapping (camera poses + sparse cloud). Structure stage: 0..100%
    #    driven by how many photos have been placed so far.
    emit(0.0, "Building rough 3D structure", "structure")

    def _map_line(line: str) -> None:
        m = _RE_REGISTER.search(line)
        if m:
            k = int(m.group(1))
            emit(min(0.99, k / max(1, total)), f"Placed {k} of {total} photos", "structure")

    _run([
        COLMAP_BIN, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(img_dir),
        "--output_path", str(sparse_dir),
        # Bundle adjustment is the CPU bottleneck; halve the iteration caps to
        # trade a little pose accuracy for speed (override via env).
        "--Mapper.ba_local_max_num_iterations",
        os.environ.get("COLMAP_BA_LOCAL_ITERS", "12"),
        "--Mapper.ba_global_max_num_iterations",
        os.environ.get("COLMAP_BA_GLOBAL_ITERS", "25"),
    ], log_path, on_line=_map_line)

    # COLMAP can split into several disconnected sub-models (sparse/0, /1, …).
    # Pick the one with the most registered images, not blindly sparse/0.
    candidates = [d for d in sparse_dir.iterdir() if d.is_dir()
                  and ((d / "cameras.bin").exists() or (d / "cameras.txt").exists())]
    if not candidates:
        raise NotAvailable(
            "We couldn't work out the camera positions from these photos. "
            "Try more photos with more overlap and varied angles."
        )
    best = max(candidates, key=count_registered_images)

    emit(1.0, "Camera positions found", "structure")
    return best


def count_registered_images(sparse_model: Path) -> int:
    """How many images COLMAP successfully registered (rough quality signal)."""
    images_bin = Path(sparse_model) / "images.bin"
    images_txt = Path(sparse_model) / "images.txt"
    if images_txt.exists():
        n = 0
        for line in images_txt.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                n += 1
        return n // 2  # two lines per image in the txt format
    if images_bin.exists():
        import struct
        data = images_bin.read_bytes()
        if len(data) >= 8:
            return struct.unpack("<Q", data[:8])[0]
    return 0
