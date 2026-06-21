"""On-device CPU depth reconstruction.

This is the real (input-dependent) reconstructor used when no GPU/COLMAP is
available. It does NOT fake a fixed scene. For every uploaded photo it:

  1. estimates a dense depth map with a monocular depth model
     (Depth Anything V2 Small, ONNX, run on CPU/CoreML),
  2. back-projects each pixel into 3D using a pinhole camera model, giving a
     colored point cloud whose *shape* comes from the photo's content,
  3. places each photo's cloud on an arc around the viewer so the photos form a
     navigable scene you can walk through.

Because the geometry is derived from the actual pixels and depth of each image,
different uploads produce visibly different worlds. It is genuine 2.5D
photo-to-3D — not multi-view-consistent 3D Gaussian Splatting (that still needs
the GPU path in `colmap_gsplat.py`), and it is honest about that in the UI.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps

from .base import NotAvailable, ProgressCallback, ReconstructionBackend
from .splat_format import write_splats

# Path to the bundled ONNX depth model (backend/models/...).
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_MODEL_PATH = _MODELS_DIR / "depth_anything_v2_small.onnx"

# Model input side (multiple of 14, per Depth Anything V2).
_INPUT_SIDE = 518
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Scene shaping.
_TOTAL_SPLAT_BUDGET = 150_000   # across all photos
_MAX_PER_IMAGE = 16_000
_MAX_IMAGES = 24                # cap panels so the arc stays readable
_NEAR, _FAR = 0.6, 3.6          # local depth range each photo is mapped into (m-ish)
_ARC_RADIUS = 4.6               # how far each photo panel sits from the viewer
_ARC_SPREAD = 2.6               # total angular spread of the arc (radians)
_HFOV_DEG = 62.0                # assumed horizontal field of view for back-projection
_EYE_HEIGHT = 1.4               # lift the scene to roughly standing eye level


class DepthReconstructor(ReconstructionBackend):
    """Monocular-depth photo-to-3D reconstructor (CPU/CoreML, ONNX)."""

    name = "depth"

    def __init__(self) -> None:
        self._session = None  # lazily created

    # -- availability ----------------------------------------------------
    def is_available(self) -> bool:
        if not _MODEL_PATH.exists():
            return False
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return False
        return True

    def _get_session(self):
        if self._session is None:
            import onnxruntime as ort

            available = ort.get_available_providers()
            # CPU is the default: ~150ms/image here, deterministic, and avoids
            # CoreML's noisy temp-dir cleanup warnings. Opt into CoreML with
            # GSW_DEPTH_PROVIDER=coreml if you want to try the Neural Engine.
            want = os.environ.get("GSW_DEPTH_PROVIDER", "cpu").lower()
            if want == "coreml" and "CoreMLExecutionProvider" in available:
                chosen = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                chosen = ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # quiet
            self._session = ort.InferenceSession(
                str(_MODEL_PATH), sess_options=opts, providers=chosen
            )
        return self._session

    # -- depth -----------------------------------------------------------
    def _estimate_depth(self, im: Image.Image) -> np.ndarray:
        """Return a (S, S) float depth map in [0,1], 1 = near, 0 = far."""
        sq = im.convert("RGB").resize((_INPUT_SIDE, _INPUT_SIDE), Image.BILINEAR)
        arr = (np.asarray(sq, dtype=np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        x = arr.transpose(2, 0, 1)[None].astype(np.float32)
        out = self._get_session().run(None, {"pixel_values": x})[0][0]
        out = np.asarray(out, dtype=np.float32)
        if out.shape != (_INPUT_SIDE, _INPUT_SIDE):
            # model may emit 14*floor(side/14); resample to the canonical grid
            out = np.asarray(
                Image.fromarray(out).resize((_INPUT_SIDE, _INPUT_SIDE), Image.BILINEAR),
                dtype=np.float32,
            )
        # Robust normalize to [0,1]; the model emits affine-invariant disparity
        # (higher = nearer), so the normalized value doubles as "nearness".
        lo, hi = np.percentile(out, 2.0), np.percentile(out, 98.0)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        return np.clip((out - lo) / (hi - lo), 0.0, 1.0)

    # -- one photo -> point cloud ---------------------------------------
    def _photo_cloud(self, im: Image.Image, budget: int):
        """Back-project one photo into (positions, scales, colors) in panel-local
        camera space (camera at origin looking down -Z)."""
        nearness = self._estimate_depth(im)  # (S,S) in [0,1], 1=near
        # Colors sampled on the same grid as the depth map.
        rgb = np.asarray(
            im.convert("RGB").resize((_INPUT_SIDE, _INPUT_SIDE), Image.BILINEAR),
            dtype=np.uint8,
        )

        # Subsample a regular grid to hit the per-photo budget.
        stride = max(1, int(round(_INPUT_SIDE / math.sqrt(max(1, budget)))))
        vs = np.arange(0, _INPUT_SIDE, stride)
        us = np.arange(0, _INPUT_SIDE, stride)
        uu, vv = np.meshgrid(us, vs)  # (H,W)
        uu = uu.ravel()
        vv = vv.ravel()

        near = nearness[vv, uu]                         # 1 = near, 0 = far
        depth = _NEAR + (1.0 - near) * (_FAR - _NEAR)    # world depth (positive)

        # Pinhole back-projection. Pixel centre is the principal point.
        f = 0.5 * _INPUT_SIDE / math.tan(math.radians(_HFOV_DEG) * 0.5)
        cx = cy = _INPUT_SIDE * 0.5
        x_cam = (uu - cx) / f * depth
        y_cam = -(vv - cy) / f * depth                   # flip so image-up is +Y
        z_cam = -depth                                   # camera looks down -Z

        pos = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)

        # Isotropic gaussian scale ~ projected pixel footprint at that depth, so
        # nearer (denser) splats stay small and far ones fill gaps.
        footprint = (stride / f) * depth
        s = np.clip(footprint * 0.6, 0.012, 0.14).astype(np.float32)
        scales = np.repeat(s[:, None], 3, axis=1)

        col = rgb[vv, uu].astype(np.uint8)               # (N,3)
        colors = np.concatenate([col, np.full((col.shape[0], 1), 255, np.uint8)], axis=1)
        return pos, scales, colors

    # -- arc placement ---------------------------------------------------
    @staticmethod
    def _panel_basis(theta: float):
        """Camera centre on an arc of radius R and its right/up/forward axes,
        forward pointing toward the origin (the viewer)."""
        c = np.array([_ARC_RADIUS * math.sin(theta), 0.0, -_ARC_RADIUS * math.cos(theta)], np.float32)
        forward = -c / (np.linalg.norm(c) + 1e-9)        # toward origin
        up = np.array([0.0, 1.0, 0.0], np.float32)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right) + 1e-9
        return c, right, up, forward

    # -- main ------------------------------------------------------------
    def reconstruct(
        self,
        project_dir: Path,
        images: List[Path],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        if not self.is_available():
            raise NotAvailable(
                "The 3D builder isn't ready on this machine. Please try again later."
            )
        if not images:
            raise NotAvailable("No usable photos were found to build a 3D world.")

        imgs = images[:_MAX_IMAGES]
        m = len(imgs)
        budget_each = min(_MAX_PER_IMAGE, max(2000, _TOTAL_SPLAT_BUDGET // m))
        step = _ARC_SPREAD / max(1, m - 1) if m > 1 else 0.0

        all_pos, all_scale, all_color = [], [], []
        for i, path in enumerate(imgs):
            try:
                with Image.open(path) as raw:
                    im = ImageOps.exif_transpose(raw)  # respect orientation
                    pos, scales, colors = self._photo_cloud(im, budget_each)
            except Exception:
                continue  # skip unreadable photo, keep going

            theta = (i - (m - 1) / 2.0) * step
            c, right, up, forward = self._panel_basis(theta)
            # world = centre + x*right + y*up + depth*forward; pos[:,2] = -depth
            world = (
                c
                + pos[:, 0:1] * right
                + pos[:, 1:2] * up
                + (-pos[:, 2:3]) * forward
            )
            world[:, 1] += _EYE_HEIGHT
            all_pos.append(world.astype(np.float32))
            all_scale.append(scales)
            all_color.append(colors)

            if progress_cb:
                frac = 0.08 + 0.88 * (i + 1) / m
                progress_cb(min(frac, 0.96), f"Lifting photo {i + 1} of {m} into 3D")

        if not all_pos:
            raise NotAvailable(
                "We couldn't read enough detail from these photos to build a 3D world. "
                "Try clearer photos with more variety."
            )

        positions = np.concatenate(all_pos, axis=0)
        scales = np.concatenate(all_scale, axis=0)
        colors = np.concatenate(all_color, axis=0)
        quats = np.zeros((positions.shape[0], 4), dtype=np.float32)
        quats[:, 0] = 1.0  # identity

        # Recenter horizontally on the content so the viewer starts well-placed.
        center_xz = positions[:, [0, 2]].mean(axis=0)
        positions[:, 0] -= center_xz[0]
        positions[:, 2] -= center_xz[1]

        asset = project_dir / "asset.splat"
        write_splats(
            asset,
            {"positions": positions, "scales": scales, "colors": colors, "quats": quats},
        )
        if progress_cb:
            progress_cb(1.0, "Your 3D world is ready to explore!")
        return asset
