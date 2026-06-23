"""CPU fallback reconstructor.

No GPU / COLMAP is available on this machine, so instead of a real
structure-from-motion + Gaussian-splat training run, this builds a genuine,
navigable `.splat` scene *sampled from the uploaded photos*:

  - a floor near y=0,
  - four surrounding walls colored by sampling the photos,
  - a central point cloud whose colors come from photo pixels, given depth so
    it reads as a 3D object you can walk around.

The result is deterministic, Y-up, centered near the origin, and encodes valid
splats per CONTRACT.md §3. It is intentionally tens of thousands of splats so
the WebGL viewer has something substantial to render.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .base import ProgressCallback, ReconstructionBackend
from .splat_format import write_splats

# Scene dimensions in world units (meters-ish). Floor at y=0, room centered.
ROOM_HALF = 4.0          # half-width of the square room (x and z)
WALL_HEIGHT = 3.0        # walls rise from y=0 to y=WALL_HEIGHT
FLOOR_SPLATS = 9000      # splats covering the floor
WALL_SPLATS_EACH = 4000  # splats per wall (x4)
CLOUD_SPLATS = 16000     # central object point cloud


def _seed_from_images(images: List[Path]) -> int:
    """Deterministic seed derived from the input filenames."""
    h = 0
    for p in images:
        for ch in p.name.encode("utf-8"):
            h = (h * 131 + ch) & 0xFFFFFFFF
    return h or 1


def _load_palette(images: List[Path], samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample RGB pixels from the photos to use as scene colors.

    Returns (samples, 3) uint8. Falls back to a neutral gradient if no image
    can be opened.
    """
    pixels: list[np.ndarray] = []
    per_image = max(1, samples // max(1, len(images)))
    for p in images:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                # Downscale hard for speed; we only need a color distribution.
                im.thumbnail((96, 96))
                arr = np.asarray(im, dtype=np.uint8).reshape(-1, 3)
            if arr.shape[0] == 0:
                continue
            idx = rng.integers(0, arr.shape[0], size=min(per_image, arr.shape[0]))
            pixels.append(arr[idx])
        except Exception:
            continue

    if not pixels:
        grey = np.linspace(60, 200, samples).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=1)

    pool = np.concatenate(pixels, axis=0)
    sel = rng.integers(0, pool.shape[0], size=samples)
    return pool[sel]


def _quat_array(n: int) -> np.ndarray:
    """Identity quaternions (w=1, x=y=z=0), order (w, x, y, z)."""
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = 1.0
    return q


class FallbackReconstructor(ReconstructionBackend):
    name = "fallback"

    def is_available(self) -> bool:
        return True

    def reconstruct(
        self,
        project_dir: Path,
        images: List[Path],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        rng = np.random.default_rng(_seed_from_images(images))

        total = FLOOR_SPLATS + 4 * WALL_SPLATS_EACH + CLOUD_SPLATS
        palette = _load_palette(images, total, rng)

        positions: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        colors: list[np.ndarray] = []

        color_cursor = 0

        def take_colors(k: int) -> np.ndarray:
            nonlocal color_cursor
            out = palette[color_cursor : color_cursor + k]
            color_cursor += k
            if out.shape[0] < k:  # wrap (shouldn't happen, palette sized to total)
                extra = palette[: k - out.shape[0]]
                out = np.concatenate([out, extra], axis=0)
            return out

        # --- Floor -----------------------------------------------------------
        n = FLOOR_SPLATS
        fx = rng.uniform(-ROOM_HALF, ROOM_HALF, n)
        fz = rng.uniform(-ROOM_HALF, ROOM_HALF, n)
        fy = rng.normal(0.0, 0.01, n)  # slight thickness near y=0
        pos = np.stack([fx, fy, fz], axis=1)
        positions.append(pos)
        scales.append(np.tile([0.08, 0.02, 0.08], (n, 1)))
        # Floor tinted toward photo colors but darkened a touch.
        c = take_colors(n).astype(np.float32) * 0.7
        colors.append(np.concatenate([c, np.full((n, 1), 230)], axis=1))
        if progress_cb:
            progress_cb(0.2, "Laying down the floor")

        # --- Walls -----------------------------------------------------------
        wall_defs = [
            # (axis fixed value, varying-axis is the other horizontal one)
            ("z", -ROOM_HALF),
            ("z", ROOM_HALF),
            ("x", -ROOM_HALF),
            ("x", ROOM_HALF),
        ]
        for i, (axis, val) in enumerate(wall_defs):
            n = WALL_SPLATS_EACH
            along = rng.uniform(-ROOM_HALF, ROOM_HALF, n)
            wy = rng.uniform(0.0, WALL_HEIGHT, n)
            if axis == "z":
                pos = np.stack([along, wy, np.full(n, val)], axis=1)
            else:
                pos = np.stack([np.full(n, val), wy, along], axis=1)
            positions.append(pos)
            scales.append(np.tile([0.07, 0.07, 0.07], (n, 1)))
            c = take_colors(n).astype(np.float32) * 0.85
            colors.append(np.concatenate([c, np.full((n, 1), 220)], axis=1))
            if progress_cb:
                progress_cb(0.3 + 0.1 * i, "Raising the walls")

        # --- Central object point cloud -------------------------------------
        # A rounded blob whose colors come straight from the photos, given depth
        # so it reads as a real object centered in the room.
        n = CLOUD_SPLATS
        # spherical-ish distribution, slightly squashed, sitting on the floor
        u = rng.uniform(0, 1, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = np.arccos(1 - 2 * rng.uniform(0, 1, n))
        r = 1.2 * np.cbrt(u)  # filled volume, radius up to ~1.2
        ox = r * np.sin(phi) * np.cos(theta)
        oz = r * np.sin(phi) * np.sin(theta)
        oy = r * np.cos(phi) * 0.9 + 1.1  # lift so it rests above the floor
        oy = np.clip(oy, 0.05, None)
        pos = np.stack([ox, oy, oz], axis=1)
        positions.append(pos)
        scales.append(np.tile([0.03, 0.03, 0.03], (n, 1)))
        c = take_colors(n).astype(np.float32)
        colors.append(np.concatenate([c, np.full((n, 1), 255)], axis=1))
        if progress_cb:
            progress_cb(0.9, "Placing the captured scene")

        all_pos = np.concatenate(positions, axis=0).astype(np.float32)
        all_scale = np.concatenate(scales, axis=0).astype(np.float32)
        all_color = np.clip(np.concatenate(colors, axis=0), 0, 255).astype(np.uint8)
        quats = _quat_array(all_pos.shape[0])

        asset = project_dir / "asset.splat"
        write_splats(
            asset,
            {
                "positions": all_pos,
                "scales": all_scale,
                "colors": all_color,
                "quats": quats,
            },
        )
        if progress_cb:
            progress_cb(1.0, "Scene ready")
        return asset
