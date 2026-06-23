"""Generate simple top-connected sky masks for outdoor COLMAP datasets."""
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageFilter


_ENABLED = os.environ.get("GSW_SKY_MASK", "1") != "0"
_MIN_FRACTION = float(os.environ.get("GSW_SKY_MASK_MIN_FRACTION", "0.03"))
_MAX_FRACTION = float(os.environ.get("GSW_SKY_MASK_MAX_FRACTION", "0.65"))


def _sky_candidate(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    chroma = mx - mn

    # Bright blue/cyan sky. The top-connected pass below prevents similarly
    # colored object pixels from being masked unless they touch the top sky.
    blue = (b > r + 0.08) & (b > g + 0.02) & (b > 0.35) & (chroma > 0.08)
    pale = (mx > 0.72) & (chroma < 0.18) & (b >= r - 0.03) & (g >= r - 0.05)
    return blue | pale


def _top_connected(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    q: deque[Tuple[int, int]] = deque()
    for x in range(w):
        if mask[0, x]:
            out[0, x] = True
            q.append((0, x))
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def _mask_for_image(path: Path) -> Image.Image | None:
    with Image.open(path) as im:
        rgb_img = im.convert("RGB")
        small = rgb_img.copy()
        small.thumbnail((320, 240))
        rgb = np.asarray(small, dtype=np.uint8)

    sky = _top_connected(_sky_candidate(rgb))
    frac = float(sky.mean())
    if frac < _MIN_FRACTION or frac > _MAX_FRACTION:
        return None

    alpha_small = np.where(sky, 0, 255).astype(np.uint8)
    alpha = Image.fromarray(alpha_small, mode="L").resize(rgb_img.size, Image.Resampling.NEAREST)
    # Slight blur avoids a hard alpha edge at the skyline.
    return alpha.filter(ImageFilter.GaussianBlur(radius=1.0))


def generate_sky_masks(image_dir: Path, mask_dir: Path) -> int:
    """Write Brush-compatible masks for top-connected sky regions.

    Returns the number of masks written. Existing masks are left untouched.
    """
    if not _ENABLED:
        return 0

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    images: Iterable[Path] = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for image_path in images:
        out = mask_dir / f"{image_path.stem}.png"
        if out.exists():
            continue
        try:
            mask = _mask_for_image(image_path)
        except Exception:  # noqa: BLE001 - masks are an optimization, not fatal
            continue
        if mask is None:
            continue
        mask.save(out)
        written += 1
    return written
