"""Image ingest: save raw, make processed + thumbnail, compute metrics.

Sharpness uses the variance of a discrete Laplacian (focus measure) computed
with numpy on a grayscale image — high variance => sharp, low => blurry.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .. import config


class UnreadableImage(Exception):
    pass


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian of a grayscale float array."""
    g = gray.astype(np.float64)
    # 4-neighbour Laplacian via padded shifts (no SciPy dependency).
    lap = (
        -4.0 * g
        + np.pad(g, ((1, 0), (0, 0)))[:-1, :]
        + np.pad(g, ((0, 1), (0, 0)))[1:, :]
        + np.pad(g, ((0, 0), (1, 0)))[:, :-1]
        + np.pad(g, ((0, 0), (0, 1)))[:, 1:]
    )
    # Trim the border which is affected by padding.
    if lap.shape[0] > 2 and lap.shape[1] > 2:
        lap = lap[1:-1, 1:-1]
    return float(lap.var())


def _open_oriented(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)  # honor EXIF orientation
    return im


def process_image(
    raw_path: Path, processed_dir: Path, thumb_dir: Path, stem: str
) -> Tuple[int, int, float]:
    """Create processed + thumbnail copies, return (width, height, sharpness).

    Raises UnreadableImage if the file isn't a decodable image.
    """
    try:
        im = _open_oriented(raw_path)
        im = im.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise UnreadableImage(str(exc)) from exc

    width, height = im.size

    # Sharpness on a modest grayscale copy for speed/consistency.
    gray_im = im.copy()
    gray_im.thumbnail((512, 512))
    gray = np.asarray(gray_im.convert("L"))
    sharpness = variance_of_laplacian(gray)

    # Processed (downscaled) copy.
    processed = im.copy()
    processed.thumbnail((config.PROCESSED_MAX, config.PROCESSED_MAX))
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed.save(processed_dir / f"{stem}.jpg", "JPEG", quality=88)

    # Thumbnail.
    thumb = im.copy()
    thumb.thumbnail((config.THUMB_MAX, config.THUMB_MAX))
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb.save(thumb_dir / f"{stem}.jpg", "JPEG", quality=82)

    return width, height, round(sharpness, 3)


def cover_thumbnail_path(project_dir: Path) -> Optional[Path]:
    """Pick a cover thumbnail (first available) for the project."""
    thumbs = sorted((project_dir / "thumbs").glob("*.jpg"))
    return thumbs[0] if thumbs else None
