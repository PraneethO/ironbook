"""Tests for simple Brush sky-mask generation."""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.reconstruction.sky_masks import generate_sky_masks


def test_generate_sky_masks_masks_top_connected_sky(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()

    arr = np.zeros((80, 120, 3), dtype=np.uint8)
    arr[:30, :] = [90, 165, 245]  # sky
    arr[30:, :] = [40, 110, 90]  # object/ground
    arr[45:65, 20:70] = [90, 165, 245]  # blue object patch, not top-connected
    Image.fromarray(arr, "RGB").save(image_dir / "frame.jpg")

    assert generate_sky_masks(image_dir, mask_dir) == 1

    mask = np.asarray(Image.open(mask_dir / "frame.png").convert("L"))
    assert mask[5, 10] < 20
    assert mask[55, 30] > 230
    assert mask[70, 10] > 230


def test_generate_sky_masks_skips_non_sky(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    arr = np.full((80, 120, 3), [80, 80, 80], dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(image_dir / "frame.jpg")

    assert generate_sky_masks(image_dir, mask_dir) == 0
    assert not list(mask_dir.glob("*.png"))
