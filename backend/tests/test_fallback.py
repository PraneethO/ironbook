"""Tests for the CPU fallback reconstructor."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.reconstruction.fallback import FallbackReconstructor
from app.reconstruction import splat_format
from tests.conftest import make_image_bytes


def _write_images(d: Path, n: int) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = d / f"img_{i}.jpg"
        p.write_bytes(make_image_bytes(seed=i))
        paths.append(p)
    return paths


def test_fallback_produces_valid_nonempty_splat(tmp_path):
    imgs = _write_images(tmp_path / "in", 5)
    backend = FallbackReconstructor()
    assert backend.is_available()

    asset = backend.reconstruct(tmp_path, imgs)
    assert asset.exists()
    assert asset.name == "asset.splat"

    raw = asset.read_bytes()
    assert len(raw) > 0
    assert len(raw) % 32 == 0
    count = len(raw) // 32
    # Tens of thousands of splats (floor + 4 walls + cloud).
    assert count > 20000

    arrays = splat_format.read_splats(asset)
    mn, mx = splat_format.bounds(arrays["positions"])
    # Y-up, floor near y=0, centered near origin.
    assert mn[1] >= -0.5
    assert abs((mn[0] + mx[0]) / 2) < 1.0
    assert abs((mn[2] + mx[2]) / 2) < 1.0


def test_fallback_deterministic(tmp_path):
    imgs = _write_images(tmp_path / "in", 4)
    b = FallbackReconstructor()
    a1 = b.reconstruct(tmp_path / "p1", imgs)
    a2 = b.reconstruct(tmp_path / "p2", imgs)
    assert a1.read_bytes() == a2.read_bytes()


def test_progress_callback_called(tmp_path):
    imgs = _write_images(tmp_path / "in", 3)
    seen = []
    FallbackReconstructor().reconstruct(
        tmp_path, imgs, progress_cb=lambda f, m=None: seen.append(f)
    )
    assert seen and seen[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in seen)
