"""Tests for the on-device depth reconstructor.

These are skipped automatically if the ONNX depth model isn't present, so the
suite still passes on a clean checkout without the ~94MB model download.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.reconstruction.depth_reconstructor import DepthReconstructor
from app.reconstruction.splat_format import read_splats

_be = DepthReconstructor()
pytestmark = pytest.mark.skipif(
    not _be.is_available(), reason="depth model / onnxruntime not available"
)


def _make_photos(tmp: Path, kind: str, n: int = 6) -> list[Path]:
    paths = []
    for i in range(n):
        im = Image.new("RGB", (640, 480), (40, 40, 60))
        d = ImageDraw.Draw(im)
        if kind == "A":
            for t in range(11):
                f = t / 10
                x0 = min(310, 320 - 300 * (1 - f) + i * 4)
                x1 = max(x0 + 4, 320 + 20 * (1 - f))
                d.rectangle([x0, 240 - 180 * (1 - f), x1, 240 + 180 * (1 - f)],
                            outline=(210, 190, 150), width=3)
        else:
            for y in range(0, 480, 16):
                d.line([(0, y), (640, y)], fill=(70, 90, 120), width=2)
            cx = 120 + i * 20
            d.ellipse([cx, 120, cx + 300, 420], fill=(220, 90, 60))
        p = tmp / f"{kind}_{i:02d}.jpg"
        im.save(p, quality=90)
        paths.append(p)
    return paths


def _geom_fp(positions: np.ndarray) -> str:
    return hashlib.md5(np.round(positions, 2).tobytes()).hexdigest()


def test_produces_valid_nonempty_splat(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    imgs = _make_photos(tmp_path, "A")
    asset = _be.reconstruct(proj, imgs, progress_cb=None)
    assert asset.exists()
    assert asset.stat().st_size % 32 == 0
    sp = read_splats(asset)
    assert sp["positions"].shape[0] > 1000
    # colors are real (sampled from photos), not all black
    assert sp["colors"][:, :3].max() > 0
    # finite, sane bounds
    assert np.isfinite(sp["positions"]).all()


def test_different_photos_give_different_geometry(tmp_path):
    pa, pb = tmp_path / "a", tmp_path / "b"
    pa.mkdir(); pb.mkdir()
    a = read_splats(_be.reconstruct(pa, _make_photos(tmp_path, "A"), None))["positions"]
    b = read_splats(_be.reconstruct(pb, _make_photos(tmp_path, "B"), None))["positions"]
    assert _geom_fp(a) != _geom_fp(b), "geometry must depend on the input photos"


def test_deterministic_for_same_photos(tmp_path):
    imgs = _make_photos(tmp_path, "A")
    p1, p2 = tmp_path / "1", tmp_path / "2"
    p1.mkdir(); p2.mkdir()
    a = read_splats(_be.reconstruct(p1, imgs, None))["positions"]
    b = read_splats(_be.reconstruct(p2, imgs, None))["positions"]
    assert np.array_equal(a, b)


def test_progress_callback_reaches_one(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    seen = []
    _be.reconstruct(proj, _make_photos(tmp_path, "A"), lambda f, m=None: seen.append(f))
    assert seen and seen[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in seen)
