"""Unit tests for post-training floater/haze cleanup."""
from __future__ import annotations

import numpy as np

from app.reconstruction.splat_cleanup import clean_splat

_STRIDE = 32


def _make_splat(path, pos, scale, alpha):
    """Write a minimal 32-byte .splat: pos(3f) scale(3f) rgba(4u8) quat(4u8)."""
    n = len(pos)
    a = np.zeros((n, _STRIDE), dtype=np.uint8)
    a[:, 0:12] = np.asarray(pos, np.float32).view(np.uint8).reshape(n, 12)
    a[:, 12:24] = np.asarray(scale, np.float32).view(np.uint8).reshape(n, 12)
    a[:, 24:27] = 128  # mid-grey color
    a[:, 27] = np.asarray(alpha, np.uint8)
    a[:, 28:32] = 128  # identity-ish quat bytes
    path.write_bytes(a.tobytes())


def _count(path):
    return path.stat().st_size // _STRIDE


def test_removes_far_floaters_keeps_core(tmp_path):
    rng = np.random.default_rng(1)
    n_core = 5000
    core_pos = rng.normal(0, 1.0, (n_core, 3)).astype(np.float32)
    core_scale = np.full((n_core, 3), 0.02, np.float32)
    core_alpha = np.full(n_core, 220, np.uint8)
    # 20 floaters flung far away (the kind that wreck viewer framing)
    far_pos = rng.uniform(80, 120, (20, 3)).astype(np.float32)
    far_scale = np.full((20, 3), 0.02, np.float32)
    far_alpha = np.full(20, 220, np.uint8)

    pos = np.vstack([core_pos, far_pos])
    scale = np.vstack([core_scale, far_scale])
    alpha = np.concatenate([core_alpha, far_alpha])
    p = tmp_path / "s.splat"
    _make_splat(p, pos, scale, alpha)

    before, after = clean_splat(p)
    assert before == n_core + 20
    assert after == n_core  # exactly the far floaters removed
    assert _count(p) == n_core


def test_removes_faint_and_haze(tmp_path):
    n = 4000
    pos = np.zeros((n, 3), np.float32)
    scale = np.full((n, 3), 0.02, np.float32)
    alpha = np.full(n, 200, np.uint8)
    # 100 faint (invisible) + 100 large-translucent haze planes
    alpha[:100] = 2
    scale[100:200] = 0.6
    alpha[100:200] = 30
    p = tmp_path / "s.splat"
    _make_splat(p, pos, scale, alpha)

    before, after = clean_splat(p)
    assert before == n
    assert after == n - 200


def test_strict_cleanup_can_remove_noisy_tail(tmp_path):
    rng = np.random.default_rng(3)
    n_core = 3000
    core_pos = rng.normal(0, 1.0, (n_core, 3)).astype(np.float32)
    core_scale = np.full((n_core, 3), 0.02, np.float32)
    core_alpha = np.full(n_core, 180, np.uint8)

    n_noise = 1000
    noise_pos = rng.normal(0, 12.0, (n_noise, 3)).astype(np.float32)
    noise_scale = np.full((n_noise, 3), 0.03, np.float32)
    noise_alpha = np.full(n_noise, 10, np.uint8)

    p = tmp_path / "s.splat"
    _make_splat(
        p,
        np.vstack([core_pos, noise_pos]),
        np.vstack([core_scale, noise_scale]),
        np.concatenate([core_alpha, noise_alpha]),
    )

    before, after = clean_splat(p, strict=True)
    assert before == n_core + n_noise
    assert after == n_core


def test_safety_rail_leaves_unusual_scene_untouched(tmp_path):
    # A scene where most points would match a heuristic must not be gutted.
    n = 2000
    pos = np.zeros((n, 3), np.float32)
    scale = np.full((n, 3), 0.02, np.float32)
    alpha = np.full(n, 2, np.uint8)  # everything "faint" -> >25% would drop
    p = tmp_path / "s.splat"
    _make_splat(p, pos, scale, alpha)

    before, after = clean_splat(p)
    assert before == after == n  # safety rail: untouched


def test_tiny_file_untouched(tmp_path):
    pos = np.zeros((10, 3), np.float32)
    scale = np.full((10, 3), 0.02, np.float32)
    alpha = np.full(10, 2, np.uint8)
    p = tmp_path / "s.splat"
    _make_splat(p, pos, scale, alpha)
    assert clean_splat(p) == (10, 10)
