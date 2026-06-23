"""Unit tests for the .splat binary codec (CONTRACT.md §3)."""
from __future__ import annotations

import numpy as np

from app.reconstruction import splat_format


def test_encode_decode_round_trip(tmp_path):
    n = 500
    rng = np.random.default_rng(7)
    positions = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
    scales = rng.uniform(0.01, 0.2, (n, 3)).astype(np.float32)
    colors = rng.integers(0, 256, (n, 4)).astype(np.uint8)
    # Normalized quaternions in (w, x, y, z) order, in [-1, 1].
    quats = rng.uniform(-1, 1, (n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    path = tmp_path / "a.splat"
    count = splat_format.write_splats(
        path,
        {"positions": positions, "scales": scales, "colors": colors, "quats": quats},
    )
    assert count == n

    raw = path.read_bytes()
    assert len(raw) % splat_format.SPLAT_BYTES == 0
    assert len(raw) // splat_format.SPLAT_BYTES == n

    out = splat_format.read_splats(path)
    # float32 fields are exact-ish round trip
    assert np.allclose(out["positions"], positions, atol=1e-5)
    assert np.allclose(out["scales"], scales, atol=1e-5)
    # colors exact
    assert np.array_equal(out["colors"], colors)
    # quats quantized to u8 => tolerance ~1/128
    assert np.allclose(out["quats"], quats, atol=1.0 / 128 + 1e-3)


def test_quat_encoding_formula():
    # q=0 -> round((0+1)*128)=128 ; q=1 -> 256 clamp 255 ; q=-1 -> 0
    quats = np.array([[0.0, 1.0, -1.0, 0.5]])
    pos = np.zeros((1, 3), np.float32)
    arrays = {"positions": pos, "scales": pos, "colors": np.zeros((1, 4)), "quats": quats}
    import io

    # Inspect the encoded bytes directly.
    encoded = splat_format._encode_quat(quats)[0]
    assert list(encoded) == [128, 255, 0, round((0.5 + 1) * 128)]


def test_default_identity_quat(tmp_path):
    n = 10
    arrays = {
        "positions": np.zeros((n, 3), np.float32),
        "scales": np.full((n, 3), 0.05, np.float32),
        "colors": np.full((n, 4), 200, np.uint8),
    }
    path = tmp_path / "id.splat"
    splat_format.write_splats(path, arrays)
    out = splat_format.read_splats(path)
    # identity (w=1) decodes near (1,0,0,0)
    assert np.allclose(out["quats"][:, 0], 1.0, atol=1.0 / 128 + 1e-3)
    assert np.allclose(out["quats"][:, 1:], 0.0, atol=1.0 / 128 + 1e-3)


def test_bounds_helper():
    pos = np.array([[-1, 0, 2], [3, -4, 5]], np.float32)
    mn, mx = splat_format.bounds(pos)
    assert mn == [-1.0, -4.0, 2.0]
    assert mx == [3.0, 0.0, 5.0]


def test_bad_buffer_length():
    import pytest

    with pytest.raises(ValueError):
        splat_format.decode_buffer(b"\x00" * 33)
