"""Tests for COLMAP sparse point cloud sanitization."""
from __future__ import annotations

import struct

import numpy as np

from app.reconstruction.colmap_sanitize import sanitize_sparse_points


def _write_points(path, xyz):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(xyz)))
        for i, p in enumerate(xyz, start=1):
            fh.write(
                struct.pack(
                    "<QdddBBBd",
                    i,
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    128,
                    128,
                    128,
                    0.5,
                )
            )
            # One two-entry track: (image_id, point2d_idx) pairs.
            fh.write(struct.pack("<Q", 2))
            fh.write(struct.pack("<ii", 1, i))
            fh.write(struct.pack("<ii", 2, i))


def _read_count(path):
    with open(path, "rb") as fh:
        return struct.unpack("<Q", fh.read(8))[0]


def test_sanitize_sparse_points_removes_far_tail(tmp_path):
    rng = np.random.default_rng(4)
    core = rng.normal(0, 1.0, (500, 3))
    far = rng.normal(80, 1.0, (10, 3))
    sparse = tmp_path / "sparse" / "1"
    points = sparse / "points3D.bin"
    _write_points(points, np.vstack([core, far]))

    stats = sanitize_sparse_points(sparse)

    assert stats.applied is True
    assert stats.before == 510
    assert stats.after == 500
    assert stats.dropped_far == 10
    assert _read_count(points) == 500
    assert (sparse / "points3D.original.bin").exists()


def test_sanitize_sparse_points_safety_rail(tmp_path):
    rng = np.random.default_rng(5)
    core = rng.normal(0, 1.0, (100, 3))
    far = rng.normal(80, 1.0, (100, 3))
    sparse = tmp_path / "sparse" / "1"
    points = sparse / "points3D.bin"
    _write_points(points, np.vstack([core, far]))

    stats = sanitize_sparse_points(sparse)

    assert stats.applied is False
    assert stats.before == stats.after == 200
    assert _read_count(points) == 200
    assert not (sparse / "points3D.original.bin").exists()
