"""Clean COLMAP sparse points before they seed Gaussian Splatting.

COLMAP can register a long image sequence successfully while still producing a
small tail of badly triangulated 3D points. Those points are especially harmful
for Gaussian Splatting because trainers initialize splats from the sparse cloud
and then densify around whatever is there. A few sky/background points hundreds
of scene units away can turn into a large floater cloud.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import struct
from typing import List

import numpy as np


@dataclass(frozen=True)
class SparseSanitizeStats:
    before: int
    after: int
    far_radius: float
    dropped_far: int
    dropped_nonfinite: int
    applied: bool

    @property
    def dropped(self) -> int:
        return self.before - self.after


@dataclass
class _Point3D:
    point_id: int
    xyz: np.ndarray
    rgb: tuple[int, int, int]
    error: float
    track: bytes


_DIST_PCT = float(os.environ.get("GSW_COLMAP_SANITIZE_DIST_PCT", "95.0"))
_DIST_FACTOR = float(os.environ.get("GSW_COLMAP_SANITIZE_DIST_FACTOR", "2.0"))
_MIN_RADIUS = float(os.environ.get("GSW_COLMAP_SANITIZE_MIN_RADIUS", "8.0"))
_MAX_DROP_FRACTION = float(os.environ.get("GSW_COLMAP_SANITIZE_MAX_DROP_FRACTION", "0.35"))


def _read_points3d_bin(path: Path) -> List[_Point3D]:
    points: List[_Point3D] = []
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(n):
            raw = fh.read(43)
            if len(raw) != 43:
                raise ValueError(f"truncated COLMAP points3D.bin: {path}")
            vals = struct.unpack("<QdddBBBd", raw)
            track_len_raw = fh.read(8)
            if len(track_len_raw) != 8:
                raise ValueError(f"truncated COLMAP points3D.bin track length: {path}")
            track_len = struct.unpack("<Q", track_len_raw)[0]
            track = fh.read(track_len * 8)
            if len(track) != track_len * 8:
                raise ValueError(f"truncated COLMAP points3D.bin track: {path}")
            points.append(
                _Point3D(
                    point_id=vals[0],
                    xyz=np.array(vals[1:4], dtype=np.float64),
                    rgb=(vals[4], vals[5], vals[6]),
                    error=float(vals[7]),
                    track=track,
                )
            )
    return points


def _write_points3d_bin(path: Path, points: List[_Point3D]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(points)))
        for p in points:
            fh.write(
                struct.pack(
                    "<QdddBBBd",
                    p.point_id,
                    float(p.xyz[0]),
                    float(p.xyz[1]),
                    float(p.xyz[2]),
                    int(p.rgb[0]),
                    int(p.rgb[1]),
                    int(p.rgb[2]),
                    float(p.error),
                )
            )
            fh.write(struct.pack("<Q", len(p.track) // 8))
            fh.write(p.track)
    tmp.replace(path)


def sanitize_sparse_points(sparse_model: Path) -> SparseSanitizeStats:
    """Remove extreme sparse points from a COLMAP model in place.

    The cutoff is intentionally robust and scene-relative: center by median,
    take the 95th percentile radius, and allow twice that radius with a floor.
    If the heuristic would remove too much, it leaves the model unchanged.
    """
    points_path = Path(sparse_model) / "points3D.bin"
    if not points_path.exists():
        return SparseSanitizeStats(0, 0, 0.0, 0, 0, False)

    points = _read_points3d_bin(points_path)
    n = len(points)
    if n < 100:
        return SparseSanitizeStats(n, n, 0.0, 0, 0, False)

    xyz = np.stack([p.xyz for p in points], axis=0)
    finite = np.isfinite(xyz).all(axis=1)
    finite_xyz = xyz[finite]
    if finite_xyz.shape[0] < 100:
        return SparseSanitizeStats(n, n, 0.0, 0, int((~finite).sum()), False)

    center = np.median(finite_xyz, axis=0)
    dist = np.linalg.norm(xyz - center, axis=1)
    base_radius = float(np.percentile(dist[finite], _DIST_PCT))
    far_radius = max(_MIN_RADIUS, base_radius * _DIST_FACTOR)
    far = finite & (dist > far_radius)
    drop = (~finite) | far
    n_drop = int(drop.sum())

    if n_drop == 0 or n_drop > n * _MAX_DROP_FRACTION:
        return SparseSanitizeStats(n, n, far_radius, int(far.sum()), int((~finite).sum()), False)

    backup = points_path.with_name("points3D.original.bin")
    if not backup.exists():
        shutil.copy2(points_path, backup)

    kept = [p for p, should_drop in zip(points, drop) if not should_drop]
    _write_points3d_bin(points_path, kept)
    return SparseSanitizeStats(n, len(kept), far_radius, int(far.sum()), int((~finite).sum()), True)
