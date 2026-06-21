"""Convert a standard 3D Gaussian Splatting `.ply` into our 32-byte `.splat`.

The trainers (Inria graphdeco / gsplat / MPS ports) all emit the same PLY:
binary_little_endian, one vertex per gaussian, float32 properties:

    x, y, z,
    nx, ny, nz,                 (unused normals)
    f_dc_0, f_dc_1, f_dc_2,     (SH degree-0 / DC term, per channel)
    f_rest_0 .. f_rest_44,      (higher SH bands; ignored by our RGBA viewer)
    opacity,                    (pre-sigmoid logit)
    scale_0, scale_1, scale_2,  (log-scale)
    rot_0, rot_1, rot_2, rot_3  (quaternion, w-first, un-normalized)

Activations to get renderable values (the viewer stores RGBA + world-space
scale + normalized quat, so we bake them here):

    color_rgb = clip(0.5 + C0 * f_dc, 0, 1) * 255,  C0 = 0.28209479177387814
    alpha     = sigmoid(opacity) * 255
    scale     = exp(scale_i)
    quat      = normalize(rot)            order (w, x, y, z)

Higher SH bands encode view-dependent color; our `.splat` viewer is RGBA-only,
so we keep just the DC term (the base albedo). That matches how web splat
viewers render these files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .splat_format import write_splats

_SH_C0 = 0.28209479177387814


def _read_binary_ply(path: Path) -> np.ndarray:
    """Parse a binary_little_endian PLY with all-scalar vertex props.

    Returns a structured numpy array of the vertex element.
    """
    raw = Path(path).read_bytes()
    # Split header (ascii, ends at 'end_header\n') from the binary body.
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError("not a valid PLY (no end_header)")
    header = raw[:idx].decode("ascii", errors="replace").splitlines()
    body = raw[idx + len(marker):]

    fmt = None
    n_vertices = 0
    props: List[Tuple[str, str]] = []
    in_vertex = False
    _np_of = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
        "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
    }
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n_vertices = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            # property <type> <name>  (no list properties expected here)
            if parts[1] == "list":
                raise ValueError("list properties not supported for vertex")
            props.append((parts[2], _np_of[parts[1]]))

    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format: {fmt!r} (need binary_little_endian)")

    dtype = np.dtype([(name, t) for name, t in props])
    verts = np.frombuffer(body, dtype=dtype, count=n_vertices)
    return verts


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ply_to_splat(ply_path: str | Path, splat_path: str | Path) -> int:
    """Convert a 3DGS `.ply` to our `.splat`. Returns the splat count."""
    v = _read_binary_ply(Path(ply_path))
    names = v.dtype.names

    def col(*candidates: str) -> np.ndarray:
        for c in candidates:
            if c in names:
                return v[c].astype(np.float32)
        raise ValueError(f"PLY missing any of {candidates}")

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1)

    f_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)
    rgb = np.clip(0.5 + _SH_C0 * f_dc, 0.0, 1.0) * 255.0

    alpha = (_sigmoid(col("opacity")) * 255.0).reshape(-1, 1)
    colors = np.concatenate([rgb, alpha], axis=1)

    scales = np.exp(np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1))

    quat = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    quat = quat / norm  # (w, x, y, z), unit

    # Trainers occasionally emit a few degenerate gaussians (NaN/Inf scale or
    # position) — exp(nan) propagates and renders as garbage. Drop them.
    finite = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(quat).all(axis=1)
    )
    if not finite.all():
        xyz, scales, colors, quat = xyz[finite], scales[finite], colors[finite], quat[finite]

    return write_splats(
        splat_path,
        {"positions": xyz, "scales": scales, "colors": colors, "quats": quat},
    )
