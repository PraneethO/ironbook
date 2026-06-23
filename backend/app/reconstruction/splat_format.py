"""
Encode / decode the antimatter15-compatible `.splat` binary format.

Layout (CONTRACT.md §3): 32 bytes per splat, little-endian, no header,
splats concatenated.

| offset | type        | field                                            |
|--------|-------------|--------------------------------------------------|
| 0      | float32 x 3 | position x, y, z                                 |
| 12     | float32 x 3 | scale x, y, z (world units)                      |
| 24     | uint8  x 4  | color r, g, b, a (0..255)                        |
| 28     | uint8  x 4  | rotation quat, round((q+1)*128) clamped 0..255,  |
|        |             | order (w, x, y, z)                               |

splat_count = filesize / 32
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

SPLAT_BYTES = 32

# Structured dtype matching the on-disk record exactly (little-endian).
_RECORD = np.dtype(
    [
        ("pos", "<f4", (3,)),
        ("scale", "<f4", (3,)),
        ("color", "u1", (4,)),
        ("quat", "u1", (4,)),
    ]
)
assert _RECORD.itemsize == SPLAT_BYTES, "record must be exactly 32 bytes"


def _encode_quat(quat: np.ndarray) -> np.ndarray:
    """Encode float quaternions in [-1, 1] to uint8 via round((q+1)*128) clamped 0..255.

    Input shape (N, 4) in (w, x, y, z) order; output uint8 (N, 4) same order.
    """
    q = np.asarray(quat, dtype=np.float64)
    encoded = np.round((q + 1.0) * 128.0)
    encoded = np.clip(encoded, 0, 255)
    return encoded.astype(np.uint8)


def _decode_quat(quat_u8: np.ndarray) -> np.ndarray:
    """Inverse of _encode_quat: u8 -> float in roughly [-1, 1]."""
    q = np.asarray(quat_u8, dtype=np.float64)
    return (q / 128.0) - 1.0


def write_splats(path: str | Path, arrays: Dict[str, np.ndarray]) -> int:
    """Write splats to `path` in the contract format. Returns splat_count.

    `arrays` keys:
      - positions: (N, 3) float
      - scales:    (N, 3) float (world units)
      - colors:    (N, 4) uint8-ish (r, g, b, a) 0..255
      - quats:     (N, 4) float in [-1, 1], order (w, x, y, z).
                   Optional; defaults to identity rotation (1, 0, 0, 0).
    """
    positions = np.asarray(arrays["positions"], dtype=np.float32).reshape(-1, 3)
    n = positions.shape[0]

    scales = np.asarray(arrays["scales"], dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(arrays["colors"]).reshape(-1, 4)
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    if "quats" in arrays and arrays["quats"] is not None:
        quats = np.asarray(arrays["quats"], dtype=np.float64).reshape(-1, 4)
    else:
        quats = np.zeros((n, 4), dtype=np.float64)
        quats[:, 0] = 1.0  # identity (w=1, x=y=z=0)

    if not (n == scales.shape[0] == colors.shape[0] == quats.shape[0]):
        raise ValueError("all splat arrays must share the same length")

    records = np.zeros(n, dtype=_RECORD)
    records["pos"] = positions
    records["scale"] = scales
    records["color"] = colors
    records["quat"] = _encode_quat(quats)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(records.tobytes())
    return n


def read_splats(path: str | Path) -> Dict[str, np.ndarray]:
    """Read a `.splat` file. Returns dict with positions/scales/colors/quats.

    `quats` is returned as decoded floats (w, x, y, z).
    """
    data = Path(path).read_bytes()
    return decode_buffer(data)


def decode_buffer(data: bytes) -> Dict[str, np.ndarray]:
    if len(data) % SPLAT_BYTES != 0:
        raise ValueError(
            f"buffer length {len(data)} is not a multiple of {SPLAT_BYTES}"
        )
    records = np.frombuffer(data, dtype=_RECORD)
    return {
        "positions": records["pos"].astype(np.float32),
        "scales": records["scale"].astype(np.float32),
        "colors": records["color"].astype(np.uint8),
        "quats": _decode_quat(records["quat"]),
    }


def splat_count(path: str | Path) -> int:
    return Path(path).stat().st_size // SPLAT_BYTES


def bounds(positions: np.ndarray) -> Tuple[list, list]:
    """Return (min[xyz], max[xyz]) of a positions array as plain python lists."""
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if pos.shape[0] == 0:
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    mn = pos.min(axis=0).astype(float).tolist()
    mx = pos.max(axis=0).astype(float).tolist()
    return (mn, mx)
