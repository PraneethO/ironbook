"""Tests for direct .ply upload (trained 3DGS, transcoded to .splat server-side)."""
from __future__ import annotations

import numpy as np

# The vertex properties ply_to_splat reads from a trained 3DGS .ply.
_PROPS = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "opacity",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "f_dc_0", "f_dc_1", "f_dc_2",
]


def _make_ply_bytes(n: int = 10) -> bytes:
    """A minimal valid binary_little_endian 3DGS .ply with n gaussians."""
    rng = np.random.default_rng(0)
    dtype = np.dtype([(p, "<f4") for p in _PROPS])
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = rng.standard_normal((3, n)).astype("f4")
    for s in ("scale_0", "scale_1", "scale_2"):
        arr[s] = np.log(0.05).astype("f4")  # exp(...) -> 0.05 world units
    arr["opacity"] = 3.0  # sigmoid(3) ~ 0.95
    arr["rot_0"] = 1.0  # identity quaternion (w=1), normalized in the converter
    for f in ("f_dc_0", "f_dc_1", "f_dc_2"):
        arr[f] = 0.5
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in _PROPS)
        + "end_header\n"
    ).encode("ascii")
    return header + arr.tobytes()


def test_upload_ply_creates_ready_project(client):
    data = _make_ply_bytes(10)
    r = client.post(
        "/api/projects/upload_ply",
        files={"file": ("bike.ply", data, "application/octet-stream")},
        data={"name": "My PLY"},
    )
    assert r.status_code == 201, r.text
    proj = r.json()
    assert proj["name"] == "My PLY"
    assert proj["status"] == "ready"
    assert proj["has_asset"] is True
    pid = proj["id"]

    info = client.get(f"/api/projects/{pid}/asset/info")
    assert info.status_code == 200
    assert info.json()["splat_count"] == 10


def test_upload_ply_rejects_non_ply_extension(client):
    r = client.post(
        "/api/projects/upload_ply",
        files={"file": ("scene.splat", b"\x00" * 32, "application/octet-stream")},
    )
    assert r.status_code == 400


def test_upload_ply_rejects_garbage(client):
    r = client.post(
        "/api/projects/upload_ply",
        files={"file": ("scene.ply", b"not actually a ply file", "application/octet-stream")},
    )
    assert r.status_code == 400
