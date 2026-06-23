"""Tests for direct .splat upload (bring-your-own Gaussian-splat scene)."""
from __future__ import annotations

import numpy as np

from app.reconstruction import splat_format


def _make_splat_bytes(n: int = 50) -> bytes:
    rng = np.random.default_rng(0)
    arrays = {
        "positions": rng.standard_normal((n, 3)).astype("float32"),
        "scales": (np.ones((n, 3), dtype="float32") * 0.05),
        "colors": rng.integers(0, 255, (n, 4)).astype("uint8"),
    }
    import io
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.splat"
        splat_format.write_splats(p, arrays)
        return p.read_bytes()


def test_upload_splat_creates_ready_project(client):
    data = _make_splat_bytes(64)
    r = client.post(
        "/api/projects/upload_splat",
        files={"file": ("scene.splat", data, "application/octet-stream")},
        data={"name": "My Splat"},
    )
    assert r.status_code == 201, r.text
    proj = r.json()
    assert proj["name"] == "My Splat"
    assert proj["status"] == "ready"
    assert proj["has_asset"] is True
    pid = proj["id"]

    # asset is served and parses to the right count
    r = client.get(f"/api/projects/{pid}/asset")
    assert r.status_code == 200
    assert len(r.content) == len(data)

    r = client.get(f"/api/projects/{pid}/asset/info")
    assert r.status_code == 200
    assert r.json()["splat_count"] == 64


def test_upload_rejects_non_splat_extension(client):
    r = client.post(
        "/api/projects/upload_splat",
        files={"file": ("scene.txt", b"\x00" * 32, "text/plain")},
    )
    assert r.status_code == 400


def test_upload_rejects_misaligned_bytes(client):
    r = client.post(
        "/api/projects/upload_splat",
        files={"file": ("scene.splat", b"\x00" * 33, "application/octet-stream")},
    )
    assert r.status_code == 400
