"""Shared pytest fixtures.

Each test gets an isolated temp storage dir and inline reconstruction, so the
suite never touches real storage and jobs complete synchronously and fast.
"""
from __future__ import annotations

import importlib
import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A FastAPI TestClient wired to an isolated storage dir + inline jobs."""
    monkeypatch.setenv("GSW_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("RECONSTRUCT_INLINE", "1")
    # API/contract tests use the fast, hermetic procedural backend — not the
    # real COLMAP+MLX pipeline (which needs real overlapping photos + minutes).
    monkeypatch.setenv("GSW_FORCE_BACKEND", "fallback")

    # Reimport modules so they pick up the patched env at import time.
    import app.config as config

    importlib.reload(config)
    import app.services.store as store

    importlib.reload(store)
    import app.services.jobs as jobs

    importlib.reload(jobs)
    import app.services.images as images

    importlib.reload(images)
    import app.services.validation as validation

    importlib.reload(validation)
    import app.routers.projects as projects_router

    importlib.reload(projects_router)
    import app.routers.health as health_router

    importlib.reload(health_router)
    import app.main as main

    importlib.reload(main)

    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


def make_image_bytes(
    seed: int = 0, size: Tuple[int, int] = (256, 256), sharp: bool = True
) -> bytes:
    """Generate a small synthetic JPEG. Sharp => high-frequency pattern."""
    rng = np.random.default_rng(seed)
    if sharp:
        # Hard checkerboard + noise => high Laplacian variance.
        x = np.indices(size)[1] // 8 % 2
        y = np.indices(size)[0] // 8 % 2
        base = ((x ^ y) * 255).astype(np.uint8)
        arr = np.stack([base, np.roll(base, 3, 0), np.roll(base, 5, 1)], axis=2)
        arr = (arr.astype(np.int16) + rng.integers(-20, 20, arr.shape)).clip(0, 255)
        arr = arr.astype(np.uint8)
    else:
        # Smooth gradient => low Laplacian variance (blurry).
        g = np.linspace(40, 90, size[1])[None, :].repeat(size[0], 0).astype(np.uint8)
        arr = np.stack([g, g, g], axis=2)
    im = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def upload_payload(n: int, sharp: bool = True, start: int = 0) -> List[tuple]:
    """Build a multipart `files` payload of n synthetic JPEGs."""
    files = []
    for i in range(n):
        data = make_image_bytes(seed=start + i, sharp=sharp)
        files.append(("files", (f"photo_{start + i}.jpg", data, "image/jpeg")))
    return files
