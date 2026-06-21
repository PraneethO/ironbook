"""Runtime configuration for the backend."""
from __future__ import annotations

import os
from pathlib import Path

# Storage root: backend/storage
BACKEND_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.environ.get("GSW_STORAGE_DIR", BACKEND_DIR / "storage"))

FRONTEND_BASE_URL = os.environ.get("GSW_FRONTEND_URL", "http://localhost:5173")

# When set, reconstruct jobs run synchronously to completion (used by tests).
RECONSTRUCT_INLINE = os.environ.get("RECONSTRUCT_INLINE", "0") == "1"

# Upload rules (CONTRACT.md / DECISIONS.md)
MIN_PHOTOS = 8           # minimum to attempt reconstruction
RECOMMENDED_PHOTOS = 20  # warn below this
MAX_PHOTOS = 400         # hard cap

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_EXTS = {".mp4", ".mov"}

# Processed-image and thumbnail sizes
PROCESSED_MAX = 1600
THUMB_MAX = 320


def ensure_storage() -> Path:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR
