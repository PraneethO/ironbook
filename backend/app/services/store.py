"""Persistent project store.

Each project lives under ``storage/<project_id>/`` with:
  - ``project.json``  : project record + upload metadata + job state
  - ``raw/``          : original uploaded files
  - ``processed/``    : downscaled images
  - ``thumbs/``       : thumbnails
  - ``asset.splat``   : generated viewer asset (after reconstruction)

An ``index.json`` at the storage root lists project ids in creation order so
``GET /api/projects`` is newest-first and survives restarts. All mutations go
through a single re-entrant lock, which is safe for the in-process worker
thread that updates job state concurrently with API requests.
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config

_LOCK = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def _index_path() -> Path:
    return config.ensure_storage() / "index.json"


def _project_dir(project_id: str) -> Path:
    return config.STORAGE_DIR / project_id


def _project_file(project_id: str) -> Path:
    return _project_dir(project_id) / "project.json"


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)


def _read_index() -> List[str]:
    p = _index_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _write_index(ids: List[str]) -> None:
    _atomic_write_json(_index_path(), ids)


# --- public API ----------------------------------------------------------


def project_dir(project_id: str) -> Path:
    return _project_dir(project_id)


def create_project(name: str) -> Dict[str, Any]:
    with _LOCK:
        pid = new_id()
        ts = now_iso()
        d = _project_dir(pid)
        (d / "raw").mkdir(parents=True, exist_ok=True)
        (d / "processed").mkdir(parents=True, exist_ok=True)
        (d / "thumbs").mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = {
            "id": pid,
            "name": name,
            "status": "draft",
            "created_at": ts,
            "updated_at": ts,
            "photo_count": 0,
            "thumbnail_url": None,
            "has_asset": False,
            # internal extras (not part of the Project schema response):
            "uploads": [],   # list of UploadItem-like dicts
            "job": None,     # Job-like dict
        }
        _atomic_write_json(_project_file(pid), record)
        idx = _read_index()
        idx.append(pid)
        _write_index(idx)
        return record


def load_project(project_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        p = _project_file(project_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None


def save_project(record: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        record["updated_at"] = now_iso()
        _atomic_write_json(_project_file(record["id"]), record)
        return record


def update_project(project_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    with _LOCK:
        record = load_project(project_id)
        if record is None:
            return None
        record.update(fields)
        return save_project(record)


def list_projects() -> List[Dict[str, Any]]:
    """Newest-first list of project records."""
    with _LOCK:
        ids = _read_index()
        out: List[Dict[str, Any]] = []
        # Index is append-order (oldest->newest); reverse for newest-first.
        for pid in reversed(ids):
            rec = load_project(pid)
            if rec is not None:
                out.append(rec)
        return out


def delete_project(project_id: str) -> bool:
    with _LOCK:
        d = _project_dir(project_id)
        if not d.exists():
            return False
        shutil.rmtree(d, ignore_errors=True)
        idx = _read_index()
        if project_id in idx:
            idx.remove(project_id)
            _write_index(idx)
        return True


def rebuild_index_from_disk() -> None:
    """Recover the index by scanning storage dirs (e.g. after a crash)."""
    with _LOCK:
        root = config.ensure_storage()
        existing = set(_read_index())
        found: List[tuple[str, str]] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            rec = load_project(child.name)
            if rec is None:
                continue
            found.append((rec.get("created_at", ""), child.name))
        found.sort()  # by created_at ascending
        ordered = [pid for _, pid in found]
        if set(ordered) != existing:
            _write_index(ordered)
