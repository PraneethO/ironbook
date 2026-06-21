"""Reconstruction job queue + in-process worker thread.

A job walks the fixed STAGES from CONTRACT.md, updating overall + per-stage
progress and appending friendly logs, then writes ``asset.splat`` and flips the
job + project to ``ready``. Errors are reported with friendly, jargon-free
messages.

A single daemon worker thread pulls jobs off a queue. Tests can set
``RECONSTRUCT_INLINE=1`` (or pass ``inline=True``) to run a job synchronously to
completion fast.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

import sentry_sdk

from .. import config
from ..reconstruction import NotAvailable, ReconstructionBackend, select_backend
from . import store

# Ordered stages: (key, friendly label). Matches CONTRACT.md §1.
STAGES: List[tuple[str, str]] = [
    ("preprocessing", "Preparing your photos"),
    ("pose_estimation", "Finding camera positions"),
    ("structure", "Building rough 3D structure"),
    ("optimization", "Optimizing visual detail"),
    ("compression", "Compressing the scene"),
    ("viewer_asset", "Preparing interactive viewer"),
]

# Simulated per-stage durations (seconds) for the background path. Kept short so
# the demo feels responsive; inline mode skips sleeping entirely.
_STAGE_SECONDS = {
    "preprocessing": 0.6,
    "pose_estimation": 1.2,
    "structure": 0.9,
    "optimization": 1.8,
    "compression": 0.5,
    "viewer_asset": 0.5,
}

_BACKEND: ReconstructionBackend = select_backend()


def backend_name() -> str:
    return _BACKEND.name


def _new_job(project_id: str) -> dict:
    return {
        "project_id": project_id,
        "status": "queued",
        "progress": 0.0,
        "current_stage": None,
        "stages": [
            {"key": k, "label": label, "status": "pending", "progress": 0.0}
            for k, label in STAGES
        ],
        "logs": [],
        "error": None,
    }


def _log(job: dict, level: str, stage: str, message: str) -> None:
    job["logs"].append(
        {
            "ts": store.now_iso(),
            "level": level,
            "stage": stage,
            "message": message,
        }
    )


def _persist(project_id: str, job: dict, project_status: Optional[str] = None) -> None:
    record = store.load_project(project_id)
    if record is None:
        return
    record["job"] = job
    if project_status is not None:
        record["status"] = project_status
    store.save_project(record)


def enqueue(project_id: str, inline: Optional[bool] = None) -> dict:
    """Create a queued job for the project and schedule it.

    Returns the freshly-created Job dict (status ``queued``).
    """
    job = _new_job(project_id)
    _log(job, "info", "preprocessing", "Your 3D world is in the queue.")
    _persist(project_id, job, project_status="queued")

    run_inline = config.RECONSTRUCT_INLINE if inline is None else inline
    if run_inline:
        _run_job(project_id, inline=True)
        refreshed = store.load_project(project_id)
        return refreshed["job"] if refreshed else job
    else:
        _WORKER.submit(project_id)
        return job


def _input_images(project_id: str) -> List[Path]:
    d = store.project_dir(project_id) / "processed"
    return sorted(d.glob("*.jpg"))


_STAGE_KEYS = [k for k, _ in STAGES]


def _run_job(project_id: str, inline: bool = False) -> None:
    record = store.load_project(project_id)
    if record is None:
        return
    job = record.get("job") or _new_job(project_id)

    n_stages = len(STAGES)
    # Throttle disk writes: the callback can fire dozens of times/sec.
    last_persist = [0.0]

    def _maybe_persist(force: bool = False) -> None:
        now = time.monotonic()
        if force or now - last_persist[0] >= 0.25:
            last_persist[0] = now
            _persist(project_id, job)

    def _stage_index(stage: Optional[str]) -> int:
        if stage in _STAGE_KEYS:
            return _STAGE_KEYS.index(stage)
        return _STAGE_KEYS.index("optimization")

    def progress_cb(frac: float, msg: Optional[str] = None, stage: Optional[str] = None) -> None:
        """Backend-driven progress. `stage` is one of the STAGES keys (or None
        => optimization). Earlier stages are auto-completed so the bar advances
        monotonically across the real pipeline."""
        idx = _stage_index(stage)
        for j in range(idx):
            job["stages"][j]["status"] = "done"
            job["stages"][j]["progress"] = 1.0
        st = job["stages"][idx]
        st["status"] = "active"
        st["progress"] = float(max(0.0, min(1.0, frac)))
        job["status"] = "processing"
        job["current_stage"] = _STAGE_KEYS[idx]
        job["progress"] = round((idx + st["progress"]) / n_stages, 3)
        if msg:
            _log(job, "info", _STAGE_KEYS[idx], msg)
            _maybe_persist(force=True)
        else:
            _maybe_persist()

    with sentry_sdk.start_transaction(
        op="job.reconstruction",
        name="reconstruction",
        sampled=True,
    ) as txn:
        txn.set_tag("project_id", project_id)
        txn.set_tag("backend", _BACKEND.name)

        try:
            with sentry_sdk.start_span(op="stage.preprocessing", description="Preparing photos"):
                job["status"] = "processing"
                job["current_stage"] = "preprocessing"
                job["stages"][0]["status"] = "active"
                _log(job, "info", "preprocessing", "Preparing your photos…")
                _persist(project_id, job, project_status="processing")
                job["stages"][0]["status"] = "done"
                job["stages"][0]["progress"] = 1.0
                job["progress"] = round(1 / n_stages, 3)
                _persist(project_id, job)

            images = _input_images(project_id)
            txn.set_data("image_count", len(images))

            with sentry_sdk.start_span(op="stage.reconstruction", description="3D reconstruction"):
                _BACKEND.reconstruct(store.project_dir(project_id), images, progress_cb)

            # All real work done — mark every stage complete.
            for st in job["stages"]:
                st["status"] = "done"
                st["progress"] = 1.0
            job["status"] = "ready"
            job["current_stage"] = None
            job["progress"] = 1.0
            _log(job, "info", "viewer_asset", "Your 3D world is ready to explore!")
            record = store.load_project(project_id) or record
            record["job"] = job
            record["status"] = "ready"
            record["has_asset"] = (store.project_dir(project_id) / "asset.splat").exists()
            store.save_project(record)
            txn.set_tag("status", "success")

        except NotAvailable as exc:
            sentry_sdk.capture_exception(exc)
            txn.set_tag("status", "failed")
            txn.set_tag("failure_reason", "backend_unavailable")
            _fail(project_id, job, str(exc))
        except Exception as exc:  # noqa: BLE001 - convert to friendly message
            sentry_sdk.capture_exception(exc)
            txn.set_tag("status", "failed")
            txn.set_tag("failure_stage", job.get("current_stage") or "unknown")
            _fail(
                project_id,
                job,
                "Something went wrong while building your 3D world. "
                "Try uploading more photos from different angles.",
            )
            # Keep the technical detail only in logs for power users.
            _log(job, "error", job.get("current_stage") or "optimization", f"detail: {exc}")
            _persist(project_id, job)


def _fail(project_id: str, job: dict, friendly: str) -> None:
    job["status"] = "failed"
    cur = job.get("current_stage")
    for st in job["stages"]:
        if st["key"] == cur:
            st["status"] = "failed"
    job["error"] = friendly
    _log(job, "error", cur or "optimization", friendly)
    _persist(project_id, job, project_status="failed")


class _Worker:
    """Single daemon thread that processes queued jobs serially."""

    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._loop, name="reconstruct-worker", daemon=True
                )
                self._thread.start()

    def _loop(self) -> None:
        while True:
            project_id = self._q.get()
            try:
                _run_job(project_id, inline=False)
            finally:
                self._q.task_done()

    def submit(self, project_id: str) -> None:
        self._ensure_thread()
        self._q.put(project_id)


_WORKER = _Worker()
