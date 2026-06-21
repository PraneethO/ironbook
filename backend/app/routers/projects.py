"""Project CRUD + upload + reconstruct + asset endpoints (CONTRACT.md §2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .. import config
from ..models import (
    AssetInfo,
    Job,
    Project,
    ProjectCreate,
    ShareResponse,
    UploadItem,
    ValidationReport,
)
from ..reconstruction import splat_format
from ..services import images as image_svc
from ..services import jobs as job_svc
from ..services import store
from ..services import validation as validation_svc

router = APIRouter(prefix="/api/projects", tags=["projects"])

# Public Project fields per the contract (internal extras like uploads/job hidden).
_PUBLIC_FIELDS = (
    "id",
    "name",
    "status",
    "created_at",
    "updated_at",
    "photo_count",
    "thumbnail_url",
    "has_asset",
)


def _public(record: Dict[str, Any]) -> Project:
    data = {k: record.get(k) for k in _PUBLIC_FIELDS}
    # Only advertise a thumbnail URL if the cover image actually exists on disk.
    # Otherwise the dashboard's <img> fires a request that 404s (a common case
    # for projects that were created/failed before any thumbnail was generated).
    pid = record.get("id")
    if data.get("thumbnail_url") and pid:
        if image_svc.cover_thumbnail_path(store.project_dir(pid)) is None:
            data["thumbnail_url"] = None
    return Project(**data)


def _require(project_id: str) -> Dict[str, Any]:
    record = store.load_project(project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="We couldn't find that 3D world.")
    return record


@router.post("", response_model=Project, status_code=201)
def create_project(body: ProjectCreate) -> Project:
    record = store.create_project(body.name.strip())
    return _public(record)


@router.get("", response_model=List[Project])
def list_projects() -> List[Project]:
    return [_public(r) for r in store.list_projects()]


@router.get("/{project_id}", response_model=Project)
def get_project(project_id: str) -> Project:
    return _public(_require(project_id))


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str) -> Response:
    if not store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="We couldn't find that 3D world.")
    return Response(status_code=204)


# --- Direct .splat upload -------------------------------------------------


@router.post("/upload_splat", response_model=Project, status_code=201)
async def upload_splat(
    file: UploadFile = File(...),
    name: str = Form("Uploaded world"),
) -> Project:
    """Create a project directly from a pre-built `.splat` asset.

    Lets the user bring their own Gaussian-splat scene (any source) and view +
    navigate it with the agent, without going through photo reconstruction.
    """
    fname = file.filename or "scene.splat"
    if Path(fname).suffix.lower() not in config.SPLAT_EXTS:
        raise HTTPException(
            status_code=400,
            detail="Please upload a .splat file (the standard Gaussian-splat format).",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="That .splat file looked empty.")
    if len(data) % splat_format.SPLAT_BYTES != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "That doesn't look like a valid .splat file "
                f"(size must be a multiple of {splat_format.SPLAT_BYTES} bytes)."
            ),
        )

    record = store.create_project((name or "Uploaded world").strip() or "Uploaded world")
    pid = record["id"]
    asset_path = store.project_dir(pid) / "asset.splat"
    asset_path.write_bytes(data)

    record = store.load_project(pid) or record
    record["status"] = "ready"
    record["has_asset"] = True
    record["photo_count"] = 0
    store.save_project(record)
    return _public(record)


# --- Uploads --------------------------------------------------------------


@router.post("/{project_id}/uploads", response_model=ValidationReport)
async def upload_files(project_id: str, files: List[UploadFile]) -> ValidationReport:
    record = _require(project_id)
    store.update_project(project_id, status="uploading")

    pdir = store.project_dir(project_id)
    raw_dir = pdir / "raw"
    processed_dir = pdir / "processed"
    thumb_dir = pdir / "thumbs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    uploads: List[Dict[str, Any]] = list(record.get("uploads", []))
    existing_names = {u["filename"] for u in uploads}
    rejected: List[Dict[str, str]] = []

    for upload in files:
        name = upload.filename or "file"
        ext = Path(name).suffix.lower()

        if ext in config.VIDEO_EXTS:
            rejected.append(
                {
                    "filename": name,
                    "reason": (
                        "Video uploads need extra processing that isn't available "
                        "here yet. For now, please upload photos instead."
                    ),
                }
            )
            continue

        if ext not in config.IMAGE_EXTS:
            rejected.append(
                {
                    "filename": name,
                    "reason": "That file type isn't supported. Please use JPG, PNG, or HEIC photos.",
                }
            )
            continue

        if len(uploads) >= config.MAX_PHOTOS:
            rejected.append(
                {
                    "filename": name,
                    "reason": f"We already have the maximum of {config.MAX_PHOTOS} photos.",
                }
            )
            continue

        # Save raw bytes.
        data = await upload.read()
        if not data:
            rejected.append({"filename": name, "reason": "That file looked empty."})
            continue

        stem = f"{len(uploads):04d}_{Path(name).stem}"
        raw_path = raw_dir / f"{stem}{ext}"
        raw_path.write_bytes(data)

        try:
            width, height, sharpness = image_svc.process_image(
                raw_path, processed_dir, thumb_dir, stem
            )
        except image_svc.UnreadableImage:
            raw_path.unlink(missing_ok=True)
            rejected.append(
                {
                    "filename": name,
                    "reason": "We couldn't read that photo. It may be corrupted.",
                }
            )
            continue

        item = {
            "filename": name,
            "stem": stem,
            "thumbnail_url": f"/api/projects/{project_id}/uploads/{stem}/thumbnail",
            "width": width,
            "height": height,
            "sharpness": sharpness,
        }
        uploads.append(item)
        existing_names.add(name)

    # Persist updated uploads + cover thumbnail + photo_count.
    record = store.load_project(project_id) or record
    record["uploads"] = uploads
    record["photo_count"] = len(uploads)
    if uploads and not record.get("thumbnail_url"):
        record["thumbnail_url"] = f"/api/projects/{project_id}/thumbnail"
    record["status"] = "uploading" if uploads else "draft"
    store.save_project(record)

    report = validation_svc.build_report(uploads, rejected)
    return ValidationReport(**report)


@router.get("/{project_id}/uploads", response_model=List[UploadItem])
def list_uploads(project_id: str) -> List[UploadItem]:
    record = _require(project_id)
    return [
        UploadItem(
            filename=u["filename"],
            thumbnail_url=u["thumbnail_url"],
            width=u["width"],
            height=u["height"],
            sharpness=u["sharpness"],
        )
        for u in record.get("uploads", [])
    ]


@router.get("/{project_id}/uploads/{stem}/thumbnail")
def get_upload_thumbnail(project_id: str, stem: str) -> FileResponse:
    _require(project_id)
    path = store.project_dir(project_id) / "thumbs" / f"{stem}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="That photo isn't available.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{project_id}/thumbnail")
def get_project_thumbnail(project_id: str) -> FileResponse:
    _require(project_id)
    path = image_svc.cover_thumbnail_path(store.project_dir(project_id))
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="No cover photo yet.")
    return FileResponse(path, media_type="image/jpeg")


# --- Reconstruction job ---------------------------------------------------


@router.post("/{project_id}/reconstruct", response_model=Job)
def reconstruct(project_id: str) -> Job:
    record = _require(project_id)
    photo_count = record.get("photo_count", 0)
    if photo_count < config.MIN_PHOTOS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"We need at least {config.MIN_PHOTOS} photos to build a 3D world. "
                f"Please add more from different angles."
            ),
        )
    job = job_svc.enqueue(project_id)
    return Job(**job)


@router.get("/{project_id}/job", response_model=Job)
def get_job(project_id: str) -> Job:
    record = _require(project_id)
    job = record.get("job")
    if job is None:
        raise HTTPException(
            status_code=404, detail="This 3D world hasn't started building yet."
        )
    return Job(**job)


# --- Assets ---------------------------------------------------------------


@router.get("/{project_id}/asset")
def get_asset(project_id: str) -> FileResponse:
    _require(project_id)
    path = store.project_dir(project_id) / "asset.splat"
    if not path.exists():
        raise HTTPException(status_code=404, detail="This 3D world isn't ready yet.")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{project_id}.splat",
    )


@router.get("/{project_id}/asset/info", response_model=AssetInfo)
def get_asset_info(project_id: str) -> AssetInfo:
    _require(project_id)
    path = store.project_dir(project_id) / "asset.splat"
    if not path.exists():
        raise HTTPException(status_code=404, detail="This 3D world isn't ready yet.")
    arrays = splat_format.read_splats(path)
    mn, mx = splat_format.bounds(arrays["positions"])
    return AssetInfo(
        splat_count=splat_format.splat_count(path),
        bytes=path.stat().st_size,
        bounds={"min": mn, "max": mx},
        format="splat",
    )


@router.get("/{project_id}/share", response_model=ShareResponse)
def share(project_id: str) -> ShareResponse:
    _require(project_id)
    return ShareResponse(url=f"{config.FRONTEND_BASE_URL}/view/{project_id}")
