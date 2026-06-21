# Gaussian Splat World — Backend

FastAPI backend for the "Gaussian Splat World" web app. Upload photos, get a
navigable `.splat` 3D world out, with friendly staged progress.

Implements [`../CONTRACT.md`](../CONTRACT.md) exactly. Runs on a **CPU fallback
reconstructor** here (no GPU / COLMAP / ffmpeg required); the real
COLMAP + gsplat path plugs in behind the same `ReconstructionBackend` interface
when a GPU server is available.

## Requirements

- macOS / Linux, Python 3.13
- Pure pip wheels only: fastapi, uvicorn, python-multipart, pillow, numpy, httpx, pytest
- No torch / COLMAP / ffmpeg needed at runtime

## Setup

```bash
cd backend

# Create an isolated virtualenv
python3.13 -m venv .venv

# Install dependencies
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

## Run the server

```bash
./.venv/bin/uvicorn app.main:app --reload --port 8000
```

- API base: `http://localhost:8000/api`
- Health check: `http://localhost:8000/api/health`
  → `{"status":"ok","reconstruction_backend":"fallback"}`
- Interactive docs: `http://localhost:8000/docs`
- CORS allows `http://localhost:5173` (the Vite dev server) and `*` in dev.

## Run the tests

```bash
./.venv/bin/python -m pytest -q
```

All tests use FastAPI's `TestClient` with an isolated temp storage dir and
`RECONSTRUCT_INLINE=1`, so jobs run synchronously and fast. They generate their
own synthetic images with Pillow — no external fixtures required.

## How reconstruction works

- `POST /api/projects/{id}/reconstruct` enqueues a job and a background worker
  thread walks the contract stages (preprocessing → pose_estimation → structure
  → optimization → compression → viewer_asset), updating progress and appending
  friendly logs, then writes `asset.splat`.
- Set `RECONSTRUCT_INLINE=1` to run the job synchronously to completion (used by
  the test suite and handy for scripting).
- The default `FallbackReconstructor` builds a real navigable scene from the
  uploaded photos: a colored floor + four walls + a central point cloud, all
  sampled from photo pixels. It is deterministic, Y-up, centered near the
  origin, with the floor near y=0, and emits tens of thousands of valid splats.

## Environment variables

| Var                  | Default                   | Meaning                                        |
|----------------------|---------------------------|------------------------------------------------|
| `GSW_STORAGE_DIR`    | `backend/storage`         | Per-project storage root                       |
| `GSW_FRONTEND_URL`   | `http://localhost:5173`   | Used to build share links                      |
| `RECONSTRUCT_INLINE` | `0`                       | `1` runs reconstruct jobs synchronously         |

## Storage layout

```
storage/
  index.json                 # project ids (newest-first listing, survives restart)
  <project_id>/
    project.json             # project record + upload metadata + job state
    raw/                     # original uploads
    processed/               # downscaled images (input to reconstruction)
    thumbs/                  # thumbnails
    asset.splat              # generated viewer asset (after reconstruction)
```

## Layout

```
app/
  main.py                    # FastAPI app, CORS, router mounting, startup
  config.py                  # paths, thresholds, env flags
  models.py                  # Pydantic schemas matching the contract
  routers/
    health.py                # GET /api/health
    projects.py              # project CRUD, uploads, reconstruct, asset, share
  services/
    store.py                 # persistent JSON project store (thread-safe)
    images.py                # save/downscale/thumbnail + sharpness (var-of-Laplacian)
    validation.py            # coverage/quality heuristics + friendly warnings
    jobs.py                  # staged job queue + background worker thread
  reconstruction/
    base.py                  # ReconstructionBackend ABC + NotAvailable
    splat_format.py          # 32-byte .splat encode/decode (CONTRACT.md §3)
    fallback.py              # FallbackReconstructor (default, CPU)
    colmap_gsplat.py         # real COLMAP+gsplat backend (stub, GPU integration point)
tests/                       # pytest suite (TestClient, synthetic images)
```
