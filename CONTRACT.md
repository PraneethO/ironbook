# Integration Contract — Gaussian Splat World App

This is the **single source of truth** that all components must agree on. Backend, viewer
engine, and frontend UI are built independently against this contract. Do not change a
shared interface without it being reflected here.

Base URLs (dev):
- Backend API: `http://localhost:8000`
- Frontend dev server: `http://localhost:5173`
- All API routes are prefixed with `/api`.
- CORS: backend allows origin `http://localhost:5173` (and `*` in dev).

---

## 1. Domain model

### Project
```json
{
  "id": "string (uuid hex)",
  "name": "string",
  "status": "draft | uploading | queued | processing | ready | failed",
  "created_at": "ISO-8601 string",
  "updated_at": "ISO-8601 string",
  "photo_count": 0,
  "thumbnail_url": "string | null",   // /api/projects/{id}/thumbnail
  "has_asset": false
}
```

### Job (reconstruction progress)
A project has at most one active/most-recent job.
```json
{
  "project_id": "string",
  "status": "queued | processing | ready | failed",
  "progress": 0.0,                 // 0..1 overall
  "current_stage": "string key",   // one of STAGES[].key below, or null
  "stages": [
    {
      "key": "preprocessing",
      "label": "Preparing your photos",   // friendly, user-facing
      "status": "pending | active | done | failed",
      "progress": 0.0                       // 0..1 within this stage
    }
  ],
  "logs": [ { "ts": "ISO-8601", "level": "info|warn|error", "stage": "string", "message": "string" } ],
  "error": "string | null"          // friendly message; never raw COLMAP/SfM jargon
}
```

### STAGES (ordered, friendly labels per 04_user_experience.md)
| key              | label (user-facing)            | technical meaning            |
|------------------|--------------------------------|------------------------------|
| `preprocessing`  | Preparing your photos          | normalize/downscale/EXIF     |
| `pose_estimation`| Finding camera positions       | SfM / COLMAP poses           |
| `structure`      | Building rough 3D structure    | sparse point cloud           |
| `optimization`   | Optimizing visual detail       | Gaussian splat training      |
| `compression`    | Compressing the scene          | compress splat asset         |
| `viewer_asset`   | Preparing interactive viewer   | build .splat for viewer      |

When all stages are `done`, job.status = `ready` and project.status = `ready`.

### Validation report (returned from upload)
```json
{
  "accepted": 0,
  "rejected": [ { "filename": "string", "reason": "friendly string" } ],
  "photo_count": 0,
  "coverage_score": 0.0,        // 0..1
  "quality_score": 0.0,         // 0..1 (avg sharpness-based)
  "warnings": [ "We have many front views but few side views ..." ],
  "ready_to_reconstruct": true  // false if too few photos, etc.
}
```

---

## 2. REST API

| Method | Path                                   | Body / Query                         | Returns |
|--------|----------------------------------------|--------------------------------------|---------|
| GET    | `/api/health`                          | —                                    | `{ "status": "ok", "reconstruction_backend": "fallback|colmap_gsplat" }` |
| POST   | `/api/projects`                        | `{ "name": "string" }`               | Project |
| GET    | `/api/projects`                        | —                                    | `Project[]` (newest first) |
| GET    | `/api/projects/{id}`                   | —                                    | Project |
| DELETE | `/api/projects/{id}`                   | —                                    | `204` |
| POST   | `/api/projects/{id}/uploads`           | `multipart/form-data` field `files`  | ValidationReport |
| GET    | `/api/projects/{id}/uploads`           | —                                    | `[ { "filename","thumbnail_url","width","height","sharpness" } ]` |
| GET    | `/api/projects/{id}/thumbnail`         | —                                    | image bytes (project cover) or 404 |
| POST   | `/api/projects/{id}/reconstruct`       | —                                    | Job (status `queued`) |
| GET    | `/api/projects/{id}/job`               | —                                    | Job |
| GET    | `/api/projects/{id}/asset`             | —                                    | `.splat` binary (Content-Type `application/octet-stream`), or 404 |
| GET    | `/api/projects/{id}/asset/info`        | —                                    | `{ "splat_count": int, "bytes": int, "bounds": {"min":[x,y,z],"max":[x,y,z]}, "format": "splat" }` |
| GET    | `/api/projects/{id}/share`             | —                                    | `{ "url": "http://localhost:5173/view/{id}" }` |

Errors: JSON `{ "detail": "friendly message" }` with appropriate 4xx/5xx status.

Upload constraints: accept `.jpg/.jpeg/.png/.heic` images and `.mp4/.mov` video (video
frame-extraction requires ffmpeg; if unavailable, reject video with a friendly message).
Min photos to reconstruct: **8** (warn below 20). Max: 400.

---

## 3. `.splat` binary asset format

Standard antimatter15-compatible layout, **32 bytes per splat**, little-endian,
splats concatenated with no header:

| offset | type        | field                              |
|--------|-------------|------------------------------------|
| 0      | float32 × 3 | position x, y, z                   |
| 12     | float32 × 3 | scale x, y, z (world units)        |
| 24     | uint8  × 4  | color r, g, b, a (0..255)          |
| 28     | uint8  × 4  | rotation quaternion, each = round((q+1)*128) clamped 0..255, order (w,x,y,z) |

`splat_count = filesize / 32`. The viewer parses exactly this layout.

---

## 4. Viewer engine interface (owned by `frontend/src/viewer/`)

Vanilla TypeScript, framework-agnostic. The frontend UI imports and wraps it.

```ts
export type CameraMode = 'orbit' | 'walk' | 'fly';

export interface SplatViewerOptions {
  canvas: HTMLCanvasElement;
  mode?: CameraMode;            // default 'orbit'
  onFps?: (fps: number) => void;
  onProgress?: (loaded: number, total: number) => void; // bytes during load
}

export class SplatViewer {
  constructor(opts: SplatViewerOptions);
  /** Load a .splat from a URL (the backend asset endpoint). Resolves when first frame drawn. */
  load(url: string): Promise<void>;
  /** Load from an ArrayBuffer (used by tests / drag-drop). */
  loadBuffer(buf: ArrayBuffer): Promise<void>;
  setMode(mode: CameraMode): void;
  getMode(): CameraMode;
  resetCamera(): void;
  /** Returns a PNG data URL of the current frame (screenshot mode). */
  capture(): string;
  /** Number of splats currently loaded. */
  get splatCount(): number;
  dispose(): void;
}
```

Controls (per 04_user_experience.md): drag = look/orbit, WASD = move, scroll = zoom/dolly,
Shift = move faster, Space/C = up/down in fly mode. Controls attach to the canvas on
construction and detach on `dispose()`.

The frontend wraps this in `src/components/SplatViewerReact.tsx` exposing:
`<SplatViewerReact src={url} mode={mode} onFps={...} />`.

---

## 5. Conventions
- Coordinate system: Y-up, right-handed. Scene roughly centered at origin; floor near y=0.
- Backend stores everything under `backend/storage/<project_id>/` (raw/, processed/, thumbs/, asset.splat, project.json).
- Friendly error tone everywhere (no "COLMAP", "SfM", "training failed" in user-facing text).
- Health endpoint reports which reconstruction backend is active so the UI can show a notice
  when running on the CPU fallback.
