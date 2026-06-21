# Gaussian Splat World

Upload photos of a real place → get an interactive 3D world you can walk through, in your
browser. This is the MVP described in `../gaussian_splat_world_docs/` (Phase 1:
Reconstruction Viewer).

- **`backend/`** — FastAPI service: project management, photo/video upload + validation &
  coverage scoring, a staged reconstruction job queue, and a pluggable reconstruction engine
  that emits a real `.splat` asset.
- **`frontend/`** — React + Vite app (Dashboard, Upload, Capture Guide, Processing, 3D Viewer,
  Scene Settings, Export/Share) with a from-scratch **WebGL2 Gaussian-splat viewer** (orbit /
  walk / fly modes, WASD + mouse).
- **`CONTRACT.md`** — the binding API + `.splat` format + viewer-interface contract.
- **`DECISIONS.md`** — how the open questions in the spec were resolved for this MVP.

## What's real vs. simulated
Everything in the product surface is real and runnable: project mgmt, upload + validation,
staged progress with friendly logs, real `.splat` generation, a real WebGL splat renderer,
export/screenshot/share, and full test suites.

**Reconstruction is real 3D Gaussian Splatting, trained fully on the Apple GPU.** On this M5 Max
the default backend (`msplat`) runs the genuine pipeline:

1. **COLMAP** (Homebrew, CPU) recovers camera poses + a sparse point cloud from your photos.
2. **msplat** — a fused-Metal trainer (projection, sort, rasterize, SSIM, backward, Adam, and
   densify all run as Metal compute kernels with no per-iteration CPU round-trip) optimizes real
   3D Gaussians **on the GPU**.
3. msplat writes the viewer's 32-byte `.splat` directly (byte-verified compatible — no convert).

Measured on this machine: truck scene → **~24 dB PSNR in ~41 s** (7000 iters, ~169 it/s) at
**~98% GPU / ~9% CPU**; the full upload→COLMAP→train→`.splat`→HTTP flow runs end-to-end.
`/api/health` reports the active backend and the UI explains it.

Fallbacks (best-available wins): if msplat isn't built → the MLX 3DGS trainer (`gaussian_3dgs`,
same algorithm but CPU-bottlenecked, ~20 min); if no trainer, or if COLMAP can't solve poses for
a given set → on-device **depth 2.5D** so you still get something navigable.

### Setup for the GPU 3DGS path
- **COLMAP:** `brew install colmap`
- **Metal toolchain** (needed to build msplat's shaders): `sudo xcodebuild -runFirstLaunch` then
  `xcodebuild -downloadComponent MetalToolchain` (already done on this machine).
- **msplat** lives under `_splat_research/msplat` (built binary at `build/msplat`). Env knobs:
  `MSPLAT_DIR`, `MSPLAT_BIN`, `MSPLAT_ITERS` (default 7000), `MSPLAT_DOWNSCALES`. The MLX
  fallback trainer's knobs (`SPLAT_RESEARCH_DIR`, `SPLAT_TRAIN_ITERS`, …) are in `DECISIONS.md`.
  Capture tip: 20+ overlapping photos taken while walking around the subject — SfM needs real
  parallax, so a few scattered shots won't solve.
- **Depth fallback model (one-time, ~94 MB):**
  ```bash
  mkdir -p backend/models
  curl -L "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx" \
    -o backend/models/depth_anything_v2_small.onnx
  ```

## Run it

**Backend** (terminal 1):
```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --reload --port 8000
```

**Frontend** (terminal 2):
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api → :8000)
```

Open http://localhost:5173 → "New 3D World" → add 8+ photos → "Create my world" → watch the
friendly stages → explore the result. Use the share link or export the `.splat`.

## Test
```bash
cd backend  && ./.venv/bin/python -m pytest -q     # 27 passed
cd frontend && npm test                            # 50 passed
cd frontend && npm run build                        # type-check + production build
```

## Verified end-to-end
Create project → upload 12 photos (validation + coverage warnings) → reconstruct through all 6
stages → real 41,000-splat asset served → frontend `SplatLoader` decodes the live asset
byte-for-byte → Vite proxy wiring → persistence across backend restart → friendly 400/404
error paths. The only thing not verifiable headlessly is the on-GPU pixel render, which needs a
real browser/GPU (the renderer logic itself is unit-tested).
# ironbook
# ironbook
