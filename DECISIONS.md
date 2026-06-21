# Decisions — resolving the open questions (06_open_questions.md)

These are the concrete choices made for this MVP build. Rationale kept short.

| Question | Decision |
|----------|----------|
| Platform (web/mac/iOS/desktop)? | **Web app** — most accessible, matches the WebGL/WebGPU viewer in the spec. |
| Reconstruction local or cloud? | **Pluggable, 5 tiers (best-available wins).** (0) CUDA COLMAP+gsplat (NVIDIA only; stubbed seam). (1) **msplat — fused-Metal 3D Gaussian Splatting, the active default on this M5 Max.** COLMAP (Homebrew, CPU) solves poses, then msplat trains real Gaussians fully on the GPU (every step a Metal compute kernel, no CPU round-trips) and writes the viewer's `.splat` directly. Measured: truck → ~24 dB in **~41 s** (7k iters, ~169 it/s, ~98% GPU). (2) COLMAP + MLX 3DGS — same idea but CPU-bottlenecked (~20 min); kept as a fallback if msplat isn't built. (3) on-device monocular-depth 2.5D — fallback when no trainer, or when COLMAP can't solve poses. (4) procedural room — last resort. |
| Optimize for speed/quality/editability? | **Speed + smooth viewing** for the MVP (Phase 1 "Reconstruction Viewer"). |
| Output for viewing/exporting/editing? | **Viewing + exporting** (`.splat` download, screenshot, share link). Editing is Phase 3+, out of MVP. |
| Hide the term "Gaussian Splatting"? | **Hidden by default** in user-facing copy ("3D World"); shown in power-user/expanded logs. |
| Photos, video, or both? | **Both.** Photos always; video frame-extraction when ffmpeg is present, else friendly reject. |
| Min photos for good output? | **≥8 to attempt, ≥20 recommended**, ≤400 max. |
| Guide capture? | **Yes** — capture-guide screen + live coverage/quality warnings on upload. |
| Detect insufficient coverage? | Heuristic coverage score from count + EXIF/orientation spread; warns about missing side views. |
| Backend first: Nerfstudio/Graphdeco/gsplat/custom? | Integration target = **COLMAP (poses) + gsplat (Splatfacto-style training)**; abstracted behind `ReconstructionBackend`. |
| Viewer: WebGL/WebGPU/Metal/Unity? | **WebGL2** custom splat renderer (broad support, no build-time GPU deps). |
| Movement feel: street view / game / editor? | **Game-like**: orbit + walk + fly modes, WASD + mouse. |
| Collide or fly freely? | **Fly/free** for MVP; collision is future work. |
| Edits destructive or layered / undo? | Out of MVP (Phase 3+). |
| Generative hidden geometry? | Out of MVP (Phase 4+); architecture leaves a generative-layer seam. |
| Business shape? | Standalone web app for the MVP demo. |
| Strongest demo? | "Upload photos → walk through the 3D world → share a link / export." |

## What is real vs. simulated in this build
- **Real reconstruction — true 3D Gaussian Splatting, GPU-saturating, on the Apple GPU.**
  `MsplatBackend` runs the genuine pipeline: COLMAP recovers camera poses + a sparse cloud, then
  **msplat** (fused-Metal: projection, sort, rasterize, SSIM, backward, Adam, densify all as
  Metal compute kernels — no per-iteration CPU round-trip) optimizes real 3D Gaussians fully on
  the GPU and writes the viewer's antimatter15 `.splat` directly (byte-verified compatible).
  Measured on this M5 Max: truck → PSNR ~24 dB, ~41 s for 7k iters (~169 it/s), **~98% GPU /
  ~9% CPU**; full upload→COLMAP→msplat→`.splat`→HTTP flow exercised end-to-end.
- **Fallback 3DGS: `Gaussian3DGSBackend`** (COLMAP + MLX trainer → `.ply` → `.splat`). Same real
  algorithm but CPU-bottlenecked (~20 min); used only if msplat isn't built.
- **Real (also on-device): depth 2.5D** — `DepthReconstructor` (monocular depth → point cloud).
  Used when no trainer is installed, or as auto-fallback when COLMAP can't solve poses.
- **The remaining stub:** CUDA COLMAP+gsplat (`ColmapGsplatBackend`) for NVIDIA GPUs — not used here.
- **Last-resort fallback:** `FallbackReconstructor` (a fixed procedural room).
- **Everything else is real:** project mgmt, upload + validation/coverage, staged job queue with
  friendly progress + logs, the from-scratch WebGL2 splat **viewer** (orbit/walk/fly), export/
  screenshot/share, and the test suites.

### Note on the 3DGS trainer location
The GPU trainer (MLX + COLMAP loader) lives in a separate venv under
`_splat_research/` (it pulls heavy ML deps — torch/mlx — that we keep out of the API server's
environment). The backend shells out to it. Env knobs: `SPLAT_RESEARCH_DIR`, `SPLAT_APPLE_DIR`,
`SPLAT_TRAINER_PY`, `SPLAT_TRAIN_ITERS` (default 7000), `SPLAT_TRAINER_BACKEND` (cpp|python),
`GSW_FORCE_BACKEND` (pin a backend; tests use `fallback`). The absolute-fastest fused-Metal
path (msplat) needs the offline Metal toolchain (`sudo xcodebuild -runFirstLaunch`), which is
currently broken on this machine — MLX avoids it by compiling kernels at runtime.
