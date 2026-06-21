# Ironbook

> Preserve the knowledge of every expert on your shop floor — before they walk out the door.

---

## The Problem: The Silver Tsunami

The manufacturing industry is facing a knowledge crisis. An entire generation of skilled machinists, technicians, and engineers — people who know exactly why a machine sounds different when a bearing is failing, or which tooling setup works for a specific aluminum alloy — are retiring. That institutional knowledge has historically lived in people's heads, passed down through years of hands-on mentorship. As these experts leave, so does the knowledge.

Younger manufacturers joining the workforce don't have decades to apprentice. They need a faster way to learn the equipment, understand the machines, and ask questions about the physical world around them.

**Ironbook bridges that gap.**

---

## The Solution

Take your phone, walk around a machine, a part, or a workstation, and photograph it from multiple angles. Ironbook reconstructs a photorealistic 3D scene — a **Gaussian splat** — and pairs it with an AI agent that can *navigate that scene in real time*.

Instead of a static photo or a flat manual, a new technician can:

- **Ask natural questions** — *"What's that valve on the left side?"* or *"Show me the oil fill port."*
- **Let the agent navigate** — the AI physically moves the camera through the 3D scene, flies to the object, and highlights it on screen
- **Speak instead of type** — voice input lets technicians keep their hands free
- **Build a living library** — every scanned asset becomes a searchable, queryable record of institutional knowledge

---

## How It Works

### 1. Capture
Upload 20+ overlapping photos of any physical object or space. The more parallax between shots, the better the reconstruction. The app guides you through coverage.

### 2. Reconstruct
Ironbook runs a real **3D Gaussian Splatting** pipeline on your machine:

1. **COLMAP** recovers precise camera poses and a sparse point cloud from your photos using structure-from-motion
2. **msplat** — a Metal-accelerated trainer — optimizes tens of thousands of 3D Gaussians on your Apple GPU (~24 dB PSNR in ~41 seconds on M-series chips)
3. The output is a `.splat` asset streamed directly to the browser

If GPU training isn't available, the system falls back to an MLX-based CPU trainer, then to a depth-based 2.5D reconstruction — you always get something navigable.

### 3. Query
An AI agent backed by **Claude** (vision + structured output) receives the user's message, the current camera frame as a screenshot, and the camera state. It reasons about the scene and returns:

- A natural-language **answer** to the user's question
- A sequence of **camera actions** the frontend executes: `fly_to`, `look_at`, `highlight`, `rotate`, `zoom`, `move`, `reset_view`, and more

The agent is connected to the viewer via an **MCP-style action protocol** — it doesn't just describe what to do, it *does it*, maneuvering through the 3D scene to answer questions spatially.

### 4. Explore
A WebGL2 Gaussian splat renderer built from scratch supports **orbit, walk, and fly modes** with WASD + mouse. The viewer executes agent actions in real time as the conversation unfolds.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Browser                         │
│                                                     │
│  ┌──────────────┐   actions   ┌──────────────────┐  │
│  │  AgentChat   │ ──────────► │  WebGL2 Splat    │  │
│  │  (voice/text)│ ◄────────── │  Viewer          │  │
│  └──────┬───────┘  screenshot └──────────────────┘  │
│         │                                           │
└─────────┼───────────────────────────────────────────┘
          │ POST /api/agent/act
          ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│                                                     │
│  agent/act ──► agent_llm.run_agent()               │
│                    │                               │
│                    ▼                               │
│             Claude (vision + structured output)    │
│             returns {answer, actions[]}            │
│                                                     │
│  projects/ ──► COLMAP → msplat → .splat            │
└─────────────────────────────────────────────────────┘
```

**Stack:**
- **Frontend:** React + Vite, custom WebGL2 Gaussian splat renderer, voice input via Deepgram
- **Backend:** FastAPI, COLMAP, msplat (Metal), MLX 3DGS fallback, depth-anything fallback
- **Agent:** Claude (claude-3-5-sonnet), structured JSON output schema, multi-turn history, screenshot-grounded reasoning
- **Observability:** Sentry (transactions, errors, Anthropic integration), Langfuse (LLM traces, token usage)

---

## Quick Start

**Prerequisites:** Python 3.11+, Node 18+, `brew install colmap` (optional, for real reconstruction)

**Backend:**
```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# Create backend/.env with at minimum:
# ANTHROPIC_API_KEY=sk-ant-...
# (see backend/.env.example for all options)

./.venv/bin/uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173
```

Open `http://localhost:5173`, click **New Project**, upload 8+ photos of any object, and hit **Reconstruct**. Once processing completes, open the 3D viewer and start asking questions.

### GPU reconstruction (Apple Silicon)

For the full msplat pipeline (~41 s on M5 Max vs ~20 min on CPU):

```bash
# Metal toolchain (one-time)
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadComponent MetalToolchain

# Depth fallback model (~94 MB, one-time)
mkdir -p backend/models
curl -L "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx" \
  -o backend/models/depth_anything_v2_small.onnx
```

Set `MSPLAT_DIR` in `backend/.env` to point at your msplat build. `GET /api/health` reports the active backend.

### Demo scene

No photos handy? Click **Try Demo** on the home screen to load a prebuilt bike scene and start querying immediately.

---

## Configuration

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required for agent queries |
| `DEEPGRAM_API_KEY` | Required for voice input |
| `SENTRY_DSN` | Error tracking + performance monitoring |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | LLM trace logging |
| `MSPLAT_DIR` | Path to msplat build (enables GPU 3DGS) |
| `MSPLAT_ITERS` | Training iterations (default: 7000) |
| `AGENT_MODEL` | Claude model to use (default: claude-3-5-sonnet-20241022) |

---

## Testing

```bash
# Backend (27 tests)
cd backend && ./.venv/bin/python -m pytest -q

# Frontend (50 tests)
cd frontend && npm test

# Type-check + production build
cd frontend && npm run build

# Sentry smoke test (fires ~100 real API calls)
python sentry_smoke_test.py
```

---

## Capture Tips

Good 3D reconstruction depends on good photos:

- **20+ overlapping shots** — walk a full circle around the subject
- **Real parallax** — move your camera position between shots, not just your phone angle
- **Even lighting** — avoid hard shadows or blown highlights
- **No motion blur** — tap to focus before each shot
- The app shows a **coverage score** as you upload and warns you if gaps exist

---

## Roadmap

- [ ] Multi-user shared scene libraries (team knowledge base)
- [ ] Video upload → automatic frame extraction
- [ ] Semantic search across scanned assets
- [ ] On-device reconstruction for field use (no cloud required)
- [ ] Export to standard formats (`.ply`, GLTF)
