<div align="center">

# Ironbook

### A living manual you can walk into.

**🏆 UC Berkeley AI Hackathon — Toolbox Grand Prize Winner**

<video src="assets/demo.mp4" controls width="100%"></video>

---

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![WebGL](https://img.shields.io/badge/WebGL2-990000?style=flat&logo=webgl&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-D97757?style=flat&logo=anthropic&logoColor=white)

</div>

---

Manufacturing is facing a knowledge crisis. An entire generation of skilled machinists, technicians, and engineers are retiring — and **57% leave with more than half their institutional knowledge still in their heads**. Apprenticeships take 4–5 years to replace one person. ~500,000 manufacturing jobs sit empty right now.

Manuals are flat. Videos are passive. CAD scanning costs $50K+ and requires specialists. None of them capture tacit, spatial knowledge.

**Ironbook captures it with a phone.**

Walk around a machine, take 20 photos, and Ironbook reconstructs a photorealistic 3D scene — paired with an AI agent that navigates it in real time. A new technician asks *"What's that valve on the left?"* and the agent flies there, highlights it, and explains it.

---

## How It Works

### 1. Capture
Walk around any machine or workstation with your phone. Take 20+ overlapping photos from different angles. The app guides you through coverage and warns about gaps.

### 2. Reconstruct
Ironbook runs a real **3D Gaussian Splatting** pipeline:

1. **COLMAP** recovers precise camera poses and a sparse point cloud from your photos
2. **msplat** — Metal-accelerated — trains tens of thousands of 3D Gaussians on your Apple GPU (~24 dB PSNR in ~41 seconds on M-series chips)
3. The output is a `.splat` asset streamed to a custom WebGL2 viewer

Fallbacks ensure you always get something navigable: Metal GPU → MLX CPU trainer → depth-based 2.5D reconstruction.

### 3. Ask
An AI agent (Claude, vision + structured output) receives your message, the current camera frame as a screenshot, and the 3D camera state. It returns a natural-language answer **and** a sequence of camera actions the frontend executes in real time:

`fly_to` · `look_at` · `highlight` · `rotate` · `zoom` · `move` · `reset_view`

The agent doesn't just describe what to do — it *does it*, maneuvering through the 3D scene to answer spatially.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Browser                        │
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
│                   FastAPI Backend                   │
│                                                     │
│  agent/act ──► agent_llm.run_agent()               │
│                    │                               │
│                    ▼                               │
│             Claude (vision + structured output)    │
│             returns { answer, actions[] }          │
│                                                     │
│  projects/ ──► COLMAP → msplat → .splat            │
└─────────────────────────────────────────────────────┘
```

| Layer | Tech |
|---|---|
| Frontend | React 18 + Vite + TypeScript, custom WebGL2 Gaussian splat renderer |
| Backend | FastAPI + Uvicorn |
| Agent | Claude (claude-3-5-sonnet), vision + structured JSON output, multi-turn history |
| Voice | Deepgram nova-2 (WebSocket, browser-direct) |
| Reconstruction | COLMAP + msplat (Metal GPU) → MLX CPU → depth-anything ONNX fallback |
| Observability | Sentry (transactions, errors), Langfuse (LLM traces, token usage) |

---

## Quick Start

**Prerequisites:** Python 3.11+, Node 18+, `brew install colmap` (optional, for real reconstruction)

**Backend:**
```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# Create backend/.env — see backend/.env.example for all options
# Required: ANTHROPIC_API_KEY=sk-ant-...

./.venv/bin/uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173
```

Open `http://localhost:5173`, click **New Project**, upload 8+ photos of any object, and hit **Reconstruct**. Once processing completes, open the 3D viewer and start asking questions.

**No photos handy?** Click **Try Demo** to load a prebuilt bike scene instantly.

### GPU Reconstruction (Apple Silicon)

For the full msplat pipeline (~41 s on M5 Max vs ~20 min on CPU):

```bash
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadComponent MetalToolchain

# Depth fallback model (~94 MB, one-time)
mkdir -p backend/models
curl -L "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx" \
  -o backend/models/depth_anything_v2_small.onnx
```

Set `MSPLAT_DIR` in `backend/.env` to point at your msplat build. `GET /api/health` reports the active reconstruction backend.

---

## Configuration

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Required for agent queries |
| `DEEPGRAM_API_KEY` | Voice input (Deepgram nova-2) |
| `SENTRY_DSN` | Error tracking + performance monitoring |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | LLM trace logging |
| `MSPLAT_DIR` | Path to msplat build (enables Metal GPU reconstruction) |
| `MSPLAT_ITERS` | Training iterations (default: 7000) |
| `AGENT_MODEL` | Claude model override (default: claude-3-5-sonnet-20241022) |

---

## Capture Tips

- **20+ overlapping shots** — walk a full circle around the subject
- **Move the camera position** between shots, not just your phone angle (real parallax)
- **Even lighting** — avoid hard shadows or blown highlights
- **Tap to focus** before each shot to avoid motion blur

---

## Testing

```bash
# Backend (27 tests)
cd backend && ./.venv/bin/python -m pytest -q

# Frontend (50 tests)
cd frontend && npm test

# Type-check + production build
cd frontend && npm run build
```

---

## Built at UC Berkeley AI Hackathon

**Praneeth Otthi** · **Langdon** · **Sanjay Sundaram**

Won the **Toolbox Grand Prize** at the UC Berkeley AI Hackathon.
