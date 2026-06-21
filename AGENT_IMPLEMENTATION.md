# Reasoning Navigation Agent — Implementation Spec (for Sonnet)

> **Goal.** Add a reasoning agent to the existing Gaussian-Splat viewer that can (a) navigate **any** `.splat` loaded into the app via natural language ("move to the fountain", "zoom in on that"), and (b) answer questions about objects in the scene ("what does this lever do?"). The agent drives the camera with the full motion set — forward / backward / strafe / rotate CW+CCW / zoom in+out — and can **highlight** an object in the splat.
>
> **Architecture (decided).** Screenshot → Claude vision for object grounding. A new backend `POST /api/agent/act` endpoint holds the Anthropic API key and runs the reasoning loop. The frontend sends the user's text plus a screenshot of the current view; Claude returns an answer plus a list of structured **actions**; the frontend executes each action against the viewer. Object localization is done by Claude returning a normalized 2D point in the screenshot, which the client converts to a 3D scene point by ray-picking against the loaded splats.
>
> **Model.** `claude-opus-4-8` via the official `anthropic` Python SDK, adaptive thinking, **structured output** (`output_config.format`) so the response is always a valid actions+answer JSON object. (Swap to `claude-sonnet-4-6` only if asked.)

Read `CONTRACT.md` §4 and these files before starting — they are the ground truth and your edits must stay consistent with them:
`frontend/src/viewer/SplatViewer.ts`, `frontend/src/viewer/controls.ts`, `frontend/src/viewer/math.ts`, `frontend/src/viewer/shaders.ts`, `frontend/src/viewer/SplatLoader.ts`, `frontend/src/components/SplatViewerReact.tsx`, `frontend/src/pages/ViewerPage.tsx`, `frontend/src/api/client.ts`, `backend/app/main.py`, `backend/app/routers/projects.py`, `backend/app/config.py`.

Use TDD where noted. Run `cd frontend && npm test` and `cd backend && ./.venv/bin/python -m pytest -q` after each phase; both suites currently pass (50 / 27) and must stay green.

---

## High-level data flow

```
User types "highlight the fountain and tell me what it is"
        │
        ▼
ViewerPage chat panel
   • viewer.capture()  → PNG data URL (current camera view)
   • viewer.getCameraSnapshot() → {eye,target,fov,aspect,bounds,mode}
        │  POST /api/agent/act { message, screenshot_b64, camera, history }
        ▼
backend /api/agent/act  (holds ANTHROPIC_API_KEY)
   • builds Claude messages: system + history + [image block + user text]
   • client.messages.create(model=claude-opus-4-8, output_config=ACTIONS_SCHEMA, tools? no)
   • returns { answer, actions:[...] }   (validated JSON)
        │
        ▼
ViewerPage action executor
   • for each action → AgentController method on the viewer
   • actions referencing an object carry target_2d:[nx,ny] in [0,1]
   • viewer.pickAt(nx,ny) ray-casts → 3D point → flyTo / highlight
        │
        ▼
SplatViewer animates camera / sets highlight uniforms; answer shown in chat
```

The only "intelligence" lives in the backend Claude call. Everything else is deterministic camera math + WebGL.

---

## PHASE 0 — Backend: Anthropic client + config

**Files:** `backend/requirements.txt`, `backend/app/config.py`, new `backend/app/services/agent_llm.py`.

1. Add `anthropic` to `backend/requirements.txt` (pin a recent version, e.g. `anthropic>=0.69`). Install into the existing venv: `./.venv/bin/pip install -r requirements.txt`.

2. In `config.py`, read the key from env. Do **not** hardcode it. Add:
   ```python
   ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
   AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")
   ```
   (Match the existing config style in that file.)

3. Create `backend/app/services/agent_llm.py`. This module owns the Claude call and is the **only** place that imports `anthropic`. Keep it swappable.

   Key facts (from the Claude API skill — follow exactly, your training prior may be stale):
   - Construct `anthropic.Anthropic()` (reads `ANTHROPIC_API_KEY` from env) — or pass `api_key=` from config.
   - `claude-opus-4-8` uses **adaptive thinking**: `thinking={"type": "adaptive"}`. **Never** send `budget_tokens`, `temperature`, `top_p`, or `top_k` — they 400 on this model.
   - Use **structured output** so the response is guaranteed-valid JSON: pass `output_config={"format": {"type": "json_schema", "schema": ACTIONS_SCHEMA}}`. The first text block of the response is then valid JSON matching the schema — parse it with `json.loads`.
   - Vision: pass the screenshot as an image content block: `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": <b64 no prefix>}}`, placed **before** the text block in the user message.
   - `max_tokens=2048` is plenty; no streaming needed (small output).
   - Handle `response.stop_reason == "refusal"` → return a friendly fallback answer with empty actions.

   ```python
   import json, anthropic
   from ..config import ANTHROPIC_API_KEY, AGENT_MODEL

   _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

   SYSTEM_PROMPT = """You are a navigation + reasoning agent embedded in a 3D Gaussian-splat
   viewer. The user sees a real-time rendering of a reconstructed 3D scene and talks to you
   about it. You are given the CURRENT camera view as an image plus a description of the camera.

   You control the camera and can highlight objects by returning a list of ACTIONS, and you
   answer the user's questions in `answer`.

   Coordinate conventions:
   - Any action that refers to an object in the scene must include `target_2d`: the [x, y]
     location of that object in the CURRENT image, normalized to [0,1] with (0,0) = top-left,
     (1,1) = bottom-right. Pick the center of the object.
   - Distances/amounts are RELATIVE and unitless in [0,3]; 1.0 ≈ one moderate step / a 30°
     turn / a comfortable zoom. The client scales them to the scene automatically.

   Available action types (see schema). Rules:
   - If the user asks to GO somewhere / move toward something, emit `fly_to` with target_2d.
   - If the target is NOT visible in the current image, do NOT guess a target_2d. Instead emit
     a `rotate` or `move` to explore, and say in `answer` that you're looking for it.
   - To answer "what is this / what does X do", reason from the image and emit a `highlight`
     on it (with target_2d) plus your explanation in `answer`. Be concise and specific; if you
     are uncertain about an object's identity or function, say so rather than inventing detail.
   - Use `clear_highlight` when the user wants the highlight removed.
   - Emit only the actions needed. `answer` is always required (one or two sentences)."""

   # See ACTIONS_SCHEMA below.

   def run_agent(message: str, screenshot_b64: str | None, camera: dict, history: list) -> dict:
       if _client is None:
           return {"answer": "The agent isn't configured (missing ANTHROPIC_API_KEY).",
                   "actions": []}
       user_content = []
       if screenshot_b64:
           user_content.append({"type": "image", "source": {
               "type": "base64", "media_type": "image/png", "data": screenshot_b64}})
       cam_note = (f"Camera mode={camera.get('mode')} fov={camera.get('fov')}. "
                   f"Scene is roughly centered at the origin, Y-up.")
       user_content.append({"type": "text", "text": f"{cam_note}\n\nUser: {message}"})

       messages = _history_to_messages(history) + [{"role": "user", "content": user_content}]
       try:
           resp = _client.messages.create(
               model=AGENT_MODEL,
               max_tokens=2048,
               thinking={"type": "adaptive"},
               system=SYSTEM_PROMPT,
               output_config={"format": {"type": "json_schema", "schema": ACTIONS_SCHEMA}},
               messages=messages,
           )
       except anthropic.APIError as e:
           return {"answer": f"Sorry, the agent hit an error: {e.message}", "actions": []}

       if resp.stop_reason == "refusal":
           return {"answer": "I can't help with that request.", "actions": []}
       text = next((b.text for b in resp.content if b.type == "text"), "{}")
       data = json.loads(text)
       data.setdefault("actions", [])
       data.setdefault("answer", "")
       return data
   ```

   `_history_to_messages(history)` converts the prior turns (list of `{role, text}` the frontend sends back each call — the API is stateless) into `[{"role": r, "content": t}, ...]`. Keep only the last ~6 turns. Do **not** resend old screenshots (cost); text-only history is fine.

4. Define `ACTIONS_SCHEMA` in the same module. **Structured-output schema constraints** (from the skill): every object needs `"additionalProperties": false` and a `"required"` list; you may use `enum`, arrays, nested objects; you may **not** use `minimum`/`maximum`/`minLength` (validate ranges client-side instead).

   ```python
   ACTIONS_SCHEMA = {
     "type": "object",
     "additionalProperties": False,
     "required": ["answer", "actions"],
     "properties": {
       "answer": {"type": "string"},
       "actions": {
         "type": "array",
         "items": {
           "type": "object",
           "additionalProperties": False,
           "required": ["type"],
           "properties": {
             "type": {"type": "string", "enum": [
               "move", "rotate", "zoom", "fly_to", "look_at",
               "highlight", "clear_highlight", "reset_view"]},
             "direction": {"type": "string", "enum": [
               "forward","backward","left","right","up","down",
               "clockwise","counterclockwise","in","out"]},
             "amount": {"type": "number"},
             "target_2d": {"type": "array", "items": {"type": "number"}},
             "label": {"type": "string"}
           }
         }
       }
     }
   }
   ```

   Semantics the frontend will rely on:
   - `move` → `direction` ∈ forward/backward/left/right/up/down, `amount` (default 1).
   - `rotate` → `direction` ∈ clockwise/counterclockwise (yaw), `amount` in turns where 1≈30°.
   - `zoom` → `direction` ∈ in/out, `amount`.
   - `fly_to` / `look_at` / `highlight` → require `target_2d` `[nx,ny]`.
   - `clear_highlight`, `reset_view` → no params.

---

## PHASE 1 — Backend: the `/api/agent/act` endpoint

**Files:** new `backend/app/routers/agent.py`, register it in `backend/app/main.py`. Add Pydantic models to `backend/app/models.py` (match existing style).

1. Request/response models:
   ```python
   class CameraSnapshot(BaseModel):
       mode: str
       fov: float
       eye: list[float]
       target: list[float]
       bounds: dict           # {"min":[...], "max":[...]}
   class AgentTurn(BaseModel):
       role: str
       text: str
   class AgentActRequest(BaseModel):
       message: str
       screenshot_b64: str | None = None    # PNG base64, NO data: prefix
       camera: CameraSnapshot
       history: list[AgentTurn] = []
   class AgentAction(BaseModel):
       type: str
       direction: str | None = None
       amount: float | None = None
       target_2d: list[float] | None = None
       label: str | None = None
   class AgentActResponse(BaseModel):
       answer: str
       actions: list[AgentAction]
   ```

2. Router (`backend/app/routers/agent.py`), prefixed `/api/agent` to match the project's `/api` convention (see how `projects.py` is mounted in `main.py`):
   ```python
   from fastapi import APIRouter
   from ..models import AgentActRequest, AgentActResponse
   from ..services.agent_llm import run_agent

   router = APIRouter(prefix="/api/agent", tags=["agent"])

   @router.post("/act", response_model=AgentActResponse)
   def act(req: AgentActRequest) -> AgentActResponse:
       result = run_agent(req.message, req.screenshot_b64, req.camera.model_dump(),
                          [t.model_dump() for t in req.history])
       return AgentActResponse(**result)
   ```
   Register in `main.py`: `app.include_router(agent.router)` (mirror the existing include for `projects`/`health`). CORS is already configured in `main.py`; no change needed.

3. **Tests** (`backend/tests/test_agent.py`): monkeypatch `agent_llm.run_agent` (or `agent_llm._client`) so tests never hit the network. Assert:
   - `/api/agent/act` returns 200 with a valid `AgentActResponse` shape for a stubbed result.
   - Missing key path: with `_client=None`, the endpoint returns a friendly `answer` and `actions==[]`.
   - A request with no screenshot still succeeds.
   Keep it consistent with `tests/conftest.py` (it already builds a TestClient — reuse the fixture). **Do not** add a test that requires `ANTHROPIC_API_KEY`.

---

## PHASE 2 — Viewer engine: imperative agent API (motion + camera snapshot)

**File:** `frontend/src/viewer/SplatViewer.ts` (and reuse pure helpers in `controls.ts`).

Add public methods to `SplatViewer` that the agent layer calls. All motion must work in **all three modes**; reuse the existing pure functions (`applyMove`, `applyLook`, `applyZoom`, `forwardFromAngles`, `fitToBounds`, `eyeForMode`, `targetForMode`) — do not duplicate camera math. Movement amounts scale with scene size via `this.state.distance` (already how `controls.ts` scales keyboard speed).

Add a small **animation system** so motions glide instead of snapping. Store a target camera state and lerp toward it inside the existing `renderOnce` (which already calls `this.controls?.update(dt)`). Add `this.anim` = `{active, t, dur, from, to}` and an `updateAnimation(dt)` called at the top of `renderOnce`.

```ts
// --- agent-facing imperative API ---
private STEP = 0.8;            // base move fraction of scene distance
private ROT  = Math.PI / 6;    // 30° == amount 1.0

moveRelative(dir: 'forward'|'backward'|'left'|'right'|'up'|'down', amount = 1) {
  const d = clampAmount(amount) * Math.max(0.5, this.state.distance) * this.STEP;
  // ensure we're in a free-move mode so position is authoritative
  const map = { forward:[0,0,d], backward:[0,0,-d], left:[-d,0,0], right:[d,0,0],
                up:[0,d,0], down:[0,-d,0] } as const;
  const mode = this.mode === 'orbit' ? 'fly' : this.mode;   // orbit move == fly nudge
  applyMove(this.state, mode, map[dir] as Vec3);
  if (this.mode === 'orbit') this.state.target = add(this.state.position,
      scale(forwardFromAngles(this.state.yaw, this.state.pitch), this.state.distance));
}

rotateView(dir: 'clockwise'|'counterclockwise', amount = 1) {
  const dy = (dir === 'clockwise' ? 1 : -1) * clampAmount(amount) * this.ROT;
  applyLook(this.state, dy, 0);
}

zoomView(dir: 'in'|'out', amount = 1) {
  applyZoom(this.state, this.mode, (dir === 'in' ? 1 : -1) * clampAmount(amount) * 0.6);
}

/** Animate camera so `point` is centered and framed; standoff scales to scene. */
flyTo(point: Vec3, standoff?: number) {
  const r = Math.max(0.5, this.state.distance);
  const so = standoff ?? r * 0.6;
  const to = structuredCloneState(this.state);
  to.target = point;
  to.distance = so;
  to.position = eyeForMode(to, 'orbit');   // keep yaw/pitch, recompute eye
  this.startAnim(to, 700);
}

/** Rotate (don't translate) to face `point`. */
lookAt(point: Vec3) {
  const eye = eyeForMode(this.state, this.mode);
  const dir = normalize(sub(point, eye));
  const to = structuredCloneState(this.state);
  to.yaw = Math.atan2(dir[0], -dir[2]);
  to.pitch = clamp(Math.asin(dir[1]), MIN_PITCH, MAX_PITCH);
  this.startAnim(to, 500);
}

getCameraSnapshot() {
  return {
    mode: this.mode,
    fov: DEFAULT_FOV,
    eye: eyeForMode(this.state, this.mode),
    target: targetForMode(this.state, this.mode),
    bounds: this.splats ? this.splats.bounds : { min:[0,0,0], max:[0,0,0] },
  };
}
```

`clampAmount(a)` = `Math.min(3, Math.max(0, a || 1))`. Add `startAnim(to,dur)` + `updateAnimation(dt)` doing eased lerp (easeInOutCubic) of `target`, `position`, `distance`, `yaw`, `pitch`; set `this.state` from the interpolation each frame; clear when `t>=1`. Lerp yaw via shortest angular path. `structuredCloneState` just deep-copies the `CameraState`.

**Tests** (extend `frontend/tests/camera.test.ts`, which already unit-tests `controls.ts` math headlessly): construct a `SplatViewer` with a stub canvas (the existing tests show the pattern), `loadBuffer` a tiny generated `.splat` (use `encodeSplat` from `SplatLoader`), then assert:
- `moveRelative('forward')` changes `getCameraSnapshot().eye` along the forward axis.
- `rotateView('clockwise')` changes yaw by ~30°.
- `zoomView('in')` reduces `distance`.
- `flyTo([x,y,z])` sets an animation whose final `target` ≈ the point after enough `renderOnce` ticks.
Keep these deterministic — drive frames by calling the private render or exposing a tiny `tickForTest(dt)` if needed (mirror how existing tests advance state).

---

## PHASE 3 — Viewer engine: ray-pick (2D → 3D) and highlight

**Files:** `frontend/src/viewer/SplatViewer.ts`, `frontend/src/viewer/shaders.ts`.

### 3a. `pickAt(nx, ny)` — screen point to scene point
Claude returns a normalized point in the **screenshot** (top-left origin). Convert it to a 3D scene point by projecting all splats and choosing the frontmost splat near that pixel, then return the median position of its local neighborhood for stability.

Algorithm (no new GL needed — pure math over `this.splats.positions`):
```ts
pickAt(nx: number, ny: number): Vec3 | null {
  const s = this.splats; if (!s || s.count === 0) return null;
  const w = this.canvas.width || 1, h = this.canvas.height || 1;
  const view = lookAt(eyeForMode(this.state,this.mode), targetForMode(this.state,this.mode),[0,1,0]);
  const proj = perspective(DEFAULT_FOV, w/h, 0.01, 1000);
  const targetPx = nx, targetPy = ny;            // both in [0,1], top-left origin
  let best = -1, bestScore = Infinity;
  const RADIUS = 0.06;                             // accept splats within 6% of frame
  for (let i = 0; i < s.count; i++) {
    const p: Vec3 = [s.positions[i*3], s.positions[i*3+1], s.positions[i*3+2]];
    const clip = mulMat4Vec4(proj, mulMat4Vec4(view, [p[0],p[1],p[2],1]));
    if (clip[3] <= 0) continue;                   // behind camera
    const ndcx = clip[0]/clip[3], ndcy = clip[1]/clip[3];
    const sx = (ndcx*0.5+0.5), sy = (1 - (ndcy*0.5+0.5));   // → [0,1] top-left
    const dx = sx - targetPx, dy = sy - targetPy;
    const d2 = dx*dx + dy*dy;
    if (d2 > RADIUS*RADIUS) continue;
    const depth = clip[2]/clip[3];                // prefer nearer (frontmost) splats
    const score = d2 + depth*0.001;
    if (score < bestScore) { bestScore = score; best = i; }
  }
  if (best < 0) return null;
  return neighborhoodMedian(s.positions, s.count, best);  // see below
}
```
- Add a `mulMat4Vec4(m, v)` helper to `math.ts` (column-major, matching `lookAt`/`perspective`). If a matrix-vector helper already exists, reuse it.
- `neighborhoodMedian`: take the chosen splat's position, gather all splats within e.g. 5% of scene radius, return the component-wise median (robust to floaters). For performance, sample if `count` is large.
- For very large scenes (>200k splats) you may stride-sample in the projection loop; if you cap, that's fine for an MVP.

### 3b. Highlight rendering
Add highlight uniforms to the shader and a sphere-region tint. **Read `shaders.ts` fully first** — the vertex shader already fetches `center`, builds `v_color`, and passes `v_offset` to the fragment shader.

In `VERTEX_SHADER`, add uniforms and a varying:
```glsl
uniform vec3 u_hlCenter;
uniform float u_hlRadius;   // <=0 means "no highlight"
uniform float u_hlPulse;    // 0..1 animated
out float v_hl;
```
After computing `center`, compute and emit the highlight factor:
```glsl
v_hl = (u_hlRadius > 0.0)
  ? smoothstep(u_hlRadius, u_hlRadius*0.4, distance(center, u_hlCenter))
  : 0.0;
```
In `FRAGMENT_SHADER`, accept `in float v_hl;` and after computing `fragColor`, brighten + tint highlighted splats:
```glsl
vec3 hlTint = vec3(1.0, 0.85, 0.2);
float k = v_hl * (0.5 + 0.5*u_hlPulse);
fragColor.rgb = mix(fragColor.rgb, hlTint, 0.55*k);
fragColor.rgb += 0.25*k;        // glow boost
```
Pass `u_hlPulse` from `Math.sin` of elapsed time. Wire the new uniforms exactly like the existing ones: get locations in `initGL` (`this.uniforms.u_hlCenter = gl.getUniformLocation(...)`, etc.) and set them in `renderOnce`'s draw block:
```ts
gl.uniform3f(this.uniforms.u_hlCenter, ...(this.hlCenter ?? [0,0,0]));
gl.uniform1f(this.uniforms.u_hlRadius, this.hlRadius);
gl.uniform1f(this.uniforms.u_hlPulse, 0.5 + 0.5*Math.sin(t*0.004));
```
Add state + API:
```ts
private hlCenter: Vec3 | null = null;
private hlRadius = 0;
highlightAt(point: Vec3, radius?: number) {
  this.hlCenter = point;
  this.hlRadius = radius ?? Math.max(0.3, this.state.distance * 0.25);
}
clearHighlight() { this.hlCenter = null; this.hlRadius = 0; }
```
Highlight radius should scale to scene size (use `distance` or a fraction of bounds extent).

**Tests:** `pickAt` is the testable unit — place a splat at a known world position, point the camera at it, project where it lands (or just put it at screen center), and assert `pickAt(0.5,0.5)` returns ≈ that position. Shader output isn't headlessly testable (documented limitation in `README.md`), so just assert `highlightAt`/`clearHighlight` set/clear `hlRadius`.

Export any new public methods through `frontend/src/viewer/index.ts` if other modules import from there.

---

## PHASE 4 — React: expose the agent API + API client method

**Files:** `frontend/src/components/SplatViewerReact.tsx`, `frontend/src/api/client.ts`, `frontend/src/api/types.ts`.

1. Extend `SplatViewerHandle` to surface the new viewer methods (the component already exposes `viewer` via ref, so the simplest path is to use `ref.viewer` directly, but add typed convenience passthroughs for `moveRelative`, `rotateView`, `zoomView`, `flyTo`, `lookAt`, `pickAt`, `highlightAt`, `clearHighlight`, `getCameraSnapshot`, `capture`). Keep the existing handle members.

2. Add types to `api/types.ts` mirroring the backend models: `AgentAction`, `AgentActResponse`, `CameraSnapshot`, `AgentTurn`.

3. Add a client method to `api/client.ts` (follow the existing `jsonRequest` pattern):
   ```ts
   agentAct(body: {
     message: string; screenshot_b64?: string; camera: CameraSnapshot; history: AgentTurn[];
   }): Promise<AgentActResponse> {
     return jsonRequest<AgentActResponse>('/agent/act', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(body),
     });
   }
   ```
   The Vite dev proxy already forwards `/api/*` → `:8000` (see `README.md`), so this Just Works.

---

## PHASE 5 — React: chat panel + action executor (the glue)

**File:** `frontend/src/pages/ViewerPage.tsx` (+ a new `frontend/src/components/AgentChat.tsx`, + styles in `global.css`).

1. Build `AgentChat.tsx`: a bottom or right-side glass panel (match the existing `.glass` overlay style used in `ViewerPage`) with a scrollable message list, a text input, and a send button. It receives the `viewerRef` (the `SplatViewerHandle`). Maintain `messages: {role:'user'|'assistant', text:string}[]` and a `busy` flag.

2. On send:
   ```ts
   const v = viewerRef.current?.viewer;
   if (!v) return;
   const screenshot = v.capture();                       // "data:image/png;base64,XXXX"
   const screenshot_b64 = screenshot.split(',')[1];      // strip the data: prefix
   const camera = v.getCameraSnapshot();
   const history = messages.slice(-6);
   setMessages(m => [...m, { role:'user', text }]);
   setBusy(true);
   try {
     const res = await apiClient.agentAct({ message: text, screenshot_b64, camera, history });
     setMessages(m => [...m, { role:'assistant', text: res.answer }]);
     executeActions(v, res.actions);
   } finally { setBusy(false); }
   ```

3. The **action executor** — the deterministic mapping from Claude's actions to viewer calls. Convert `target_2d` to 3D via `pickAt`; if pick fails (object not actually visible), fall back to `lookAt` of frame center or just skip and rely on the agent's text.
   ```ts
   function executeActions(v: SplatViewer, actions: AgentAction[]) {
     for (const a of actions) {
       switch (a.type) {
         case 'move':   if (a.direction) v.moveRelative(a.direction as any, a.amount ?? 1); break;
         case 'rotate': if (a.direction) v.rotateView(a.direction as any, a.amount ?? 1); break;
         case 'zoom':   if (a.direction) v.zoomView(a.direction as any, a.amount ?? 1); break;
         case 'reset_view':     v.resetCamera(); break;
         case 'clear_highlight':v.clearHighlight(); break;
         case 'fly_to':
         case 'look_at':
         case 'highlight': {
           if (!a.target_2d) break;
           const p = v.pickAt(a.target_2d[0], a.target_2d[1]);
           if (!p) break;
           if (a.type === 'fly_to') v.flyTo(p);
           else if (a.type === 'look_at') v.lookAt(p);
           else { v.highlightAt(p); v.lookAt(p); }
           break;
         }
       }
     }
   }
   ```
   Note: a "move to X then tell me about it" turn naturally arrives as `[{fly_to,target_2d}, {highlight,target_2d}]` + an `answer`; executing them in order does the right thing.

4. Add a toggle button in the `ViewerPage` bottom bar ("🤖 Ask") that shows/hides the panel. Keep it out of the `shared` (public) view if you want, or allow it everywhere — your call; default to showing it in the owner view.

5. **Tests** (`frontend/tests/AgentChat.test.tsx`, mirror `ViewerPage.test.tsx`): mock `apiClient.agentAct` to return a canned `{answer, actions}`; render the panel with a fake `viewerRef` whose viewer is a jest/vitest mock exposing the agent methods; type a message, click send, and assert (a) the user + assistant messages render, (b) the correct viewer methods were called for each action type, (c) `pickAt` is consulted for `fly_to`/`highlight`. This validates the executor without GL.

---

## PHASE 6 — Wire-up, polish, verify

- `capture()` requires `preserveDrawingBuffer: true` — it's already set in `SplatViewer`'s context creation, so screenshots will work.
- Disable the send button while `busy`; show a small spinner. Show agent errors inline (the backend always returns a friendly `answer`).
- Empty-scene / no-asset guard: if `v.splatCount === 0`, the panel should say "Load a world first."
- Run the full gates and keep them green:
  ```
  cd backend  && ./.venv/bin/python -m pytest -q
  cd frontend && npm test
  cd frontend && npm run build      # tsc typecheck + prod build
  ```
- Manual smoke test (needs a real browser + GPU, per README): start backend + frontend, open a project with a `.splat`, open the agent panel, try: "what am I looking at?", "move closer to the object on the left", "rotate clockwise", "zoom in", "highlight the brightest object and tell me what it might be", "reset the view".

---

## Acceptance criteria

1. `POST /api/agent/act` returns `{answer, actions}`; works without a screenshot; returns a friendly answer when `ANTHROPIC_API_KEY` is unset; never 500s on a normal request.
2. The viewer exposes `moveRelative / rotateView / zoomView / flyTo / lookAt / pickAt / highlightAt / clearHighlight / getCameraSnapshot`, all reusing existing camera math, all covered by unit tests.
3. Camera motions are smooth (animated), and work regardless of orbit/walk/fly mode.
4. Highlight visibly tints + glows splats in a sphere around a picked 3D point and can be cleared.
5. The chat panel sends the current screenshot + camera, renders the answer, and executes every returned action against the loaded splat — for **any** `.splat`, not a hardcoded scene.
6. Existing 50 frontend + 27 backend tests still pass; `npm run build` is clean.

---

## Notes / gotchas (read before coding)

- **Never put the API key in the frontend.** The browser only ever talks to `/api/agent/act`.
- **Adaptive thinking only** on `claude-opus-4-8`; sending `temperature`/`top_p`/`top_k`/`budget_tokens` returns HTTP 400.
- **Structured output**: parse `response.content`'s first text block with `json.loads`; the schema guarantees shape but still `setdefault` defensively.
- Screenshot base64 from `canvas.toDataURL` includes a `data:image/png;base64,` prefix — strip it before sending; the API wants the raw base64.
- `target_2d` is in the **screenshot's** pixel frame (top-left origin, [0,1]); `pickAt` must use the same convention — note the `sy = 1 - (...)` flip in the projection.
- Keep `agent_llm.py` the single Anthropic dependency so the provider is swappable later.
- The whole feature is additive: you are not changing `CONTRACT.md` interfaces, the `.splat` format, or the reconstruction pipeline.

---
---

# ADDENDUM v2 — Confirmed decisions, verified facts, current status, and the remaining work

This addendum supersedes the earlier phases where they differ. It records (a) the
decisions the user locked in, (b) facts verified by actually running code, (c) what is
**already implemented and tested on disk**, and (d) the **precise remaining work** (all
frontend) with enough detail to implement without guessing.

## A. Confirmed decisions (from the user)

1. **Model:** `claude-sonnet-4-6` (configurable via `AGENT_MODEL` env). Opus is a one-line swap later.
2. **Grounding:** screenshot → Claude vision → normalized `target_2d` → client ray-pick to 3D. (unchanged)
3. **Agent runtime:** backend `POST /api/agent/act` holds the key. (unchanged)
4. **Direct `.splat` upload:** the user can upload a pre-built `.splat` (any source) and the
   agent navigates it — not only photo→reconstruct scenes.
5. **Agent panel lives in the PUBLIC view** (`/view/:id`), not just the owner view.
6. **Real-time agent viewpoint:** the single rendered canvas **is** the agent's viewpoint.
   Agent moves animate smoothly so the user *watches* the navigation happen. Show an
   "agent is navigating…" affordance while a move animates.

## B. Verified facts (ran live — do not second-guess these)

- `anthropic==0.111.0` + `python-dotenv` install cleanly into `backend/.venv`.
- The Claude call shape **works as written** with `claude-sonnet-4-6`:
  `client.messages.create(model=..., max_tokens=2048, thinking={"type":"adaptive"},
  system=SYSTEM_PROMPT, output_config={"format":{"type":"json_schema","schema":ACTIONS_SCHEMA}},
  messages=[...])`. A live call returned valid JSON, e.g. user "turn right a bit" →
  `{"answer": "...", "actions": [{"type":"rotate","direction":"clockwise","amount":0.5}]}`.
  **Do NOT send `temperature`/`top_p`/`top_k`/`budget_tokens`** — they 400 on this model.
- Structured-output schema must have `additionalProperties:false` + `required` on every
  object, and must NOT use `minimum`/`maximum`/`minLength` (range-check client-side).
- Backend test suite: **43 passed, 4 skipped** (the 4 skips are the pre-existing depth-model
  tests; unrelated). The new agent + upload tests pass.

## C. ALREADY IMPLEMENTED AND TESTED (backend — on disk, green)

Do not re-create these; read them as the source of truth. If you change them, keep tests green.

- `backend/requirements.txt` — added `anthropic>=0.69`, `python-dotenv>=1.0,<2`.
- `backend/.env` (git-ignored; added `.env` to `backend/.gitignore`) — holds
  `ANTHROPIC_API_KEY` and `AGENT_MODEL=claude-sonnet-4-6`. **The key in it was shared in
  plaintext and must be rotated.**
- `backend/app/config.py` — loads `.env` via `load_dotenv(BACKEND_DIR/".env")`; adds
  `ANTHROPIC_API_KEY`, `AGENT_MODEL` (default `claude-sonnet-4-6`), `SPLAT_EXTS = {".splat"}`.
- `backend/app/services/agent_llm.py` — the only Anthropic-touching module. Contains
  `SYSTEM_PROMPT`, `ACTIONS_SCHEMA`, `_history_to_messages`, and
  `run_agent(message, screenshot_b64, camera, history) -> {"answer", "actions"}`. Builds the
  vision message (image block before text), calls Claude with the verified shape, parses the
  structured JSON, and returns a friendly fallback (no raise) when the key is missing, on
  `stop_reason=="refusal"`, on API error, or on bad JSON.
- `backend/app/models.py` — added `CameraSnapshot`, `AgentTurn`, `AgentActRequest`,
  `AgentAction`, `AgentActResponse`.
- `backend/app/routers/agent.py` — `POST /api/agent/act` → `agent_llm.run_agent(...)`.
- `backend/app/routers/projects.py` — added `POST /api/projects/upload_splat` (multipart
  `file` + `name` form field): validates `.splat` extension and `len % 32 == 0`, creates a
  project, writes `asset.splat`, sets `status="ready"`, `has_asset=True`, returns the Project.
- `backend/app/main.py` — `app.include_router(agent.router)`.
- `backend/tests/test_agent.py` — endpoint + `run_agent` tests with the **LLM fully mocked**
  (never hits network, never needs the key). `backend/tests/test_splat_upload.py` — upload
  happy path + two rejection cases.

**Backend API contract the frontend depends on:**
```
POST /api/agent/act
  body: { message: string,
          screenshot_b64?: string,        // PNG base64, NO "data:" prefix
          camera: { mode, fov, eye:[x,y,z], target:[x,y,z], bounds:{min,max} },
          history: [{ role:"user"|"assistant", text:string }] }
  -> { answer: string, actions: AgentAction[] }

AgentAction = { type, direction?, amount?, target_2d?:[nx,ny], label? }
  type ∈ move|rotate|zoom|fly_to|look_at|highlight|clear_highlight|reset_view
  direction ∈ forward|backward|left|right|up|down|clockwise|counterclockwise|in|out
  target_2d in [0,1], top-left origin (same frame as the screenshot)

POST /api/projects/upload_splat   (multipart: file=<x.splat>, name=<string>)
  -> Project { id, status:"ready", has_asset:true, ... }
```

## D. PARTIALLY IMPLEMENTED (viewer engine — verify/finish)

The following edits to the viewer were started and should be **reviewed, completed, and
tested** (Phase 3 below lists exactly what must be true). Re-read each file before editing.

- `frontend/src/viewer/math.ts` — added `transformPoint4(m, p) -> [x,y,z,w]` (no perspective
  divide; lets callers reject points behind the camera via `w<=0`).
- `frontend/src/viewer/SplatViewer.ts` — added imports (`applyLook/applyMove/applyZoom/
  forwardFromAngles/MIN_PITCH/MAX_PITCH/MIN_DISTANCE` from controls; `add/sub/scale/normalize/
  multiply/clamp/transformPoint4/Vec3` from math); fields (`anim`, `hlCenter`, `hlRadius`,
  `MOVE_STEP=0.8`, `ROT_STEP=π/6`); methods `moveRelative / rotateView / zoomView / flyTo /
  lookAtPoint / getCameraSnapshot / pickAt / neighborhoodMedian / highlightAt / clearHighlight /
  isAnimating / startAnim / updateAnimation`; wired `updateAnimation(dt*1000)` into `renderOnce`
  and added highlight uniform sets (`u_hlCenter/u_hlRadius/u_hlPulse`) in the draw block.

**STILL TO DO in the viewer (this is why `npm run build` will currently fail — finish these):**

1. **Module-scope helpers** referenced by the new methods are NOT defined yet. Add them near
   the existing `now()` function at the bottom of `SplatViewer.ts`:
   - `clampAmount(a:number):number` → `Math.min(3, Math.max(0, a || 1))`
   - `cloneState(s:CameraState):CameraState` → deep copy (`target/position` are 3-tuples, copy them)
   - `easeInOut(t:number):number` → easeInOutCubic
   - `lerp(a,b,e)`, `lerpVec(a:Vec3,b:Vec3,e):Vec3`
   - `lerpAngle(a,b,e)` → lerp along the **shortest** angular path (wrap (b-a) into [-π,π])
2. **Highlight uniform locations** must be registered in `initGL`'s `this.uniforms = {...}`
   object: add `u_hlCenter`, `u_hlRadius`, `u_hlPulse` via `gl.getUniformLocation(program, ...)`,
   exactly like the existing `u_view`/`u_proj` entries. (The draw block already *sets* them.)
3. **Shader edits** in `frontend/src/viewer/shaders.ts` (see Phase 3b in the original doc —
   still required): add `uniform vec3 u_hlCenter; uniform float u_hlRadius; uniform float
   u_hlPulse; out float v_hl;` to the vertex shader, compute
   `v_hl = u_hlRadius>0.0 ? smoothstep(u_hlRadius, u_hlRadius*0.4, distance(center,u_hlCenter)) : 0.0;`
   after `center` is known, and in the fragment shader add `in float v_hl;` then tint+glow:
   `float k=v_hl*(0.5+0.5*u_hlPulse); fragColor.rgb=mix(fragColor.rgb, vec3(1.0,0.85,0.2), 0.55*k);
   fragColor.rgb += 0.25*k;`
4. Note the method is named **`lookAtPoint`** (not `lookAt`, to avoid colliding with the
   `lookAt` matrix function imported from `math.ts`). The action executor must call
   `lookAtPoint`.

### Exact behavior the finished viewer methods must have
- `moveRelative(dir, amount=1)`: distance = `clampAmount(amount) * max(0.5,distance) * 0.8`;
  uses `applyMove` in `fly` mode when current mode is `orbit` (and re-derives the orbit target);
  cancels any running animation.
- `rotateView('clockwise'|'counterclockwise', amount=1)`: yaw by `±amount*π/6` via `applyLook`.
- `zoomView('in'|'out', amount=1)`: `applyZoom` (orbit shrinks distance; walk/fly dollies, scaled
  by scene distance).
- `flyTo(point, standoff?)`: turn toward `point` (set yaw/pitch from current eye→point), set
  `target=point`, `distance=standoff ?? sceneDist*0.55`, `position=point - forward*standoff`,
  then `startAnim(to, 700)`. Works in all three modes.
- `lookAtPoint(point)`: turn to face `point` keeping the eye fixed; `startAnim(to, 450)`.
- `pickAt(nx,ny)`: project every splat with `proj*view` (`transformPoint4`), skip `w<=0`,
  convert to top-left `[0,1]` (`sy = 1-(ndcy*0.5+0.5)`), keep the frontmost splat within ~8% of
  the frame, return `neighborhoodMedian(best)` (component median of splats within ~4% of scene
  extent). Stride-sample when `count>150000`. Return `null` if nothing qualifies.
- `getCameraSnapshot()`: `{ mode, fov:DEFAULT_FOV, eye, target, bounds }` — eye/target from
  `eyeForMode/targetForMode`; bounds from the parsed splats (or zeros).
- `highlightAt(point, radius?)` / `clearHighlight()` / `isAnimating` getter as described.

### Viewer unit tests to add (`frontend/tests/camera.test.ts` or a new file)
Headless, no GL needed (mirror existing tests): build a `SplatViewer` on a stub canvas,
`loadBuffer` a small `.splat` made with `encodeSplat`, then assert:
- `moveRelative('forward')` moves `getCameraSnapshot().eye` forward.
- `rotateView('clockwise')` changes yaw by ≈ +30°.
- `zoomView('in')` reduces `distance`.
- Put a single splat at a known world point, aim the camera at it, assert `pickAt(0.5,0.5)`
  returns ≈ that point.
- `highlightAt(p)` sets a positive `hlRadius`; `clearHighlight()` zeroes it.

## E. REMAINING FRONTEND WORK (not started — implement per the original Phases 4–5, with these deltas)

1. **`frontend/src/api/types.ts`** — add `AgentAction`, `AgentActResponse`, `CameraSnapshot`,
   `AgentTurn` mirroring the backend contract in section C.
2. **`frontend/src/api/client.ts`** — add:
   - `agentAct(body)` → `POST /agent/act` (note: base is `/api`, so path is `/agent/act`).
   - `uploadSplat(file: File, name: string)` → `POST /projects/upload_splat` as `FormData`
     (`form.append('file', file); form.append('name', name)`), returns `Project`. Follow the
     existing `uploadFiles` multipart pattern (do NOT set Content-Type manually).
3. **`frontend/src/components/SplatViewerReact.tsx`** — the handle already exposes `viewer`;
   the chat can call `ref.current.viewer.<method>` directly. Optionally add typed passthroughs.
   No structural change required.
4. **`frontend/src/components/AgentChat.tsx`** (new) — glass panel (match existing `.glass`
   overlay style) with message list, text input, send button, busy state, and an
   "agent is navigating…" hint shown while `viewer.isAnimating`. On send:
   - `screenshot_b64 = viewer.capture().split(',')[1]` (strip `data:` prefix).
   - `camera = viewer.getCameraSnapshot()`.
   - `history = messages.slice(-6)`.
   - call `apiClient.agentAct({message, screenshot_b64, camera, history})`.
   - push the `answer` as an assistant message; run `executeActions(viewer, res.actions)`.
   - Guard: if `viewer.splatCount === 0`, show "Load a world first."
5. **Action executor** (in `AgentChat.tsx`): map each action to a viewer call. For
   `fly_to`/`look_at`/`highlight`, call `viewer.pickAt(target_2d[0], target_2d[1])` → if a 3D
   point comes back, `flyTo` / `lookAtPoint` / (`highlightAt` + `lookAtPoint`); skip if pick is
   null. `move`/`rotate`/`zoom` → the matching method with `direction` + `amount`.
   `clear_highlight` → `clearHighlight()`; `reset_view` → `resetCamera()`. Execute in array
   order (so "fly_to then highlight" reads naturally).
6. **`frontend/src/pages/ViewerPage.tsx`** — mount `<AgentChat viewerRef={viewerRef} />`.
   Mount it in **both** the owner and the public (`shared`) layouts (requirement A5). It only
   needs the `viewerRef`, which already exists on the page.
7. **`frontend/src/pages/NewWorld.tsx`** — add an "Upload a .splat directly" affordance: a
   second file input accepting `.splat`; on choose, call `apiClient.uploadSplat(file, name)` and
   `navigate('/projects/'+proj.id+'/viewer')`. Keep it visually separate from the photo flow.
8. **`frontend/src/styles/global.css`** — styles for the chat panel (reuse existing glass/blur
   tokens; position bottom-right or a right rail; scrollable message list; input row).
9. **Tests** — `frontend/tests/AgentChat.test.tsx` (mirror `ViewerPage.test.tsx`): mock
   `apiClient.agentAct`, render with a mock `viewerRef` whose viewer exposes the agent methods
   as vi.fn()s, send a message, and assert (a) user+assistant messages render, (b) the correct
   viewer methods are called per action type, (c) `pickAt` is consulted for fly_to/highlight.
   Keep the existing 50 frontend tests green.

## F. VERIFICATION (Phase 5 — required; the user asked to prove it works on a REAL splat)

1. **Gates:** `cd backend && ./.venv/bin/python -m pytest -q` (expect 46+ passed, 4 skipped),
   `cd frontend && npm test` (≥50 passed), `cd frontend && npm run build` (clean).
2. **Get a real `.splat`:** download a known public Gaussian-splat asset (network is available
   in this environment) — e.g. an antimatter15-format `.splat` — OR, if download is blocked,
   synthesize a *recognizable multi-object* scene with `splat_format.write_splats` (e.g. a red
   sphere on the left, a blue cube on the right, a green ground plane) so "find the red object"
   has a deterministic answer. Upload it via `POST /api/projects/upload_splat`.
3. **Live agent verification (this is the real-splat proof):** with the backend running and the
   key loaded, send real prompts to `/api/agent/act` *with a real rendered screenshot of the
   scene* and assert the returned actions are sensible:
   - "find the red object" → expect a `highlight`/`fly_to` with a `target_2d` over the red region.
   - "what does the object on the right do?" → expect a `highlight` + a non-empty `answer`.
   - "move forward", "rotate clockwise", "zoom in", "reset the view" → matching action types.
   Because a true WebGL screenshot needs a real browser/GPU (documented limitation in README —
   not possible headlessly), do the **vision** verification by feeding the live endpoint a real
   image of a scene (a photo or a pre-rendered PNG of the test splat) and confirming valid,
   on-target actions; and verify `pickAt` against the real splat geometry in a unit test. State
   clearly in the final report which parts were verified headlessly vs. require a browser.
4. **Manual browser smoke (hand to the user):** start both servers, open the uploaded splat at
   `/view/:id`, open the agent panel, and run the search/explain prompts; confirm the camera
   visibly flies/rotates (agent viewpoint in real-time) and the highlight glows.

## G. Run instructions (for the user)
```
# backend (terminal 1) — key is in backend/.env
cd backend && ./.venv/bin/uvicorn app.main:app --reload --port 8000
# frontend (terminal 2)
cd frontend && npm install && npm run dev      # http://localhost:5173
```
Open "New 3D World" → upload a `.splat` (or build from photos) → open it → use the agent panel.
**Rotate the leaked API key** in the Anthropic console and replace it in `backend/.env`.
