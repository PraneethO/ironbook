/**
 * SplatViewer — WebGL2 Gaussian-splat renderer using antimatter15/splat's
 * rendering + sorting approach, wired to our own camera controls.
 *
 * Pipeline:
 *  - Splat data is packed (in a worker) into an RGBA32UI texture: position +
 *    color and the 6 half-float entries of the 3D covariance (see
 *    splatTexture.ts).
 *  - Each frame we build the view/projection in the OpenCV convention the
 *    antimatter shader expects (math.ts viewMatrixCV / projectionCV) from our
 *    controls' eye/target, and draw one instanced [-2,2] quad per splat
 *    (drawArraysInstanced TRIANGLE_FAN). The vertex shader projects the
 *    covariance and stretches the quad along its screen-space axes (~2.83σ).
 *  - Draw order comes from an O(n) 16-bit counting sort (sort.ts), run in the
 *    worker and re-issued only when the view direction changes meaningfully.
 *  - Fragments are premultiplied and composited with antimatter's blend for
 *    correct, halo-free alpha.
 *
 * The camera controls (orbit/walk/fly) are unchanged from the previous viewer.
 * A synchronous fallback (no Worker) keeps the engine usable/testable.
 */

import {
  applyLook,
  applyMove,
  applyZoom,
  attachControls,
  CameraMode,
  CameraState,
  createCameraState,
  DEFAULT_FOV,
  eyeForMode,
  fitToBounds,
  forwardFromAngles,
  MAX_PITCH,
  MIN_DISTANCE,
  MIN_PITCH,
  syncStateForMode,
  targetForMode,
} from './controls';
import {
  add,
  clamp,
  cross,
  focalLengthPx,
  length,
  lookAt,
  Mat4,
  multiply,
  normalize,
  perspective,
  projectionCV,
  scale,
  scale as vscale,
  sub,
  transformPoint4,
  Vec3,
  viewMatrixCV,
} from './math';
import { parseSplat, ParsedSplats } from './SplatLoader';
import { FRAGMENT_SHADER, VERTEX_SHADER } from './shaders';
import { sortByDepth } from './sort';
import { generateSplatTexture } from './splatTexture';
import { GridOverlay } from './grid';

export type { CameraMode };

export interface SplatViewerOptions {
  canvas: HTMLCanvasElement;
  mode?: CameraMode;
  onFps?: (fps: number) => void;
  onProgress?: (loaded: number, total: number) => void;
}

// Instanced quad corners in σ-ish units; the shader scales these by √(2λ).
const QUAD = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);

export class SplatViewer {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext | null = null;
  private mode: CameraMode;
  private onFps?: (fps: number) => void;
  private onProgress?: (loaded: number, total: number) => void;

  private state: CameraState = createCameraState();
  private controls: { update: (dt: number) => void; detach: () => void } | null = null;

  private splats: ParsedSplats | null = null;
  private _splatCount = 0;
  private sceneRadius = 1;

  private grid: GridOverlay | null = null;
  private gridVisible = true;

  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private quadBuffer: WebGLBuffer | null = null;
  private indexBuffer: WebGLBuffer | null = null;
  private dataTexture: WebGLTexture | null = null;
  private textureReady = false;
  private uniforms: Record<string, WebGLUniformLocation | null> = {};

  private worker: Worker | null = null;
  private sortGeneration = 0;
  private order: Uint32Array | null = null;
  private orderDirty = false;
  private lastSortDir: [number, number, number] | null = null;

  private rafId = 0;
  private disposed = false;
  private splatScale = 1.0;
  private bgColor: [number, number, number] = [0.118, 0.118, 0.118]; // #1e1e1e

  // --- agent-driven camera animation + highlight ---
  private anim: { from: CameraState; to: CameraState; t: number; dur: number } | null = null;
  private hlCenter: Vec3 | null = null;
  private hlRadius = 0;
  private static readonly MOVE_STEP = 0.8; // move = amount * sceneDist * STEP
  private static readonly ROT_STEP = Math.PI / 6; // amount 1.0 == 30 degrees

  private frameCount = 0;
  private lastFpsTs = 0;
  private lastFrameTs = 0;

  private firstFrameResolvers: Array<() => void> = [];

  constructor(opts: SplatViewerOptions) {
    this.canvas = opts.canvas;
    this.mode = opts.mode ?? 'orbit';
    this.onFps = opts.onFps;
    this.onProgress = opts.onProgress;

    // antimatter's premultiplied-alpha blend needs an alpha channel; the page
    // shows through where splats are transparent, so we paint the background via
    // CSS rather than the GL clear color.
    this.gl = this.canvas.getContext('webgl2', {
      antialias: false,
      premultipliedAlpha: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });

    if (this.gl) this.initGL(this.gl);
    this.applyBgToCanvas();

    this.controls = attachControls({
      canvas: this.canvas,
      state: this.state,
      getMode: () => this.mode,
    });

    this.lastFrameTs = now();
    this.lastFpsTs = this.lastFrameTs;
    if (typeof requestAnimationFrame !== 'undefined') {
      this.rafId = requestAnimationFrame(this.frame);
    }
  }

  // -- public API (CONTRACT §4) ------------------------------------------

  async load(url: string): Promise<void> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Could not load the 3D world (status ${res.status}).`);
    }
    const total = Number(res.headers.get('content-length') ?? 0);

    if (res.body && typeof res.body.getReader === 'function' && this.onProgress) {
      const reader = res.body.getReader();
      const chunks: Uint8Array[] = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          loaded += value.length;
          this.onProgress(loaded, total || loaded);
        }
      }
      const merged = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) {
        merged.set(c, off);
        off += c.length;
      }
      await this.loadBuffer(merged.buffer);
      return;
    }

    const buf = await res.arrayBuffer();
    if (this.onProgress) this.onProgress(buf.byteLength, buf.byteLength);
    await this.loadBuffer(buf);
  }

  async loadBuffer(buf: ArrayBuffer): Promise<void> {
    const parsed = parseSplat(buf);
    this.splats = parsed;
    this._splatCount = parsed.count;
    this.textureReady = false;
    this.order = null;
    this.lastSortDir = null;

    if (parsed.count > 0) {
      fitToBounds(this.state, parsed.bounds.min, parsed.bounds.max, DEFAULT_FOV);
      this.sceneRadius = Math.max(
        1e-3,
        length(vscale(sub(parsed.bounds.max, parsed.bounds.min), 0.5)),
      );
      if (this.gl) this.grid?.setBounds(this.gl, parsed.bounds.min, parsed.bounds.max);
    }

    this.initWorker(buf, parsed.count);

    await new Promise<void>((resolve) => {
      this.firstFrameResolvers.push(resolve);
      if (typeof requestAnimationFrame === 'undefined') {
        this.renderOnce();
      }
    });
  }

  setMode(mode: CameraMode): void {
    if (mode === this.mode) return;
    syncStateForMode(this.state, this.mode, mode);
    this.mode = mode;
  }

  getMode(): CameraMode {
    return this.mode;
  }

  /** Show/hide the ground grid + XYZ axes overlay. */
  setGridVisible(visible: boolean): void {
    this.gridVisible = visible;
  }

  isGridVisible(): boolean {
    return this.gridVisible;
  }

  resetCamera(): void {
    if (this.splats && this.splats.count > 0) {
      fitToBounds(this.state, this.splats.bounds.min, this.splats.bounds.max, DEFAULT_FOV);
    } else {
      this.state = createCameraState();
    }
  }

  // -- agent-facing imperative API --------------------------------------
  // All motion reuses the pure camera math in controls.ts and scales to the
  // scene via state.distance, so it feels right on any loaded .splat.

  private sceneDist(): number {
    return Math.max(0.5, this.state.distance);
  }

  /** Step the camera; orbit nudges behave like a free 'fly' move. */
  moveRelative(
    dir: 'forward' | 'backward' | 'left' | 'right' | 'up' | 'down',
    amount = 1,
  ): void {
    const d = clampAmount(amount) * this.sceneDist() * SplatViewer.MOVE_STEP;
    const map: Record<string, Vec3> = {
      forward: [0, 0, d],
      backward: [0, 0, -d],
      left: [-d, 0, 0],
      right: [d, 0, 0],
      up: [0, d, 0],
      down: [0, -d, 0],
    };
    const mode: CameraMode = this.mode === 'orbit' ? 'fly' : this.mode;
    applyMove(this.state, mode, map[dir]);
    if (this.mode === 'orbit') {
      // keep the orbit target in front so subsequent orbit math stays sane
      const fwd = forwardFromAngles(this.state.yaw, this.state.pitch);
      this.state.target = add(this.state.position, scale(fwd, this.state.distance));
    }
    this.anim = null;
  }

  /** Yaw the view. clockwise = +yaw. amount 1.0 == 30 degrees. */
  rotateView(dir: 'clockwise' | 'counterclockwise', amount = 1): void {
    const dy = (dir === 'clockwise' ? 1 : -1) * clampAmount(amount) * SplatViewer.ROT_STEP;
    applyLook(this.state, dy, 0);
    this.anim = null;
  }

  /** Dolly / zoom. in = closer. Animates smoothly. */
  zoomView(dir: 'in' | 'out', amount = 1): void {
    const a = (dir === 'in' ? 1 : -1) * clampAmount(amount) * 0.6;
    const to = cloneState(this.state);
    if (this.mode === 'orbit') applyZoom(to, this.mode, a);
    else applyZoom(to, this.mode, a * this.sceneDist());
    this.startAnim(to, 350);
  }

  /** Smoothly turn toward + approach a 3D point so it's centered and framed. */
  flyTo(point: Vec3, standoff?: number): void {
    const eye = eyeForMode(this.state, this.mode);
    const dir = normalize(sub(point, eye));
    const yaw = Math.atan2(dir[0], -dir[2]);
    const pitch = clamp(Math.asin(clamp(dir[1], -1, 1)), MIN_PITCH, MAX_PITCH);
    const fwd = forwardFromAngles(yaw, pitch);
    const so = Math.max(MIN_DISTANCE, standoff ?? this.sceneDist() * 0.55);
    const to = cloneState(this.state);
    to.yaw = yaw;
    to.pitch = pitch;
    to.target = point;
    to.distance = so;
    to.position = sub(point, scale(fwd, so)); // authoritative eye for walk/fly
    this.startAnim(to, 700);
  }

  /** Smoothly turn to face a 3D point without travelling. */
  lookAtPoint(point: Vec3): void {
    const eye = eyeForMode(this.state, this.mode);
    const dir = normalize(sub(point, eye));
    const yaw = Math.atan2(dir[0], -dir[2]);
    const pitch = clamp(Math.asin(clamp(dir[1], -1, 1)), MIN_PITCH, MAX_PITCH);
    const fwd = forwardFromAngles(yaw, pitch);
    const to = cloneState(this.state);
    to.yaw = yaw;
    to.pitch = pitch;
    to.position = eye; // keep eye fixed
    to.distance = this.state.distance;
    to.target = add(eye, scale(fwd, this.state.distance));
    this.startAnim(to, 450);
  }

  /** Camera + scene info sent to the agent backend each turn. */
  getCameraSnapshot(): {
    mode: CameraMode;
    fov: number;
    eye: Vec3;
    target: Vec3;
    bounds: { min: Vec3; max: Vec3 };
  } {
    const b = this.splats
      ? this.splats.bounds
      : { min: [0, 0, 0] as Vec3, max: [0, 0, 0] as Vec3 };
    return {
      mode: this.mode,
      fov: DEFAULT_FOV,
      eye: eyeForMode(this.state, this.mode),
      target: targetForMode(this.state, this.mode),
      bounds: { min: b.min as Vec3, max: b.max as Vec3 },
    };
  }

  /**
   * Convert a normalized screen point (nx, ny in [0,1], top-left origin — the
   * same frame as capture()/the agent's screenshot) to a 3D scene point by
   * projecting splats and choosing the frontmost one near that pixel, then
   * returning the local median for stability. Returns null if nothing is there.
   */
  pickAt(nx: number, ny: number): Vec3 | null {
    const s = this.splats;
    if (!s || s.count === 0) return null;
    const w = this.canvas.width || 1;
    const h = this.canvas.height || 1;
    const view = lookAt(
      eyeForMode(this.state, this.mode),
      targetForMode(this.state, this.mode),
      [0, 1, 0],
    );
    const vp = multiply(perspective(DEFAULT_FOV, w / h, 0.01, 1000), view);
    const RADIUS = 0.08; // accept splats within 8% of the frame
    const r2max = RADIUS * RADIUS;
    let best = -1;
    let bestScore = Infinity;
    // Stride-sample very large scenes to keep picks instant.
    const stride = s.count > 150000 ? Math.ceil(s.count / 150000) : 1;
    for (let i = 0; i < s.count; i += stride) {
      const px = s.positions[i * 3];
      const py = s.positions[i * 3 + 1];
      const pz = s.positions[i * 3 + 2];
      const c = transformPoint4(vp, [px, py, pz]);
      if (c[3] <= 0) continue; // behind camera
      const ndcx = c[0] / c[3];
      const ndcy = c[1] / c[3];
      const sx = ndcx * 0.5 + 0.5;
      const sy = 1 - (ndcy * 0.5 + 0.5); // flip to top-left origin
      const dx = sx - nx;
      const dy = sy - ny;
      const d2 = dx * dx + dy * dy;
      if (d2 > r2max) continue;
      const depth = c[2] / c[3]; // prefer nearer (frontmost) splats
      const score = d2 + depth * 0.0015;
      if (score < bestScore) {
        bestScore = score;
        best = i;
      }
    }
    if (best < 0) return null;
    return this.neighborhoodMedian(best);
  }

  /** Median position of splats near the chosen splat (robust to floaters). */
  private neighborhoodMedian(index: number): Vec3 {
    const s = this.splats!;
    const cx = s.positions[index * 3];
    const cy = s.positions[index * 3 + 1];
    const cz = s.positions[index * 3 + 2];
    const ext = Math.max(
      0.001,
      s.bounds.max[0] - s.bounds.min[0],
      s.bounds.max[1] - s.bounds.min[1],
      s.bounds.max[2] - s.bounds.min[2],
    );
    const rad = ext * 0.04;
    const rad2 = rad * rad;
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    const stride = s.count > 150000 ? Math.ceil(s.count / 150000) : 1;
    for (let i = 0; i < s.count; i += stride) {
      const dx = s.positions[i * 3] - cx;
      const dy = s.positions[i * 3 + 1] - cy;
      const dz = s.positions[i * 3 + 2] - cz;
      if (dx * dx + dy * dy + dz * dz <= rad2) {
        xs.push(s.positions[i * 3]);
        ys.push(s.positions[i * 3 + 1]);
        zs.push(s.positions[i * 3 + 2]);
      }
    }
    if (xs.length === 0) return [cx, cy, cz];
    const med = (a: number[]) => {
      a.sort((p, q) => p - q);
      return a[a.length >> 1];
    };
    return [med(xs), med(ys), med(zs)];
  }

  /** Always returns a 3D point — uses pickAt if available, otherwise ray-casts
   *  to the scene midpoint depth. Used as fallback when pickAt returns null. */
  rayToSceneDepth(nx: number, ny: number): Vec3 {
    // Try real pick first
    const picked = this.pickAt(nx, ny);
    if (picked) return picked;

    // Fallback: construct a ray from the camera through the normalized screen point
    // and return a point at the current scene distance
    const eye = eyeForMode(this.state, this.mode);
    const tgt = targetForMode(this.state, this.mode);
    const w = this.canvas.width || 400;
    const h = this.canvas.height || 300;
    const aspect = w / h;

    // Convert normalized [0,1] top-left to NDC [-1,1]
    const ndcx = nx * 2 - 1;
    const ndcy = -(ny * 2 - 1); // flip Y

    // Unproject: compute the ray direction in world space
    // Use the same FOV as the viewer
    const fovHalfTan = Math.tan(DEFAULT_FOV / 2);
    const localX = ndcx * fovHalfTan * aspect;
    const localY = ndcy * fovHalfTan;

    // Camera axes
    const forward = normalize(sub(tgt, eye));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);

    // Ray direction in world space
    const rayDir = normalize(
      add(add(scale(right, localX), scale(up, localY)), forward)
    );

    // Return point at scene distance along the ray
    const depth = Math.max(0.5, this.state.distance * 0.6);
    return add(eye, scale(rayDir, depth));
  }

  /** Tint + glow splats within a sphere around `point`. */
  highlightAt(point: Vec3, radius?: number): void {
    this.hlCenter = point;
    this.hlRadius = radius ?? Math.max(0.25, this.sceneDist() * 0.25);
  }

  clearHighlight(): void {
    this.hlCenter = null;
    this.hlRadius = 0;
  }

  /** True while a flyTo/lookAt animation is playing (agent is navigating). */
  get isAnimating(): boolean {
    return this.anim !== null;
  }

  private startAnim(to: CameraState, dur: number): void {
    this.anim = { from: cloneState(this.state), to, t: 0, dur: Math.max(1, dur) };
  }

  private updateAnimation(dtMs: number): void {
    if (!this.anim) return;
    this.anim.t = Math.min(1, this.anim.t + dtMs / this.anim.dur);
    const e = easeInOut(this.anim.t);
    const a = this.anim.from;
    const b = this.anim.to;
    this.state.target = lerpVec(a.target, b.target, e);
    this.state.position = lerpVec(a.position, b.position, e);
    this.state.distance = lerp(a.distance, b.distance, e);
    this.state.yaw = lerpAngle(a.yaw, b.yaw, e);
    this.state.pitch = lerp(a.pitch, b.pitch, e);
    if (this.anim.t >= 1) this.anim = null;
  }

  capture(): string {
    if (this.gl) this.renderOnce();
    try {
      return this.canvas.toDataURL('image/png');
    } catch {
      return '';
    }
  }

  get splatCount(): number {
    return this._splatCount;
  }

  setSplatScale(scale: number): void {
    this.splatScale = Math.max(0.05, scale);
  }

  /** Change the background color by CSS-like name or hex. */
  setBackgroundByName(name: string): void {
    const colors: Record<string, [number, number, number]> = {
      black: [0, 0, 0],
      white: [1, 1, 1],
      dark: [0.04, 0.05, 0.07],
      midnight: [0.02, 0.02, 0.08],
      navy: [0.03, 0.06, 0.18],
      gray: [0.15, 0.15, 0.15],
      slate: [0.08, 0.09, 0.12],
    };
    const c = colors[name.toLowerCase()] ?? colors.dark;
    this.setBackgroundColor(c[0], c[1], c[2]);
  }

  /** Multiply background and splat brightness. */
  setBrightness(factor: number): void {
    const f = Math.max(0.1, Math.min(3.0, factor));
    const base = [0.04, 0.05, 0.07];
    this.setBackgroundColor(base[0] * f, base[1] * f, base[2] * f);
    this.setSplatScale(Math.max(0.3, Math.min(3.0, f)));
  }

  setBackgroundColor(r: number, g: number, b: number): void {
    this.bgColor = [r, g, b];
    this.applyBgToCanvas();
  }

  dispose(): void {
    this.disposed = true;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.controls?.detach();
    this.controls = null;
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    const gl = this.gl;
    if (gl) {
      if (this.program) gl.deleteProgram(this.program);
      [this.quadBuffer, this.indexBuffer].forEach((b) => b && gl.deleteBuffer(b));
      if (this.dataTexture) gl.deleteTexture(this.dataTexture);
      if (this.vao) gl.deleteVertexArray(this.vao);
      this.grid?.dispose(gl);
    }
    this.resolveFirstFrame();
  }

  // -- GL setup ----------------------------------------------------------

  private initGL(gl: WebGL2RenderingContext) {
    const program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    if (!program) return;
    this.program = program;

    this.uniforms = {
      projection: gl.getUniformLocation(program, 'projection'),
      view: gl.getUniformLocation(program, 'view'),
      focal: gl.getUniformLocation(program, 'focal'),
      viewport: gl.getUniformLocation(program, 'viewport'),
      u_texture: gl.getUniformLocation(program, 'u_texture'),
      u_splatScale: gl.getUniformLocation(program, 'u_splatScale'),
      // agent highlight — null no-ops with the covariance shader; kept so the
      // agent's highlightAt/clearHighlight bookkeeping has valid locations.
      u_hlCenter: gl.getUniformLocation(program, 'u_hlCenter'),
      u_hlRadius: gl.getUniformLocation(program, 'u_hlRadius'),
      u_hlPulse: gl.getUniformLocation(program, 'u_hlPulse'),
    };

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    this.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.INT, 0, 0);
    gl.vertexAttribDivisor(1, 1);

    this.dataTexture = gl.createTexture();

    gl.bindVertexArray(null);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
      gl.ONE_MINUS_DST_ALPHA,
      gl.ONE,
      gl.ONE_MINUS_DST_ALPHA,
      gl.ONE,
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    this.grid = new GridOverlay(gl);
  }

  private uploadTexture(
    gl: WebGL2RenderingContext,
    texdata: Uint32Array,
    texwidth: number,
    texheight: number,
  ) {
    if (!this.dataTexture) return;
    gl.bindTexture(gl.TEXTURE_2D, this.dataTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32UI,
      texwidth,
      texheight,
      0,
      gl.RGBA_INTEGER,
      gl.UNSIGNED_INT,
      texdata,
    );
    this.textureReady = true;
  }

  private uploadOrder(gl: WebGL2RenderingContext) {
    if (!this.order || !this.indexBuffer) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.order, gl.DYNAMIC_DRAW);
    this.orderDirty = false;
  }

  // -- worker / sorting --------------------------------------------------

  private initWorker(buf: ArrayBuffer, count: number) {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    if (count === 0) return;

    // Transfer a copy so we never detach the caller's buffer.
    const workerBuf = buf.slice(0);

    try {
      if (typeof Worker === 'undefined') throw new Error('no worker');
      this.worker = new Worker(new URL('./sortWorker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e: MessageEvent) => {
        const msg = e.data;
        if (msg?.type === 'texture') {
          if (this.gl) this.uploadTexture(this.gl, msg.texdata, msg.texwidth, msg.texheight);
        } else if (msg?.type === 'sorted' && msg.generation === this.sortGeneration) {
          this.order = msg.depthIndex as Uint32Array;
          this.orderDirty = true;
        }
      };
      this.worker.postMessage({ type: 'init', buffer: workerBuf, count }, [workerBuf]);
    } catch {
      // Synchronous fallback: build texture + initial order on the main thread.
      this.worker = null;
      const tex = generateSplatTexture(workerBuf, count);
      if (this.gl) this.uploadTexture(this.gl, tex.texdata, tex.texwidth, tex.texheight);
      // positions for the fallback sort come straight from the parse.
    }
  }

  private requestSort(viewProj: Mat4) {
    if (!this.splats || this.splats.count === 0) return;

    // Only re-sort when the view direction (viewProj row 2) changes enough;
    // pure translation keeps the order stable (matches antimatter).
    const dir = normalize3(viewProj[2], viewProj[6], viewProj[10]);
    if (this.lastSortDir && this.order) {
      const dot =
        dir[0] * this.lastSortDir[0] +
        dir[1] * this.lastSortDir[1] +
        dir[2] * this.lastSortDir[2];
      if (dot > 0.999) return;
    }
    this.lastSortDir = dir;
    this.sortGeneration++;

    if (this.worker) {
      const vp = new Float32Array(viewProj);
      this.worker.postMessage({ type: 'sort', viewProj: vp, generation: this.sortGeneration }, [
        vp.buffer,
      ]);
    } else {
      this.order = sortByDepth(this.splats.positions, this.splats.count, viewProj);
      this.orderDirty = true;
    }
  }

  // -- render loop -------------------------------------------------------

  private frame = () => {
    if (this.disposed) return;
    const t = now();
    this.renderOnce(t);
    this.frameCount++;
    if (this.onFps && t - this.lastFpsTs >= 500) {
      const fps = (this.frameCount * 1000) / (t - this.lastFpsTs);
      this.onFps(Math.round(fps));
      this.frameCount = 0;
      this.lastFpsTs = t;
    }
    this.rafId = requestAnimationFrame(this.frame);
  };

  private renderOnce(t = now()) {
    const gl = this.gl;
    const dt = Math.min(0.1, Math.max(0, (t - this.lastFrameTs) / 1000));
    this.lastFrameTs = t;

    this.updateAnimation(dt * 1000);
    this.controls?.update(dt);

    if (!gl || !this.program) {
      this.resolveFirstFrame();
      return;
    }

    this.resizeIfNeeded();

    const w = this.canvas.width || 1;
    const h = this.canvas.height || 1;

    const eye = eyeForMode(this.state, this.mode);
    const target = targetForMode(this.state, this.mode);
    const near = Math.max(1e-3, this.sceneRadius * 0.01);
    const far = this.sceneRadius * 1000 + 1000;
    const view = viewMatrixCV(eye, target, [0, 1, 0]);
    const proj = projectionCV(DEFAULT_FOV, w, h, near, far);
    const viewProj = multiply(proj, view);
    const focal = focalLengthPx(DEFAULT_FOV, h);

    this.requestSort(viewProj);
    if (this.orderDirty) this.uploadOrder(gl);

    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); // transparent; CSS supplies the background
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Floor grid + axes drawn first so the splats composite on top.
    if (this.gridVisible && this.splats && this.splats.count > 0) {
      this.grid?.draw(gl, view, proj);
    }

    if (
      this.splats &&
      this.splats.count > 0 &&
      this.vao &&
      this.order &&
      this.textureReady
    ) {
      gl.useProgram(this.program);
      gl.bindVertexArray(this.vao);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.dataTexture);
      gl.uniform1i(this.uniforms.u_texture, 0);

      gl.uniformMatrix4fv(this.uniforms.projection, false, proj);
      gl.uniformMatrix4fv(this.uniforms.view, false, view);
      gl.uniform2f(this.uniforms.focal, focal, focal);
      gl.uniform2f(this.uniforms.viewport, w, h);
      gl.uniform1f(this.uniforms.u_splatScale, this.splatScale);

      const hc = this.hlCenter ?? [0, 0, 0];
      gl.uniform3f(this.uniforms.u_hlCenter, hc[0], hc[1], hc[2]);
      gl.uniform1f(this.uniforms.u_hlRadius, this.hlRadius);
      gl.uniform1f(this.uniforms.u_hlPulse, 0.5 + 0.5 * Math.sin(t * 0.004));

      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.splats.count);
      gl.bindVertexArray(null);
    }

    this.resolveFirstFrame();
  }

  private applyBgToCanvas() {
    if (!this.canvas.style) return; // minimal/headless canvas (tests) — nothing to paint
    const [r, g, b] = this.bgColor;
    const to255 = (v: number) => Math.round(Math.max(0, Math.min(1, v)) * 255);
    this.canvas.style.backgroundColor = `rgb(${to255(r)}, ${to255(g)}, ${to255(b)})`;
  }

  private resolveFirstFrame() {
    if (this.firstFrameResolvers.length) {
      const rs = this.firstFrameResolvers;
      this.firstFrameResolvers = [];
      rs.forEach((r) => r());
    }
  }

  private resizeIfNeeded() {
    const dpr = Math.min(2, (typeof window !== 'undefined' && window.devicePixelRatio) || 1);
    const cssW = this.canvas.clientWidth || this.canvas.width || 1;
    const cssH = this.canvas.clientHeight || this.canvas.height || 1;
    const targetW = Math.max(1, Math.floor(cssW * dpr));
    const targetH = Math.max(1, Math.floor(cssH * dpr));
    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }
  }
}

function normalize3(x: number, y: number, z: number): [number, number, number] {
  const l = Math.hypot(x, y, z) || 1;
  return [x / l, y / l, z / l];
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

// --- agent camera helpers ------------------------------------------------

function clampAmount(a: number): number {
  return Math.min(3, Math.max(0, a || 1));
}

function cloneState(s: CameraState): CameraState {
  return {
    target: [s.target[0], s.target[1], s.target[2]],
    distance: s.distance,
    yaw: s.yaw,
    pitch: s.pitch,
    position: [s.position[0], s.position[1], s.position[2]],
  };
}

function easeInOut(t: number): number {
  // easeInOutCubic
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function lerp(a: number, b: number, e: number): number {
  return a + (b - a) * e;
}

function lerpVec(a: Vec3, b: Vec3, e: number): Vec3 {
  return [lerp(a[0], b[0], e), lerp(a[1], b[1], e), lerp(a[2], b[2], e)];
}

function lerpAngle(a: number, b: number, e: number): number {
  // shortest angular path: wrap (b-a) into [-π, π]
  let diff = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (diff < -Math.PI) diff += 2 * Math.PI;
  return a + diff * e;
}

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader | null {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error('Splat shader compile error: ' + log);
  }
  return sh;
}

function createProgram(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram | null {
  const v = compileShader(gl, gl.VERTEX_SHADER, vs);
  const f = compileShader(gl, gl.FRAGMENT_SHADER, fs);
  if (!v || !f) return null;
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, v);
  gl.attachShader(program, f);
  gl.linkProgram(program);
  gl.deleteShader(v);
  gl.deleteShader(f);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error('Splat program link error: ' + log);
  }
  return program;
}
