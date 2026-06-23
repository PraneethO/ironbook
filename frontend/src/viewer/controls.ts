/**
 * controls.ts — camera state, pure camera math, and DOM control attachment for
 * orbit / walk / fly modes (per 04_user_experience.md and CONTRACT.md §4).
 *
 * The camera math (eye/forward computation, orbit/dolly/move updates, fit to
 * bounds) lives in pure functions so it can be unit-tested without a browser.
 * `attachControls` wires pointer/keyboard/wheel events to the canvas and
 * returns a detach function used by SplatViewer.dispose().
 */

import {
  add,
  clamp,
  cross,
  length,
  normalize,
  scale,
  sub,
  Vec3,
} from './math';

export type CameraMode = 'orbit' | 'walk' | 'fly';

export interface CameraState {
  /** Orbit target / focus point. */
  target: Vec3;
  /** Spherical orbit params around target. */
  distance: number;
  yaw: number; // radians, around Y (up)
  pitch: number; // radians, vertical look
  /** Free position used directly in walk/fly modes. */
  position: Vec3;
}

export const DEFAULT_FOV = (60 * Math.PI) / 180;
export const MIN_PITCH = -Math.PI / 2 + 0.05;
export const MAX_PITCH = Math.PI / 2 - 0.05;
export const MIN_DISTANCE = 0.05;

export function createCameraState(): CameraState {
  return {
    target: [0, 0, 0],
    distance: 5,
    yaw: 0,
    pitch: -0.2,
    position: [0, 0, 5],
  };
}

/** Unit forward vector implied by yaw/pitch (camera looks toward -Z at yaw=0). */
export function forwardFromAngles(yaw: number, pitch: number): Vec3 {
  const cp = Math.cos(pitch);
  return normalize([
    Math.sin(yaw) * cp,
    Math.sin(pitch),
    -Math.cos(yaw) * cp,
  ]);
}

/**
 * Resolve the camera eye position for the current mode.
 * - orbit: eye orbits around target at `distance`.
 * - walk/fly: eye is `position` directly.
 */
export function eyeForMode(state: CameraState, mode: CameraMode): Vec3 {
  if (mode === 'orbit') {
    const fwd = forwardFromAngles(state.yaw, state.pitch);
    return sub(state.target, scale(fwd, state.distance));
  }
  return state.position;
}

/** Resolve the look-at target for the current mode. */
export function targetForMode(state: CameraState, mode: CameraMode): Vec3 {
  if (mode === 'orbit') return state.target;
  const fwd = forwardFromAngles(state.yaw, state.pitch);
  return add(state.position, fwd);
}

/** Apply a drag delta (in radians) to orbit/look angles, clamping pitch. */
export function applyLook(state: CameraState, dYaw: number, dPitch: number): void {
  state.yaw += dYaw;
  state.pitch = clamp(state.pitch + dPitch, MIN_PITCH, MAX_PITCH);
}

/** Dolly/zoom: orbit shrinks distance; walk/fly moves along forward. */
export function applyZoom(state: CameraState, mode: CameraMode, amount: number): void {
  if (mode === 'orbit') {
    // amount > 0 means zoom in
    const factor = Math.exp(-amount);
    state.distance = Math.max(MIN_DISTANCE, state.distance * factor);
  } else {
    const fwd = forwardFromAngles(state.yaw, state.pitch);
    state.position = add(state.position, scale(fwd, amount));
  }
}

/**
 * Move in walk/fly mode. `move` is local-space (x=strafe, y=up, z=forward).
 * In walk mode vertical movement and the forward's Y component are flattened
 * to the ground plane. Mutates state.position.
 */
export function applyMove(state: CameraState, mode: CameraMode, move: Vec3): void {
  let fwd = forwardFromAngles(state.yaw, state.pitch);
  if (mode === 'walk') {
    fwd = normalize([fwd[0], 0, fwd[2]]);
  }
  const right = normalize(cross(fwd, [0, 1, 0]));
  const up: Vec3 = [0, 1, 0];

  let delta = add(scale(right, move[0]), scale(fwd, move[2]));
  if (mode === 'fly') {
    delta = add(delta, scale(up, move[1]));
  }
  state.position = add(state.position, delta);
}

/**
 * When switching modes, keep the view continuous: derive position from the
 * current orbit eye, or recompute orbit angles from a free position.
 */
export function syncStateForMode(
  state: CameraState,
  from: CameraMode,
  to: CameraMode,
): void {
  if (from === to) return;
  if (from === 'orbit') {
    // Adopt the orbit eye as the free position; keep yaw/pitch (they already
    // describe the look direction from eye toward target).
    state.position = eyeForMode(state, 'orbit');
  } else if (to === 'orbit') {
    // Build an orbit target in front of the free camera at current distance.
    const fwd = forwardFromAngles(state.yaw, state.pitch);
    state.target = add(state.position, scale(fwd, state.distance));
  }
}

/**
 * Fit the camera to axis-aligned bounds: center the target and back off far
 * enough that the whole scene fits in view for the given vertical FOV.
 */
export function fitToBounds(
  state: CameraState,
  min: Vec3,
  max: Vec3,
  fovY = DEFAULT_FOV,
): void {
  const center: Vec3 = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const radius = Math.max(0.5, length(scale(sub(max, min), 0.5)));
  const dist = (radius / Math.sin(fovY / 2)) * 1.1;

  state.target = center;
  state.distance = Math.max(MIN_DISTANCE, dist);
  state.yaw = 0;
  state.pitch = -0.25;
  // keep position consistent for walk/fly
  state.position = eyeForMode(state, 'orbit');
}

// ---------------------------------------------------------------------------
// DOM control attachment
// ---------------------------------------------------------------------------

export interface ControlsContext {
  canvas: HTMLCanvasElement;
  state: CameraState;
  getMode: () => CameraMode;
}

/**
 * Attach pointer/keyboard/wheel handlers to the canvas. Returns a per-frame
 * `update(dt)` to apply held-key movement, and a `detach()` to remove all
 * listeners. Movement speed scales with scene distance for sensible feel.
 */
export function attachControls(ctx: ControlsContext): {
  update: (dtSeconds: number) => void;
  detach: () => void;
} {
  const { canvas, state } = ctx;
  const keys = new Set<string>();
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const LOOK_SPEED = 0.005;

  const onPointerDown = (e: PointerEvent) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture?.(e.pointerId);
  };

  const onPointerMove = (e: PointerEvent) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    const mode = ctx.getMode();
    if (mode === 'orbit') {
      applyLook(state, dx * LOOK_SPEED, dy * LOOK_SPEED);
    } else {
      // walk/fly: invert so dragging looks naturally
      applyLook(state, dx * LOOK_SPEED, -dy * LOOK_SPEED);
    }
  };

  const onPointerUp = (e: PointerEvent) => {
    dragging = false;
    canvas.releasePointerCapture?.(e.pointerId);
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    const mode = ctx.getMode();
    const step = -e.deltaY * 0.002;
    if (mode === 'orbit') {
      applyZoom(state, mode, step);
    } else {
      applyZoom(state, mode, step * Math.max(0.5, state.distance));
    }
  };

  const onKeyDown = (e: KeyboardEvent) => {
    keys.add(e.key.toLowerCase());
  };
  const onKeyUp = (e: KeyboardEvent) => {
    keys.delete(e.key.toLowerCase());
  };

  const update = (dt: number) => {
    const mode = ctx.getMode();
    if (mode === 'orbit') return;
    const fast = keys.has('shift') ? 3 : 1;
    const speed = Math.max(0.5, state.distance) * 0.8 * fast * dt;
    let mx = 0;
    let my = 0;
    let mz = 0;
    if (keys.has('w')) mz += 1;
    if (keys.has('s')) mz -= 1;
    if (keys.has('a')) mx -= 1;
    if (keys.has('d')) mx += 1;
    if (mode === 'fly') {
      if (keys.has(' ')) my += 1;
      if (keys.has('c')) my -= 1;
    }
    if (mx || my || mz) {
      applyMove(state, mode, [mx * speed, my * speed, mz * speed]);
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  const detach = () => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointercancel', onPointerUp);
    canvas.removeEventListener('wheel', onWheel);
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('keyup', onKeyUp);
    keys.clear();
  };

  return { update, detach };
}
