import { describe, expect, it } from 'vitest';
import {
  applyLook,
  applyMove,
  applyZoom,
  createCameraState,
  eyeForMode,
  fitToBounds,
  forwardFromAngles,
  MAX_PITCH,
  MIN_PITCH,
  syncStateForMode,
  targetForMode,
} from '../src/viewer/controls';
import { lookAt, perspective, viewDepth } from '../src/viewer/math';
import { sortByDepth } from '../src/viewer/sort';

describe('camera math (pure)', () => {
  it('forwardFromAngles points toward -Z at yaw=0, pitch=0', () => {
    const f = forwardFromAngles(0, 0);
    expect(f[0]).toBeCloseTo(0, 5);
    expect(f[1]).toBeCloseTo(0, 5);
    expect(f[2]).toBeCloseTo(-1, 5);
  });

  it('applyLook clamps pitch to the allowed range', () => {
    const s = createCameraState();
    applyLook(s, 0, 100);
    expect(s.pitch).toBeLessThanOrEqual(MAX_PITCH);
    applyLook(s, 0, -100);
    expect(s.pitch).toBeGreaterThanOrEqual(MIN_PITCH);
  });

  it('orbit zoom shrinks distance and never goes non-positive', () => {
    const s = createCameraState();
    s.distance = 5;
    applyZoom(s, 'orbit', 1); // zoom in
    expect(s.distance).toBeLessThan(5);
    for (let i = 0; i < 100; i++) applyZoom(s, 'orbit', 5);
    expect(s.distance).toBeGreaterThan(0);
  });

  it('walk move stays on the ground plane (no Y change)', () => {
    const s = createCameraState();
    s.position = [0, 1.6, 0];
    s.pitch = -0.6; // looking down
    applyMove(s, 'walk', [0, 0, 1]); // move forward
    expect(s.position[1]).toBeCloseTo(1.6, 5); // Y unchanged in walk
  });

  it('fly move can change Y via the up component', () => {
    const s = createCameraState();
    s.position = [0, 0, 0];
    applyMove(s, 'fly', [0, 1, 0]); // move up
    expect(s.position[1]).toBeGreaterThan(0);
  });

  it('eyeForMode/targetForMode are consistent in orbit mode', () => {
    const s = createCameraState();
    s.target = [1, 2, 3];
    s.distance = 4;
    const eye = eyeForMode(s, 'orbit');
    const tgt = targetForMode(s, 'orbit');
    expect(tgt).toEqual([1, 2, 3]);
    // eye should be `distance` away from target
    const d = Math.hypot(eye[0] - tgt[0], eye[1] - tgt[1], eye[2] - tgt[2]);
    expect(d).toBeCloseTo(4, 4);
  });

  it('fitToBounds centers the target and backs off', () => {
    const s = createCameraState();
    fitToBounds(s, [-1, -1, -1], [1, 1, 1]);
    expect(s.target).toEqual([0, 0, 0]);
    expect(s.distance).toBeGreaterThan(1);
  });

  it('syncStateForMode keeps a continuous position when leaving orbit', () => {
    const s = createCameraState();
    s.target = [0, 0, 0];
    s.distance = 5;
    const eyeBefore = eyeForMode(s, 'orbit');
    syncStateForMode(s, 'orbit', 'fly');
    expect(s.position[0]).toBeCloseTo(eyeBefore[0], 4);
    expect(s.position[1]).toBeCloseTo(eyeBefore[1], 4);
    expect(s.position[2]).toBeCloseTo(eyeBefore[2], 4);
  });
});

describe('matrices', () => {
  it('perspective maps a centered point to the screen center (x=y=0)', () => {
    const p = perspective((60 * Math.PI) / 180, 1, 0.1, 100);
    // a point straight ahead at view -Z projects to ndc (0,0)
    // emulate by checking matrix structure: m[11] === -1
    expect(p[11]).toBe(-1);
  });

  it('lookAt + viewDepth: farther points have larger depth', () => {
    const view = lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]);
    const near = viewDepth(view, [0, 0, 0]);
    const far = viewDepth(view, [0, 0, -10]);
    expect(far).toBeGreaterThan(near);
  });
});

describe('sortByDepth', () => {
  it('orders indices farthest-first for back-to-front blending', () => {
    // three splats along Z; camera at +5 looking toward -Z
    const positions = new Float32Array([
      0, 0, 0, // mid
      0, 0, -5, // far
      0, 0, 4, // near (behind nothing, closest to camera)
    ]);
    const view = lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]);
    const viewRow: [number, number, number, number] = [view[2], view[6], view[10], view[14]];
    const order = sortByDepth(positions, 3, viewRow);
    // farthest first => the z=-5 splat (index 1) should come before z=4 (index 2)
    expect(order[0]).toBe(1);
    expect(order[2]).toBe(2);
  });

  it('reuses the provided output buffer when large enough', () => {
    const positions = new Float32Array([0, 0, 0, 0, 0, 1]);
    const out = new Uint32Array(2);
    const res = sortByDepth(positions, 2, [0, 0, 1, 0], out);
    expect(res).toBe(out);
  });
});
