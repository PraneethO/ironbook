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
import {
  lookAt,
  Mat4,
  perspective,
  projectionCV,
  Vec3,
  viewDepth,
  viewMatrixCV,
} from '../src/viewer/math';
import { sortByDepth } from '../src/viewer/sort';

/** Column-major mat4 * vec4(p, 1) -> [x, y, z, w] (no perspective divide). */
function mulMatVec(m: Mat4, p: Vec3): [number, number, number, number] {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15],
  ];
}

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

describe('viewMatrixCV / projectionCV (antimatter OpenCV convention)', () => {
  // Camera at +5 on Z looking at the origin, world-up = +Y.
  const eye: Vec3 = [0, 0, 5];
  const target: Vec3 = [0, 0, 0];
  const up: Vec3 = [0, 1, 0];
  const view = viewMatrixCV(eye, target, up);
  const proj = projectionCV((60 * Math.PI) / 180, 800, 600, 0.1, 1000);

  it('puts points in front of the camera at positive view-z (cam.z > 0)', () => {
    const cam = mulMatVec(view, [0, 0, 0]); // scene center, in front
    expect(cam[2]).toBeGreaterThan(0);
    const behind = mulMatVec(view, [0, 0, 10]); // behind the camera
    expect(behind[2]).toBeLessThan(0);
  });

  it('projects in-front points with positive clip.w', () => {
    const cam = mulMatVec(view, [0, 0, 0]);
    const clip = mulMatVec(proj, [cam[0], cam[1], cam[2]]);
    expect(clip[3]).toBeGreaterThan(0);
  });

  it('maps world-up to the top and world-right to the right of the screen', () => {
    const project = (p: Vec3) => {
      const cam = mulMatVec(view, p);
      const clip = mulMatVec(proj, [cam[0], cam[1], cam[2]]);
      return { x: clip[0] / clip[3], y: clip[1] / clip[3] }; // NDC
    };
    expect(project([0, 1, 0]).y).toBeGreaterThan(0); // above center -> top
    expect(project([0, -1, 0]).y).toBeLessThan(0); // below center -> bottom
    expect(project([1, 0, 0]).x).toBeGreaterThan(0); // right -> +x
    expect(project([-1, 0, 0]).x).toBeLessThan(0); // left -> -x
  });
});

describe('sortByDepth (counting sort)', () => {
  it('orders indices by view-projected depth (viewProj row 2)', () => {
    // depth = viewProj[2]*x + viewProj[6]*y + viewProj[10]*z; here depth = z.
    const positions = new Float32Array([
      0, 0, 0, // idx 0: depth 0
      0, 0, -5, // idx 1: depth -5 (smallest)
      0, 0, 4, // idx 2: depth 4 (largest)
    ]);
    const viewProj = new Float32Array(16);
    viewProj[10] = 1; // only the z term contributes
    const order = sortByDepth(positions, 3, viewProj);
    // counting sort is ascending by quantized depth: -5, 0, 4
    expect(Array.from(order)).toEqual([1, 0, 2]);
  });

  it('reuses the provided output buffer when large enough', () => {
    const positions = new Float32Array([0, 0, 0, 0, 0, 1]);
    const viewProj = new Float32Array(16);
    viewProj[10] = 1;
    const out = new Uint32Array(2);
    const res = sortByDepth(positions, 2, viewProj, out);
    expect(res).toBe(out);
  });

  it('handles an empty splat set', () => {
    const order = sortByDepth(new Float32Array(0), 0, new Float32Array(16));
    expect(order.length).toBe(0);
  });
});
