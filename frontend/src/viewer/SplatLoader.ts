/**
 * SplatLoader — pure, framework-agnostic parser for the `.splat` binary asset
 * format defined in CONTRACT.md §3.
 *
 * Layout: 32 bytes per splat, little-endian, no header. splat_count = bytes / 32.
 *   offset 0  : float32 x3  position (x, y, z)
 *   offset 12 : float32 x3  scale (x, y, z) world units
 *   offset 24 : uint8 x4    color (r, g, b, a) 0..255
 *   offset 28 : uint8 x4    rotation quaternion, each byte = round((q+1)*128)
 *               clamped 0..255, stored in order (w, x, y, z)
 *
 * The parse is a pure function with no WebGL dependency so it is fully
 * unit-testable. Decoded data is returned as flat typed arrays ready for
 * upload as GPU instance attributes.
 */

export const SPLAT_STRIDE = 32;

export interface ParsedSplats {
  /** Number of splats parsed. */
  count: number;
  /** count * 3 float32 — positions (x, y, z). */
  positions: Float32Array;
  /** count * 3 float32 — scales (x, y, z), world units. */
  scales: Float32Array;
  /** count * 4 uint8 — colors (r, g, b, a), 0..255. */
  colors: Uint8Array;
  /** count * 4 float32 — quaternions decoded to floats, order (w, x, y, z). */
  quats: Float32Array;
  /** Axis-aligned bounds of all splat centers. */
  bounds: { min: [number, number, number]; max: [number, number, number] };
}

/** Decode one stored quaternion byte back to its float value in [-1, 1]. */
export function decodeQuatByte(b: number): number {
  // Encoding: round((q + 1) * 128) clamped 0..255  =>  q = b / 128 - 1
  return b / 128 - 1;
}

/** Encode a quaternion component to a byte, matching the contract encoding. */
export function encodeQuatByte(q: number): number {
  return Math.max(0, Math.min(255, Math.round((q + 1) * 128)));
}

/**
 * Parse a `.splat` buffer into typed arrays. Robust to empty or
 * short/truncated buffers: any trailing partial splat is ignored.
 */
export function parseSplat(buffer: ArrayBuffer | ArrayBufferView): ParsedSplats {
  const bytes =
    buffer instanceof ArrayBuffer
      ? new Uint8Array(buffer)
      : new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const total = bytes.byteLength;
  const count = Math.floor(total / SPLAT_STRIDE);

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 4);
  const quats = new Float32Array(count * 4);

  if (count === 0) {
    return {
      count: 0,
      positions,
      scales,
      colors,
      quats,
      bounds: { min: [0, 0, 0], max: [0, 0, 0] },
    };
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, count * SPLAT_STRIDE);

  for (let i = 0; i < count; i++) {
    const base = i * SPLAT_STRIDE;

    // position (3 x float32 LE)
    const px = view.getFloat32(base + 0, true);
    const py = view.getFloat32(base + 4, true);
    const pz = view.getFloat32(base + 8, true);
    positions[i * 3 + 0] = px;
    positions[i * 3 + 1] = py;
    positions[i * 3 + 2] = pz;

    // scale (3 x float32 LE)
    scales[i * 3 + 0] = view.getFloat32(base + 12, true);
    scales[i * 3 + 1] = view.getFloat32(base + 16, true);
    scales[i * 3 + 2] = view.getFloat32(base + 20, true);

    // color (4 x uint8)
    colors[i * 4 + 0] = view.getUint8(base + 24);
    colors[i * 4 + 1] = view.getUint8(base + 25);
    colors[i * 4 + 2] = view.getUint8(base + 26);
    colors[i * 4 + 3] = view.getUint8(base + 27);

    // rotation quaternion (4 x uint8 -> float), stored order (w, x, y, z)
    quats[i * 4 + 0] = decodeQuatByte(view.getUint8(base + 28)); // w
    quats[i * 4 + 1] = decodeQuatByte(view.getUint8(base + 29)); // x
    quats[i * 4 + 2] = decodeQuatByte(view.getUint8(base + 30)); // y
    quats[i * 4 + 3] = decodeQuatByte(view.getUint8(base + 31)); // z
  }

  return { count, positions, scales, colors, quats, bounds: robustBounds(positions, count) };
}

/**
 * Robust scene bounds for camera framing. A few far floaters (reconstruction
 * noise) can sit hundreds of units from the real content; a raw min/max box
 * would then balloon and the camera would back off so far the scene renders as
 * a speck. Instead frame to the dense core: median center + a high-percentile
 * radius (sampled for speed). Floaters beyond stay in the scene as faint specks
 * but no longer dictate the camera distance.
 */
function robustBounds(
  positions: Float32Array,
  count: number,
): { min: [number, number, number]; max: [number, number, number] } {
  if (count < 100) {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < count; i++) {
      const px = positions[i * 3 + 0];
      const py = positions[i * 3 + 1];
      const pz = positions[i * 3 + 2];
      if (px < min[0]) min[0] = px;
      if (py < min[1]) min[1] = py;
      if (pz < min[2]) min[2] = pz;
      if (px > max[0]) max[0] = px;
      if (py > max[1]) max[1] = py;
      if (pz > max[2]) max[2] = pz;
    }
    return { min, max };
  }

  const target = Math.min(count, 40000);
  const stride = Math.max(1, Math.floor(count / target));
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  for (let i = 0; i < count; i += stride) {
    xs.push(positions[i * 3 + 0]);
    ys.push(positions[i * 3 + 1]);
    zs.push(positions[i * 3 + 2]);
  }
  const median = (arr: number[]): number => {
    const s = arr.slice().sort((a, b) => a - b);
    return s[s.length >> 1];
  };
  const cx = median(xs);
  const cy = median(ys);
  const cz = median(zs);
  const dists: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - cx;
    const dy = ys[i] - cy;
    const dz = zs[i] - cz;
    dists.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  dists.sort((a, b) => a - b);
  // 98th percentile distance: excludes the floater tail, keeps the real scene.
  const r = Math.max(0.5, dists[Math.floor(dists.length * 0.98)] ?? 1);
  return {
    min: [cx - r, cy - r, cz - r],
    max: [cx + r, cy + r, cz + r],
  };
}

/**
 * Encode parsed splat fields back into a `.splat` buffer (used by tests for
 * round-tripping and by potential export tooling). Inputs must be the same
 * flat-array layout produced by {@link parseSplat}.
 */
export function encodeSplat(opts: {
  count: number;
  positions: ArrayLike<number>;
  scales: ArrayLike<number>;
  colors: ArrayLike<number>;
  quats: ArrayLike<number>;
}): ArrayBuffer {
  const { count, positions, scales, colors, quats } = opts;
  const buffer = new ArrayBuffer(count * SPLAT_STRIDE);
  const view = new DataView(buffer);

  for (let i = 0; i < count; i++) {
    const base = i * SPLAT_STRIDE;
    view.setFloat32(base + 0, positions[i * 3 + 0], true);
    view.setFloat32(base + 4, positions[i * 3 + 1], true);
    view.setFloat32(base + 8, positions[i * 3 + 2], true);
    view.setFloat32(base + 12, scales[i * 3 + 0], true);
    view.setFloat32(base + 16, scales[i * 3 + 1], true);
    view.setFloat32(base + 20, scales[i * 3 + 2], true);
    view.setUint8(base + 24, colors[i * 4 + 0]);
    view.setUint8(base + 25, colors[i * 4 + 1]);
    view.setUint8(base + 26, colors[i * 4 + 2]);
    view.setUint8(base + 27, colors[i * 4 + 3]);
    view.setUint8(base + 28, encodeQuatByte(quats[i * 4 + 0]));
    view.setUint8(base + 29, encodeQuatByte(quats[i * 4 + 1]));
    view.setUint8(base + 30, encodeQuatByte(quats[i * 4 + 2]));
    view.setUint8(base + 31, encodeQuatByte(quats[i * 4 + 3]));
  }

  return buffer;
}
