import { describe, expect, it } from 'vitest';
import {
  decodeQuatByte,
  encodeQuatByte,
  encodeSplat,
  parseSplat,
  SPLAT_STRIDE,
} from '../src/viewer/SplatLoader';

/**
 * Build a known 32-byte splat record exactly per CONTRACT.md §3 and assert the
 * loader decodes every field correctly (byte-for-byte against the encoding).
 */
function buildRecord(
  pos: [number, number, number],
  scale: [number, number, number],
  color: [number, number, number, number],
  quatBytes: [number, number, number, number], // raw bytes (w,x,y,z)
): ArrayBuffer {
  const buf = new ArrayBuffer(SPLAT_STRIDE);
  const dv = new DataView(buf);
  dv.setFloat32(0, pos[0], true);
  dv.setFloat32(4, pos[1], true);
  dv.setFloat32(8, pos[2], true);
  dv.setFloat32(12, scale[0], true);
  dv.setFloat32(16, scale[1], true);
  dv.setFloat32(20, scale[2], true);
  dv.setUint8(24, color[0]);
  dv.setUint8(25, color[1]);
  dv.setUint8(26, color[2]);
  dv.setUint8(27, color[3]);
  dv.setUint8(28, quatBytes[0]);
  dv.setUint8(29, quatBytes[1]);
  dv.setUint8(30, quatBytes[2]);
  dv.setUint8(31, quatBytes[3]);
  return buf;
}

describe('SplatLoader.parseSplat', () => {
  it('decodes a single known record byte-for-byte', () => {
    // quat bytes: 128 -> 0, 255 -> ~0.992, 0 -> -1, 192 -> 0.5
    const buf = buildRecord(
      [1.5, -2.25, 3.0],
      [0.1, 0.2, 0.3],
      [10, 20, 30, 255],
      [128, 255, 0, 192],
    );

    const parsed = parseSplat(buf);
    expect(parsed.count).toBe(1);

    expect(parsed.positions[0]).toBeCloseTo(1.5, 5);
    expect(parsed.positions[1]).toBeCloseTo(-2.25, 5);
    expect(parsed.positions[2]).toBeCloseTo(3.0, 5);

    expect(parsed.scales[0]).toBeCloseTo(0.1, 5);
    expect(parsed.scales[1]).toBeCloseTo(0.2, 5);
    expect(parsed.scales[2]).toBeCloseTo(0.3, 5);

    expect(Array.from(parsed.colors)).toEqual([10, 20, 30, 255]);

    // quat decode: q = b/128 - 1
    expect(parsed.quats[0]).toBeCloseTo(0, 5); // w from 128
    expect(parsed.quats[1]).toBeCloseTo(255 / 128 - 1, 5); // x from 255
    expect(parsed.quats[2]).toBeCloseTo(-1, 5); // y from 0
    expect(parsed.quats[3]).toBeCloseTo(0.5, 5); // z from 192
  });

  it('decodes multiple records and computes bounds over centers', () => {
    const a = buildRecord([-1, -2, -3], [1, 1, 1], [0, 0, 0, 255], [128, 128, 128, 128]);
    const b = buildRecord([4, 5, 6], [1, 1, 1], [255, 255, 255, 255], [128, 128, 128, 128]);
    const merged = new Uint8Array(SPLAT_STRIDE * 2);
    merged.set(new Uint8Array(a), 0);
    merged.set(new Uint8Array(b), SPLAT_STRIDE);

    const parsed = parseSplat(merged.buffer);
    expect(parsed.count).toBe(2);
    expect(parsed.bounds.min).toEqual([-1, -2, -3]);
    expect(parsed.bounds.max).toEqual([4, 5, 6]);
  });

  it('handles an empty buffer gracefully', () => {
    const parsed = parseSplat(new ArrayBuffer(0));
    expect(parsed.count).toBe(0);
    expect(parsed.positions.length).toBe(0);
    expect(parsed.bounds).toEqual({ min: [0, 0, 0], max: [0, 0, 0] });
  });

  it('ignores a trailing partial (truncated) record', () => {
    const full = buildRecord([1, 1, 1], [1, 1, 1], [1, 2, 3, 4], [128, 128, 128, 128]);
    const buf = new Uint8Array(SPLAT_STRIDE + 10); // 10 extra bytes
    buf.set(new Uint8Array(full), 0);
    const parsed = parseSplat(buf.buffer);
    expect(parsed.count).toBe(1);
  });

  it('round-trips through encodeSplat -> parseSplat', () => {
    const original = parseSplat(
      buildRecord([2, 3, 4], [0.5, 0.6, 0.7], [11, 22, 33, 200], [200, 100, 50, 150]),
    );
    const reencoded = encodeSplat({
      count: original.count,
      positions: original.positions,
      scales: original.scales,
      colors: original.colors,
      quats: original.quats,
    });
    const reparsed = parseSplat(reencoded);

    expect(Array.from(reparsed.positions)).toEqual(Array.from(original.positions));
    expect(Array.from(reparsed.colors)).toEqual(Array.from(original.colors));
    // quats survive the byte quantization within the encoder's resolution
    for (let i = 0; i < 4; i++) {
      expect(reparsed.quats[i]).toBeCloseTo(original.quats[i], 5);
    }
  });
});

describe('quat byte encode/decode', () => {
  it('decodeQuatByte inverts encodeQuatByte for representable values', () => {
    for (const q of [-1, -0.5, 0, 0.5, 0.992]) {
      const b = encodeQuatByte(q);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThanOrEqual(255);
      expect(decodeQuatByte(b)).toBeCloseTo(q, 2);
    }
  });

  it('clamps out-of-range quaternion components', () => {
    expect(encodeQuatByte(5)).toBe(255);
    expect(encodeQuatByte(-5)).toBe(0);
  });
});
