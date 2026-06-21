import { describe, it, expect } from 'vitest';
import {
  generateSplatTexture,
  TEX_WIDTH,
  packHalf2x16,
  floatToHalf,
} from '../src/viewer/splatTexture';
import { encodeSplat } from '../src/viewer/SplatLoader';

describe('generateSplatTexture', () => {
  it('packs position floats and color bytes at the antimatter offsets', () => {
    const buf = encodeSplat({
      count: 1,
      positions: [1, 2, 3],
      scales: [0.1, 0.2, 0.3],
      colors: [10, 20, 30, 200],
      quats: [1, 0, 0, 0], // (w, x, y, z) ~identity
    });
    const { texdata, texwidth, texheight } = generateSplatTexture(buf, 1);

    expect(texwidth).toBe(TEX_WIDTH);
    expect(texheight).toBeGreaterThanOrEqual(1);

    // texel 0: position as float bits in the first 3 uint32 lanes
    const f = new Float32Array(texdata.buffer);
    expect(f[0]).toBeCloseTo(1, 5);
    expect(f[1]).toBeCloseTo(2, 5);
    expect(f[2]).toBeCloseTo(3, 5);

    // color packed into the 4th uint32 lane (uint32 index 7 => byte offset 28)
    const c = new Uint8Array(texdata.buffer);
    expect(c[7 * 4 + 0]).toBe(10);
    expect(c[7 * 4 + 1]).toBe(20);
    expect(c[7 * 4 + 2]).toBe(30);
    expect(c[7 * 4 + 3]).toBe(200);

    // texel 1: covariance half-floats are populated for a non-degenerate splat
    expect((texdata[4] | texdata[5] | texdata[6]) >>> 0).not.toBe(0);
  });

  it('packHalf2x16 stores x in the low lane and y in the high lane', () => {
    const packed = packHalf2x16(1, 2);
    expect(packed & 0xffff).toBe(floatToHalf(1));
    expect((packed >>> 16) & 0xffff).toBe(floatToHalf(2));
  });
});
