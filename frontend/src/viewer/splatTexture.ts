/**
 * splatTexture.ts — pack a `.splat` buffer into the RGBA32UI data texture the
 * renderer samples, ported from antimatter15/splat.
 *
 * Each splat occupies 2 texels (8 uint32) in a fixed 2048-wide texture:
 *   texel 0 (uint32 8i+0..3): position.x, position.y, position.z (float bits),
 *                              and RGBA color packed into the 4th uint32.
 *   texel 1 (uint32 8i+4..7): the 6 unique entries of the 3D covariance
 *                              (Σ = M·Mᵀ, M = R·S), scaled ×4 and packed as
 *                              half-floats (packHalf2x16), 4th uint32 unused.
 *
 * The shader reads splat `index` at ivec2((index & 0x3ff) << 1, index >> 10),
 * so the width MUST stay 2048 (1024 splats per row × 2 texels).
 *
 * Pure and framework-agnostic so it runs in the worker and is unit-testable.
 */

export const TEX_WIDTH = 1024 * 2;

const _floatView = new Float32Array(1);
const _int32View = new Int32Array(_floatView.buffer);

/** IEEE-754 float32 → float16 bit pattern (antimatter15's implementation). */
export function floatToHalf(float: number): number {
  _floatView[0] = float;
  const f = _int32View[0];

  const sign = (f >> 31) & 0x0001;
  const exp = (f >> 23) & 0x00ff;
  let frac = f & 0x007fffff;

  let newExp;
  if (exp === 0) {
    newExp = 0;
  } else if (exp < 113) {
    newExp = 0;
    frac |= 0x00800000;
    frac = frac >> (113 - exp);
    if (frac & 0x01000000) {
      newExp = 1;
      frac = 0;
    }
  } else if (exp < 142) {
    newExp = exp - 112;
  } else {
    newExp = 31;
    frac = 0;
  }

  return (sign << 15) | (newExp << 10) | (frac >> 13);
}

/** Pack two floats into one uint32 as two half-floats (x in low 16 bits). */
export function packHalf2x16(x: number, y: number): number {
  return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
}

export interface SplatTexture {
  texdata: Uint32Array;
  texwidth: number;
  texheight: number;
}

/**
 * Build the RGBA32UI texture payload for `vertexCount` splats from a raw
 * `.splat` buffer (32 bytes/splat: 3×f32 pos, 3×f32 scale, 4×u8 color,
 * 4×u8 quaternion in (w,x,y,z) order encoded as (q+1)*128).
 */
export function generateSplatTexture(
  buffer: ArrayBuffer,
  vertexCount: number,
): SplatTexture {
  const f_buffer = new Float32Array(buffer);
  const u_buffer = new Uint8Array(buffer);

  const texwidth = TEX_WIDTH;
  const texheight = Math.max(1, Math.ceil((2 * vertexCount) / texwidth));
  const texdata = new Uint32Array(texwidth * texheight * 4);
  const texdata_c = new Uint8Array(texdata.buffer);
  const texdata_f = new Float32Array(texdata.buffer);

  for (let i = 0; i < vertexCount; i++) {
    // position (float bits)
    texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
    texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
    texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

    // color rgba (uint8) packed into the 4th uint32 of texel 0
    texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
    texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
    texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
    texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

    const scale = [
      f_buffer[8 * i + 3 + 0],
      f_buffer[8 * i + 3 + 1],
      f_buffer[8 * i + 3 + 2],
    ];
    const rot = [
      (u_buffer[32 * i + 28 + 0] - 128) / 128,
      (u_buffer[32 * i + 28 + 1] - 128) / 128,
      (u_buffer[32 * i + 28 + 2] - 128) / 128,
      (u_buffer[32 * i + 28 + 3] - 128) / 128,
    ];

    // M = R · S  (rotation matrix from quaternion, columns scaled)
    const M = [
      1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
      2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
      2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

      2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
      1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
      2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

      2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
      2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
      1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
    ].map((k, idx) => k * scale[Math.floor(idx / 3)]);

    // Σ = M · Mᵀ, upper triangle (6 unique entries)
    const sigma = [
      M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
      M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
      M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
      M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
      M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
      M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
    ];

    texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
    texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
    texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
  }

  return { texdata, texwidth, texheight };
}
