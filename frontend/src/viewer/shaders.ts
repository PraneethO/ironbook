/**
 * shaders.ts — GLSL ES 3.00 source for the projected-quad Gaussian splat
 * renderer (antimatter15 / standard approach).
 *
 * Splat attributes live in float textures (u_data). Each drawn instance reads
 * a splat index from a per-instance integer attribute (`a_index`) that is
 * populated, in back-to-front order, from the depth-sort worker. This lets us
 * change *draw order* every frame (for correct alpha blending) without
 * re-uploading the (large, static) splat data.
 *
 * The vertex shader projects the splat center to clip space, builds the 2D
 * screen-space covariance by projecting the 3D covariance (scale + rotation)
 * through the Jacobian of the perspective projection, then offsets the quad
 * corners along the covariance principal axes. The fragment shader evaluates
 * the Gaussian falloff times the splat color/alpha.
 */

export const VERTEX_SHADER = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

// Per-vertex quad corner in sigma units (e.g. [-2, 2]).
layout(location = 0) in vec2 a_corner;
// Per-instance splat index (into the data texture), back-to-front sorted.
layout(location = 1) in uint a_index;

uniform sampler2D u_data;  // RGBA32F, packed splat records
uniform int u_texWidth;    // texels per row
uniform int u_recordTexels; // texels per splat record

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport;
uniform float u_splatScale;

out vec4 v_color;
out vec2 v_offset;

vec4 fetch(int recordTexel) {
  int linear = int(a_index) * u_recordTexels + recordTexel;
  int x = linear % u_texWidth;
  int y = linear / u_texWidth;
  return texelFetch(u_data, ivec2(x, y), 0);
}

mat3 quatToMat3(vec4 q) {
  float w = q.x, x = q.y, y = q.z, z = q.w;
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2, zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;
  return mat3(
    1.0 - (yy + zz), xy + wz,         xz - wy,
    xy - wz,         1.0 - (xx + zz), yz + wx,
    xz + wy,         yz - wx,         1.0 - (xx + yy)
  );
}

void main() {
  // record layout (3 texels): [pos.xyz, scale.x], [scale.yz, quat.wx], [quat.yz, color.rg(packed?)]
  // We use 4 texels for clarity: pos+sx, scale.yz+_, color rgba, quat wxyz.
  vec4 t0 = fetch(0); // pos.xyz, scale.x
  vec4 t1 = fetch(1); // scale.y, scale.z, _, _
  vec4 t2 = fetch(2); // color rgba (0..1)
  vec4 t3 = fetch(3); // quat w,x,y,z

  vec3 center = t0.xyz;
  vec3 s = max(vec3(t0.w, t1.x, t1.y), vec3(1e-6)) * u_splatScale;
  vec4 color = t2;
  vec4 quat = normalize(t3);

  vec4 viewPos = u_view * vec4(center, 1.0);
  if (viewPos.z > 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_color = vec4(0.0);
    v_offset = vec2(0.0);
    return;
  }

  mat3 R = quatToMat3(quat);
  mat3 S = mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 M = R * S;
  mat3 Sigma = M * transpose(M);

  float fx = u_proj[0][0] * u_viewport.x * 0.5;
  float fy = u_proj[1][1] * u_viewport.y * 0.5;
  float z = viewPos.z;
  float zinv = 1.0 / z;
  float zinv2 = zinv * zinv;
  mat3 J = mat3(
    fx * zinv, 0.0,       0.0,
    0.0,       fy * zinv, 0.0,
    -fx * viewPos.x * zinv2, -fy * viewPos.y * zinv2, 0.0
  );
  mat3 W = mat3(u_view);
  mat3 T = J * W;
  mat3 cov = T * Sigma * transpose(T);

  float a = cov[0][0] + 0.3;
  float b = cov[0][1];
  float c = cov[1][1] + 0.3;

  float trace = a + c;
  float det = a * c - b * b;
  float mid = 0.5 * trace;
  float disc = sqrt(max(mid * mid - det, 0.0));
  float l1 = mid + disc;
  float l2 = max(mid - disc, 0.0);

  vec2 e1 = (abs(b) < 1e-6)
    ? ((a >= c) ? vec2(1.0, 0.0) : vec2(0.0, 1.0))
    : normalize(vec2(b, l1 - a));
  vec2 e2 = vec2(-e1.y, e1.x);

  float r1 = sqrt(l1);
  float r2 = sqrt(l2);

  vec4 clip = u_proj * viewPos;
  vec2 ndc = clip.xy / clip.w;
  vec2 px = a_corner.x * r1 * e1 + a_corner.y * r2 * e2;
  vec2 ndcOffset = 2.0 * px / u_viewport;

  gl_Position = vec4(ndc + ndcOffset, clip.z / clip.w, 1.0);
  v_color = color;
  v_offset = a_corner;
}
`;

export const FRAGMENT_SHADER = /* glsl */ `#version 300 es
precision highp float;

in vec4 v_color;
in vec2 v_offset;
out vec4 fragColor;

void main() {
  float r2 = dot(v_offset, v_offset);
  float alpha = exp(-0.5 * r2) * v_color.a;
  if (alpha < 0.004) discard;
  fragColor = vec4(v_color.rgb, alpha);
}
`;

/** Texels per splat record in the data texture (RGBA32F). */
export const RECORD_TEXELS = 4;
