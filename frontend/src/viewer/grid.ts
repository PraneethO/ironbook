/**
 * grid.ts — a ground grid + XYZ axes overlay for the splat viewer.
 *
 * Gives the floating scene a floor reference so it reads as "standing on the
 * ground" instead of drifting in space. Self-contained: owns its own GL
 * program/VAO/VBO and draws a set of GL_LINES on the scene's floor plane.
 * Premultiplied-alpha output to match the splat blend pipeline; drawn BEFORE
 * the splats so the model composites on top.
 */
import type { Mat4, Vec3 } from './math';

const GRID_VS = /* glsl */ `#version 300 es
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_color;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_color;
void main() {
  gl_Position = u_proj * u_view * vec4(a_pos, 1.0);
  v_color = a_color;
}`;

const GRID_FS = /* glsl */ `#version 300 es
precision mediump float;
in vec3 v_color;
uniform float u_alpha;
out vec4 fragColor;
void main() {
  fragColor = vec4(v_color * u_alpha, u_alpha); // premultiplied
}`;

const GRID_COLOR: Vec3 = [0.38, 0.42, 0.5];
const X_COLOR: Vec3 = [0.92, 0.3, 0.32];
const Y_COLOR: Vec3 = [0.4, 0.85, 0.45];
const Z_COLOR: Vec3 = [0.34, 0.55, 0.96];

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader | null {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.warn('grid shader compile failed:', gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

/** Build interleaved [x,y,z, r,g,b] line vertices for the floor grid + axes. */
function buildGeometry(min: Vec3, max: Vec3): { verts: Float32Array; gridCount: number; axisCount: number } {
  const cx = (min[0] + max[0]) / 2;
  const cz = (min[2] + max[2]) / 2;
  const floorY = min[1];
  const ext = Math.max(max[0] - min[0], max[2] - min[2], 1e-3);
  const half = ext * 0.9;
  const step = ext / 10; // ~20 divisions across

  const grid: number[] = [];
  const push = (arr: number[], x: number, y: number, z: number, c: Vec3) =>
    arr.push(x, y, z, c[0], c[1], c[2]);

  for (let d = -half; d <= half + 1e-6; d += step) {
    // lines parallel to X (vary z)
    push(grid, cx - half, floorY, cz + d, GRID_COLOR);
    push(grid, cx + half, floorY, cz + d, GRID_COLOR);
    // lines parallel to Z (vary x)
    push(grid, cx + d, floorY, cz - half, GRID_COLOR);
    push(grid, cx + d, floorY, cz + half, GRID_COLOR);
  }

  const axis: number[] = [];
  push(axis, cx - half, floorY, cz, X_COLOR); push(axis, cx + half, floorY, cz, X_COLOR); // X
  push(axis, cx, floorY, cz - half, Z_COLOR); push(axis, cx, floorY, cz + half, Z_COLOR); // Z
  push(axis, cx, floorY, cz, Y_COLOR); push(axis, cx, floorY + ext * 0.6, cz, Y_COLOR);   // Y (up)

  return {
    verts: new Float32Array([...grid, ...axis]),
    gridCount: grid.length / 6,
    axisCount: axis.length / 6,
  };
}

export class GridOverlay {
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private vbo: WebGLBuffer | null = null;
  private uView: WebGLUniformLocation | null = null;
  private uProj: WebGLUniformLocation | null = null;
  private uAlpha: WebGLUniformLocation | null = null;
  private gridCount = 0;
  private axisCount = 0;

  constructor(gl: WebGL2RenderingContext) {
    const v = compile(gl, gl.VERTEX_SHADER, GRID_VS);
    const f = compile(gl, gl.FRAGMENT_SHADER, GRID_FS);
    if (!v || !f) return;
    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, v);
    gl.attachShader(program, f);
    gl.linkProgram(program);
    gl.deleteShader(v);
    gl.deleteShader(f);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.warn('grid program link failed:', gl.getProgramInfoLog(program));
      return;
    }
    this.program = program;
    this.uView = gl.getUniformLocation(program, 'u_view');
    this.uProj = gl.getUniformLocation(program, 'u_proj');
    this.uAlpha = gl.getUniformLocation(program, 'u_alpha');
    this.vao = gl.createVertexArray();
    this.vbo = gl.createBuffer();
  }

  setBounds(gl: WebGL2RenderingContext, min: Vec3, max: Vec3): void {
    if (!this.program || !this.vao || !this.vbo) return;
    const { verts, gridCount, axisCount } = buildGeometry(min, max);
    this.gridCount = gridCount;
    this.axisCount = axisCount;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    gl.bindVertexArray(null);
  }

  draw(gl: WebGL2RenderingContext, view: Mat4, proj: Mat4): void {
    if (!this.program || !this.vao || this.gridCount + this.axisCount === 0) return;
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniformMatrix4fv(this.uView, false, view);
    gl.uniformMatrix4fv(this.uProj, false, proj);
    gl.uniform1f(this.uAlpha, 0.26);
    gl.drawArrays(gl.LINES, 0, this.gridCount);
    gl.uniform1f(this.uAlpha, 0.95);
    gl.drawArrays(gl.LINES, this.gridCount, this.axisCount);
    gl.bindVertexArray(null);
  }

  dispose(gl: WebGL2RenderingContext): void {
    if (this.program) gl.deleteProgram(this.program);
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.vbo) gl.deleteBuffer(this.vbo);
    this.program = null;
    this.vao = null;
    this.vbo = null;
  }
}
