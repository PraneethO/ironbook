/**
 * sort.ts — pure depth-sort logic shared by the main thread (synchronous
 * fallback) and the Web Worker. Splats must be drawn back-to-front for correct
 * alpha blending, so we sort indices by view-space depth (farthest first).
 *
 * To compute depth we only need the view matrix's third row (the part that
 * projects a world point onto the view -Z axis). We pass that row plus the
 * positions, keeping the function allocation-light and testable.
 */

/**
 * Sort splat indices back-to-front (farthest first) given positions and the
 * view matrix's depth row. `viewRow` = [m2, m6, m10, m14] from a column-major
 * view matrix; depth = -(m2*x + m6*y + m10*z + m14).
 *
 * Returns a Uint32Array of indices ordered farthest -> nearest.
 */
export function sortByDepth(
  positions: Float32Array,
  count: number,
  viewRow: [number, number, number, number],
  out?: Uint32Array,
): Uint32Array {
  const indices = out && out.length >= count ? out : new Uint32Array(count);
  const depths = new Float32Array(count);

  const [m2, m6, m10, m14] = viewRow;
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3 + 0];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    // farther points have more negative view-z; we store -z so larger = farther
    depths[i] = -(m2 * x + m6 * y + m10 * z + m14);
    indices[i] = i;
  }

  // Sort farthest first (descending depth) for back-to-front blending.
  const view = indices.subarray(0, count);
  // Array.prototype.sort on a typed array subarray view.
  Array.prototype.sort.call(view, (a: number, b: number) => depths[b] - depths[a]);

  return indices;
}

export interface SortRequest {
  positions: Float32Array;
  count: number;
  viewRow: [number, number, number, number];
}
