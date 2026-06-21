/**
 * sort.ts — fast back-to-front depth ordering for correct alpha compositing,
 * ported from antimatter15/splat.
 *
 * A 16-bit single-pass counting sort (O(n)) replaces the old O(n·log n)
 * comparison sort, whose per-comparison JS callback dominated frame time and
 * made large scenes stutter. Depth is the view-projected Z of each splat
 * center, so we only need row 2 of the column-major viewProj matrix
 * (indices 2, 6, 10). The resulting index order is the draw order the renderer
 * (premultiplied-alpha blend) expects.
 *
 * Shared by the Web Worker and the synchronous fallback so the logic is
 * identical and unit-testable.
 */

const BUCKETS = 256 * 256;

/**
 * Return splat indices ordered by view-projected depth (the antimatter draw
 * order). `viewProj` is the flattened column-major projection·view matrix;
 * only indices [2], [6], [10] are read. Pass `out` to reuse a buffer.
 */
export function sortByDepth(
  positions: Float32Array,
  count: number,
  viewProj: ArrayLike<number>,
  out?: Uint32Array,
): Uint32Array {
  const depthIndex = out && out.length >= count ? out : new Uint32Array(count);
  if (count === 0) return depthIndex;

  const vp2 = viewProj[2];
  const vp6 = viewProj[6];
  const vp10 = viewProj[10];

  // Quantize each depth to an integer and track the range.
  const sizeList = new Int32Array(count);
  let maxDepth = -Infinity;
  let minDepth = Infinity;
  for (let i = 0; i < count; i++) {
    const depth =
      ((vp2 * positions[i * 3 + 0] +
        vp6 * positions[i * 3 + 1] +
        vp10 * positions[i * 3 + 2]) *
        4096) |
      0;
    sizeList[i] = depth;
    if (depth > maxDepth) maxDepth = depth;
    if (depth < minDepth) minDepth = depth;
  }

  // 16-bit single-pass counting sort over the quantized depth range.
  const depthInv = (BUCKETS - 1) / (maxDepth - minDepth || 1);
  const counts = new Uint32Array(BUCKETS);
  for (let i = 0; i < count; i++) {
    const bucket = ((sizeList[i] - minDepth) * depthInv) | 0;
    sizeList[i] = bucket;
    counts[bucket]++;
  }
  const starts = new Uint32Array(BUCKETS);
  for (let i = 1; i < BUCKETS; i++) starts[i] = starts[i - 1] + counts[i - 1];
  for (let i = 0; i < count; i++) depthIndex[starts[sizeList[i]]++] = i;

  return depthIndex;
}
