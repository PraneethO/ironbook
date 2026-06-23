/**
 * sortWorker.ts — off-main-thread splat preparation + depth sorting, following
 * antimatter15/splat's worker design.
 *
 * On `init` it receives the raw `.splat` buffer (transferred), builds the
 * RGBA32UI data texture once, and posts it back. On each `sort` it runs the
 * O(n) counting sort for the supplied viewProj matrix and posts the depth
 * order back (transferred). The texture-gen and sort logic are shared with the
 * synchronous fallback via splatTexture.ts / sort.ts.
 */

import { generateSplatTexture } from './splatTexture';
import { sortByDepth } from './sort';

interface InitMessage {
  type: 'init';
  buffer: ArrayBuffer; // raw .splat bytes, count*32
  count: number;
}

interface SortMessage {
  type: 'sort';
  viewProj: Float32Array; // flattened column-major projection·view
  generation: number;
}

type InMessage = InitMessage | SortMessage;

// Contiguous positions (count*3) kept for fast re-sorting on camera moves.
let positions: Float32Array | null = null;
let count = 0;

self.onmessage = (e: MessageEvent<InMessage>) => {
  const msg = e.data;

  if (msg.type === 'init') {
    count = msg.count;
    const f = new Float32Array(msg.buffer); // stride 8 floats per splat
    positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3 + 0] = f[8 * i + 0];
      positions[i * 3 + 1] = f[8 * i + 1];
      positions[i * 3 + 2] = f[8 * i + 2];
    }

    const tex = generateSplatTexture(msg.buffer, count);
    (self as unknown as Worker).postMessage(
      {
        type: 'texture',
        texdata: tex.texdata,
        texwidth: tex.texwidth,
        texheight: tex.texheight,
      },
      [tex.texdata.buffer],
    );
    return;
  }

  if (msg.type === 'sort') {
    if (!positions || count === 0) return;
    const depthIndex = sortByDepth(positions, count, msg.viewProj);
    (self as unknown as Worker).postMessage(
      { type: 'sorted', depthIndex, generation: msg.generation },
      [depthIndex.buffer],
    );
  }
};
