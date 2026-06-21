/**
 * sortWorker.ts — depth-sort Web Worker. Receives splat positions once
 * (transferred), then on each `sort` message computes a back-to-front index
 * order for the supplied view row and posts the sorted Uint32Array back
 * (transferred) for the main thread to upload.
 *
 * The actual sort uses the shared pure `sortByDepth` so logic is identical to
 * the synchronous fallback and is unit-testable.
 */

import { sortByDepth } from './sort';

interface InitMessage {
  type: 'init';
  positions: ArrayBuffer; // Float32Array buffer, length count*3
  count: number;
}

interface SortMessage {
  type: 'sort';
  viewRow: [number, number, number, number];
  generation: number;
}

type InMessage = InitMessage | SortMessage;

let positions: Float32Array | null = null;
let count = 0;

self.onmessage = (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    positions = new Float32Array(msg.positions);
    count = msg.count;
    return;
  }
  if (msg.type === 'sort') {
    if (!positions) return;
    const order = sortByDepth(positions, count, msg.viewRow);
    // Transfer the buffer back to avoid a copy.
    (self as unknown as Worker).postMessage(
      { type: 'sorted', order, generation: msg.generation },
      [order.buffer],
    );
  }
};
