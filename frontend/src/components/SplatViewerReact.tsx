/**
 * SplatViewerReact — React wrapper around the framework-agnostic SplatViewer
 * engine (CONTRACT.md §4). Mounts a canvas, instantiates the viewer once, and
 * keeps mode / src in sync. Exposes the underlying viewer via a ref so parent
 * screens can call capture(), resetCamera(), setSplatScale(), etc.
 */

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import { SplatViewer, type CameraMode } from '../viewer/SplatViewer';

export interface SplatViewerHandle {
  viewer: SplatViewer | null;
  capture: () => string;
  resetCamera: () => void;
  setSplatScale: (s: number) => void;
  setBackgroundColor: (r: number, g: number, b: number) => void;
  setGridVisible: (visible: boolean) => void;
}

export interface SplatViewerReactProps {
  src?: string;
  buffer?: ArrayBuffer;
  mode?: CameraMode;
  onFps?: (fps: number) => void;
  onProgress?: (loaded: number, total: number) => void;
  onLoaded?: (splatCount: number) => void;
  onError?: (message: string) => void;
  className?: string;
}

export const SplatViewerReact = forwardRef<SplatViewerHandle, SplatViewerReactProps>(
  function SplatViewerReact(props, ref) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const viewerRef = useRef<SplatViewer | null>(null);
    const [, force] = useState(0);

    useImperativeHandle(ref, () => ({
      get viewer() {
        return viewerRef.current;
      },
      capture: () => viewerRef.current?.capture() ?? '',
      resetCamera: () => viewerRef.current?.resetCamera(),
      setSplatScale: (s: number) => viewerRef.current?.setSplatScale(s),
      setBackgroundColor: (r: number, g: number, b: number) =>
        viewerRef.current?.setBackgroundColor(r, g, b),
      setGridVisible: (visible: boolean) => viewerRef.current?.setGridVisible(visible),
    }));

    // Create the engine once.
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const viewer = new SplatViewer({
        canvas,
        mode: props.mode ?? 'orbit',
        onFps: props.onFps,
        onProgress: props.onProgress,
      });
      viewerRef.current = viewer;
      force((n) => n + 1);
      return () => {
        viewer.dispose();
        viewerRef.current = null;
      };
      // Intentionally create once; callbacks are read from latest closure via refs.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Load source (url or buffer) whenever it changes.
    useEffect(() => {
      const viewer = viewerRef.current;
      if (!viewer) return;
      let cancelled = false;
      const run = async () => {
        try {
          if (props.buffer) {
            await viewer.loadBuffer(props.buffer);
          } else if (props.src) {
            await viewer.load(props.src);
          } else {
            return;
          }
          if (!cancelled) props.onLoaded?.(viewer.splatCount);
        } catch (e) {
          if (!cancelled) {
            props.onError?.(
              e instanceof Error ? e.message : 'Could not load the 3D world.',
            );
          }
        }
      };
      void run();
      return () => {
        cancelled = true;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [props.src, props.buffer, viewerRef.current]);

    // Keep mode in sync.
    useEffect(() => {
      if (props.mode) viewerRef.current?.setMode(props.mode);
    }, [props.mode]);

    return (
      <canvas
        ref={canvasRef}
        className={props.className}
        style={{ width: '100%', height: '100%', display: 'block', touchAction: 'none' }}
      />
    );
  },
);
