/**
 * 3D Viewer (screen 5). Full-screen canvas wrapping the engine via
 * SplatViewerReact. Mode switcher (Orbit / Walk / Fly / Screenshot), an
 * on-screen controls-help overlay, and an FPS + splat-count readout. Loads
 * /api/projects/{id}/asset. Also serves the public /view/:id shared route.
 */
import { useCallback, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { apiClient } from '../api/client';
import { SplatViewerReact, type SplatViewerHandle } from '../components/SplatViewerReact';
import type { CameraMode } from '../viewer/SplatViewer';

const MODES: { key: CameraMode; label: string }[] = [
  { key: 'orbit', label: 'Orbit' },
  { key: 'walk', label: 'Walk' },
  { key: 'fly', label: 'Fly' },
];

export function ViewerPage({ shared = false }: { shared?: boolean }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const viewerRef = useRef<SplatViewerHandle | null>(null);

  const [mode, setMode] = useState<CameraMode>('orbit');
  const [fps, setFps] = useState(0);
  const [splatCount, setSplatCount] = useState(0);
  const [progress, setProgress] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const src = useMemo(() => (id ? apiClient.assetUrl(id) : undefined), [id]);

  const onFps = useCallback((f: number) => setFps(f), []);
  const onProgress = useCallback((l: number, t: number) => {
    setProgress(t > 0 ? l / t : 0);
  }, []);
  const onLoaded = useCallback((count: number) => {
    setSplatCount(count);
    setLoaded(true);
  }, []);
  const onError = useCallback((m: string) => setError(m), []);

  const onScreenshot = () => {
    const data = viewerRef.current?.capture();
    if (!data) return;
    const a = document.createElement('a');
    a.href = data;
    a.download = `world-${id ?? 'scene'}.png`;
    a.click();
  };

  return (
    <div className="viewer-root">
      <div className="viewer-canvas-wrap">
        <SplatViewerReact
          ref={viewerRef}
          src={src}
          mode={mode}
          onFps={onFps}
          onProgress={onProgress}
          onLoaded={onLoaded}
          onError={onError}
        />
      </div>

      {!loaded && !error && (
        <div className="viewer-loading">
          <div className="spinner" />
          <div>Loading your world… {progress > 0 ? `${Math.round(progress * 100)}%` : ''}</div>
        </div>
      )}

      {error && (
        <div className="viewer-error">
          <div style={{ fontSize: 32 }}>😕</div>
          <div>{error}</div>
          <Link className="btn" to="/">Back to my worlds</Link>
        </div>
      )}

      {/* Top-left: navigation / title */}
      <div className="viewer-overlay top-left">
        <div className="glass">
          {shared ? (
            <Link to="/" className="brand" style={{ fontSize: 15 }}>
              <span className="logo" aria-hidden /> Gaussian Splat World
            </Link>
          ) : (
            <button className="btn btn-ghost" onClick={() => navigate('/')}>
              ← My worlds
            </button>
          )}
        </div>
      </div>

      {/* Top-right: actions */}
      {!shared && id && (
        <div className="viewer-overlay top-right">
          <div className="glass row-actions">
            <button className="btn" onClick={() => navigate(`/projects/${id}/settings`)}>
              Settings
            </button>
            <button className="btn" onClick={() => navigate(`/projects/${id}/share`)}>
              Share / Export
            </button>
          </div>
        </div>
      )}

      {/* Bottom-left: mode switch + screenshot */}
      <div className="viewer-overlay bottom-left">
        <div className="glass">
          <div className="mode-switch">
            {MODES.map((m) => (
              <button
                key={m.key}
                className={mode === m.key ? 'active' : ''}
                onClick={() => setMode(m.key)}
                data-testid={`mode-${m.key}`}
              >
                {m.label}
              </button>
            ))}
            <button onClick={onScreenshot} data-testid="mode-screenshot" title="Save screenshot">
              📸 Screenshot
            </button>
          </div>
        </div>
      </div>

      {/* Bottom-right: controls help + readout */}
      <div className="viewer-overlay bottom-right">
        <div className="glass" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <ul className="help-list">
            <li><kbd>Drag</kbd> look / orbit</li>
            <li><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</li>
            <li><kbd>Scroll</kbd> zoom</li>
            <li><kbd>Shift</kbd> move faster</li>
            <li><kbd>Space</kbd>/<kbd>C</kbd> up / down (fly)</li>
          </ul>
          <div className="readout">
            <span><b>{fps}</b> fps</span>
            <span><b>{splatCount.toLocaleString()}</b> splats</span>
          </div>
        </div>
      </div>
    </div>
  );
}
