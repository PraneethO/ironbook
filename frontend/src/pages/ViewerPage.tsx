/**
 * 3D Viewer (screen 5). Full-screen canvas wrapping the engine via
 * SplatViewerReact, with a left worlds/annotations sidebar, a right-side
 * navigation-agent panel, and a bottom status/tool bar. Loads
 * /api/projects/{id}/asset. Also serves the public /view/:id shared route.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { apiClient } from '../api/client';
import { AgentChat } from '../components/AgentChat';
import { SplatViewerReact, type SplatViewerHandle } from '../components/SplatViewerReact';
import type { Project } from '../api/types';
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
  const [gridOn, setGridOn] = useState(true);
  const [worlds, setWorlds] = useState<Project[] | null>(null);
  const [panelsOpen, setPanelsOpen] = useState(true);

  useEffect(() => {
    if (shared) return;
    apiClient.listProjects().then(setWorlds).catch(() => setWorlds([]));
  }, [shared]);

  const onToggleGrid = () => {
    setGridOn((on) => {
      const next = !on;
      viewerRef.current?.setGridVisible(next);
      return next;
    });
  };

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
          <div style={{ color: 'var(--danger)' }}>Error</div>
          <div>{error}</div>
          <Link className="btn" to="/">Back to my worlds</Link>
        </div>
      )}

      {/* Top-left: panel toggle + navigation */}
      <div className="viewer-overlay top-left">
        <div className="glass row-actions" style={{ gap: 2 }}>
          {!shared && (
            <button
              className="btn btn-ghost"
              onClick={() => setPanelsOpen((o) => !o)}
              title="Toggle panels"
              data-testid="toggle-panels"
            >
              ☰ Files
            </button>
          )}
          {shared ? (
            <Link to="/" className="brand">
              <span className="logo" aria-hidden /> IRONBOOK
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

      {/* Left sidebar: past worlds (file system) + annotations */}
      {!shared && panelsOpen && (
        <aside className="viewer-sidebar left" data-testid="worlds-panel">
          <div className="sidebar-section grow">
            <div className="sidebar-header">Worlds</div>
            <div className="sidebar-body">
              <div className="world-list">
                {worlds === null && <div className="annot-empty">Loading…</div>}
                {worlds && worlds.length === 0 && (
                  <div className="annot-empty">No worlds yet.</div>
                )}
                {worlds?.map((w) => (
                  <button
                    key={w.id}
                    className={`world-item${w.id === id ? ' active' : ''}`}
                    onClick={() => navigate(`/projects/${w.id}/viewer`)}
                    title={w.name}
                    data-testid="worlds-item"
                  >
                    <span className={`wi-dot ${w.status}`} aria-hidden />
                    <span className="wi-name">{w.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-header">Annotations</div>
            <div className="sidebar-body">
              <div className="annot-empty">
                No annotations yet. Pin notes to points in the scene to label and
                measure features.
              </div>
              <div className="annot-add" title="Coming soon">+ Add annotation (soon)</div>
            </div>
          </div>
        </aside>
      )}

      {/* Navigation agent — right sidebar */}
      <AgentChat viewerRef={viewerRef} />

      {/* Bottom status / tool bar */}
      <div className="status-bar">
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
            Capture
          </button>
          <button
            className={gridOn ? 'active' : ''}
            onClick={onToggleGrid}
            data-testid="toggle-grid"
            title="Toggle ground grid + axes"
          >
            {gridOn ? 'Grid on' : 'Grid off'}
          </button>
        </div>

        <div className="spacer" />
        <div className="sb-item">
          Drag look · WASD move · Scroll zoom · Shift faster · Space/C up/down
        </div>
        <div className="spacer" />
        <div className="sb-item"><b>{fps}</b> fps</div>
        <div className="sb-item"><b>{splatCount.toLocaleString()}</b> splats</div>
      </div>
    </div>
  );
}
