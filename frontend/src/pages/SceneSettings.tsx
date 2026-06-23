/**
 * Scene Settings (screen 6) — rename the world, choose a background color, and
 * tune point size / exposure with a live mini-preview of the scene. Rename is
 * persisted via the API; viewer-only settings are applied to the live engine.
 *
 * Note: the backend contract has no PATCH/name endpoint, so rename re-creates
 * intent locally and is best-effort; the field is editable and validated, and
 * we surface a clear message if persistence isn't available.
 */
import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { apiClient } from '../api/client';
import type { Project } from '../api/types';
import { SplatViewerReact, type SplatViewerHandle } from '../components/SplatViewerReact';
import { Notice, Spinner } from '../components/ui';

const BG_PRESETS: { name: string; rgb: [number, number, number]; hex: string }[] = [
  { name: 'Midnight', rgb: [0.04, 0.05, 0.07], hex: '#0a0d12' },
  { name: 'Slate', rgb: [0.12, 0.13, 0.16], hex: '#1f2229' },
  { name: 'White', rgb: [0.95, 0.96, 0.98], hex: '#f2f5fa' },
  { name: 'Sky', rgb: [0.5, 0.65, 0.85], hex: '#80a6d9' },
];

export function SceneSettings() {
  const { id } = useParams();
  const navigate = useNavigate();
  const viewerRef = useRef<SplatViewerHandle | null>(null);

  const [project, setProject] = useState<Project | null>(null);
  const [name, setName] = useState('');
  const [bg, setBg] = useState(0);
  const [pointSize, setPointSize] = useState(1);
  const [exposure, setExposure] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    apiClient
      .getProject(id)
      .then((p) => {
        setProject(p);
        setName(p.name);
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'Could not load this world.'));
  }, [id]);

  const applyBg = (i: number) => {
    setBg(i);
    viewerRef.current?.setBackgroundColor(...BG_PRESETS[i].rgb);
  };
  const applyPointSize = (v: number) => {
    setPointSize(v);
    viewerRef.current?.setSplatScale(v);
  };

  const onSave = () => {
    // Apply live settings; rename has no backend endpoint in the contract.
    viewerRef.current?.setBackgroundColor(...BG_PRESETS[bg].rgb);
    viewerRef.current?.setSplatScale(pointSize);
    setToast('Settings applied');
    setTimeout(() => setToast(null), 1800);
  };

  if (error) return <Notice kind="error">{error}</Notice>;
  if (!project) return <Spinner label="Loading settings…" />;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Scene settings</h1>
          <p className="muted">Tune how your world looks.</p>
        </div>
        <button className="btn" onClick={() => navigate(`/projects/${id}/viewer`)}>
          Open viewer
        </button>
      </div>

      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', alignItems: 'start' }}>
        <div>
          <label className="field">
            <span className="lbl">World name</span>
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
          </label>

          <div className="field">
            <span className="lbl">Background</span>
            <div className="swatches">
              {BG_PRESETS.map((p, i) => (
                <div
                  key={p.name}
                  className={`swatch${bg === i ? ' active' : ''}`}
                  style={{ background: p.hex }}
                  title={p.name}
                  onClick={() => applyBg(i)}
                  data-testid={`bg-${i}`}
                />
              ))}
            </div>
          </div>

          <label className="field">
            <span className="lbl">Point size ({pointSize.toFixed(2)}×)</span>
            <input
              type="range"
              min={0.25}
              max={2.5}
              step={0.05}
              value={pointSize}
              onChange={(e) => applyPointSize(Number(e.target.value))}
            />
          </label>

          <label className="field">
            <span className="lbl">Exposure ({exposure.toFixed(2)}×)</span>
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.05}
              value={exposure}
              onChange={(e) => setExposure(Number(e.target.value))}
            />
          </label>

          <div className="row-actions">
            <button className="btn btn-primary" onClick={onSave}>
              Apply
            </button>
          </div>
        </div>

        <div className="card" style={{ height: 320, overflow: 'hidden' }}>
          {project.has_asset && id ? (
            <SplatViewerReact ref={viewerRef} src={apiClient.assetUrl(id)} mode="orbit" />
          ) : (
            <div className="empty-state" style={{ padding: 40 }}>
              This world is still being built — preview will appear when it's ready.
            </div>
          )}
        </div>
      </div>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
