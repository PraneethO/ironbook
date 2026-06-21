/**
 * Dashboard (screen 1) — lists 3D Worlds as cards with thumbnail + status,
 * a "New 3D World" entry point, and delete. Shows a notice when the backend is
 * running the demo (fallback) reconstructor.
 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../api/client';
import type { Health, Project } from '../api/types';
import { Notice, Spinner, StatusBadge } from '../components/ui';

export function Dashboard() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const list = await apiClient.listProjects();
      setProjects(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load your worlds.');
      setProjects([]);
    }
  };

  useEffect(() => {
    void refresh();
    apiClient.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const onOpen = (p: Project) => {
    if (p.status === 'ready') navigate(`/projects/${p.id}/viewer`);
    else if (p.status === 'processing' || p.status === 'queued')
      navigate(`/projects/${p.id}/processing`);
    else navigate(`/projects/${p.id}/upload`);
  };

  const onDelete = async (e: React.MouseEvent, p: Project) => {
    e.stopPropagation();
    if (!confirm(`Delete "${p.name}"? This can't be undone.`)) return;
    try {
      await apiClient.deleteProject(p.id);
      setProjects((cur) => (cur ? cur.filter((x) => x.id !== p.id) : cur));
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Could not delete this world.');
    }
  };

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Your 3D Worlds</h1>
          <p className="muted">Turn your photos into worlds you can walk through.</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/new')}>
          + New 3D World
        </button>
      </div>

      {(health?.reconstruction_backend === 'brush' ||
        health?.reconstruction_backend === 'msplat' ||
        health?.reconstruction_backend === 'colmap_gsplat' ||
        health?.reconstruction_backend === 'gaussian_3dgs') && (
        <Notice kind="info">
          Worlds are built with real <b>3D Gaussian Splatting</b>, trained on your Apple GPU from
          your photos. Best results come from 20+ overlapping photos taken while walking around
          the subject.
        </Notice>
      )}
      {health?.reconstruction_backend === 'depth' && (
        <Notice kind="info">
          Worlds are reconstructed <b>on this device</b> from the depth in your photos (a fast 2.5D
          build). Full multi-view 3D Gaussian Splatting runs on the GPU engine.
        </Notice>
      )}
      {health?.reconstruction_backend === 'fallback' && (
        <Notice kind="info">
          You're in <b>demo mode</b>: worlds are built with a quick preview reconstructor so you
          can try the full flow. Results improve with the full engine.
        </Notice>
      )}

      {error && <Notice kind="error">{error}</Notice>}

      {projects === null ? (
        <Spinner label="Loading your worlds…" />
      ) : projects.length === 0 ? (
        <div className="empty-state">
          <div className="big">🌍</div>
          <h2>No worlds yet</h2>
          <p>Upload photos of a room, object, or place to build your first 3D world.</p>
          <button
            className="btn btn-primary"
            style={{ marginTop: 16 }}
            onClick={() => navigate('/new')}
          >
            Create your first world
          </button>
        </div>
      ) : (
        <div className="grid grid-cards">
          {projects.map((p) => (
            <div
              key={p.id}
              className="card project-card"
              onClick={() => onOpen(p)}
              data-testid="project-card"
            >
              <div className="project-thumb">
                {p.thumbnail_url ? (
                  <img src={apiClient.thumbnailUrl(p.id)} alt={p.name} />
                ) : (
                  <span>🖼️</span>
                )}
              </div>
              <div className="project-body">
                <div className="row">
                  <span className="project-name">{p.name}</span>
                  <StatusBadge status={p.status} />
                </div>
                <div className="row">
                  <span className="faint">{p.photo_count} photos</span>
                  <button
                    className="btn btn-danger"
                    onClick={(e) => onDelete(e, p)}
                    aria-label={`Delete ${p.name}`}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
