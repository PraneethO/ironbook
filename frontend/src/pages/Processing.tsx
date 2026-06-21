/**
 * Processing Status (screen 4) — polls GET /job, shows friendly staged
 * progress with per-stage bars + current stage, an expandable technical-logs
 * panel, auto-advances to the viewer when ready, and shows a friendly error on
 * failure.
 */
import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { apiClient } from '../api/client';
import type { Job } from '../api/types';
import { Notice, ProgressBar, Spinner } from '../components/ui';

const POLL_MS = 1500;

export function Processing() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!id) return;
    let active = true;

    const poll = async () => {
      try {
        const j = await apiClient.getJob(id);
        if (!active) return;
        setJob(j);
        if (j.status === 'ready') {
          // brief pause so the user sees it complete, then open the world
          timer.current = setTimeout(() => navigate(`/projects/${id}/viewer`), 800);
          return;
        }
        if (j.status === 'failed') return; // stop polling on failure
        timer.current = setTimeout(poll, POLL_MS);
      } catch (e) {
        if (!active) return;
        setError(e instanceof Error ? e.message : 'Lost connection to the build.');
        timer.current = setTimeout(poll, POLL_MS * 2);
      }
    };

    void poll();
    return () => {
      active = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [id, navigate]);

  const stageIcon = (status: string) =>
    status === 'done' ? '✓' : status === 'failed' ? '!' : status === 'active' ? '•' : '';

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Building your world…</h1>
          <p className="muted">This can take a little while. You can keep this tab open.</p>
        </div>
      </div>

      {error && <Notice kind="error">{error}</Notice>}

      {!job ? (
        <Spinner label="Getting started…" />
      ) : job.status === 'failed' ? (
        <div>
          <Notice kind="error">
            {job.error ?? "We couldn't finish building this world. Try adding more photos from different angles."}
          </Notice>
          <div className="row-actions">
            <button className="btn" onClick={() => navigate(`/projects/${id}/upload`)}>
              Add more photos
            </button>
            <button className="btn btn-ghost" onClick={() => navigate('/')}>
              Back to worlds
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="section">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <span className="muted">Overall progress</span>
              <span className="muted">{Math.round(job.progress * 100)}%</span>
            </div>
            <ProgressBar value={job.progress} />
          </div>

          <div className="stage-list">
            {job.stages.map((s) => (
              <div className={`stage ${s.status}`} key={s.key} data-testid="stage">
                <div className="dot">{stageIcon(s.status)}</div>
                <div className="stage-main">
                  <div className="stage-label">{s.label}</div>
                  <ProgressBar value={s.status === 'done' ? 1 : s.progress} />
                </div>
              </div>
            ))}
          </div>

          {job.status === 'ready' && (
            <Notice kind="info">Your world is ready! Opening the viewer…</Notice>
          )}

          <details className="logs">
            <summary>Show technical logs</summary>
            <div className="log-lines">
              {job.logs.length === 0 ? (
                <div className="log-line">No logs yet.</div>
              ) : (
                job.logs.map((l, i) => (
                  <div className={`log-line ${l.level}`} key={i}>
                    [{l.level}] {l.stage}: {l.message}
                  </div>
                ))
              )}
            </div>
          </details>
        </>
      )}
    </div>
  );
}
