/**
 * Export / Share (screen 7) — download the .splat asset, copy the share link
 * (/view/{id}), capture a screenshot via the live viewer, and a "coming soon"
 * video-flythrough stub.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { apiClient } from '../api/client';
import type { AssetInfo } from '../api/types';
import { SplatViewerReact, type SplatViewerHandle } from '../components/SplatViewerReact';
import { Notice, Spinner, Stat } from '../components/ui';

export function ExportShare() {
  const { id } = useParams();
  const navigate = useNavigate();
  const viewerRef = useRef<SplatViewerHandle | null>(null);

  const [shareUrl, setShareUrl] = useState<string>('');
  const [info, setInfo] = useState<AssetInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const assetUrl = useMemo(() => (id ? apiClient.assetUrl(id) : ''), [id]);

  useEffect(() => {
    if (!id) return;
    apiClient
      .share(id)
      .then((s) => setShareUrl(s.url))
      .catch(() => setShareUrl(`${window.location.origin}/view/${id}`));
    apiClient
      .assetInfo(id)
      .then(setInfo)
      .catch((e) => setError(e instanceof Error ? e.message : 'World assets not ready yet.'));
  }, [id]);

  const flash = (m: string) => {
    setToast(m);
    setTimeout(() => setToast(null), 1800);
  };

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      flash('Link copied!');
    } catch {
      flash('Copy failed — select and copy manually.');
    }
  };

  const onDownload = () => {
    if (!assetUrl) return;
    const a = document.createElement('a');
    a.href = assetUrl;
    a.download = `world-${id}.splat`;
    a.click();
  };

  const onScreenshot = () => {
    const data = viewerRef.current?.capture();
    if (!data) {
      flash('Open the viewer first to capture a screenshot.');
      return;
    }
    const a = document.createElement('a');
    a.href = data;
    a.download = `world-${id}.png`;
    a.click();
    flash('Screenshot saved');
  };

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Share &amp; export</h1>
          <p className="muted">Send your world to anyone, or take it with you.</p>
        </div>
        <button className="btn" onClick={() => navigate(`/projects/${id}/viewer`)}>
          Open viewer
        </button>
      </div>

      {error && <Notice kind="error">{error}</Notice>}

      <div className="section">
        <h2>Share link</h2>
        <p className="muted">Anyone with this link can explore your world.</p>
        <div className="row-actions" style={{ alignItems: 'center' }}>
          <input type="text" readOnly value={shareUrl} style={{ maxWidth: 460 }} />
          <button className="btn btn-primary" onClick={onCopy} disabled={!shareUrl}>
            Copy link
          </button>
        </div>
      </div>

      <div className="section">
        <h2>Export</h2>
        {info && (
          <div className="stat-row">
            <Stat label="Splats" value={info.splat_count.toLocaleString()} />
            <Stat label="File size" value={`${(info.bytes / 1024 / 1024).toFixed(1)} MB`} />
            <Stat label="Format" value={info.format} />
          </div>
        )}
        <div className="row-actions">
          <button className="btn btn-primary" onClick={onDownload}>
            Download 3D file (.splat)
          </button>
          <button className="btn" onClick={onScreenshot}>
            Save screenshot
          </button>
          <button className="btn" disabled title="Coming soon">
            Video flythrough (coming soon)
          </button>
        </div>
      </div>

      {/* Hidden-ish live viewer so screenshot capture works from this screen. */}
      <div className="card" style={{ height: 260, overflow: 'hidden', marginTop: 12 }}>
        {assetUrl ? (
          <SplatViewerReact ref={viewerRef} src={assetUrl} mode="orbit" />
        ) : (
          <Spinner label="Preparing preview…" />
        )}
      </div>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
