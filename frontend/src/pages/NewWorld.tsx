/**
 * New World / Upload (screen 2). Lets the user name a world, drag-drop or pick
 * photos (and try video), preview them, see live file count + a quality
 * estimate and coverage warnings from the backend's ValidationReport, then
 * start the build. Shows demo-mode and video-unavailable notices.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Link } from 'react-router-dom';
import { apiClient } from '../api/client';
import type { Health, ValidationReport } from '../api/types';
import { Notice, Spinner, Stat } from '../components/ui';
import { convertHeicFiles } from '../lib/heic';

const IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.heic', '.heif'];
const VIDEO_EXT = ['.mp4', '.mov'];

function extOf(name: string): string {
  const i = name.lastIndexOf('.');
  return i >= 0 ? name.slice(i).toLowerCase() : '';
}

function qualityLabel(score: number): string {
  if (score >= 0.75) return 'Great';
  if (score >= 0.5) return 'Good';
  if (score >= 0.3) return 'Fair';
  return 'Low';
}

export function NewWorld() {
  const navigate = useNavigate();
  const params = useParams();
  const existingId = params.id ?? null;

  const [name, setName] = useState('My 3D World');
  const [splatUploading, setSplatUploading] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);
  const splatInputRef = useRef<HTMLInputElement | null>(null);
  const [projectId, setProjectId] = useState<string | null>(existingId);
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoNotice, setVideoNotice] = useState(false);
  const [converting, setConverting] = useState(false);
  const [convertError, setConvertError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [starting, setStarting] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onUploadSplat = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSplatUploading(true);
    setError(null);
    try {
      const isPly = file.name.toLowerCase().endsWith('.ply');
      const worldName = name.trim() || 'Uploaded world';
      const proj = isPly
        ? await apiClient.uploadPly(file, worldName)
        : await apiClient.uploadSplat(file, worldName);
      navigate(`/projects/${proj.id}/viewer`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not upload that file.');
      setSplatUploading(false);
    }
  };

  const onLoadDemo = async () => {
    setDemoLoading(true);
    setError(null);
    try {
      const proj = await apiClient.loadDemo();
      navigate(`/projects/${proj.id}/viewer`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load the demo scene.');
      setDemoLoading(false);
    }
  };

  useEffect(() => {
    apiClient.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  // Build object-URL previews; revoke on change/unmount to avoid leaks.
  useEffect(() => {
    const urls = files.map((f) => URL.createObjectURL(f));
    setPreviews(urls);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [files]);

  const imageFiles = useMemo(
    () => files.filter((f) => IMAGE_EXT.includes(extOf(f.name))),
    [files],
  );

  const addFiles = async (incoming: FileList | File[]) => {
    const arr = Array.from(incoming);
    const hasVideo = arr.some((f) => VIDEO_EXT.includes(extOf(f.name)));
    // Video frame-extraction needs ffmpeg on the backend; show friendly note.
    if (hasVideo) setVideoNotice(true);
    const accepted = arr.filter((f) => IMAGE_EXT.includes(extOf(f.name)));
    if (accepted.length === 0) return;
    setReport(null);
    setConvertError(null);

    // HEIC/HEIF (iPhone photos) can't be previewed or decoded downstream, so
    // convert them to JPEG in the browser before they enter the upload set.
    const needsConversion = accepted.some(
      (f) => extOf(f.name) === '.heic' || extOf(f.name) === '.heif',
    );
    if (!needsConversion) {
      setFiles((cur) => [...cur, ...accepted]);
      return;
    }

    setConverting(true);
    try {
      const { files: converted, failed } = await convertHeicFiles(accepted);
      setFiles((cur) => [...cur, ...converted]);
      if (failed.length > 0) {
        setConvertError(
          `We couldn't convert ${failed.length} HEIC photo(s): ` +
            `${failed.join(', ')}. Try exporting them as JPEG and adding again.`,
        );
      }
    } finally {
      setConverting(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer?.files) addFiles(e.dataTransfer.files);
  };

  const ensureProject = async (): Promise<string> => {
    if (projectId) return projectId;
    const p = await apiClient.createProject(name.trim() || 'My 3D World');
    setProjectId(p.id);
    return p.id;
  };

  const onUpload = async () => {
    if (imageFiles.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const id = await ensureProject();
      const rep = await apiClient.uploadFiles(id, imageFiles);
      setReport(rep);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const onStart = async () => {
    if (!projectId) return;
    setStarting(true);
    setError(null);
    try {
      await apiClient.reconstruct(projectId);
      navigate(`/projects/${projectId}/processing`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start building your world.');
      setStarting(false);
    }
  };

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>{existingId ? 'Add photos' : 'New 3D World'}</h1>
          <p className="muted">
            Upload photos of your space.{' '}
            <Link to="/guide">See capture tips →</Link>
          </p>
        </div>
      </div>

      {(health?.reconstruction_backend === 'brush' ||
        health?.reconstruction_backend === 'msplat' ||
        health?.reconstruction_backend === 'colmap_gsplat' ||
        health?.reconstruction_backend === 'gaussian_3dgs') && (
        <Notice kind="info">
          We'll build a real 3D Gaussian Splatting world, trained on your Apple GPU. For the best
          result, upload 20+ overlapping photos taken while walking around your subject.
        </Notice>
      )}
      {health?.reconstruction_backend === 'depth' && (
        <Notice kind="info">
          We'll build your world on-device from the depth in your photos. It's a fast 2.5D
          reconstruction — great for walking through your capture. (Full multi-view 3D needs a GPU.)
        </Notice>
      )}
      {health?.reconstruction_backend === 'fallback' && (
        <Notice kind="info">
          Demo mode is on — we'll build a quick preview world so you can try the whole flow.
        </Notice>
      )}

      {!existingId && (
        <label className="field" style={{ maxWidth: 420 }}>
          <span className="lbl">World name</span>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My living room"
          />
        </label>
      )}

      <div
        className={`dropzone${dragging ? ' drag' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        data-testid="dropzone"
      >
        <p style={{ margin: '0 0 6px', fontWeight: 500 }}>
          [ DROP PHOTOS &mdash; OR CLICK ]
        </p>
        <p className="faint">JPG · PNG · HEIC · MP4/MOV when available</p>
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.heic,.heif,.mp4,.mov,image/*,video/*"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => e.target.files && addFiles(e.target.files)}
          data-testid="file-input"
        />
      </div>

      {videoNotice && (
        <Notice kind="warn">
          Video uploads need extra processing that isn't available right now. We've kept your
          photos and skipped the video — please add still photos for the best results.
        </Notice>
      )}

      {converting && (
        <Notice kind="info">Converting HEIC photos to JPEG…</Notice>
      )}
      {convertError && <Notice kind="warn">{convertError}</Notice>}

      <div className="stat-row">
        <Stat label="Photos selected" value={imageFiles.length} />
        {report && <Stat label="Accepted" value={report.accepted} />}
        {report && (
          <Stat
            tone="success"
            label="Quality"
            value={`${qualityLabel(report.quality_score)} (${Math.round(report.quality_score * 100)}%)`}
          />
        )}
        {report && (
          <Stat tone="success" label="Coverage" value={`${Math.round(report.coverage_score * 100)}%`} />
        )}
      </div>

      {imageFiles.length > 0 && imageFiles.length < 8 && (
        <Notice kind="warn">
          We recommend at least 8 photos (20+ for a great result). Add more angles for a fuller
          world.
        </Notice>
      )}

      {error && <Notice kind="error">{error}</Notice>}

      {report && (
        <div className="section">
          {report.warnings.length > 0 &&
            report.warnings.map((w, i) => (
              <Notice kind="warn" key={i}>
                {w}
              </Notice>
            ))}
          {report.rejected.length > 0 && (
            <Notice kind="warn">
              {report.rejected.length} file(s) were skipped:{' '}
              {report.rejected.map((r) => `${r.filename} (${r.reason})`).join('; ')}
            </Notice>
          )}
          {report.ready_to_reconstruct ? (
            <Notice kind="info">Looks good! You're ready to build your world.</Notice>
          ) : (
            <Notice kind="warn">
              Add more photos from different angles before we can build a good world.
            </Notice>
          )}
        </div>
      )}

      {previews.length > 0 && (
        <div className="section">
          <h2>Preview ({imageFiles.length})</h2>
          <div className="grid grid-thumbs">
            {previews.slice(0, 60).map((u, i) => (
              <div className="thumb" key={i}>
                <img src={u} alt={files[i]?.name ?? 'photo'} />
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="row-actions" style={{ marginTop: 24 }}>
        <button className="btn btn-ghost" onClick={() => navigate('/')}>
          Cancel
        </button>
        <div className="spacer" />
        {!report ? (
          <button
            className="btn btn-primary"
            disabled={imageFiles.length === 0 || uploading || converting}
            onClick={onUpload}
          >
            {uploading ? 'Uploading…' : 'Upload photos'}
          </button>
        ) : (
          <>
            <button className="btn" disabled={uploading} onClick={onUpload}>
              Re-check
            </button>
            <button
              className="btn btn-primary"
              disabled={!report.ready_to_reconstruct || starting}
              onClick={onStart}
            >
              {starting ? 'Starting…' : 'Create my world'}
            </button>
          </>
        )}
      </div>

      {converting && <Spinner label="Converting HEIC photos…" />}
      {uploading && <Spinner label="Checking your photos…" />}

      {/* Direct .splat/.ply upload — skip reconstruction, navigate straight to viewer */}
      <div className="section" style={{ marginTop: 32, borderTop: '1px solid var(--border)', paddingTop: 24 }}>
        <h2 style={{ marginBottom: 8 }}>Already have a .splat or .ply file?</h2>
        <p className="muted" style={{ marginBottom: 12 }}>
          Upload a pre-built Gaussian-splat scene directly — no photos needed. A
          trained 3DGS <code>.ply</code> (from any tool) is converted automatically.
          The agent will navigate it for you.
        </p>
        <input
          ref={splatInputRef}
          type="file"
          accept=".splat,.ply"
          style={{ display: 'none' }}
          onChange={onUploadSplat}
          data-testid="splat-file-input"
        />
        <button
          className="btn"
          disabled={splatUploading}
          onClick={() => splatInputRef.current?.click()}
          data-testid="upload-splat-btn"
        >
          {splatUploading ? 'Uploading…' : 'Upload .splat / .ply'}
        </button>

        <p className="muted" style={{ margin: '16px 0 8px' }}>
          Just want to see it in action? Load a real built-in scene — no file needed:
        </p>
        <button
          className="btn"
          disabled={demoLoading}
          onClick={onLoadDemo}
          data-testid="load-demo-btn"
        >
          {demoLoading ? 'Loading demo…' : 'See the demo bike'}
        </button>
      </div>
    </div>
  );
}
