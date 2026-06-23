/** Small shared UI primitives used across screens. */
import type { ReactNode } from 'react';
import type { ProjectStatus } from '../api/types';

export function StatusBadge({ status }: { status: ProjectStatus }) {
  const friendly: Record<ProjectStatus, string> = {
    draft: 'Draft',
    uploading: 'Uploading',
    queued: 'In queue',
    processing: 'Building',
    ready: 'Ready',
    failed: 'Needs attention',
  };
  return <span className={`badge badge-${status}`}>{friendly[status]}</span>;
}

export function Notice({
  kind = 'info',
  children,
}: {
  kind?: 'info' | 'warn' | 'error';
  children: ReactNode;
}) {
  return <div className={`notice notice-${kind}`}>{children}</div>;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="center" style={{ padding: 48 }}>
      <div className="spinner" style={{ margin: '0 auto' }} />
      {label && <p className="muted" style={{ marginTop: 14 }}>{label}</p>}
    </div>
  );
}

export function Stat({
  label,
  value,
  tone = 'default',
}: {
  label: string;
  value: ReactNode;
  tone?: 'default' | 'success';
}) {
  return (
    <div className={`stat${tone === 'success' ? ' stat-conf' : ''}`}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

export function ProgressBar({ value }: { value: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="progress" role="progressbar" aria-valuenow={pct}>
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}
