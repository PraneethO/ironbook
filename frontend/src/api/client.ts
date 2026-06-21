/**
 * client.ts — typed REST client for the backend (CONTRACT.md §2).
 *
 * The API base is configurable via VITE_API_BASE (default `/api`, which the
 * Vite dev server proxies to http://localhost:8000). All methods return the
 * typed models from ./types and throw ApiError (with the backend's friendly
 * `detail` message) on non-2xx responses.
 */

import type {
  AgentActResponse,
  AgentTurn,
  AssetInfo,
  CameraSnapshot,
  Health,
  Job,
  Project,
  ShareLink,
  UploadedImage,
  ValidationReport,
} from './types';

export const API_BASE: string =
  (import.meta.env?.VITE_API_BASE as string | undefined) ?? '/api';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function parseError(res: Response): Promise<never> {
  let detail = `Something went wrong (status ${res.status}).`;
  try {
    const body = await res.json();
    if (body && typeof body.detail === 'string') detail = body.detail;
  } catch {
    /* non-JSON error body */
  }
  throw new ApiError(res.status, detail);
}

async function jsonRequest<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) await parseError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const apiClient = {
  /** Absolute base used for building asset/thumbnail URLs for <img>/viewer. */
  base: API_BASE,

  // GET /api/health
  health(): Promise<Health> {
    return jsonRequest<Health>('/health');
  },

  // POST /api/projects
  createProject(name: string): Promise<Project> {
    return jsonRequest<Project>('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
  },

  // GET /api/projects
  listProjects(): Promise<Project[]> {
    return jsonRequest<Project[]>('/projects');
  },

  // GET /api/projects/{id}
  getProject(id: string): Promise<Project> {
    return jsonRequest<Project>(`/projects/${id}`);
  },

  // DELETE /api/projects/{id}
  async deleteProject(id: string): Promise<void> {
    await jsonRequest<void>(`/projects/${id}`, { method: 'DELETE' });
  },

  // POST /api/projects/{id}/uploads  (multipart field `files`)
  uploadFiles(id: string, files: File[]): Promise<ValidationReport> {
    const form = new FormData();
    for (const f of files) form.append('files', f, f.name);
    return jsonRequest<ValidationReport>(`/projects/${id}/uploads`, {
      method: 'POST',
      body: form,
    });
  },

  // GET /api/projects/{id}/uploads
  listUploads(id: string): Promise<UploadedImage[]> {
    return jsonRequest<UploadedImage[]>(`/projects/${id}/uploads`);
  },

  // GET /api/projects/{id}/thumbnail (URL builder for <img src>)
  thumbnailUrl(id: string): string {
    return `${API_BASE}/projects/${id}/thumbnail`;
  },

  // POST /api/projects/{id}/reconstruct
  reconstruct(id: string): Promise<Job> {
    return jsonRequest<Job>(`/projects/${id}/reconstruct`, { method: 'POST' });
  },

  // GET /api/projects/{id}/job
  getJob(id: string): Promise<Job> {
    return jsonRequest<Job>(`/projects/${id}/job`);
  },

  // GET /api/projects/{id}/asset (URL builder for viewer / download)
  assetUrl(id: string): string {
    return `${API_BASE}/projects/${id}/asset`;
  },

  // GET /api/projects/{id}/asset/info
  assetInfo(id: string): Promise<AssetInfo> {
    return jsonRequest<AssetInfo>(`/projects/${id}/asset/info`);
  },

  // GET /api/projects/{id}/share
  share(id: string): Promise<ShareLink> {
    return jsonRequest<ShareLink>(`/projects/${id}/share`);
  },

  // POST /api/projects/upload_splat — bring-your-own .splat file
  uploadSplat(file: File, name: string): Promise<Project> {
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('name', name);
    // Do NOT set Content-Type manually; the browser sets multipart/form-data + boundary.
    return jsonRequest<Project>('/projects/upload_splat', {
      method: 'POST',
      body: form,
    });
  },

  // POST /api/agent/act — reasoning navigation agent
  agentAct(body: {
    message: string;
    screenshot_b64?: string;
    camera: CameraSnapshot;
    history: AgentTurn[];
  }): Promise<AgentActResponse> {
    return jsonRequest<AgentActResponse>('/agent/act', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  },
};

export type ApiClient = typeof apiClient;
