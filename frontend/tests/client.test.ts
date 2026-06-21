import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiClient, ApiError, API_BASE } from '../src/api/client';

type FetchCall = { url: string; init?: RequestInit };

function mockFetch(impl: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const calls: FetchCall[] = [];
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    return impl(url, init);
  });
  vi.stubGlobal('fetch', fn);
  return { calls };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('apiClient endpoints', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('health -> GET /health', async () => {
    const { calls } = mockFetch(() =>
      jsonResponse({ status: 'ok', reconstruction_backend: 'fallback' }),
    );
    const h = await apiClient.health();
    expect(calls[0].url).toBe(`${API_BASE}/health`);
    expect(h.reconstruction_backend).toBe('fallback');
  });

  it('createProject -> POST /projects with JSON body', async () => {
    const { calls } = mockFetch(() => jsonResponse({ id: 'abc', name: 'Hi' }));
    await apiClient.createProject('Hi');
    expect(calls[0].url).toBe(`${API_BASE}/projects`);
    expect(calls[0].init?.method).toBe('POST');
    expect(calls[0].init?.headers).toMatchObject({ 'Content-Type': 'application/json' });
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({ name: 'Hi' });
  });

  it('listProjects -> GET /projects', async () => {
    const { calls } = mockFetch(() => jsonResponse([]));
    await apiClient.listProjects();
    expect(calls[0].url).toBe(`${API_BASE}/projects`);
    expect(calls[0].init?.method ?? 'GET').toBe('GET');
  });

  it('getProject -> GET /projects/{id}', async () => {
    const { calls } = mockFetch(() => jsonResponse({ id: 'p1' }));
    await apiClient.getProject('p1');
    expect(calls[0].url).toBe(`${API_BASE}/projects/p1`);
  });

  it('deleteProject -> DELETE /projects/{id} and tolerates 204', async () => {
    const { calls } = mockFetch(() => new Response(null, { status: 204 }));
    await apiClient.deleteProject('p1');
    expect(calls[0].url).toBe(`${API_BASE}/projects/p1`);
    expect(calls[0].init?.method).toBe('DELETE');
  });

  it('uploadFiles -> POST /projects/{id}/uploads with multipart field "files"', async () => {
    const { calls } = mockFetch(() =>
      jsonResponse({ accepted: 2, rejected: [], photo_count: 2, warnings: [] }),
    );
    const f1 = new File([new Uint8Array([1])], 'a.jpg', { type: 'image/jpeg' });
    const f2 = new File([new Uint8Array([2])], 'b.jpg', { type: 'image/jpeg' });
    await apiClient.uploadFiles('p1', [f1, f2]);
    expect(calls[0].url).toBe(`${API_BASE}/projects/p1/uploads`);
    expect(calls[0].init?.method).toBe('POST');
    const body = calls[0].init?.body as FormData;
    expect(body).toBeInstanceOf(FormData);
    expect(body.getAll('files').length).toBe(2);
  });

  it('listUploads -> GET /projects/{id}/uploads', async () => {
    const { calls } = mockFetch(() => jsonResponse([]));
    await apiClient.listUploads('p1');
    expect(calls[0].url).toBe(`${API_BASE}/projects/p1/uploads`);
  });

  it('reconstruct -> POST /projects/{id}/reconstruct', async () => {
    const { calls } = mockFetch(() => jsonResponse({ status: 'queued', stages: [], logs: [] }));
    await apiClient.reconstruct('p1');
    expect(calls[0].url).toBe(`${API_BASE}/projects/p1/reconstruct`);
    expect(calls[0].init?.method).toBe('POST');
  });

  it('getJob -> GET /projects/{id}/job', async () => {
    const { calls } = mockFetch(() => jsonResponse({ status: 'processing', stages: [], logs: [] }));
    await apiClient.getJob('p1');
    expect(calls[0].url).toBe(`${API_BASE}/projects/p1/job`);
  });

  it('assetInfo -> GET /projects/{id}/asset/info', async () => {
    const { calls } = mockFetch(() =>
      jsonResponse({ splat_count: 5, bytes: 160, bounds: { min: [0, 0, 0], max: [1, 1, 1] }, format: 'splat' }),
    );
    const info = await apiClient.assetInfo('p1');
    expect(calls[0].url).toBe(`${API_BASE}/projects/p1/asset/info`);
    expect(info.splat_count).toBe(5);
  });

  it('share -> GET /projects/{id}/share', async () => {
    const { calls } = mockFetch(() =>
      jsonResponse({ url: 'http://localhost:5173/view/p1' }),
    );
    const s = await apiClient.share('p1');
    expect(calls[0].url).toBe(`${API_BASE}/projects/p1/share`);
    expect(s.url).toContain('/view/p1');
  });

  it('URL builders produce contract paths', () => {
    expect(apiClient.thumbnailUrl('p1')).toBe(`${API_BASE}/projects/p1/thumbnail`);
    expect(apiClient.assetUrl('p1')).toBe(`${API_BASE}/projects/p1/asset`);
  });

  it('throws ApiError carrying the backend detail on non-2xx', async () => {
    mockFetch(() => jsonResponse({ detail: 'Too few photos to build a world.' }, 400));
    await expect(apiClient.getProject('nope')).rejects.toMatchObject({
      name: 'ApiError',
      status: 400,
      message: 'Too few photos to build a world.',
    });
  });

  it('ApiError is an Error subclass', () => {
    const e = new ApiError(500, 'x');
    expect(e).toBeInstanceOf(Error);
  });
});
