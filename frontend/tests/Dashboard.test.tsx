import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import type { Project } from '../src/api/types';

// Mock the API client module before importing the component.
vi.mock('../src/api/client', () => {
  const projects: Project[] = [
    {
      id: 'p1',
      name: 'Living Room',
      status: 'ready',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
      photo_count: 42,
      thumbnail_url: '/api/projects/p1/thumbnail',
      has_asset: true,
    },
    {
      id: 'p2',
      name: 'Garden',
      status: 'processing',
      created_at: '2026-01-02T00:00:00Z',
      updated_at: '2026-01-02T00:00:00Z',
      photo_count: 18,
      thumbnail_url: null,
      has_asset: false,
    },
  ];
  return {
    API_BASE: '/api',
    apiClient: {
      listProjects: vi.fn(async () => projects),
      health: vi.fn(async () => ({ status: 'ok', reconstruction_backend: 'fallback' })),
      thumbnailUrl: (id: string) => `/api/projects/${id}/thumbnail`,
      deleteProject: vi.fn(async () => undefined),
    },
  };
});

import { Dashboard } from '../src/pages/Dashboard';

function renderDash() {
  return render(
    <MemoryRouter>
      <Dashboard />
    </MemoryRouter>,
  );
}

describe('Dashboard', () => {
  afterEach(() => vi.clearAllMocks());

  it('renders projects from the mocked client', async () => {
    renderDash();
    await waitFor(() => {
      expect(screen.getByText('Living Room')).toBeInTheDocument();
      expect(screen.getByText('Garden')).toBeInTheDocument();
    });
    const cards = screen.getAllByTestId('project-card');
    expect(cards).toHaveLength(2);
    expect(screen.getByText('42 photos')).toBeInTheDocument();
  });

  it('does not surface a demo-mode notice', async () => {
    renderDash();
    await waitFor(() => {
      expect(screen.getAllByTestId('project-card').length).toBeGreaterThan(0);
    });
    expect(screen.queryByText(/demo mode/i)).toBeNull();
  });

  it('renders friendly status labels (no jargon)', async () => {
    renderDash();
    await waitFor(() => {
      expect(screen.getByText('Ready')).toBeInTheDocument();
      expect(screen.getByText('Building')).toBeInTheDocument();
    });
  });
});
