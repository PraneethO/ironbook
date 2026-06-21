import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import type { Job } from '../src/api/types';

const job: Job = {
  project_id: 'p1',
  status: 'processing',
  progress: 0.45,
  current_stage: 'optimization',
  stages: [
    { key: 'preprocessing', label: 'Preparing your photos', status: 'done', progress: 1 },
    { key: 'pose_estimation', label: 'Finding camera positions', status: 'done', progress: 1 },
    { key: 'structure', label: 'Building rough 3D structure', status: 'active', progress: 0.5 },
    { key: 'optimization', label: 'Optimizing visual detail', status: 'pending', progress: 0 },
    { key: 'compression', label: 'Compressing the scene', status: 'pending', progress: 0 },
    { key: 'viewer_asset', label: 'Preparing interactive viewer', status: 'pending', progress: 0 },
  ],
  logs: [
    { ts: '2026-01-01T00:00:00Z', level: 'info', stage: 'preprocessing', message: 'Normalized 42 photos' },
    { ts: '2026-01-01T00:00:01Z', level: 'warn', stage: 'structure', message: 'Sparse overlap in region 3' },
  ],
  error: null,
};

vi.mock('../src/api/client', () => ({
  API_BASE: '/api',
  apiClient: {
    getJob: vi.fn(async () => job),
  },
}));

import { Processing } from '../src/pages/Processing';

function setup() {
  return render(
    <MemoryRouter initialEntries={['/projects/p1/processing']}>
      <Routes>
        <Route path="/projects/:id/processing" element={<Processing />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('Processing', () => {
  afterEach(() => vi.clearAllMocks());

  it('renders friendly stage labels from the mocked Job', async () => {
    setup();
    await waitFor(() => {
      expect(screen.getByText('Preparing your photos')).toBeInTheDocument();
      expect(screen.getByText('Finding camera positions')).toBeInTheDocument();
      expect(screen.getByText('Optimizing visual detail')).toBeInTheDocument();
    });
    const stages = screen.getAllByTestId('stage');
    expect(stages).toHaveLength(6);
  });

  it('shows overall progress percentage', async () => {
    setup();
    await waitFor(() => expect(screen.getByText('45%')).toBeInTheDocument());
  });

  it('exposes technical logs in an expandable panel', async () => {
    setup();
    await waitFor(() => {
      expect(screen.getByText('Show technical logs')).toBeInTheDocument();
      expect(screen.getByText(/Normalized 42 photos/)).toBeInTheDocument();
      expect(screen.getByText(/Sparse overlap in region 3/)).toBeInTheDocument();
    });
  });
});
