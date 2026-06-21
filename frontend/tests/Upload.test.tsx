import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import type { ValidationReport } from '../src/api/types';

const report: ValidationReport = {
  accepted: 10,
  rejected: [{ filename: 'blurry.jpg', reason: 'too blurry' }],
  photo_count: 10,
  coverage_score: 0.4,
  quality_score: 0.62,
  warnings: ['We have many front views but few side views. The 3D result may have holes on the sides.'],
  ready_to_reconstruct: true,
};

vi.mock('../src/api/client', () => ({
  API_BASE: '/api',
  apiClient: {
    health: vi.fn(async () => ({ status: 'ok', reconstruction_backend: 'colmap_gsplat' })),
    createProject: vi.fn(async () => ({ id: 'newp', name: 'My 3D World' })),
    uploadFiles: vi.fn(async () => report),
    reconstruct: vi.fn(async () => ({ status: 'queued', stages: [], logs: [] })),
  },
}));

import { NewWorld } from '../src/pages/NewWorld';
import { apiClient } from '../src/api/client';

function setup() {
  return render(
    <MemoryRouter>
      <NewWorld />
    </MemoryRouter>,
  );
}

function makeImage(name: string): File {
  return new File([new Uint8Array([1, 2, 3])], name, { type: 'image/jpeg' });
}

describe('NewWorld / Upload', () => {
  afterEach(() => vi.clearAllMocks());

  it('shows live file count after choosing photos', async () => {
    setup();
    const input = screen.getByTestId('file-input') as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [makeImage('a.jpg'), makeImage('b.jpg')] },
    });
    await waitFor(() => {
      expect(screen.getByText('Photos selected')).toBeInTheDocument();
      expect(screen.getByText('2')).toBeInTheDocument();
    });
  });

  it('surfaces coverage warnings and rejected files from the ValidationReport', async () => {
    setup();
    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const files = Array.from({ length: 10 }, (_, i) => makeImage(`img${i}.jpg`));
    fireEvent.change(input, { target: { files } });

    fireEvent.click(screen.getByRole('button', { name: /upload photos/i }));

    await waitFor(() => {
      expect(screen.getByText(/few side views/i)).toBeInTheDocument();
      expect(screen.getByText(/blurry\.jpg/i)).toBeInTheDocument();
      expect(screen.getByText(/ready to build/i)).toBeInTheDocument();
    });
    expect(apiClient.uploadFiles).toHaveBeenCalledWith('newp', expect.any(Array));
  });

  it('shows a friendly video-unavailable notice when a video is added', async () => {
    setup();
    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const video = new File([new Uint8Array([0])], 'clip.mp4', { type: 'video/mp4' });
    fireEvent.change(input, { target: { files: [video, makeImage('a.jpg')] } });
    await waitFor(() => {
      expect(screen.getByText(/video uploads need extra processing/i)).toBeInTheDocument();
    });
  });

  it('enables "Create my world" once the report is ready and starts reconstruction', async () => {
    setup();
    const input = screen.getByTestId('file-input') as HTMLInputElement;
    const files = Array.from({ length: 10 }, (_, i) => makeImage(`img${i}.jpg`));
    fireEvent.change(input, { target: { files } });
    fireEvent.click(screen.getByRole('button', { name: /upload photos/i }));

    const startBtn = await screen.findByRole('button', { name: /create my world/i });
    expect(startBtn).not.toBeDisabled();
    fireEvent.click(startBtn);
    await waitFor(() => expect(apiClient.reconstruct).toHaveBeenCalledWith('newp'));
  });
});
