import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';

// Mock the WebGL engine so jsdom never needs a GL context. We record method
// calls and let load() resolve immediately, invoking onLoaded via the wrapper.
// `vi.hoisted` lets the (hoisted) mock factory reference shared state safely.
const { instances, MockViewer } = vi.hoisted(() => {
  const instances: any[] = [];
  class MockViewer {
    mode: string;
    splatCount = 1234;
    onFps?: (n: number) => void;
    disposed = false;
    loadedUrls: string[] = [];
    constructor(opts: { mode?: string; onFps?: (n: number) => void }) {
      this.mode = opts.mode ?? 'orbit';
      this.onFps = opts.onFps;
      instances.push(this);
    }
    async load(url: string) {
      this.loadedUrls.push(url);
    }
    async loadBuffer() {}
    setMode(m: string) {
      this.mode = m;
    }
    getMode() {
      return this.mode;
    }
    resetCamera() {}
    capture() {
      return 'data:image/png;base64,AAAA';
    }
    setSplatScale() {}
    setBackgroundColor() {}
    dispose() {
      this.disposed = true;
    }
  }
  return { instances, MockViewer };
});

vi.mock('../src/viewer/SplatViewer', () => ({
  SplatViewer: MockViewer,
}));

vi.mock('../src/api/client', () => ({
  API_BASE: '/api',
  apiClient: {
    assetUrl: (id: string) => `/api/projects/${id}/asset`,
    agentAct: vi.fn(),
    voiceConfig: vi.fn().mockResolvedValue({ deepgram_key: 'test-key', model: 'nova-2' }),
    listProjects: vi.fn().mockResolvedValue([]),
  },
}));

import { ViewerPage } from '../src/pages/ViewerPage';

function setup(shared = false) {
  const path = shared ? '/view/p1' : '/projects/p1/viewer';
  const pattern = shared ? '/view/:id' : '/projects/:id/viewer';
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path={pattern} element={<ViewerPage shared={shared} />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('ViewerPage', () => {
  afterEach(() => {
    instances.length = 0;
    vi.clearAllMocks();
  });

  it('mounts the viewer engine and loads the project asset URL', async () => {
    setup();
    await waitFor(() => {
      expect(instances.length).toBeGreaterThan(0);
      expect(instances[0].loadedUrls).toContain('/api/projects/p1/asset');
    });
  });

  it('renders the mode switcher with Orbit/Walk/Fly/Screenshot', async () => {
    setup();
    expect(screen.getByTestId('mode-orbit')).toBeInTheDocument();
    expect(screen.getByTestId('mode-walk')).toBeInTheDocument();
    expect(screen.getByTestId('mode-fly')).toBeInTheDocument();
    expect(screen.getByTestId('mode-screenshot')).toBeInTheDocument();
    await waitFor(() => expect(instances[0].loadedUrls.length).toBeGreaterThan(0));
  });

  it('switching modes calls setMode on the engine', async () => {
    setup();
    await waitFor(() => expect(instances.length).toBeGreaterThan(0));
    fireEvent.click(screen.getByTestId('mode-fly'));
    await waitFor(() => expect(instances[0].getMode()).toBe('fly'));
  });

  it('shows the controls help overlay (WASD, scroll, shift)', async () => {
    setup();
    expect(screen.getByText(/WASD move/i)).toBeInTheDocument();
    expect(screen.getByText(/Shift faster/i)).toBeInTheDocument();
    await waitFor(() => expect(instances[0].loadedUrls.length).toBeGreaterThan(0));
  });

  it('reports the splat count after load', async () => {
    setup();
    await waitFor(() => {
      expect(screen.getByText(/splats/i)).toBeInTheDocument();
      expect(screen.getByText('1,234')).toBeInTheDocument();
    });
  });

  it('shared mode hides the in-app action buttons', async () => {
    setup(true);
    expect(screen.queryByText(/Share \/ Export/i)).not.toBeInTheDocument();
    await waitFor(() => expect(instances[0].loadedUrls.length).toBeGreaterThan(0));
  });

  it('disposes the engine on unmount', async () => {
    const { unmount } = setup();
    await waitFor(() => expect(instances.length).toBeGreaterThan(0));
    const inst = instances[0];
    unmount();
    expect(inst.disposed).toBe(true);
  });
});
