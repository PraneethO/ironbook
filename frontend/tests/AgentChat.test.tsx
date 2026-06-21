/**
 * AgentChat tests — mocks the API and viewer, exercises the executor and UI.
 * No WebGL needed.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AgentChat } from '../src/components/AgentChat';

// ---------------------------------------------------------------------------
// Mock apiClient.agentAct so we never hit the network
// ---------------------------------------------------------------------------
vi.mock('../src/api/client', () => ({
  apiClient: {
    agentAct: vi.fn(),
    voiceConfig: vi.fn().mockResolvedValue({ deepgram_key: 'test-key', model: 'nova-2' }),
  },
}));
import { apiClient } from '../src/api/client';

// ---------------------------------------------------------------------------
// Minimal SplatViewer mock that records method calls
// ---------------------------------------------------------------------------
function makeMockViewer(overrides: Record<string, unknown> = {}) {
  return {
    splatCount: 100,
    isAnimating: false,
    capture: vi.fn(() => 'data:image/png;base64,AAAA'),
    getCameraSnapshot: vi.fn(() => ({
      mode: 'orbit',
      fov: 1.0,
      eye: [0, 0, 5],
      target: [0, 0, 0],
      bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    })),
    pickAt: vi.fn(() => [0.5, 0.5, 0] as [number, number, number]),
    rayToSceneDepth: vi.fn(() => [0.5, 0.5, 0] as [number, number, number]),
    setSplatScale: vi.fn(),
    setBackgroundByName: vi.fn(),
    setBrightness: vi.fn(),
    moveRelative: vi.fn(),
    rotateView: vi.fn(),
    zoomView: vi.fn(),
    flyTo: vi.fn(),
    lookAtPoint: vi.fn(),
    highlightAt: vi.fn(),
    clearHighlight: vi.fn(),
    resetCamera: vi.fn(),
    ...overrides,
  };
}

function makeViewerRef(viewer: ReturnType<typeof makeMockViewer>) {
  return { current: { viewer } } as any;
}

afterEach(() => {
  vi.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Helper: open the chat panel
// ---------------------------------------------------------------------------
function openPanel() {
  const btn = screen.getByTestId('agent-toggle');
  fireEvent.click(btn);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AgentChat panel', () => {
  it('renders the panel open by default; header closes it and the toggle reopens', () => {
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);
    // Panel is open by default
    expect(screen.getByTestId('agent-chat')).toBeTruthy();
    // Header ✕ closes it (the toggle is hidden while open)
    fireEvent.click(screen.getByTestId('agent-close'));
    expect(screen.queryByTestId('agent-chat')).toBeNull();
    // The toggle reopens it
    openPanel();
    expect(screen.getByTestId('agent-chat')).toBeTruthy();
  });

  it('shows the hint message when no messages yet', () => {
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);
    // Panel is open by default — hint is visible immediately
    expect(screen.getByText(/navigate/i)).toBeTruthy();
  });

  it('sends user message and renders agent answer', async () => {
    (apiClient.agentAct as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'I rotated the view.',
      actions: [],
    });
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), {
      target: { value: 'rotate clockwise' },
    });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() => screen.getAllByTestId('agent-msg-assistant'));
    const msgs = screen.getAllByTestId('agent-msg-assistant');
    expect(msgs.some((el) => el.textContent?.includes('I rotated the view.'))).toBe(true);
  });

  it('calls moveRelative for a move action', async () => {
    (apiClient.agentAct as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'Moving forward.',
      actions: [{ type: 'move', direction: 'forward', amount: 1.5 }],
    });
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), { target: { value: 'move forward' } });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() => expect(viewer.moveRelative).toHaveBeenCalledWith('forward', 1.5));
  });

  it('calls rotateView for a rotate action', async () => {
    (apiClient.agentAct as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'Rotating.',
      actions: [{ type: 'rotate', direction: 'clockwise', amount: 1 }],
    });
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), { target: { value: 'turn right' } });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() => expect(viewer.rotateView).toHaveBeenCalledWith('clockwise', 1));
  });

  it('calls zoomView for a zoom action', async () => {
    (apiClient.agentAct as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'Zooming.',
      actions: [{ type: 'zoom', direction: 'in', amount: 1 }],
    });
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), { target: { value: 'zoom in' } });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() => expect(viewer.zoomView).toHaveBeenCalledWith('in', 1));
  });

  it('calls rayToSceneDepth + flyTo for a fly_to action', async () => {
    (apiClient.agentAct as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'Flying to target.',
      actions: [{ type: 'fly_to', target_2d: [0.6, 0.4] }],
    });
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), { target: { value: 'go to the fountain' } });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() => expect(viewer.rayToSceneDepth).toHaveBeenCalledWith(0.6, 0.4));
    expect(viewer.flyTo).toHaveBeenCalledWith([0.5, 0.5, 0]);
  });

  it('calls rayToSceneDepth + highlightAt + lookAtPoint for a highlight action', async () => {
    (apiClient.agentAct as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'Highlighting.',
      actions: [{ type: 'highlight', target_2d: [0.3, 0.5], label: 'door' }],
    });
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), { target: { value: 'highlight the door' } });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() => expect(viewer.rayToSceneDepth).toHaveBeenCalledWith(0.3, 0.5));
    expect(viewer.highlightAt).toHaveBeenCalledWith([0.5, 0.5, 0]);
    expect(viewer.lookAtPoint).toHaveBeenCalledWith([0.5, 0.5, 0]);
  });

  it('calls clearHighlight for a clear_highlight action', async () => {
    (apiClient.agentAct as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'Cleared.',
      actions: [{ type: 'clear_highlight' }],
    });
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), { target: { value: 'clear highlight' } });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() => expect(viewer.clearHighlight).toHaveBeenCalled());
  });

  it('calls resetCamera for a reset_view action', async () => {
    (apiClient.agentAct as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'Reset.',
      actions: [{ type: 'reset_view' }],
    });
    const viewer = makeMockViewer();
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), { target: { value: 'reset' } });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() => expect(viewer.resetCamera).toHaveBeenCalled());
  });

  it('shows "Load a world first" when splatCount is 0', async () => {
    const viewer = makeMockViewer({ splatCount: 0 });
    render(<AgentChat viewerRef={makeViewerRef(viewer)} />);

    fireEvent.change(screen.getByTestId('agent-input'), { target: { value: 'hello' } });
    fireEvent.click(screen.getByTestId('agent-send'));

    await waitFor(() =>
      screen.getAllByTestId('agent-msg-assistant').some((el) =>
        el.textContent?.includes('Load a world first'),
      ),
    );
    // Should NOT have called the API
    expect(apiClient.agentAct).not.toHaveBeenCalled();
  });
});
