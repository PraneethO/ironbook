/**
 * AgentChat — the reasoning navigation agent chat panel.
 *
 * Rendered inside the full-screen viewer (both owner and public/shared views).
 * The panel takes a screenshot of the current viewer frame, sends it + the
 * user's message to the backend agent endpoint, then executes the returned
 * actions against the viewer so the user can watch the camera navigate in
 * real-time.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import * as Sentry from '@sentry/react';
import { apiClient } from '../api/client';
import type { AgentAction, AgentTurn } from '../api/types';
import type { SplatViewerHandle } from './SplatViewerReact';
import type { SplatViewer } from '../viewer/SplatViewer';
import type { Vec3 } from '../viewer/math';
import { VoiceButton } from './VoiceButton';

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

interface Props {
  viewerRef: React.RefObject<SplatViewerHandle | null>;
}

/** Map agent actions to viewer calls. */
function executeActions(viewer: SplatViewer, actions: AgentAction[]): void {
  for (const action of actions) {
    switch (action.type) {
      case 'move':
        if (action.direction) {
          viewer.moveRelative(
            action.direction as Parameters<typeof viewer.moveRelative>[0],
            action.amount ?? 1,
          );
        }
        break;
      case 'rotate':
        if (action.direction) {
          viewer.rotateView(
            action.direction as Parameters<typeof viewer.rotateView>[0],
            action.amount ?? 1,
          );
        }
        break;
      case 'zoom':
        if (action.direction) {
          viewer.zoomView(
            action.direction as Parameters<typeof viewer.zoomView>[0],
            action.amount ?? 1,
          );
        }
        break;
      case 'reset_view':
        viewer.resetCamera();
        break;
      case 'clear_highlight':
        viewer.clearHighlight();
        break;
      case 'fly_to': {
        if (!action.target_2d) break;
        const p = viewer.rayToSceneDepth(action.target_2d[0], action.target_2d[1]);
        viewer.flyTo(p as Vec3);
        break;
      }
      case 'look_at': {
        if (!action.target_2d) break;
        const p = viewer.rayToSceneDepth(action.target_2d[0], action.target_2d[1]);
        viewer.lookAtPoint(p as Vec3);
        break;
      }
      case 'highlight': {
        if (!action.target_2d) break;
        const p = viewer.rayToSceneDepth(action.target_2d[0], action.target_2d[1]);
        viewer.highlightAt(p as Vec3);
        viewer.lookAtPoint(p as Vec3);
        break;
      }
      case 'set_splat_scale':
        viewer.setSplatScale(action.amount ?? 1.0);
        break;
      case 'set_background':
        viewer.setBackgroundByName(action.label ?? 'dark');
        break;
      case 'set_brightness':
        viewer.setBrightness(action.amount ?? 1.0);
        break;
    }
  }
}

export function AgentChat({ viewerRef }: Props) {
  const [open, setOpen] = useState(true); // open by default
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [animating, setAnimating] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Poll viewer animation state to show the "navigating…" hint
  useEffect(() => {
    if (!open) return;
    const id = setInterval(() => {
      const v = viewerRef.current?.viewer;
      setAnimating(v ? v.isAnimating : false);
    }, 100);
    return () => clearInterval(id);
  }, [open, viewerRef]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (typeof bottomRef.current?.scrollIntoView === 'function') {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Core dispatch — works for both text input and voice transcript.
  const sendText = useCallback(async (text: string) => {
    if (!text.trim() || busy) return;
    const v = viewerRef.current?.viewer;
    if (!v) return;

    if (v.splatCount === 0) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: 'Load a world first — open or upload a .splat scene.' },
      ]);
      return;
    }

    setMessages((m) => [...m, { role: 'user', text }]);
    setBusy(true);

    const camera = v.getCameraSnapshot();

    const span = Sentry.startInactiveSpan({
      name: 'ironbook.agent.act',
      op: 'ai.pipeline',
      attributes: {
        'gen_ai.request.model': 'claude-sonnet-4-6',
        'ai.message_length': text.length,
        'ai.history_turns': messages.length,
        'ai.camera_mode': camera.mode,
        'ai.splat_count': v.splatCount,
      },
    });

    try {
      // Capture screenshot and measure how long it takes.
      const screenshotStart = performance.now();
      const screenshot = v.capture();
      const screenshotMs = Math.round(performance.now() - screenshotStart);
      const screenshot_b64 = screenshot.split(',')[1];

      Sentry.addBreadcrumb({
        category: 'agent.screenshot',
        message: 'viewer frame captured',
        data: {
          capture_ms: screenshotMs,
          data_chars: screenshot_b64.length,
          splat_count: v.splatCount,
        },
        level: 'info',
      });
      span.setAttribute('ai.screenshot_capture_ms', screenshotMs);

      const history: AgentTurn[] = messages.slice(-6).map((m) => ({
        role: m.role,
        text: m.text,
      }));

      Sentry.addBreadcrumb({
        category: 'agent.request',
        message: 'sending to backend agent',
        data: {
          message_preview: text.slice(0, 120),
          history_turns: history.length,
          camera_mode: camera.mode,
        },
        level: 'info',
      });

      const res = await apiClient.agentAct({
        message: text,
        screenshot_b64,
        camera,
        history,
      });

      span.setAttribute('ai.action_count', res.actions.length);
      span.setAttribute('ai.action_types', res.actions.map((a) => a.type).join(','));
      span.setAttribute('ai.answer_length', res.answer.length);
      span.setStatus({ code: 1 });

      Sentry.addBreadcrumb({
        category: 'agent.response',
        message: 'agent responded',
        data: {
          answer_preview: res.answer.slice(0, 120),
          action_count: res.actions.length,
          action_types: res.actions.map((a) => a.type),
        },
        level: 'info',
      });

      setMessages((m) => [...m, { role: 'assistant', text: res.answer }]);

      // Breadcrumb per action so the trace shows exactly what the agent did.
      for (const action of res.actions) {
        Sentry.addBreadcrumb({
          category: 'agent.action',
          message: `execute: ${action.type}`,
          data: {
            type: action.type,
            ...(action.direction ? { direction: action.direction } : {}),
            ...(action.amount != null ? { amount: action.amount } : {}),
            ...(action.label ? { label: action.label } : {}),
          },
          level: 'info',
        });
      }

      executeActions(v, res.actions);
    } catch (err) {
      span.setStatus({ code: 2, message: err instanceof Error ? err.message : 'error' });
      Sentry.captureException(err, {
        tags: { source: 'agent_chat', camera_mode: camera.mode },
        extra: { message_preview: text.slice(0, 200), splat_count: v.splatCount },
      });
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: err instanceof Error ? err.message : 'Something went wrong. Try again.',
        },
      ]);
    } finally {
      span.end();
      setBusy(false);
    }
  }, [busy, messages, viewerRef]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text) return;
    setInput('');
    await sendText(text);
  }, [input, sendText]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  return (
    <>
      {/* Toggle button — bottom-right, hidden while the panel is open */}
      {!open && (
        <button
          className="btn agent-toggle glass"
          style={{ position: 'absolute', bottom: 52, right: 16, zIndex: 60, pointerEvents: 'auto' }}
          onClick={() => setOpen(true)}
          title="Ask the navigation agent"
          data-testid="agent-toggle"
        >
          Ask Agent
        </button>
      )}

      {open && (
        <div className="agent-chat glass" data-testid="agent-chat">
          <div className="agent-chat-header">
            <span>Navigation Agent</span>
            {animating && (
              <span className="agent-navigating" data-testid="agent-navigating">
                Navigating…
              </span>
            )}
            <button
              className="btn btn-ghost"
              style={{ padding: '2px 8px', fontSize: 12 }}
              onClick={() => setOpen(false)}
              data-testid="agent-close"
            >
              ✕
            </button>
          </div>

          <div className="agent-chat-messages" data-testid="agent-messages">
            {messages.length === 0 && (
              <p className="agent-hint">
                Ask me to navigate: "go to the fountain", "highlight the door",
                "what is this object?", "rotate clockwise", "zoom in"…
              </p>
            )}
            {messages.map((m, i) => (
              <div
                key={i}
                className={`agent-msg agent-msg-${m.role}`}
                data-testid={`agent-msg-${m.role}`}
              >
                <span className="agent-msg-role">{m.role === 'user' ? 'You' : 'Agent'}</span>
                <span className="agent-msg-text">{m.text}</span>
              </div>
            ))}
            {busy && (
              <div className="agent-msg agent-msg-assistant">
                <span className="agent-msg-role">Agent</span>
                <span className="agent-msg-text agent-thinking">Thinking…</span>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="agent-chat-input-row">
            <VoiceButton
              onTranscript={(text) => void sendText(text)}
              disabled={busy}
            />
            <input
              ref={inputRef}
              type="text"
              className="agent-input"
              placeholder="Type a command, or hold mic to speak…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={busy}
              data-testid="agent-input"
            />
            <button
              className="btn btn-primary"
              onClick={() => void send()}
              disabled={busy || !input.trim()}
              data-testid="agent-send"
            >
              {busy ? '…' : 'Send'}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
