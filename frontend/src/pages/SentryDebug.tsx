import { useEffect, useState } from 'react';
import * as Sentry from '@sentry/react';

type Result = { name: string; id: string | null; ok: boolean };

export function SentryDebug() {
  const [results, setResults] = useState<Result[]>([]);

  useEffect(() => {
    const fired: Result[] = [];

    // 1. Unhandled promise rejection (auto-captured by Sentry's global handler)
    Promise.reject(new Error('SentryDebug: unhandled promise rejection in viewer init'));

    // 2. Manual captureException — simulates a failed asset decode
    const decodeErr = new TypeError('SentryDebug: splat asset header magic bytes mismatch (expected 0x504C5953, got 0x00000000)');
    const id2 = Sentry.captureException(decodeErr, {
      tags: { component: 'SplatLoader', phase: 'decode' },
    });
    fired.push({ name: 'TypeError: splat decode failure', id: id2, ok: true });

    // 3. Manual captureException — simulates agent JSON parse failure
    const agentErr = new SyntaxError('SentryDebug: agent response was not valid JSON — model returned truncated output');
    const id3 = Sentry.captureException(agentErr, {
      tags: { component: 'AgentChat', model: 'claude-sonnet-4-6' },
    });
    fired.push({ name: 'SyntaxError: agent JSON parse failure', id: id3, ok: true });

    // 4. captureMessage at error level — simulates missing Deepgram key warning
    const id4 = Sentry.captureMessage(
      'SentryDebug: DEEPGRAM_API_KEY not configured — voice input disabled',
      { level: 'error', tags: { component: 'VoiceButton', feature: 'voice-input' } },
    );
    fired.push({ name: 'Error message: voice input misconfiguration', id: id4, ok: true });

    // 5. ErrorBoundary trigger — throw inside a render to exercise the boundary
    //    (done via a thrown error from a child, tested via captureException here)
    const renderErr = new RangeError('SentryDebug: WebGL2 context lost — too many active splat scenes');
    const id5 = Sentry.captureException(renderErr, {
      tags: { component: 'WebGLRenderer', context: 'webgl2' },
    });
    fired.push({ name: 'RangeError: WebGL2 context lost', id: id5, ok: true });

    Sentry.flush(5000).then(() => setResults(fired));
  }, []);

  return (
    <div style={{ padding: '2rem', fontFamily: 'monospace' }}>
      <h2 style={{ marginBottom: '1rem' }}>Sentry Frontend Debug</h2>
      {results.length === 0 ? (
        <p style={{ opacity: 0.6 }}>Firing errors and flushing to Sentry…</p>
      ) : (
        <>
          <p style={{ color: '#4ade80', marginBottom: '1rem' }}>
            Flushed {results.length} events to Sentry.
          </p>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {results.map((r) => (
              <li key={r.id} style={{ marginBottom: '0.5rem' }}>
                <span style={{ color: '#4ade80' }}>✓</span> {r.name}
                <br />
                <span style={{ opacity: 0.5, fontSize: '0.75rem' }}>event id: {r.id}</span>
              </li>
            ))}
          </ul>
          <p style={{ marginTop: '1.5rem', opacity: 0.6, fontSize: '0.85rem' }}>
            Check{' '}
            <a
              href="https://praneeth-otthi.sentry.io/issues/"
              target="_blank"
              rel="noreferrer"
              style={{ color: '#818cf8' }}
            >
              sentry.io/issues
            </a>
          </p>
        </>
      )}
    </div>
  );
}
