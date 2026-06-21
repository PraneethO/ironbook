/**
 * VoiceButton — push-to-talk microphone using Deepgram streaming STT.
 *
 * Hold the button (or Space key) to speak. Deepgram streams a live transcript
 * back as you talk. On release, the final transcript is passed to onTranscript.
 *
 * The Deepgram key is fetched from the backend (/api/agent/voice-config) so it
 * never ships in the compiled JS bundle.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '../api/client';

interface Props {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}

type VoiceState = 'idle' | 'connecting' | 'listening' | 'error';

export function VoiceButton({ onTranscript, disabled }: Props) {
  const [state, setState] = useState<VoiceState>('idle');
  const [liveText, setLiveText] = useState('');
  const [keyReady, setKeyReady] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const finalRef = useRef('');
  const dgKeyRef = useRef('');
  const dgModelRef = useRef('nova-2');

  // Fetch config once on mount
  useEffect(() => {
    apiClient
      .voiceConfig()
      .then((cfg) => {
        dgKeyRef.current = cfg.deepgram_key;
        dgModelRef.current = cfg.model ?? 'nova-2';
        setKeyReady(!!cfg.deepgram_key);
      })
      .catch(() => setKeyReady(false));
  }, []);

  const stopAll = useCallback(() => {
    // Stop MediaRecorder
    if (mediaRef.current && mediaRef.current.state !== 'inactive') {
      mediaRef.current.stop();
    }
    mediaRef.current = null;

    // Stop microphone tracks
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    // Close WebSocket
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
    }
  }, []);

  const startListening = useCallback(async () => {
    if (!keyReady || state !== 'idle') return;
    setState('connecting');
    finalRef.current = '';
    setLiveText('');

    try {
      // Request microphone
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Connect to Deepgram streaming WebSocket
      const params = new URLSearchParams({
        model: dgModelRef.current,
        language: 'en-US',
        smart_format: 'true',
        interim_results: 'true',
        utterance_end_ms: '800',
        vad_events: 'true',
        encoding: 'linear16',
        sample_rate: '16000',
        channels: '1',
      });
      const ws = new WebSocket(
        `wss://api.deepgram.com/v1/listen?${params}`,
        ['token', dgKeyRef.current],
      );
      wsRef.current = ws;

      ws.onopen = () => {
        setState('listening');

        // Use AudioContext to downsample to 16kHz PCM
        const audioCtx = new AudioContext({ sampleRate: 16000 });
        const source = audioCtx.createMediaStreamSource(stream);
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const float32 = e.inputBuffer.getChannelData(0);
          // Convert Float32 → Int16 PCM
          const int16 = new Int16Array(float32.length);
          for (let i = 0; i < float32.length; i++) {
            int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
          }
          ws.send(int16.buffer);
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);

        // Store ref to stop on release
        (ws as any)._audioCtx = audioCtx;
        (ws as any)._processor = processor;
        (ws as any)._source = source;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data as string);
          const alt = data?.channel?.alternatives?.[0];
          if (!alt) return;
          const transcript: string = alt.transcript ?? '';
          const isFinal: boolean = data.is_final ?? false;

          if (transcript) {
            setLiveText(transcript);
            if (isFinal) {
              finalRef.current = (finalRef.current + ' ' + transcript).trim();
              setLiveText('');
            }
          }
        } catch {}
      };

      ws.onerror = () => {
        setState('error');
        stopAll();
      };

      ws.onclose = () => {
        // Cleanup AudioContext nodes
        try {
          (ws as any)._processor?.disconnect();
          (ws as any)._source?.disconnect();
          (ws as any)._audioCtx?.close();
        } catch {}
        setState((s) => (s === 'listening' || s === 'connecting' ? 'idle' : s));
      };
    } catch (err) {
      setState('error');
      stopAll();
      setTimeout(() => setState('idle'), 2000);
    }
  }, [keyReady, state, stopAll]);

  const stopListening = useCallback(() => {
    if (state !== 'listening' && state !== 'connecting') return;
    stopAll();

    // Small delay so Deepgram finalises the last utterance
    setTimeout(() => {
      setState('idle');
      setLiveText('');
      const text = finalRef.current.trim();
      if (text) {
        onTranscript(text);
        finalRef.current = '';
      }
    }, 300);
  }, [state, stopAll, onTranscript]);

  // Space key as hold-to-talk (only when input isn't focused)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return;
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA'
      )
        return;
      e.preventDefault();
      startListening();
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA'
      )
        return;
      stopListening();
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [startListening, stopListening]);

  // Clean up on unmount
  useEffect(() => () => stopAll(), [stopAll]);

  const label =
    state === 'connecting'
      ? '⏳'
      : state === 'listening'
        ? '🔴'
        : state === 'error'
          ? '⚠️'
          : '🎙️';

  const title =
    state === 'idle'
      ? 'Hold to speak (or hold Space)'
      : state === 'connecting'
        ? 'Connecting…'
        : state === 'listening'
          ? 'Listening — release to send'
          : 'Microphone error';

  return (
    <div className="voice-button-wrap">
      <button
        className={`btn voice-btn voice-btn-${state}`}
        onMouseDown={() => void startListening()}
        onMouseUp={stopListening}
        onMouseLeave={state === 'listening' ? stopListening : undefined}
        onTouchStart={(e) => {
          e.preventDefault();
          void startListening();
        }}
        onTouchEnd={(e) => {
          e.preventDefault();
          stopListening();
        }}
        disabled={disabled || !keyReady}
        title={title}
        aria-label={title}
        data-testid="voice-btn"
      >
        {label}
      </button>
      {liveText && (
        <span className="voice-live-text" data-testid="voice-live-text">
          {liveText}
        </span>
      )}
    </div>
  );
}
