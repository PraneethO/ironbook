import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, useLocation, useNavigationType, createRoutesFromChildren, matchRoutes } from 'react-router-dom';
import * as Sentry from '@sentry/react';
import { App } from './App';
import './styles/global.css';

const sentryDsn = import.meta.env.VITE_SENTRY_DSN as string | undefined;

Sentry.init({
  dsn: sentryDsn,
  release: (import.meta.env.VITE_APP_VERSION as string | undefined) ?? '0.1.0',
  integrations: [
    Sentry.reactRouterV6BrowserTracingIntegration({
      useEffect: React.useEffect,
      useLocation,
      useNavigationType,
      createRoutesFromChildren,
      matchRoutes,
    }),
    Sentry.replayIntegration({
      // Don't mask text — we want to see what the user typed/asked the agent.
      maskAllText: false,
      blockAllMedia: false,
    }),
  ],
  // Capture 100% of transactions in dev; tune down in production.
  tracesSampleRate: 1.0,
  // Record 10% of sessions normally; 100% of sessions that hit an error.
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,
  environment: import.meta.env.MODE,
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
