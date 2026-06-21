/**
 * App — router + shared chrome. Routes map to the 8 key screens from
 * 04_user_experience.md plus the public /view/:id shared-link route.
 */
import { Link, Route, Routes, useLocation } from 'react-router-dom';
import * as Sentry from '@sentry/react';
import { Dashboard } from './pages/Dashboard';
import { NewWorld } from './pages/NewWorld';
import { CaptureGuide } from './pages/CaptureGuide';
import { Processing } from './pages/Processing';
import { ViewerPage } from './pages/ViewerPage';
import { SceneSettings } from './pages/SceneSettings';
import { ExportShare } from './pages/ExportShare';
import { AddPhotos } from './pages/AddPhotos';

function TopBar() {
  return (
    <header className="topbar">
      <Link className="brand" to="/">
        <span className="logo" aria-hidden />
        Gaussian Splat World
      </Link>
      <Link className="btn btn-primary" to="/new">
        + New 3D World
      </Link>
    </header>
  );
}

function CrashFallback({ error }: { error: Error }) {
  return (
    <div style={{ padding: '2rem', color: '#fff' }}>
      <h2>Something went wrong</h2>
      <pre style={{ opacity: 0.6, fontSize: 12 }}>{error.message}</pre>
      <button className="btn btn-primary" onClick={() => window.location.reload()}>
        Reload
      </button>
    </div>
  );
}

export function App() {
  const location = useLocation();
  // The full-screen viewer route hides the app chrome.
  const isImmersive =
    location.pathname.startsWith('/view/') ||
    /^\/projects\/[^/]+\/viewer$/.test(location.pathname);

  if (isImmersive) {
    return (
      <Sentry.ErrorBoundary fallback={CrashFallback}>
        <Routes>
          <Route path="/view/:id" element={<ViewerPage shared />} />
          <Route path="/projects/:id/viewer" element={<ViewerPage />} />
        </Routes>
      </Sentry.ErrorBoundary>
    );
  }

  return (
    <div className="app-shell">
      <TopBar />
      <main className="container">
        <Sentry.ErrorBoundary fallback={CrashFallback}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/new" element={<NewWorld />} />
            <Route path="/guide" element={<CaptureGuide />} />
            <Route path="/projects/:id/upload" element={<NewWorld />} />
            <Route path="/projects/:id/add" element={<AddPhotos />} />
            <Route path="/projects/:id/processing" element={<Processing />} />
            <Route path="/projects/:id/settings" element={<SceneSettings />} />
            <Route path="/projects/:id/share" element={<ExportShare />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </Sentry.ErrorBoundary>
      </main>
    </div>
  );
}
