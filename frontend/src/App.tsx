/**
 * App — router + shared chrome. Routes map to the 8 key screens from
 * 04_user_experience.md plus the public /view/:id shared-link route.
 */
import { Link, Route, Routes, useLocation } from 'react-router-dom';
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

export function App() {
  const location = useLocation();
  // The full-screen viewer route hides the app chrome.
  const isImmersive =
    location.pathname.startsWith('/view/') ||
    /^\/projects\/[^/]+\/viewer$/.test(location.pathname);

  if (isImmersive) {
    return (
      <Routes>
        <Route path="/view/:id" element={<ViewerPage shared />} />
        <Route path="/projects/:id/viewer" element={<ViewerPage />} />
      </Routes>
    );
  }

  return (
    <div className="app-shell">
      <TopBar />
      <main className="container">
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
      </main>
    </div>
  );
}
