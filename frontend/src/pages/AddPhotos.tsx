/**
 * Add More Photos (screen 8) — re-opens the upload flow for an existing
 * project. Reuses NewWorld, which already supports the /projects/:id/upload
 * route (it skips name entry and uploads into the existing project).
 */
import { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

export function AddPhotos() {
  const { id } = useParams();
  const navigate = useNavigate();
  useEffect(() => {
    if (id) navigate(`/projects/${id}/upload`, { replace: true });
  }, [id, navigate]);
  return null;
}
