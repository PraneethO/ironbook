/** Capture Guide (screen 3) — friendly capture tips from 04_user_experience.md. */
import { useNavigate } from 'react-router-dom';

const TIPS: { icon: string; title: string; body: string }[] = [
  { icon: '🚶', title: 'Walk all the way around', body: 'Move around the object or space so every side gets seen.' },
  { icon: '🔁', title: 'Overlap your shots', body: 'Each photo should share a good chunk with the last one.' },
  { icon: '✨', title: 'Keep it sharp', body: 'Avoid blurry photos — hold steady and use good light.' },
  { icon: '📐', title: 'Many angles', body: 'High, low, near and far. Variety builds a fuller world.' },
  { icon: '🎥', title: 'Move slowly on video', body: 'If you record video, pan slowly so frames stay crisp.' },
  { icon: '🙈', title: 'Capture hidden sides', body: 'Anything you skip may end up with holes in the 3D world.' },
];

export function CaptureGuide() {
  const navigate = useNavigate();
  return (
    <div>
      <div className="page-head">
        <div>
          <h1>How to capture a great world</h1>
          <p className="muted">A few tips make a big difference to the result.</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/new')}>
          I'm ready — upload photos
        </button>
      </div>
      <ul className="guide">
        {TIPS.map((t) => (
          <li key={t.title}>
            <span className="ic" aria-hidden>{t.icon}</span>
            <div>
              <div style={{ fontWeight: 600 }}>{t.title}</div>
              <div className="muted">{t.body}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
